#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

// Constants
constexpr int TILE_SIZE     = 16;
constexpr int BLOCK_SIZE    = 256;
constexpr int WARP_SIZE     = 32;
constexpr float TEMPERATURE = 1.0f;  // Adjust as needed
constexpr float CLAMP_MIN   = 1e-8f;

// Utility functions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / ; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        val = shared[lane];
    } else {
        val = 0.0f;
    }

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel 1: Normalize and reshape
template <typename T>
__global__ void normalize_reshape_kernel(const T* input, float* output, int B, int H, int L, int D, int split_heads, int total_heads, int head_dim) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int seq   = blockIdx.z;
    int tid   = threadIdx.x;

    if (batch >= B || head >= split_heads || seq >= L)
        return;

    int input_head     = head;
    int start_feature  = 0;
    int feature_stride = 1;

    // Compute norm for this (batch, head, seq) slice
    __shared__ float s_norm[BLOCK_SIZE];
    float local_sum = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        int input_idx = ((batch * total_heads + input_head) * L + seq) * head_dim + start_feature + i * feature_stride;
        float val     = static_cast<float>(input[input_idx]);
        local_sum += val * val;
    }

    s_norm[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_norm[tid] += s_norm[tid + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(s_norm[0] + 1e-12f);

    // Normalize and write to output
    for (int i = tid; i < D; i += blockDim.x) {
        int input_idx      = ((batch * total_heads + input_head) * L + seq) * head_dim + start_feature + i * feature_stride;
        int output_idx     = ((batch * split_heads + head) * L + seq) * D + i;
        float val          = static_cast<float>(input[input_idx]) / norm;
        output[output_idx] = val;
    }
}

// Kernel 2: Compute relation matrix (batched matrix multiplication)
__global__ void compute_relation_kernel(const float* A, const float* B, float* C, int B_dim, int M, int N, int K) {
    // Batched matrix multiplication: C = A * B^T
    // A: [B*split_heads, L, D]
    // B: [B*split_heads, L, D]
    // C: [B*split_heads, L, L]

    int batch = blockIdx.z;
    int row   = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col   = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            int a_idx                    = (batch * M + row) * K + a_col;
            As[threadIdx.y][threadIdx.x] = A[a_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile (transposed)
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_row < K) {
            int b_idx                    = (batch * N + col) * K + b_row;
            Bs[threadIdx.y][threadIdx.x] = B[b_idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int c_idx = (batch * M + row) * N + col;
        C[c_idx]  = sum / TEMPERATURE;
    }
}

// Kernel 3: Compute softmax and KL divergence
__global__ void compute_kl_loss_kernel(const float* s_logits, const float* t_logits, float* loss_buffer, int B, int H, int L) {
    int batch_head      = blockIdx.x;
    int tid             = threadIdx.x;
    int total_sequences = B * H * L;

    if (batch_head >= total_sequences)
        return;

    __shared__ float s_max_shared[WARP_SIZE];
    __shared__ float t_max_shared[WARP_SIZE];
    __shared__ float s_sum_shared[WARP_SIZE];
    __shared__ float t_sum_shared[WARP_SIZE];

    // Find max for numerical stability
    float s_max = -INFINITY;
    float t_max = -INFINITY;

    for (int i = tid; i < L; i += blockDim.x) {
        int idx = batch_head * L + i;
        s_max   = max(s_max, s_logits[idx]);
        t_max   = max(t_max, t_logits[idx]);
    }

    s_max = warp_reduce_max(s_max);
    t_max = warp_reduce_max(t_max);

    if (tid % WARP_SIZE == 0) {
        s_max_shared[tid / WARP_SIZE] = s_max;
        t_max_shared[tid / WARP_SIZE] = t_max;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        s_max = s_max_shared[tid];
        t_max = t_max_shared[tid];
    }

    s_max = warp_reduce_max(s_max);
    t_max = warp_reduce_max(t_max);

    // Compute exponentials and sums
    float s_exp_sum = 0.0f;
    float t_exp_sum = 0.0f;
    float kl_sum    = 0.0f;

    for (int i = tid; i < L; i += blockDim.x) {
        int idx     = batch_head * L + i;
        float s_val = s_logits[idx] - s_max;
        float t_val = t_logits[idx] - t_max;

        float s_exp = expf(s_val);
        float t_exp = expf(t_val);

        s_exp_sum += s_exp;
        t_exp_sum += t_exp;

        s_val = __shfl_sync(0xffffffff, s_exp, i % WARP_SIZE);
        t_val = __shfl_sync(0xffffffff, t_exp, i % WARP_SIZE);

        float s_prob = max(s_val / s_exp_sum, CLAMP_MIN);
        float t_prob = max(t_val / t_exp_sum, CLAMP_MIN);

        kl_sum += t_prob * (logf(t_prob) - logf(s_prob));
    }

    s_exp_sum = warp_reduce_sum(s_exp_sum);
    t_exp_sum = warp_reduce_sum(t_exp_sum);
    kl_sum    = warp_reduce_sum(kl_sum);

    if (tid == 0) {
        loss_buffer[batch_head] = kl_sum;
    }
}

// Main CUDA function
torch::Tensor compute_attention_distillation_loss_cuda(const std::vector<torch::Tensor>& student_states, const std::vector<torch::Tensor>& teacher_states,
                                                       int64_t distill_layer, int64_t split_heads) {
    cudaSetDevice(student_states[0].get_device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(student_states[0].device());

    // Get tensor dimensions
    auto s_shape = student_states[0].sizes();
    auto t_shape = teacher_states[0].sizes();

    int64_t B           = s_shape[1];
    int64_t total_heads = s_shape[2];
    int64_t L           = s_shape[3];
    int64_t head_dim    = s_shape[4];
    int64_t D           = total_heads * head_dim / split_heads;

    // Validate dimensions
    TORCH_CHECK(D * split_heads == total_heads * head_dim, "D * split_heads must equal total_heads * head_dim");

    // Allocate buffers
    int64_t total_batch_heads = B * split_heads;
    int64_t normalized_size   = total_batch_heads * L * D;
    int64_t relation_size     = total_batch_heads * L * L;

    auto s_normalized_buffer = torch::empty({total_batch_heads, L, D}, options);
    auto t_normalized_buffer = torch::empty({total_batch_heads, L, D}, options);
    auto s_relation_buffer   = torch::empty({total_batch_heads, L, L}, options);
    auto t_relation_buffer   = torch::empty({total_batch_heads, L, L}, options);
    auto loss_buffer         = torch::zeros({total_batch_heads * L}, options);

    float total_loss = 0.0f;

    // Process Q, K, V
    for (int i = 0; i < 3; ++i) {
        auto s_tensor = student_states[i].contiguous();
        auto t_tensor = teacher_states[i].contiguous();

        // Step 1: Normalize and reshape
        dim3 norm_grid(B, split_heads, L);
        dim3 norm_block(BLOCK_SIZE);

        if (s_tensor.dtype() == torch::kBFloat16) {
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(s_tensor.data_ptr<at::BFloat16>(), s_normalized_buffer.data_ptr<float>(), B,
                                                                           total_heads, L, D, split_heads, total_heads, head_dim);
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(t_tensor.data_ptr<at::BFloat16>(), t_normalized_buffer.data_ptr<float>(), B,
                                                                           total_heads, L, D, split_heads, total_heads, head_dim);
        } else if (s_tensor.dtype() == torch::kFloat16) {
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(s_tensor.data_ptr<at::Half>(), s_normalized_buffer.data_ptr<float>(), B, total_heads,
                                                                           L, D, split_heads, total_heads, head_dim);
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(t_tensor.data_ptr<at::Half>(), t_normalized_buffer.data_ptr<float>(), B, total_heads,
                                                                           L, D, split_heads, total_heads, head_dim);
        } else {
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(s_tensor.data_ptr<float>(), s_normalized_buffer.data_ptr<float>(), B, total_heads, L,
                                                                           D, split_heads, total_heads, head_dim);
            normalize_reshape_kernel<<<norm_grid, norm_block, 0, stream>>>(t_tensor.data_ptr<float>(), t_normalized_buffer.data_ptr<float>(), B, total_heads, L,
                                                                           D, split_heads, total_heads, head_dim);
        }

        // Step 2: Compute relation matrices
        dim3 matmul_grid((L + TILE_SIZE - 1) / TILE_SIZE, (L + TILE_SIZE - 1) / TILE_SIZE, total_batch_heads);
        dim3 matmul_block(TILE_SIZE, TILE_SIZE);

        compute_relation_kernel<<<matmul_grid, matmul_block, 0, stream>>>(s_normalized_buffer.data_ptr<float>(), s_normalized_buffer.data_ptr<float>(),
                                                                          s_relation_buffer.data_ptr<float>(), total_batch_heads, L, L, D);

        compute_relation_kernel<<<matmul_grid, matmul_block, 0, stream>>>(t_normalized_buffer.data_ptr<float>(), t_normalized_buffer.data_ptr<float>(),
                                                                          t_relation_buffer.data_ptr<float>(), total_batch_heads, L, L, D);

        // Step 3: Compute KL divergence loss
        auto s_relation_flat = s_relation_buffer.reshape({-1, L});
        auto t_relation_flat = t_relation_buffer.reshape({-1, L});

        int num_blocks = (total_batch_heads * L + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_kl_loss_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(s_relation_flat.data_ptr<float>(), t_relation_flat.data_ptr<float>(),
                                                                      loss_buffer.data_ptr<float>(), B, split_heads, L);

        // Sum losses
        cudaStreamSynchronize(stream);
        auto loss_sum = loss_buffer.sum().item<float>();
        total_loss += loss_sum;
    }

    // Average the loss
    float avg_loss = total_loss / (3.0f * B * split_heads * L);

    return torch::tensor(avg_loss, options);
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_attention_distillation_loss", &compute_attention_distillation_loss_cuda, "Compute attention distillation loss (CUDA)");
}