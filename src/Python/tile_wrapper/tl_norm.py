import torch
import tilelang
import tilelang.language as T


def rms_norm_splitk(M, N, blk_m, blk_k):
    dtype = T.float

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

            for k in range(num_k_step):
                # reverse, better cache hit rate
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


@tilelang.jit(out_idx=[-1])
def rms_norm(M, N, blk_m):
    dtype = T.float

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return main


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)

@tl.jit(out_idx=[1])
def softmax_kernel(
    M,    N,    dtype: T.dtype = T.float16,
) -> "Callable":
    BN = min(tl.next_power_of_2(N), 8192)
    NN = tl.cdiv(N, BN)

    accum_dtype = T.float32

    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),
        Y: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)        #   lse - Log of Sum of Exponentials
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                T.reduce_max(x, max_x, dim=0, clear=True)

                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)

                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)

                lse[0] = max_x[0] * scale + T.log2(T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])

                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main

def soft_max_test():
    M = 8192
    N = 8192
    kernel = softmax_kernel(M, N)
    dtype = torch.float16
    X = torch.randn(M, N, dtype=dtype, device="cuda")
    Y = kernel(X)
    Y_ref = X.softmax(dim=1)

    torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)

    t1 = do_bench(lambda: X.softmax(dim=1), warmup=25, rep=100)
    t2 = do_bench(lambda: kernel(X), warmup=25, rep=100)
    print(f"torch latency: {t1:.3f} ms")
    print(f"TileLang latency: {t2:.3f} ms")
    print(f"Speedup: {t1 / t2:.3f}x")

if __name__ == "__main__":
    M, N, blk_m, blk_k = 8192, 8192, 1, 512
    kernel = rms_norm(M, N, blk_m)
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
