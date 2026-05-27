import tilelang
import tilelang.language as T
import re
from .tl_utils import Codes2Lines, get_gemm_heuristic_sweep, get_gemm_sweep
from tilelang.autotuner import AutoTuner

'''
    1. PyTorch’s nn.Linear defines weight as [out_features, in_features], and the forward pass is: y = X @ W.T + bias
    2. Use Larger Blocks (Increase Arithmetic Intensity) 
        Ampere (A100)   block_M=128, block_N=128/256
        Ada / Hopper    block_M=128/256, block_N=128/256
    3. Increase block_Kto Reduce Global Memory Traffic

    todo
    1. double-buffer of A/B

'''

TL_threads_per_block = 128  # if reg/smem allows
block_M=128; block_N=128; block_K=32

@tilelang.jit(
    #    graph_capture=True,
    out_idx=[-1]
)
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    assert (block_M % 16 == 0 and block_N % 16 == 0)    #   Align Shared Memory Accesses

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=TL_threads_per_block) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            '''
                alloc_fragment 
                1. It's only a logical tile description, each thread might hold a few dozen registers, not thousands.
                2. the compiler (TileLang + LLVM/NVCC) does the heavy lifting, to sure: 1) Register usage stays within hardware limits 2) Each thread does the correct slice of work
                3. Accumulator fragment - Special accumulator registers; Input fragment - General-purpose registers; Tensor core fragment - WMMA / MMA registers
            '''
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype) 
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                # T.async_copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                # T.async_copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local) #deterministic=True

            T.copy(C_local, C[by * block_M, bx * block_N])            

    return gemm

@tilelang.jit(out_idx=[-1])
def matmul_transposed_b(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_transposed_b(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),  # B is (N, K) not (K, N)
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=TL_threads_per_block) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)  # CHANGED: (block_N, block_K) for B^T's block
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)                
                # Load B^T block: we need a (block_N, block_K) block from B^T
                # B^T[x: x+block_N, k*block_K: k*block_K+block_K]
                # corresponds to B[bx*block_N: bx*block_N+block_N, k*block_K: k*block_K+block_K]
                # But B_shared shape is (block_N, block_K)
                T.copy(B[bx * block_N, k * block_K], B_shared)  # Load from B (not B^T)                
                # Compute: C_local += A_shared × B_shared^T
                # Where A_shared: (block_M, block_K)
                #       B_shared: (block_N, block_K) ← this is the tricky part
                # We want: A_shared × B_shared^T
                # Which is: (block_M, block_K) × (block_K, block_N)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # A_shared × B_shared^T

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_transposed_b

@tilelang.jit(out_idx=[-1])
def matmul_transposed_a(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
    """
    Compute C = A.T @ B
    A shape: [K, M]  (transposed: [M, K])
    B shape: [K, N]
    C shape: [M, N]
    """
    @T.prim_func
    def gemm_transposed_a(
        A: T.Tensor((K, M), dtype),  # Note: shape is (K, M) for transpose
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=TL_threads_per_block) as (bx, by):
            A_shared = T.alloc_shared((block_K, block_M), dtype)  # Transposed shape!
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Load A^T tile: access A[k, by*block_M] instead of A[by*block_M, k]
                T.copy(A[k * block_K, by * block_M], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                
                # A_shared is (block_K, block_M), need to transpose for GEMM
                # Option 1: Use transpose gemm
                # T.gemm(T.transpose(A_shared), B_shared, C_local)
                
                # Option 2: Or use gemm with transA=True
                T.gemm(A_shared, B_shared, C_local, transpose_A=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_transposed_a

def ref_program(A, B):
    return A @ B.T

'''
    Since AutoTuner.from_kernel(...).set_compile_args(            out_idx=[-1],            target="auto",        )
    there is no no "@tl.jit(out_idx=[-1])" before "def kernel ..."
'''
def get_best_config(
    M,
    N,
    K,
    with_roller: bool = False,
    profile_backend: str = "event",
):
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
    ):
        dtype = T.bfloat16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_gemm_sweep(M, N, K, with_roller))
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=3, rep=20)

def run_regression_perf(M, N, K, use_autotune: bool = False, with_roller: bool = False, profile_backend: str = "event",):
    if use_autotune:
        result = get_best_config(
            M,  N, K, with_roller=with_roller, profile_backend=profile_backend,
        )
        print(f"run_regression_perf={result.config}")
        kernel = result.kernel
        block_M, block_N, block_K = result.config["block_M"],result.config["block_N"],result.config["block_K"]
    else:
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tl_latency = profiler.do_bench(
        backend=profile_backend,
    )
    ref_latency = profiler.do_bench(
        ref_program,
        backend=profile_backend,
    )
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    flops = 2 * M * N * K * 1e-9
    print(f"\tTileLang TFlops={flops/tl_latency : .3g} / Ref={flops/ref_latency : .3g}")
    return kernel, block_M, block_N, block_K

def GenKernesTable(desc_kernel, kernel_metas, all_codes, M, N, K, block_M=128, block_N=128, block_K=32 ):    
    for desc, func, TRANS in desc_kernel:
        id = len(kernel_metas)
        title = f"_tl_M{M}_N{N}_K{K}"
        kernel, block_M, block_N, block_K = run_regression_perf(M, N, K, use_autotune=True)
        # kernel = func(M, N, K, block_M, block_N, block_K)
        code_0 = kernel.get_kernel_source()
        codes,header = Codes2Lines(id, code_0,title)
        all_codes.append(codes)

        assert header is not None
        kernel_name = re.split(r"[,:;| ()\t]", header)[4]
        # "m":M,"n":N,"k":K,"trans":TRANS,
        #                 "grid_x":(int)(N/block_N),"grid_y":(int)(M/block_M),"grid_z":1,
        #                 "block_x":TL_threads_per_block,"block_y":1,"block_z":1,
        params = [M,N,K,TRANS, (int)(N/block_N),(int)(M/block_M),1,TL_threads_per_block,1,1]       
        params = ",".join(map(str, params))   
        kernel_meata = {"declare":header,"name":kernel_name,
                        "params":params, "desc":desc}
        
        kernel_metas.append(kernel_meata)
        print(kernel_meata)
    return 



def Matmul_wraper(jConfig, path, header):
    header.append("""
using TL_GEMM_KERNEL = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
struct TL_GEMM {
    const int params[10];
    TL_GEMM_KERNEL kernel;
    const char* desc;
    int grid_x, grid_y, grid_z, block_x = 128, block_y = 1, block_z = 1;
    int m, n, k, trans;

    bool isMatch(int m, int n, int k, int trans, int flag = 0x0) {
        if (params[0] == m && params[1] == n && params[2] == k && params[3] == trans) {
            grid_x = params[4], grid_y = params[5], grid_z = params[6];
            block_x = params[7], block_y = params[8], block_z = params[9];
            return true;
        }
        return false;
    }
};
    """)
    kernels = []
    codes = []
    params = jConfig["model"]["parameter"]["transformer"]
    batch = jConfig["train"]["batch"]
    ctx,embed,ffn,q_head,kv_head,head_dim = params["Ctx"],params["Embed"],params["Ffn"],params["Head"],params["KVHead"],params["head_dim"]
    
# Q.Forw(Q.out, inpQ)   8192,2048,1024 no-bias
    M=batch*ctx; N=head_dim*q_head; K=embed    #   
    # tl_Q = matmul(M, N, K, block_M, block_N, block_K)   
    GenKernesTable([("Q",matmul,0),("trans(Q)",matmul_transposed_b,1)], kernels, codes, M, N, K ) 
    if False:
    #     tl_Qt = matmul_transposed_b(M, N, K, block_M, block_N, block_K)   
    #     GenKernesTable("trans(Q)", kernels, codes, tl_Qt,M, N, K, 1)
    # # K.Forw(Q.out, inpQ)   8192,1024,1024 no-bias
        M=batch*ctx; N=head_dim*kv_head; K=embed    #   
    #     tl_KV = matmul(M, N, K, block_M, block_N, block_K)   
        GenKernesTable([("KV",matmul,0),("trans(KV)",matmul_transposed_b,1)], kernels, codes, M, N, K) 
    #     tl_KVt = matmul_transposed_b(M, N, K, block_M, block_N, block_K)   
    #     GenKernesTable("trans(KV)", kernels, codes, tl_KVt,M, N, K, 1)
    # # proj_cat.Forw      8192,1024, 2048 no-bias
        M=batch*ctx; N=embed; K= head_dim*q_head  #   
        GenKernesTable([("cat",matmul,0),("trans(cat)",matmul_transposed_b,1)], kernels, codes, M, N, K) 
    # FFN
        M=batch*ctx; N=ffn; K= embed  #   
        GenKernesTable([("up",matmul,0),("trans(up)",matmul_transposed_b,1)], kernels, codes, M, N, K) 
        M=batch*ctx; N=embed; K=ffn   #   
        GenKernesTable([("down",matmul,0),("trans(down)",matmul_transposed_b,1)], kernels, codes, M, N, K) 
    # OutCLS
        M=2*ctx; N=151936; K= embed  #   
        GenKernesTable([("CLS",matmul,0),("trans(CLS)",matmul_transposed_b,1)], kernels, codes, M, N, K) 

    with open(path, "w") as f:
        for id, code in enumerate(codes):            
            f.write(code)  
  
    for kenel in kernels:
        header.append(kenel["declare"])
    header.append("\nconst TL_GEMM TL_GEMM_tables[] = {")
    for kenel in kernels:
        header.append(f"\t{{ {{{kenel["params"]}}},{kenel["name"]},\"{kenel["desc"]}\" }},")
    header.append(f"}};")    

def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)    

    import torch
    a = torch.randn(1024, 1024).cuda().bfloat16()
    b = torch.randn(1024, 1024).cuda().bfloat16()
    c = kernel(a, b)
    ref_c = a @ b
    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")
    # kernel = tilelang.compile(matmul, target="cuda")  ### so strange! Always failed to export_library
    # kernel.export_library("topk.so")
    path = "/home/cys/rnd/lic/src/Device/tilelang/gemm_v0.cu"
    with open(path, "w") as f:
        f.write(kernel.get_kernel_source())    
    print(f"CUDA Source@{path}")

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")





if __name__ == "__main__":
    main()
