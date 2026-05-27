"""
    from    tilelang/examples/flash_attention/
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as tl
import argparse
from functools import partial
import re
from .tl_utils import Codes2Lines, tl_pick_tiling_shape, Kernel2Codes, TL_threads_per_block

tpTorch = torch.bfloat16
bM_forw,bN_forw = 128,64

@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(block_M, block_N, dtype, batch, heads, seq_len, dim_qk, dim_v, is_causal, groups=1):
    scale = (1.0 / dim_qk) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    accum_dtype = tl.float32
    print(f"Q={q_shape} K={k_shape}")

    @tl.prim_func
    def flash_fwd(
        Q: tl.Tensor(q_shape, dtype),  # type: ignore
        K: tl.Tensor(k_shape, dtype),  # type: ignore
        V: tl.Tensor(v_shape, dtype),  # type: ignore
        Output: tl.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        lse: tl.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with tl.Kernel(tl.ceildiv(seq_len, block_M), heads, batch, threads=TL_threads_per_block) as (bx, by, bz):
            # alloc_shared m*(dim_qk+n) + n*(dim_qk+dim_v)
            Q_shared = tl.alloc_shared([block_M, dim_qk], dtype)
            K_shared = tl.alloc_shared([block_N, dim_qk], dtype)
            V_shared = tl.alloc_shared([block_N, dim_v], dtype)
            acc_s = tl.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = tl.alloc_shared([block_M, block_N], dtype)
            acc_o = tl.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = tl.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = tl.alloc_fragment([block_M], accum_dtype)
            scores_scale = tl.alloc_fragment([block_M], accum_dtype)
            scores_sum = tl.alloc_fragment([block_M], accum_dtype)
            logsum = tl.alloc_fragment([block_M], accum_dtype)

            tl.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            tl.fill(acc_o, 0)
            tl.fill(logsum, 0)
            tl.fill(scores_max, -tl.infinity(accum_dtype))
            loop_range = tl.ceildiv((bx + 1) * block_M, block_N) if is_causal else tl.ceildiv(seq_len, block_N)
            for k in tl.Pipelined(loop_range, num_stages=1):
                tl.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                if is_causal:
                    for i, j in tl.Parallel(block_M, block_N):
                        acc_s[i, j] = tl.if_then_else(bx * block_M + i >= k * block_N + j, 0, -tl.infinity(acc_s.dtype))
                else:
                    for i, j in tl.Parallel(block_M, block_N):
                        acc_s[i, j] = tl.if_then_else(k * block_N + j >= seq_len, -tl.infinity(acc_s.dtype), 0)
                tl.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=tl.GemmWarpPolicy.FullRow)
                tl.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                tl.copy(scores_max, scores_max_prev)
                tl.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in tl.Parallel(block_M):
                    scores_max[i] = tl.max(scores_max[i], scores_max_prev[i])
                for i in tl.Parallel(block_M):
                    scores_scale[i] = tl.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in tl.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in tl.Parallel(block_M, block_N):
                    acc_s[i, j] = tl.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                tl.copy(acc_s, acc_s_cast)
                tl.gemm(acc_s_cast, V_shared, acc_o, policy=tl.GemmWarpPolicy.FullRow)
                tl.reduce_sum(acc_s, scores_sum, dim=1)
                for i in tl.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in tl.Parallel(block_M, dim_v):
                acc_o[i, j] /= logsum[i]
            tl.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            for i in tl.Parallel(block_M):
                logsum[i] = tl.log2(logsum[i]) + scores_max[i] * scale
            tl.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd

'''
    O: output of attention (after softmax × V) dO: gradient of loss w.r.t. O
    O*dO
'''
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(block_M, block_N, dtype, batch, heads, seq_len, dim_v):
    accum_dtype = tl.float32
    shape = [batch, seq_len, heads, dim_v]
    blk = block_M   #32

    @tl.prim_func
    def flash_bwd_preprocess(
        O: tl.Tensor(shape, dtype),  # type: ignore
        dO: tl.Tensor(shape, dtype),  # type: ignore
        Delta: tl.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with tl.Kernel(heads, tl.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = tl.alloc_fragment([blk, blk], dtype)
            do = tl.alloc_fragment([blk, blk], dtype)
            acc = tl.alloc_fragment([blk, blk], accum_dtype)
            delta = tl.alloc_fragment([blk], accum_dtype)
            tl.clear(acc)
            for k in range(tl.ceildiv(dim_v, blk)):
                tl.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                tl.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in tl.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            tl.reduce_sum(acc, delta, 1)
            tl.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_preprocess


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return tl.Layout(dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_postprocess(block_M, block_N, dtype, batch, heads, seq_len, dim_qk):
    accum_dtype = tl.float32
    shape = [batch, seq_len, heads, dim_qk]
    blk = block_M   #64

    @tl.prim_func
    def flash_bwd_postprocess(
        dQ: tl.Tensor(shape, accum_dtype),  # type: ignore
        dQ_out: tl.Tensor(shape, dtype),  # type: ignore
    ):
        with tl.Kernel(tl.ceildiv(seq_len, blk), heads, batch, threads=TL_threads_per_block) as (bx, by, bz):
            tl.annotate_layout({dQ: make_dq_layout(dQ)})
            tl.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_postprocess

def make_qkT_layout(qkT):
    return tl.Layout(qkT.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])

# dQ is accumulated in fp32 for numerical stability, since it is
# directly used by optimizer and cannot be corrected later.
# dK and dV are intermediate buffers that will be reduced across
# groups and written back, so bf16 precision is sufficient.

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd_atomic_add(block_M, block_N, dtype, batch, heads, seq_len, dim_qk, dim_v, is_causal, threads=TL_threads_per_block, num_stages=2, groups=1):
    sm_scale = (1.0 / dim_qk) ** 0.5
    scale = (1.0 / dim_qk) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    accum_dtype = tl.float32

    @tl.prim_func
    def flash_bwd(
        Q: tl.Tensor(q_shape, dtype),  # type: ignore
        K: tl.Tensor(k_shape, dtype),  # type: ignore
        V: tl.Tensor(v_shape, dtype),  # type: ignore
        dO: tl.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        lse: tl.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        Delta: tl.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        dQ: tl.Tensor(q_shape, accum_dtype),  # type: ignore
        dK: tl.Tensor(k_shape, accum_dtype),  # type: ignore
        dV: tl.Tensor(v_shape, accum_dtype),  # type: ignore
    ):
        with tl.Kernel(heads, tl.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):    # 16,16, 8
            # alloc_shared: block_M*(dim_qk*2+dim_v*2+block_N*3)+block_N*(dim_qk+dim_v+2)
            K_shared = tl.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = tl.alloc_shared([block_M, block_N], dtype)
            q = tl.alloc_shared([block_N, dim_qk], dtype)
            V_shared = tl.alloc_shared([block_M, dim_v], dtype)
            qkT = tl.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = tl.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = tl.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = tl.alloc_fragment([block_M, block_N], dtype)
            lse_shared = tl.alloc_shared([block_N], accum_dtype)
            delta = tl.alloc_shared([block_N], accum_dtype)
            do = tl.alloc_shared([block_N, dim_v], dtype)
            dv = tl.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = tl.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = tl.alloc_fragment([block_N, dim_qk], accum_dtype)
            dk_shared = tl.alloc_shared([block_M, dim_qk], accum_dtype)
            dv_shared = tl.alloc_shared([block_M, dim_v], accum_dtype)

            tl.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                }
            )

            tl.copy(K[bz, by * block_M : (by + 1) * block_M, bx // groups, :], K_shared)
            tl.copy(V[bz, by * block_M : (by + 1) * block_M, bx // groups, :], V_shared)
            tl.clear(dv)
            tl.clear(dk)
            loop_st = tl.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = tl.ceildiv(seq_len, block_N)
            for k in tl.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                tl.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                tl.clear(qkT)
                tl.gemm(K_shared, q, qkT, transpose_B=True, policy=tl.GemmWarpPolicy.FullRow)
                tl.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in tl.Parallel(block_M, block_N):
                    qkT[i, j] = tl.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in tl.Parallel(block_M, block_N):
                        qkT[i, j] = tl.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)
                tl.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                tl.clear(dsT)
                tl.gemm(V_shared, do, dsT, transpose_B=True, policy=tl.GemmWarpPolicy.FullRow)
                tl.copy(qkT, qkT_cast)
                
                tl.gemm(qkT_cast, do, dv, policy=tl.GemmWarpPolicy.FullRow)

                tl.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                for i, j in tl.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                tl.gemm(dsT_cast, q, dk, policy=tl.GemmWarpPolicy.FullRow)

                tl.copy(dsT_cast, dsT_shared)
                tl.clear(dq)
                tl.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in tl.Parallel(block_N, dim_qk):
                    tl.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            tl.copy(dv, dv_shared)
            tl.atomic_add(dV[bz, by * block_M : (by + 1) * block_M, bx // groups, :], dv_shared)
            tl.copy(dk, dk_shared)
            tl.atomic_add(dK[bz, by * block_M : (by + 1) * block_M, bx // groups, :], dk_shared)
    return flash_bwd
    
def MyTest(kernels,codes,dtype,
    BATCH: int = 1,H: int = 32,N_CTX: int = 256,D_HEAD_QK: int = 192,D_HEAD_V: int = 128,groups: int = 16,
    causal: bool = False,    use_atomic: bool = True,
):
    rtol, atol = 1.0e-1, 5e-2   # for bfloat16
    flops_per_qk = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_QK
    flops_per_v = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_V
    total_flops = 3 * flops_per_qk + 2 * flops_per_v
    if causal:
        total_flops *= 0.5
    head_kv = H // groups
    Q = torch.empty(BATCH, N_CTX, H, D_HEAD_QK, dtype=tpTorch, device="cuda").normal_().requires_grad_()    
    K = torch.empty(BATCH, N_CTX, head_kv, D_HEAD_QK, dtype=tpTorch, device="cuda").normal_().requires_grad_()
    V = torch.empty(BATCH, N_CTX, head_kv, D_HEAD_V, dtype=tpTorch, device="cuda").normal_().requires_grad_()
    dO = torch.empty(BATCH, N_CTX, H, D_HEAD_V, dtype=tpTorch, device="cuda").normal_().requires_grad_()
    shape_q = [BATCH, N_CTX, H, D_HEAD_QK]
    shape_k = [BATCH, N_CTX, head_kv, D_HEAD_QK]
    shape_v = [BATCH, N_CTX, head_kv, D_HEAD_V]
    dq = torch.zeros(shape_q, dtype=torch.float32, device=Q.device)
    dk = torch.zeros(shape_k, dtype=torch.float32, device=Q.device)
    dv = torch.zeros(shape_v, dtype=torch.float32, device=Q.device)

    mod = flashattn_fwd(bM_forw,bN_forw, tl.bfloat16, BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, groups)
    O, lse = mod(Q,K,V)

    bM_back,bN_back = 64,32
    bM_back,bN_back = 64,16
    bM_back,bN_back,sm_back = tl_pick_tiling_shape(bM_back,bN_back, dtype=dtype, func=lambda m, n: m*(512+n*3)+n*(258), most=True)
    bwd_prep = flashattn_bwd_preprocess(bM_back,bN_back,tl.bfloat16,BATCH, H, N_CTX, D_HEAD_V)    
    delta = bwd_prep(O, dO) # [batch, heads, seq_len]
    kernel = flashattn_bwd_atomic_add(bM_back,bN_back, tl.bfloat16, BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, threads=TL_threads_per_block, num_stages=2, groups=groups            )

    kernel(Q, K, V, dO, lse, delta, dq, dk, dv)
    bwd_post = flashattn_bwd_postprocess(bM_back,bN_back,tl.bfloat16,BATCH, H, N_CTX, D_HEAD_QK)
    dq = bwd_post(dq)
    dk = dk.to(tpTorch)
    dv = dv.to(tpTorch)
    Kernel2Codes([("flash_bwd",flashattn_bwd_atomic_add,0)], kernels, codes, bM_back,bN_back, dtype, sm_back, BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, True, TL_threads_per_block, 2, groups )                                                            

    # attention = _attention.apply
    # O = attention(Q, K, V, causal, groups, use_atomic)
    # O.backward(dO, retain_graph=True)
    # dQ, Q.grad = Q.grad.clone(), None
    # dK, K.grad = K.grad.clone(), None
    # dV, V.grad = V.grad.clone(), None

    O_ref = ref_program(Q, K, V, causal, groups)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    torch.testing.assert_close(O, O_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dv, dV_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dk, dK_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dq, dQ_ref, rtol=rtol, atol=atol)
    print("All checks passed.✅")

    # def run():
    #     O_ref.backward(dO, retain_graph=True)

    # def run1():
    #     O.backward(dO, retain_graph=True)

    # from tilelang.profiler import do_bench

    # latency = do_bench(run, warmup=500)
    # print("torch: {:.2f} ms".format(latency))
    # print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = do_bench(run1, warmup=500)
    # print("tilelang: {:.2f} ms".format(latency))
    # print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))

def QKV_wraper(jConfig, path, header):
#     header.append("""
# // using TL_QKV_KERNEL = void (*)(const __nv_bfloat16* , __nv_bfloat16* , const __nv_bfloat16* , const __nv_bfloat16*, float* );
# struct TL_QKV {
#     const unsigned int params[12];    
#     const char* desc;
    
#     dim3 Grid3() {
#         unsigned int N = sizeof(params) / sizeof(int);
#         return {params[N - 6], params[N - 5], params[N - 4]};
#     }
#     dim3 Block3() {
#         unsigned int N = sizeof(params) / sizeof(int);
#         return {params[N - 3], params[N - 2], params[N - 1]};
#     }
# };
#     """)
    kernels = []
    codes = []
    params = jConfig["model"]["parameter"]["transformer"]
    batch = jConfig["train"]["batch"]
    ctx,embed,ffn,q_head,kv_head,head_dim = params["Ctx"],params["Embed"],params["Ffn"],params["Head"],params["KVHead"],params["head_dim"]
    groups = q_head//kv_head
    dtype = tl.bfloat16
    assert head_dim==128
    global bM_forw,bN_forw
    
    bM_forw,bN_forw,sm_forw = tl_pick_tiling_shape(bM_forw,bN_forw, dtype=dtype, func=lambda m, n: m*(128+n)+n*256) 

    MyTest(kernels,codes,dtype, batch,q_head,ctx,head_dim,head_dim,q_head//kv_head,True,True)
   
        #  m*(dim_qk+n) + n*(dim_qk+dim_v)
    Kernel2Codes([("flash_fwd",flashattn_fwd,0)], kernels, codes, bM_forw,bN_forw, dtype, sm_forw, batch, q_head, ctx, head_dim, head_dim, True, groups )
    #   bwd_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD_V)    
    block_M,block_N,sm_usage = tl_pick_tiling_shape(block_M=32, block_N = 32, dtype=dtype)
    Kernel2Codes([("flash_bwd_preprocess",flashattn_bwd_preprocess,0)], kernels, codes, block_M, block_N, dtype, sm_usage, batch, q_head, ctx, head_dim )
    #   bwd_post = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD_QK)
    block_M,block_N,sm_usage = tl_pick_tiling_shape(block_M=64, block_N = 64, dtype=dtype)
    Kernel2Codes([("flash_bwd_postprocess",flashattn_bwd_postprocess,0)], kernels, codes, block_M, block_N, dtype, sm_usage, batch, q_head, ctx, head_dim )
    
    with open(path, "w") as f:
        for id, code in enumerate(codes):            
            f.write(code)  
    
    for kenel in kernels:
        header.append(kenel["declare"])
    # header.append("\nconst TL_QKV TL_QKV_tables[] = {")
    # for kenel in kernels:
    #     header.append(f"\t{{ {{{kenel["params"]}}},{kenel["name"]},\"{kenel["desc"]}\" }},")
    # header.append(f"}};")  

def ref_program(Q, K, V, is_causal, groups=1):
    # Q: [B, T, HQ, D_QK]
    # K: [B, T, HK, D_QK]
    # V: [B, T, HV, D_V]
    # HQ = HKV * groups
    assert Q.size(2) == K.size(2) * groups, f"Q.size(2): {Q.size(2)}, K.size(2): {K.size(2)}, groups: {groups}"
    assert Q.size(2) == V.size(2) * groups, f"Q.size(2): {Q.size(2)}, V.size(2): {V.size(2)}, groups: {groups}"

    dim_qk = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim_qk, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--h", type=int, default=16, help="Number of heads")
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--d_head_qk", type=int, default=128, help="Head dimension for Q/K")
    parser.add_argument("--d_head_v", type=int, default=128, help="Head dimension for V")
    parser.add_argument("--causal", action="store_true", help="Causal flag")
    parser.add_argument("--groups", type=int, default=2, help="groups")
    parser.add_argument("--use_atomic", action="store_true", default=False, help="Use atomic add for dK/dV")
    parser.add_argument("--use_split", action="store_true", default=False, help="Use split for dK/dV")
    args = parser.parse_args()

    # Handle backward compatibility and logic
    if args.use_split:
        use_atomic = False
    elif args.use_atomic:
        use_atomic = True
    else:
        # Default: use atomic
        use_atomic = True
    args.causal = True
    MyTest(args.batch, args.h, args.n_ctx, args.d_head_qk, args.d_head_v, args.groups, args.causal, use_atomic)
