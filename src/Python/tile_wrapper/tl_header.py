import re
from .tl_utils import Codes2Lines, tl_pick_tiling_shape, Kernel2Codes, TL_threads_per_block
import tilelang as T
import tilelang.language as tl

import tilelang as T
import tilelang.language as tl

'''
Top-k tokens by confidence per sample.

Outputs:
  topk_token_ids   : [B, k]
  @T.jit(out_idx=[2])
def topk_confidence_kernel(
    block_T: int,
    k: int,
    dtype,
    V: int,
    threads: int = 128,
) -> "Callable":

    B_sym = tl.dynamic("B", dtype="int32")
    T_sym = tl.dynamic("T", dtype="int32")
    accum_dtype = tl.float32

    BV = min(T.next_power_of_2(V), 128)
    NV = tl.cdiv(V, BV)

    print(f"topk_confidence_kernel threads={threads} V={V}({BV}x{NV}) k={k}")

    @tl.prim_func
    def topk_confidence(
        logits: tl.Tensor([B_sym, T_sym, V], dtype),
        topk_token_ids: tl.Tensor([B_sym, k], tl.int32)
    ):
        # Shared memory for confidence + token index
        conf_shared = tl.shared([block_T], accum_dtype)
        idx_shared  = tl.shared([block_T], tl.int32)

        # ------------------------------------------------------------
        # Step 1: compute confidence = max(logits) per token
        # ------------------------------------------------------------
        with tl.Kernel(B_sym, T_sym, threads=threads) as (b, t):

            max_val = tl.min_value(accum_dtype)

            # Reduce over vocabulary
            for nv in tl.range(NV):
                for v in tl.range(BV):
                    if nv * BV + v < V:
                        val = logits[b, t, nv * BV + v]
                        max_val = tl.maximum(max_val, val)

            conf_shared[t] = max_val
            idx_shared[t]  = t

            tl.syncthreads()

            # --------------------------------------------------------
            # Step 2: Warp-level top-k selection
            # --------------------------------------------------------
            # Each warp maintains k best (confidence, token_id) pairs
            topk_val = tl.full([k], -tl.infinity(accum_dtype), accum_dtype)
            topk_idx = tl.full([k], -1, tl.int32)

            for i in tl.range(block_T):
                c = conf_shared[i]
                idx = idx_shared[i]

                # Insert into top-k
                for j in tl.range(k):
                    if c > topk_val[j]:
                        # Shift down
                        for m in tl.range(k - 1, j, -1):
                            topk_val[m] = topk_val[m - 1]
                            topk_idx[m] = topk_idx[m - 1]
                        topk_val[j] = c
                        topk_idx[j] = idx
                        break

            # Write output
            for j in tl.range(k):
                topk_token_ids[b, j] = topk_idx[j]

    return topk_confidence
'''

'''
    1. N is the total number of tokens across the batch, and Vis the vocabulary size.
'''
@T.jit(out_idx=[2, 3])
def llm_header(
    block_M, block_N, dtype, arg1, nValidToken, threads=TL_threads_per_block
) -> "Callable":    
    N_sym = tl.dynamic("N", dtype="int32")
    V_sym = tl.dynamic("V", dtype="int32")
    accum_dtype = tl.float32
    scale = 1.44269504  # log2(e)   
    BV = 128    #min(T.next_power_of_2(V_sym), 128)
    # assert(V % BV == 0)
    print(f"llm_header_tilelang threads={threads} ...") #V={V}({BV}x{NV}) 

    @tl.prim_func
    def header_cls(
        pre_logits: tl.Tensor([N_sym, V_sym], dtype),
        labels: tl.Tensor([N_sym], tl.int32),
        losses: tl.Tensor([N_sym], accum_dtype),
        grad_pre_logits: tl.Tensor([N_sym, V_sym], dtype),
        nValidToken:tl.int
    ):
        with tl.Kernel(N_sym, threads=TL_threads_per_block) as n:
            label = labels[n]            
            NV = tl.cdiv(V_sym, BV) 
            label_tile = label // BV
            label_off = label % BV
            mask = 0 if label >= 0 else 1

            if mask == 0: 
                z = tl.alloc_fragment([BV], dtype)
                max_z = tl.alloc_fragment([1], dtype)
                sum_exp = tl.alloc_fragment([1], accum_dtype)
                lse = tl.alloc_fragment([1], accum_dtype)    #   lse - Log of Sum of Exponentials            
                tl.fill(lse, -tl.infinity(accum_dtype))     # get lse byonline fomula: for first tile, exp2(lse - m) = exp2(-∞) = 0

                for i_v in tl.Pipelined(0, NV): #   Online LSE
                    start = i_v * BV
                    end = tl.min(V_sym,start + BV)
                    tl.copy(pre_logits[n, start:end], z)
                    tl.reduce_max(z, max_z, dim=0, clear=True)
                    for j in tl.Parallel(BV):
                        z[j] = tl.exp2(z[j] * scale - max_z[0] * scale)                
                    tl.reduce_sum(z, sum_exp, dim=0, clear=True)
                    lse[0] = (
                        max_z[0] * scale + tl.log2(
                            tl.exp2(lse[0] - max_z[0] * scale)
                            + sum_exp[0]
                        )
                    )

                for i_v in tl.Pipelined(0, NV): 
                    start = i_v * BV
                    end = tl.min(V_sym,start + BV)                
                    tl.copy(pre_logits[n, start:end], z)
                    for j in tl.Parallel(BV):# recompute exp(z_label - lse)
                        z[j] = tl.exp2(z[j] * scale - lse[0]) / nValidToken
                    tl.copy(z, grad_pre_logits[n, start:end])    #online version                
                    # p_label[i_v] = tl.cast(z[label_off], tl.float32) * nValidToken   #z[label_off]  
                tl.sync_threads()
            
                p_label = grad_pre_logits[n, label] * nValidToken
                p_label = tl.max(p_label, 1e-20)  
                losses[n] = -tl.log(p_label) 
                grad_pre_logits[n, label] = (p_label - 1.0) / nValidToken
            else:
                # row = tl.alloc_shared([V_sym], dtype)
                # tl.fill(row, dtype(0))
                # tl.copy(row, pre_logits[n, :])
                tl.fill(pre_logits[n, :], dtype(0))
                losses[n] = 0

    return header_cls


def CustomKernel(desc_kernel, kernel_metas, all_codes, N, vocab, *args ):    
    #batch, heads, seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups
    for desc, func, TRANS in desc_kernel:
        id = len(kernel_metas)
        # title = '_'.join(str(arg) for arg in args)
        title = ""  #f"_B{batch}_S{seq_len}_H{heads}_D{dim_qk}"
        more = '_'.join(str(arg) for arg in args)
        kernel = func(N, vocab, *args)
        code_0 = kernel.get_kernel_source()
        codes,header = Codes2Lines(id, code_0,title)
        all_codes.append(codes)

        assert header is not None
        kernel_name = re.split(r"[,:;| ()\t]", header)[4]
        params = [N, vocab, -1,-1,-1,-1, N, 1, 1, TL_threads_per_block,1,1 ]       
        params = ",".join(map(str, params))   
        kernel_meata = {"declare":header,"name":kernel_name, "params":params, "desc":desc}
        
        kernel_metas.append(kernel_meata)
        print(kernel_meata)
    return 

def HeaderCLS_wraper(jConfig, path, header):
    header.append("""

    """)
    kernels = []
    codes = []
    params = jConfig["model"]["parameter"]["transformer"]  
    ctx,embed = params["Ctx"],params["Embed"]   
    dtype = tl.bfloat16
    block_M, block_N, sm_usage = tl_pick_tiling_shape(64,64,dtype=dtype,most=True)    
    # Deprecated, to support dialect
    # vocabs = [151936, 66]
    # for vocab in vocabs:
    #     Kernel2Codes([(f"_V{vocab}",llm_header,0)], kernels, codes, block_M, block_N, dtype, sm_usage, vocab, 0 )   
    Kernel2Codes([(f"_",llm_header,0)], kernels, codes, block_M, block_N, dtype, sm_usage, 0, 0 )  

    with open(path, "w") as f:
        for id, code in enumerate(codes):            
            f.write(code)  
  
    for kenel in kernels:
        header.append(kenel["declare"])
    # header.append("\nconst TL_GEMM TL_GEMM_tables[] = {")
    # for kenel in kernels:
    #     header.append(f"\t{{ {{{kenel["params"]}}},{kenel["name"]},\"{kenel["desc"]}\" }},")
    # header.append(f"}};")    

def main():
    pass


if __name__ == "__main__":
    main()