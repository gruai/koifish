import tilelang as T
import tilelang.language as tl
import re
from .tl_utils import Codes2Lines, tl_pick_tiling_shape, Kernel2Codes, TL_threads_per_block

'''
    v0
@tl.prim_func
def header_cls(
    pre_logits: tl.Tensor([N_sym, V], dtype),
    labels: tl.Tensor([N_sym], tl.int32),
    losses: tl.Tensor([N_sym], accum_dtype),
    grad_pre_logits: tl.Tensor([N_sym, V], dtype),
    nValidToken:tl.int
):
        with tl.Kernel(N_sym, threads=TL_threads_per_block) as n:
            z = tl.alloc_fragment([BV], dtype)
            max_z = tl.alloc_fragment([1], dtype)
            sum_exp = tl.alloc_fragment([1], accum_dtype)
            lse = tl.alloc_fragment([1], accum_dtype)    #   lse - Log of Sum of Exponentials            
            tl.fill(lse, -tl.infinity(accum_dtype))     # get lse byonline fomula: for first tile, exp2(lse - m) = exp2(-∞) = 0

            label = labels[n]
            label_tile = label // BV
            label_off = label % BV

            for i_v in tl.Pipelined(0, NV): #   Online LSE
                start = i_v * BV
                end = tl.min(V,start + BV)
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

            for i_v in tl.Pipelined(0, NV): #   Online LSE
                start = i_v * BV
                end = tl.min(V,start + BV)                
                tl.copy(pre_logits[n, start:end], z)
                for j in tl.Parallel(BV):# recompute exp(z_label - lse)
                    z[j] = tl.exp2(z[j] * scale - lse[0]) / nValidToken
                tl.copy(z, grad_pre_logits[n, start:end])    #online version                
                # p_label[i_v] = tl.cast(z[label_off], tl.float32) * nValidToken   #z[label_off]  
            tl.sync_threads()
            p_label = grad_pre_logits[n, label] * nValidToken
            # p_label[label_tile] = tl.max(p_label[label_tile], 1e-20)  
            losses[n] = -tl.log(p_label) 
            grad_pre_logits[n, label] = (p_label - 1.0) / nValidToken
'''         
      

'''
    1. N is the total number of tokens across the batch, and Vis the vocabulary size.
'''
@T.jit(out_idx=[2, 3])
def llm_header(
    block_M, block_N, dtype, V, nValidToken, threads=TL_threads_per_block
) -> "Callable":    
    N_sym = tl.dynamic("N", dtype="int32")
    accum_dtype = tl.float32
    scale = 1.44269504  # log2(e)   
    BV = min(T.next_power_of_2(V), 128)
    NV = tl.cdiv(V, BV) #1187
    assert(V % BV == 0)
    print(f"llm_header threads={threads} V={V}({BV}x{NV}) ...")

    @tl.prim_func
    def header_cls(
        pre_logits: tl.Tensor([N_sym, V], dtype),
        labels: tl.Tensor([N_sym], tl.int32),
        losses: tl.Tensor([N_sym], accum_dtype),
        grad_pre_logits: tl.Tensor([N_sym, V], dtype),
        nValidToken:tl.int
    ):
        with tl.Kernel(N_sym, threads=TL_threads_per_block) as n:
            label = labels[n]
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
                    end = tl.min(V,start + BV)
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

                for i_v in tl.Pipelined(0, NV): #   Online LSE
                    start = i_v * BV
                    end = tl.min(V,start + BV)                
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
                row = tl.alloc_fragment([V], dtype)
                tl.fill(row, dtype(0))
                tl.copy(row, pre_logits[n, :])
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
    params,vocab = jConfig["model"]["parameter"]["transformer"], jConfig["model"]["vocab_size"]
    # batch = 2 #jConfig["train"]["batch"]
    ctx,embed = params["Ctx"],params["Embed"]   
    dtype = tl.bfloat16
    block_M, block_N, sm_usage = tl_pick_tiling_shape(64,64,dtype=dtype,most=True)    
    Kernel2Codes([("classification@llm header",llm_header,0)], kernels, codes, block_M, block_N, dtype, sm_usage, vocab, 0 )    #batch*ctx

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