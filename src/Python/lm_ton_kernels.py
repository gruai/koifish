import torch
import json
import argparse
import triton
import triton.language as tl
from triton_wrapper.ton_qkv import QKV_wraper


"""
nvdisasm /home/cys/rnd/lic/src/Device/ton_kernel/add_kernel.cubin | grep add_kernel

"""

def InitJConfig(path):
    if path is not None:
        with open(path, 'r', encoding='utf-8') as f:
            jConfig = json.load(f)    
            return jConfig
    # some trial value
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--groups", type=int, default=16, help="groups")
    args = parser.parse_args()
    # kernel = gemm(1024, 1024, 1024, 128, 128, 32)   
    return None

def main():
    root = "/home/cys/rnd/lic"
    print(f"\n======== triton_{triton.__version__} target =  ========\n")
    
    
    return 

    metas = [
        {"config":f'{root}/scripts/qwen3.json', "path":f"{root}/src/Device/ton_kernel/qkv_cubin", "func": QKV_wraper},
        # {"config":f'{root}/scripts/qwen3.json', "path":f"{root}/src/Device/tile_kernel/utils.cu", "func": Utils_wraper},
        # {"config":f'{root}/scripts/qwen3.json', "path":f"{root}/src/Device/tile_kernel/gemm_v0.cu", "func": Matmul_wraper},
        
        # {"config":f'{root}/scripts/qwen3.json', "path":f"{root}/src/Device/tile_kernel/header_cls.cu", "func": HeaderCLS_wraper}
    ]
    
    #print(tilelang.__version__)
    header =[f"#pragma once\n\n#define KOIFISH_TL_INFO \"tilelang_{tilelang.__version__}\"\n\n#ifdef __USE_TILELANG__\n"]
    
    for meta in metas:     
        config = InitJConfig(meta["config"])
        model_param = config
        meta["func"](model_param, meta["path"], header)
    
    
    header.append(f"\n#else\n\n#endif")
    with open(f"{root}/src/Device/tile_kernel/tl_kernels.hpp", "w") as f:
        codes = '\n'.join(header)
        f.write(codes)

    print(f"\n======== generated all kernels!!! {meta["func"]}@{meta["path"]} ========")


if __name__ == "__main__":
    main()