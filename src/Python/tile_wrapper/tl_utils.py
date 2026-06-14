# Adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py
# ruff: noqa
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import re
import tilelang as T
import tilelang.language as tl
from tilelang.language import PrimFunc
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import itertools
from pathlib import Path
import shutil
from tilelang.autotuner import AutoTuner
import os
import glob
import tilelang
from tilelang.utils.target import determine_target
from tilelang.contrib.nvcc import default_compile_options
import json
import subprocess
import ctypes
# if 256, flashattn_bwd_atomic_add would "LayoutInference conflict" for ugly/strange reanson
TL_threads_per_block = 128

def Verify_ENV():
    print(f"\n======== Torch_{torch.version.cuda} cuda = {torch.cuda.get_device_capability()} ========")
    t_major, t_minor = map(int, torch.version.cuda.split("."))
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = subprocess.check_output([nvcc, "--version"]).decode()
        for line in out.splitlines():
            if "release" in line:
                v = line.split("release")[-1].strip().split(",")[0]
                n_major, n_minor = map(int, v.split("."))
        print(f"======== nvcc_{n_major}.{n_minor} ========")
    if n_major*1000 + n_minor != t_major*1000 + t_minor:
        print(f"\n======== CUDA version mismatch between tilelang({t_major}.{t_minor}) & nvcc({n_major}.{n_minor})! Dangerous!!! ========\n")        
    print(f"======== tilelang_{tilelang.__version__} target = {determine_target()} ========\n")   

    cuda = ctypes.CDLL("libcudart.so")
    MAX_SMEM_PER_BLOCK = 97      # cudaDevAttrMaxSharedMemoryPerBlockOptin
    device = torch.cuda.current_device()
    value = ctypes.c_int()
    ret = cuda.cudaDeviceGetAttribute(
        ctypes.byref(value),
        MAX_SMEM_PER_BLOCK,
        device
    )
    assert ret == 0
    print(f"Max shared memory per block (opt-in): {value.value} bytes")

    flags = default_compile_options()
    print(json.dumps(flags, indent=2))

def tl_clear_cache():
    cache_root = Path.home() / ".tilelang" / "cache" / "autotuner"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    AutoTuner._memory_cache.clear()

def auto_infer_current_arch():
    if torch.version.hip is not None:
        return get_arch("hip")
    if torch.cuda.is_available():
        return get_arch("cuda")
    elif torch.mps.is_available():
        return get_arch("metal")
    else:
        return get_arch("llvm")
    
def tl_smem_size(M, N, dtype):
    element_size = tl.dtype(dtype).itemsize
    return M * N * element_size

'''
    1. TileLang (and many other GPU DSLs) defaults to block_M == block_Nbecause it is simple, safe, and works well for the most common workloads, especially attention-style GEMMs.
    2. But TileLang does not allocate all buffers simultaneously! for example
        T.Pipelined(..., num_stages=2)means:
            Stage 0 uses one set of buffers
            Stage 1 reuses the same physical memory
'''
def tl_pick_tiling_shape(block_M, block_N, dtype=torch.float16, device_id=0, func=None, most=False):    
    props = torch.cuda.get_device_properties(device_id)
    sm_per_block = props.shared_memory_per_block
    target_ratio = 0.95  # avoid OOM
    target_sm = sm_per_block * target_ratio
    element_size = tl.dtype(dtype).itemsize
    if most:
        return block_M, block_N, sm_per_block

    while True:
        sm_usage = 0 if func is None else func(block_M,block_N)*element_size
        if sm_usage <= target_sm:
            break;
        block_M = block_M//2
        block_N = block_N//2

    sm_usage = 0 if func is None else func(block_M,block_N)*element_size
    return block_M, block_N, sm_usage

def get_gemm_heuristic_sweep() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version in {80}:
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version in {90}:
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 0, "thread_num": 128, "enable_rasteration": True}

def get_gemm_sweep(M, N, K, with_roller=False, topk=20):
    """
    Generate a list of kernel tuning configuration dictionaries for a tiled matrix-multiply.

    When with_roller is True this queries the MatmulTemplate roller to produce up to `topk` recommended
    configurations (device-specific TensorCore-friendly tilings). Each returned dict contains:
      - block_M, block_N, block_K: tile sizes
      - num_stages: pipeline staging (0 means no explicit staging)
      - thread_num: total threads used for the block
      - enable_rasteration: whether a rasterization/swizzle layout was recommended (note spelling)

    When with_roller is False this returns the Cartesian product of a fixed set of candidate
    parameters; the returned dicts use the backward-compatible key name "enable_rasteration" for that flag.

    Parameters:
        M, N, K (int): GEMM dimensions used to generate valid tile sizes.
        with_roller (bool): If True, use MatmulTemplate's roller to generate device-aware hints;
            otherwise use a predefined candidate grid.
        topk (int): Maximum number of roller hints to request when with_roller is True.

    Returns:
        List[dict]: A list of configuration dictionaries as described above.

    Raises:
        ValueError: if with_roller is True but the roller returns no hints.
    """
    if with_roller:
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        block_M = [64, 128]
        block_N = [64, 128]
        block_K = [32, 64]
        num_stages = [1, 2, 3]  #[0, 1, 2, 3]
        thread_num = [128] #[128, 256]
        enable_rasterization = [True, False]
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                enable_rasterization,
            )
        )

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # keep param name for backward-compat
            }
            for c in _configs
        ]
    return configs

def tl_pick_tiling_shape_v0(block_M=128, block_N = 128, dtype=torch.float16, device_id=0, scaleM=1, scaleN=1):    
    props = torch.cuda.get_device_properties(device_id)
    sm_per_block = props.shared_memory_per_block
    target_ratio = 0.95  # avoid OOM
    target_sm = sm_per_block * target_ratio

    while True:
        sm_usage = tl_smem_size(block_M,scaleM,dtype) + tl_smem_size(block_N,scaleN,dtype)
        if sm_usage <= target_sm:
            break;
        block_M = block_M//2
        block_N = block_N//2

    sm_usage = tl_smem_size(block_M,scaleM,dtype) + tl_smem_size(block_N,scaleN,dtype)
    return block_M, block_N, sm_usage


def get_optimal_block_sizes(gpu_model="default"):
    gpu_configs = {
        "A100": (128, 64),  # 107KB
        "V100": (64, 64),   # ~53KB
        "T4": (64, 32),     # ~27KB
        "RTX4090": (64, 64), # ~53KB
        "default": (64, 32), # 
    }
    return gpu_configs.get(gpu_model, (64, 32))



def Codes2Lines(id, code_0, title, replace16=False):
    assert(code_0 is not None)
    codes = code_0.replace("_kernel", title)
    # Both types are 16-bit​ and use the same IEEE 754 bfloat16 format:
    if replace16:
        codes = codes.replace("bfloat16_t", "__nv_bfloat16")
    lines = []
    header = None
    for line in codes.splitlines():        
        if id>0 and line.strip().startswith('#include'):
            continue
        if line=="#include <tl_templates/cuda/debug.h>":
            continue
        if header == None and line.strip().startswith('extern \"C\" __global__ void'):
            header = line
            header = header.replace("bfloat16_t", "__nv_bfloat16")
            continue
        lines.append(line)
    codes = '\n'.join(lines)
    return codes,header

DEVICE_NAMES = {
    "device_kernel.cu",
    "kernel.cu",
}

def Find_kernel_source(func_name, key="*.cu", copy_so = False):
    cache_dir = os.path.expanduser("~/.tilelang/cache")  #"/home/cys/.tilelang/cache/0.1.9_cuda_git84c5f812-x86_64/kernels/"
    key_files = glob.glob(os.path.join(cache_dir, "**", key), recursive=True)
    key_files = sorted( key_files, key=lambda f: os.path.getmtime(f), reverse=True )    # sort by modification time
    assert(len(key_files)>0)
    pattern = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(',
        re.MULTILINE
    )
    for key_file in key_files:
        dir_path = os.path.dirname(key_file)
        for cu_file in os.listdir(dir_path):
            full_path = os.path.join(dir_path, cu_file)        
            if not os.path.basename(cu_file) in DEVICE_NAMES:
                continue
            
            # print(full_path)
            with open(full_path, "r") as f:
                codes = f.read()
            isFind = False
            for m in pattern.finditer(codes):
                start = m.start()
                line_start = codes.rfind("\n", 0, start) + 1
                line_end = codes.find("\n", start)
                line = codes[line_start:line_end]                
                if m.group(1)==func_name+"_kernel":
                    print(f"\"{line}\"\n@{full_path}" )#m.group(1)
                    isFind = True
                    break
            if isFind:
                if copy_so:
                    so_from, so_to = dir_path + "/executable.so", f"/home/cys/rnd/lic/src/Device/tile_kernel/{func_name}.so"
                    shutil.copy(so_from, so_to)
                return codes

    # latest = max(cufiles, key=os.path.getmtime)
    # print("Latest cufiles:", latest)
    return None


'''
    Some lite kernels
'''
@T.jit(out_idx=[-1])
def inplace_scale_(
    block_M=128, block_N=128, dtype=tl.bfloat16, threads=TL_threads_per_block
):
    M_sym = tl.dynamic("M", dtype="int32")
    N_sym = tl.dynamic("N", dtype="int32")
    @tl.prim_func
    def tl_scale_(
        C: tl.Tensor((M_sym, N_sym), dtype),
        alpha: tl.float32
    ):
        grid_M = tl.ceildiv(M_sym, block_M)
        grid_N = tl.ceildiv(N_sym, block_N)
        with tl.Kernel(grid_N,grid_M,threads=threads) as (bx, by):     
            
            C_shared = tl.alloc_shared((block_M, block_N), dtype)
            C_local  = tl.alloc_fragment((block_M, block_N), dtype)
            tl.copy(
                C[by * block_M : by * block_M + block_M,
                  bx * block_N : bx * block_N + block_N],
                C_shared
            )

            # Compute
            for local_y, local_x in tl.Parallel(block_M, block_N):
                C_local[local_y, local_x] = C_shared[local_y, local_x] * alpha

            # TileLang automatically inserts boundary checks here as well
            tl.copy(
                C_local,
                C[by * block_M : by * block_M + block_M,
                  bx * block_N : bx * block_N + block_N]
            )

    return tl_scale_

def kernel_hash(func):
    src = inspect.getsource(func)
    return hashlib.sha256(src.encode()).hexdigest()

def Kernel2Codes(desc_kernel, kernel_metas, all_codes, block_M, block_N, dtype,sm_usage, *args ):
    # copy_so = True    
    assert(len(args)>=2)
    M,N = args[0], args[1]  
    # smem = tl_smem_size(block_M, block_N, dtype)  
    props = torch.cuda.get_device_properties(0)
    assert( sm_usage<=props.shared_memory_per_block )
    dim3 = [sm_usage, tl.ceildiv(N, block_N),tl.ceildiv(M, block_M),1, TL_threads_per_block,1,1]
    #batch, heads, seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups
    for desc, func, TRANS in desc_kernel:
        id = len(kernel_metas)
        # title = '_'.join(str(arg) for arg in args)
        title = f"_T{block_M}_{block_N}_S{sm_usage}_{dtype}"
        more = '_'.join(str(arg) for arg in args)
        kernel = func(block_M, block_N, dtype, *args)
        # print(kernel)   #assert isinstance(kernel, PrimFunc)
        # mod = kernel.get_module()
        # for k in mod.kernels:
        #     print(k.name, k.shared_memory)
        code_0 = kernel.get_kernel_source()
        # code_0 = Find_kernel_source(desc)
        codes,header = Codes2Lines(id, code_0,title)
        all_codes.append(codes)        

        assert header is not None
        kernel_name = re.split(r"[,:;| ()\t]", header)[4]
        params = []    
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                params.append(arg)

        
        params.extend(dim3)
        params = ",".join(map(str, params))   
        kernel_meata = {"declare":header,"name":kernel_name, "params":params, "desc":desc}
        
        kernel_metas.append(kernel_meata)
        print(f"---- {dim3}@{header} desc={desc}")
        # print(kernel_meata)
    return 

def Utils_wraper(jConfig, path, header):
    header.append("""

    """)
    kernels = []
    codes = []
    params,vocab = jConfig["model"]["parameter"]["transformer"], jConfig["model"]["vocab_size"]
    batch = jConfig["train"]["batch"]
    ctx,embed = params["Ctx"],params["Embed"]   
    dtype = tl.bfloat16
    block_M,block_N,sm_usage = tl_pick_tiling_shape(block_M=128, block_N = 128, dtype=dtype, func=lambda m, n: m*n )    
    Kernel2Codes([("delta_prelogits",inplace_scale_,0)], kernels, codes, block_M, block_N, dtype,sm_usage, 2*ctx, vocab )

    with open(path, "w") as f:
        for id, code in enumerate(codes):            
            f.write(code)  
  
    for kenel in kernels:
        header.append(kenel["declare"])
    # header.append("\nconst TL_GEMM TL_GEMM_tables[] = {")
    # for kenel in kernels:
    #     header.append(f"\t{{ {{{kenel["params"]}}},{kenel["name"]},\"{kenel["desc"]}\" }},")
    # header.append(f"}};")                        