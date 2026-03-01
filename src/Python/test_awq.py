# - The last tested configuration used Torch 2.6.0 and Transformers 4.51.3.  
# https://github.com/vllm-project/llm-compressor
# pip uninstall -y transformers
# pip install transformers==4.51.3 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import torch
import pandas as pd
import numpy as np
from awq.utils.utils import get_best_device

device = get_best_device()

def Quant(model_path):    
    quant_path = 'Qwen-awq'
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data="/home/cys/Datasets/pile-val-backup", split="validation")

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)
    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros

def save_dequantized_to_csv(
    dequantized_tensor: torch.Tensor,
    title: str,
    csv_path: str = None,    
    flatten: bool = False,  # True = single column, False = 2D grid
    decimal_places: int = 6  # Control precision to avoid bloated files
) -> None:    
    dequantized_np = dequantized_tensor.cpu().float().numpy()
    if flatten:
        # Flatten to 1D and reshape into a single column
        flattened = dequantized_np.flatten().reshape(-1, 1)
        # Round to reduce precision (optional but recommended)
        flattened = flattened.round(decimal_places)
        # Use pandas to save (simple and fast)
        df = pd.DataFrame(flattened, columns=["Dequantized_Weight"])
        # df.to_csv(csv_path, index=False)
    else:
        # Keep as 2D grid (each row = out_feature, each column = in_feature)
        dequantized_np = dequantized_np.round(decimal_places)
        # Save with row/column labels for clarity
        df = pd.DataFrame(
            dequantized_np,
            columns=[f"{i}" for i in range(dequantized_np.shape[1])],
            index=[f"{title}_{i}" for i in range(dequantized_np.shape[0])]
        )
    if csv_path is not None:
        df.to_csv(csv_path, index=True)  # Keep index for row labels
    else:
        # pd.set_option('display.max_columns', 80)          # Show max 5 columns (reduces line width)
        # pd.set_option('display.width', 80)               # Max line width (characters)
        pd.set_option('display.max_colwidth', 80)        # Max width per column (characters)
        print(df)
    return 

# @def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
def Dequant_1(title, qweight,scales,qzeros,group_size: int = 128,    bits: int = 4):
    dot_index = title.find('.')
    title = title[dot_index+1:]
    in_features, out_features_packed = qweight.shape
    out_features = out_features_packed * (32 // bits)    
    
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    save_dequantized_to_csv(torch.bitwise_and(iweight, (2**bits) - 1), "qW") 
    # save_dequantized_to_csv(torch.bitwise_and(izeros, (2**bits) - 1), "zero")    
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)# Reverse the order of the iweight and izeros tensors
    save_dequantized_to_csv(torch.bitwise_and(iweight, (2**bits) - 1), "qWi") 
    
    # overflow checks

    save_dequantized_to_csv(scales, "scales")    
    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    save_dequantized_to_csv(scales, "scales_1")    
    izeros = izeros.repeat_interleave(group_size, dim=0)
    save_dequantized_to_csv(izeros, "zeroi")
    dequantized = (iweight - izeros) * scales
    dequantized = dequantized.to(dtype=torch.bfloat16)
    # save_dequantized_to_csv(dequantized, title)    #, "dequantized_.csv"
    tensor_flat = dequantized.flatten().cpu().float().numpy()
    n = 8
    formatted_list = [f"{num:.6g}" for num in np.concatenate((tensor_flat[:n], tensor_flat[-n:]), axis=0).tolist() ]
    # np.set_printoptions(precision=4, suppress=False, linewidth=np.inf)
    print(f"{title} = {', '.join(formatted_list)}")
    # print(f"{title} = {formatted_list}")

    return dequantized


def Dequant(model_path):
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )    
    level = 1
    # Iterate through all modules
    for name, module in model.named_modules():
        # Check for attributes of a quantized linear layer
        if not hasattr(module, 'qweight') and not hasattr(module, 'scales'):
            # if hasattr(module, 'weight') and name=="model.model.layers.0.input_layernorm":
            #     print(f"{name} = {module.weight}")
            continue
        if level == 0:
            print(f"Layer: {name}")
            print(f"  qweight shape: {module.qweight.shape}")
            print(f"  scales shape: {module.scales.shape}")
            if hasattr(module, 'qzeros'):
                print(f"  qzeros shape: {module.qzeros.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  bias shape: {module.bias.shape}")
            print("-" * 80)
        qzeros = None
        if hasattr(module, 'qzeros'):
            qzeros = module.qzeros      # INT4 zero points, packed in INT32            
        Dequant_1(name, module.qweight,module.scales,qzeros)

def Chat(quant_path):    
    tokenizer = AutoTokenizer.from_pretrained(quant_path)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model = AutoAWQForCausalLM.from_quantized(
        quant_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    prompt = [
        
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "when the smoke is going down,"},
    # {"role": "user", "content": \
    #         "You're standing on the surface of the Earth. "\
    #         "You walk one mile south, one mile west and one mile north. "\
    #         "You end up exactly where you started. Where are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    ).to(device)

    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=256,
        streamer=streamer,
    )
    # print(outputs)

if __name__ == '__main__':
    model_path = "/home/cys/rnd/lic/Models/Qwen3-0.6B"    #'Qwen/Qwen2.5-14B-Instruct'
    quant_path = "/home/cys/Github/infer/AutoAWQ-main/Qwen3-0.6B-awq/"
    # quant_path = "/home/cys/rnd/lic/Models/Qwen3-600M-AWQ"  #   600M 4B 8B 32B(OOM) packed_8B(fail)
    # Quant(model_path)
    Dequant("/home/cys/Github/infer/AutoAWQ-main/Qwen3-0.6B-awq/")  
    # Chat(quant_path)