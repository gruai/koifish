

from transformers import AutoTokenizer
import json
# import tiktoken
from pathlib import Path

MAX_LEN = 1024

def filter_and_truncate(example, tokenizer):
    messages = example["messages"]

    # Tokenize each role
    total_tokens = 0
    for m in messages:
        total_tokens += len(tokenizer.encode(m["content"], add_special_tokens=False))

    # Drop if assistant response is too long
    assistant_msg = messages[-1]["content"]
    assistant_len = len(tokenizer.encode(assistant_msg, add_special_tokens=False))
    if assistant_len > MAX_LEN:
        return False

    # Truncate user message if needed
    while total_tokens > MAX_LEN and len(messages) >= 2:
        user_msg = messages[-2]["content"]
        user_tokens = tokenizer.encode(user_msg, add_special_tokens=False)
        if len(user_tokens) <= 10:
            break
        # Remove ~10% of tokens from the start
        user_tokens = user_tokens[len(user_tokens)//10:]
        messages[-2]["content"] = tokenizer.decode(user_tokens)
        total_tokens = sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in messages
        )

    return True

def count_tokens(enc, text: str) -> int:
    return len(enc.encode(text))

def truncate_to_max_tokens(enc, text: str, max_tokens: int) -> str:
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def ToSTR(x0):
    assert (x0 is not None)
    x = x0[0].detach().float().cpu()
    str= f"|avg|={x.abs().mean().item():.3e} [{x.min().item():.3e},{x.max().item():.3e}] n={x.numel()}"
    return str
'''
1.  PyTorch names things from the forward perspective, not the backward flow.
'''
def hook_model(model, HookGrad=0):   
    handles = [] 
    if HookGrad<=0:
        return handles
    
    assert (hasattr(model, "model"))    # GPT‑2 hasattr(model, "transformer"):
    layer_grads = {}  # layer_idx -> grad_input tensor
    ffn_down_inputs = {}  # layer_idx -> activation (forward)
    ffn_down_grads  = {}  # layer_idx -> gradient (backward)

    def make_layer_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_input[0] == Δin
            if grad_input is not None and grad_input[0] is not None:
                g = grad_input[0].detach()
                layer_grads[layer_idx] = g
                print(f"Layer {layer_idx:02d} | δ |avg|={g.abs().mean().item():.3e} | max={g.abs().max().item():.3e}")
        return hook
    
    def make_ffn_down_hook(layer_idx):
        def hook(module, input, output):
            # input is a tuple; first element is what we want
            ffn_down_inputs[layer_idx] = input[0].detach()
        return hook

    def make_ffn_down_grad_hook(layer_idx):
        def hook(module, grad_in, grad_out):
            # grad_input[0] = gradient w.r.t. down_proj input
            print(f"FFN_down {layer_idx:02d} | δ_in={ToSTR(grad_out[0])} δ_out={ToSTR(grad_in[0])}")
            
            # if grad_input[0] is not None:
            #     g = grad_input[0].detach()
            #     ffn_down_grads[layer_idx] = g
            #     print(f"FFN_down {layer_idx:02d} | δ_ffn |avg|={g.abs().mean().item():.3e} [{g.min().item():.3e},{g.max().item():.3e}]")
        return hook   
    
    final_rms_norm = model.model.norm
    for idx, layer in enumerate(model.model.layers):
        # h = layer.register_full_backward_hook(make_layer_hook(idx))
        # handles.append(h)
        mlp = layer.mlp
        h_fwd = mlp.down_proj.register_forward_hook(            make_ffn_down_hook(idx)        )
        h_bwd = mlp.down_proj.register_full_backward_hook( make_ffn_down_grad_hook(idx)        )
        handles += [h_fwd, h_bwd]
    
    def rms_forward_hook(module, inp, out):
        global rms_input, rms_output
        rms_input = inp[0].detach().float().cpu()
        rms_output = out.detach().float().cpu()

    def rms_backward_hook(module, grad_in, grad_out):
        global rms_grad
        # gout = grad_out[0].detach().float().cpu()
        # gin = grad_in[0].detach().float().cpu()
        print(f"lastRMS | δ_in={ToSTR(grad_out)} δ_out={ToSTR(grad_in)}")

    final_rms_norm.register_forward_hook(rms_forward_hook)
    final_rms_norm.register_full_backward_hook(rms_backward_hook)

    return handles