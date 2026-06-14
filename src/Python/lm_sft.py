'''
1. lr
    12e-5is widely considered the de facto industry-standard starting learning ratefor full-parameter supervised fine-tuning (SFT) of pretrained LLMs, especially in the 7B–13B range.
    30B – 70B 5e-6 – 2e-5 Larger models → lower LR
'''

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import argparse 
from some_utils import OnInitInstance
from PreTokenizer import ChatJSONL2SFT,collate_Chat_JSONL
from functools import partial
import torch.nn.functional as F
from lm_utils import hook_model, ToSTR

def main():
    OnInitInstance(42)
    isPreToken = False
    HookGrad = 0
    model_name = "/home/cys/rnd/lic/Models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"\n==========  isPreToken={isPreToken} tokenizer pad={tokenizer.pad_token} eos={tokenizer.eos_token_id} ========== ")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.train()

    handles = hook_model(model, HookGrad)    

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    #load_dataset("trl-lib/Capybara", split="train")    # Example: conversational dataset -> apply chat template -> tokenize

    if isPreToken:
        dataset = load_dataset("json", data_files="assets/chat_data.jsonl", split="train")
        ds = JsonFile2Dataset(dataset, tokenizer, "out_file")   
        loader = DataLoader(ds, batch_size=2, shuffle=True)
    else:
        ds = Dataset.from_json("assets/chat_data.jsonl")    
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=partial(collate_Chat_JSONL, tokenizer=tokenizer, padding="max_length"))
    
    optimizer.zero_grad()

    for batch in loader:        
        batch = {k: v.to(model.device) if hasattr(v, "to") else torch.as_tensor(v).to(model.device) for k, v in batch.items()}
        # print(batch)
        # ---------- forward WITHOUT internal loss ----------
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=None,  # ✅ critical
            output_hidden_states=True
        )
        
        logits = outputs.logits
        labels = batch["labels"]
        last_hidden = outputs.hidden_states[-1]
        last_hidden_grad = None
        last_hidden.register_hook(lambda g: setattr(last_hidden, "grad_buf", g.detach()))

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # ---------- manual loss ----------
        # loss =            F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
        per_token_loss =    F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(shift_labels.shape)
        mask = shift_labels != -100
        num_valid = mask.sum().item()
        true_losses = per_token_loss[mask]
        loss = true_losses.sum() / len(true_losses)    
        # Very important    loss != per_token_loss.mean()      
        print(f"loss={loss.item()} n={len(true_losses)}")     
        # ---------- hook ----------
        shift_logits_grad = None
        shift_logits.register_hook(
            lambda g: setattr(shift_logits, "grad_buf", g.detach())
        )        
        loss.backward()
        grads = shift_logits.grad_buf   # ∂{LOSS}/∂{prelogits}

        # ---------- optional dump ----------
        mask = shift_labels != -100
        B, T, V = shift_logits.shape
        label_grads = grads[torch.arange(B).unsqueeze(1),torch.arange(T).unsqueeze(0),shift_labels]
        if HookGrad > 0:
            items = zip(
                shift_labels[mask].flatten().tolist(),
                true_losses.flatten().tolist(),
                label_grads[mask].flatten().tolist()
            )
            print( "per-token = ", " ".join(f"({lbl},{loss:.2f},{grad:.2f})" for lbl, loss, grad in items)        )
            g = last_hidden.grad_buf        # ∂{LOSS}/∂{prelogits} x trans(W_proj) = ∂{LOSS}/∂{hidden state of last RMS-NORM)
            print(f"last_hidden={ToSTR(g)}", )
        # print(f"|avg|={g.abs().mean().item():.5e}({g.numel()}) [{g.min().item():.3e},{g.max().item():.3e}]", )

        optimizer.step()
        optimizer.zero_grad()

    for h in handles:   # Cleanup hooks
        h.remove()
    model.save_pretrained("/tmp/qwen3-sft-full")
    tokenizer.save_pretrained("/tmp/qwen3-sft-full")

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="log path of sweep test")
    parser.add_argument("--csv", type=str, help="log path of single csv file")
    parser.add_argument("--x_dir", type=str)
    parser.add_argument("--x", action='store_false') 
    # parser.add_argument("--stat", action='store_true')    
    args = parser.parse_args()
    main()
    exit()    