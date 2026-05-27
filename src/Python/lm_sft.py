'''
4.0699334144592285  2.8075459003448486  2.0822982788085938  1.5184664726257324  1.329068660736084   1.4290963411331177

isPreToken=True
4.07163143157959    2.786594867706299   2.107296943664551   1.561959147453308   1.4008532762527466  1.4658167362213135
'''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import argparse 
from some_utils import OnInitInstance
from PreTokenizer import ChatJSONL2SFT,collate_Chat_JSONL
from functools import partial

def main():
    OnInitInstance(42)
    isPreToken = False
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    #load_dataset("trl-lib/Capybara", split="train")    # Example: conversational dataset -> apply chat template -> tokenize

    if isPreToken:
        dataset = load_dataset("json", data_files="assets/chat_data.jsonl", split="train")
        ds = JsonFile2Dataset(dataset, tokenizer, "out_file")   
        loader = DataLoader(ds, batch_size=2, shuffle=True)
    else:
        ds = Dataset.from_json("assets/chat_data.jsonl")    
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=partial(collate_Chat_JSONL, tokenizer=tokenizer))
    
    for epoch in range(1):
        for batch in loader:
            # print(batch)
            batch = {k: v.to(model.device) if hasattr(v, "to") else torch.as_tensor(v).to(model.device) for k, v in batch.items()}
            # print(batch)
            outputs = model(**batch)      # model computes CE loss if labels provided
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())
    
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