import sys
import os
import glob
import struct
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from PreTokenizer import TokenizedFile
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]
        

# python src/Python/parquet.py
def main(directory, output_dir, text_column, model_name, MAX_TOKENS=-1):
    """
    Main function to load parquet files, tokenize, and save.
    
    Args:
        directory: Directory containing parquet files
        output_file: Output .bin file path
        text_column: Name of column containing text (auto-detected if None)
        model_name: Qwen model name (default: Qwen/Qwen2.5-0.5B)
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)   #, use_fast=False
    pad_id = tokenizer.encode(tokenizer.pad_token)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})

    print(f"Loading parquet files from: {directory} ...\n")
    parquet_files = glob.glob(os.path.join(directory, "*.parquet"))      
    model = None
    max_length = MAX_TOKENS if MAX_TOKENS>0 else tokenizer.model_max_length
    most_line = 1000
    max_tokens_per_batch = (most_line+1) * (max_length+1)  # 100 sequences * max length
    buffer = np.zeros(max_tokens_per_batch, dtype=np.int32)
    print(f"max_length={max_length}\nparquet_files={parquet_files}")
    # for idx, file in enumerate(tqdm(parquet_files, desc="Process all parquet files")):
    for idx, file in enumerate(parquet_files):
        path = f"{output_dir}{Path(file).stem}___.bin"
        if os.path.isfile(path):
            continue
        print(f"[parquet] @{path} ...")  
        with TokenizedFile(model, path, tokenizer.vocab_size, masking=False) as f:
            dataset = load_dataset("parquet", data_files=file, split="train") #   split="train", "test"
            nAllSamp = len(dataset)
            if nAllSamp==0:
                print(f"No data found in the {file}!")
                continue

            nPass = 0  
            buffer_pos = 0         
            pbar = tqdm(dataset, total=nAllSamp, desc=f"Processing")
            for step, samp in enumerate(pbar):
                sample = samp['text']    
                try:
                    line_encoded = tokenizer.encode(sample, add_special_tokens=False,max_length=tokenizer.model_max_length,truncation=True )
                    if len(line_encoded) >= max_length or len(line_encoded)==0:
                        nPass = nPass+1
                        continue                    
                    line_encoded = line_encoded + [tokenizer.eos_token_id]
                    # line_encoded += pad_id
                except Exception as e:
                    print(f"Failed@{step} error={str(e)} sample={sample}")
                    continue
                arr = np.array(line_encoded, dtype=np.int32)
                buffer[buffer_pos:buffer_pos+len(arr)] = arr
                buffer_pos += len(arr)
                assert(buffer_pos<=max_tokens_per_batch)
                if (step+1) % most_line == 0:
                    try:                            
                        f.add_document(buffer[:buffer_pos])
                        pbar.set_description(f"{step:8d} samples(nPass={nPass}): toks={f.toks/1.0e6:.5g}M")
                        buffer_pos = 0    
                    except Exception as e:
                        print("❌ Batch concat / write failed:", e)
            
        
        


def read_tokens_from_bin(file_path):
    """Read tokens from the .bin file format"""
    with open(file_path, 'rb') as f:
        # Read total number of tokens (int64)
        total_tokens = struct.unpack('q', f.read(8))[0]
        
        # Read all tokens (int32)
        tokens = []
        for _ in range(total_tokens):
            token = struct.unpack('i', f.read(4))[0]
            tokens.append(token)
        
        return tokens, total_tokens



if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Tokenize parquet data with Qwen tokenizer')
    parser.add_argument('--directory', default="/home/cys/rnd/lic/Datasets/fine_code/data/", help='Directory containing parquet files')
    parser.add_argument('--output', default='/home/cys/rnd/lic/Datasets/fine_code/bin/1.bin', help='Output .bin file path')
    parser.add_argument('--text-column', default='text', help='Name of text column (auto-detected if not specified)')
    parser.add_argument('--model', default='/home/cys/rnd/lic/Models/Qwen3-0.6B/', help='Qwen model name')
    # parser.add_argument('--model-size', choices=['0.5B', '1.5B', '3B', '7B', '14B', '72B'], help='Qwen model size (shorthand)')
    
    args = parser.parse_args()    
    main(args.directory, args.output, args.text_column, args.model)

    # Example usage
    # tokens, count = read_tokens_from_bin('tokens.bin')
    # print(f"Loaded {count} tokens")
    # print(f"First 10 tokens: {tokens[:10]}")


