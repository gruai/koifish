#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import enum
import faulthandler
import functools
import itertools
import json
import math
import traceback
import os
import pickle
import re
import signal
import struct
import sys
import time
import zipfile
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeVar
import multiprocessing as mp
import numpy as np
try:
    import datasets
    from transformers import AutoConfig, AutoTokenizer
    from types import SimpleNamespace
except ImportError:
    print("Error: transformers package is required.")
    print("Please run `pip install transformers` to install it.")


if TYPE_CHECKING:
    from typing import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray: TypeAlias = 'np.ndarray[Any, Any]'

# ARCH = gguf.MODEL_ARCH.LLAMA

DEFAULT_CONCURRENCY = 8

def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )


HEADERS_INFO = {
    "Qwen2.5": {   #   Qwen2.5-0.5B
        "magic": 20250520,
        "version": 1,
        "token_dtype": np.uint32,
        "bytes_per_token": 4,
        "alias": ["qwen25","Qwen_2.5","Qwen-2.5"]
    },
    "Qwen3": {
        "magic": 20251218,
        "version": 1,
        "bytes_per_token": 4,
        "token_dtype": np.uint32,
    },
}

def LoadTokenizer(model_path : dir, tokenizer_path=None):
    print(f"Loading tokenizer @{model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #   Token indices sequence length is longer than the specified maximum sequence length for this model (154484 > 131072). Running this sequence through the model will result in indexing errors
    max_len = tokenizer.model_max_length    #131072 ???
    hf_config = AutoConfig.from_pretrained(model_path)

    model = SimpleNamespace()
    model.tokenizer = tokenizer
    model.bos_token_id = hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    model.eos_token_id = hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    model.name = model_path.name
    model.version = 1
    for k,v in HEADERS_INFO.items():
        if k in model.name:
            model.head_info = v

    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]
    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())
    # Extract vocab and merge rules
    model.vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {''.join(tuple(merge if isinstance(merge, list) else merge.split())): i for i, merge in enumerate(merges)}
    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)
        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score
    model.max_token_length = max(len(t) for t in all_tokens)    

    if tokenizer_path is not None:
        with open(tokenizer_path, "wb") as out_f:
            # Header: max_token_length, bos_token_id, eos_token_id
            out_f.write(struct.pack("<I", model.max_token_length))
            out_f.write(struct.pack("<I", model.bos_token_id))
            out_f.write(struct.pack("<I", model.eos_token_id))

            for id, token in enumerate(all_tokens):
                token_bytes = internal_to_bytes(U2B, token)
                out_f.write(struct.pack("f", pseudo_scores[token])) # merge score
                out_f.write(struct.pack("<I", len(token_bytes))) # 4 bytes: token length
                out_f.write(token_bytes)                         # UTF-8 bytes

        print(f"Written tokenizer model to {tokenizer_path}")
    
    print(f"Successfully loaded tokenizer of {model_path}. bos=f{tokenizer.bos_token},eos=f{tokenizer.eos_token},pad=f{tokenizer.pad_token}")
    return model

worker_tokenizer = None

class TokenizedFile:
    def __init__(self, model, file_name: str,  vocab_size: int, masking: bool = False):
        self.file_name = file_name
        self.file_handle = None
        self.header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
        self.header[0] = model.head_info["magic"]
        self.header[1] = model.head_info["version"]   
        # self.header[2] = self.toks
        self.header[3] = model.head_info["bytes_per_token"]  
        self.toks = 0
        self.vocab_size = vocab_size
        self.has_masks = masking
        self.mask_list = []
        self.mask_rest = None

    def __enter__(self):
        self.file_handle = open(self.file_name, "wb+")
        # reserve space for the file header
        self.file_handle.write(('*' * 1023 + '\n').encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_masks:
            self._write_masks()
        self._write_header()
        self.file_handle.close()
        self.file_handle = None

    def add_document(self, tokens: np.ndarray, mask: Optional[np.ndarray] = None):
        assert self.file_handle is not None
        if mask is not None and self.has_masks is False:
            raise ValueError("Cannot add masking to a file that was not created with masking enabled")
        elif mask is None and self.has_masks is True:
            raise ValueError("Cannot add maskless tokens to a file that was created with masking enabled")

        tokens = np.array(tokens , dtype=np.int32)
        assert tokens.ndim == 1

        if mask is not None:
            assert len(mask) == len(tokens)
            self._record_mask(mask)

        self.file_handle.write(tokens.tobytes())
        self.toks += len(tokens)
        if self.toks >= 2**31:
            raise RuntimeError("cannot have more than 2**31 tokens in a single file")

    def _record_mask(self, mask: np.ndarray):
        mask = mask.astype(np.bool)
        if self.mask_rest is not None:
            full_mask = np.concatenate([self.mask_rest, mask], dtype=np.bool)
        else:
            full_mask = mask

        full_bytes = len(full_mask) // 8 * 8
        mask_bytes = full_mask[:full_bytes]
        self.mask_rest = full_mask[full_bytes:]
        self.mask_list.append(np.packbits(mask_bytes, bitorder='little'))

    def _write_masks(self):
        if self.mask_rest is not None and len(self.mask_rest) > 0:
            self.mask_list.append(np.packbits(self.mask_rest, bitorder='little'))
        for part in self.mask_list:
            self.file_handle.write(part.tobytes())

    def _write_header(self):
        assert self.file_handle is not None
        assert self.toks < 2**31, "token count too large" # ~2.1B tokens

        # construct the header        
        self.header[2] = self.toks # number of tokens after the 256*4 bytes of header
        self.header[9] = self.vocab_size
        self.header[10] = self.has_masks

        self.file_handle.seek(0)
        nWrite = self.file_handle.write(self.header)
        self.file_handle.seek(self.header.size*4)        
        # header_str = "BIN.TOK\n"  # 8 bytes
        # version = 2
        bytes_per_token = 4
        # self.file_handle.write(header_str.encode("ascii"))
        # self.file_handle.write(np.array([version, bytes_per_token, self.toks, self.vocab_size, self.has_masks], dtype=np.int32).tobytes())
        # self.file_handle.seek(256*4)



def init_worker(model):
    global worker_tokenizer
    worker_tokenizer = model.tokenizer  #AutoTokenizer.from_pretrained(model_name_arg)


def tokenize_example_worker(args: tuple) -> dict:
    example, key_func, seq_len = args
    """Worker function for multiprocessing"""
    if callable(key_func):
        example = key_func(example)
    else:
        example = example[key_func]
    if isinstance(example, str):
        tokens = worker_tokenizer(example, return_tensors='np', split_special_tokens=True,truncation=True).input_ids[0, ...]
        tokens = np.concatenate([tokens, [worker_tokenizer.eos_token_id]])
        return {"tokens": tokens}
    elif isinstance(example, tuple):
        assert len(example) == 2
        prompt = worker_tokenizer(example[0], return_tensors='np', split_special_tokens=True).input_ids[0, ...]
        response = worker_tokenizer(example[1], return_tensors='np', split_special_tokens=True).input_ids[0, ...]

        # fix up BOS/EOS tokens
        if response[0] == worker_tokenizer.bos_token_id:
            response = response[1:]
        if prompt[-1] == worker_tokenizer.eos_token_id:
            prompt = prompt[:-1]
        if prompt[0] != worker_tokenizer.bos_token_id and worker_tokenizer.bos_token_id is not None:
            prompt = np.concatenate([[worker_tokenizer.bos_token_id], prompt])

        tokens = np.concatenate([prompt, response, [worker_tokenizer.eos_token_id]])
        mask = np.concatenate([np.zeros_like(prompt), np.ones_like(response), [1]])
        if seq_len is not None:
            if len(tokens) > seq_len:
                # truncate, but ensure last token remains EOS
                tokens = tokens[:seq_len]
                tokens[seq_len-1] = worker_tokenizer.eos_token_id
                mask = mask[:seq_len]
                mask[seq_len-1] = 1
            else:
                tokens = np.pad(tokens, (0, seq_len - len(tokens)), mode="constant")
                mask = np.pad(mask, (0, seq_len - len(mask)), mode="constant")
    else:
        raise ValueError(f"unknown example type {type(example)}")
    return {"tokens": tokens, "mask": mask}

def process_single_file(model, ds_iter, key, file_name: str, vocab_size: int, max_tokens: int, masking: bool, seq_len: int, pool) -> tuple[bool, int]:
    def example_generator(f: TokenizedFile):
        try:
            while f.toks < max_tokens:
                yield next(ds_iter), key, seq_len
        except StopIteration:
            return

    has_more_data = False
    with TokenizedFile(model, file_name, vocab_size, masking=masking) as f:
        tokenized_examples = pool.imap(
            tokenize_example_worker,
            example_generator(f),
            chunksize=256
        )

        for tokens in tokenized_examples:
            f.add_document(**tokens)

            if f.toks >= max_tokens:
                has_more_data = True
                break

        return has_more_data, f.toks


def ProcessDataset(file_name: str, ds, key, split_name: str, max_tokens: int = None, *, model, is_tiny: bool = False, first_is_eval: int = -1, masking: bool = False, seq_len: int = None):
    num_processes = max(1, min(mp.cpu_count() // 2, 16))
    tokenizer = model.tokenizer
    ds_iter = iter(ds)
    file_index = 0
    total_tokens = 0
    max_size = 100_000_000
    if max_tokens and max_tokens < max_size:
        max_size = max_tokens

    Path(f"{file_name}").mkdir(parents=True, exist_ok=True)
    print(f"ProcessDataset mp={num_processes} @{file_name} ...")
    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model,)) as pool:
        while True:
            if first_is_eval > 0:
                if file_index == 0:
                    max_size = first_is_eval
                    output_filename = f"{file_name}/eval.bin"
                else:
                    max_size = 100_000_000
                    output_filename = f"{file_name}/{split_name}-{file_index-1:03d}.bin"
            elif is_tiny:
                output_filename = f"{file_name}/{split_name}.bin"
            else:
                output_filename = f"{file_name}/{split_name}-{file_index:03d}.bin"

            has_more_data, tokens_written = process_single_file(model, ds_iter, key, output_filename, tokenizer.vocab_size, max_size, masking, seq_len, pool)

            total_tokens += tokens_written
            print(f"Completed file {output_filename} with {tokens_written:,} tokens")
            file_index += 1

            if not has_more_data:
                break

            if max_tokens and total_tokens >= max_tokens:
                break

def _extract_gsm8k_example(example):
    return example["question"], example["answer"]


def _extract_limo_example(example):
    return example["question"], example["solution"]

def TokenizeDataset(dataset: str, model, out_dir: Path = "preTokenData", seq_len: int = None, localdir: Path = None):
    out_dir = str(out_dir)
    localdir = str(localdir)
    subsample = None
    is_tiny = False
    masking = False
    default_seq_len = None

    if dataset == "tiny-shakespeare":
        d = datasets.load_dataset("/home/cys/rnd/lic/Datasets/shake")
        dst = "tiny-shakespeare"
        key = "Text"
        test_split = "test"
        is_tiny = True
    elif dataset == "hellaswag":
        d = datasets.load_dataset("/home/cys/rnd/lic/Datasets/hellaswag")
        dst = "hellaswag"
        key = "text"
        test_split = "validation"
    elif dataset == "tiny-stories":
        d = datasets.load_dataset("roneneldan/TinyStories")
        dst = "tiny-stories"
        key = "text"
        test_split = "validation"
    elif dataset == "gsm8k":
        d = datasets.load_dataset("openai/gsm8k", "main")
        dst = "gsm8k"
        key = _extract_gsm8k_example
        test_split = "test"
        is_tiny = True
        masking = True
        default_seq_len = 512
    elif dataset == "fineweb-1b":
        d = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True)
        dst = "fineweb-1b"
        key = "text"
        test_split = 10_000_000
        subsample = 1_000_000_000    
    elif dataset in ["climb-1b", "climb-10b"]:
        dst = dataset
        key = "text"
        test_split = 10_000_000
        subsample = 1_000_000_000 if dataset=="climb-1b" else 10_000_000_000
        cluster_datasets = []
        cluster_sizes = [86, 121, 145, 386, 189, 1778, 1673, 117, 81, 734, 156, 2568, 90, 28, 23, 728, 702, 227, 116, 51]
        cluster_sizes = [c / sum(cluster_sizes) for c in cluster_sizes]
        for cluster_id in range(1,21):  # clusters 1-20            
            try:
                if localdir is not None:
                    cluster_ds = datasets.load_dataset(localdir,  f"cluster_id={cluster_id}", split="train", streaming=True)
                else:
                    cluster_ds = datasets.load_dataset("gvlassis/ClimbMix",  f"cluster_id={cluster_id}", split="train", streaming=True)                
            except:  
                error_details = traceback.format_exc()
                print(f"Error details: {error_details}")
                continue
            cluster_datasets.append(cluster_ds)
        mixed_dataset = datasets.interleave_datasets(
            cluster_datasets,
            probabilities=cluster_sizes,
            seed=42,
            stopping_strategy="all_exhausted"
        )
        d = {"train": mixed_dataset}
    elif dataset == "limo":
        d = datasets.load_dataset("GAIR/LIMO")["train"].train_test_split(test_size=40, seed=42)
        dst = "limo"
        key = _extract_limo_example
        test_split = "test"
        is_tiny = True
        masking = True
        default_seq_len = 16383
    else:
        assert False, f"unknown dataset {dataset}"   

    dst = out_dir #+ "/" + dst

    if seq_len is None and default_seq_len is not None:
        seq_len = default_seq_len
    elif seq_len is not None and seq_len <= 0:
        seq_len = None

    if isinstance(test_split, int):
        ProcessDataset(dst, d["train"], key, "train", subsample, is_tiny=is_tiny, first_is_eval=test_split, model=model, masking=masking, seq_len=seq_len)
    else:
        ProcessDataset(dst, d["train"], key, "train", subsample, is_tiny=is_tiny, model=model, masking=masking, seq_len=seq_len)
        ProcessDataset(dst, d[test_split], key, "eval", None, is_tiny=True, model=model, masking=masking, seq_len=seq_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tokenizer.egg from HF card")
    
    vocab_types = ["spm", "bpe", "hfft"]
    output_choices = ["f16","bf16","f8","nf4"]
    parser.add_argument("--awqpath",     type=Path,              help="Path to scale awq cache file", default=None)
    parser.add_argument("--dump",         action="store_true",    help="don't convert, just show what's in the model")
    parser.add_argument("--dumpsingle",  action="store_true",    help="don't convert, just show what's in a single model file")
    parser.add_argument("--vocabonly",   action="store_true",    help="extract only the vocab")
    parser.add_argument("--outtype",      choices=output_choices, help="output format - note: q8_0 may be very slow (default: f16 or f32 based on input)")
    parser.add_argument("--vocabdir",    type=Path,              help="directory containing tokenizer.model, if separate from model file")
    parser.add_argument("--vocabtype",   choices=vocab_types,    help="The vocabulary format used to define the tokenizer model (default: spm)", default="spm")
    parser.add_argument("--outdir",      type=Path,              help="path to write to; default: based on input")
    parser.add_argument("--localdir",      type=Path,             help="path of local dataset files")
    parser.add_argument("model",          type=Path,              help="Path to the local Hugging Face model directory (used for both input and output).")
    parser.add_argument("--ctx",          type=int,               help="model training context (default: based on input)")
    parser.add_argument("--concurrency",  type=int,               help=f"concurrency used for conversion (default: {DEFAULT_CONCURRENCY})", default=DEFAULT_CONCURRENCY)
    parser.add_argument("--bigendian",   action="store_true",    help="model is executed on big endian machine")
    parser.add_argument("--padvocab",    action="store_true",    help="add pad tokens when model vocab expects more than tokenizer metadata provides")
    parser.add_argument("--skipunknown", action="store_true",    help="skip unknown tensor names instead of failing")

    parser.add_argument("--dataset")
    parser.add_argument("--seqlen", default=None, type=int,
                        help="Sequence length to be used for padding in fine-tuning datasets. Set to -1 to disable padding."
                        "If no value is given, a dataset-specific default value will be used.")

    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Directory created: {args.outdir}")
        
    tokenizer_path = os.path.join(args.outdir, "tokenizer.dat")
    model_wrap = LoadTokenizer(args.model,tokenizer_path=tokenizer_path)
    if model_wrap is None:
        exit()
    
    if(args.dataset is not None):
        TokenizeDataset(dataset=args.dataset, localdir = args.localdir, model=model_wrap, out_dir=args.outdir, seq_len=args.seqlen)


