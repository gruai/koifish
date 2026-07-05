'''
    1.  tokenizer.convert_tokens_to_ids("<|endoftext|>")  # → 151643
        tokenizer.convert_ids_to_tokens(151643)          # → "<|endoftext|>"
'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import traceback
import os
import signal
import struct
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeVar
from some_utils import list_depth
import multiprocessing as mp
import numpy as np
from functools import partial
import torch
import random
from tqdm import tqdm
import re
import threading
try:
    import datasets
    from datasets import load_dataset, Features, Sequence, Value
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
        self.fd = None
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
        self.lock = threading.Lock()

    def __enter__(self):
        self.fd = os.open(self.file_name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        # self.fd = os.open(self.file_name, "wb+")
        # reserve space for the file header
        os.write(self.fd, ('*' * 1023 + '\n').encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_masks:
            self._write_masks()
        self._write_header()
        os.close(self.fd)
        self.fd = None

    def add_document(self, tokens: np.ndarray, mask: Optional[np.ndarray] = None):
        assert self.fd is not None
        if mask is not None and self.has_masks is False:
            raise ValueError("Cannot add masking to a file that was not created with masking enabled")
        elif mask is None and self.has_masks is True:
            raise ValueError("Cannot add maskless tokens to a file that was created with masking enabled")

        tokens = np.array(tokens , dtype=np.int32)
        assert tokens.ndim == 1

        if mask is not None:
            assert len(mask) == len(tokens)
            self._record_mask(mask)

        os.write(self.fd, tokens.tobytes())
        # os.write(self.fd, memoryview(tokens))
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
        assert self.fd is not None
        assert self.toks < 2**31, "token count too large" # ~2.1B tokens

        # construct the header        
        self.header[2] = self.toks # number of tokens after the 256*4 bytes of header
        self.header[9] = self.vocab_size
        self.header[10] = self.has_masks

        os.lseek(self.fd, 0, os.SEEK_SET)    #self.file_handle.seek(0)
        nWrite = os.write(self.fd, self.header)
        os.lseek(self.fd, self.header.size*4, os.SEEK_SET)    #self.file_handle.seek(self.header.size*4)        
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


def ProcessDataset(file_name: str, ds, key, split_name: str, max_tokens: int = None, *, model, num_processes: int = -1, is_tiny: bool = False, first_is_eval: int = -1, masking: bool = False, seq_len: int = None):
    if num_processes <= 0:
        num_processes = max(1, min(mp.cpu_count() // 2, 16))
    tokenizer = model.tokenizer
    ds_iter = iter(ds)
    file_index = 0
    total_tokens = 0
    max_size = 100_000_000
    if max_tokens and max_tokens < max_size:
        max_size = max_tokens

    Path(f"{file_name}").mkdir(parents=True, exist_ok=True)
    print(f"ProcessDataset_ mp={num_processes} @{file_name} ...")
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

def ProcessPileBackup(dataset, model, out_file):
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    n_samples=128
    max_seq_len=512
    text_column="text"
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = model.tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = line_encoded
        if len(sample) == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = np.concatenate(samples)
    with TokenizedFile(model, out_file, model.tokenizer.vocab_size, masking=False) as f:
        # for tokens in tokenized_examples:
        f.add_document(cat_samples)
    print(f"ProcessPileBackup_ to {out_file}\n")
    # n_split = cat_samples.shape[1] // max_seq_len
    # assert len(cat_samples) % max_seq_len == 0
    # cat_samples = cat_samples.reshape(n_split, max_seq_len)
    # return cat_samples

def ProcessDatabricksDolly(dataset, model, in_file, out_file, SYSTEM_PROMPT = "You are a helpful assistant.", enable_thinking=False, MAX_TOKENS = -1024):
    isEncode = True
    processed_data = []
    tokens = []
    with open(in_file, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # str = model.tokenizer.apply_chat_template(line.strip(), tokenize=False, add_generation_prompt=False)

            instruction = obj.get("instruction", "")
            context = obj.get("context", "")
            response = obj.get("response", "")
            user_content = instruction
            if context:
                user_content += f"\n\n{context}"
            if enable_thinking:
                chatml = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>")
            else:
                #"          <|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
                chatml = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{response}<|im_end|>")

            processed_data.append({"text": chatml})
            if isEncode:
                line = chatml.strip()
                line_encoded = model.tokenizer.encode(line)
                if len(line_encoded) > MAX_TOKENS or len(line_encoded)==0:
                    continue
                tokens.append(line_encoded)
    
    with open(out_file+".jsonl", "w", encoding="utf-8") as fout:
        json.dump(processed_data, fout, ensure_ascii=False, indent=2)
    print(f"ProcessDolly: JSON output written to {out_file}")
    if isEncode:
        cat_tokens = np.concatenate(tokens)
        with TokenizedFile(model, out_file + ".bin", model.tokenizer.vocab_size, masking=False) as f:
            # for tokens in tokenized_examples:
            f.add_document(cat_tokens)
        print(f"ProcessDolly: to {out_file}\n")

# 70%  → with system prompt     30%  → without system prompt (pure UA)
def pre_processing_chat(conversations, add_system_ratio=0.7):
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是koifish，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是koifish，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are koifish, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are koifish, a small but useful language model."
    ]

    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # if '<think>\n\n</think>\n\n' in prompt_content: and random.random() > empty_think_ratio:
    #     prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    prompt_content = re.sub(r'jingyaogong', 'YingshiChen', prompt_content, flags=re.IGNORECASE)
    a = prompt_content
    def ensure_think_tags(data_str: str) -> str:
        """Inserts empty <think> tags into assistant responses if they are missing."""        
        def replace_assistant(match):
            full_block = match.group(0)  # The entire assistant block
            content = match.group(1)     # Everything inside the block
            
            # Check if the content already starts with a <think> tag
            if content.strip().startswith("<think>"):
                return full_block  # Keep it exactly as it is
            
            # If missing, insert the empty think tags right after the assistant start token
            return f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>"

        # Regex pattern to match everything between assistant start and end tokens
        # re.DOTALL ensures that the '.' matches newlines as well
        pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        
        return re.sub(pattern, replace_assistant, data_str, flags=re.DOTALL)


    prompt_content = ensure_think_tags(prompt_content)
    # print(f"\n----before---\n{a}\n----after replace_handler ---\n{prompt_content}")
    
    return prompt_content
'''
    1.  Base Qwen3 (non-reasoning): No built-in think template
'''
def conversations2chatml(tokenizer, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    assert(tools is None)
    chatml = tokenizer.apply_chat_template(
        messages,        tokenize=False,        add_generation_prompt=False,    tools=tools
    )
    return chatml.strip()

#   python src/Python/PreTokenizer.py /home/cys/rnd/lic/Models/Qwen3-0.6B/ --dataset minimind --localdir /home/cys/rnd/lic/Datasets/minimind/sft_t2t_mini.jsonl --outdir /home/cys/rnd/lic/Datasets/chatml/ 
def ProcessMinimind(dataset, model, in_file, out_file, SYSTEM_PROMPT = "You are a helpful assistant.", enable_thinking=False, MAX_TOKENS = -1024):
    from torch.utils.data import Dataset
    
    max_length = MAX_TOKENS if MAX_TOKENS>0 else 1024
    features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
    samples = load_dataset('json', data_files=in_file, split='train', features=features)
    start,end = 0, min(len(samples),10000000)    #len(samples) 
 
    samples = samples.select(range(start,end)) 
    print(f"ProcessMinimind: {len(samples)} samples({start}:{end})")
    pad_id = model.tokenizer.encode(model.tokenizer.pad_token)
    eos_id = model.tokenizer(f'{model.tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    processed_data = []
    path = f"{out_file}_{len(samples)}.bin"
    counter = 0
    with TokenizedFile(model, path, model.tokenizer.vocab_size, masking=False) as f:
        tokens = []
        pbar = tqdm(samples, desc="Processing SFT samples")
        for sample in pbar:
            try:
                conversations = pre_processing_chat(sample['conversations'])
                chatml = conversations2chatml(model.tokenizer, conversations)
                chatml = post_processing_chat(chatml)    
                # processed_data.append({"text": chatml})        
                line = chatml
                line_encoded = model.tokenizer.encode(line)
                if len(line_encoded) > max_length-1 or len(line_encoded)==0:
                    continue
                line_encoded += pad_id
            except Exception as e:
                print(f"Failed@{counter} error={str(e)} sample={sample}")
                continue
            # print(f"{line}\n{line_encoded}")
            tokens.append(line_encoded)
            counter += 1
            if counter % 1000 == 0:
                try:
                    # cat_tokens = np.concatenate(tokens)
                    cat_tokens = np.fromiter(
                        (tok for seq in tokens for tok in seq),
                        dtype=np.int32,
                        count=sum(map(len, tokens))
                    ).copy()
                    f.add_document(cat_tokens)
                    pbar.set_description(f"{counter:8d} samples: toks={f.toks/1.0e6:.5g}M")
                    tokens = []
                except Exception as e:
                    print("❌ Batch concat / write failed:", e)
                    tokens.clear()
    print(f"ProcessMinimind: to {path}\n")  
    
def create_assistant_labels(tokenizer, input_ids, im_start, im_end):
    """
    仅对 assistant 块中  之后的真实回答计算损失
    system/user 块、所有分隔符、 标签全部设为 -100
    """
    # 初始化 labels 为 -100（全部不计算损失）
    labels = torch.full_like(input_ids, -100)
    
    # 获取特殊标记的 token id
    start_id = tokenizer.convert_tokens_to_ids(im_start)
    end_id = tokenizer.convert_tokens_to_ids(im_end)
    
    # 获取 assistant 关键词 token id
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    think_start_id = tokenizer.convert_tokens_to_ids("")
    think_end_id = tokenizer.convert_tokens_to_ids("")

    # 遍历每一条样本
    for b_idx in range(input_ids.shape[0]):
        seq = input_ids[b_idx]
        seq_len = len(seq)
        
        # 寻找 <|im_start|>assistant 位置
        in_assistant = False
        found_think_end = False
        
        for i in range(seq_len - 1):
            # 匹配 <|im_start|>assistant
            if seq[i] == start_id and seq[i+1] == assistant_id:
                in_assistant = True  # 进入 assistant 块
            
            # 匹配 ：之后才是真正需要学习的回答内容
            if in_assistant and seq[i] == think_end_id:
                found_think_end = True
            
            # 匹配 <|im_end|>：退出 assistant 块
            if in_assistant and seq[i] == end_id:
                in_assistant = False
                found_think_end = False
            
            # 只有 assistant 块中  之后的 token 才需要计算损失
            if in_assistant and found_think_end:
                # 过滤掉末尾空行，只保留有效回答
                if seq[i] not in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                    labels[b_idx, i] = seq[i]

    return labels
# '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nFine<|im_end|>\n'
def build_sft_loss_labels(tokenizer, input_ids_tensor, start_tok, end_tok):
    """
    Rule:
    All system/user blocks, <|im_start|>, <|im_end|>, ... are context (-100 no loss).
    Only plain assistant answer AFTER  uses original token id for loss calculation.
    HF Trainer ignores -100 in cross-entropy loss by default.
    """
    # labels = tokens["input_ids"].clone()
    # labels[labels == tokenizer.pad_token_id] = -100     #   nn.CrossEntropyLoss ignores targets == -100
    batch_size, seq_len = input_ids_tensor.shape
    # Initialize all positions to -100 (no loss)
    labels = torch.full_like(input_ids_tensor, fill_value=-100)

    # Fetch special token IDs
    id_im_start = tokenizer.convert_tokens_to_ids(start_tok)
    id_im_end = tokenizer.convert_tokens_to_ids(end_tok)
    id_assistant = tokenizer.convert_tokens_to_ids("assistant")
    id_think_open = tokenizer.convert_tokens_to_ids("<think>")
    id_think_close = tokenizer.convert_tokens_to_ids("</think>")
    true_answer = ""
    for batch_idx in range(batch_size):
        seq = input_ids_tensor[batch_idx]
        in_assistant_block = False
        think_closed_flag = False
        pos = -1
        while pos+1 < seq_len:
            pos += 1
            current = seq[pos].item()            
            # Detect start of assistant turn: <|im_start|> followed by "assistant"
            if pos+1 < seq_len and current==id_im_start and seq[pos+1]==id_assistant:
                in_assistant_block = True
            if not in_assistant_block: continue

            if current == id_think_close:
                think_closed_flag = True
                a1,a2 = tokenizer.decode(seq[pos+1]),tokenizer.decode(seq[pos+2])
                if a1 == "\n\n":    pos = pos+1
                if a1 == "\n" and a2 == "\n":    pos = pos+2                
            elif current == id_im_end:
                in_assistant_block = False
                think_closed_flag = False
            else:                # Only enable loss for tokens after  inside assistant block
                if think_closed_flag:                    # Skip padding tokens
                    a = tokenizer.decode(current)
                    if current != tokenizer.pad_token_id :                        
                        labels[batch_idx, pos] = current
                        true_answer += a

    return labels,true_answer

'''
1 Chat-style JSONL or OpenAI chat format: {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "How do I bake a chocolate cake?"},
      {"role": "assistant", "content": "First, preheat the oven to 350°F. Then mix flour, sugar, cocoa powder, eggs, butter, and baking powder..."}
    ]
  },
2 padding="max_length" padding=True;        Padding tokens contribute to the loss unless you explicitly exclude them!!!
'''
def collate_Chat_JSONL(batch, tokenizer, padding=True):
    if isinstance(batch, list):
        texts = [
            tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            for ex in batch
        ]
    else:
        texts = tokenizer.apply_chat_template(batch["messages"], tokenize=False, add_generation_prompt=False)
    # print(texts)
    # attention_mask[i] = 1 if input_ids[i] != pad_token_id else 0
    tokens = tokenizer(texts,        padding=padding,       truncation=True,       max_length=128,     return_tensors="pt",)
    # print(tokens)
    if(list_depth(tokens["input_ids"])==2):  # In some case, it's batch even for 1 samp
        labels = tokens["input_ids"][0].copy() # Ignore padding tokens in loss      
    else:  
        labels = tokens["input_ids"].detach().clone()   #.copy()
    IM_START, IM_END = "<|im_start|>", "<|im_end|>"
    labels,true_answer = build_sft_loss_labels(tokenizer, tokens["input_ids"], IM_START, IM_END)
    print(f"answer={true_answer}")
    # print(labels)
    tokens["labels"] = labels
    # print(enc)
    return tokens

'''
    1. padding token=151643 => labels=-100
    2.  attention_mask[i] = 1 if input_ids[i] != pad_token_id else 0
'''
def ChatJSONL2SFT(dataset, tokenizer, out_file):        
    # dataset = dataset.shuffle(seed=42)
    pad_id = tokenizer.pad_token_id
    print(f"tokenizer pad={pad_id} eos={tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   
    
    print(dataset)
    dataset = dataset.map(partial(collate_Chat_JSONL, tokenizer=tokenizer), batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")    #   Change all elements of dataset to PyTorch tensors, not Python lists or NumPy arrays
    # all_sample = [] #   only for debug
    # for sample in dataset:
    #     token_sample = collate_Chat_JSONL(sample, tokenizer)
    #     all_sample.append(token_sample)
    # dataset = datasets.Dataset.from_list(all_sample)
    first_row = dataset[0]
    print(f"dataset first_row = {first_row}")
    return dataset
    # for key in ['input_ids', 'attention_mask', 'labels']:
    #     value = first_row[key]
    #     print(f"\n{key} (type: {type(value)}):")
    #     print(f"  Shape/Length: {len(value)}")
    #     print(f"  First 10 values: {value[:10]}")
    #     print(f"  Last 10 values: {value[-10:] if len(value) > 10 else value}")      
    #     if key in ['input_ids', 'labels'] and hasattr(dataset, '_tokenizer'):
    #         decoded = dataset._tokenizer.decode(value)
    #         print(f"  Decoded text: {decoded}")


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
    elif dataset == "pile-val-backup":  # mit-han-lab/pile-val-backup only 10k txt slice with ~30k tokens! (original pie>million!)
        d = datasets.load_dataset("/home/cys/rnd/lic/Datasets/pile-val-backup", split="validation")        
        ProcessPileBackup(d, model, out_dir+"/eval.bin")
        return
    elif dataset == "databricks-dolly":  
        # d = datasets.load_dataset("/home/cys/rnd/lic/Datasets/pile-val-backup", split="validation")        
        ProcessDatabricksDolly(None, model, localdir, out_dir+"/bricks_chatml", MAX_TOKENS = 1024)
        return
    elif dataset == "minimind":        
        ProcessMinimind(None, model, localdir, out_dir+"/minimind", MAX_TOKENS = 1024)
        return
    elif dataset == "json_file":  
        test_split = 0
        key = "messages"
        is_tiny = True
        # dataset = datasets.load_dataset("json", data_files=localdir, split="train")   # From hf's strange design, we woulg get a single datasets with all json items
        dataset = datasets.load_dataset("json", data_files=localdir)   
        d = dataset
        print(d["train"])
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


