import lzma
import os
import tarfile
from pathlib import Path
import glob
import sys
import json
import argparse  
import re
import datasets
# import ftfy
import json
# from langdetect import detect
import numpy as np
import time
import os
import sys 
# from tokenizer import Tokenizer
# https://blog.csdn.net/xiaoaoniubi/article/details/138313464
# https://huggingface.co/datasets/Skylion007/openwebtext/blob/main/openwebtext.py
# Size of downloaded dataset files: 13.51 GB    Size of the generated dataset: 41.70 GB     Total amount of disk used: 55.21 GB
_CITATION = """\
@misc{Gokaslan2019OpenWeb,
  title={OpenWebText Corpus},
  author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex},
  howpublished{\\url{http://Skylion007.github.io/OpenWebTextCorpus}},
  year={2019}
}
"""
 
MIN_DOCUMENT_LENGHT = 128

def decompress_xz_files(src_dir, dest_dir, start_index=1, end_index=1000):
    """Decompress .xz files containing multiple documents and copy each document to the destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
 
    # for i in range(start_index, end_index + 1):
    #     src_file = f"{src_dir}/urlsf_subset20-{i}_data.xz"
    # directory = os.fsencode(src_dir)   
    i = 0 
    for src_file in os.listdir(src_dir):
        src_file = src_dir+src_file
        filename = os.fsdecode(src_file)
        if filename.endswith(".xz"):            # Handle regular .xz files
            try:
                dest_file_path = os.path.join(dest_dir, f"extracted__{i}.txt")
                with lzma.open(src_file, 'rt') as file:
                    content = file.read()
                with open(dest_file_path, 'w') as out_file:
                    out_file.write(content)
                print(f"Decompressed and copied content from {src_file} to {dest_file_path}") 
            except:
                print(f"Exception @{src_file} to {dest_file_path}!!!")
                continue
            i = i+1
            
        #   for f in *.tar; do tar xf "$f"; done
        # elif tarfile.is_tarfile(src_file):            # Open the tarball file
        #     with tarfile.open(src_file, mode='r:xz') as tar:
        #         tar.extractall(path=dest_dir)
        #         print(f"Extracted all contents of {src_file} to {dest_dir}")            
        else:
            print(f"File {src_file} does not exist")
 

def merge():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".",
                        help="path where all the json files are located")
 
    parser.add_argument("--output_file", type=str, default="merged_output.json",
                        help="filename where the merged json should go")
 
    args = parser.parse_args()
    data_path = args.data_path
    out_file = args.output_file
    text_files = glob.glob(data_path + '/*.txt')
    counter = 0
 
    with open(out_file, 'w', encoding='UTF-8') as outfile:
        for fname in text_files:
            counter += 1
 
            if counter % 1024 == 0:
                print("Merging at ", counter, flush=True)
 
            with open(fname, 'r', encoding='UTF-8') as infile:
                for row in infile:
                    tmp = {}
                    tmp['text'] = row
                    outfile.write(json.dumps(tmp))
                    outfile.write('\n')
 
    print("Merged file", out_file, flush=True)

if __name__ == '__main__':
    source_directory = '/home/cys/Documents/temp/openwebtext/' #'/media/cys/E0/openweb/openwebtext/'
    destination_directory = '/home/cys/Documents/temp/xtext/'
    decompress_xz_files(source_directory, destination_directory)

