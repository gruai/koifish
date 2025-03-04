# Koifish

**Koifish** is a c++ framework focused on efficient training/fine-tuning language model on edge devices & PC. 
1. Efficient  on-device training of billion parameter language model.
2. Efficient  fine-tuning 10-billion parameter LLMs on edge device.

## Features

- Rematerialisation and fusion of operators
- Mixture of models
- Support DEEPSEEK/LLAMA/GPT ...
- CPU, GPU and Hybrid training
- Json config file
- Pure C++ project

## Download & Build

```bash
git clone https://github.com/gruai/koifish
cd koifish
# build ggml lib first
cd llama.cpp
mkdir build && cd build && cmake .. 
make clean && make VERBOSE=TRUE
cd ../../

mkdir build && cd build && cmake ..
# export CPATH=~/cudnn-frontend/include/:/usr/local/cuda-12.1/include:$CPATH        # maybe need this to export CPATH
make clean && make VERBOSE=TRUE
```

## Tutorial

1.    [Training of GPT2(774M/124M) on single 3090](cases/tutorial_gpt2.md)


## Working plan
- Hybrid 1-bit Optimizer
- Support MAMBA
- Sparse mapping of token-embedding to logits

## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `koifish` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the outstanding work of [llm.c](https://github.com/karpathy/llm.c) & [ggml](https://github.com/ggerganov/ggml).