# Koifish

**Koifish** is a c++ framework focused on efficient training/fine-tuning language model on edge devices & PC. 
1. Efficient on-device training of billion parameter language model.
2. Efficient fine-tuning 10-billion parameter LLMs on edge device.

## Features

- Rematerialisation and fusion of operators
- Mixture of models
- Support DEEPSEEK/LLAMA/GPT ...
- CPU, GPU and Hybrid training
- Json config file
- Self-contained C++ project with minimal dependencies

## Download & Build

```bash
# sudo apt-get install libicu-dev
# export CPATH=~/cudnn-frontend/include/:/usr/local/cuda-12.1/include:$CPATH        # maybe need this to export CPATH
git clone https://github.com/gruai/koifish
cd koifish
mkdir build && cd build && cmake ..
make clean && make VERBOSE=TRUE
```

## Tutorial

1.    [Training of GPT2(774M/124M) on single 3090](cases/tutorial_gpt2.md)

## Training tricks
- Subsampling
- [Weight Tying](cases/tricks/WeightTying.md)


## Working plan
- Hybrid 16/8 bit Optimizer
- Support MAMBA
- Sparsing

## Contributing

- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the highly instructive work of [calm](https://github.com/zeux/calm), [llm.c](https://github.com/karpathy/llm.c) & [ggml](https://github.com/ggerganov/ggml).
