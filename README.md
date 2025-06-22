# Koifish

**Koifish** is a c++ framework focused on efficient training/fine-tuning language model on edge devices & PC. 
1. Efficient on-device training of ~1B language model.
2. Efficient fine-tuning ~10B LLMs on edge device.

## Features

- Rematerialisation and fusion of operators
- Mixture of models
- CPU, GPU and Hybrid training
- Json config file
- Pure C++ tokenizer
- Self-contained C++/cu project with minimal dependencies

## Minimum dependencies
- cudnn(may removed in future version)
- 16GB+ VRAM CUDA Device
- CUDA Toolkit (12.5+)

## Download & Build

```bash
# sudo apt-get -y install libcudnn9-dev-cuda-12  # maybe need this to install CUDNN
# sudo apt-get install libicu-dev
# export CPATH=/usr/local/cuda-12.5/include:$CPATH        # maybe need this to export CPATH
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


## Changelog

* 06/14/2025: Support bit representation(binary[-1,1], ternary[-1,0,1]) 

## Working plan
- Hybrid 16/8/1 bit training
- Support QWen/DeepSeek
- Sparsing

## Contributing

- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the highly instructive work of [calm](https://github.com/zeux/calm), [llm.c](https://github.com/karpathy/llm.c) & [ggml](https://github.com/ggerganov/ggml).
