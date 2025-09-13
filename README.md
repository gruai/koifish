# Koifish

Sparse & quantized LLM training in C++/cu.

Training of ~2B language models with only one GPU.

## Features
- Hybrid 16/8/1 bit training
- Rematerialisation and fusion of operators
- Mixture of models 
- Automatic detection of training instability
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
# export CPATH=/usr/local/cuda/include:$CPATH        # maybe need this to export CPATH
git clone https://github.com/gruai/koifish
cd koifish
mkdir build && cd build && cmake ..
make clean && make VERBOSE=TRUE
```

## Tutorial

1.    [Training of GPT2_1558M_ on single 4090](cases/tutorial_gpt2_1558M.md)
1.    [Training of GPT2(774M/124M) on single 3090](cases/tutorial_gpt2.md)

## [Techniques/Tricks](cases/tricks/Tricks.md)

## History
* 08/31/2025: Training 1558M sparse GPT2 model on single 4090 with throughput > 40K tokens/second
* 08/17/2025: A new framework of multiscale deep learning
* 08/01/2025: Training 1558M sparse GPT2 model on single 4090 with throughput > 20K tokens/second
* 07/24/2025: Support tile quantization 
* 06/29/2025: Rope with pre/post normalization 
* 06/14/2025: Support bit representation(binary[-1,1], ternary[-1,0,1]) 

## Working plan
- Support 1-bit version of QWen/DeepSeek
- Sparsing(Predict sparse neurons by GBDT method)

## Contributing

- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the highly instructive work of [calm](https://github.com/zeux/calm), [llm.c](https://github.com/karpathy/llm.c) & [ggml](https://github.com/ggerganov/ggml).

## More
QQ group: 167653113
