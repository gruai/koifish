# Koifish

Sparse & quantized LLM training in C++/cu.

Koifish needs much less training resource than other frameworks. It needs only one day to train ~2B model on single 4090 as shown in the following table.

| Model | Parameter  | Loss(Baseline) |GPU Memory|Total Time (Training+Evaluating)|Throughput|Log|
|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| GPT2-124M         | 124M           | 3.287(3.425)  | ~6.8G   |~8 hours|~140k/s|[log](cases/gpt2/124M_shard50_F6_lr0.001)|
| GPT2-774M         | 774M           | 3.146(3.00)   | ~15G |~18 hours|~70k/s|[log](cases/gpt2/774M_Shard50_F6_B80)|
| GPT2-1558M         | 1558M           | 3.04(2.83)   | ~23G   |~30 hours|~50k/s|[log](cases/gpt2/1558M_F8_B80)|

Note
1. The baseline results are from [GPT2](https://github.com/openai/gpt-2). Our training only used 5B tokens of FineWeb.   
2. The evaluating time depends on the frequency of testing and the sampling ratio(We use only ~10% randomely sampled tokens to reduce total time). 

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

1.    [Training of GPT2_1558M_ on single 4090](cases/tutorial/tutorial_gpt2_1558M.md)
1.    [Training of GPT2(774M/124M) on single 3090](cases/tutorial/tutorial_gpt2.md)

## [Techniques/Tricks](cases/tricks/Tricks.md)

## History
* 08/17/2025: A new framework of multiscale deep learning
* 08/01/2025: Training GPT2_1558M on single 4090 with throughput > 20K tokens/second
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
