# Koifish

Sparse & quantized LLM training/inference in C++/cu.

Koifish needs much less training resource than other frameworks. It needs only one day to train ~2B model on single 4090 as shown in the following table.

| Model | Parameter  | Loss(Baseline) |GPU Memory|Total Time (Training+Evaluating)|Throughput|Log|
|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| [GPT2-124M](./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json)         | 124M           | 3.293(3.425)  | ~6.8G   |~8 hours|~140k/s|[log](cases/gpt2/124M_shard50_F6_lr0.001)|
| [GPT2-774M](./cases/gpt2/774M_Shard50_F6_B80/F6_B80.json)         | 774M           | 3.146(3.00)   | ~15G |~18 hours|~70k/s|[log](cases/gpt2/774M_Shard50_F6_B80)|
| [GPT2-1558M](./cases/gpt2/1558M_F8_B80/F8_B80.json)        | 1558M           | 3.04(2.83)   | ~23G   |~30 hours|~50k/s|[log](cases/gpt2/1558M_F8_B80)|

Note
1. The baseline results are from [GPT2](https://github.com/openai/gpt-2). Our training only used 5B tokens of FineWeb.   
2. The evaluating time depends on the frequency of testing and the sampling ratio(We use only ~10% randomly sampled tokens to reduce total time). 

## Features
- Hybrid 16/8/4/3/1 bit training
- Rematerialisation and fusion of operators
- [Evolutionary optimization of experts](https://arxiv.org/abs/2509.24436)
- Inference of QWen3-32B on single 4090
- Automatic detection of training instability
- Json config file
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
make clean && make -j$(nproc)
```

## Tutorial

1.    [Training of GPT2_1558M_ on single 4090](cases/tutorial/tutorial_gpt2_1558M.md)
1.    [Training of GPT2(774M/124M) on single 3090](cases/tutorial/tutorial_gpt2.md)

## [Techniques/Tricks](cases/tricks/Tricks.md)

## Working plan
- Support 1-bit version of QWen/DeepSeek
- Sparsing(Predict sparse neurons by GBDT method)

## Citation
Please use the following bibtex entry:
```bibtex
@article{chen2025eoe,
  title={EOE: Evolutionary Optimization of Experts for Training Language Models},
  author={Chen, Yingshi},
  journal={arXiv preprint arXiv:2509.24436},
  year={2025}
}
```

## Contributing

- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the highly instructive work of [qwen600](https://github.com/yassa9/qwen600), [calm](https://github.com/zeux/calm), [llm.c](https://github.com/karpathy/llm.c)
 & [ggml](https://github.com/ggerganov/ggml).

## More
### [History](cases/tutorial/history.md)
QQ group: 167653113
