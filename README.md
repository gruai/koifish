# Koifish

**Koifish** is a c++ framework focused on efficient training/fine-tuning language model on edge devices & PC. 
1. Efficient  on-device training of billion parameter language model.
2. Efficient  fine-tuning 10-billion parameter LLMs on edge device.

## Features

- Mixture of models
- Support LLAMA/GPT/MAMBA ...
- CPU, GPU and Hybrid training
- Json config file
- Pure C++ project

## Build

```bash
git clone https://github.com/gruai/koifish
cd koifish

mkdir build && cd build && cmake ..
make clean && make VERBOSE=TRUE
# cmake --build . --config Release -j 8 VERBOSE=TRUE
```

## Datasets

## Training

```bash

```

## Fine-tuning



## Working plan
- Sign adam optimizer

## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `koifish` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues, PRs and projects is very appreciated!
  
## Acknowledgements

* Thanks very much for the outstanding work of [llama.cpp](https://github.com/ggerganov/llama.cpp) & [ggml](https://github.com/ggerganov/ggml).