
## 1. Enviroment
1) Linux x86 64bit Ubuntu 22.04 with CUDA 12.5+
2) Install CUDNN & cudnn-frontend
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
```

## 2. Download project & build
```bash
# export CPATH=/usr/local/cuda-12.5/include:$CPATH        # maybe need this to export CPATH
git clone https://github.com/gruai/koifish
cd koifish
mkdir build && cd build && cmake ..
make clean && make VERBOSE=TRUE
```

## 3. Datasets & Tokenizer   
The datasets for this tutorial comes from [karpathy/fineweb-edu-100B-gpt2-token-shards](https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards), which includes about 1000 .bin files. Each file contain 100M tokens.    Thank Andrej K very much for providing this dataset!

As the following json configuration shows, Koifish would load token files from user specified folder. In this tutorial, "most"=20, so Koifish would only load at most 20 train*.bin files(with 2B tokens). Users could try more tokens to get higher accuracy with more time.

## 4. Train 
```shell
    ./bin/koifish ./cases/gpt2/gpt2_1558M.json
```