make train_gpt2 VERBOSE=TRUE

cc -g -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -march=native -fopenmp -DOMP   train_gpt2.c -lm -lgomp -o train_gpt2

pip install datasets --break-system-packages -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

make train_gpt2cu USE_CUDNN=1 VERBOSE=TRUE

./train_gpt2cu \
    -i "/home/cys/Datasets/edu_fineweb/*fineweb_train_*.bin" \
    -j "/home/cys/Datasets/edu_fineweb/*fineweb_val_*.bin" \
    -o log124M \
    -e "d12" \
    -b 32 -t 1024 -d 524288 -r 1 -z 1 -c 0.1 -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -v 250 -s 20000 \
    -h 1