import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path  
import os
import json
import argparse 
import os
import shutil
import json
import time
import subprocess

sz = "774M"
loss_baseline = {
    "124M": 3.424958,
    "350M": 3.083089,
    "774M": 3.000580,
    "1558M": 2.831273,
}[sz]
hella2_baseline = { # for GPT-2
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}[sz]
hella3_baseline = { # for GPT-3
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}[sz]

# optional function that smooths out the loss some
def smooth_moving_average(signal, window_size):
    if signal.ndim != 1:
        raise ValueError("smooth_moving_average only accepts 1D arrays.")
    if signal.size < window_size:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_size < 3:
        return signal

    s = np.pad(signal, (window_size//2, window_size-1-window_size//2), mode='edge')
    w = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(s, w, mode='valid')
    return smoothed_signal

def plt_1(df,df2,baseline=None,title="Loss",ylabel="loss" ):
    x_="iter";      y_="loss" 
    xs, ys = df[x_],df[y_]
    # smooth out ys using a rolling window
    # ys = smooth_moving_average(ys, 21) # optional
    plt.plot(xs, ys, label=f'Koifish ({sz})')
    if df2 is not None:
        xs2, ys2 = df2[x_],df2[y_]
        plt.plot(xs2, ys2, label=f'Koifish ({sz}) eval')
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
    y0,y1 = min(ys),max(ys)
    print(f"{title} = [{y0},{y1}]")
    plt.xlabel(x_);    plt.ylabel(ylabel)
    # plt.yscale('log')
    # plt.ylim(top=4.0)
    plt.grid(True)
    plt.legend()
    plt.title(title)

def plt_losscurve(dfTrain,dfEval=None,df_hellaswag=None):
    assert dfTrain is not None
    plt.figure(figsize=(16, 6))
    plt.subplot(121)        # Panel 1: losses: both train and val
    plt_1(dfTrain,dfEval,loss_baseline)
    
    if df_hellaswag is not None:    # Panel 2: HellaSwag eval
        plt.subplot(122)      
        plt_1(df_hellaswag,None,hella2_baseline,title="Accuracy on HellaSwag",ylabel="accuracy")  

    plt.show(block=True) 
    path = args.train+".png"
    plt.savefig(path)
    
    print(f"Save losscurve@{path} train={dfTrain.shape}  eval={dfEval.shape}")

def get_gpu_stats():
    try:
        # Get full GPU info
        result = subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)        
        # Get specific memory info
        memory_query = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv'],
            stdout=subprocess.PIPE,
            text=True
        )
        
        print("Full GPU Status:")
        print(result.stdout)        
        # print("\nMemory Details:")
        print(memory_query.stdout)
        
    except FileNotFoundError:
        print("nvidia-smi not found. Are you using NVIDIA GPU?")
        
if __name__ == '__main__':
    sExe = "./bin/koifish "
    fJson = "./scripts/gpt2.json"
    with open(fJson, 'r') as f:
        jConfig_0 = json.load(f)  
    
    batch_size = [200,160,40,80,120]
    cross_over=[0.6,0.4,0.2,0.1]
    no = 0
    for b in batch_size:
        for cross in cross_over:
            title = "crossover"+ str(cross) + "_B" + str(b)
            sOutput = title+".info" #log/cross_over=0.6.info"
            path = "./SWEEP/"+title+"/"
            os.makedirs(path, exist_ok=True)
            jConfig = jConfig_0
            jConfig["train"]["batch"] = b
            jConfig["model"]["fuyou"]["crossover"] = cross
            with open(path+title+".json", 'w') as f:
                json.dump(jConfig, f, indent=4)  # Write JSON to file
            
            # cmd = sExe+ path+title+".json 2>&1 | tee "+path+sOutput 
            if b==200:
                continue
            get_gpu_stats()
            cmd = sExe+ path+title+".json > "+path+sOutput + " 2>&1"
            print(f"{no}\t{cmd} ...")
            start = time.time()                 
            exit_code = os.system(cmd)
            elapsed = time.time() - start
            print(f"{no}...OK! code={exit_code} Time={elapsed:.4f} seconds")
            get_gpu_stats()
            shutil.copy('./Train@[edu_fineweb1B]_info_.csv', path)
            shutil.copy('./Eval@[edu_fineweb1B]_info_.csv', path)
            
    exit()    
   
