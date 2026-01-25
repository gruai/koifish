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
import fnmatch
import seaborn as sns
import glob
import atexit
import sys
import tkinter as tk
from tkinter import messagebox

PID_FILE = "/tmp/__SweepParams__.pid"

def PID_cleanup():
    if os.path.exists(PID_FILE):
        os.unlink(PID_FILE)
    
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
        
        # print(f"Full GPU Status:{result.stdout}")
        # print("\nMemory Details:")
        print(memory_query.stdout)
        
    except FileNotFoundError:
        print("nvidia-smi not found. Are you using NVIDIA GPU?")

def MessageBox(msg):
    try:
        root = tk.Tk()
        root.withdraw();    root.option_add('*Dialog.msg.font', 'Arial 24')
        response = messagebox.askyesno(f"Warning", )
        if response is False:
            sys.exit()  
    except tk.TclError as e:
        print(f"Tkinter failed: {e} (DISPLAY issue)")

def koifish_one(title, sExe, jsFile0, path="./tests/", most_iter=-1,train_csv=None):
    with open(jsFile0, 'r') as f:
        jConfig = json.load(f)  
    if most_iter>0:
        if  "debug" not in jConfig:
            jConfig["debug"] = {}
        jConfig["debug"]["most_iter"] = most_iter
        # jConfig["train"]["dump-every"] = 1
    if train_csv is not None:
        if os.path.exists(train_csv):
            os.remove(train_csv)
    
    jsFile = path+title+".json"
    sOutput = title+".info"
    with open(jsFile, 'w') as f:
        json.dump(jConfig, f, indent=4)  # Write JSON to file       
    
    # if b==200:                continue
    get_gpu_stats()
    cmd = sExe + jsFile + "> "+path+sOutput + " 2>&1"        # cmd = sExe+ path+title+".json 2>&1 | tee "+path+sOutput 
    print(f"{title}\t{cmd} ...")
    start = time.time()    
    exit_code = os.system(cmd)
    elapsed = time.time() - start
    
    # get_gpu_stats()
    dfTrain = None
    szTrain, szEval = 0, 0
    if train_csv is None:
        train_csv = './Train@[edu_fineweb1B]_info_.csv' #default old path
    if os.path.exists(train_csv):
        shutil.copy(train_csv, path+"./Train.csv")
        szTrain = os.path.getsize(path+"./Train.csv")
        dfTrain = pd.read_csv(path+"./Train.csv", sep=' ',index_col=False)
    if os.path.exists('./Eval@[edu_fineweb1B]_info_.csv'):
        shutil.copy('./Eval@[edu_fineweb1B]_info_.csv', path+"./Eval.csv")
        szEval = os.path.getsize(path+"./Eval.csv")
    print(f"{title}...OK! code={exit_code}. nByte of fTrain={szTrain}; nByte of fEval={szEval} Time={elapsed:.4f} seconds")
    assert dfTrain is not None
    assert "loss" in dfTrain.columns
    
    return dfTrain

def pangpi_one(title, sExe, sArgs, path="./tests/", most_iter=-1):    
    sOutput = title+".info"   
    
    get_gpu_stats()
    cmd = sExe + sArgs + "> "+path+sOutput + " 2>&1"        # cmd = sExe+ path+title+".json 2>&1 | tee "+path+sOutput 
    print(f"{title}\t{cmd} ...")
    start = time.time()    
    exit_code = os.system(cmd)
    elapsed = time.time() - start
    if exit_code!=0:
        sys.exit(exit_code)
    
    # get_gpu_stats()
    dfLoss = None
    szLoss = 0
    fSrc,ftarget = './Eval_loss.csv',path+"./loss.csv"
    if os.path.exists(fSrc):
        shutil.copy(fSrc, ftarget)
        szLoss = os.path.getsize(ftarget)
        dfLoss = pd.read_csv(ftarget, sep=' ',index_col=False)
    print(f"{title}...OK! code={exit_code}. nByte of fLoss={szLoss} Time={elapsed:.4f} seconds")
    assert dfLoss is not None
    
    return dfLoss, exit_code

def bubble_one(title,  sArgs, sExe ="./bin/bubble ", path="./tests/", most_iter=-1):    
    sOutput = title+".info"   
    cmd = sExe + sArgs + "> "+path+sOutput + " 2>&1"        # cmd = sExe+ path+title+".json 2>&1 | tee "+path+sOutput 
    print(f"[{title}]\t{cmd} ...")    

    if os.path.exists('chat.csv'):
        os.remove('chat.csv')
    exit_code = os.system(cmd)
    if not os.path.exists('chat.csv'):
        return ""
    try:
        with open('chat.csv', 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()  # 读取全部内容（返回字符串）
        print(content)    
        lines = content.splitlines()
        line_numbers = list(range(1, len(lines) + 1))
        # Get the last line
        last_line = lines[-1] if lines else None
        return last_line
    except UnicodeDecodeError:
        return(f"Failed with encoding: utf-8")
    
# python cases/SweepHyParams.py --dir ./SWEEP/124M --json ./scripts/gpt2.json   
#  python cases/SweepHyParams.py --dir ./SWEEP/Shard50 --json ./scripts/gpt2.json
#  python cases/SweepHyParams.py --dir ./SWEEP/Shard50 --json ./cases/gpt2/gpt2_774M.json
#  python cases/SweepHyParams.py --dir ./SWEEP/1558M --json ./cases/gpt2/gpt2_1558M.json
# python cases/SweepHyParams.py --dir ./SWEEP/MUON --json ./cases/gpt2_1558M.json  
if __name__ == '__main__':
    if os.path.exists(PID_FILE):
        print("./cases/SweepHyParams.py is already running!")
        sys.exit()
    else:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        atexit.register(PID_cleanup)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./SWEEP/",help="log path of sweep test")
    parser.add_argument("--symbol", action='store_true')    
    parser.add_argument("--json", type=str)  
    args = parser.parse_args()
    MessageBox(msg=f"SweepHyParams would rm all csv results@{args.dir}!\nDo you want to continue?")

    sExe = "./bin/koifish "
    # fJson = "./cases/gpt2/gpt2_774M.json"        #cases/gpt2/gpt2_774M.json
    with open(args.json, 'r') as f:
        jConfig_0 = json.load(f)  
    
    # batch_size = [200,160,40,80,120]
    #   cross_over=[0.6,0.4,0.2,0.1]
    branches = [6]     #48
    lrs =[0.0006,0.001]    #[0.002,0.001,0.0006,0.0002]
    lr = 0.0006
    no = 0;     batchs = [80,120,160,200,240]   
    for branch in branches:        
        for b in batchs:
            title = "F"+ str(branch)  + "_B" + str(b)      #+ "_crossover"+ str(cross)
            sOutput = title+".info" #log/cross_over=0.6.info"
            path = args.dir+"/"+title+"/"
            os.makedirs(path, exist_ok=True)
            for file in glob.glob(os.path.join(path, "*.csv")):  # Or "*.txt" for specific extensions
                if os.path.isfile(file):
                    os.remove(file)
                    
            jConfig = jConfig_0
            jConfig["train"]["batch"] = b
            jConfig["datasets"]["train"]["most"] = 100
            jConfig["train"]["learning-rate"] = lr
            # jConfig["model"]["parameter"]["transformer"]["Ctx"] = 1536        # needs much more memory
            # jConfig["model"]["fuyou"]["crossover"] = cross
            jConfig["model"]["fuyou"]["branch"] = branch
            with open(path+title+".json", 'w') as f:
                json.dump(jConfig, f, indent=4)  # Write JSON to file       
            
            # if b==200:                continue
            get_gpu_stats()
            cmd = sExe+ path+title+".json > "+path+sOutput + " 2>&1"        # cmd = sExe+ path+title+".json 2>&1 | tee "+path+sOutput 
            print(f"{no}\t{cmd} ...")
            start = time.time()    
            if args.symbol:             
                exit_code = 0  
            else:
                exit_code = os.system(cmd)
            elapsed = time.time() - start
            
            # get_gpu_stats()
            szTrain, szEval = 0, 0
            if not args.symbol and os.path.exists('./Train@[edu_fineweb1B]_info_.csv'):
                shutil.copy('./Train@[edu_fineweb1B]_info_.csv', path+"./Train.csv")
                szTrain = os.path.getsize(path+"./Train.csv")
            if os.path.exists('./Eval@[edu_fineweb1B]_info_.csv'):
                shutil.copy('./Eval@[edu_fineweb1B]_info_.csv', path+"./Eval.csv")
                szEval = os.path.getsize(path+"./Eval.csv")
            print(f"{no}...OK! code={exit_code} fTrain={szTrain} fEval={szEval} Time={elapsed:.4f} seconds")
            
    sys.exit()    
   
