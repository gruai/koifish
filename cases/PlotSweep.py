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

class SWEEP_result:
    # Class attribute (shared by all instances)
    species = ""

    def __init__(self, title, path,jConfig,dfTrain,dfEval,fLog=None):
        # Instance attributes
        self.title = title
        self.path = path
        self.jConfig = jConfig
        self.dfTrain = dfTrain
        self.dfEval = dfEval
        self.fLog = fLog

    # Instance method
    def bark(self):
        return f"{self.name} says woof!"

def plt_result(result,baseline=None,ylabel="loss" ):
    dfSrc = result.dfEval
    x_="iter";      y_="loss"     
    xs, ys = dfSrc.iloc[:,0],dfSrc.iloc[:,1]      
    # smooth out ys using a rolling window
    # ys = smooth_moving_average(ys, 21) # optional
    # print(dfSrc.head())
    # print(f"x={xs} y={ys}")
    plt.plot(xs, ys, label=f'{result.title}')
    if result.dfEval is not None:
        # xs2, ys2 = result.dfEval[x_],result.dfEval[y_]
        xs2, ys2 = result.dfEval.iloc[:,0],result.dfEval.iloc[:,1]
        plt.plot(xs2, ys2, label=f'{result.title} eval')
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
    y0,y1 = min(ys),max(ys)
    print(f"{result.title} = [{y0},{y1}]")
    plt.xlabel(x_);    plt.ylabel(ylabel)
    # plt.yscale('log')
    # plt.ylim(top=4.0)
    plt.grid(True)
    plt.legend()
    # plt.title(result.title)

def plt_df(df,title,path ):
    df = df.sort_index(axis = 1)
    print(df)
    nResult = len(df.columns)
    plt.figure(figsize=(20, 12))  # Width=10 inches, Height=6 inches
    plt.grid(True)
    # palette='Dark2'      # palette='Set2')  # Other options: 'Paired', 'Dark2', 'tab10'
    sns.lineplot(data=df,linewidth=1) #   The smallest usable linewidth is 0.1
    # sns.scatterplot(data=df)
    plt.title(title)
    plt.savefig(path)
    plt.show(block=True)                    
    print(f"Save losscurve@{path} n={nResult}"  )

def SWEEP_plt(all_results,path,yCol='loss'):
    nResult = len(all_results)
    assert nResult>0
    df = pd.DataFrame()
    no = 0
    # df['iter'] = all_results[0].dfTrain['iter'] 
    for result in all_results:
        dfSrc = result.dfEval   #result.dfTrain
        if yCol=="lr" or yCol=="gNorm":
            dfSrc = result.dfTrain
        assert dfSrc is not None
        rows,cols = dfSrc.shape
        minY,maxY = dfSrc[yCol].min(),dfSrc[yCol].max()  
        new_df = pd.DataFrame()
        # indices = np.linspace(0, rows-1, 200, dtype=int)  # 20 evenly spaced points
        new_df[str(minY)+"@"+result.title] = dfSrc[yCol]        #.iloc[:,2]         
        
        print(f"new_df shape={new_df.shape} head = {new_df.head()}")    
        df = pd.concat([df, new_df], axis=1)
        
        no = no+1
    # Melt DataFrame for seaborn
    # df_melted = df.melt(id_vars='x', var_name='curve', value_name='y')
    plt_df(df,title = f"\"{yCol}\" {nResult} sweeps @'{path}'",path=path)

    # plt.figure(figsize=(20, 12))  # Width=10 inches, Height=6 inches
    # plt.grid(True)
    # sns.lineplot(data=df,linewidth=0.5) #   The smallest usable linewidth is 0.1
    # # sns.scatterplot(data=df)
    # title = f"\"{yCol}\" {nResult} sweeps @'{path}'"
    # plt.title(title)
    # plt.savefig(path)
    # plt.show(block=True)                    
    # print(f"Save losscurve@{path} n={nResult}"  )

# 
def SWEEP_stat(root_dir,plot_path,isNeedLog=False):
    all_results = []
    for root, dirs, files in os.walk(root_dir):
        for cur_dir in dirs:
            # if cur_dir != "fuyouo6_crossover0.6_B40":                continue
            cur_path = os.path.join(root, cur_dir)
            jConfig = None; fLog=None;  dfTrain=None;  dfEval=None
            for f in os.listdir(cur_path):
                f = cur_path + "/" + f 
                if not os.path.isfile(f):   continue                
                fsize = os.path.getsize(f)
                file_path = Path(f)
                if file_path.suffix==".json":
                    with open(f, 'r') as fr:
                        jConfig = json.load(fr)  
                if file_path.suffix==".info":
                    if fsize>2048:
                        with open(f, 'r') as fr:
                            fLog = fr.read() 
                if file_path.suffix==".csv":
                    if fnmatch.fnmatch(f,"*Train*"):
                        dfTrain = pd.read_csv(f, sep=' ',index_col=False)
                    else:
                        dfEval = pd.read_csv(f, sep=' ',index_col=False)        
                        # print(dfEval)    
            if dfTrain is None or dfEval is None:
                continue   
            if isNeedLog and fLog is None:
                continue
            print(f"dfTrain shape={dfTrain.shape} head = {dfTrain.head()}\n loss={dfTrain["loss"]}")    
            title = cur_dir
            result = SWEEP_result(title, cur_dir,jConfig,dfTrain,dfEval,fLog)
            all_results.append(result)
            # SWEEP_plt(all_results,plot_path)
    assert(len(all_results)>0)
    yCol='loss'     #   loss  lr max_|G|  max_|W| gNorm
    SWEEP_plt(all_results,plot_path,yCol=yCol)
    pass

def Plot_csv(path):
    dfAll = pd.read_csv(path, sep=' ',index_col=False)
    picks = []
    filter = ("loss")      #   "G_ffn_up" G_qkv   G_cat_ G_ffn_down loss
    for column in dfAll:        
        if str(column).startswith(filter):
            picks.append(column)
    df = dfAll[picks]
    rows,cols = df.shape
    plot_path = path + ".png"
    plt_df(df,f"{filter}_{cols}_.png",plot_path )
    pass

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="log path of sweep test")
    parser.add_argument("--csv", type=str, help="log path of single csv file")
    # parser.add_argument("--stat", action='store_true')    
    args = parser.parse_args()
    if args.dir:
        SWEEP_stat(args.dir,args.dir+"/sweep_results.png")
    elif args.csv:
        Plot_csv(args.csv)
    exit()    
   
