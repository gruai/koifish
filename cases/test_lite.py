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
import glob
import atexit
import sys
import math #Use math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
from SweepHyParams import koifish_one

# source /home/cys/anaconda3/bin/activate base
# pytest -v -s ./cases/

def add(a, b):
    return a + b

sExe = "./bin/koifish "
most_iter = 10
def CheckResult(df,iter,golden,title=""):
    # print(df)
    a = df["loss"][iter]
    print(f"{title} loss={a} golden={golden}\n")
    assert math.isclose(a,golden,rel_tol=1e-05, abs_tol=0.0) 

def test_gpt2_124M_fuyou6():    
    most_iter = 70
    title = "124M"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json", most_iter=most_iter)    
    CheckResult(dfTrain,most_iter,7.498,title=title)

def test_gpt2_124M():    
    most_iter = 70
    title = "124M_no_fuyou"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/no_fuyou.json", most_iter=most_iter)    
    CheckResult(dfTrain,most_iter,7.467,title=title)

def test_gpt2_774M():    
    title = "774M"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/774M_Shard50_F6_B80/F9_B40.json", most_iter=most_iter)
    CheckResult(dfTrain,most_iter,9.409,title=title)    #   61     loss=7.318967
    # assert math.isclose(a,9.504,rel_tol=1e-05, abs_tol=0.0) 
    
def test_gpt2_1558M():    
    title = "1558M"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/1558M_F8_B80/F8_B40.json", most_iter=most_iter)
    CheckResult(dfTrain,most_iter,9.472,title=title) 

# def test_add_negative_numbers():
#     assert add(-1, -5) == -6

# def test_add_zero():
#     assert add(0, 7) == 7


    
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./SWEEP/",help="log path of sweep test")
    parser.add_argument("--symbol", action='store_true')    
    parser.add_argument("--json", type=str)  
    parser.add_argument("--csv", type=str, help="log path of single csv file")
    args = parser.parse_args()    
    
    sExe = "./bin/koifish "
    test_gpt2_124M()
    # test_gpt2_124M_fuyou6()
    # # test_gpt2_774M()
    # test_gpt2_1558M()
    # koifish_one("124M", sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json", most_iter=most_iter)
    


