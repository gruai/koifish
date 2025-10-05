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

# pytest -v -s ./cases/
def add(a, b):
    return a + b

sExe = "./bin/koifish "
most_iter = 10
def test_gpt2_124M():    
    most_iter = 70
    dfTrain = koifish_one("124M", sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json", most_iter=most_iter)    
    # print(dfTrain)
    a = dfTrain["loss"][most_iter]
    assert math.isclose(a,7.498,rel_tol=1e-05, abs_tol=0.0) 
    #   [epoch_0]_601    loss=6.142322

def test_gpt2_774M():    
    dfTrain = koifish_one("774M", sExe, "./cases/gpt2/774M_Shard50_F6_B80/F6_B80.json", most_iter=most_iter)
    # print(dfTrain)
    a = dfTrain["loss"][most_iter]
    assert math.isclose(a,9.504,rel_tol=1e-05, abs_tol=0.0) 
    
def test_gpt2_1558M():    
    dfTrain = koifish_one("774M", sExe, "./cases/gpt2/1558M_F8_B80/F8_B80.json", most_iter=most_iter)
    # print(dfTrain)
    a = dfTrain["loss"][most_iter]
    assert math.isclose(a,9.463,rel_tol=1e-05, abs_tol=0.0)

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
    # test_gpt2_774M()
    # test_gpt2_1558M()
    # koifish_one("124M", sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json", most_iter=most_iter)
    


