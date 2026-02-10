import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path  
import json
import argparse 
import json
import time
import math #Use math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
from SweepHyParams import koifish_one,bubble_one,pangpi_one

# source /home/cys/anaconda3/bin/activate base
# clear && pytest -v -s ./cases/

sExe = "./bin/koifish "
most_iter = 10
def CheckResult(df,iter,golden,title="",rel_tol=1e-05):
    # print(df)
    a = df["loss"][iter]
    print(f"{title} loss={a} golden={golden}\n")
    assert math.isclose(a,golden,rel_tol=rel_tol, abs_tol=0.0) 

def test_chat_qwen3_0_6B():  
    content = bubble_one("chat_qwen3_0.6b","--tokenizer ./assets/tokenizer_151936.bin --hf ./Models/Qwen3-0.6B/ --prompts \"hello\"")  #./cases/qwen3/qwen3_0.6B.json
    assert "Hello! How can I assist you today?" in content

def test_chat_qwen3_4B():    
    content = bubble_one("chat_qwen3_4B","--tokenizer ./assets/tokenizer_151936.bin --hf ./Models/Qwen3-4B/ --prompts \"Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?\"")  #  "./cases/qwen3/qwen3_4B.json"
    assert "Answer: \\boxed{1}" in content  or "Answer: 1" in content or "Answer:1" in content or "answer:1" in content  # "Answer: 1"   "✅ Final Answer:1 ✅"  "Answer: 1 ✅"  "✅Answer: 1"
    # assert content=="Hello! How can I assist you today? 😊\n" or content=="Hello! It seems there was a small glitch. 😊 How can I assist you today?\n"
def test_chat_qwen3_4B_1():    
    content = bubble_one("chat_qwen3_4B","p1 --tokenizer ./assets/tokenizer_151936.bin --hf ./Models/Qwen3-4B/ --prompts \"Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?\"")  #  "./cases/qwen3/qwen3_4B.json"
    assert "Answer: \\boxed{1}" in content  or "Answer: 1" in content or "Answer:1" in content or "answer:1" in content  # "Answer: 1"   "✅ Final Answer:1 ✅"  "Answer: 1 ✅"  "✅Answer: 1"

def xtest_batch_qwen3_4B():  
    nlayer = 36 #   28 36
    path = "./cases/qwen3/qwen3_0.6B.json" if nlayer==28 else"./cases/qwen3/qwen3_4B.json"
    with open(path, 'r') as f:
        jConfig_0 = json.load(f)    
    start = time.time()
    for layer in range (nlayer):
        jConfig = jConfig_0
        jPath = f"./tests/qwen3_4B_layer{layer}.json"
        with open(jPath, 'w') as f:
            jConfig.setdefault("quantizer", {})["MINI"] = [f"model.layers.{layer}.mlp"]
            jConfig["debug"]["prompts"] = ["hello"]
            jConfig["gpt"]["max_seq_len"] = 128
            json.dump(jConfig, f, indent=4)  # Write JSON to file   
        print(f"\n------- @LAY_{layer} {time.time() - start :.3f}s------")    # ~5s
        content = bubble_one("chat_qwen3_4B",jPath)  
        # Hello! It seems like there might be a small mix-up. I'm Qwen, a large-scale language model developed by Alibaba Cloud. I'm here to help you with any questions or tasks you might have. How can I assist you today? 😊

def test_qwen3_596M():    
    most_iter = 180
    title = "QWen3_596M"
    dfTrain = koifish_one(title, sExe, "./cases/qwen3/qwen3_1.json", most_iter=most_iter, train_csv="./Train@[climb]_info_.csv")    
    CheckResult(dfTrain,most_iter,6.942,title=title,rel_tol=0.001)      #   6.942169    7.589036

def test_qwen2_494M():    
    most_iter = 70
    title = "QWen2.5_494M"
    dfTrain = koifish_one(title, sExe, "./cases/qwen3/qwen25_1.json", most_iter=most_iter, train_csv="./Train@[shake]_info_.csv")    
    CheckResult(dfTrain,most_iter,2.979,title=title,rel_tol=0.001)      #   2.873

def test_gpt2_1558M():    
    title = "1558M"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/1558M_F8_B80/F8_B32.json", most_iter=most_iter)
    CheckResult(dfTrain,most_iter,9.446,title=title) 

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



def xtest_gpt2_774M():    
    title = "774M"
    dfTrain = koifish_one(title, sExe, "./cases/gpt2/774M_Shard50_F6_B80/F9_B40.json", most_iter=most_iter)
    CheckResult(dfTrain,most_iter,9.409,title=title)    #   61     loss=7.318967
    # assert math.isclose(a,9.504,rel_tol=1e-05, abs_tol=0.0)     

def no_test_pp_gpt2():    
    most_iter = 70
    title = "pangpi_gpt2"
    sExe = "./bin/pangpi "
    dfLoss,exit_code = pangpi_one(title, sExe, "./hy-tmp/case/124M/GPT2_fuyou.fish --hellaswag ./cases/datasets/hellaswag_val.bin") 
    # dfLoss = pd.read_csv("/home/cys/rnd/lic/Eval_loss.csv", sep=' ',index_col=False)
    # print(dfLoss)
    CheckResult(dfLoss,0,0.2476,title=title,rel_tol=1e-03)       #   0.24766387 0.01475318    


    
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
    # test_gpt2_774M()

    # test_chat_qwen3_0_6B()  
    #test_qwen3_596M()
    # test_chat_qwen3_4B_1()
    # xtest_batch_qwen3_4B()

    # test_pp_gpt2()
    # test_gpt2_124M()
    # test_gpt2_124M_fuyou6()
    # test_gpt2_1558M()
    test_qwen2_494M()
    # # 
    # test_gpt2_1558M()
    # koifish_one("124M", sExe, "./cases/gpt2/124M_shard50_F6_lr0.001/F6_lr0.001.json", most_iter=most_iter)
    


