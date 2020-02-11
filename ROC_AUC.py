# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:04:05 2020

@author: HAN
"""

import pandas as pd
import csv
import re
import matplotlib.pyplot as plt

file_0="L:/ICL/Vase_project/code_done/test0.csv"
file_1="L:/ICL/Vase_project/code_done/test1.csv"

y = []
y_hat = []
y_hat_prob = []

def append_0(n):
    for i in range (n):
        y.append(0)

def append_1(n):
    for i in range(n):
        y.append(1)
        
append_0(1069)
append_1(2235)

if __name__ == "__main__":
    f = open(file_0,"r")
    lines = f.readlines()
    for line in lines:
        if "bad" in line:
            y_hat.append(0)
        elif "good" in line:
            y_hat.append(1)
            
if __name__ == "__main__":
    g = open(file_1,"r")
    lines = g.readlines()
    for line in lines:
        if "bad" in line:
            y_hat.append(0)
        elif "good" in line:
            y_hat.append(1)


            
file0=pd.read_csv("L:/ICL/Vase_project/code_done/test0.csv")
data=list(file0.iloc[:,1])

for i in data:
    tmp=i.split(':')
    if tmp[0]=='good':
        y_hat_prob.append(float(tmp[1]))
    else:
        y_hat_prob.append(1-float(tmp[1]))

file1=pd.read_csv("L:/ICL/Vase_project/code_done/test1.csv")
data1=list(file1.iloc[:,1])

for i in data1:
    tmp=i.split(':')
    if tmp[0]=='good':
        y_hat_prob.append(float(tmp[1]))
    else:
        y_hat_prob.append(1-float(tmp[1]))

def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive

def get_precision(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive

def get_tnr(y, y_hat):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative

def get_roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
    return ret

def get_auc(y, y_hat_prob):
    roc = iter(get_roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc


points = get_roc(y,y_hat_prob)
df = pd.DataFrame(points, columns=["True positive rate", "False positive rate"])
print("AUC is %.3f." % get_auc(y, y_hat_prob))
df.plot(x="False positive rate", y="True positive rate", label="ROC")

plt.savefig("test.svg", format="svg")


