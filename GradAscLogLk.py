#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:49:40 2017

@author: salvatore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_table("/home/salvatore/Documents/bank-additional.csv",sep=";")
Data.drop('duration', axis=1, inplace=True)
Data.replace({"pdays":{999:0}},inplace=True)
Data.replace({"y":{"no":0,"yes":1}},inplace=True)
Data["0"] = np.ones(len(Data))
Data[["y","0"]] = Data[["0","y"]]
msk = np.random.rand(len(Data)) < 0.8  #Random assign\n",
tr = Data[msk]
tst = Data[~msk]
tr.head()
Xtr = pd.get_dummies(tr[tr.columns[0:20]])
Ytr = np.array([tr['0']]).T
ptr = np.array([[0.5]]*len(Ytr))

Xtst = pd.get_dummies(tst[tst.columns[0:20]])
Ytst = np.array([tst['0']]).T
ptst = np.array([[0.5]]*len(Ytst))

Btr = np.array([[0]]*len(Xtr.T)) #Initialized to 0

def PofX(B,x):
    PoofX = 1/(1+np.exp(-1*x.dot(B)))
    return PoofX
    
def LossLog(B,x,y):
    p1 = PofX(B,x)
    Loss_i= y*np.log(p1) + (1-y)*np.log(1 - p1)
    Loss_m= np.sum(Loss_i)[0]/-len(y)
    return Loss_m

def Lglklh(B,x,y):
    p = PofX(B,x)
    Liklh = np.sum(y*np.log(p) + (1-y)*np.log(1 - p))
    return Liklh

def LogRegGa(B,x,y,alph,i):
    n = 0
    ALV = [] #Lists to store data
    #Lss = []
    #AlphV = []
    while n<i:
        n = n+1
        print(n)
        Bn = B + alph*x.T.dot(y-PofX(B,x))
        Bn = np.array([Bn[:][0].values]).T
        Err = (Lglklh(B,x,y) - Lglklh(Bn,x,y)) #Differences in likelihoods between Bt-1 vs Bt
        #print(Err)
        ALV.append(Err[0]) #Storing the error value
        B=Bn #Actualizing B values for the next iteration
    else:
        plt.plot(ALV)
        plt.title("Error vs iteration")
        return Bn, ALV