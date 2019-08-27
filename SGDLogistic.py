#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:10:19 2017

@author: salvatore
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_table("/home/salvatore/Documents/bank-additional.csv",sep=";")
Data.drop('duration', axis=1, inplace=True)
Data.replace({"pdays":{999:0}},inplace=True)
Data.replace({"y":{"no":0,"yes":1}},inplace=True)
Data["Ones"] = np.ones(len(Data))
DT = pd.get_dummies(Data)
DT = DT[['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school', 'education_illiterate',
       'education_professional.course', 'education_university.degree',
       'education_unknown', 'default_no', 'default_unknown', 'default_yes',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',
       'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
       'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure',
       'poutcome_nonexistent', 'poutcome_success', 'Ones','y']]#This is just to assure that
                                                               # the "y" values are at the end
msk = np.random.rand(len(DT)) < 0.8  #Random assign\n",
tr = DT[msk]
tst = DT[~msk]
Xtr = np.array(tr[tr.columns[0:63]])
Ytr = np.array([tr['y']]).T

Xtst = np.array(tst[tr.columns[0:63]])
Ytst = np.array([tst['y']]).T

def PofX(B,x):
    PoofX = 1/(1+np.exp(-1*x.dot(B)))
    return PoofX

def Lglklh(B,x,y):
    p = PofX(B,x)
    Liklh = np.sum(y*np.log(p) + (1-y)*np.log(1 - p))
    return Liklh

def LossLog(B,x,y):
    p1 = PofX(B,x)
    Loss_i= y*np.log(p1) + (1-y)*np.log(1 - p1)
    Loss_m= np.sum(Loss_i)/-len(y)
    return Loss_m

def SGALogReg(x,y,mu,epoch):
    Lss = []
    ALV = []
    #h = 0
    Btr = np.array([[0]]*len(Xtr.T)) #Initialized to 0
    for n in np.arange(epoch):
        for i in np.arange(len(x)):
            p= PofX(Btr,x[i])
            Gd = np.array([x[i]*((y[i]-p))]).T
            #h = h + Gd*Gd
            Bn = Btr+mu*Gd
            Error = Lglklh(Btr,x[i],y[i]) - Lglklh(Bn,x[i],y[i]) 
            Loss  = LossLog(Bn,x,y)
            Lss.append(Loss)
            ALV.append(Error)
            #print ("Epoch",n+1,"It#:",i+1)
            #print("h Vector", np.round(h.T,2))
            #print("B:",np.round(Bn.T,2))
            #print("Error",np.round(Error.T,2))
            #print("Loss",np.round(Loss,2))
            Btr = Bn
    plt.plot(Lss)
    plt.title("Loss vs iterations")
    plt.xlabel("Iteration")    
    plt.ylabel("Error as Log of Prob")
    plt.show()
    plt.plot(ALV)
    plt.title("Error vs iterations")
    plt.xlabel("Iteration")    
    plt.ylabel("Error as LogLikelihood")
    plt.show()
    return #Lss, ALV,Bn


def AdaGSGALogReg(x,y,mu,epoch):
    Lss = []
    ALV = []
    h = 0
    Btr = np.array([[0]]*len(Xtr.T)) #Initialized to 0
    for n in np.arange(epoch):
        for i in np.arange(len(x)):
            p= PofX(Btr,x[i])
            Gd = np.array([x[i]*((y[i]-p))]).T
            h = h + Gd*Gd
            Bn = Btr+mu/(h**0.5)*Gd
            Error = Lglklh(Btr,x[i],y[i]) - Lglklh(Bn,x[i],y[i]) 
            Loss  = LossLog(Bn,x,y)
            Lss.append(Loss)
            ALV.append(Error)
            print ("Epoch",n+1,"It#:",i+1)
            print("h Vector", np.round(h.T,2))
            #print("B:",np.round(Bn.T,2))
            print("Error",Error.T)
            #print("Loss",np.round(Loss,2))
            Btr = Bn
    plt.plot(Lss)
    plt.title("Loss vs iterations")
    plt.xlabel("Iteration")    
    plt.ylabel("Error as Log of Prob")
    plt.show()
    plt.plot(ALV)
    plt.title("Error vs iterations")
    plt.xlabel("Iteration")    
    plt.ylabel("Error as LogLikelihood")
    plt.show()
    return Lss, ALV,Bn





