# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:05:20 2020

@author: Freedom
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xlwt as xl
import pickle
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn import preprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_excel("Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by=['Početak zastoja'])

df = df[['Pocetak_zastoja_u_minutima', 'Vreme_zastoja', 'Vreme_rada', 'Kraj_zastoja_u_miutama']]

k = int(len(df['Vreme_zastoja']))
i = 0

df.reset_index(inplace=True, drop=True)

lista2 = []
lista1 = []

for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	if df["Vreme_zastoja"].iloc[i] > 2000:
		continue
	if df["Vreme_rada"].iloc[i] > 3000:
		continue
	lista1.append(df["Pocetak_zastoja_u_minutima"].iloc[i])
	lista2.append(df["Kraj_zastoja_u_miutama"].iloc[i])

podatci1 = np.array(lista1)
podatci1 = podatci1 - podatci1[0]
podatci2 = np.array(lista2)
podatci2 = podatci2 - podatci1[0]
#podatci1 = podatci1/max(podatci1)

def gen_lambda(podatci1,podatci2, seq_len):
    a = np.zeros(podatci2[-1])
    for i in podatci1:
        a[0] = 1
        if i == 0: continue
        a[i-1] = 1
    lambd = []
    start = 0
    end = seq_len
    for i in range(int(len(a)/seq_len)):
        ls = a[start:end]
        lambd.append(sum(ls))
        start = end
        end += seq_len
    return lambd

def gen_mi(podatci2, seq_len):
    a = np.zeros(podatci2[-1])
    for i in podatci2:
        a[i-1] = 1
    lambd = []
    start = 0
    end = seq_len
    for i in range(int(len(a)/seq_len)):
        ls = a[start:end]
        lambd.append(sum(ls))
        start = end
        end += seq_len
    return lambd

def ploting(lambd, mi, graph_name): 
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(lambd, 'r-')
    plt.ylabel('oktz/intervalu')
    plt.xlabel('vreme') 
    plt.title(graph_name)
    plt.subplot(2, 1, 2)
    plt.plot(mi, 'b-')
    plt.ylabel('popravki/intervalu')
    plt.xlabel('vreme') 
    plt.show()


seq_leng = np.random.uniform(10, 480, size = 15) 
for seq_len in seq_leng:
    a = int(seq_len/60)
    graph_name = str(a) + 'hours interval' + '.png'
    lambd = gen_lambda(podatci1,podatci2, int(seq_len))
    mi = gen_mi(podatci2, int(seq_len))
    ploting(lambd, mi, graph_name )

    
        
    
