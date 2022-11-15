# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:50:51 2020

@author: Freedom
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xlwt as xl
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_excel("Zastoji.xlsx", index_col = 0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Vreme_rada','Vreme_zastoja', 'Vrsta_zastoja' ]]
df.reset_index(inplace = True, drop = True)


lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Vreme_rada"].iloc[i])
	lista1.append(df["Vrsta_zastoja"].iloc[i])

data_X = np.array(lista).reshape(-1,1)
labels_raw = np.array(lista1)

data_Y = []
for label in labels_raw:
    if label == 'Masinski':
        data_Y.append(0)
    elif label == 'Elektro':
        data_Y.append(1)
    elif label == 'Ostalo':
        data_Y.append(2)

dataX = np.array(data_Y).reshape(-1,1)
dataY = np.array(data_Y).reshape(-1,1)

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length )):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length ]
        x.append(_x)
        y.append(_y)	
    return np.array(x),np.array(y)
window_size = 50
   
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
    
x,y = sliding_windows(dataX, dataY, window_size)
split = int(0.8*len(x))
x_train = x[:split]
y_train = y[:split]
x_valid = x[split:]
y_valid = y[split:]
batch_size = 8
shuffle_buffer_size = 1


model = tf.keras.models.load_model( 'proba' )
model.predict(x_valid)