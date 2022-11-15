# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 05:12:07 2020

@author: Freedom
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:20:14 2020

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

df = df[['Vreme_rada','Objekat', 'Vrsta_zastoja' ]]
df.reset_index(inplace = True, drop = True)


lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Vreme_rada"].iloc[i])
	lista1.append(df["Objekat"].iloc[i])

data_X = np.array(lista).reshape(-1,1)
labels_raw = np.array(lista1)

data_Y = []
for label in labels_raw:
    if label == 'BAGER SchRs-800':
        data_Y.append(0)
    elif label == 'DROBILANA':
        data_Y.append(1)
    else:
        data_Y.append(2)

series = np.array(data_Y)

dataY = np.array(data_Y).reshape(-1,1)

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length - 1)):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)	
    return np.array(x),np.array(y)

window_size = 50   

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

x,y = sliding_windows(dataY, dataY, window_size)
split = int(0.8*len(x))
x_train = x[:split]
y_train = y[:split]
x_valid = x[split:]
y_valid = y[split:]
batch_size = 32

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


# window_size = 20
batch_size = 32
shuffle_buffer_size = 150

# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True,input_shape=[50, 1], dropout=0.1), 
    tf.keras.layers.LSTM(25, dropout=0.1),
    tf.keras.layers.Dense(3, activation='softmax')
])
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=200, shuffle=False, validation_split=0.1)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

rnn_eval = model.evaluate(x_valid, y_valid)

# wb = xl.Workbook ()
# ws1 = wb.add_sheet("LSTM_seq_Vrsta_predvidja_Vrstu")
# ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
# ws1.row(0).write(0, ws1_kolone[0])
# ws1.row(0).write(1, ws1_kolone[1])
# ws1.row(0).write(2, ws1_kolone[2])
# ws1.row(0).write(3, ws1_kolone[3])

# simulation_name = 'LSTM_seq_Vrsta_predvidja_Vrstu' 
# path =  simulation_name 

# ws1.row(1).write(0, simulation_name + "_" +'LSTM')
# ws1.row(1).write(1, history.history["loss"][-1])
# ws1.row(1).write(2, (rnn_eval[0]))
# ws1.row(1).write(3, (rnn_eval[1]))
