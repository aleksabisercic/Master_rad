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

df = df[['Vreme_rada', 'Vrsta_zastoja' ]]
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
print(y_valid.shape)
batch_size = 32

model = tf.keras.models.load_model('LSTM encoder_decoder Multivere Timeseries.pt')   #MEO_class_model.pt
model.evaluate(x_valid, y_valid)
model2 = tf.keras.models.load_model('Classification_model')
model2.evaluate(x_valid, y_valid)
#model3 = tf.keras.models.load_model('MEO_class_model.pt')
#model3.evaluate(x_valid, y_valid)
'''def plot_series(time, series, format="-", start=0, end=None):
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

simulation_name = 'MEO_class_model.pt' 
path =  simulation_name

model.save(simulation_name)'''
