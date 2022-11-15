# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:22:56 2020

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

df = df[['Vreme_zastoja','Objekat', 'Vrsta_zastoja' ]]
df.reset_index(inplace = True, drop = True)


lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Vreme_zastoja"].iloc[i])
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

dataX = np.array(lista).reshape(-1)
dataY = np.array(data_Y).reshape(-1,1)

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length - 1)):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length-1]
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
batch_size = 32
shuffle_buffer_size = 150

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(35, activation="relu"), 
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(3, activation='softmax')
])
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=250, shuffle=False)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

rnn_eval = model.evaluate(x_valid, y_valid)
print(model.predict(x_valid)[-100:])
# wb = xl.Workbook ()
# ws1 = wb.add_sheet("DNN_seq_popravke_u_otk")
# ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
# ws1.row(0).write(0, ws1_kolone[0])
# ws1.row(0).write(1, ws1_kolone[1])
# ws1.row(0).write(2, ws1_kolone[2])
# ws1.row(0).write(3, ws1_kolone[3])

# simulation_name = 'TF_DNN_seq_popravke_predvidja_dropout otkaz' 
# path =  simulation_name 

# ws1.row(1).write(0, simulation_name + "_" +'LSTM')
# ws1.row(1).write(1, history.history["loss"][-1])
# ws1.row(1).write(2, (rnn_eval[0]))
# ws1.row(1).write(3, (str(rnn_eval[1])))

# wb.save( simulation_name + ".xls")


# forecast = []
# for time in range(len(series) - window_size):
#   forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# results = []
# forecast = forecast[split_time-window_size:]
# forecast = np.array(forecast)
# forecast = forecast.reshape(forecast.shape[0],forecast.shape[2])
# for probabilities in forecast:
#     if probabilities[0] > probabilities[1] and probabilities[0] > probabilities[2]:
#         results.append(0)
#     elif probabilities[1] > probabilities[0] and probabilities[1] > probabilities[2]:
#         results.append(1)
#     else:
#         results.append(2)
# results = np.array(results)
# plt.figure(figsize=(10, 6))

# plot_series(time_valid[100:150], x_valid[100:150])
# plot_series(time_valid[100:150], results[100:150])

# print(accuracy_score(x_valid, results))