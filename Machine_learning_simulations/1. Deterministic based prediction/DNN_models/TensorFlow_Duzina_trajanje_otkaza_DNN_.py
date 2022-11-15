# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:30:32 2020

@author: Freedom
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:29:08 2020

@author: Freedom
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import xlwt as xl
import pickle

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
df = pd.read_excel("Dataset/Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Vreme_zastoja', 'Vrsta_zastoja' , 'Vreme_rada']]
df.reset_index(inplace = True, drop = True)


df = df[df.Vreme_zastoja < 2000]

lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
#	lista.append(df["Vreme_zastoja"].iloc[i])
#	lista.append(df["Vrsta_zastoja"].iloc[i])
	lista.append(df["Vreme_rada"].iloc[i])
#	lista1.append(df["Vrsta_zastoja"].iloc[i])
    
series = np.array(lista).reshape(-1)
labels_raw = np.array(lista1)

def windowed_dataset(series, window_size, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.map(lambda window: (window[:-2], window[-2]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

def sliding_windows(datax, seq_length):
    x = []
    y = []

    for i in range(len(datax) - seq_length - 1):
        if (i % 2) != 0: continue
        _x = datax[i:(i + seq_length)]
#        _x = preprocessing.normalize(_x)
        _y = datax[i + seq_length]
        x.append(_x)
        y.append(_y)			

    return np.array(x), np.array(y).reshape(-1,1)

window_size = 50
batch_size = 52

x,y = sliding_windows(series,window_size )

split_time = int((x.shape[0])*0.8)
time = np.arange(len(series))
x_train = x[:split_time]
y_train = y[:split_time]
x_valid = x[split_time:]
y_valid = y[split_time:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, input_shape = [x_train.shape[1]] ,activation="relu"), 
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(30, activation="relu"), 
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 1000)
])
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
history = model.fit(x_train, y_train, epochs=500, verbose=1,  batch_size=batch_size, validation_split = 0.1, shuffle=False)

loss = history.history['loss']
print(loss[-1])
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()


rnn_eval = model.evaluate(x_valid, y_valid)


wb = xl.Workbook ()
ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

simulation_name = 'TF_Duzina_trajanje_rada_DNN__' 
path = 'DNN model/'+ simulation_name + '.pt'

ws1.row(1).write(0, simulation_name + "_" +'RNN')
ws1.row(1).write(1, history.history["loss"][-1])
ws1.row(1).write(2, int(rnn_eval[0]))
ws1.row(1).write(3, int(rnn_eval[1]))

wb.save("Excel tabels (results)/"+ simulation_name + ".xls")

model.save(path) 
# new_model = tf.keras.models.load_model(path)