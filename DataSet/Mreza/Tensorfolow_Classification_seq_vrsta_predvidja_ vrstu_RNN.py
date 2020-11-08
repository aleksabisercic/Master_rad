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

df = df[['Vreme_zastoja', 'Vrsta_zastoja' ]]
df.reset_index(inplace = True, drop = True)


df = df[df.Vreme_zastoja < 2000]

lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Vreme_zastoja"].iloc[i])
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

series_zastoj = np.array(lista).reshape(-1)
series = np.array(data_Y).reshape(-1)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


def sliding_windows(datax, datay, seq_length):
    x = []
    y = []

    for i in range(len(datax) - seq_length - 1):
        _x = datax[i:(i + seq_length)]
#        _x = preprocessing.normalize(_x)
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)			

    return np.array(x), np.array(y)

split_time = int(len(series)*0.8)
time = np.arange(len(series))
time_train = time[:split_time]
x_train = series_zastoj[:split_time]
y_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
y_valid = series[split_time:]

window_size = 40
shuffle_buffer_size = 300

x_t, y_t = sliding_windows(x_train, y_train, window_size)
x_v, y_v = sliding_windows(x_valid, y_valid, window_size)
print(x_t.shape, y_t.shape)

#train_set = windowed_dataset(x_train, window_size, batch_size=1, shuffle_buffer=shuffle_buffer_size)
    
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(30),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(3, activation='softmax'),
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam()
model.compile(loss= 'sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(x_t,y_t, epochs=150)

#test_set = windowed_dataset(x_t, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
print( model.evaluate(x_t, y_t) )
aleksa = model.predict(x_t)

results = []
for probabilities in aleksa:
    if probabilities[0] > probabilities[1] and probabilities[0] > probabilities[2]:
        results.append(0)
    elif probabilities[1] > probabilities[0] and probabilities[1] > probabilities[2]:
        results.append(1)
    else:
        results.append(2)

results = np.array(results)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_v, results))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])

# loss = history.history['loss']
# epochs = range(len(loss))
# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.show()

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