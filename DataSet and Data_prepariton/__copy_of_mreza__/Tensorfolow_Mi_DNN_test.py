# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:29:08 2020

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

series = np.load('mi_sirovi_podatci_izbrisi.npy') #mi_series
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def windowed_dataset(series, window_size, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

split_time = int(len(series)*0.75)
time = np.arange(len(series))
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 50
batch_size = 52

dataset = windowed_dataset(x_train, window_size, batch_size)

    
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(15, activation="relu"), 
    tf.keras.layers.Dense(1)
])
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))
# optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam())
history = model.fit(dataset, epochs=150)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
forecast = np.array(forecast)
forecast = forecast.reshape(1,-1)

results = forecast.reshape(-1)
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

popravke_prethodni = x_valid[-window_size:]
mi = []
for i in range(1000):
    lambda_dt = model.predict(popravke_prethodni[np.newaxis])
    mi.append(int(lambda_dt.reshape(-1)))
    popravke_prethodni = np.roll(popravke_prethodni, -1)
    popravke_prethodni[-1] = lambda_dt

model.save('saved_model/my_model') 
new_model = tf.keras.models.load_model('Tensorflow_Mi_model_DNN')