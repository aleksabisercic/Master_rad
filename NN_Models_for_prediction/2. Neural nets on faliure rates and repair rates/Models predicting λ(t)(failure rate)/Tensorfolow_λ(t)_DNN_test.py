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

''' Loading  λ(t)(failure rate) generated from folder path '''
series = np.load(r'C:\Users\Freedom\Documents\GitHub\Master_rad\Machine_learning_simulations\2. Exponential distribution based prediction\Generating λ(t)(failure rate) and μ(t)(Repair rate)\Numpy λ(t)(failure rate) and μ(t)(Repair rate)\Failure_rates_for_NN.npy')

series_faliur = series.reshape(-1) 
split_time = int(len(series_faliur)*0.8)
time = np.arange(len(series_faliur))
time_train = time[:split_time]
x_train = series_faliur[:split_time]
time_valid = time[split_time:]
x_valid = series_faliur[split_time:]

window_size = 50
batch_size = 256

dataset = windowed_dataset(x_train, window_size, batch_size)
validation_dataset = windowed_dataset(x_valid, window_size, batch_size = len(x_valid))
    
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
history = model.fit(dataset, epochs=1)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()

model.evaluate(validation_dataset)

for x,y in validation_dataset:
    results = model.predict(x)
results = results.reshape(-1)
plt.figure(figsize=(10, 6))

plot_series(time_valid[-10330:], x_valid[-10330:])
plot_series(time_valid[-10330:], results[-10330:])


# model.save('saved_model/my_model') 
# new_model = tf.keras.models.load_model('Tensorflow_Mi_model_DNN')