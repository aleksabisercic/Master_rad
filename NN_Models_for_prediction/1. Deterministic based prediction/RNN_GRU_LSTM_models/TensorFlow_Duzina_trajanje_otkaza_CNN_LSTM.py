# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:54:42 2020

@author: Freedom
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt as xl

df = pd.read_excel(r"C:\Users\Freedom\Documents\GitHub\Master_rad\DataSet\Zastoji.xlsx", index_col=0)
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

series = np.array(lista).reshape(-1)
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
split_time = int(len(series)*0.8)
time = np.arange(len(series))
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

def windowed_dataset(series, window_size, batch_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

window_size = 50
batch_size = 32
shuffle_buffer_size = 1
train_set = windowed_dataset(x_train, window_size, batch_size)
val_set = windowed_dataset(x_valid, window_size, batch_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  # tf.keras.layers.Conv1D(filters=32, kernel_size=5,
  #                     strides=1, padding="causal",
  #                     activation="relu",
  #                     input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
  tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 70)
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#      lambda epoch: 1e-4 / 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
model.fit(train_set, epochs=10)
# history = model.fit(train_set, epochs=100 )#, callbacks=[lr_schedule])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 60])
rnn_eval = model.evaluate(val_set)
# rnn_forecast = model_forecast(history.model, series[..., np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# wb = xl.Workbook ()
# ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
# ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
# ws1.row(0).write(0, ws1_kolone[0])
# ws1.row(0).write(1, ws1_kolone[1])
# ws1.row(0).write(2, ws1_kolone[2])
# ws1.row(0).write(3, ws1_kolone[3])

# simulation_name = 'LSTM encoder_decoder Multivere Timeseries' 
# path = 'modelLSTM/'+ simulation_name + '.pt'

# ws1.row(1).write(0, simulation_name + "_" +'RNN')
# ws1.row(1).write(1, history.history["loss"][-1])
# ws1.row(1).write(2, int(rnn_eval[0]))
# ws1.row(1).write(3, int(rnn_eval[1]))

# wb.save("Excel tabels (results)/"+ simulation_name + ".xls")

# model.save(path)