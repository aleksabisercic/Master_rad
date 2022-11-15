# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:29:08 2020

@author: Freedom
"""
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import xlwt as xl

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
series = np.load('Repair_rates_for_NN.npy')


series_faliur = series.reshape(-1) 
split_time = int(len(series_faliur)*0.8)
time = np.arange(len(series_faliur))
time_train = time[:split_time]
x_train = series_faliur[:split_time]
time_valid = time[split_time:]
x_valid = series_faliur[split_time:]

window_size = 50
batch_size = 512


dataset_valid = windowed_dataset(x_valid, window_size, batch_size=x_valid.shape[0])  
dataset = windowed_dataset(x_train, window_size, batch_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 60.0)
])


optimizer = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=150)

rnn_eval = model.evaluate(dataset_valid)

forcast = model.predict(dataset_valid)
forcast = forcast.reshape(-1)
plot_series(time_valid[:-window_size], x_valid[:-window_size])
plot_series(time_valid[:-window_size], forcast)

wb = xl.Workbook ()
ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

simulation_name = 'μ(t)_LSTM' 
path = 'modelLSTM/'+ simulation_name + '.pt'

ws1.row(1).write(0, simulation_name + "_" +'LSTM')
ws1.row(1).write(1, history.history["loss"][-1])
ws1.row(1).write(2, int((rnn_eval[1])**2))
ws1.row(1).write(3, int(rnn_eval[1]))

wb.save("Excel tabels (results)/"+ simulation_name + ".xls")

model.save(path)
#new_model = tf.keras.models.load_model('modelLSTM/'+ simulation_name + '.pt') #loading model

# forecast = []
# for time in range(len(series) - window_size):
#   forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# forecast = forecast[split_time-window_size:]
# forecast = np.array(forecast)
# forecast = forecast.reshape(1,-1)

# results = forecast.reshape(-1)
# plt.figure(figsize=(10, 6))

# plot_series(time_valid, x_valid)
# plot_series(time_valid, results)

# popravke_prethodni = x_valid[-window_size:]
# mi = []
# for i in range(1000):
#     lambda_dt = model.predict(popravke_prethodni[np.newaxis])
#     mi.append(int(lambda_dt.reshape(-1)))
#     popravke_prethodni = np.roll(popravke_prethodni, -1)
#     popravke_prethodni[-1] = lambda_dt

# model.save('saved_model/my_model') 
# new_model = tf.keras.models.load_model('Tensorflow_Mi_model_DNN')