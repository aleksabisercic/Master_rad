# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:54:42 2020

@author: Freedom
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlwt as xl

''' Loading  λ(t)(failure rate) generated from folder path '''
series = np.load('Failure_rates_for_NN.npy')

series_faliur = series.reshape(-1) 
split_time = int(len(series_faliur)*0.8)
time = np.arange(len(series_faliur))
time_train = time[:split_time]
x_train = series_faliur[:split_time]
time_valid = time[split_time:]
x_valid = series_faliur[split_time:]

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

window_size = 256
batch_size = 512
shuffle_buffer_size = 1000
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
test_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 70)
])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-4 / 10**(epoch / 50))

optimizer = tf.keras.optimizers.Adam(lr=1e-8)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=1) #, callbacks=[lr_schedule])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 60])
rnn_eval = model.evaluate(test_set)

wb = xl.Workbook ()
ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

simulation_name = 'Tensorfolow_?(t)_CNNandLSTM' 
path = 'modelLSTM/'+ simulation_name + '.pt'

ws1.row(1).write(0, simulation_name + "_" +'LSTM')
ws1.row(1).write(1, history.history["loss"][-1])
ws1.row(1).write(2, int(rnn_eval[0]))
ws1.row(1).write(3, int(rnn_eval[1]))

wb.save("Excel tabels (results)/"+ simulation_name + ".xls")

model.save(path)
#new_model = tf.keras.models.load_model('modelLSTM/'+ simulation_name + '.pt') #loading model