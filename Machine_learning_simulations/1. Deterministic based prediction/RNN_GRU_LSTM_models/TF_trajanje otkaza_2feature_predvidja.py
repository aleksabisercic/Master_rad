# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 00:46:00 2020

@author: Freedom
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:40:45 2020

@author: Freedom
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xlwt as xl
import pickle
import seaborn as sns
import scipy as sc
import tensorflow as tf

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn import preprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch.utils.data import TensorDataset, DataLoader

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_excel("Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by=['Početak zastoja'])

df = df[['Pocetak_zastoja_u_minutima', 'Vreme_zastoja', 'Vreme_rada']]
df.reset_index(inplace=True, drop=True)

#drop outliers
z_scores = sc.stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = df[filtered_entries]
new_df.reset_index(inplace=True, drop=True)
print(new_df)

# bottom 10 and upper 10 percentail of data drop

lista = []
lista1 = []

for i in range(0, len(new_df.index)):  # df['Vreme_zastoja']:
	if new_df["Vreme_zastoja"].iloc[i] > 2000:
		continue
	if new_df["Vreme_rada"].iloc[i] > 2000:
		continue
	lista1.append(new_df["Vreme_zastoja"].iloc[i])
	lista1.append(new_df["Vreme_rada"].iloc[i])
	lista.append(df["Vreme_zastoja"].iloc[i])

podatci = np.array(lista)
podatci2 = np.array(lista1)
podatci = podatci.reshape(-1, 1)
podatci1 = np.array(lista)
podatci1 = podatci1.reshape(-1, 1)

# Scaling the input data
scaler = MinMaxScaler()
scaler.fit(podatci1)

datax = scaler.transform(podatci1)
datay = podatci


# Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length - 2)):
        if (i % 2) != 0: continue
        _x = datax[i:(i + seq_length)]
#        _x = preprocessing.normalize(_x)
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)	
    y = np.array(y)
    x = np.array(x)
    
    return x.reshape(len(x),int(seq_length/2),2), y.reshape(-1) #y.reshape(len(x),1)

x,y = sliding_windows(datax, datay, 50)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
 			
trainX = np.array(x[0:train_size])			
trainY = np.array(y[0:train_size])
 			
testX = np.array(x[train_size:len(x)])
testY = np.array(y[train_size:len(y)])

time = np.arange(len(x))
time_train = time[:train_size]
time_valid = time[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
batch_size = 16
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


def model_eval(model,test_dataset ):
    forecast = model.evaluation(test_dataset)
    return forecast

model = tf.keras.models.Sequential([
  # tf.keras.layers.Conv1D(filters=32, kernel_size=5,
  #                     strides=1, padding="causal",
  #                     activation="relu",
  #                     input_shape=[None, 2]),
  tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[25, 2]),
  tf.keras.layers.LSTM(25, dropout=0.2),
#  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1,  activation="relu"),
  tf.keras.layers.Lambda(lambda x: x * 2000)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
     lambda epoch: 1e-4 / 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(lr=1e-4)

model.compile(loss= 'MSE',
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(trainX, trainY, batch_size=16,  epochs=300, validation_split=0.2, shuffle=False)# , callbacks=[lr_schedule])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 60])
rnn_eval = model.evaluate(test_dataset)
rnn1_forecast = model.predict(test_dataset)

wb = xl.Workbook ()
ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

simulation_name = 'TF_trajanje otkaza_2feature_predvidjanja' 
path =  simulation_name 

ws1.row(1).write(0, simulation_name + "_" +'LSTM')
ws1.row(1).write(1, history.history["loss"][-1])
ws1.row(1).write(2, int(rnn_eval[0]))
ws1.row(1).write(3, int(rnn_eval[1]))

wb.save("Excel tabels (results)/"+ simulation_name + ".xls")

model.save(path)
#new_model = tf.keras.models.load_model( simulation_name ) #loading model