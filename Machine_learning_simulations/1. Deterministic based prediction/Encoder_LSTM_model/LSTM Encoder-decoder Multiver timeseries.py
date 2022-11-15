
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

k = int(len(df['Vreme_zastoja']))
i = 0

#drop outliers
# z_scores = sc.stats.zscore(df)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# new_df = df[filtered_entries]
# new_df.reset_index(inplace=True, drop=True)

df.reset_index(inplace=True, drop=True)

lista = []
lista1 = []

for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	if df["Vreme_zastoja"].iloc[i] > 2000:
		continue
	if df["Vreme_rada"].iloc[i] > 2000:
		continue
	lista.append(df["Vreme_zastoja"].iloc[i])
	lista.append(df["Vreme_rada"].iloc[i])

sc = MinMaxScaler()

podatci = np.array(lista)
podatci = podatci.reshape(-1, 1)
podatci1 = np.array(lista)
podatci1 = podatci1.reshape(-1, 1)
#datax =  sc.fit_transform(podatci1)

datay = podatci
datax = podatci1
#datax = preprocessing.normalize(datax)

# Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length - 2)):
        if (i % 2) != 0: continue
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length]
        _y1 = datay[i + seq_length + 1]
        x.append(_x)
        y.append(_y)	
        y.append(_y1)	
    y = np.array(y)
    x = np.array(x)
    
    return x.reshape(len(x),int(seq_length/2),2), y.reshape(len(x),2)

timesteps = 50
x,y = sliding_windows(datax, datay, timesteps)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
 			
trainX = np.array(x[0:train_size])			
trainY = np.array(y[0:train_size])
#reshape outputs into (samples, timesteps, features) 	
trainY	= np.expand_dims(trainY, axis=1)	

testX = np.array(x[train_size:len(x)])
testY = np.array(y[train_size:len(y)])
#reshape outputs into (samples, timesteps, features) 	
testY	= np.expand_dims(testY, axis=1)	

time = np.arange(len(x))
time_train = time[:train_size]
time_valid = time[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

n_timestaps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
num_neurons_LSTM = [timesteps]#, 2*timesteps, 3*timesteps]
wb = xl.Workbook ()
ws1 = wb.add_sheet("LSTM_Enc_Dec_razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss","Validation accuracy(MAE)" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])
i = 1
for num_neur in num_neurons_LSTM:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(num_neur, activation='relu', input_shape=[n_timestaps, n_features]))
    model.add(tf.keras.layers.RepeatVector(n_outputs))
    model.add(tf.keras.layers.LSTM(num_neur, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_neur/2, activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2)))
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=["mae"])
    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 / 10**(epoch / 40))

    history = model.fit(trainX, trainY, batch_size=batch_size, validation_split=0.1, epochs=500) # , callbacks=[lr_schedule])
    
    # plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis([1e-8, 1e-4, 0, 60])
    rnn_eval = model.evaluate(test_dataset)
    rnn1_forecast = model.predict(test_dataset)
    
    #reevalueting with prediction
    rnn1_forecast = model.predict(testX)
    final_forcast = rnn1_forecast.reshape(rnn1_forecast.shape[0],rnn1_forecast.shape[2])
    for_test_y = testY.reshape(rnn1_forecast.shape[0],rnn1_forecast.shape[2])
    losses = tf.keras.losses.mean_absolute_error(for_test_y, final_forcast)
    LOSS = sum(losses)/len(losses)                                             
    
    simulation_name = 'LSTM_encoder_decoder' 
    
    ws1.row(i).write(0, simulation_name + "_" + str(num_neur))
    ws1.row(i).write(1, history.history["loss"][-1])
    ws1.row(i).write(2, int(rnn_eval[0]))
    ws1.row(i).write(3, int(rnn_eval[1]))
    i = i + 1
wb.save("Excel tabels (results)/"+ simulation_name + ".xls")
path = simulation_name + str(num_neur) + '.pt'
model.save(simulation_name)
print(history.history["loss"][-100:])
#new_model = tf.keras.models.load_model( simulation_name ) #loading model