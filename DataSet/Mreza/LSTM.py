# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:24:58 2020

@author: Freedom
"""

import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
import xlwt as xl

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_excel("Zastoji.xlsx", index_col = 0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Pocetak_zastoja_u_minutima', 'Vreme_zastoja', 'Vreme_rada']]
df.reset_index(inplace = True, drop = True)
			   
k = int(len(df['Vreme_zastoja']))
i=0


while(i<k):
    if (df['Vreme_zastoja'].iloc[i] > 2000 or df['Vreme_rada'].iloc[i] > 2000):
        df.drop(df.index[i],inplace=True)
        k = k - 1
    i = i + 1
		
df.reset_index(inplace = True, drop = True)

lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Pocetak_zastoja_u_minutima"].iloc[i])
	lista1.append(df["Pocetak_zastoja_u_minutima"].iloc[i])
	
Podatci = np.array(lista)
starty= int(len(Podatci)*0.8)

test_data = Podatci[starty:int(len(Podatci)*0.9)]
train_data = Podatci[:starty]

scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))

class LSTM_model(nn.Module):
  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(LSTM_model, self).__init__()
    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers
    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )
    self.linear = nn.Linear(in_features=n_hidden, out_features=1)
  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )
  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred

def train_model(
  model,
  train_data,
  train_labels,
  test_data=None,
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')
  optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 60
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)
  for t in range(num_epochs):
    model.reset_hidden_state()
    y_pred = model(X_train)
    loss = loss_fn(y_pred.float(), y_train)
    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()
      if t % 10 == 0:
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')
    train_hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
  return model.eval(), train_hist, test_hist

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50	
a1 = [ 30  ]
a2 = [ 2 ]

wb = xl.Workbook ()
ws1 = wb.add_sheet("Rezultat simulacije")
ws1_kolone = ["Ime simulacije","Poslednji rezultat", "Najmanji rezultat", "Srednja vrednost rezultat" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

counter = 1 
for num_hid in a1:
	for num_lay in a2:
			X_train, y_train = create_sequences(train_data, seq_length)
			X_test, y_test = create_sequences(test_data, seq_length)
			X_train = torch.from_numpy(X_train).float()
			y_train = torch.from_numpy(y_train).float()
			X_test = torch.from_numpy(X_test).float()
			y_test = torch.from_numpy(y_test).float()
			
			
			model = LSTM_model(
			  n_features=1,
			  n_hidden=num_hid,
			  seq_len=seq_length,
			  n_layers=num_lay
			)
			model, train_hist, test_hist = train_model(
			  model,
			  X_train,
			  y_train,
			  X_test,
			  y_test
			)
			plt.plot(train_hist, label="Training loss")
			plt.plot(test_hist, label="Test loss")
			plt.legend();
			
			with torch.no_grad():
			  test_seq = X_test[:1]
			  preds = []
			  for _ in range(len(X_test)):
			    y_test_pred = model(test_seq)
			    pred = torch.flatten(y_test_pred).item()
			    preds.append(pred)
			    new_seq = test_seq.numpy().flatten()
			    new_seq = np.append(new_seq, [pred])
			    new_seq = new_seq[1:]
			    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
			
			true_cas = scaler.inverse_transform(
			    np.expand_dims(y_test.flatten().numpy(), axis=0)
			).flatten()
			predicted_cas = scaler.inverse_transform(
			  np.expand_dims(preds, axis=0)
			).flatten()
			
			plt.plot(
			  Podatci.index[:len(train_data)],
			  scaler.inverse_transform(train_data).flatten(),
			  label='Historical'
			)
			plt.plot(
			  Podatci.index[len(train_data):len(train_data) + len(true_cas)],
			  true_cas,
			  label='Real'
			)
			plt.plot(
			  Podatci.index[len(train_data):len(train_data) + len(true_cas)],
			  predicted_cas,
			  label='Predicted'
			)
			plt.legend();

