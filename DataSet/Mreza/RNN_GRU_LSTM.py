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
import time

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch.autograd import Variable
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

df.reset_index(inplace=True, drop=True)

lista = []
lista1 = []

for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	if df["Vreme_zastoja"].iloc[i] > 2000:
		continue
	lista.append(df["Vreme_zastoja"].iloc[i])

# Scaling the input data
sc = MinMaxScaler()

podatci = np.array(lista)
podatci = podatci.reshape(-1, 1)
podatci1 = np.array(lista)
podatci1 = podatci1.reshape(-1, 1)
#datax =  sc.fit_transform(podatci1)

datay = podatci
datax = podatci1


# Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation

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


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim , n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data  #ovaj korak nije mi bas jasan
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(RNNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def train(train_loader, learn_rate, hidden_dim, number_of_layers, EPOCHS=5, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = number_of_layers
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    elif model_type == "RNN":
        model = RNNNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    LOSS = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):  
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:  #for i in range seq_len.
            counter += 1
            if model_type == "GRU" or model_type == "RNN":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.clock()
        LOSS.append((avg_loss / counter))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    #    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model, LOSS


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    loss = []
    h = model.init_hidden(testY.shape[0])
    for test_x, test_y in test_loader:
        out, h = model(test_x.to(device).float(), h)
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.numpy())
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    for i in range(len(test_x)):
        MSEloss = (outputs[0][i]-targets[0][i])**2
        loss.append(MSEloss)
    Loss = sum(loss)/len(loss)	
    print("Validation loss: {}".format(Loss))
    return outputs, targets, Loss

seq_length =  [ 50 ]
hidden_dim = [ 50  ]
number_of_layers = [ 2 ]

wb = xl.Workbook ()
ws1 = wb.add_sheet("RNN razultati")
ws1_kolone = ["Ime simulacije","Validation Loss", "Training L", "Srednja vrednost rezultat" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])
ws2 = wb.add_sheet("GRU razultati")
ws2_kolone = ["Ime simulacije","Poslednji rezultat", "Najmanji rezultat", "Srednja vrednost rezultat" ]
ws2.row(0).write(0, ws1_kolone[0])
ws2.row(0).write(1, ws1_kolone[1])
ws2.row(0).write(2, ws1_kolone[2])
ws2.row(0).write(3, ws1_kolone[3])
ws3 = wb.add_sheet("LSTM ruzultati")
ws3_kolone = ["Ime simulacije","Poslednji rezultat", "Najmanji rezultat", "Srednja vrednost rezultat" ]
ws3.row(0).write(0, ws1_kolone[0])
ws3.row(0).write(1, ws1_kolone[1])
ws3.row(0).write(2, ws1_kolone[2])
ws3.row(0).write(3, ws1_kolone[3])


counter = 1 
for seq_len in seq_length:
	for hid_dim in hidden_dim:
		for num_layers in number_of_layers:
			
			x, y = sliding_windows(datax, datay, seq_len)
			
			train_size = int(len(y) * 0.9)
			test_size = len(y) - train_size
			
			dataX = np.array(x)
			dataY = np.array(y)
			trainX = np.array(x[0:train_size])
			
			trainY = np.array(y[0:train_size])
			
			testX = np.array(x[train_size:len(x)])
			testY = np.array(y[train_size:len(y)])
			
			#Data loader
			batch_size = 128		
			train_data = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
			train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
			test_data = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))
			test_loader = DataLoader(test_data, shuffle=True, batch_size=testY.shape[0], drop_last=True)
			lr = 0.001
			
			#Training and Validating RNN_model
			rnn_model, rnn_training_loss = train(train_loader, lr, hid_dim,num_layers, model_type="RNN")
			rnn_outputs, targets, rnn_val_loss = evaluate(rnn_model, test_loader)
			
			#Training and Validating GRU_model
			gru_model, gru_training_loss = train(train_loader, lr, hid_dim,num_layers, model_type="GRU")
			gru_outputs, targets, gru_val_loss = evaluate(gru_model, test_loader)
			
			#Training and Validating LSTM_model
			lstm_model, lstm_training_loss = train(train_loader, lr, hid_dim,num_layers, model_type="LSTM")
			lstm_outputs, targets, lstm_val_loss = evaluate(lstm_model, test_loader)
			
			simulation_name = 'seq_len' + '_' + str(seq_len) + 'hid_dim' + '_' +str(hid_dim) + 'num_layers' + '_' + str(num_layers)
			pathRNN = 'modelRNN/'+ simulation_name + '.pth'
			pathGRU = 'modelGRU/'+ simulation_name + '.pth'
			pathLSTM = 'modelLSTM/'+ simulation_name + '.pth'

			#RNN
			last_element = rnn_val_loss[-1]
			min_element = rnn_training_loss[-1]
			mean_element = 'Za sada nista'
			ws1.row(counter).write(0, simulation_name + "_" +'RNN')
			ws1.row(counter).write(1, last_element)
			ws1.row(counter).write(2, min_element)
			ws1.row(counter).write(3, mean_element)

			#save model parametre RNN
			torch.save(rnn_model.state_dict(), pathRNN)
			
						#GRU
			last_element = gru_val_loss[-1]
			min_element = gru_training_loss[-1]
			mean_element = 'Za sada nista'
			ws2.row(counter).write(0, simulation_name + "_" +'RNN')
			ws2.row(counter).write(1, last_element)
			ws2.row(counter).write(2, min_element)
			ws2.row(counter).write(3, mean_element)

			#save model parametre GRU
			torch.save(gru_model.state_dict(), pathGRU)
			
						#LSTM
			last_element = lstm_val_loss[-1]
			min_element = lstm_training_loss[-1]
			mean_element = 'Za sada nista'
			ws3.row(counter).write(0, simulation_name + "_" +'RNN')
			ws3.row(counter).write(1, last_element)
			ws3.row(counter).write(2, min_element)
			ws3.row(counter).write(3, mean_element)

			#save model parametre LSTM
			torch.save(lstm_model.state_dict(), pathLSTM)
			
			counter += 1
wb.save("Rezultati_RNN_LSTM_GRU.xls")