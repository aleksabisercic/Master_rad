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

from tensorflow import keras
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

''' Loading  λ(t)(failure rate) generated from folder path '''
series = np.load('Failure_rates_for_NN.npy')

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []

    for i in range(len(datax) - seq_length - 1):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)			

    return np.array(x), np.array(y)

lambd = series
dataY = np.array(lambd).reshape(-1, 1)
dataX = np.array(lambd).reshape(-1, 1)

x, y = sliding_windows(dataX, dataY, 50)

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
        hidden = weight.new(self.n_layers, 1, self.hidden_dim).zero_().to(device)
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
        hidden = (weight.new(self.n_layers, 1, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, 1, self.hidden_dim).zero_().to(device))
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
        hidden = weight.new(self.n_layers, 1, self.hidden_dim).zero_().to(device)
        return hidden


def train(  trainX, trainY, train_loader, learn_rate, hidden_dim, number_of_layers, EPOCHS=1000, model_type="GRU"):
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
        h = model.init_hidden(1)
        avg_loss = 0.
        counter = 0
        for i in range(len(trainY)):  #for i in range seq_len.
            counter += 1
            if model_type == "GRU" or model_type == "RNN":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            train_X = trainX[i]
            train_X = np.expand_dims(train_X, axis=0)
            train_X = torch.from_numpy(train_X)
            train_Y = trainY[i]
            train_Y = np.expand_dims(train_Y, axis=0)
            train_Y = torch.from_numpy(train_Y)
            if i < len(trainY)*3/4: #ovo je ubaceno
                out, h = model(train_X.to(device).float(), h)
            else:
                out, h = model(train_X.to(device).float(), h)
                if i == (len(trainY)-1): continue
                trainX[i+1] = trainX[i] 
                trainX[i+1][-1] = out.cpu().detach().numpy()
            loss = criterion(out, train_Y.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
#            if counter % 200 == 0:
 #               print("Epoch {} Average Loss for Epoch: {}".format(epoch, avg_loss / counter))
        LOSS.append((avg_loss / counter))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(trainY)))
    #    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model, LOSS


def evaluate_future_pred(model,testX, testY):
    model.eval()
    Loss = []
    prediction = []
    criterion = nn.MSELoss()
    h = model.init_hidden(1)
    for i in range(len(testY)):
        test_X = testX[i]
        test_X = np.expand_dims(test_X, axis=0)
        test_X = torch.from_numpy(test_X)
        test_Y = trainY[i]
        test_Y = np.expand_dims(test_Y, axis=0)
        test_Y = torch.from_numpy(test_Y)
        out, h = model(test_X.to(device).float(), h)
        loss = criterion(out, test_Y.to(device).float())
        Loss.append(loss.item())
        if i == (len(testY)-1): continue
        testX[i+1] = testX[i] 
        testX[i+1][-1] = out.cpu().detach().numpy()    
        prediction.append(out.cpu().detach().numpy())
    print("Validation loss: {}".format(sum(Loss)/len(Loss)))
    return Loss, sum(Loss)/len(Loss), prediction

def evaluate(model, test_loader):
    model.eval()
    model.to(device)
    outputs = []
    targets = []
    loss = []
    h = model.init_hidden(1)
    for test_x, test_y in test_loader:
        out, h = model(test_x.to(device).float(), h)
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.numpy())
    outputs = np.array(outputs).reshape(-1,2)
    targets = np.array(targets).reshape(-1,2)
    loss =np.array(keras.losses.MSE(targets,outputs)).reshape(-1,1)
    Loss = sum(loss)/len(loss)
    print("Validation loss: {}".format(Loss))
    return outputs, targets, Loss

seq_length =  [ 50 ]
hidden_dim = [ 50  ]
number_of_layers = [ 2 ]

wb = xl.Workbook ()
ws1 = wb.add_sheet("RNN razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws2 = wb.add_sheet("GRU razultati")
ws2_kolone = ["Ime simulacije", "Training L","Validation Loss"]
ws2.row(0).write(0, ws1_kolone[0])
ws2.row(0).write(1, ws1_kolone[1])
ws2.row(0).write(2, ws1_kolone[2])
ws3 = wb.add_sheet("LSTM ruzultati")
ws3_kolone = ["Ime simulacije", "Training L","Validation Loss"]
ws3.row(0).write(0, ws1_kolone[0])
ws3.row(0).write(1, ws1_kolone[1])
ws3.row(0).write(2, ws1_kolone[2])

counter = 1 
for seq_len in seq_length:
	for hid_dim in hidden_dim:
		for num_layers in number_of_layers:
			
			x, y = sliding_windows(dataX, dataY, seq_len)
			
			train_size = int(len(y) * 0.8)
			test_size = len(y) - train_size
			
			dataX = np.array(x)
			dataY = np.array(y)
			trainX = np.array(x[0:train_size])
#			trainX = torch.from_numpy(trainX)
			
			trainY = np.array(y[0:train_size])
#			trainY = torch.from_numpy(trainY)
			
			testX = np.array(x[train_size:len(x)])
			testY = np.array(y[train_size:len(y)])
			
			#Data loader
			batch_size = 512		
			train_data = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
			train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
			test_data = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))
			test_loader = DataLoader(test_data, shuffle=False, batch_size=1, drop_last=True)
			lr = 0.0001

#			rnn_model, rnn_training_loss = train(trainX, trainY, train_loader, lr, hid_dim,num_layers, model_type="RNN")
#			rnn_outputs, targets, rnn_test_loss = evaluate(rnn_model, test_loader)
            
			#Training and Validating GRU_model
#			gru_model, gru_training_loss = train(trainX, trainY, train_loader, lr, hid_dim,num_layers, model_type="GRU")
#			gru_outputs, targets, gru_test_loss = evaluate(gru_model, test_loader)
#			PATH_gru = "modelRNN/2features predict 2 outputs GRU.pt"
            
			#Training and Validating LSTM_model
			lstm_model, lstm_training_loss = train( trainX, trainY, train_loader, lr, hid_dim,num_layers, model_type="LSTM")
			lstm_outputs, lstm_targets, lstm_test_loss = evaluate(lstm_model, test_loader)
            
			simulation_name = '2features predict 2(update after each x)' 
			pathRNN = 'modelRNN/'+ simulation_name + '.pt'
			pathGRU = 'modelGRU/'+ simulation_name + '.pt'
			pathLSTM = 'modelLSTM/'+ simulation_name + '.pt'
            
			'''Ime simulacije","Validation Loss", "Training L'''
			#RNN
#			ws1.row(counter).write(0, simulation_name + "_" +'RNN')
#			ws1.row(counter).write(1, rnn_training_loss[-1])
#			ws1.row(counter).write(2, int(rnn_test_loss[-1]))

			#save model parametre RNN
#			torch.save(rnn_model, pathRNN)
			
						#GRU
#			ws2.row(counter).write(0, simulation_name + "_" +'GRU')
#			ws2.row(counter).write(1, gru_training_loss[-1])
#			ws2.row(counter).write(2, int(gru_test_loss[-1]))

			#save model parametre GRU
#			torch.save(gru_model, pathGRU)
			
						#LSTM
			ws3.row(counter).write(0, simulation_name + "_" +'LSTM')
			ws3.row(counter).write(1, lstm_training_loss[-1])
			ws3.row(counter).write(2, int(lstm_test_loss[-1]))

			#save model parametre LSTM
			torch.save(lstm_model, pathLSTM)
			
wb.save( 'update_1_by_1' + ".xls")
