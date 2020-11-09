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
from sklearn import preprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch.utils.data import TensorDataset, DataLoader

x = np.load('graph_Lambda_mi/x.npy')
y = np.load('graph_Lambda_mi/y.npy')

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


def train(train_loader, learn_rate, hidden_dim, number_of_layers, batch_size, EPOCHS=30, model_type="LSTM"):
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
    outputs = []
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
            if epoch == 30:
                outputs.append(out.cpu().detach().numpy())
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
    return model, LOSS, outputs


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
def ploting(lambd, mi): 
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(lambd, 'r-')
    plt.ylabel('oktz/intervalu')
    plt.xlabel('vreme') 
    plt.title('graph_name')
    plt.subplot(2, 1, 2)
    plt.plot(mi, 'b-')
    plt.ylabel('popravki/intervalu')
    plt.xlabel('vreme') 
    plt.show()
    
counter = 1 
for seq_len in seq_length:
	for hid_dim in hidden_dim:
		for num_layers in number_of_layers:
			
			train_size = int(len(y)*0.7)
			test_size = len(y) - train_size
			
			dataX = np.array(x)
			dataY = np.array(y)
			trainX = np.array(x[0:train_size])
			
			trainY = np.array(y[0:train_size])
			
			testX = np.array(x[train_size:])
			testY = np.array(y[train_size:])			
			#Data loader
			batch_size = 512		
			train_data = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
			train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
			test_data = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))
			test_loader = DataLoader(test_data, shuffle=False, batch_size=testY.shape[0], drop_last=True)
			lr = 0.01
						
			#Training and Validating LSTM_model
			lstm_model, lstm_training_loss, outputs = train(train_loader, lr, hid_dim,num_layers, batch_size, model_type="LSTM")
			lstm_outputs, targets, lstm_val_loss = evaluate(lstm_model, test_loader)
			PATH = "Lambda_Mi_model.pt"
			torch.save(lstm_model, PATH)
            
			a = np.array(lstm_outputs).reshape(-1)   
			b = np.array(targets).reshape(-1)   
			ploting(a,b)