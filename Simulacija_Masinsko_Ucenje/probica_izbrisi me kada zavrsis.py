# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:54:16 2020

@author: Freedom
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn


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
        hidden = (weight.new(self.n_layers, 1, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, 1, self.hidden_dim).zero_())
        return hidden

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
PATH = "modelLSTM/entire_model2.pt"
model = torch.load(PATH)
model.eval()
t = np.load('test_X.npy' )

def pred(model, test_x):
    model.eval()
    inp = torch.from_numpy(np.expand_dims(test_x, axis=0))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)
    return np.squeeze(out.detach().numpy())

prediction = pred(model, t)
