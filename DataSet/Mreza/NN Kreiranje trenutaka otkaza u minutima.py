# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:52:16 2020

@author: Freedom
"""

#Kreiranje trenutaka otkaza


import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

df = pd.read_excel("Zastoji.xlsx", index_col = 0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df.sort_values(by = ['Početak zastoja'])

df1 = df[['Pocetak_zastoja_u_minutima', 'Vreme_zastoja']]

n = int(len(df1['Pocetak_zastoja_u_minutima']))
i=0

while(i<n):
    if df1['Vreme_zastoja'].iloc[i] > 2000 :
        df1.drop(df.index[i],inplace=True)
        n = n - 1
    i = i + 1
		
df1.reset_index(inplace = True, drop = True)

lista = []

for i in range (0,len(df1.index)): #df['Vreme_zastoja']:
	lista.append(df1["Pocetak_zastoja_u_minutima"].iloc[i])

	
Podatci = np.array(lista)

a = Podatci
n = 20
size_i = int(len(a)/2)
size_j = n+1

mat = np.zeros((size_i,size_j))
br = 0
mat[0][size_j-1] = a[0]
for i in range(1,size_i):
    mat[i] = mat[i-1]
    mat[i] = np.roll(mat[i],-1)
    mat[i][size_j - 1] = a[br+1]
    br = br + 1


X_list = 0
X = []
for i in range (0, size_i): 
    X_list = mat[i][:(n)]
    X.append(X_list)
	
Y = np.array(lista)	    
X = np.array(X) 

startx= int(len(X)*0.8)
starty= int(len(Y)*0.8)

X2 = X[startx:]
X1 = X[:startx]
Y2 = Y[starty:]
Y1 = Y[:starty]
Y2 = Y2.reshape(int(len(Y2)),1 )
	
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class NNtDataset(Dataset):
    def __init__(self, X1, Y1):
        self.x = torch.from_numpy(X1).type(torch.FloatTensor)
        self.y = torch.from_numpy(Y1).type(torch.FloatTensor)
        self.y = self.y.view(-1,1)
		     	   
    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
  
class NNvDataset(Dataset):
    def __init__(self, X2, Y2):
        self.x = torch.from_numpy(X2).type(torch.FloatTensor)
        self.y = torch.from_numpy(Y2).type(torch.FloatTensor)
        self.y = self.y.view(-1,1)
		     	   
    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

 
train_data = NNtDataset( X1, Y1 )
test_data = NNvDataset( X2, Y2 )


# dataloaders
train_loader = DataLoader(dataset = train_data, batch_size = 64,  shuffle=False) #da li sam dobro razumeo batch size
validation_loader = DataLoader(dataset = test_data, shuffle=False)

#ploting 
def plot_accuracy_loss(training_results): 
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')   
    plt.show()
	
class Net(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
    
    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

def accuracy(y1, yhat1):
    y_pred = yhat1.detach().numpy() 
    y_true = y1.detach().numpy()
    return (mean_squared_error(y_true, y_pred))

def train( model, criterion, train_loader,validation_loader, optimizer, data_set1, epochs=100):
    model.train()
    LOSS = []
#    ACC = []
    useful_stuff = {'training_loss': [],'validation_accuracy': []}  
    for epoch in range(epochs):       
        for x, y in train_loader: 
            yhat = model(x.view(-1, n * 1))
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            if epoch%5 == 0:
                print(loss)			
            useful_stuff['training_loss'].append(loss.item())
       # ACC.append(accuracy(model, data_set))
  
        for x,y in validation_loader:
            yhat1 = model(x)
            y1 = y
            useful_stuff['validation_accuracy'].append(accuracy(y1, yhat1))

    return useful_stuff  
   
data_set = train_data
data_set1 = test_data
Layers = [n, 10, 10, 1] 
model = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam
criterion = nn.MSELoss() #nadji mean square error	
training_results = train(model, criterion, train_loader, validation_loader, optimizer,data_set1, epochs=100)
plot_accuracy_loss(training_results)

