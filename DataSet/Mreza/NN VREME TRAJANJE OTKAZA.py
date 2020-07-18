# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:38:07 2020

@author: Freedom
"""
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
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Vreme_zastoja', 'Vreme_rada']]

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
	lista.append(df["Vreme_zastoja"].iloc[i])
	lista1.append(df["Vreme_zastoja"].iloc[i])
	lista.append(df["Vreme_rada"].iloc[i])
	
Podatci = np.array(lista)
Podatciy = np.array(lista1)

a = Podatci
n = 100
size_i = int(len(a)/2)
size_j = 2*n+2

mat = np.zeros((size_i,size_j))
br = 2
mat[0][size_j-2] = a[0]
mat[0][size_j-1] = a[1]
for i in range(1,size_i):
    mat[i] = mat[i-1]
    mat[i] = np.roll(mat[i],-2)
    mat[i][size_j - 2] = a[br]
    mat[i][size_j - 1] = a[br+1]
    br = br + 2


X_list = 0
X = []
for i in range (0, size_i): 
    X_list = mat[i][:(2*n)]
    X.append(X_list)
	
Y = np.array(lista1)	    
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
train_loader = DataLoader(dataset = train_data, batch_size = 512,  shuffle=False) #da li sam dobro razumeo batch size
validation_loader = DataLoader(dataset = test_data, batch_size = 512, shuffle=False)

#ploting 
def plot_accuracy_loss(training_results): 
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r-')
    plt.ylabel('loss')
    plt.xlabel('epochs') 
    plt.title('training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'], 'b-')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')   
    plt.show()
	
class Net(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.Batchn = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
            self.Batchn.append(nn.BatchNorm1d(input_size))
			
    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        k = 0
        for (l, linear_transform) in zip(range(L), self.hidden ):
            if l < L - 1:
                activation = self.Batchn[k](F.relu(linear_transform(activation)))
                k = k + 1
                #activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

def accuracy(y1, yhat1):
    y_pred = yhat1.detach().numpy() 
    y_true = y1.detach().numpy()
    return (mean_squared_error(y_true, y_pred))

def train( model, criterion, train_loader,validation_loader, optimizer, data_set1, epochs=500):
 #   LOSS = []
#    ACC = []
    useful_stuff = {'training_loss': [],'validation_accuracy': []}  
    for epoch in range(epochs):       
        for x, y in train_loader: 
            model.train()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 #           LOSS.append(loss.item())
#            if epoch%5 == 0:
#                print(loss)
            useful_stuff['training_loss'].append(loss.item())
            
       # ACC.append(accuracy(model, data_set))
  
        for x,y in validation_loader:
            model.eval()
            yhat1 = model(x)
            loss1 = criterion(yhat1, y)
            print(loss1)
            useful_stuff['validation_accuracy'].append(loss1.item())
#            ACC.append(loss1.item())

    return useful_stuff  
#dropout   
data_set = train_data
data_set1 = test_data
Layers = [2*n, 15, 10, 15, 5, 1] #15_15_15_10_500ep_n400_lr0_005_bs512
model = Net(Layers)
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam
criterion = nn.MSELoss() #nadji mean square error	
training_results = train(model, criterion, train_loader, validation_loader, optimizer,data_set1, epochs=500)
plot_accuracy_loss(training_results)

#ako ostatak posle deljenja sa epoch%5 = 0 print(loss) e