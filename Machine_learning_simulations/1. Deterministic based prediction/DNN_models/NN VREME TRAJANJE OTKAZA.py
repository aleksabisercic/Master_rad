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
import xlwt as xl
import pickle
import os
import seaborn as sns

from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
from tqdm import tqdm
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import  optim

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_excel("Dataset/Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Vreme_zastoja', 'Vreme_rada']]


lista = []
lista1 = []

for i in range (len(df.index)):
	lista.append(df["Vreme_zastoja"].iloc[i])
	lista1.append(df["Vreme_zastoja"].iloc[i])
	lista.append(df["Vreme_rada"].iloc[i])
	
	
 
Podatci = np.array(lista)
Podatciy = np.array(lista1)

a = Podatci

def Dataa (n,a):
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
		
	X = X[ n : ]			    
	X = np.array(X) 

		
	startx= int(len(X)*0.8)
	
	X2 = X[startx:]
	X1 = X[:startx]
	
	transformx1 = preprocessing.normalize(X1)
	transformx2 = preprocessing.normalize(X2)
	return(transformx1,transformx2)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class NNtDataset(Dataset):
    def __init__(self, transformx1, y1):
        self.x = torch.from_numpy(transformx1).type(torch.FloatTensor)
        self.y = torch.from_numpy(y1).type(torch.FloatTensor)
		     	   
    def __len__(self):
        return (self.x.shape[0])
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
#ploting 
def plot_accuracy_loss(training_results, graph_name): 
    plt.clf()
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
    plt.savefig(graph_name)
    
class Net(nn.Module):
    
    # Constructor
    def __init__(self, Layers,p=0.9):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)		
        self.hidden = nn.ModuleList()
        self.Batchn = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
#            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu') 
            batchq = nn.BatchNorm1d(output_size)
            self.hidden.append(linear)
            self.Batchn.append(batchq)
    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        k = 0
        for (l, linear_transform) in zip(range(L), self.hidden ):
            if l < L - 1:
                activation = self.Batchn[k](F.leaky_relu(self.drop(linear_transform(activation))))
                k = k + 1
            else:
                activation = linear_transform(activation)
        return activation

def accuracy(y1, yhat1):
    y_pred = yhat1.detach().numpy() 
    y_true = y1.detach().numpy()
    return (mean_squared_error(y_true, y_pred))

def train( model, criterion, train_loader,validation_loader, optimizer, epochs=150):
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
#            LOSS.append(loss.item())
#            if epoch%5 == 0:
#                print(loss)
            useful_stuff['training_loss'].append(loss.item())
  
        for x,y in validation_loader:
            model.eval()
            yhat1 = model(x)
            loss1 = criterion(yhat1, y)
            useful_stuff['validation_accuracy'].append(loss1.item())
        
        if epoch%5 == 0:
                print('Training loss: {}, Validation loss: {}'.format (str(loss), str(loss1)))
           
    return useful_stuff, model

#dropout   
n =  [ 50 ]
a1 = [ 15 ]
a2 = [ 15 ]
a3 = [ 10 ]


wb = xl.Workbook ()
ws1 = wb.add_sheet("Rezultat simulacije")
ws1_kolone = ["Ime simulacije","Poslednji rezultat", "Najmanji rezultat", "Training_Loss" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

counter = 1 
for seq_len in n:
	for i in a1:
		for j in a2:
			for k in a3:
				x1,x2 = Dataa (seq_len,a)
				lista1 = lista1[seq_len:]
				Y = np.array(lista1)
				starty= int(len(Y)*0.8)
				Y2 = (Y[starty:])
				Y1 = (Y[:starty])
				Y2 = Y2.reshape(-1,1)
				Y1 = Y1.reshape(-1,1)
				train_data = NNtDataset( x1, Y1 )
				test_data = NNtDataset( x2, Y2 )                
				#train_data = TensorDataset(torch.from_numpy(x1), torch.from_numpy(Y1).view(1,-1))
				#test_data = TensorDataset(torch.from_numpy(x2), torch.from_numpy(Y2))
				# dataloaders
				train_loader = DataLoader(dataset = train_data, batch_size = 56,  shuffle=False) #da li sam dobro razumeo batch size
				validation_loader = DataLoader(dataset = test_data, batch_size = Y2.shape[0], shuffle=False)
				Layers = [2*seq_len, i,j,k, 1] 
				model = Net(Layers, p=0)
				learning_rate = 0.001
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam
				criterion = nn.MSELoss() #nadji mean square error	
				training_results, nn_model = train(model, criterion, train_loader, validation_loader, optimizer, epochs=300)
				simulation_name = 'NN_Vreme_trajanje_otkaza'
				graph_name = 'DNN model/'+ simulation_name + '.png'
				#ispisi poslednji iz exel tabele
				last_element = training_results['validation_accuracy'][-1]
				min_element = min(training_results['validation_accuracy'])
				last_training_loss = sum(training_results['training_loss'][-seq_len:])/len(training_results['training_loss'][-seq_len:])
				ws1.row(counter).write(0, simulation_name)
				ws1.row(counter).write(1, last_element)
				ws1.row(counter).write(2, min_element)
				ws1.row(counter).write(3, last_training_loss)
				#pikl u poseban folder (kao objekat)
				PathNN = 'DNN model/'+ simulation_name + '.pt'
				torch.save(nn_model, PathNN)
#				plot_accuracy_loss(training_results,graph_name )
				counter += 1
wb.save("Excel tabels (results)\Rezultati_DNN_trajanje_otkaza_torch.xls")			 


