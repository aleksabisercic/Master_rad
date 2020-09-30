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

from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing

df = pd.read_excel("Zastoji.xlsx", index_col = 0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by = ['Početak zastoja'])

df = df[['Vreme_zastoja', 'Vreme_rada']]
df.reset_index(inplace = True, drop = True)


df = df[df.Vreme_zastoja < 2000]
df = df[df.Vreme_rada < 2000]

lista = []
lista1 = []

for i in range (0,len(df.index)): #df['Vreme_zastoja']:
	lista.append(df["Vreme_zastoja"].iloc[i])
	lista1.append(df["Vreme_rada"].iloc[i])
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
	X = np.array(X) 
	startx= int(len(X)*0.8)
	X2 = X[startx:]
	X1 = X[:startx]
	transformx1 = preprocessing.normalize(X1)
	transformx2 = preprocessing.normalize(X2)
	return(transformx1,transformx2)


Y = np.array(lista1)
starty= int(len(Y)*0.8)
Y2 = Y[starty:]
Y1 = Y[:starty]



def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class NNtDataset(Dataset):
    def __init__(self, transformx1, transformy1):
        self.x = torch.from_numpy(transformx1).type(torch.FloatTensor)
        self.y = torch.from_numpy(transformy1).type(torch.FloatTensor)
        self.y = self.y.view(-1,1)
		     	   
    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
  
class NNvDataset(Dataset):
    def __init__(self, transformx2, transformy2):
        self.x = torch.from_numpy(transformx2).type(torch.FloatTensor)
        self.y = torch.from_numpy(transformy2).type(torch.FloatTensor)
        self.y = self.y.view(-1,1)
		     	   
    def __len__(self):
        return (len(self.x))
    
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
    def __init__(self, Layers,p=0.6):
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
                activation = self.Batchn[k](F.relu(self.drop(linear_transform(activation))))
                k = k + 1
            else:
                activation = linear_transform(activation)
        return activation

def accuracy(y1, yhat1):
    y_pred = yhat1.detach().numpy() 
    y_true = y1.detach().numpy()
    return (mean_squared_error(y_true, y_pred))

def train( model, criterion, train_loader,validation_loader, optimizer, epochs=500):
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
            if epoch%1 == 0:
                print(loss1)			
            useful_stuff['validation_accuracy'].append(loss1.item())

    return useful_stuff  

#dropout   
n =  [ 100, 200, 300, 50 ]
a1 = [ 100, 50, 15  ]
a2 = [ 100, 15 ]
a3 = [ 15 ]

wb = xl.Workbook ()
ws1 = wb.add_sheet("Rezultat simulacije")
ws1_kolone = ["Ime simulacije","Poslednji rezultat", "Najmanji rezultat", "Srednja vrednost rezultat" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws1.row(0).write(3, ws1_kolone[3])

counter = 1 
for l in n:
	for i in a1:
		for j in a2:
			for k in a3:
				x1,x2 = Dataa (l,a)
				train_data = NNtDataset( x1, Y1 )
				test_data = NNtDataset( x2, Y2 )
				# dataloaders
				train_loader = DataLoader(dataset = train_data, batch_size = 512,  shuffle=False) 
				validation_loader = DataLoader(dataset = test_data, batch_size = Y2.shape[0], shuffle=False)
				Layers = [2*l, i,j,k, 1] 
				model = Net(Layers, p=0)
				learning_rate = 0.005
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #adam
				criterion = nn.MSELoss() #nadji mean square error	
				training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=250)
				simulation_name = str(i) +'_'+str(j)+'_'+str(k)+'_'+str(l)+  '_' +'t_rada'
				graph_name = 'graph/'+ simulation_name + '.png'
				pickle_name = 'pikl/'+ simulation_name + '.obj'
				#ispisi poslednji iz exel tabele
				last_element = training_results['validation_accuracy'][-1]
				min_element = min(training_results['validation_accuracy'])
				mean_element = sum(training_results['validation_accuracy'])/len((training_results['validation_accuracy']))
				ws1.row(counter).write(0, simulation_name)
				ws1.row(counter).write(1, last_element)
				ws1.row(counter).write(2, min_element)
				ws1.row(counter).write(3, mean_element)
				#pikl u poseban folder (kao objekat)
				pickle.dump(model, open(pickle_name , 'wb'))
				plot_accuracy_loss(training_results,graph_name )
				counter += 1
wb.save("Rezultati_NN1.xls")			 
