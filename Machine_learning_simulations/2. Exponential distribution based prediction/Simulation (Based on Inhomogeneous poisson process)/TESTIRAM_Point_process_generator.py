# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 02:47:57 2020

@author: Freedom
"""

import numpy as np
from tick.base import TimeFunction
import torch
import torch.nn as nn
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson
from torch.utils.data import TensorDataset, DataLoader

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
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#PATH = "Mi_model.pt"
PATH = 'modelLSTM/probaj_ovaj_za_mi.pt'
Path = "Lambda_Mi_model.pt"
model = torch.load(PATH)
model_lambda = torch.load(Path)
popravke_prethodni = np.load('test_podatci_za_predvidjanje_mi.npy' )
popravke_prethodni = popravke_prethodni[0]
otkazi_prethodni = np.load('aleksa_lambda_proba.npy' )
Y = np.load('testY_podatci_za_predvidjanje_mi.npy')
#test_data = TensorDataset(torch.from_numpy(popravke_prethodni), torch.from_numpy(Y))
#test_loader = DataLoader(test_data, shuffle=False, batch_size=Y.shape[0], drop_last=True)

#otkazi_prethodni = otkazi_prethodni[0]
mi_ls = []
lam_ls = []
run_time =  10000
max_tim = run_time
def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    loss = []
    h = model.init_hidden(Y.shape[0])
    for test_x, test_y in test_loader:
        out, h = model(test_x.to(device).float(), h)
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.numpy())
    for i in range(len(test_x)):
        MSEloss = (outputs[0][i]-targets[0][i])**2
        loss.append(MSEloss)
    Loss = sum(loss)/len(loss)	
    print("Validation loss: {}".format(Loss))
    return outputs, targets, Loss
def pred(model, test_x):
    with torch.no_grad():
        model.eval()
        inp = torch.from_numpy(np.expand_dims(test_x, axis=0))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
    return np.squeeze(out.detach().numpy())

def Lambda_interp(t,vreme,Lambda):
    lambda_val = np.interp(t,vreme,Lambda)
    return lambda_val

def generate_times_opt(max_t, delta, vreme, Lambda):
    time = np.arange(delta,max_t, delta)
   
    lamb_val_t = Lambda_interp(time,vreme,Lambda) 
    
    tf = TimeFunction((time, lamb_val_t), dt=delta)
    Psim = SimuInhomogeneousPoisson([tf], end_time = max_t, verbose = False)
    Psim.simulate()
    simulacija = Psim.timestamps 
    return simulacija[0]

''' Generisemo lambda podatke '''
for i in range(int(run_time/60)): 
    lambda_dt = pred(model, popravke_prethodni)
    mi_ls.append(int(lambda_dt))
    popravke_prethodni = np.roll(popravke_prethodni, -1)
    popravke_prethodni = popravke_prethodni.reshape(-1,1)
    popravke_prethodni[-1] = lambda_dt
 
for i in range(int(run_time/60)): 
    lambda_dt = pred(model_lambda, otkazi_prethodni)
    lam_ls.append(int(lambda_dt))
    otkazi_prethodni = np.roll(otkazi_prethodni, -1)
    otkazi_prethodni = otkazi_prethodni.reshape(-1,1)
    otkazi_prethodni[-1] = lambda_dt

lam_ls = np.array(lam_ls).reshape(-1)
lam_ls = lam_ls/15/24/60  

mi_ls = np.array(mi_ls).reshape(-1)
mi_ls = mi_ls/15/24/60     

delta = 30                                         #dt na koj gledamo u simulaciji

otkazi = [] #prvi clan iz simulacije, a sledeci je mi[-1] + Prvi izlaz iz simulacije
popravke = []

try:
    while True:
        vreme_lam =  np.linspace(0, run_time, lam_ls.shape[0]) # instance za koje imamo lambda 
        vreme_mi =  np.linspace(0, run_time, mi_ls.shape[0]) # instance za koje imamo lambda
        time = np.arange(delta,run_time, delta)   
        
        #kako se generisu otkazi posle
        lamb_val_t = Lambda_interp(time,vreme_lam,lam_ls) 
        tf = TimeFunction((time, lamb_val_t), dt=delta)
        Psim = SimuInhomogeneousPoisson([tf], end_time = run_time, verbose = False)
        Psim.simulate()
        simulacija_lambd = Psim.timestamps 
        array_lam = np.array(simulacija_lambd)
        array_lam = array_lam.reshape(-1)
        if not popravke:
            otkazi.append(int(array_lam[0]))
        else:
            otkazi.append(int(array_lam[0] + popravke[-1]))
            
        mi_val_t = Lambda_interp(time,vreme_mi,mi_ls) 
        tf = TimeFunction((time, mi_val_t), dt=delta)
        Psim = SimuInhomogeneousPoisson([tf], end_time = run_time, verbose = False)
        Psim.simulate()
        simulacija_mi = Psim.timestamps 
        array_mi = np.array(simulacija_lambd)
        array_mi = array_lam.reshape(-1)
        popravke.append(int(array_mi[0] + otkazi[-1]))
        
        update_array = int(array_mi[0]/30)
        lam_ls = lam_ls[update_array:]
        mi_ls = mi_ls[update_array:]
        
        run_time = max_tim - popravke[-1]
except: 
    print(otkazi)
    print(popravke)

# nova_lista = []
# simulacija_lambd = np.array(simulacija_lambd).reshape(-1)
# simulacija_mi = np.array(simulacija_mi).reshape(-1)
# for i in range(0, len(simulacija_lambd)):
#     m = 0
#     while m < len(simulacija_mi):
#         if simulacija_lambd[i] < simulacija_mi[m]:
#             nova_lista.append(simulacija_lambd[i])
#             nova_lista.append(simulacija_mi[m])
#             m = len(simulacija_mi) + 1
#         else:
#             m += 1
            
# konacno = np.array(nova_lista).reshape(-1,2)
# ls = []
# for i in range(0,len(konacno)):
#     if i == len(konacno) - 1: continue
#     if konacno[i][1] > konacno[i+1][0]:
#         ls.append(i+1)
#     else:
#         continue
# a= 1
# novo_konacno = np.delete(konacno, ls, 0)