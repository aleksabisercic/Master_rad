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
from sklearn.metrics import mean_squared_error

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

def predict(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    h = model.init_hidden(1)
    for test_x, test_y in test_loader:
        out, h = model(test_x.to(device).float(), h)
        outputs.append(abs(out.cpu().detach().numpy()))
        targets.append(test_y.numpy())
    outc = np.array(outputs).reshape(-1)
    tar = np.array(targets).reshape(-1)
    Loss = mean_squared_error(tar, outc)	
    print("Validation loss: {}".format(Loss))
    return outputs

is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
PATH = 'μ(t)(Repair rate)_LSTM_GRU_RNN.pt'
Path = "λ(t)(failure rate)_LSTM_GRU_RNN.pt"
model_mi = torch.load(PATH)
model_lambda = torch.load(Path)

#Numpy λ(t) and μ(t) loading and generating datasets
def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range(len(datax) - seq_length - 1):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)			
    return np.array(x), np.array(y)

#μ(t) Repair_rates
mi = np.load('Numpy λ(t) and μ(t)/Repair_rates_for_NN.npy') #Repair_rates dt(step) = 30min (step),  window = 21600min

dataY = np.array(mi).reshape(-1, 1)
dataX = np.array(mi).reshape(-1, 1)

x_mi, y_mi = sliding_windows(dataX, dataY, 50)	
train_size = int(len(y_mi) * 0.8)
test_size = len(y_mi) - train_size
testX_mi = np.array(x_mi[train_size:])
testY_mi = np.array(y_mi[train_size:])

#Data loader
test_data_mi = TensorDataset(torch.from_numpy(testX_mi), torch.from_numpy(testY_mi))
test_loader_mi = DataLoader(test_data_mi, shuffle=False, batch_size=1, drop_last=True)

mi_ls = predict(model_mi, test_loader_mi) #μ(t) Repair_rates predictions list
mi_ls = np.array(mi_ls).reshape(-1)

#λ(t) Failure_rates 
lamb = np.load('Numpy λ(t) and μ(t)/Failure_rates_for_NN.npy')
dataY = np.array(lamb).reshape(-1, 1)
dataX = np.array(lamb).reshape(-1, 1)

x, y = sliding_windows(dataX, dataY, 50)	
train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
testX = np.array(x[train_size:])
testY = np.array(y[train_size:])

#Data loader
test_data_lamb = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))
test_loader_lamb = DataLoader(test_data_lamb, shuffle=False, batch_size=1, drop_last=True)

lam_ls = predict(model_lambda, test_loader_lamb) #λ(t) Failure_rates predictions list
lam_ls = np.array(lam_ls).reshape(-1)


run_time = int(len(lam_ls)*30) # len in minutes of test seta (step is 30min)
max_tim = int(len(lam_ls)*30) 
print('max_time {}min'.format(max_tim))

def Lambda_interp(t,vreme,Lambda):
    """
    Interp Failure rates and Repair rates
    """
    return np.interp(t,vreme,Lambda)

def generate_times_opt(max_t, delta, vreme, Lambda):
    time = np.arange(delta,max_t, delta)       
    lamb_val_t = Lambda_interp(time,vreme,Lambda)     
    tf = TimeFunction((time, lamb_val_t), dt=delta)
    print('Funion_time {},\n lamb_val_t {},'.format(time.shape, lamb_val_t.shape))
    Psim = SimuInhomogeneousPoisson([tf], end_time = max_t, verbose = False)
    Psim.simulate()
    simulacija = Psim.timestamps 
    return simulacija[0]

lam_ls = np.array(lam_ls).reshape(-1)
lam_ls = lam_ls/(60*24*15) #get rates in minutes 

mi_ls = np.array(mi_ls).reshape(-1)
mi_ls = mi_ls/(60*24*15)   #get rates in minutes 

delta = 30  #dt (step)

otkazi = [] #prvi clan iz simulacije, a sledeci je mi[-1] + Prvi izlaz iz simulacije
popravke = []

try:
    while True:
        vreme_lam =  np.linspace(0, run_time, lam_ls.shape[0]) # instance za koje imamo lambda 
        time = np.arange(delta,run_time, delta)   
        
        #kako se generisu otkazi posle
        lamb_val_t = Lambda_interp(time,vreme_lam,lam_ls) 
        tf = TimeFunction((time, lamb_val_t), dt=delta)
        Psim = SimuInhomogeneousPoisson([tf], end_time = run_time, verbose = False)
        Psim.simulate()
        simulacija_lambd = Psim.timestamps 
        array_lam = np.array(simulacija_lambd)
        array_lam = array_lam.reshape(-1)
 #       print('Funion_time {},\n lamb_val_t {},'.format(time.shape, lamb_val_t.shape))
        update_array_mi = int(array_lam[0]/30) # proveri, da li delim sa dt=30
        mi_ls = mi_ls[update_array_mi:]
        run_time = run_time - array_lam[0]
   #     print('Funion_time_update {},\n mi_val_t_update {},'.format(time.shape, lamb_val_t.shape))
        if not popravke:
            otkazi.append(int(array_lam[0]))
        else:
            otkazi.append(int(array_lam[0] + popravke[-1]))

        vreme_mi =  np.linspace(0, run_time, mi_ls.shape[0]) # instance za koje imamo lambda
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
        
        run_time = max_tim - popravke[-1]
except: 
   print(otkazi)
   print(popravke)

vreme_simulacije = 259200 # duzina test seta ( 6 meseci )
vremena_otkaza = np.array(otkazi)
vremena_popravke = np.array(popravke)
podatci1 = vremena_otkaza.reshape(-1)
podatci2 = vremena_popravke.reshape(-1)

def gen_lambda_and_mi(podatci1,podatci2, seq_len, t):
    matrix = np.zeros(vremena_popravke[-1])
    for i in podatci1:
        matrix[int(i)] = 1        
    matrix1 = np.zeros(vremena_popravke[-1] + 1)
    for i in podatci2:
        matrix1[int(i)] = 1  
    lambd = []
    mi = []    
    start = 0
    end = seq_len
    for i in range(int((len(matrix)-seq_len)/t)):
        ls = matrix[start:end]
        lambd.append(sum(ls))
        ls1 = matrix1[start:end]
        mi.append(sum(ls1))
        start += t
        end += t
    lambd.append(sum(matrix[-seq_len:]))
    mi.append(sum(matrix1[-seq_len:]))
    return lambd, mi
        
seq_leng = [15*24*60]
dt = [8*60]

for seq_len in seq_leng:
    for t in dt:
        lamb, mi =  gen_lambda_and_mi(podatci1,podatci2, int(seq_len), t)
        mi_gen_simulacija = np.array(mi).reshape(-1, 1)
        lamb_gen_simulacija = np.array(lamb).reshape(-1, 1)
        sim_name_lam = 'Failure_rates_' + str(t) + 'dt_' + str(seq_len) + 'min_simulacija' + '.npy'
        sim_name_mi = 'Repair_rates' + str(t) + 'dt' + str(seq_len) + 'min_simulacija' + '.npy'
        np.save(sim_name_lam, lamb_gen_simulacija)
        np.save(sim_name_mi +str(t), mi_gen_simulacija)




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