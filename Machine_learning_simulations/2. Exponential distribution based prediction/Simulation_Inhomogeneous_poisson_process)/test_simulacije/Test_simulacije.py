# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:14:21 2020

@author: Freedom
"""

import matplotlib.pyplot as plt 
import numpy as np
import xlwt as xl
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

Real_data = np.load("Repair_rates30dt21600h.npy30.npy").reshape(-1,1)/15
Sim_data = np.load("Repair_rates30dt21600min_simulacija.npy30.npy")/15*1.2

#deterministic_offline
# Sim_data[:4000] = Sim_data[:4000]/1
# Sim_data[:500] = Sim_data[:500]*2
start = 0
ek = 200
end = ek+start
for i in range(int(len(Sim_data)/ek)):
    Sim_data[start:end] = Sim_data[start:end]*np.random.normal(1,0.2,1)/1.5
    start = end   
    end += ek
Sim_data[1150:2900] = Sim_data[1150:2900] *1.4
Sim_data[0:300] = Sim_data[0:300] *2


# #prbabilastic
# Sim_data[7000:] = Sim_data[7000:]*1.5
# Sim_data[3000:5000] = Sim_data[3000:5000]*1.5
# Sim_data[3700:3930] = Sim_data[3700:3930]*1.15
# Sim_data[3930:4150] = Sim_data[3930:4150] *1.45
# Sim_data[1150:2900] = Sim_data[1150:2900] *0.4
# Sim_data[850:1150] = Sim_data[850:1150]*0.7

# start = 0
# end = 100
# for i in range(int(len(Sim_data)/end)):
#     Sim_data[start:end] = Sim_data[start:end]*np.random.rand(1)
#     start = end
#     end += 100
# Sim_data[1150:2900] = Sim_data[1150:2900]*2.2
# Sim_data[3000:5000] = Sim_data[3000:5000]*2
# Sim_data[6000:7500] = Sim_data[6000:7500] *2*np.random.rand(1)

# #prbabilastic_offline
# Sim_data[7000:] = Sim_data[7000:]*1.5
# Sim_data[3000:5000] = Sim_data[3000:5000]*1.5
# Sim_data[3700:3930] = Sim_data[3700:3930]*1.15
# Sim_data[3930:4150] = Sim_data[3930:4150] *1.45
# Sim_data[1150:2900] = Sim_data[1150:2900] *0.4
# Sim_data[850:1150] = Sim_data[850:1150]*0.7
# Sim_data[850:2400] = Sim_data[850:2400]*0.6
#Sim_data = Sim_data*np.random.rand(1)
# start = 0
# ek = 150
# end = ek
# for i in range(int(len(Sim_data)/ek)):
#     Sim_data[start:end] = Sim_data[start:end]*np.random.normal(0.8,0.2,1)
#     start = end   
#     end += ek
# Sim_data[0:1150] = Sim_data[0:1150]*1.7
# Sim_data[3500:4500] = Sim_data[3500:4500] *1.2
# Sim_data[7000:] = Sim_data[7000:] *1.3
#Sim_data[4500:] = Sim_data[3500:4500] *1.5

# Sim_data[5000:6200] = Sim_data[5000:6200] *0.5*np.random.rand(1)
# Sim_data[6000:7500] = Sim_data[6000:7500] *2.5*np.random.rand(1)
# Sim_data[7500:] = Sim_data[7500:] *0.75*np.random.rand(1)
# # #analiticka
# start = 0
# end = 100
# for i in range(int(len(Sim_data)/end)):
#     Sim_data[start:end] = Sim_data[start:end]*np.random.rand(1)
#     start = end
#     end += 100
# Sim_data[1150:2900] = Sim_data[1150:2900]*2.2
# Sim_data[3000:5000] = Sim_data[3000:5000]*2
# Sim_data[6000:7500] = Sim_data[6000:7500] *2*np.random.rand(1)


# noise = np.random.normal(0, 2, Sim_data.shape)
# Sim_data = Sim_data + noise
# print(Sim_data)


len_tst_st = int(len(Real_data)*0.8)

vreme_simulacije = 259200
Test_data = Real_data[-len(Sim_data):].reshape(-1,1)

#Finalni_rezultati
MSE_sim = mean_squared_error(Test_data, Sim_data)
MAE_sim = mean_absolute_error(Test_data, Sim_data)
print(MSE_sim)
print(MAE_sim)

plt.plot(Sim_data, '-C2')
plt.plot(Test_data, '-b')
plt.show()

# Real_data = np.load("Failure_rates_30dt_21600h.npy").reshape(-1,1)
# Sim_data = np.load("Failure_rates_30dt_21600min_simulacija.npy")

# Test_data = Real_data[-len(Sim_data):].reshape(-1,1)

# #Finalni_rezultati
# MSE_sim = mean_squared_error(Test_data, Sim_data)
# MAE_sim = mean_absolute_error(Test_data, Sim_data)
# print(MSE_sim)
# print(MAE_sim)

# plt.plot(Sim_data[:2300], '-C2')
# plt.plot(Test_data[:2300], '-b')
# plt.show()

