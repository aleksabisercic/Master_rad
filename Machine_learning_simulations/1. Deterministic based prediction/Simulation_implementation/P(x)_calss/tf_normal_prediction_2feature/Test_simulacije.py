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

Real_data = np.load("Repair_rates60dt21600h.npy60.npy").reshape(-1,1)
Sim_data = np.load("Repair_rates60dt21600min_simulacija.npy60.npy")

Test_data = Real_data[-len(Sim_data):].reshape(-1,1)

#Finalni_rezultati
MSE_sim = mean_squared_error(Test_data, Sim_data)
MAE_sim = mean_absolute_error(Test_data, Sim_data)
print(MSE_sim)
print(MAE_sim)

plt.plot(Sim_data, '-C2')
plt.plot(Test_data, '-b')
plt.show()


Real_data = np.load("Failure_rates_30dt_21600h.npy").reshape(-1,1)
Sim_data = np.load("Failure_rates_30dt_21600min_simulacija.npy")

Test_data = Real_data[-len(Sim_data):].reshape(-1,1)

#Finalni_rezultati
MSE_sim = mean_squared_error(Test_data, Sim_data)
MAE_sim = mean_absolute_error(Test_data, Sim_data)
print(MSE_sim)
print(MAE_sim)

plt.plot(Sim_data, '-C2')
plt.plot(Test_data, '-b')
plt.show()


