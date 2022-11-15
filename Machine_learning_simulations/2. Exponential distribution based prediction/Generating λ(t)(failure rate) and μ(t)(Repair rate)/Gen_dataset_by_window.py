# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:55:33 2020

@author: Freedom
"""

import numpy as np 
import matplotlib.pyplot as plt
import time
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.special import kl_div

def podeli(x,y):
        try:
            return x/y
        except ZeroDivisionError:
            return 0

df = pd.read_excel("Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by=['Početak zastoja'])

df = df[['Pocetak_zastoja_u_minutima', 'Vreme_zastoja', 'Vreme_rada', 'Kraj_zastoja_u_miutama', 'Vrsta_zastoja']]

k = int(len(df['Vreme_zastoja']))
i = 0

df.reset_index(inplace=True, drop=True)

lista2 = []
lista1 = []
lista3 = []
for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	lista1.append(df["Pocetak_zastoja_u_minutima"].iloc[i])
	lista2.append(df["Kraj_zastoja_u_miutama"].iloc[i])
	lista3.append(df["Vrsta_zastoja"].iloc[i])

podatci3 = []
for label in lista3:
    if label == 'Masinski':
        podatci3.append(0)
    elif label == 'Elektro':
        podatci3.append(1)
    elif label == 'Ostalo':
        podatci3.append(2)

podatci1 = np.array(lista1)
podatci1 = podatci1[int(len(podatci1)*0.751):]
podatci1 = podatci1 - podatci1[0]

podatci2 = np.array(lista2)
podatci2 = podatci2[int(len(podatci2)*0.751):]
podatci2 = podatci2 - podatci2[0]

podatci3 = np.array(podatci3)
podatci3 = podatci3[int(len(podatci3)*0.751):]

def gen_lambda_and_mi(podatci1,podatci2, podatci3, seq_len, t):
        '''
        This Funcion consist of 3 parts:
        1. We generate 3 matrix with len(len_of_simulation(in minutes))
        2.a) We put 1 in moment (minut) in which the event happend 
             and 0 otherwise (for repair and failure rates)
        2.b) We put class (1,2,3) of falilure in moment (minut) in which the event happend 
             and 0 otherwise (for class of failures)
        3. We go through Matrixes with sliding_window (window size, and step) 
        '''

        #1. step: Generate matrix for each list
        matrix = np.zeros(podatci1[-1]+1)
        matrix1 = np.zeros(podatci2[-1]+1)
        matrix2 = np.zeros(podatci1[-1]+1)

        #2. step: Fill the each Matrix with mooments 
        #(in which minute) event happend (or what type of event happend in that moment (matrix2))
        for index, value in enumerate(podatci1):
            matrix[int(value)] = 1
            matrix2[int(value)] = podatci3[index] + 1      
        for i in podatci2:
            matrix1[int(i)] = 1 

        #3. step: Evaluate for window_size(seq_len) and step(dt)         
        lambd = []
        mi = [] 
        fail_distribution = []   
        start = 0
        end = seq_len
        one_day = (24*60) / seq_len #when multiplied with window_size(seq_len) returns values scaled to 1 day
        for i in range(int((len(matrix)-seq_len)/t)):
            #Generate fail number per window and step
            ls = matrix[start:end]
            lambd.append(sum(ls)*one_day) #Number of Failures per 1 day for current window and step
            #Generate repair number per window and step
            ls1 = matrix1[start:end]
            mi.append(sum(ls1)*one_day) #Number of Repairs per 1 day for current window and step
            #Generate fail_distribution per window and step
            types_of_fail_ = []
            types_of_fail_.extend(matrix2[start:end])
            Mechanical_fail = types_of_fail_.count(1)
            Electro_fail = types_of_fail_.count(2) 
            Other_fail = types_of_fail_.count(3) 
            total_fail = Mechanical_fail + Electro_fail + Other_fail
            ls_ = [podeli(Mechanical_fail, total_fail), podeli(Electro_fail,total_fail), podeli(Other_fail,total_fail)]
            fail_distribution.append(ls_) 
             
            start += t #update for step(dt)
            end += t #update for step(dt)
        return lambd, mi, fail_distribution
        
for i in range(1):
    ls_ = []
    #make list vector with cosistant(same) len
    #sometimes one vector is longer than another for single value
    len_1 = len(podatci1)
    len_2 = len(podatci2)
    len_3 = len(podatci3)
    ls_.extend((len_1, len_2, len_3))
    min_len = min(ls_)
    podatci1 = podatci1[:min_len]
    podatci2 = podatci2[:min_len]
    podatci3 = podatci3[:min_len]

    seq_leng = [7*24*60, 15*24*60, 30*24*60] #windows of 7, 15 and 30 days
    dt = [8*60] #step of 60 minutes
    Results = []
    for seq_len in seq_leng:
        for t in dt:
            lamb, mi, fail_distribution =  gen_lambda_and_mi(podatci1,podatci2,podatci3, int(seq_len), t)
            mi_gen = np.array(mi).reshape(-1, 1)
            lamb_gen = np.array(lamb).reshape(-1, 1)
            fail_distribution = np.array(fail_distribution).reshape(-1, 3)

            #Optinal: Save into lists
            np.save('fail_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)), int(t/60)), lamb_gen)
            np.save('repair_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)), int(t/60)), mi_gen)
            np.save('class_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)), int(t/60)), fail_distribution)
            