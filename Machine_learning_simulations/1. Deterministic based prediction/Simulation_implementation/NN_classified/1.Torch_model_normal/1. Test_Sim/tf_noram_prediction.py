# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:55:33 2020

@author: Freedom
"""
import numpy as np 
import matplotlib.pyplot as plt

vreme_simulacije = 259200 # duzina test seta ( 6 meseci )
vremena_otkaza = np.load('vremena_otkaza_mean_tf.npy')
vremena_popravke = np.load('vremena_popravke_mean_tf.npy')

podatci1 = vremena_otkaza.reshape(-1)
podatci2 = vremena_popravke.reshape(-1)

def gen_lambda_and_mi(podatci1,podatci2, seq_len, t):
    matrix = np.zeros(vreme_simulacije)
    for i in podatci1:
        matrix[int(i)] = 1        
    matrix1 = np.zeros(vreme_simulacije)
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
dt = [30]

for seq_len in seq_leng:
    for t in dt:
        lamb, mi =  gen_lambda_and_mi(podatci1,podatci2, int(seq_len), t)
        mi_gen_simulacija = np.array(mi).reshape(-1, 1)
        lamb_gen_simulacija = np.array(lamb).reshape(-1, 1)
        sim_name_lam = 'Failure_rates_' + str(t) + 'dt_' + str(seq_len) + 'min_simulacija' + '.npy'
        sim_name_mi = 'Repair_rates' + str(t) + 'dt' + str(seq_len) + 'min_simulacija' + '.npy'
        np.save(sim_name_lam, lamb_gen_simulacija)
        np.save(sim_name_mi +str(t), mi_gen_simulacija)
        plt.plot(mi)