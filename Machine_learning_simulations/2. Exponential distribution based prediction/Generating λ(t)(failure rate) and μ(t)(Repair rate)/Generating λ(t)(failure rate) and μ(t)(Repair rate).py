# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:05:20 2020

@author: Freedom
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
	if df["Vreme_zastoja"].iloc[i] > 75000:
		continue
	if df["Vreme_rada"].iloc[i] > 75000:
		continue
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
podatci1 = podatci1 - podatci1[0]
podatci2 = np.array(lista2)
podatci2 = podatci2 - podatci1[0]
podatci3 = np.array(podatci3)

def gen_lambda_and_mi(podatci1,podatci2, seq_len, t):
    a = np.zeros(podatci2[-1])
    for i in podatci1:
        a[0] = 1
        if i == 0: continue
        a[i] = 1
        
    a1 = np.zeros(podatci2[-1])
    for i in podatci2:
        a1[i-1] = 1  

    lambd = []
    mi = []    
    start = 0
    end = seq_len
    for i in range(int((len(a)-seq_len)/t)):
        ls = a[start:end]
        lambd.append(sum(ls))
        ls1 = a1[start:end]
        mi.append(sum(ls1))
        start += t
        end += t
    lambd.append(sum(a[-seq_len:]))
    mi.append(sum(a1[-seq_len:]))
    return lambd, mi

def gen_mi(podatci2, seq_len, t):
    a = np.zeros(podatci2[-1])
    for i in podatci2:
        a[i-1] = 1
    mi = []
    start = 0
    end = seq_len
    for i in range(int((len(a)-seq_len)/t)):
        ls = a[start:end]
        mi.append(sum(ls))
        start += t
        end += t
    mi.append(sum(a[-seq_len:]))
    return mi

def gen_lambda(podatci1, podatci2 ,seq_len, t):
    a = np.zeros(podatci2[-1])
    for i in podatci1:
        a[0] = 1
        if i == 0: continue
        a[i] = 1
    lamb = []
    start = 0
    end = seq_len
    for i in range(int((len(a)-seq_len)/t)):
        ls = a[start:end]
        lamb.append(sum(ls))
        start += t
        end += t
    lamb.append(sum(a[-seq_len:]))
    return lamb

def ploting(lambd, mi): 
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(lambd, 'r-')
    plt.ylabel('oktz/intervalu')
    plt.xlabel('vreme') 
    plt.title('poredjenje')
    plt.subplot(2, 1, 2)
    plt.plot(mi, 'b-')
    plt.ylabel('popravki/intervalu')
    plt.xlabel('vreme') 
    plt.show()


#seq_leng = np.random.uniform(300, 15*60*24, size = 100) 
seq_leng = [15*24*60, 7*24*60, 30*24*60]
dt = [10]#15, 30, 60]

for seq_len in seq_leng:
    for t in dt:
        lamb, mi =  gen_lambda_and_mi(podatci1,podatci2, int(seq_len), t)
        mi_gen = np.array(mi).reshape(-1, 1)
        lamb_gen = np.array(lamb).reshape(-1, 1)
        print(lamb_gen, mi_gen.shape)
        sim_name_lam = 'Failure_rates_' + str(t) + 'dt_' + str(seq_len) + 'h' + '.npy'
        sim_name_mi = 'Repair_rates' + str(t) + 'dt' + str(seq_len) + 'h' + '.npy'
        np.save(sim_name_lam, lamb_gen)
        np.save(sim_name_mi +str(t), mi_gen)
        
