# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:42:00 2020

@author: Freedom
"""
import pandas as pd
import numpy as np

df  = pd.read_excel(r'C:\Users\Freedom\Documents\GitHub\Master_rad\DataSet\Mreza\Zastoji.xlsx')
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by=['Početak zastoja'])

df = pd.read_excel(r"C:\Users\Freedom\Documents\GitHub\Master_rad\DataSet\Zastoji.xlsx", index_col=0)

k = int(len(df['Vreme_zastoja']))
i = 0

df.reset_index(inplace=True, drop=True)

lista2 = []
lista1 = []

for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	if df["Vreme_zastoja"].iloc[i] > 75000:
		continue
	if df["Vreme_rada"].iloc[i] > 75000:
		continue
	lista1.append(df["Pocetak_zastoja_u_minutima"].iloc[i])
	lista2.append(df["Kraj_zastoja_u_miutama"].iloc[i])

podatci1 = np.array(lista1)