# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:56:55 2020

@author: Andri
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder




df = pd.read_excel("Zastoji.xlsx", index_col = 0)
step = 20
one_hot = pd.get_dummies(df['Vrsta_zastoja'])
df = df.drop('Vrsta_zastoja',axis = 1).join(one_hot)

mapping = {"JANUAR":0,"FEBRUAR":1,"MART":2,"APRIL":3,"MAJ":4,"JUN":5,
      "JUL":6,"Jul":6,"AVGUST":7,"SEPTEMBAR":8,"OKTOBAR":9,"NOVEMBAR":10,"DECEMBAR":11}

df.replace({'Mesec':mapping},inplace = True)

kolone = ["Sistem","Mesec","Vreme zastoja","Vreme_rada","Elektro","Masinski","Ostalo"]

df = df.loc[:,kolone]


