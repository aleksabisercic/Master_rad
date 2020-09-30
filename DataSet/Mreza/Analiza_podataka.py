# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:12:37 2020

@author: Andrija Master
"""

import pandas as pd
import datetime



df_mas = pd.read_excel('Mašinski zastoji Drmno.xlsx', header = 1 )
df_ele = pd.read_excel('Elektro-zastoji Drmno.xlsx', header = 1)
df_ost = pd.read_excel('Ostali-zastoji Drmno.xlsx', header = 1)
df_mas['Vrsta_zastoja'] = 'Masinski'
df_ele['Vrsta_zastoja'] = 'Elektro'
df_ost['Vrsta_zastoja'] = 'Ostalo'

kolone = ['Datum','Mesec','Godina','Sistem','Objekat','Početak zastoja',
          'Ukupno vreme zastoja u minutima','Kraj zastoja','Vrsta_zastoja']

mapping = {'DATUM':'Datum','SISTEM':'Sistem','OBJEKAT':'Objekat',
           'POČETAK ZASTOJA':'Početak zastoja','KRAJ ZASTOJA':'Kraj zastoja'}

df_ele.rename(columns = mapping, inplace = True)
df_ost.rename(columns = mapping, inplace = True)

df_mas = df_mas[kolone]
df_ele = df_ele[kolone]
df_ost = df_ost[kolone]


df = pd.concat([df_mas,df_ele,df_ost])
df['Početak zastoja'] = pd.to_datetime(df['Početak zastoja'],dayfirst=True)

df.dropna(inplace = True)
df.sort_values(by = 'Početak zastoja',inplace = True)

df_new = pd.DataFrame(columns = kolone)
    
broj = 0    
df = df[df["Sistem"] != "BTD ERs-710 U"]
df =  df[df["Sistem"] != "I BTO ERs-710 J"]
df = df[df["Sistem"] != "IV BTO ERs-710 J"]
df = df[df["Sistem"] != "III BTO ERs-710 J"]
df = df[df["Sistem"] != "I BTO ERs-710 U"]
df = df[df["Sistem"] != "III BTO ERs-710 U"]


for i in df["Sistem"].unique():
    df1 = df[df["Sistem"] == i]
    df1.sort_values(by = ['Početak zastoja'])
    k=0
    while k < len(df1.index)-1:
        df_new.loc[k+broj] = df1.iloc[k,:]
        k1 = k
        if k1 < len(df1.index)-1:
            while df1["Kraj zastoja"].iloc[k1] >= df1["Početak zastoja"].iloc[k1+1]:
                df_new.loc[k+broj,"Kraj zastoja"] = df1["Kraj zastoja"].iloc[k1+1]
                k1+=1
                if k1 >= len(df1.index)-1:
                    break
        if k1>k:    
            k=k1+1
        else:
            k+=1
    broj+=len(df1.index)

df_new.reset_index(inplace = True, drop = True)
df_new["Vreme zastoja"] = df_new["Kraj zastoja"] - df_new["Početak zastoja"]
df_new["Vreme zastoja"] = df_new["Vreme zastoja"]//datetime.timedelta(minutes=1)
df_new = df_new[df_new["Vreme zastoja"]>0]

df_new.to_csv('Zastoji.csv')          
df_new.to_excel("Zastoji.xlsx")

df = pd.read_excel("Zastoji.xlsx", index_col = 0)
df['Početak zastoja'] = pd.to_datetime(df['Početak zastoja'],  dayfirst=True)
df['Kraj zastoja'] = pd.to_datetime(df['Kraj zastoja'], dayfirst=True)
df.sort_values(by = 'Početak zastoja',inplace = True)


df['Vreme_rada'] = 0
for i in df["Sistem"].unique():
    df1 = df[df["Sistem"]==i]
    razlika = df1["Početak zastoja"] - df1["Kraj zastoja"].shift()
    razlika = razlika//datetime.timedelta(minutes=1)
    indeksi = df[df["Sistem"]==i].index
    df.loc[indeksi,'Vreme_rada'] = razlika.values
df.loc[df[df['Vreme_rada']<0].index,'Vreme_rada'] = 0

df.dropna(inplace=True)
df.drop(columns="Ukupno vreme zastoja u minutima",inplace = True)

df.reset_index(inplace = True, drop = True)

df.to_excel("Zastoji.xlsx")

df = pd.read_excel("Zastoji.xlsx", index_col = 0)

lokacije = []
for i in df["Sistem"].unique():
    if df[df["Sistem"]==i].shape[0] > 1000:
        print(df[df["Sistem"]==i].shape[0],i)
        lokacije.append(i)
		
Jan2016 = pd.datetime(2016, 1, 1) 
Jan1st2016 =  pd.to_datetime(Jan2016 ,dayfirst=True)

df = df[df.Godina != 2019]	
	
df.sort_values(by = 'Početak zastoja',inplace = True)
df["Pocetak_zastoja_u_minutima"] = df["Početak zastoja"] - Jan1st2016  
#df["Pocetak zastoja u minutima"] = int(df["Pocetak zastoja u minutima"].total_seconds() / 60)
df["Pocetak_zastoja_u_minutima"] = df["Pocetak_zastoja_u_minutima"]//datetime.timedelta(minutes=1)
df["Kraj_zastoja_u_miutama"] = df["Kraj zastoja"] - Jan1st2016
df["Kraj_zastoja_u_miutama"] = df["Kraj_zastoja_u_miutama"]//datetime.timedelta(minutes=1)
        
df = df[df["Sistem"].isin(lokacije)]
df.reset_index(inplace = True, drop = True)
df.to_excel("Zastoji.xlsx")        
#reviews.iloc[:, 0]
#reviews.loc[0, 'country']

brojac = 0 
t = 10
X_osa = int(len(Training_lista)/2)
A = np.zeros(( X_osa , t+2 ))
print(A.shape)
x=0
for i in range (0, X_osa):
	A[i][-2] = Training_lista[x]
	A[i][-1] = Training_lista[x+1]
	for n in range (0,t):
		while (x-n) >= 0:
			if brojac == t:
				break			
			for g in range (3,t+1):
				A[i][-g] = Training_lista[(x-1)+3-g]	
		brojac = brojac + 1
	x = x + 2
print(A)