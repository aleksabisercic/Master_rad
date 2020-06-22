# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:44:21 2020

@author: Freedom
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
df['Kraj zastoja'] = pd.to_datetime(df['Kraj zastoja'],dayfirst=True)
df['Datum'] = pd.to_datetime(df['Datum'],dayfirst=True)

df.dropna(inplace = True)
df.sort_values(by = 'Početak zastoja',inplace = True)
        
df = df[df["Sistem"] != "BTD ERs-710 U"]
df =  df[df["Sistem"] != "I BTO ERs-710 J"]
df = df[df["Sistem"] != "IV BTO ERs-710 J"]
df = df[df["Sistem"] != "III BTO ERs-710 J"]
df = df[df["Sistem"] != "I BTO ERs-710 U"]
df = df[df["Sistem"] != "III BTO ERs-710 U"]

df["Vreme zastoja"] = df["Kraj zastoja"] - df["Početak zastoja"]
df["Vreme zastoja"] = df["Vreme zastoja"]//datetime.timedelta(minutes=1)
df = df[df["Vreme zastoja"]>0]

Jan2016 = pd.datetime(2016, 1, 1) 
Jan1st2016 =  pd.to_datetime(Jan2016 ,dayfirst=True)
df = df[df.Godina != 2019]

#Resenje za preklapanje sistema (moj predlog)
#for i in range (0,a):
#	n=i
#	while df1["Kraj_zastoja_u_miutama"].iloc[i] <= df1["Pocetak_zastoja_u_minutima"].iloc[n+1]: #df1.loc[i,"Kraj_zastoja_u_miutama"] <= df1.loc[n+1,"Pocetak_zastoja_u_minutima"]  #df.loc[i]['Kraj zastoja u miutama'].values
#		if df1.loc[i,"Sistem"] == df1.loc[n+1,"Sistem"]:
#			df1.loc[i,"Kraj_zastoja_u_miutama"] = df1.loc[n+1,"Kraj_zastoja_u_miutama"] 
#			df1.drop(index=[n+1] , inplace=False) #za odredjivanje indeksa df.loc[n+1,"Kraj zastoja u miutama"].index.values
#		n=+1
#	a = len(df1.index)

df_new = pd.DataFrame(columns = kolone)
broj = 0 
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
	
#df_new["Kraj_zastoja_u_miutama"] = df_new.Kraj_zastoja_u_miutama.astype(float)
#df_new["Pocetak_zastoja_u_minutima"] = df_new.Pocetak_zastoja_u_minutima.astype(float)
df_new.sort_values(by = 'Početak zastoja',inplace = True)
df_new["Pocetak_zastoja_u_minutima"] = df_new["Početak zastoja"] - Jan1st2016  
#df["Pocetak zastoja u minutima"] = int(df["Pocetak zastoja u minutima"].total_seconds() / 60)
df_new["Pocetak_zastoja_u_minutima"] = df_new["Pocetak_zastoja_u_minutima"]//datetime.timedelta(minutes=1)
df_new["Kraj_zastoja_u_miutama"] = df_new["Kraj zastoja"] - Jan1st2016
df_new["Kraj_zastoja_u_miutama"] = df_new["Kraj_zastoja_u_miutama"]//datetime.timedelta(minutes=1)



#df_new['Vreme_rada'] = 0
#for i in df_new["Sistem"].unique():
 #   df2 = df_new[df_new["Sistem"]==i]
  #  razlika = df2["Početak zastoja"] - df2["Kraj zastoja"].shift()
   # razlika = razlika//datetime.timedelta(minutes=1)
    #indeksi = df_new[df_new["Sistem"]==i].index
    #df_new.loc[indeksi,'Vreme_rada'] = razlika.values
#df_new.loc[df_new[df_new['Vreme_rada']<0].index,'Vreme_rada'] = 0

#df_new.dropna(inplace=True)
#df_new.drop(columns="Ukupno vreme zastoja u minutima",inplace = True)

#df_new.reset_index(inplace = True, drop = True)	
	
#df['Vreme rada sistema'] = 0
#for i in df["Sistem"]:
#	df.sort_values(by=['Sistem', 'Početak zastoja'])
#	if df.at[i,"Sistem"] == df.at[i+1,"Sistem"]:
#		df.loc[i,"Vreme rada sistema"] = df.loc[0,"Pocetak zastoja u minutima"] + df1.loc[n+1,"Pocetak zastoja u minutima"] - df1.loc[n+1,"Kraj zastoja u miutama"]:
		
         
df.to_excel("Zastoji.xlsx")
df_new.to_excel("Zastoji2.xlsx")