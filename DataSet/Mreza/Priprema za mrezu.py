# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:19:43 2020

@author: Freedom
"""

import pandas as pd
import numpy as np


def Podatci():
	df = pd.read_excel("Zastoji.xlsx", index_col = 0)
	df = df[df["Sistem"] == "BTD SchRs-800"]
	df.sort_values(by = ['Početak zastoja'])
	
	
	#df_a = pd.DataFrame(columns = kolone)
	
	df1 = df[['Vreme_zastoja', 'Vreme_rada']]
	df1.reset_index(inplace = True, drop = True)
	
	#df1["Vreme_zastoja"] = pd.to_numeric(df["Vreme_zastoja"], downcast="float")
	#df1["Vreme_rada"] = pd.to_numeric(df["Vreme_rada"], downcast="float")
	
	lista = []
	
	for i in range (0,len(df1.index)): #df['Vreme_zastoja']:
		lista.append(df1["Vreme_zastoja"].iloc[i])
		lista.append(df1["Vreme_rada"].iloc[i])
		
	Podatci = np.array(lista)
	Training_lista = Podatci[:int(len(lista)*0.8)]
	Validation_data = Podatci[int(len(lista)*0.8):]
	
	#Training DATA
	a = Training_lista
	n = 4
	size_i = int(len(a)/2)
	size_j = 2*n+2
	
	mat = np.zeros((size_i,size_j))
	br = 2
	mat[0][size_j-2] = a[0]
	mat[0][size_j-1] = a[1]
	for i in range(1,size_i):
	    mat[i] = mat[i-1]
	    mat[i] = np.roll(mat[i],-2)
	    mat[i][size_j - 2] = a[br]
	    mat[i][size_j - 1] = a[br+1]
	    br = br + 2
	
	Y_t1 = 0
	Y_t2 = 0
	X_list = 0
	X = []
	Y = []
	for i in range (0, size_i): 
	    X_list = mat[i][:(2*n)]
	    Y_t1 = mat[i][-2]
	    Y.append(Y_t1)
	    Y_t2 = mat[i][-1] 
	    Y.append(Y_t2)
	    X.append(X_list)
	    
	X = np.array(X) 
	Y = np.array(Y) 
	
	#Validation DATA
	a = Validation_data
	n = 4
	size_i = int(len(a)/2)
	size_j = 2*n+2
	
	mat = np.zeros((size_i,size_j))
	br = 2
	mat[0][size_j-2] = a[0]
	mat[0][size_j-1] = a[1]
	for i in range(1,size_i):
	    mat[i] = mat[i-1]
	    mat[i] = np.roll(mat[i],-2)
	    mat[i][size_j - 2] = a[br]
	    mat[i][size_j - 1] = a[br+1]
	    br = br + 2
	print(mat)
	
	Y1_t1 = 0
	Y1_t2 = 0
	X1_list = 0
	X1 = []
	Y1 = []
	for i in range (0, size_i): 
	    X1_list = mat[i][:(2*n)]
	    Y1_t1 = mat[i][-2]
	    Y1.append(Y_t1)
	    Y1_t2 = mat[i][-1] 
	    Y1.append(Y_t2)
	    X1.append(X_list)
	    
	X1 = np.array(X) 
	Y1 = np.array(Y)  
	
	return X1, X, Y1, Y 
if __name__ == '__main__':
    Podatci()
    print (X)	
	# instaliraj env, instaliraj sve pakete, visual studio code
	#Drugi nacin
	#brojac = 0 
	#t = 2
	#X_osa = int(len(Training_lista)/2)
	#A = np.zeros(( X_osa , t+2 ))
	#print(A.shape)
	#x=0
	#for i in range (0, X_osa):
	#	A[i][-2] = Training_lista[x]
	#	A[i][-1] = Training_lista[x+1]
	#	for g in range (3,t+1):
	#		if (x+3-g) > 0:
	#			A[i][-g] = Training_lista[x-1+3-g]	
	#	x = x + 2
	#print
	
	#n = 4
	#size_i = int(len(a)/2)
	#size_j = 2*n+2
	
	#mat = np.zeros((size_i,size_j))
	
	#br = 2
	#mat[0][size_j-2] = a[0]
	#mat[0][size_j-1] = a[1]
	#for i in range(1,size_i):
	 #   mat[i] = mat[i-1]
	  #  mat[i] = np.roll(mat[i],-2)
	  #  mat[i][size_j - 2] = a[br]
	   # mat[i][size_j - 1] = a[br+1]
	    #br = br + 2
	
	#print(mat)