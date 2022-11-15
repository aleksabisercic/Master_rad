# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:07:39 2020

@author: Freedom
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:46:46 2020

@author: Freedom
"""
import numpy as np
from scipy.optimize import fsolve
import torch
import torch.nn as nn
import pandas as pd
import tensorflow as tf
import statistics as stat
""" FUNKCIJE KOJE SE KORISTE U OKVIRU PROCEDURE """
def podeli(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


""" ZADATI PRE STARTA PROGRAMA """

Bisection_izbor = False
vreme_ispisa = 5000

""" PARAMETRI KOJI SE NE MENJAJU """ 

# Lbe=0.0000669323/60 #prebacio oz min u sec
# Lbm=0.0000433066/60
# Lbo=0.000073458/60
# Ldo=0.0000588205/60
# Lt1e=0.0000680535/60
# Lt1m=0.0000531802/60
# Lt1=0.000121234/60
# Lt2e=0.0000564661/60
# Lt2m=0.000059579/60
# Lt2=0.000116045/60
# lambda_t3e=0.000144174/60
# Lt3m=0.0000758069/60
# Ltxo=0.0000913767/60
# mbe=0.00027154/60 #Proveri da li si lepo sracunao(podelio sa 60)
# mbm=0.000474394/60
# mbo=0.0000731771/60
# mb=0.000323161/60
# mdo=0.000226708/60
# mt1e=0.000393955/60
# mt1m=0.000817231/60
# mt2e=0.000445557/60
# mt2m=0.00090446/60
# mt3e=0.000534383/60
# mt3m=0.000627307/60
# mtxo=0.000385339/60
# Lb=0.000183697/60
# Ld=0.0000588205/60
# Ltx=0.000548637/60
# L_btd=0.000791154/60
# Pvo1=0.18988903
# Pvo2=0.36405672
# Pvo3=0.446054254
Peo1=0.42532468 
Peo2=0.02435065
Peo3=0.1461039
Peo4=0.40422078
Pmo1=0.303979678238781
Pmo2=0.191363251481795
Pmo3=0.171041490262489
Pmo4=0.333615580016935
Poo1=0.111264685556323
Poo2=0.840359364201797
Poo3=0.0483759502418798
Ptt1=0.01428571
Ptt2=0.257142857
Ptt3=0.728571429 # ctrl+k+c or shift+alt+a


# broj_sekundi_u_godini=31536000
# broj_sekundi_u_godini = 7948800
""" INICIJALNE VREDNOSTI PARAMETARA """

VrRb=0
VrCnRb=0
VrRADb=0
VrOEb=0
brOEb=0
VrOMb=0
brOMb=0
VrOOb=0
brOOb=0
VrOTKb=0
brOTKb=0
VrRd=0
VrCnRd=0
VrRADd=0
VrOOd=0
brOOd=0
VrOTKd=0
brOTKd=0
VrRt=0
VrCnRt=0
VrRADt=0
VrOTKt=0
brOTKt=0
VrRt1=0
VrCnRt1=0
VrRADt1=0
VrOEt1=0
brOEt1=0
VrOMt1=0
brOMt1=0
VrOOt1=0
brOOt1=0
VrOTKt1=0
brOTKt1=0
VrRt2=0
VrCnRt2=0
VrRADt2=0
VrOEt2=0
brOEt2=0
VrOMt2=0
brOMt2=0
VrOOt2=0
brOOt2=0
VrOTKt2=0
brOTKt2=0
VrRt3=0
VrCnRt3=0
VrRADt3=0
VrOEt3=0
brOEt3=0
VrOMt3=0
brOMt3=0
VrOOt3=0
brOOt3=0
VrOTKt3=0
brOTKt3=0
brCnRb=0
topr = 0 
VrRbtd=0
brCnRd=0
brCnRt1=0
brCnRt2=0
brCnRt3=0
SrVrRbt = 0
Abt = 0	 
VrRbt = 0  

STbtd="11"
STb="11"
STd="11"
STt="11"
STt1="11"
STt2="11"
STt3="11"
	
brR=0
vreme_simulacije = 259200 #  6 meseci 
vremena_otkaza = []
vremena_popravke = []

'''Loading Deterministic Model and Data'''

df = pd.read_excel("Zastoji.xlsx", index_col=0)
df = df[df["Sistem"] == "BTD SchRs-800"]
df = df.sort_values(by=['Početak zastoja'])

df = df[['Vrsta_zastoja', 'Pocetak_zastoja_u_minutima', 'Vreme_zastoja', 'Vreme_rada']]

k = int(len(df['Vreme_zastoja']))
i = 0

df.reset_index(inplace=True, drop=True)

lista = []
lista11 = []

for i in range(0, len(df.index)):  # df['Vreme_zastoja']:
	lista.append(df["Vreme_zastoja"].iloc[i])
	lista.append(df["Vreme_rada"].iloc[i])
	lista11.append(df["Vrsta_zastoja"].iloc[i])

podatci = np.array(lista)
podatci = podatci.reshape(-1, 1)
podatci1 = np.array(lista)
podatci1 = podatci1.reshape(-1, 1)
#datax =  sc.fit_transform(podatci1)

datay = podatci
datax = podatci1
#datax = preprocessing.normalize(datax)

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length - 2)):
        if (i % 2) != 0: continue
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length]
        _y1 = datay[i + seq_length + 1]
        x.append(_x)
        y.append(_y)	
        y.append(_y1)	
    y = np.array(y).reshape(len(x),2)
    x = np.array(x).reshape(len(x),int(seq_length/2),2)
    
    return x, y

timesteps = 50
x,y = sliding_windows(datax, datay, timesteps)


train_size = int(len(y) * 0.8)
test_size = len(y) - train_size
 			
trainX = np.array(x[0:train_size])			
trainY = np.array(y[0:train_size])
#reshape outputs into (samples, timesteps, features) 	
trainY	= np.expand_dims(trainY, axis=1)	

testX = np.array(x[train_size:len(x)])
testY = np.array(y[train_size:len(y)])
#reshape outputs into (samples, timesteps, features) 	
testY	= np.expand_dims(testY, axis=1)	

time = np.arange(len(x))
time_train = time[:train_size]
time_valid = time[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

'''path = 'LSTM encoder_decoder Multivere Timeseries.pt'
LSTM_model = tf.keras.models.load_model(path) #loading model

prediction = LSTM_model.predict(testX) #predicting faliures and repairs 
prediction = prediction.reshape(prediction.shape[0], prediction.shape[2]) 

repair_len =  trainY[:,:,0].reshape(-1) #select only repair len
repair_len = [i for i in repair_len if i <= 1500] #drop outliers
stdev_repair = stat.stdev(repair_len) #standar diviation

working_len =  trainY[:,:,1].reshape(-1)
working_len = [i for i in working_len if i <= 2000] #drop outliers
stdev_work = stat.stdev(working_len) #standar diviation

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

counter = 0
tp0 = np.random.normal(prediction[counter][1], stdev_work)
tp0 = np.abs(np.round(tp0))'''

'''Loading Class model and Data for it'''

labels_raw = np.array(lista11)

data_Y = []
for label in labels_raw:
    if label == 'Masinski':
        data_Y.append(0)
    elif label == 'Elektro':
        data_Y.append(1)
    elif label == 'Ostalo':
        data_Y.append(2)

dataX = np.array(data_Y).reshape(-1,1)
dataY = np.array(data_Y).reshape(-1,1)
def sliding_windows(datax, datay, seq_length):
    x = []
    y = []
    for i in range( int(len(datax) - seq_length )):
        _x = datax[i:(i + seq_length)]
        _y = datay[i + seq_length ]
        x.append(_x)
        y.append(_y)	
    return np.array(x),np.array(y)

window_size = 50

x,y = sliding_windows(dataX, dataY, window_size)
split = int(0.8*len(x))
x_test = x[split:]
y_test = y[split:]
path = 'Classification_model'
class_model = tf.keras.models.load_model(path) #loading model

MEO_prediction = class_model.predict(x_test) #predicting class of failure
MEO_prediction = MEO_prediction.reshape(MEO_prediction.shape[0], 3) 

Pvo1 = MEO_prediction[counter][1] #elektro
Pvo2 = MEO_prediction[counter][0] #masinski
Pvo3 = MEO_prediction[counter][2] #ostali

''' Pvs1 = MEO_prediction[counter][0] #bager 
Pvs2 = MEO_prediction[counter][1] #drobilana
Pvs3 = MEO_prediction[counter][2] #trakasti ''' #ja bih izbacio ovo
failure_type = []
for t in range(1,vreme_simulacije,1):   
    if t == tp0:        
        brR = brR + 1
        r = np.random.uniform(low=0.0, high=1.0,size=None)
        if r < Pvo1:
            failure_type.append(1)
            VRotk="1"
            r1=np.random.uniform(low=0.0, high=1.0,size=None)            
            if r1 < Peo1:
                OBJeo = "1"
                brOEb = brOEb + 1
                STb = "21"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STt2 = "12"
                STt3 = "12"            
            elif r1 < (Peo2+Peo2):
                OBJeo = "2"
                brOEt1 = brOEt1 + 1
                STt1 = "21"
                r12 = np.random.uniform(low=0.0, high=1.0,size=None)
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STb = "12"
                STt2 = "12"
                STt3 = "12"
            elif r1 < (Peo1+Peo2+Peo3):
                OBJeo = "3"
                brOEt2 = brOEt2 + 1
                STt2 = "21"
                r13 = np.random.uniform(low=0.0, high=1.0,size=None)
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STb = "12"
                STt3 = "12"
            elif r1 <= 1: #<=1
                OBJeo = "4"
                brOEt3 = brOEt3 + 1
                STt3 = "21"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STt2 = "12"
                STb = "12"
        elif r < (Pvo1+Pvo2):
            failure_type.append(2)
            Vrotk="2"
            r2=np.random.uniform(low=0.0, high=1.0,size=None)
            if r2<Pmo1:
                OBJmo = "1"
                brOMb = brOMb + 1
                STb="22"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STt3="12"
            elif r2 < (Pmo1+Pmo2):
                OBJmo = "2"
                brOMt1 = brOMt1 +1
                STt1 = "22"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STb = "12"
                STt2 = "12"
                STt3 = "12"
            elif r2 < (Pmo1+Pmo2+Pmo3):
                OBJmo = "3"
                brOMt2 = brOMt2 + 1
                STt2 = "22"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STb = "12"
                STt3 = "12"
            elif r2 <= (1):
                OBJmo = "4"
                brOMt3 = brOMt3 + 1
                STt3 = "22"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STt2 = "12"
                STb = "12"

        elif r <= 1: 
            failure_type.append(2)
            Vrotk = "3"
            r3=np.random.uniform(low=0.0, high=1.0,size=None)
            if r3 < Poo1:
                OBJoo = "1"
                brOOb = brOOb + 1
                STb = "23"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STd = "12"
                STt1 = "12"
                STt2 = "12"
                STt3 = "12"

            elif r3 < (Poo1+Poo2):
                OBJoo = "2"
                brOOd = brOOd + 1
                STd = "23"
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT
                STt1 = "12"
                STb = "12"
                STt2 = "12"
                STt3 = "12"
                
            elif r3 <= (1):
                OBJoo = "3"
                r4 = np.random.uniform(low=0.0, high=1.0,size=None)

                if r4 < Ptt1:
                    OBJtt = "1"
                    brOOt1 = brOOt1 + 1
                    STt1 = "23"
                    STd = "12"
                    STb = "12"
                    STt2 = "12"
                    STt3 = "12"

                elif r4 < (Ptt1+Ptt2):
                    OBJtt = "2"
                    brOOt2 = brOOt2 + 1
                    STt2 = "23"
                    STt1 = "12"
                    STd = "12"
                    STb = "12"
                    STt3 = "12"

                elif r4 <= 1:
                    OBJtt ="3"
                    brOOt3 = brOOt3 + 1
                    STt3 = "23"
                    STt1 = "12"
                    STd = "12"
                    STt2 = "12"
                    STb = "12"
            
                DT = np.random.normal(prediction[counter][1], stdev_repair)
                DT = np.abs(np.round(DT))
                topr = t+DT

    if t==topr:
        vremena_popravke.append(topr)   
        STd = "11"
        STt1 = "11"
        STt2 = "11"
        STt3 = "11"
        STb = "11"
        STbtd ="11"
		
        counter += 1
        tp0 = np.random.normal(prediction[counter][1], stdev_work)
        tp0 = np.abs(np.round(tp0))
        tp0 = t + tp0
        vremena_otkaza.append(tp0)
        Pvo1 = MEO_prediction[counter][1]
        Pvo2 = MEO_prediction[counter][0]
        Pvo3 = MEO_prediction[counter][2]

    if STb == '11':
        VrRb = VrRb + 1
        VrRADb = VrRADb + 1
    
    if STd == "11":
        VrRd = VrRd + 1
        VrRADd = VrRADd + 1

    if STt1 == "11":
        VrRt1 = VrRt1 + 1
        VrRADt1 = VrRADt1 + 1
    
    if STt2 == "11":
        VrRt2 = VrRt2 + 1
        VrRADt2 = VrRADt2 + 1
    
    if STt3 == "11":
        VrRt3 = VrRt3 + 1
        VrRADt3 = VrRADt3 + 1
	
    if STb == "12":  
        VrCnRb = VrCnRb + 1
        VrRADb = VrRADb + 1
    if STb=="21":
        VrOEb = VrOEb + 1
        VrOTKb = VrOTKb + 1
    if STb == "22":
        VrOMb = VrOMb + 1
        VrOTKb = VrOTKb + 1
    if STb == "23":
        VrOOb = VrOOb + 1
        VrOTKb = VrOTKb + 1
    if STt1 == "12":  
        VrCnRt1 = VrCnRt1 + 1
        VrRADt1 = VrRADt1 + 1
    if STt1 == "21":
        VrOEt1 = VrOEt1 + 1
        VrOTKt1 = VrOTKt1 + 1
    if STt1 == "22":
        VrOMt1 = VrOMt1 + 1
        VrOTKt1 = VrOTKt1 + 1
    if STb == "23":
        VrOOt1 = VrOOt1 + 1
        VrOTKt1 = VrOTKt1 + 1
    if STt2 == "12":  
        VrCnRt2 = VrCnRt2 + 1
        VrRADt2 = VrRADt2 + 1
    if STt2 == "21":
        VrOEt2 = VrOEt2 + 1
        VrOTKt2 = VrOTKt2 + 1
    if STt2 == "22":
        VrOMt2 = VrOMt2 + 1
        VrOTKt2 = VrOTKt2 + 1
    if STt2 == "23":
        VrOOt2 = VrOOt2 + 1
        VrOTKt2 = VrOTKt2 + 1
    if STt3 == "12":  
        VrCnRt3 = VrCnRt3 + 1
        VrRADt3 = VrRADt3 + 1 
    if STt3 == "21":
        VrOEt3 = VrOEt3 + 1
        VrOTKt3 = VrOTKt3 + 1
    if STt3 == "22":
        VrOMt3 = VrOMt3 + 1
        VrOTKt3 = VrOTKt3 + 1
    if STt3 == "23":
        VrOOt3 = VrOOt3 + 1
        VrOTKt3 = VrOTKt3 + 1
    if STd == "12":  
        VrCnRd = VrCnRd + 1
        VrRADd = VrRADd + 1
    if STd == "23":
        VrOOd = VrOOd + 1
        VrOTKd = VrOTKd + 1
    if STb == "11" and  STd == "11" and STt1 == "11" and  STt1 == "11" and STt2 == "11" and STt3 == "11":
        VrRbtd = VrRbtd + 1

    # if t%vreme_ispisa == 0:
    #     print(t,tp0,topr)

""" PROVERA """
if (VrRt1 == VrRt2 and VrRt1 == VrRt3):
    VrRt = VrRt1
    VrCnRt = VrOEb + VrOMb + VrOOb + VrOOd
    brCnRt = brOEb + brOMb + brOOb + brOOd 
    VrRADt = VrRt + VrCnRt
    VrOTKt = VrOTKt1 + VrOTKt2 + VrOTKt3
    brOTKt = brOEt1 + brOMt1 + brOOt1 + brOEt2 + brOMt2 + brOOt2 + brOEt3 + brOMt3 + brOOt3
    brCnRb = brOOd + brOTKt
    brCnRd = brOEb + brOMb + brOOb + brOTKt
else:
    print(VrRt1, VrRt2, VrRt3)
    print("Greska_Kraj")

""" RACUNANJE STATISTIKA ZA JEDNU SIMULACIJU """
#bager statistika

SrVrRb = podeli(VrRb,brR) 
SrVrCnRb = podeli(VrCnRb, brCnRb)
SrVrRADb = podeli(VrRADb, (brR + brCnRb))
SrVrOEb = podeli(VrOEb, brOEb)
SrVrOMb = podeli(VrOMb,brOMb)
SrVrOOb = podeli(VrOOb,brOOb)
brOTKb = brOEb + brOMb + brOOb
SrVrOTKb = podeli(VrOTKb, brOTKb)
Ab = VrRADb / ( VrRADb + VrOTKb )
Aeb = VrRADb / ( VrRADb + VrOEb )
Amb = VrRADb / ( VrRADb + VrOMb )
Aob = VrRADb / ( VrRADb + VrOOb )

#drobilica statistika

SrVrRd = podeli(VrRd, brR) 
SrVrCnRd = podeli(VrCnRd, brCnRd)
SrVrRADd = podeli(VrRADd, ( brR + brCnRd ))
SrVrOOd = podeli(VrOOd, brOOd)
brOTKd =  brOOd
SrVrOTKd = podeli(VrOTKd, brOTKd)
SrVrOOd = podeli(VrOOd, brOOd)
SrVrOTKd = SrVrOOd
Ad = VrRADd / ( VrRADd + VrOTKd )
Aod = VrRADd / ( VrRADd + VrOOd )

#Trakasti transporteri statistika
SrVrRt = podeli(VrRt, brR)
SrVrCnRt = podeli(VrCnRt, brCnRt)
SrVrRADt = VrRADt / ( brR + brCnRt )
SrVrOTKt = podeli(VrOTKt, brOTKt)
At = VrRADt / ( VrRADt + VrOTKt )

#Trakasti transporter 1 stat.
brCnRt1 = brOTKb + brOTKd + brOEt2 + brOMt2 + brOOt2 + brOEt3 + brOMt3 + brOOt3
SrVrRt1 = podeli(VrRt1, brR) 
SrVrCnRt1 = podeli(VrCnRt1, brCnRt1)
SrVrRADt1 = podeli(VrRADt1, ( brR + brCnRt1 ))
SrVrOEt1 = podeli(VrOEt1, brOEt1) # ovde mi izbacuje gresku deljenje sa nulom ali moguce da nije doslo do otkaza Elektro motora T1
SrVrOMt1 = podeli(VrOMt1, brOMt1)
SrVrOOt1 = podeli(VrOOt1, brOOt1)
brOTKt1 = brOEt1 + brOMt1 + brOOt1
SrVrOTKt1 = podeli(VrOTKt1, brOTKt1)
At1 = VrRADt1 / ( VrRADt1 + VrOTKt1 )
Aet1 = VrRADt1 / ( VrRADt1 + VrOEt1 )
Amt1 = VrRADt1 / ( VrRADt1 + VrOMt1 )
Aot1 = VrRADt1 / ( VrRADt1 + VrOOt1 )

#Trakasti transporter 2 stat.

brCnRt2 = brOTKb + brOTKd + brOEt1 + brOMt1 + brOOt1 + brOEt3 + brOMt3 + brOOt3
SrVrRt2 = podeli(VrRt2, brR) 
SrVrCnRt2 = podeli(VrCnRt2, brCnRt2)
SrVrRADt2 = podeli(VrRADt2, ( brR + brCnRt2 ))
SrVrOEt2 = podeli(VrOEt2, brOEt2)
SrVrOMt2 = podeli(VrOMt2, brOMt2)
SrVrOOt2 = podeli(VrOOt2, brOOt2)
brOTKt2 = brOEt2 + brOMt2 + brOOt2
SrVrOTKt2 = podeli(VrOTKt2, brOTKt2)
At2 = VrRADt2 / ( VrRADt2 + VrOTKt2 )
Aet2 = VrRADt2 / ( VrRADt2 + VrOEt2 )
Amt2 = VrRADt2 / ( VrRADt2 + VrOMt2 )
Aot2 = VrRADt2 / ( VrRADt2 + VrOOt2 )

#Trakasti transporter 2 stat.

brCnRt3 = brOTKb + brOTKd + brOEt1 + brOMt1 + brOOt1 + brOEt2 + brOMt2 + brOOt2
SrVrRt3 = VrRt3 / brR 
SrVrCnRt3 = podeli(VrCnRt3, brCnRt3)
SrVrRADt3 = podeli(VrRADt3, ( brR + brCnRt3 ))
SrVrOEt3 = podeli(VrOEt3, brOEt3)
SrVrOMt3 = podeli(VrOMt3, brOMt3)
SrVrOOt3 = podeli(VrOOt3, brOOt3)
brOTKt3 = brOEt3 + brOMt3 + brOOt3
SrVrOTKt3 = podeli(VrOTKt3, brOTKt3)
At3 = VrRADt3 / ( VrRADt3 + VrOTKt3 )
Aet3 = VrRADt3 / ( VrRADt3 + VrOEt3 )
Amt3 = VrRADt3 / ( VrRADt3 + VrOMt3 )
Aot3 = VrRADt3 / ( VrRADt3 + VrOOt3 )

#Statistika sistema
VrRbtd = VrRb #**proveriti
SrVrRbtd = podeli(VrRbtd, brR)
Abtd = VrRbtd / ( VrRbtd + VrOTKb + VrOTKd + VrOTKt)
	#Statistika bt_sistema
SrVrRbt = SrVrRbtd 
VrRbt = VrRbtd
Abt = Abtd	

list_stat_B = [brCnRb, SrVrRb , SrVrCnRb , SrVrRADb , SrVrOEb , SrVrOMb , SrVrOOb , brOTKb , SrVrOTKb , Ab , Aeb , Aob , Amb]
list_stat_D = [brCnRd, SrVrRd , SrVrCnRd , SrVrRADd , SrVrOOd , brOTKd , SrVrOTKd ,  SrVrOOd , SrVrOTKd, Ad , Aod]
list_stat_T1 = [brCnRt1, SrVrRt1 , SrVrCnRt1 , SrVrRADt1 , SrVrOEt1 , SrVrOMt1 , SrVrOOt1 , brOTKt1 , SrVrOTKt1,  At1, Aet1 , Aot1 , Amt1]
list_stat_T2 = [brCnRt2, SrVrRt2, SrVrCnRt2, SrVrRADt2, SrVrOEt2, SrVrOMt2, SrVrOOt2, brOTKt2, SrVrOTKt2, At2, Aet2, Aot2, Amt2]
list_stat_T3 = [brCnRt3, SrVrRt3 , SrVrCnRt3 , SrVrRADt3 , SrVrOEt3 , SrVrOMt3 , SrVrOOt3 , brOTKt3 , SrVrOTKt3 , At3 , Aet3 , Aot3 , Amt3]
list_stat_T = [SrVrRt , SrVrCnRt , SrVrRADt , SrVrOTKt , At]
list_stat_BTD = [VrRbtd , SrVrRbtd , Abtd]
lista_stat_BT = [VrRbt , SrVrRbt , Abt]

vremena_otkaza = np.asarray(vremena_otkaza)
vremena_popravke = np.asarray(vremena_popravke)
   

podatci1 = vremena_otkaza.reshape(-1)
podatci2 = vremena_popravke.reshape(-1)

'''def gen_lambda_and_mi(podatci1,podatci2, seq_len, t):
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
        
seq_leng = [15*24*60, 7*24*60, 30*24*60]
dt = [10, 30, 60]

for seq_len in seq_leng:
    for t in dt:
        lamb, mi =  gen_lambda_and_mi(podatci1,podatci2, int(seq_len), t)
        mi_gen_simulacija = np.array(mi).reshape(-1, 1)
        lamb_gen_simulacija = np.array(lamb).reshape(-1, 1)
        sim_name_lam = 'Failure_rates_' + str(t) + 'dt_' + str(seq_len) + 'min_simulacija' + '.npy'
        sim_name_mi = 'Repair_rates' + str(t) + 'dt' + str(seq_len) + 'min_simulacija' + '.npy'
        np.save(sim_name_lam, lamb_gen_simulacija)
        np.save(sim_name_mi +str(t), mi_gen_simulacija)  '''

    
    
    
       
    

