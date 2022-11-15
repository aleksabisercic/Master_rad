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

#Time of simulation
vreme_simulacije = 259200 # len of  test in min ( 6 month period )

#Load datasets from simulation
vremena_otkaza = np.load('lista_vremena_otkz_4000_BTD.npy', allow_pickle=True) #shape(num of simulations, len of failures in each sim)
vremena_popravke = np.load('lista_vremena_pop_4000_BTD.npy', allow_pickle=True) #shape(num of simulations, len of repairs in each sim)
vrsta_otkaza = np.load('lista_vrsta_pop_4000_BTD.npy', allow_pickle=True)  #shape(num of simulations, len of failures in each sim)

#Load real data for evaluation
name_fail = 'Data/fail_window{}_dt{}h_real.npy'.format(int(7*24*60/(24*60)),int(60/60))
name_repair = 'Data/repair_window{}_dt{}h_real.npy'.format(int(7*24*60/(24*60)),int(60/60))
name_class = 'Data/class_window{}_dt{}h_real.npy'.format(int(7*24*60/(24*60)),int(60/60))
Real_data_fail = np.load(name_fail, allow_pickle=True)  
Real_data_rep = np.load(name_repair, allow_pickle=True)
Real_data_class = np.load(name_class, allow_pickle=True)


def division(a,b):
    '''
    Returns 0 if division by 0
    '''
    if b == 0 :
        return 0 
    else:
        return a/b

def kl_divergence(p, q):
    '''
    This is known as the relative entropy 
    or Kullback-Leibler divergence,
    or KL divergence, between the distributions p(x) and q(x).
    '''
    return sum(p[i] * np.log((p[i]/q[i])) for i in range(len(p)))

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
        matrix = np.zeros(vreme_simulacije)
        matrix1 = np.zeros(vreme_simulacije)
        matrix2 = np.zeros(vreme_simulacije)

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
            if total_fail == 0:
                total_fail += 1 # to avoid divison with 0
            ls_ = [Mechanical_fail/total_fail, Electro_fail/total_fail, Other_fail/total_fail]
            fail_distribution.append(ls_) 
             
            start += t #update for step(dt)
            end += t #update for step(dt)
        return lambd, mi, fail_distribution
Results = [] 
count = 0      
for i in range(100):
    start = time.time()
    ls_ = []
    #Load data form simulation[i]
    podatci1 = vremena_otkaza[i].reshape(-1)
    podatci2 = vremena_popravke[i].reshape(-1)
    podatci3 = vrsta_otkaza[i].reshape(-1)

    #drop if the last value is larger then len_of_simulation
    if len(podatci1) > 1:
        if vreme_simulacije < podatci1[-1]:
            podatci1[:-1]
    if len(podatci2 )> 1: 
        if vreme_simulacije < podatci2[-1]:
            podatci2[:-1]

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
    for seq_len in seq_leng:
        for t in dt:
            count += 1
            lamb, mi, fail_distribution =  gen_lambda_and_mi(podatci1,podatci2,podatci3, int(seq_len), t)
            mi_gen = np.array(mi).reshape(-1, 1)
            lamb_gen = np.array(lamb).reshape(-1, 1)
            fail_distribution = np.array(fail_distribution).reshape(-1, 3) 

            #Optinal: Save into lists
            # np.save('npy_lists/fail_window{}_dt{}h_sim{}.npy'.format(seq_len/(24*60), int(t/60), i), lamb_gen)
            # np.save('npy_lists/repair_window{}_dt{}h_sim{}.npy'.format(seq_len/(24*60),int(t/60), i), lamb_gen)
            # np.save('npy_lists/class_window{}_dt{}h_sim{}.npy'.format(seq_len/(24*60),int(t/60), i), lamb_gen)
            
            #Load real data for evaluation
            name_fail = 'Data/fail_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)),int(t/60))
            name_repair = 'Data/repair_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)),int(t/60))
            name_class = 'Data/class_window{}_dt{}h_real.npy'.format(int(seq_len/(24*60)),int(t/60))
            Real_data_fail = np.load(name_fail, allow_pickle=True)
            Real_data_repair = np.load(name_repair, allow_pickle=True)
            Real_data_class = np.load(name_class, allow_pickle=True)
            
            #MSE/Evaluation of real vs predicted failure rate
            len_tst_st = int(len(Real_data_fail)*0.8)
            Test_data_fail = Real_data_fail[len_tst_st:len_tst_st + len(lamb_gen)].reshape(-1,1)
            MSE_sim_f = mean_squared_error(Test_data_fail, lamb_gen) 

            #MSE/Evaluation of real vs predicted repair rate
            len_tst_st = int(len(Real_data_fail)*0.8)
            Test_data_repair = Real_data_repair[len_tst_st:len_tst_st + len(mi_gen)].reshape(-1,1)
            MSE_sim_r = mean_squared_error(Test_data_repair, mi_gen)

            #KL_divergance/Evaluation KL_divergance for mi_gen distribution
            len_tst_st = int(len(Real_data_class)*0.8)
            Test_data_class = Real_data_class[len_tst_st:len_tst_st + len(fail_distribution)]
            res =[] 
            for i in range(len(fail_distribution)): 
                kl_ = kl_divergence(fail_distribution[i], Test_data_class[i])
                res.append(kl_)
            res_ = [x for x in res if x < 1000 and x > -1000]
            KL_div = division(sum(res_),len(res_))
            
            #Append results in order 1. Failure_rate 2. Repair_rate 3. Class
            Results.extend([MSE_sim_f, MSE_sim_r, KL_div])
            print(np.array(Results).shape)
            
    end = time.time()
    print('Time:{}sec'.format(end - start)) #print time for loop[i]
                      
Results = np.array(Results)
print(Results.shape)
np.save('Results.npy', Results)
Results = Results.reshape(-1,3)
df = pd.DataFrame(Results, columns=['MSE_sim_fail', 'MSE_sim_repair', 'KL_div'])
df.to_excel("Output.xlsx", sheet_name='Rezultati')