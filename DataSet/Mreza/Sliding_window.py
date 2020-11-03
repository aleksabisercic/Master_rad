# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 23:08:41 2020

@author: Freedom
"""

import numpy as np

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []

    for i in range(len(datax) - seq_length - 1):
        _x = datax[i:(i + seq_length)]
#        _x = preprocessing.normalize(_x)
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)			

    return np.array(x), np.array(y)
mi = np.load('mi_sirovi_podatci_izbrisi.npy')
dataY = np.array(mi).reshape(-1, 1)
dataX = np.array(mi).reshape(-1, 1)
x_mi, y_mi = sliding_windows(dataX, dataY, 50)
np.save('x_mi.npy', x_mi)
np.save('y_mi.npy', y_mi)

