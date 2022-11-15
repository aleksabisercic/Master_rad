# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:35:39 2020

@author: Freedom
"""

import numpy as np
import matplotlib.pyplot as plt

a1 = np.load('fail_window15_dt8h_real.npy')[:2890]
print(a1.shape)

class3 = np.load('class_window15_dt8h_real.npy')[:2890]
print(type(class3))

a2 = np.load('repair_window15_dt8h_real.npy')
print(a2.shape)

plt.subplot(2, 1, 1)
plt.plot(a1)
plt.subplot(2,1,2)
plt.plot(a2)
plt.show()

