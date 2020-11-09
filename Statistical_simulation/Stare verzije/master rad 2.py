# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:15:58 2020

@author: Freedom
"""

import numpy as np
import scipy as sp
from scipy.optimize import fsolve

def F_btd_inv(t,L_btd,lambda_t3e, r):
    return (1+lambda_t3e*t)*np.exp(-L_btd*t)-(1-r)

L_btd = 0.047469/2
lambda_t3e= 0.0086504
r=np.random.uniform(low=0.0, high=1.0,size=None)
tp0 = fsolve(F_btd_inv,0,args = (L_btd, lambda_t3e, r))
print(tp0)
