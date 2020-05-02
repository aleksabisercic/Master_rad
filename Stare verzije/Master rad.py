# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:08:35 2020

@author: Freedom
"""

import numpy as np
import scipy as sp

def F_btd_inv(t,L_btd,lambda_t3e, r):
    return (1+lambda_t3e*t)*np.exp(-L_btd*t)-(1-r)

L_btd = 0.047469
lambda_t3e= 0.0086504
r=np.random.uniform(low=0.0, high=1.0,size=None)
def bisection(f,a,b,N, L_btd, lambda_t3e, r):
    
    if f(a, L_btd, lambda_t3e, r)*f(b, L_btd, lambda_t3e, r) >= 0:
        print("Polovljenje intervala fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n, L_btd, lambda_t3e, r)
        if f(a_n, L_btd, lambda_t3e, r)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n, L_btd, lambda_t3e, r)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Tacno resenje.")
            return m_n
        else:
            print("Polovljenje intervala fails.")
            return None
    return (a_n + b_n)/2
y = bisection(F_btd_inv,0,300,1000, L_btd, lambda_t3e, r)
print(y)
