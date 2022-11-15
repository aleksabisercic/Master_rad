# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:32:42 2020

@author: Freedom
"""

from scipy.special import kl_div
import numpy as np
# define distribution
p = []
for i in range(10):
    p.append([i/10, 0.40, 0.50]) 

q = []
for i in range(10):
    q.append([0.80, 0.15, 0.05]) 

# calculate (P || Q)
res =[] 
for i in  range(len(p)):
    kl_pq = kl_div(p[i], q[i])
    res.append(sum(kl_pq))

print(np.mean(res))
