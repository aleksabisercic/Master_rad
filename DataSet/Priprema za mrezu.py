# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:19:43 2020

@author: Freedom
"""

import pandas as pd
import numpy as np
import datetime



df1 = pd.read_excel('Zastoji.xlsx', header = 0 )
data = df1['Pocetak zastoja u minutima','Kraj zastoja u miutama' ].values.astype(float)
print(data)

  