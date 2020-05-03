import numpy as np
import scipy as sp
from Jedna_simulacija import Simulacija 

broj_simulacija = 1
vreme_otk = []
vreme_pop = []

for i in range(broj_simulacija):
    print(i)
    vremena_otkaza, vremena_popravke, list_stat_B, list_stat_D, list_stat_T, list_stat_BTD = Simulacija()
    vreme_otk.append(vremena_otkaza)
    vreme_pop.append(vremena_popravke)


