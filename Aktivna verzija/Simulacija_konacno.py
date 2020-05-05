import numpy as np
from Jedna_simulacija import Simulacija 
import xlwt as xl
import matplotlib.pyplot as plt

wb = xl.Workbook()
ws1 = wb.add_sheet("Bager statistika")
ws2 = wb.add_sheet("Drobilana statistika")
ws3 = wb.add_sheet("T1 statistika")
ws4 = wb.add_sheet("T2 statistika")
ws5 = wb.add_sheet("T3 statistika")
ws6 = wb.add_sheet("T statistika")
ws7 = wb.add_sheet("BTD statistika")
broj_simulacija = 1
vreme_otk = []
vreme_pop = []

ws1_kolone = ["Broj simulacije", "SrVrRb" , "SrVrCnRb" , "SrVrRADb" , "SrVrOEb" , 
                "SrVrOMb" , "SrVrOOb" , "brOTKb" , "SrVrOTKb" , "Ab" , "Aeb" , "Aob" , "Amb"]

k = 0
for i in ws1_kolone: 
    ws1.row(0).write(k,i)
    k+=1   

for i in range(broj_simulacija):
    print(i)
    vremena_otkaza, vremena_popravke, list_stat_B, list_stat_D, list_stat_T1, list_stat_T2, list_stat_T3, list_stat_T, list_stat_BTD = Simulacija()
    vreme_otk.append(vremena_otkaza)
    vreme_pop.append(vremena_popravke)

	#BAGER
    ws1.row(i+1).write(0,i)
    ws1.row(i+1).write(1, list_stat_B[0])
    ws1.row(i+1).write(2, list_stat_B[1])
    ws1.row(i+1).write(3, list_stat_B[2])
    ws1.row(i+1).write(4, list_stat_B[3])
    ws1.row(i+1).write(5, list_stat_B[4])
    ws1.row(i+1).write(6, list_stat_B[5])
    ws1.row(i+1).write(7, list_stat_B[6])
    ws1.row(i+1).write(8, list_stat_B[7])
    ws1.row(i+1).write(9, list_stat_B[8])
    ws1.row(i+1).write(10, list_stat_B[9])
    ws1.row(i+1).write(11, list_stat_B[10])
    ws1.row(i+1).write(12, list_stat_B[11])
    
    #T1
    ws3.row(i+1).write(0,i)
    ws3.row(i+1).write(1, list_stat_T1[0])
    ws3.row(i+1).write(2, list_stat_T1[1])
    ws3.row(i+1).write(3, list_stat_T1[2])
    ws3.row(i+1).write(4, list_stat_T1[3])
    ws3.row(i+1).write(5, list_stat_T1[4])
    ws3.row(i+1).write(6, list_stat_T1[5])
    ws3.row(i+1).write(7, list_stat_T1[6])
    ws3.row(i+1).write(8, list_stat_T1[7])
    ws3.row(i+1).write(9, list_stat_T1[8])
    ws3.row(i+1).write(10, list_stat_T1[9])
    ws3.row(i+1).write(11, list_stat_T1[10])
    ws3.row(i+1).write(12, list_stat_T1[11])
    
	#T2
    ws4.row(i+1).write(0,i)
    ws4.row(i+1).write(1, list_stat_T2[0])
    ws4.row(i+1).write(2, list_stat_T2[1])
    ws4.row(i+1).write(3, list_stat_T2[2])
    ws4.row(i+1).write(4, list_stat_T2[3])
    ws4.row(i+1).write(5, list_stat_T2[4])
    ws4.row(i+1).write(6, list_stat_T2[5])
    ws4.row(i+1).write(7, list_stat_T2[6])
    ws4.row(i+1).write(8, list_stat_T2[7])
    ws4.row(i+1).write(9, list_stat_T2[8])
    ws4.row(i+1).write(10, list_stat_T2[9])
    ws4.row(i+1).write(11, list_stat_T2[10])
    ws4.row(i+1).write(12, list_stat_T2[11])
    
	#T3
    ws5.row(i+1).write(0,i)
    ws5.row(i+1).write(1, list_stat_T3[0])
    ws5.row(i+1).write(2, list_stat_T3[1])
    ws5.row(i+1).write(3, list_stat_T3[2])
    ws5.row(i+1).write(4, list_stat_T3[3])
    ws5.row(i+1).write(5, list_stat_T3[4])
    ws5.row(i+1).write(6, list_stat_T3[5])
    ws5.row(i+1).write(7, list_stat_T3[6])
    ws5.row(i+1).write(8, list_stat_T3[7])
    ws5.row(i+1).write(9, list_stat_T3[8])
    ws5.row(i+1).write(10, list_stat_T3[9])
    ws5.row(i+1).write(11, list_stat_T3[10])
    ws5.row(i+1).write(12, list_stat_T3[11])
    
	#T
    ws6.row(i+1).write(0,i)
    ws6.row(i+1).write(1, list_stat_T[0])
    ws6.row(i+1).write(2, list_stat_T[1])
    ws6.row(i+1).write(3, list_stat_T[2])
    ws6.row(i+1).write(4, list_stat_T[3])
    ws6.row(i+1).write(5, list_stat_T[4])
    
    #D
    ws2.row(i+1).write(0,i)
    ws2.row(i+1).write(1, list_stat_D[0])
    ws2.row(i+1).write(2, list_stat_D[1])
    ws2.row(i+1).write(3, list_stat_D[2])
    ws2.row(i+1).write(4, list_stat_D[3])
    ws2.row(i+1).write(5, list_stat_D[4])
    ws2.row(i+1).write(6, list_stat_D[5])
    ws2.row(i+1).write(7, list_stat_D[6])
    ws2.row(i+1).write(8, list_stat_D[7])
    ws2.row(i+1).write(9, list_stat_D[8])
    ws2.row(i+1).write(10, list_stat_D[9])
    
	#BTD
    ws7.row(i+1).write(0,i)
    ws7.row(i+1).write(1, list_stat_T2[0])
    ws7.row(i+1).write(2, list_stat_T2[1])
    ws7.row(i+1).write(3, list_stat_T2[2])
	
    
    #x=np.arange(31536000)
    #y1=[]
    #for t in range(x)
	#	if t >= vreme_otkaza[br] and t < vreme_popravke[br]:
	#			 y = 0
	#	else:
	#		 	 y=1
	#	y1.append(y)
	#	if t == vreme_popravke[br]:
	#		 br += 1	 
				 		 
    y=np.asarray(y)
    fig,ax = plt.subplots()
    ax.plot(x, y)
	
	
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
	       title='About as simple as it gets, folks')
    ax.grid()
	
    plt.show()

wb.save("Rezultati.xls")
