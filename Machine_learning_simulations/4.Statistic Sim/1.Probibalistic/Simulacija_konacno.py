import numpy as np
from Jedna_simulacija_BT_D import Simulacija_BT_D 
from Jedna_simulacija import Simulacija 
import xlwt as xl
import pickle



def Simulacija_konacno(name = "BTD" , broj_simulacija = 2):

    wb = xl.Workbook()
    ws1 = wb.add_sheet("Bager statistika")
    ws2 = wb.add_sheet("Drobilana statistika")
    ws3 = wb.add_sheet("T1 statistika")
    ws4 = wb.add_sheet("T2 statistika")
    ws5 = wb.add_sheet("T3 statistika")
    ws6 = wb.add_sheet("T statistika")
    ws7 = wb.add_sheet("BTD statistika")
    ws8 = wb.add_sheet("BT statistika")

    vreme_otk = []
    vreme_pop = []
    types_of_failures = []

    ws1_kolone = ["Broj simulacije","brCnRb", "SrVrRb" , "SrVrCnRb" , "SrVrRADb" , "SrVrOEb" , 
                    "SrVrOMb" , "SrVrOOb" , "brOTKb" , "SrVrOTKb" , "Ab" , "Aeb" , "Aob" , "Amb"]
    ws2_kolone = ["Broj simulacije", "brCnRd", "SrVrRd" , "SrVrCnRd" , "SrVrRADd" , "SrVrOOd" , "brOTKd" , "SrVrOTKd" ,  "SrVrOOd" , "SrVrOTKd", "Ad" , "Aod"]

    ws3_kolone = ["Broj simulacije", "brCnRt1","SrVrRt1" , "SrVrCnRt1" , "SrVrRADt1" , "SrVrOEt1" , "SrVrOMt1" , "SrVrOOt1" ,"brOTKt1","SrVrOTKt1", "At1", "Aet1" , "Aot1" , "Amt1"]
    ws4_kolone = ["Broj simulacije", "brCnRt2","SrVrRt2"," SrVrCnRt2", "SrVrRADt2", "SrVrOEt2", "SrVrOMt2", "SrVrOOt2", "brOTKt2", "SrVrOTKt2", "At2", "Aet2", "Aot2", "Amt2"]
    ws5_kolone = ["Broj simulacije", "brCnRt3","SrVrRt3"," SrVrCnRt3", "SrVrRADt3", "SrVrOEt3", "SrVrOMt3", "SrVrOOt3", "brOTKt3", "SrVrOTKt3", "At3", "Aet3", "Aot3", "Amt3"]
    ws6_kolone = ["Broj simulacije", "SrVrRt" , "SrVrCnRt" , "SrVrRADt" , "SrVrOTKt" , "At"]
    ws7_kolone = ["Broj simulacije", "VrRbtd" , "SrVrRbtd" , "Abtd"]
    ws8_kolone = ["Broj simulacije", "VrRbt" , "SrVrRbt" , "Abt"]


    k = 0
    for i in ws1_kolone: 
        ws1.row(0).write(k,i)
        k+=1   

    k = 0    
    for i in ws2_kolone: 
        ws2.row(0).write(k,i)
        k+=1

    k = 0
    for i in ws3_kolone: 
        ws3.row(0).write(k,i)
        k+=1

    k = 0	
    for i in ws4_kolone: 
        ws4.row(0).write(k,i)
        k+=1	

    k=0
    for i in ws5_kolone: 
        ws5.row(0).write(k,i)
        k+=1

    k = 0	
    for i in ws6_kolone: 
        ws6.row(0).write(k,i)
        k+=1

    k = 0    
    for i in ws7_kolone: 
        ws7.row(0).write(k,i)
        k+=1

    k = 0    
    for i in ws8_kolone: 
        ws8.row(0).write(k,i)
        k+=1


    for i in range(broj_simulacija):
        print(i+1)
        if name == 'BTD':
            vremena_otkaza, vremena_popravke, vrste_otkaza, list_stat_B, list_stat_D, list_stat_T1, list_stat_T2, list_stat_T3, list_stat_T, list_stat_BTD, list_stat_BT  = Simulacija()
        elif name == "BT":
            vremena_otkaza, vremena_popravke, vrste_otkaza, list_stat_B, list_stat_D, list_stat_T1, list_stat_T2, list_stat_T3, list_stat_T, list_stat_BTD, list_stat_BT  = Simulacija_BT_D()

        vreme_otk.append(vremena_otkaza)
        vreme_pop.append(vremena_popravke)
        types_of_failures.append(vrste_otkaza)
        
        #BAGER
        ws1.row(i+1).write(0,i+1)
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
        ws1.row(i+1).write(13, list_stat_B[12])
        
        #T1
        ws3.row(i+1).write(0,i+1)
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
        ws3.row(i+1).write(13, list_stat_T1[12])
        
        #T2
        ws4.row(i+1).write(0,i+1)
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
        ws4.row(i+1).write(13, list_stat_T2[12])
        
        #T3
        ws5.row(i+1).write(0,i+1)
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
        ws5.row(i+1).write(13, list_stat_T3[12])    
        #T
        ws6.row(i+1).write(0,i+1)
        ws6.row(i+1).write(1, list_stat_T[0])
        ws6.row(i+1).write(2, list_stat_T[1])
        ws6.row(i+1).write(3, list_stat_T[2])
        ws6.row(i+1).write(4, list_stat_T[3])
        ws6.row(i+1).write(5, list_stat_T[4])
        
        #D
        ws2.row(i+1).write(0,i+1)
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
        ws2.row(i+1).write(11, list_stat_D[10])	
        
        #BTD
        ws7.row(i+1).write(0,i+1)
        ws7.row(i+1).write(1, list_stat_BTD[0])
        ws7.row(i+1).write(2, list_stat_BTD[1])
        ws7.row(i+1).write(3, list_stat_BTD[2])
        
        #BT
        ws8.row(i+1).write(0,i+1)
        ws8.row(i+1).write(1, list_stat_BT[0])
        ws8.row(i+1).write(2, list_stat_BT[1])
        ws8.row(i+1).write(3, list_stat_BT[2])

    with open('Rezultati/vremena_otkaza_{}.pkl'.format(name), 'wb') as f3:
        pickle.dump(vreme_otk, f3)
    with open('Rezultati/vremena_popravki_{}.pkl'.format(name), 'wb') as f4:
        pickle.dump(vreme_pop, f4)

    np.save('lista_vremena_otkz_{}_{}.npy'.format(broj_simulacija, name), vreme_otk)
    np.save('lista_vremena_pop_{}_{}.npy'.format(broj_simulacija, name), vreme_pop)
    np.save('lista_vrsta_pop_{}_{}.npy'.format(broj_simulacija, name), types_of_failures)
    
    wb.save("Rezultati/rezultati_{}_{}.xls".format(broj_simulacija, name))


