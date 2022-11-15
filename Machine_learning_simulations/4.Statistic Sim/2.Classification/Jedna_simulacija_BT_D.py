# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:46:46 2020

@author: Freedom
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
""" FUNKCIJE KOJE SE KORISTE U OKVIRU PROCEDURE """

def Simulacija_BT_D(): 

    def F_btd_inv(t,L_btd,lambda_t3e, r):
        return (1+lambda_t3e*t)*np.exp(-L_btd*t)-(1-r)

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

    def podeli(x,y):
        try:
            return x/y
        except ZeroDivisionError:
            return 0



    """ ZADATI PRE STARTA PROGRAMA """

    Bisection_izbor = False
    vreme_ispisa = 5000

    """ PARAMETRI KOJI SE NE MENJAJU """ 
    
    #Repair and Faliur rates in minutes
    Lbe=0.0000669323 
    Lbm=0.0000433066
    Lbo=0.000073458
    Ldo=0.0000588205
    Lt1e=0.0000680535
    Lt1m=0.0000531802
    Lt1=0.000121234
    Lt2e=0.0000564661
    Lt2m=0.000059579
    Lt2=0.000116045
    lambda_t3e=0.000144174
    Lt3m=0.0000758069
    Ltxo=0.0000913767
    mbe=0.00027154
    mbm=0.000474394
    mbo=0.0000731771
    mb=0.000323161
    mdo=0.000226708
    mt1e=0.000393955
    mt1m=0.000817231
    mt2e=0.000445557
    mt2m=0.00090446
    mt3e=0.000534383
    mt3m=0.000627307
    mtxo=0.000385339
    Lb=0.000183697
    Ld=0.0000588205
    Ltx=0.000548637
    L_btd=0.000791154
    Pvo1=0.18988903
    Pvo2=0.36405672
    Pvo3=0.446054254
    Peo1=0.42532468 
    Peo2=0.02435065
    Peo3=0.1461039
    Peo4=0.40422078
    Pmo1=0.303979678238781
    Pmo2=0.191363251481795
    Pmo3=0.171041490262489
    Pmo4=0.333615580016935
    Poo1=0.111264685556323
    Poo2=0.840359364201797
    Poo3=0.0483759502418798
    Ptt1=0.01428571
    Ptt2=0.257142857
    Ptt3=0.728571429

    """ INICIJALNE VREDNOSTI PARAMETARA """

    VrRb=0
    VrCnRb=0
    VrRADb=0
    VrOEb=0
    brOEb=0
    VrOMb=0
    brOMb=0
    VrOOb=0
    brOOb=0
    VrOTKb=0
    brOTKb=0
    VrRd=0
    VrCnRd=0
    VrRADd=0
    VrOOd=0
    brOOd=0
    VrOTKd=0
    brOTKd=0
    VrRt=0
    VrCnRt=0
    VrRADt=0
    VrOTKt=0
    brOTKt=0
    VrRt1=0
    VrCnRt1=0
    VrRADt1=0
    VrOEt1=0
    brOEt1=0
    VrOMt1=0
    brOMt1=0
    VrOOt1=0
    brOOt1=0
    VrOTKt1=0
    brOTKt1=0
    VrRt2=0
    VrCnRt2=0
    VrRADt2=0
    VrOEt2=0
    brOEt2=0
    VrOMt2=0
    brOMt2=0
    VrOOt2=0
    brOOt2=0
    VrOTKt2=0
    brOTKt2=0
    VrRt3=0
    VrCnRt3=0
    VrRADt3=0
    VrOEt3=0
    brOEt3=0
    VrOMt3=0
    brOMt3=0
    VrOOt3=0
    brOOt3=0
    VrOTKt3=0
    brOTKt3=0
    topr = 0
    VrRbtd = 0 
    VrRbt = 0
    brCnRb=0
    brCnRt1=0
    brCnRt2=0
    brCnRt3=0
	
    STbtd="11"
    STb="11"
    STd="11"
    STt="11"
    STt1="11"
    STt2="11"
    STt3="11"
    brR=0

    vreme_simulacije = 259200 # len of prediction in minutes ( 6 month )
    vremena_otkaza = [] #time of failuer
    vremena_popravke = [] #time of repair
    vrsta_otakaza = [] #type of faiuler where 0 is 'mechanical', 1 is 'electrical' and 2 is 'other'
    """ POCETAK PROGRAMA """ 

    r_vremena_otkaza=np.random.uniform(low=0.0, high=1.0,size=None)

    if Bisection_izbor == True: 
        tp0 = bisection(F_btd_inv,0,108000,1000, L_btd, lambda_t3e, r_vremena_otkaza)
        tp0 = np.round(tp0)
    else:
        tp0 = fsolve(F_btd_inv,0,args = (L_btd, lambda_t3e, r_vremena_otkaza))
        tp0 = np.round(tp0)
    vremena_otkaza.append(tp0)

    for t in range(1,vreme_simulacije):   
        if t == tp0:
            
            brR = brR + 1
            r = np.random.uniform(low=0.0, high=1.0,size=None)

            if r < Pvo1:
                VRotk="1"
                vrsta_otakaza.append(1)
                r1=np.random.uniform(low=0.0, high=1.0,size=None)
                
                if r1 < Peo1:
                    OBJeo = "1"
                    brOEb = brOEb + 1
                    STb = "21"
                    r11 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mbe)*np.log(r11)
                    DT = np.round(DT)
                    topr = t+DT
                    STd = "12"
                    STt1 = "12"
                    STt2 = "12"
                    STt3 = "12"
                
                elif r1 < (Peo1+Peo2):
                    OBJeo = "2"
                    brOEt1 = brOEt1 + 1
                    STt1 = "21"
                    r12 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mt1e)*np.log(r12)
                    DT = np.round(DT)
                    topr = t+DT
                    STd = "12"
                    STb = "12"
                    STt2 = "12"
                    STt3 = "12"
                elif r1 < (Peo1+Peo2+Peo3):
                    OBJeo = "3"
                    brOEt2 = brOEt2 + 1
                    STt2 = "21"
                    r13 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT=-(1/mt2e)*np.log(r13)
                    DT = np.round(DT)
                    topr = t+DT
                    STd = "12"
                    STt1 = "12"
                    STb = "12"
                    STt3 = "12"
                elif r1 <= 1: #<=1
                    OBJeo = "4"
                    brOEt3 = brOEt3 + 1
                    STt3 = "21"
                    r14 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mt3e)*np.log(r14)
                    DT = np.round(DT)
                    topr = t+DT
                    STd = "12"
                    STt1 = "12"
                    STt2 = "12"
                    STb = "12"
                    
            elif r < (Pvo1+Pvo2):
                Vrotk="2"
                vrsta_otakaza.append(0)
                r2=np.random.uniform(low=0.0, high=1.0,size=None)
                if r2<Pmo1:
                    OBJmo = "1"
                    brOMb = brOMb + 1
                    STb="22"
                    r21=np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mbm)*np.log(r21)
                    DT = np.round(DT)
                    topr = t + DT
                    STd="12"
                    STt1="12"
                    STt2="12"
                    STt3="12"
                elif r2 < (Pmo1+Pmo2):
                    OBJmo = "2"
                    brOMt1 = brOMt1 +1
                    STt1 = "22"
                    r22=np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mt1m)*np.log(r22)
                    DT = np.round(DT)
                    topr = t+DT
                    STd = "12"
                    STb = "12"
                    STt2 = "12"
                    STt3 = "12"
                elif r2 < (Pmo1+Pmo2+Pmo3):
                    OBJmo = "3"
                    brOMt2 = brOMt2 + 1
                    STt2 = "22"
                    r23 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mt2m)*np.log(r23)
                    DT = np.round(DT)
                    topr = t + DT
                    STd = "12"
                    STt1 = "12"
                    STb = "12"
                    STt3 = "12"
                elif r2 <= (1):
                    OBJmo = "4"
                    brOMt3 = brOMt3 + 1
                    STt3 = "22"
                    r24 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mt3m)*np.log(r24)
                    DT = np.round(DT)
                    topr = t + DT
                    STd = "12"
                    STt1 = "12"
                    STt2 = "12"
                    STb = "12"
            elif r <= 1: 
                Vrotk = "3"
                vrsta_otakaza.append(2)
                r3=np.random.uniform(low=0.0, high=1.0,size=None)
                if r3 < Poo1:
                    OBJoo = "1"
                    brOOb = brOOb + 1
                    STb = "23"
                    r31 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mbo)*np.log(r31)
                    DT = np.round(DT)
                    topr = t + DT
                    STd = "12"
                    STt1 = "12"
                    STt2 = "12"
                    STt3 = "12"

                elif r3 < (Poo1+Poo2):
                    OBJoo = "2"
                    brOOd = brOOd + 1
                    STd = "23"
                    r32 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mdo)*np.log(r32)
                    DT = np.round(DT)
                    topr = t + DT


                elif r3 <= (1):
                    OBJoo = "3"
                    r4 = np.random.uniform(low=0.0, high=1.0,size=None)

                    if r4 < Ptt1:
                        OBJtt = "1"
                        brOOt1 = brOOt1 + 1
                        STt1 = "23"
                        STd = "12"
                        STb = "12"
                        STt2 = "12"
                        STt3 = "12"

                    elif r4 < (Ptt1+Ptt2):
                        OBJtt = "2"
                        brOOt2 = brOOt2 + 1
                        STt2 = "23"
                        STt1 = "12"
                        STd = "12"
                        STb = "12"
                        STt3 = "12"

                    elif r4 <= 1:
                        OBJtt ="3"
                        brOOt3 = brOOt3 + 1
                        STt3 = "23"
                        STt1 = "12"
                        STd = "12"
                        STt2 = "12"
                        STb = "12"
                
                    r41 = np.random.uniform(low=0.0, high=1.0,size=None)
                    DT = -(1/mtxo)*np.log(r41) 
                    DT = np.round(DT)
                    topr = t + DT

        if t==topr:
            vremena_popravke.append(topr)   
            STd = "11"
            STt1 = "11"
            STt2 = "11"
            STt3 = "11"
            STb = "11"
            STbtd ="11"

            if Bisection_izbor == True: 
                tp0 = bisection(F_btd_inv,0,108000,1000, L_btd, lambda_t3e, r_vremena_otkaza)
            else:
                tp0 = fsolve(F_btd_inv,0,args = (L_btd, lambda_t3e, r_vremena_otkaza))
            
            tp0 = np.round(tp0)
            tp0 = t + tp0
            vremena_otkaza.append(tp0)
        
		
        if STb == '11':
            VrRb = VrRb + 1
            VrRADb = VrRADb + 1
            VrRbt = VrRbt + 1
        
        if STd == "11":
            VrRd = VrRd + 1
            VrRADd = VrRADd + 1

        if STt1 == "11":
            VrRt1 = VrRt1 + 1
            VrRADt1 = VrRADt1 + 1
        
        if STt2 == "11":
            VrRt2 = VrRt2 + 1
            VrRADt2 = VrRADt2 + 1
        
        if STt3 == "11":
            VrRt3 = VrRt3 + 1
            VrRADt3 = VrRADt3 + 1
	
        if STb == "12":  
            VrCnRb = VrCnRb + 1
            VrRADb = VrRADb + 1
        if STb=="21":
            VrOEb = VrOEb + 1
            VrOTKb = VrOTKb + 1
        if STb == "22":
            VrOMb = VrOMb + 1
            VrOTKb = VrOTKb + 1
        if STb == "23":
            VrOOb = VrOOb + 1
            VrOTKb = VrOTKb + 1
        if STt1 == "12":  
            VrCnRt1 = VrCnRt1 + 1
            VrRADt1 = VrRADt1 + 1
        if STt1 == "21":
            VrOEt1 = VrOEt1 + 1
            VrOTKt1 = VrOTKt1 + 1
        if STt1 == "22":
            VrOMt1 = VrOMt1 + 1
            VrOTKt1 = VrOTKt1 + 1
        if STb == "23":
            VrOOt1 = VrOOt1 + 1
            VrOTKt1 = VrOTKt1 + 1
        if STt2 == "12":  
            VrCnRt2 = VrCnRt2 + 1
            VrRADt2 = VrRADt2 + 1
        if STt2 == "21":
            VrOEt2 = VrOEt2 + 1
            VrOTKt2 = VrOTKt2 + 1
        if STt2 == "22":
            VrOMt2 = VrOMt2 + 1
            VrOTKt2 = VrOTKt2 + 1
        if STt2 == "23":
            VrOOt2 = VrOOt2 + 1
            VrOTKt2 = VrOTKt2 + 1
        if STt3 == "12":  
            VrCnRt3 = VrCnRt3 + 1
            VrRADt3 = VrRADt3 + 1 
        if STt3 == "21":
            VrOEt3 = VrOEt3 + 1
            VrOTKt3 = VrOTKt3 + 1
        if STt3 == "22":
            VrOMt3 = VrOMt3 + 1
            VrOTKt3 = VrOTKt3 + 1
        if STt3 == "23":
            VrOOt3 = VrOOt3 + 1
            VrOTKt3 = VrOTKt3 + 1
        if STd == "12":  
            VrCnRd = VrCnRd + 1
            VrRADd = VrRADd + 1
        if STd == "23":
            VrOOd = VrOOd + 1
            VrOTKd = VrOTKd + 1
        if STb == "11" and  STd == "11" and STt1 == "11" and  STt1 == "11" and STt2 == "11" and STt3 == "11":
            VrRbtd = VrRbtd + 1
		
        # if t%vreme_ispisa == 0:
        #     print(t,tp0,topr)


    """ PROVERA """
    if ( VrRt1 == VrRt2 and VrRt1 == VrRt3 ):
        VrRt = VrRt1
        VrCnRt = VrOEb + VrOMb + VrOOb 
        brCnRt = brOEb + brOMb + brOOb  
        VrRADt = VrRt + VrCnRt
        VrOTKt = VrOTKt1 + VrOTKt2 + VrOTKt3
        brOTKt = brOEt1 + brOMt1 + brOOt1 + brOEt2 + brOMt2 + brOOt2 + brOEt3 + brOMt3 + brOOt3
        brCnRb = brOTKt
        brCnRd = brOEb + brOMb + brOOb + brOTKt
    else:
        print("Greska_Kraj")
    
    """ RACUNANJE STATISTIKA ZA JEDNU SIMULACIJU """
    #bager statistika

    SrVrRb = podeli(VrRb,(brR-brOOd)) 
    SrVrCnRb = podeli(VrCnRb, brCnRb)
    SrVrRADb = podeli(VrRADb, ( (brR-brOOd) + brCnRb ))
    SrVrOEb = podeli(VrOEb, brOEb)
    SrVrOMb = podeli(VrOMb,brOMb)
    SrVrOOb = podeli(VrOOb,brOOb)
    brOTKb = brOEb + brOMb + brOOb
    SrVrOTKb = podeli(VrOTKb, brOTKb)
    Ab = VrRADb / ( VrRADb + VrOTKb )
    Aeb = VrRADb / ( VrRADb + VrOEb )
    Amb = VrRADb / ( VrRADb + VrOMb )
    Aob = VrRADb / ( VrRADb + VrOOb )

#     #drobilica statistika

    SrVrRd = podeli(VrRd, brR) 
    SrVrCnRd = podeli(VrCnRd, brCnRd)
    SrVrRADd = podeli(VrRADd, ( brR + brCnRd ))
    SrVrOOd = podeli(VrOOd, brOOd)
    brOTKd =  brOOd
    SrVrOTKd = podeli(VrOTKd, brOTKd)
    SrVrOOd = podeli(VrOOd, brOOd)
    SrVrOTKd = SrVrOOd
    Ad = podeli( VrRADd , ( VrRADd + VrOTKd ) )
    Aod = podeli( VrRADd , ( VrRADd + VrOOd ) )

#     #Trakasti transporteri statistika
    SrVrRt = podeli(VrRt, (brR-brOOd))
    SrVrCnRt = podeli(VrCnRt, brCnRt)
    SrVrRADt = podeli(VrRADt, ( (brR-brOOd) + brCnRt ))
    SrVrOTKt = podeli(VrOTKt, brOTKt)
    At = podeli( VrRADt , ( VrRADt + VrOTKt ) )

#     #Trakasti transporter 1 stat.
    brCnRt1 = brOTKb + brOEt2 + brOMt2 + brOOt2 + brOEt3 + brOMt3 + brOOt3
    SrVrRt1 = podeli(VrRt1, (brR-brOOd)) 
    SrVrCnRt1 = podeli(VrCnRt1, brCnRt1)
    SrVrRADt1 = podeli(VrRADt1, ( (brR-brOOd) + brCnRt1 ))
    SrVrOEt1 = podeli(VrOEt1, brOEt1) 
    SrVrOMt1 = podeli(VrOMt1, brOMt1)
    SrVrOOt1 = podeli(VrOOt1, brOOt1)
    brOTKt1 = brOEt1 + brOMt1 + brOOt1
    SrVrOTKt1 = podeli(VrOTKt1, brOTKt1)
    At1 = podeli( VrRADt1 , ( VrRADt1 + VrOTKt1 ) )
    Aet1 = podeli( VrRADt1 , ( VrRADt1 + VrOEt1 ) )
    Amt1 = podeli( VrRADt1 , ( VrRADt1 + VrOMt1 ) )
    Aot1 = podeli( VrRADt1 , ( VrRADt1 + VrOOt1 ) )

    #Trakasti transporter 2 stat.

    brCnRt2 = brOTKb  + brOEt1 + brOMt1 + brOOt1 + brOEt3 + brOMt3 + brOOt3
    SrVrRt2 = podeli(VrRt2, (brR-brOOd)) 
    SrVrCnRt2 = podeli(VrCnRt2, brCnRt2)
    SrVrRADt2 = podeli(VrRADt2, ( (brR-brOOd) + brCnRt2 ))
    SrVrOEt2 = podeli(VrOEt2, brOEt2)
    SrVrOMt2 = podeli(VrOMt2, brOMt2)
    SrVrOOt2 = podeli(VrOOt2, brOOt2)
    brOTKt2 = brOEt2 + brOMt2 + brOOt2
    SrVrOTKt2 = podeli(VrOTKt2, brOTKt2)
    At2 = podeli(VrRADt2 , ( VrRADt2 + VrOTKt2 ))
    Aet2 = podeli(VrRADt2, ( VrRADt2 + VrOEt2 ))
    Amt2 = podeli(VrRADt2 , ( VrRADt2 + VrOMt2 ) )
    Aot2 = podeli( VrRADt2 , ( VrRADt2 + VrOOt2 ) )
    
    #Trakasti transporter 2 stat.

    brCnRt3 = brOTKb + brOEt1 + brOMt1 + brOOt1 + brOEt2 + brOMt2 + brOOt2
    SrVrRt3 = podeli(VrRt3 , (brR-brOOd) )
    SrVrCnRt3 = podeli(VrCnRt3, brCnRt3)
    SrVrRADt3 = podeli(VrRADt3, ( (brR-brOOd) + brCnRt3 ))
    SrVrOEt3 = podeli(VrOEt3, brOEt3)
    SrVrOMt3 = podeli(VrOMt3, brOMt3)
    SrVrOOt3 = podeli(VrOOt3, brOOt3)
    brOTKt3 = brOEt3 + brOMt3 + brOOt3
    SrVrOTKt3 = podeli(VrOTKt3, brOTKt3)
    At3 = podeli( VrRADt3 , ( VrRADt3 + VrOTKt3 ) )
    Aet3 = podeli( VrRADt3 , ( VrRADt3 + VrOEt3 ) )
    Amt3 = podeli( VrRADt3 , ( VrRADt3 + VrOMt3 ) )
    Aot3 = podeli( VrRADt3 , ( VrRADt3 + VrOOt3 ) )

    #Statistika btd_sistema
    
    SrVrRbtd = podeli(VrRbtd, brR)
    Abtd = VrRbtd / ( VrRbtd + VrOTKb + VrOTKd + VrOTKt)
	
	#Statistika bt_sistema
    SrVrRbt = podeli(VrRbt, (brR-brOOd))
    Abt = VrRbt / ( VrRbt + VrOTKb + VrOTKt)
	

    list_stat_B = [brCnRb, SrVrRb , SrVrCnRb , SrVrRADb , SrVrOEb , SrVrOMb , SrVrOOb , brOTKb , SrVrOTKb , Ab , Aeb , Aob , Amb]
    list_stat_D = [brCnRb, SrVrRd , SrVrCnRd , SrVrRADd , SrVrOOd , brOTKd , SrVrOTKd ,  SrVrOOd , SrVrOTKd, Ad , Aod]
    list_stat_T1 = [brCnRt1, SrVrRt1 , SrVrCnRt1 , SrVrRADt1 , SrVrOEt1 , SrVrOMt1 , SrVrOOt1 , brOTKt1 , SrVrOTKt1,  At1, Aet1 , Aot1 , Amt1]
    list_stat_T2 = [brCnRt2, SrVrRt2, SrVrCnRt2, SrVrRADt2, SrVrOEt2, SrVrOMt2, SrVrOOt2, brOTKt2, SrVrOTKt2, At2, Aet2, Aot2, Amt2]
    list_stat_T3 = [brCnRt3, SrVrRt3 , SrVrCnRt3 , SrVrRADt3 , SrVrOEt3 , SrVrOMt3 , SrVrOOt3 , brOTKt3 , SrVrOTKt3 , At3 , Aet3 , Aot3 , Amt3]
    list_stat_T = [SrVrRt , SrVrCnRt , SrVrRADt , SrVrOTKt , At]
    list_stat_BTD = [VrRbtd , SrVrRbtd , Abtd]
    lista_stat_BT = [VrRbt , SrVrRbt , Abt]
	
    vremena_otkaza = np.asarray(vremena_otkaza)
    vremena_popravke = np.asarray(vremena_popravke)
    vrsta_otakaza = np.array(vrsta_otakaza)
    # np.save('vremena_otkaza.npy',vremena_otkaza)
    # np.save('vremena_popravke.npy',vremena_popravke)
    # np.save('vrsta_otkaza.npy', vrsta_otakaza)
    return vremena_otkaza, vremena_popravke, vrsta_otakaza, list_stat_B, list_stat_D, list_stat_T1, list_stat_T2, list_stat_T3, list_stat_T, list_stat_BTD, lista_stat_BT

if __name__ == "__main__":
    Simulacija_BT_D()
# vremena_otkaza, vremena_popravke, vrsta_otakaza, list_stat_B, list_stat_D, list_stat_T1, list_stat_T2, list_stat_T3, list_stat_T, list_stat_BTD, lista_stat_BT = Simulacija_BT_D()    
# vreme_simulacije = 259200 # len of prediction in minutes ( 6 month )
# podatci1 = vremena_otkaza.reshape(-1)
# podatci2 = vremena_popravke.reshape(-1)
# podatci3 = vrsta_otakaza.reshape(-1)

# def gen_lambda_and_mi(podatci1,podatci2, seq_len, t):
#     matrix = np.zeros(vreme_simulacije)
#     for i in podatci1:
#         matrix[i] = 1        
#     matrix1 = np.zeros(vreme_simulacije)
#     for i in podatci2:
#         matrix1[i] = 1  
#     lambd = []
#     mi = []    
#     start = 0
#     end = seq_len
#     for i in range(int((len(matrix)-seq_len)/t)):
#         ls = matrix[start:end]
#         lambd.append(sum(ls))
#         ls1 = matrix1[start:end]
#         mi.append(sum(ls1))
#         start += t
#         end += t
#     lambd.append(sum(matrix[-seq_len:]))
#     mi.append(sum(matrix1[-seq_len:]))
#     return lambd, mi
        
# seq_leng = [15*24*60, 7*24*60, 30*24*60]
# dt = [15, 30, 60]

# for seq_len in seq_leng:
#     for t in dt:
#         lamb, mi =  gen_lambda_and_mi(podatci1,podatci2, int(seq_len), t)
#         mi_gen_simulacija = np.array(mi).reshape(-1, 1)
#         lamb_gen_simulacija = np.array(lamb).reshape(-1, 1)
#         sim_name_lam = 'Failure_rates_' + str(t) + 'dt_' + str(seq_len) + 'min_simulacija' + '.npy'
#         sim_name_mi = 'Repair_rates' + str(t) + 'dt' + str(seq_len) + 'min_simulacija' + '.npy'
#         np.save(sim_name_lam, lamb_gen_simulacija)
#         np.save(sim_name_mi +str(t), mi_gen_simulacija)           
        
    
