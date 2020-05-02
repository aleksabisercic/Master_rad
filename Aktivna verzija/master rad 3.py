# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:46:46 2020

@author: Freedom
"""
import numpy as np
import scipy as sp
from scipy.optimize import fsolve

def F_btd_inv(t,L_btd,lambda_t3e, r):
    return (1+lambda_t3e*t)*np.exp(-L_btd*t)-(1-r)
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
Lbe=0.0000669323/60 #prebacio oz min u sec
Lbm=0.0000433066/60
Lbo=0.000073458/60
Ldo=0.0000588205/60
Lt1e=0.0000680535/60
Lt1m=0.0000531802/60
Lt1=0.000121234/60
Lt2e=0.0000564661/60
Lt2m=0.000059579/60
Lt2=0.000116045/60
lambda_t3e=0.000144174/60
Lt3m=0.0000758069/60
Ltxo=0.0000913767/60
mbe=0.00027154/60 #Proveri da li si lepo sracunao(podelio sa 60)
mbm=0.000474394/60
mbo=0.0000731771/60
mb=0.000323161/60
mdo=0.000226708/60
mt1e=0.000393955/60
mt1m=0.000817231/60
mt2e=0.000445557/60
mt2m=0.00090446/60
mt3e=0.000534383/60
mt3m=0.000627307/60
mtxo=0.000385339/60
Lb=0.000183697/60
Ld=0.0000588205/60
Ltx=0.000548637/60
L_btd=0.000791154/60
Pvo1=0.18988903
Pvo2=0.36405672
Pvo3=0.446054254
Peo1=0.42532468 #stavi na pocetak ne menjaju
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

STbtd="11"
STb="11"
STd="11"
STt="11"
STt1="11"
STt2="11"
STt3="11"
brR=0

y = bisection(F_btd_inv,0,108000,1000, L_btd, lambda_t3e, r)
y0 = fsolve(F_btd_inv,0,args = (L_btd, lambda_t3e, r))
tp0=0
if y is None: 
    tp0=y0
    print(tp0)
else:
    tp0=y
    print(tp0)

broj_sekundi_u_godini=31536000
for t in range(1,31536000,1):
    if t==tp0:
        brR=brR+1
        r=np.random.uniform(low=0.0, high=1.0,size=None)
        if r<Pvo1:
            VRotk="1"
            r1=np.random.uniform(low=0.0, high=1.0,size=None)
            if r1<Peo1:
                OBJeo="1"
                brOEb=brOEb+1
                STb="21"
                r11=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mbe)*np.ln(r11)
                topr=t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STt3="12"
            elif r1<(Peo1+Peo2):
                OBJeo="2"
                brOEt1=brOEt1+1
                STt1="21"
                r12=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt1e)*np.ln(r12)
                topr=t+DT
                STd="12"
                STb="12"
                STt2="12"
                STt3="12"
            elif r1<(Peo1+Peo2+Peo3):
                OBJeo="3"
                brOEt2=brOEt2+1
                STt2="21"
                r13=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt2e)*np.ln(r13)
                topr=t+DT
                STd="12"
                STt1="12"
                STb="12"
                STt3="12"
            elif r1<=(1): #<=1
                OBJeo="4"
                brOEt3=brOEt3+1
                STt3="21"
                r14=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt3e)*np.ln(r14)
                topr=t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STb="12"
        elif r<(Pvo1+Pvo2):
            Vrotk="2"
            r2=np.random.uniform(low=0.0, high=1.0,size=None)
            if r2<Pmo1:
                OBJmo="1"
                brOMb=brOMb+1
                STb="22"
                r21=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mbm)*np.ln(r21)
                topr=t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STt3="12"
            elif r2<(Pmo1+Pmo2):
                OBJmo="2"
                brOMt1=brOMt1+1
                STt1="22"
                r22=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt1m)*np.ln(r22)
                topr=t+DT
                STd="12"
                STb="12"
                STt2="12"
                STt3="12"
            elif r2<(Pmo1+Pmo2+Pmo3):
                OBJmo="3"
                brOMt2=brOMt2+1
                STt2="22"
                r23=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt2m)*np.ln(r23)
                topr=t+DT
                STd="12"
                STt1="12"
                STb="12"
                STt3="12"
            elif r2<=(1):
                OBJmo="4"
                brOMt3=brOMt3+1
                STt3="22"
                r24=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mt3m)*np.ln(r24)
                topr=t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STb="12"
        elif r<=(1): 
            Vrotk="3"
            r3=np.random.uniform(low=0.0, high=1.0,size=None)
            if r3<Poo1:
                OBJoo="1"
                brOOb=brOOb+1
                STb="23"
                r31=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mbo)*np.ln(r31)
                topr=t+DT
                STd="12"
                STt1="12"
                STt2="12"
                STt3="12"
            elif r3<(Poo1+Poo2):
                OBJoo="2"
                brOOd=brOOd+1
                STd="23"
                r32=np.random.uniform(low=0.0, high=1.0,size=None)
                DT=-(1/mdo)*np.ln(r32)
                topr=t+DT
                STt1="12"
                STb="12"
                STt2="12"
                STt3="12"
            elif r3<=(1):
                OBJoo="3"
                r4=np.random.uniform(low=0.0, high=1.0,size=None)
                if r4<Ptt1:
                    OBJtt="1"
                    brOOt1=brOOt1+1
                    STb="23"
                    STd="12"
                    STb="12"
                    STt2="12"
                    STt3="12"
                elif r4<(Poo1+Poo2):
                    OBJtt="2"
                    brOOt2=brOOt2+1
                    STd="23"
                    STt1="12"
                    STb="12"
                    STt3="12"
                elif r4<=(1):
                    OBJtt="3"
                    brOOt3=brOOt3+1
                    STd="23"
                    STt1="12"
                    STt2="12"
                    STb="12"
            r41=np.random.uniform(low=0.0, high=1.0,size=None)
            DT=-(1/mtxo)*np.ln(r41) 
            topr=t+DT
        if t==topr: 
         STd="11"
         STt1="11"
         STt2="11"
         STt3="11"
         STb="11"
         y1 = bisection(F_btd_inv,0,108000,1000, L_btd, lambda_t3e, r)
         y01 = fsolve(F_btd_inv,0,args = (L_btd, lambda_t3e, r))
         tp01=0
         if y is None: 
             tp01=y01
             print(tp01)
         else:
             tp0=y1
             print(tp01)
         VrRb=VrRb+1
         VrRADb=VrRADd+1
         VrRd=VrRd+1
         VrRADd=VrRADd+1
         VrRt1=VrRt1+1
         VrRADt1=VrRADt1+1
         VrRt2=VrRt2+1
         VrRADt2=VrRADt2+1
         VrRt3=VrRt3+1
         VrRADt3=VrRADt3+1
        if STb=="12":  
            VrCnRb=VrCnRb+1
            VrRADb=VrRADb+1
        if STb=="21":
            VrOEb=VrOEb+1
            VrOTKb=VrOTKb+1
        if STb=="22":
            VrOMb=VrOMb+1
            VrOTKb=VrOTKb+1
        if STb=="23":
            VrOOb=VrOOb+1
            VrOTKb=VrOTKb+1
        if STt1=="12":  
            VrCnRt1=VrCnRt1+1
            VrRADt1=VrRADt1+1
        if STt1=="21":
            VrOEt1=VrOEt1+1
            VrOTKt1=VrOTKt1+1
        if STt1=="22":
            VrOMt1=VrOMt1+1
            VrOTKt1=VrOTKt1+1
        if STb=="23":
            VrOOt1=VrOOt1+1
            VrOTKt1=VrOTKt1+1
        if STt2=="12":  
            VrCnRt2=VrCnRt2+1
            VrRADt2=VrRADt2+1
        if STt2=="21":
            VrOEt2=VrOEt2+1
            VrOTKt2=VrOTKt2+1
        if STt2=="22":
            VrOMt2=VrOMt2+1
            VrOTKt2=VrOTKt2+1
        if STt2=="23":
            VrOOt2=VrOOt2+1
            VrOTKt2=VrOTKt2+1
        if STt3=="12":  
            VrCnRt3=VrCnRt3+1
            VrRADt3=VrRADt3+1
        if STt3=="21":
            VrOEt3=VrOEt3+1
            VrOTKt3=VrOTKt3+1
        if STt3=="22":
            VrOMt3=VrOMt3+1
            VrOTKt3=VrOTKt3+1
        if STt3=="23":
            VrOOt3=VrOOt3+1
            VrOTKt3=VrOTKt3+1
        if STd=="12":  
            VrCnRd=VrCnRd+1
            VrRADd=VrRADd+1
        if STd=="23":
            VrOOd=VrOOd+1
            VrOTKd=VrOTKd+1
       
        

    
        
        
        
           
        
    
