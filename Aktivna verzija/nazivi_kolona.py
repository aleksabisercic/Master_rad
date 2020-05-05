# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:49:34 2020

@author: Freedom
"""

ws1_kolone = ["Broj simulacije", "SrVrRb" , "SrVrCnRb" , "SrVrRADb" , "SrVrOEb" , 
                "SrVrOMb" , "SrVrOOb" , "brOTKb" , "SrVrOTKb" , "Ab" , "Aeb" , "Aob" , "Amb"]
ws2_kolone = ["SrVrRd" , "SrVrCnRd" , "SrVrRADd" , "SrVrOOd" , "brOTKd" , "SrVrOTKd" ,  "SrVrOOd" , "SrVrOTKd", "Ad" , "Aod"]

ws3_kolone = ["SrVrRt1" , "SrVrCnRt1" , "SrVrRADt1" , "SrVrOEt1" , "SrVrOMt1" , "SrVrOOt1" ,"SrVrOTKt1", "brOTKt1", "At1", "Aet1" , "Aot1" , "Amt1"]
ws4_kolone = ["SrVrRt2"," SrVrCnRt2", "SrVrRADt2", "SrVrOEt2", "SrVrOMt2", "SrVrOOt2", "brOTKt2", "SrVrOTKt2", "At2", "Aet2", "Aot2", "Amt2"]
ws5_kolone = ["SrVrRt3"," SrVrCnRt3", "SrVrRADt3", "SrVrOEt3", "SrVrOMt3", "SrVrOOt3", "brOTKt3", "SrVrOTKt3", "At3", "Aet3", "Aot3", "Amt3"]
ws6_kolone = ["SrVrRt" , "SrVrCnRt" , "SrVrRADt" , "SrVrOTKt" , "At"]
ws7_kolone = ["VrRbtd" , "SrVrRbtd" , "Abtd"]


k = 0
for i in ws1_kolone: 
    ws1.row(0).write(k,i)
    k+=1   
for i in ws2_kolone: 
    ws2.row(0).write(k,i)
    k+=1
for i in ws3_kolone: 
    ws3.row(0).write(k,i)
    k+=1	
for i in ws4_kolone: 
    ws4.row(0).write(k,i)
    k+=1	
for i in ws5_kolone: 
    ws5.row(0).write(k,i)
    k+=1	
for i in ws6_kolone: 
    ws7.row(0).write(k,i)
    k+=1
for i in ws6_kolone: 
    ws7.row(0).write(k,i)
    k+=1