#define _FILE_SPECS "-rw-r--r-- 1 chris 3907 Mar 24 22:14 hosesim_data.py"
#define _MAGIC_NUMBER 412389958
import numpy as np
import math, os,   sys,  time
 
 
import statistics as st
import random as rn
import matplotlib.pyplot as plt
from numpy import random as rnn
 
#equations
# F = beta * p * S  + eps
# p = alpha R/(1000+R)
# code to generate some data
# assume radius of hose = 2cm  
# cross section = 2*2 PI  = 4 PI cm^2  = 12.566
# critical cross section = 10.0528
# flow is 3 L /min = 0.05 L /sec = 50 cc/sec when RPM=1000
# set alpha=1  so RPM=1000 --->  p= 1 * 0.5= .5
#  F= beta (cm/s) * 0.5 * 12.566 cm^2 ---> 50 =
#     beta * 6.283 ---> beta= 3.979 (cm/s)
# when RPM=5000 --->  p= 5000/(1000+5000)=0.8333
# RPM=1000, 2000, 3000, 4000, 5000
# times=[2 min, 5 min, 10 min]
# sample rate = every 10 sec
# want to discriminate these regions w. SVM
#[[F1, R1], [F, R2]....]
 
# code to generate synthetic data

def make_one_data_set(RPMvals, TIMEvals, LABELvals, Svals):
    
    Slabels=list(zip(Svals, LABELvals))
# generate 20 min of data = 1200 secs = 120 points

# nominal
#S=12.566
#S0=10.0528
    timeInts=[]
    jindex=0
    totTime=0
    rpmChoices=[]
    errTerm=[]

    Spair=rn.choice(Slabels)

    S=Spair[0]

    while totTime < 600 and jindex < 100:
        t= rn.choice(TIMEvals)
        if totTime + t <= 700:
           totTime=totTime + t
           timeInts.append(t)
           r=rn.choice(RPMvals)
           rpmChoices.append(r)
        jindex=jindex+1
        
    print("time intervals")
    print(timeInts)
    print("rpm values")
    print(rpmChoices)
    print("total time %8.2f" % totTime)

    li=len(timeInts)
# labels
    Lvals=[]
# Flow
    Fvals=[]
    rFvals=[]
# RPM
    Rvals=[]
#alpha=1.0
    beta= 3.979

    for i in range(0,li):
# 10 second intervals
        j= int(timeInts[i]/10)
    
        for k in range(0, j):
            rvals= 1.0 * rpmChoices[i]
            Rvals.append(rvals)
            pvals = 1.0 * rvals/(1000 + rvals)
            fvals = beta * pvals * S
            Fvals.append(fvals)
            Lvals.append(Spair[1])

    rFvals=[round(x, 2) for x in Fvals]


    featurePairs=[]
    for i in range(0, len(Rvals)):
        x=rFvals[i]
        y=Rvals[i]
        featurePairs.append([x,y])
    
    print(featurePairs[:60])
    print("Parameters")
    print( Spair[1])
    
    return [featurePairs, Lvals]


RPMvals=[500, 750,  1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 4500, 5000]
TIMEvals=[60, 120, 180, 240, 300,360, 420, 480,540, 600]
Svals=[11.1,  12.3, 11.8, 11.5, 10.5, 10.9, 9.8, 9.0, 8.2, 9.5, 7.0, 7.5]
LABELvals=['good', 'good', 'good', 'good','good','good','bad','bad','bad','bad','bad','bad']

fout =open('hosesvmdata.dat','w')

X=[]
y=[]
goodcount=0
badcount=0
XrpmGood=[]
XflowGood=[]
XrpmBad=[]
XflowBad=[]

Nexamples=500 

for i in range(0, Nexamples):
       featureSet, Lvals=make_one_data_set(RPMvals, TIMEvals, LABELvals, Svals)
       if Lvals[0]=='good':
           goodcount+=1
       if Lvals[0]=='bad':
           badcount+=1
           
       for j in range(0,len(featureSet)):
           fout.write("%5.2f, %5.2f, %s\n" %(featureSet[j][0], featureSet[j][1], Lvals[j]))
           X.append([featureSet[j][0], featureSet[j][1]])
           y.append(Lvals[j])
           
           if Lvals[0]=='good':
              XrpmGood.append(featureSet[j][1])
              XflowGood.append(featureSet[j][0])
           elif Lvals[0]=='bad':
              XrpmBad.append(featureSet[j][1])
              XflowBad.append(featureSet[j][0])   
           else:
               print("Oops: label = %s" % Lvals[0])
           
fout.close()

print("goodcount %d badcount %d" % (goodcount, badcount))

# INSERT ML FITTING CODE HERE  (X, y)
 
plt.scatter(XrpmGood, XflowGood,c='b',marker='o')
plt.scatter(XrpmBad, XflowBad,c='r',marker='+')
plt.xlabel("rpm")
plt.ylabel("flow")

plt.show()


  
    
    






