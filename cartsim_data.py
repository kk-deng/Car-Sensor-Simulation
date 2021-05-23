#define _FILE_SPECS "-rw-r--r-- 1 chris 8007 Mar 25 12:33 cartsim_data.py"
#define _MAGIC_NUMBER 1147484068
import numpy as np
import math, os,   sys,  time
from time import gmtime, strftime
# from logger import logger
from datetime import date, datetime, timezone
import statistics as st
import random as rn
import matplotlib.pyplot as plt
from numpy import random as rnn
 
from sklearn import metrics
import pickle

#equations
# M \ddot P = - C( X - X_0)  - D \dot X
#  P = X  + Z(Y(t))
#  Y(t)= vt

#  v=60km/hour = 16.666 m / sec
#
#  M= 1000kg
#  X_0 =  5cm
#  C spring const so 1000kg @ 10 m/sec^2 gives 5 cm ie
#    =   5 x 10e-06
#  X roughly +- 10cm
#  road 3 components period   1 sec, 2 sec, 4 sec (random amplitude)
#  sin(Y/(16.666)),  cos(Y/(16.666)), sin(Y/(2*16.66)), cos(Y/(2*16.66))
#  choose D0 to efold in 1 second ie. D/2M = 1.0 --> D=2000
#  too small is 'bad'
#  choose sampling rate @ 4 Hz
#  magnitude road = +- 5 max * sin( Y / 16.66 m)

#  10 minute samples = 600 x 4 points @ 4 Hz
#

def Zbase ( trigtype, period, K, Y):
    
    if trigtype=='sin':
        return math.sin( K * period * Y)
    if trigtype=='cos':
        return math.cos( K * period * Y)
    
def Zbaseddot ( trigtype, period, K, Y, v):
    
#     d^2/dt^ (VT)=0
    if trigtype=='sin':
        return -math.cos( K * period * Y) * v * v * (K /period) * (K/period)
    if trigtype=='cos':
        return -math.sin( K * period * Y) * v * v * (K /period) * (K/period)
    
def Xdot(Xn, Xnm1, dT):
    return (Xn - Xnm1)/dT

def Xddot(Xn, Xnm1, Xnm2, dT):
    return (Xn - 2 * Xnm1  + Xnm2)/(dT * dT)

def getXnp1(LHS, M, D, C, Xn, Xnm1, dT):
# solve for Xnp1
#  LHS = M(Xnp1 - 2 Xn + Xnm1)/delT^2 + D(Xnp1 - Xnm1)/2delT + C Xn
  
    rval = (LHS * dT * dT - (M - D * dT/2) * Xnm1  + (2 *M - C * dT * dT) * Xn  )/(M + D * dT/2) 
    
    lv = M* (rval - 2 * Xn + Xnm1)/(dT * dT) + D* (rval - Xnm1)/(2*dT) + C * Xn
    
#    print("check %f = %f" % (LHS,lv))
          
    return  rval

def getLHSval(Zddval, Ms, Vs, Cs, X0s):
# Cs * X0 =spring force
    return - Ms * Zddval + Cs * X0s

def zRoad(coeffs, v, Y, period, maxfreq):
    
    zR =0
   
    if maxfreq >= 0.5:
       zR = coeffs[0]* Zbaseddot('cos', period, 0.5, Y, v) + zR
       zR = coeffs[1]* Zbaseddot('sin', period, 0.5, Y, v) + zR

    if maxfreq >= 1.0:
       zR = coeffs[2]* Zbaseddot('cos', period, 1.0, Y, v) + zR
       zR = coeffs[3]* Zbaseddot('sin', period, 1.0, Y, v) + zR
    
    if maxfreq >=2.0:
       zR = coeffs[4]* Zbaseddot('cos', period, 2.0, Y, v) + zR
       zR = coeffs[5]* Zbaseddot('sin', period, 2.0, Y, v) + zR
       
    if maxfreq >=4.0:
       zR = coeffs[6]* Zbaseddot('cos', period, 4.0, Y, v) + zR
       zR = coeffs[7]* Zbaseddot('sin', period, 4.0, Y, v) + zR
    
    return zR

def  getRandomCoeffs(N):
    
     ampChoice=[0.01, 0.02, 0.025, 0.03, 0.035]
     coeffs=[]

     for i in range(0,N):
          rAmpl1=rn.choice(ampChoice) 
#          rAmpl1=rAmpl
          coeffs.append(rAmpl1)
          
     return coeffs
 
 

def  compute_sim(M, D0, V, C, X0, delT, period, maxfreq, coeffs, topsample):
    
    Y=0
# start with spring at rest
    Xnp1=X0
    Xn=X0
    Xnm1=X0


    springPos=[]
    timeVal=[]
    roadSurf=[]
    
     
    for i in range(0, topsample):
    
        t= i * delT
        timeVal.append(t)
#        print("                T=%8.2f" % t)
    
        Y= V * t  
    
        Zddval= zRoad(coeffs, V, Y, period, maxfreq)
    
#        print("Zddval %f" % Zddval)
    
# LHS= -M (ddot (Z(Y))) + CX0
        LHS= getLHSval(Zddval, M, V, C, X0)
        roadSurf.append(LHS)
    
        Xnp1 = getXnp1(LHS, M, D0, C, Xn, Xnm1, delT)
        springPos.append(Xnp1)
    
#        print("Xnp1 %8.3f  Xn %8.3f Xnm1 %8.3f" % (Xnp1, Xn, Xnm1))
    
        Xnm1=Xn
        Xn=Xnp1
        
    return [roadSurf, timeVal, springPos, t]

def add_sample(M, D0, V, C, X0, delT, period, maxfreq, botsample, topsample, lval,Ncoeffs, Rcoeffs, road_input_type):
 # compute discriminant
    disc= D0*D0 - 4 * M * C   
    
    [roadSurf, timeVals, springPos, tmax]= compute_sim(M, D0, V, C, X0, delT, period, maxfreq, Rcoeffs, topsample)

    yval=[]
    Xdat=[]

#    print("sample D0=%d label=%s" % (D0,lval )) 
#    print("D=%12.2f disc  %12.2f   maxfreq= %f" % (D0, disc,   maxfreq))
#    if disc < 0:
#        print("sqrt = %f" % math.sqrt(-disc))
 
# assume botsample is > 3

    if road_input_type=='vibration':
        tupleLen=8
    elif road_input_type=='surface':
        tupleLen=2
    else:
        print("unknown road_type %s" % road_type)
        tupleLen=2
 
# in this case include roadSurf = LHS as variable
    xnorm=10000 
    for i in range(botsample,topsample):
        if tupleLen==2:
# road input
           Xdat.append([roadSurf[i]/xnorm, springPos[i-2], springPos[i-1], springPos[i]])
# in vehicle vibration
        elif tupleLen==5:
           Xdat.append([springPos[i-5], springPos[i-4], springPos[i-3], springPos[i-2], springPos[i-1], springPos[i]])
#  in vehicle vibration (long)
        elif tupleLen==8:
           Xdat.append([springPos[-8], springPos[-7], springPos[-6], springPos[i-5], springPos[i-4], springPos[i-3], springPos[i-2], springPos[i-1], springPos[i]])
        else:
            print("Unsupported tupleLen %d" % tupleLen)
            return [[],[]]
        
        yval.append(lval)
 
    return [Xdat, yval]
    

# mass    
M=2000 
#   5cm compression
X0= 0.05
# spring const    

C= 0.6  * 10e+04
delT=0.25
# interesting values 500, 5000, 15000, 25000
 
# damping
Dvalues=[5000, 5500, 6000, 6500, 4500, 4000, 3500, 500, 600, 700, 800, 400, 300, 200]

LABELvalues=['good','good', 'good', 'good', 'good','good','good','bad','bad','bad','bad','bad','bad','bad']

dindexlist=[x for x in range(0,14)]

 
# car moves at 16.66 m/s
V= 16.66 
period= 16.66 
maxfreq=4.0
 
botsample=400 
topsample=500
Ncoeffs=8

X=[]
y=[]
ngood=0
nbad=0

nruns=500
testfraction=0.3
experiment_type='random_roads'
#experiment_type='standard_road'

# initialize road
Rcoeffs=getRandomCoeffs(Ncoeffs) 

ldindexlist=[x for x in range(0, 100* nruns)]

for i in range(0,nruns):
    
           dindex=rn.choice(dindexlist)

           D0=Dvalues[dindex]
           labval=LABELvalues[dindex]
    
 
           if experiment_type=='random_roads':
              Rcoeffs=getRandomCoeffs(Ncoeffs)
              

 
#  For  in vehicle vibration road_input_type='vibration'
#  For  road input set road_input_type='surface'
           
           road_input_type='surface'
 
           [Xdat, yval]= add_sample(M, D0, V, C, X0, delT, period, maxfreq, botsample, topsample, labval,Ncoeffs, Rcoeffs, road_input_type)
           
# [roadsurf, Xn-2, Xn-1, Xn]
           if yval[0]=='good':
               ngood+=1
           if yval[0]=='bad':
               nbad+=1
 
           for j in range(0, len(Xdat)):
                X.append(Xdat[j])
                y.append(yval[j])
                

print("shuffling %d entries" % len(ldindexlist))
rn.shuffle(ldindexlist)
# sanity check
print(ldindexlist[0:10])

X_rn=[]
y_rn=[]

for i in range(0, len(X)):
    X_rn.append(X[ldindexlist[i]])
    y_rn.append(y[ldindexlist[i]])
    
    
           
print("made %d samples with botsample %d topsample %d" % (nruns,botsample, topsample))
print("maxfreq= %8.2f M= %8.2f V=%8.2f C=%8.2f" % (maxfreq, M, V, C))

totalN=len(X)
print("Total samples %d good runs %d bad runs %d" % (totalN, ngood, nbad))

#print(Xdat)

print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
 
# INSERT ML algorithm here (X_rn, y_rn) train vs test sets


#Mc=metrics.confusion_matrix(y_rn[int(testfraction*totalN):], y_pred)

#totalN=Mc[0][0] + Mc[0][1] + Mc[1][0] + Mc[1][1]
#misclassifiedN = Mc[0][1] + Mc[1][0]

#errorRate= misclassifiedN / totalN

#print("confusion matrix: on test data set")
#print(Mc)

#print("errorRate %5.3f" % errorRate)
print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print("N=%d : experiment_type: %s  road_input_type: %s" % (nruns,experiment_type, road_input_type))



 


 

 
    
    
    
