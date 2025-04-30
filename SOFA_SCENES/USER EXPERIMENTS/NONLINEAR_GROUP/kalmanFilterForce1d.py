#Function for estimating force using a Kalman Filter

import numpy as np
from numpy.linalg import inv
#import matplotlib.pyplot as plt

def kalmanFilter(reactionForceVector:np.ndarray, empiricalForceVector:np.ndarray, dt):

    #STEP 1: Initialize matrices*********************************************************************************************
    #Process matrix, A
    A = np.array([[1,1],
                  [0,1],
                ],dtype=np.float64)
    
    A[0,1] = dt
    
    #Measurement matrix, H
    H = np.array([[1,0],
                ],dtype=np.float64)
    
    #Process noise matrix, Q
    a = 7.0275 
    b = 0.2772 
    c = 6.1024 
    Q = np.array([[a,b],
                  [b,c],
                ],dtype=np.float64)
    
    #Measurement noise matrix, R
    r = 0.7742
    R_emp = np.array([[r],
                ],dtype=np.float64)
    
    R_FEM = np.array([[0.001],
                ],dtype=np.float64)
    
    Pplus = np.zeros((2),dtype=np.float64)
    xhat = np.zeros((2),dtype=np.float64)
    I = np.identity((2),dtype=np.float64)
    z = np.zeros((1,1),dtype=np.float64) #Sensor measurement vector
    
    
    #STEP 2: Estimate forces************************************************************************************************
                
    #STEP 2a: Take measurement of simulation reaction forces
    zTransposed = reactionForceVector
    z = np.transpose(zTransposed)
    R = R_FEM

    #STEP 2b: Estimate force based on simulation forces    
    #Predict
    xhat = np.matmul(A,xhat)
    Pmin = np.matmul(np.matmul(A,Pplus), np.transpose(A)) + Q
    #Update
    K = np.matmul(np.matmul(Pmin,np.transpose(H)), np.linalg.inv(np.matmul(np.matmul(H,Pmin),np.transpose(H)) + R))
    xhat = xhat + np.matmul(K,(z - np.matmul(H,xhat)))
    Pplus = np.matmul((I - np.matmul(K,H)),Pmin)
    
    #STEP 2c: Take measurement of empirical meniscus probing reaction forces
    zTransposed = empiricalForceVector
    z = np.transpose(zTransposed)
    R = R_emp

    #STEP 2d: Estimate force based on empirical forces
    #Predict
    xhat = np.matmul(A,xhat)
    Pmin = np.matmul(np.matmul(A,Pplus), np.transpose(A)) + Q
    #Update
    K = np.matmul(np.matmul(Pmin,np.transpose(H)), np.linalg.inv(np.matmul(np.matmul(H,Pmin),np.transpose(H)) + R))
    xhat = xhat + np.matmul(K,(z - np.matmul(H,xhat)))
    Pplus = np.matmul((I - np.matmul(K,H)),Pmin)
    
    return xhat[0]
            
    