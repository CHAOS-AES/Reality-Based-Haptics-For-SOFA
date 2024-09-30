#Function for reading empirical force values and performing interpolation

import numpy as np
from numpy.linalg import inv
#import matplotlib.pyplot as plt

def getProbingForces(contactPointDisplacement):

    #Import data from csv file
    #data = np.genfromtxt("forceDisplacementData.csv", delimiter=",") #Timestamp (s), displacement (m), force magnitude (N)
    #data = np.delete(data, 0, 0) #remove header
    #empiricalDisplacementArray = data[:,1]*(-1)*(1000) #Convert to mm, and invert if necessary
    #empiricalForceArray = data[:,2]

    # Define manually
    empiricalDisplacementArray = np.array([0.00,1.93,3.91,5.02,5.82,6.52,7.12,7.52,7.62])
    empiricalDisplacementArray = empiricalDisplacementArray*(2/3)

    empiricalForceArray = np.array([0.31,0.85,1.77,2.97,4.36,5.57,6.55,7.20,7.51])
    empiricalForceArray = empiricalForceArray - empiricalForceArray[0]

    #Remove any possible negative values
    for k in range(0, len(empiricalDisplacementArray)):
        if empiricalDisplacementArray[k] < 0:
            empiricalDisplacementArray[k] = 0

    empiricalForce = np.interp(contactPointDisplacement, empiricalDisplacementArray, empiricalForceArray)
    
    return empiricalForce
    

    
    