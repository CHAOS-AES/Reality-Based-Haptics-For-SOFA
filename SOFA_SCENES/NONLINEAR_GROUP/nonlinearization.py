#Function for calculating nonlinear haptic feedback from linear

import numpy as np
from numpy.linalg import inv
#import matplotlib.pyplot as plt

def makeNonlinear(contactPointDisplacement, sofaReactionForce):

    #perform integration:
    # y' = ax --> y = 1/2 *y'*x = 1/2*2ax^2
    nonLinearForce = sofaReactionForce*(1/2)*contactPointDisplacement
    
    return nonLinearForce
    

def getMeniscusProbingForce(beta, sofaReactionForce):

    probingForce = beta*sofaReactionForce*sofaReactionForce    
    return probingForce
    