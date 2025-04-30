import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def calculatePrincipalComponents(displacement:np.ndarray, force:np.ndarray):
    
    #STEP 1: Calculate Principle Direction of Force Vector
    forceMatrix = force[0,0:3]
    #forceMagnitude = np.zeros((1,1))
         
    forceMagnitude = np.linalg.norm(forceMatrix) # Calculate magnitude of force vector
    if forceMagnitude == 0:
        forceMagnitude = 1e-6 #Avoid division by zero

    unitVector = forceMatrix / forceMagnitude
    principalDirection = unitVector
       
    #STEP2: Calculate displacement along principle direction (projecting position vector onto principal force direction vector)
    displacementAlongPrincipalDirection = np.zeros((1,1))
    posVector = displacement[0,0:3]
    displacementAlongPrincipalDirection = np.dot(posVector, principalDirection)
    displacementAlongPrincipalDirection = np.absolute(displacementAlongPrincipalDirection) #Return absolute value

    return [displacementAlongPrincipalDirection, forceMagnitude, principalDirection]