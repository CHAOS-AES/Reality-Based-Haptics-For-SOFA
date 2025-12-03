#*************************************************
# Scene by Øystein Bjelland, Dept. of ICT and Natural Sciences, NTNU Ålesund
# Email: oystein.bjelland@ntnu.no

#Units
#Mass [metric tonne (or Mg)], length [mm], time [s], force [N], Stress [kPa], density [tonne/mm^3], Young's modulus [MPa], Poisson's ratio [-], gravity [mm/s^2], stiffness [N/mm]

#Gravity: 9.81 [m/s^2] --> 9807 [mm/s^2]
#Density of bone: 0.55e-06 [kg/mm^3] 
#Density of meniscus: 1.09e-6 [kg/mm^3] -->  1.09e-9 [tonne/mm^3]
#Meniscus Young's modulus: 3 [MPa]

#Instrument mass: 0.035 [kg] --> 3.5e-05 [tonne]
#*************************************************

import Sofa.Core

# Required import for python
import Sofa
import SofaRuntime
import time
from matplotlib import pyplot as plt
import numpy as np

from kalmanFilterForce1d import kalmanFilter
from getEmpiricalForces import getProbingForces
from principalDirectionFunction import calculatePrincipalComponents
from nonlinearization import makeNonlinear, getMeniscusProbingForce


#OPTIONS BEGIN******************************************************************************************

# Enable/disable contact force extraction
extractContactForce = True
exportCSV = True

# Select  anatomy to include
addLateralMeniscus = True

#Haptic rendering options
useKalmanFilter = True
sendToHapticDevice = True
useInstrumentPosition = True

#OPTIONS END******************************************************************************************

# Function called when the scene graph is being created
def createScene(rootNode):
     
    rootNode.dt=0.01 #0.01, 0.05, 0.005
    rootNode.gravity=[0, -9.81e3, 0] 

    confignode = rootNode.addChild("Config")
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Geometry")
    confignode.addObject('RequiredPlugin', name="SofaPython3", printLog=False)
    confignode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('DefaultPipeline', name="pipeline", depth="6", verbose="0")
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('DefaultContactManager', name="response", response="FrictionContactConstraint") #responseParams="mu=0.1") #Does this call on the penalty method, instead of Lagrange M?
    rootNode.addObject('LocalMinDistance', alarmDistance="3.408", contactDistance="1.136", angleCone="0.0")
    rootNode.addObject('FreeMotionAnimationLoop') #, parallelCollisionDetectionAndFreeMotion = True, parallelODESolving = True)
    LCP = rootNode.addObject('LCPConstraintSolver', tolerance="0.001", maxIt="1000", computeConstraintForces="True", mu="0.000001")

    #Camera
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')
    rootNode.addObject('InteractiveCamera',name='camera') 
    rootNode.camera.position.value = [-150,30,0]

    #Haptics
    rootNode.addObject('RequiredPlugin', name='Geomagic')
    driver = rootNode.addObject('GeomagicDriver', name='GeomagicDevice', deviceName="Default Device", scale="10", drawDeviceFrame="1", positionBase="0 0 0", drawDevice="0", orientationBase="0 0.707 0 -0.707", maxInputForceFeedback = "3")  #scale attribute only affects position
    
    #LATERAL MENISCUS NODE BEGIN*************************************************************************************************************************************************************************
    if addLateralMeniscus == True:
        lateralMeniscus = rootNode.addChild('lateralMeniscus')
        lateralMeniscus.gravity = [0, -9.81e3, 0]
        lateralMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        lateralMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModelSTP.msh", rotation="90 270 -182", translation="150 -31.5 -26", scale3d="26 26 26")
        lateralMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="-0.5 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        lateralMeniscusDofs = lateralMeniscus.addObject('MechanicalObject', name="lateralMeniscusDofs", src="@meshLoader")
        lateralMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        lateralMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-9", topology="@topo", geometryState="@lateralMeniscusDofs") # massDensity="10.9e-03"
        lateralMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.29", youngModulus="5.05") # 3.84
        lateralMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")
        lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="1 2 5 6 7 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 73 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 251 252 253 254 255 256 257")

        #Lateral meniscus visual model
        lateralMeniscusVisual = lateralMeniscus.addChild('Lateral Meniscus Visual Model')
        lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/LATERAL_MENISCUS_VISUAL_MODEL.obj", rotation="-85 277 -185", translation="270 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        lateralMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_LM", color="1.0 0.2 0.2 1.0")
        lateralMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../lateralMeniscusDofs", output="@VisualModel")

        #Lateral meniscus collision model (extracts triangular mesh from computation model)
        lateralMeniscusTriangularSurface = lateralMeniscus.addChild('Lateral Meniscus Triangular Surface Model')
        lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyContainer', name="Container")
        lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        lateralMeniscusTriangularSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        lateralMeniscusTriangularSurface.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")
    #LATERAL MENISCUS NODE END***************************************************************************************************************************************************************************

    #LATERAL MENISCUS NODE BEGIN*************************************************************************************************************************************************************************
    if addLateralMeniscus == True:
        lateralMeniscus2 = rootNode.addChild('lateralMeniscus2')
        lateralMeniscus2.gravity = [0, -9.81e3, 0]
        lateralMeniscus2.addObject('EulerImplicitSolver', name="cg_odesolver")
        lateralMeniscus2.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        lateralMeniscus2.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModelSTP.msh", rotation="90 270 -182", translation="150 -31.5 34", scale3d="26 26 26")
        lateralMeniscus2.addObject('TetrahedronSetTopologyContainer', name="topo", position="-0.5 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        lateralMeniscusDofs2 = lateralMeniscus2.addObject('MechanicalObject', name="lateralMeniscusDofs", src="@meshLoader")
        lateralMeniscus2.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        lateralMeniscus2.addObject('DiagonalMass', name="mass", massDensity="1.09e-9", topology="@topo", geometryState="@lateralMeniscusDofs") 
        lateralMeniscus2.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.29", youngModulus="5.05") # 3.84
        lateralMeniscus2.addObject('PrecomputedConstraintCorrection', recompute="true")
        lateralMeniscus2.addObject('FixedConstraint', name="FixedConstraint", indices="1 2 5 6 7 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 73 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 251 252 253 254 255 256 257")

        #Lateral meniscus visual model
        lateralMeniscusVisual2 = lateralMeniscus2.addChild('Lateral Meniscus 2 Visual Model')
        lateralMeniscusVisual2.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/LATERAL_MENISCUS_VISUAL_MODEL.obj", rotation="-85 277 -185", translation="270 105 -20", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        lateralMeniscusVisual2.addObject('OglModel', name="VisualModel", src="@meshLoader_LM", color="1.0 0.2 0.2 1.0")
        lateralMeniscusVisual2.addObject('BarycentricMapping', name="visual mapping", input="@../lateralMeniscusDofs", output="@VisualModel")

        #Lateral meniscus collision model (extracts triangular mesh from computation model)
        lateralMeniscusTriangularSurface2 = lateralMeniscus2.addChild('Lateral Meniscus Triangular Surface Model')
        lateralMeniscusTriangularSurface2.addObject('TriangleSetTopologyContainer', name="Container")
        lateralMeniscusTriangularSurface2.addObject('TriangleSetTopologyModifier', name="Modifier")
        lateralMeniscusTriangularSurface2.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        lateralMeniscusTriangularSurface2.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")
    #LATERAL MENISCUS NODE END***************************************************************************************************************************************************************************


    #OMNI NODE BEGIN*************************************************************************************************************************************************************************************
    omni = rootNode.addChild('Omni')
    omniDOFs = omni.addObject('MechanicalObject', name='DOFs', template='Rigid3d', position='@GeomagicDevice.positionDevice')
    #OMNI NODE END*********************************************************************************************************************************************************************************************

    #INSTRUMENT NODE BEGIN*************************************************************************************************************************************************************************************
    instrument = rootNode.addChild('Instrument')
    instrument.addObject('EulerImplicitSolver', name="ODE solver") 
    instrument.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
    instrument.addObject('MechanicalObject', name='instrumentState', template='Rigid3d')
    instrument.addObject('UniformMass', name="mass", totalMass="3.5e-05") #[kg] 3.5e-05 Mass of instrument measured to be 35 g. 
    if sendToHapticDevice == False:
        instrument.addObject('LCPForceFeedback', activate="true", forceCoef="1.0") #Multiply haptic force by this coefficient to scale force in haptic device. 
    instrument.addObject('LinearSolverConstraintCorrection')
    instrument.addObject('RestShapeSpringsForceField', stiffness="1000000", angularStiffness="1000000", external_rest_shape="@/Omni", points="0")

            
    #Instrument visual model
    instrumentVisu = instrument.addChild('InstrumentVisualModel')
    instrumentVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/ARTHREX_AR_10000_HookProbe.obj", scale3d="1 1 1", handleSeams="1")
    instrumentVisu.addObject('OglModel', name="InstrumentVisualmodel", src="@loader", color="gray", rx="270", ry="90", rz="90", dy="5", dz="0")
    instrumentVisu.addObject('RigidMapping', name="MM->VM mapping", input="@instrumentState", output="@InstrumentVisualModel") 
            
    #Instrument collision model
    instrumentCollision = instrument.addChild('instrumentCollision')
    instrumentCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/ARTHREX_AR_10000_HookProbe.obj", scale3d="1 1 1", handleSeams="1")
    instrumentCollision.addObject('MeshTopology', src="@loader")
    instrumentCollision.addObject('MechanicalObject', name="instrumentCollisionState1", src="@loader",rx="270", ry="90", rz="90", dy="5", dz="0")
    instrumentCollision.addObject('LineCollisionModel', contactStiffness="0.44e-20") 
    instrumentCollision.addObject('PointCollisionModel', contactStiffness="0.44e-20") 
    instrumentCollision.addObject('RigidMapping', name="MM->CM mapping", input="@instrumentState", output="@instrumentCollisionState1")
    #INSTRUMENT NODE END**********************************************************************************************************************************************************************************************

    #Access data
    if extractContactForce == True: 
        rootNode.addObject(MatrixAccessController('MatrixAccessor', name='matrixAccessor', target=lateralMeniscusDofs, target2=LCP, target3=omniDOFs, target4=driver, target5=lateralMeniscusDofs2, rootNode=rootNode))
    
    return rootNode



class MatrixAccessController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.target = kwargs.get("target", None)
        self.target2 = kwargs.get("target2", None)
        self.target3 = kwargs.get("target3", None)
        self.target4 = kwargs.get("target4", None)
        self.target5 = kwargs.get("target5", None)
        self.rootNode = kwargs.get("rootNode", None)

        self.contactOngoing = False
        self.contactPointStart = np.zeros((1,3))
        self.movingAverage = np.zeros((1,3)) #Size of moving average window
        self.previousTime = time.time()

    def onAnimateEndEvent(self, event):
        
        currentTime = time.time()
        dt = currentTime - self.previousTime
        self.previousTime = currentTime

        #STEP 0: Initialize
        constraint = self.target.constraint.value
        constraint2 = self.target5.constraint.value
        forcesNorm = self.target2.constraintForces.value
        instrumentPosition = self.target3.getData("position")
        GeomagicDevice = self.rootNode.GeomagicDevice

        #Detect which meniscus is in contact
        if len(constraint2) > 0:
            constraint = constraint2
        else:
            constraint = constraint

        constraintMatrixInline = np.fromstring(constraint, sep='  ')

        pointId = []
        constraintId = []
        constraintDirections = []
        index = 0
        i = 0

        contactforce_x = 0
        contactforce_y = 0
        contactforce_z = 0

        if len(constraintMatrixInline) > 0:
            #STEP 1: CREATE LISTS OF CONSTRAINTS AND POINTS*******************************************************************************************************
            while index < len(constraintMatrixInline):
                nbConstraint   = int(constraintMatrixInline[index+1])
                currConstraintID = int(constraintMatrixInline[index])
                for pts in range(nbConstraint):
                    currIDX = index+2+pts*4
                    pointId = np.append(pointId, constraintMatrixInline[currIDX])
                    constraintId.append(currConstraintID)
                    constraintDirections.append([constraintMatrixInline[currIDX+1],constraintMatrixInline[currIDX+2],constraintMatrixInline[currIDX+3]])
                index = index + 2 + nbConstraint*4

            #print("Point ID list:  " + str(pointId))

            #STEP 2: Get number DOFs from mechanical object*******************************************************************************************************
            nbDofs = len(self.target.position.value) 
            #print('nbDofs', nbDofs)

            #STEP 3: INITIALIZE EMPTY FORCE VECTOR*******************************************************************************************************
            forces = np.zeros((nbDofs,3))

            #print('pointid', pointId)
            #print('constraintId', constraintId)

            #STEP 4: GO THROUGH LIST OF POINT IDs AND COMPUTE FORCES X,Y and Z*******************************************************************************************************
            for i in range(len(pointId)):
                indice = int(pointId[i])
                forces[indice][0] = forces[indice][0] + constraintDirections[i][0] * forcesNorm[constraintId[i]] / dt
                forces[indice][1] = forces[indice][1] + constraintDirections[i][1] * forcesNorm[constraintId[i]] / dt
                forces[indice][2] = forces[indice][2] + constraintDirections[i][2] * forcesNorm[constraintId[i]] / dt

            #print('indice', i, indice)          
            #print('force', forces)                     

            #STEP 5: GO THROUGH LIST OF NBDOFs AND COMPUTE FORCES
            for i in range (nbDofs):
                contactforce_x += forces[i][0]
                contactforce_y += forces[i][1]
                contactforce_z += forces[i][2]

        else: #If there is no interaction between object and instrument, there is no reaction force
            contactforce_x = 0
            contactforce_y = 0
            contactforce_z = 0   

        reactionForceVector = np.array(([contactforce_x, contactforce_y, contactforce_z]))
        reactionForceVector = np.reshape(reactionForceVector, (1,3))
      
        #COMPUTE KALMAN FILTERED HAPTIC FORCE*******************************************************************************************************
        if ((len(constraintMatrixInline) > 0) & (useKalmanFilter == True)):
            
            #Get empirical force vector from interpolation
            #Get displacement of contact node
            
            if useInstrumentPosition:
                instrumentPos = instrumentPosition.value #Get instrument position (x,y,z)
                nodePosition = instrumentPos[0, 0:3]
                nodePosition = np.array([nodePosition]) #Convert to array (1x3)
            else:
                positionOfDOFs = self.target.position.value #[pointId[0] , :] #displacement vector of random node in contact
                positionOfDOFs = np.array(positionOfDOFs) 
                nodePosition = positionOfDOFs[int(pointId[0]), 0:3]
                nodePosition = np.array([nodePosition])
                
            #Detect first position and then subtract from all subsequent positions to get displacement
            if self.contactOngoing == False:
                self.contactPointStart = nodePosition
                contactPointDisplacement = np.zeros((1,3))
                self.contactOngoing = True #Only run this once

            else:
                contactPointDisplacement = nodePosition - self.contactPointStart
            
            #Get force magnitude and projected displacement along force vector of simulation reaction force
            [projectedContactPointDisplacement, reactionForceMagnitude, principalDirection] = calculatePrincipalComponents(contactPointDisplacement, reactionForceVector)   

            #Get Reality-Based Reference Signal
            beta = 1/(10.7*3) #Defined as beta = 1/F, where F is the force at which the empirical force is equal to the reaction force
            empiricalForceMagnitude = getMeniscusProbingForce(beta, reactionForceMagnitude)

            #Perform Kalman Filter sensor fusion
            kfForceMagnitude = kalmanFilter(reactionForceMagnitude, empiricalForceMagnitude, dt)

            #Calculate haptic force
            scaleFactor =  0.10*0.3 #This value must be tuned manually to fit force capacity of haptic device.
            hapticForceMagnitude = kfForceMagnitude*scaleFactor #Scale force 

            maxHapticForce = 1.99
            if hapticForceMagnitude > maxHapticForce:  #Prevent excessive force on haptic device
                hapticForceMagnitude = maxHapticForce

            hapticForce = np.multiply(hapticForceMagnitude,principalDirection)
           
            #SEND HAPTIC FORCE TO DEVICE*******************************************************************************************************
            if sendToHapticDevice:
                GeomagicDevice.inputForceFeedback[0] = hapticForce[0] *(-1)
                GeomagicDevice.inputForceFeedback[1] = hapticForce[1] *(-1)
                GeomagicDevice.inputForceFeedback[2] = hapticForce[2] *(-1)
                
                #print('inputForceFeedback: ' + str(GeomagicDevice.inputForceFeedback[0]) + str(GeomagicDevice.inputForceFeedback[1]) + str(GeomagicDevice.inputForceFeedback[2]))
                
        else:
            self.contactOngoing = False
            contactPointDisplacement = np.zeros((1,3))
            projectedContactPointDisplacement = np.float64(0)
            reactionForceMagnitude = float(0)
            empiricalForceMagnitude = np.float64(0)
            kfForceMagnitude = np.float64(0)
            hapticForceMagnitude = np.float64(0)
            principalDirection = np.zeros(3)
            
            GeomagicDevice.inputForceFeedback[0] = 0
            GeomagicDevice.inputForceFeedback[1] = 0
            GeomagicDevice.inputForceFeedback[2] = 0


    

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()