#*************************************************
# Scene by Øystein Bjelland, Dept. of ICT and Natural Sciences, NTNU Ålesund
# Email: oystein.bjelland@ntnu.no

#Units
#Mass [metric ton (1e3 kg)], length [mm], time [s], force [N], Stress [MPa], density [1e3kg/mm^3], Young's modulus [MPa], Poisson's ratio [-], gravity [mm/ms^2], stiffness [N/mm]

#Gravity: 9.81 [m/s^2] --> 9.81e3 [mm/s^2]
#Density of bone: 0.55e-06 [kg/mm^3] --> 5.5e-10 [ton/mm^3]
#Density of meniscus: 1.09e-6 [kg/mm^3] --> 1.09e-9 [ton/mm^3]
#Meniscus Young's modulus: 16 [MPa]

#Instrument mass: 0.035 [kg] --> 3.5e-5 [ton]
#Instrument stiffness: 1.2347 [N/mm]
#*************************************************

import Sofa.Core

# Required import for python
import Sofa
import SofaRuntime
from scipy import sparse
from scipy import linalg
#from matplotlib import pyplot as plt
import numpy as np
from functools import wraps

from kalmanFilterForce1d import kalmanFilter
from getEmpiricalForces import getProbingForces
from principalDirectionFunction import calculatePrincipalComponents
from nonlinearization import makeNonlinear

#OPTIONS BEGIN******************************************************************************************

# Enable/disable contact force extraction
extractContactForce = False
exportCSV = True

#showImage = True

# Select  anatomy to include
addPortal = False
addPCL = True
addACL = True
addTibia = True
addFemur = True
addSkin = False
addLateralMeniscus = True
addLateralMeniscusWithTear = False
addMedialMeniscus = True
addCartilage = False 
addCollateralLigaments = True
addMuscle = True
addTendons = False
addVeins = False
addArteries = True
addNerve = False

#Haptic rendering options
useKalmanFilter = False
sendToHapticDevice = False
useInstrumentPosition = True

#OPTIONS END******************************************************************************************

# Function called when the scene graph is being created
def createScene(rootNode):
     
    rootNode.dt=0.01 #0.01, 0.05, 0.005
    #rootNode.gravity=[0, -9.81e03, 0] 
    rootNode.gravity=[0, -9.81e3, 0] 

    confignode = rootNode.addChild("Config")
    confignode.addObject('RequiredPlugin', name="SofaMiscCollision", printLog=False)
    confignode.addObject('RequiredPlugin', name="SofaPython3", printLog=False)
    confignode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('DefaultPipeline', name="pipeline", depth="6", verbose="0")
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('DefaultContactManager', name="response", response="FrictionContactConstraint") #responseParams="mu=0.1")
    #rootNode.addObject('LocalMinDistance', alarmDistance="3.14", contactDistance="1.14", angleCone="0.0")
    rootNode.addObject('LocalMinDistance', alarmDistance="3.408", contactDistance="1.136", angleCone="0.0")
    rootNode.addObject('FreeMotionAnimationLoop')
    LCP = rootNode.addObject('LCPConstraintSolver', tolerance="0.001", maxIt="1000", computeConstraintForces="True", mu="0.000001")

    #Haptics
    rootNode.addObject('RequiredPlugin', name='Geomagic')
    driver = rootNode.addObject('GeomagicDriver', name='GeomagicDevice', deviceName="Default Device", scale="10", drawDeviceFrame="1", positionBase="-200 0 50", drawDevice="0", orientationBase="0 0.707 0 -0.707", maxInputForceFeedback = "3")  #scale attribute only affects position
    


    #FEMUR AND PATELLA NODE BEGIN*************************************************************************************************************************************************************************************
    if addFemur == True:
        femur = rootNode.addChild("femur")
        
        #Femur visual model
        femurVisu = femur.addChild("VisualModel")
        femurVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/FEMUR_AND_PATELLA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")
        femurVisu.addObject('OglModel', name="FemurVisualModel", src="@loader", color="white") 

        #Femur collision model
        femurCollision = femur.addChild('femurCollision')
        femurCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/FEMUR_AND_PATELLA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") 
        femurCollision.addObject('MeshTopology', src="@loader", name="FemurCollisionModel")
        femurCollision.addObject('MechanicalObject', src="@loader", name="FemurState")
        femurCollision.addObject('TriangleCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        femurCollision.addObject('LineCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        femurCollision.addObject('PointCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
    #FEMUR NODE END***************************************************************************************************************************************************************************************
    
    #TIBIA AND FIBULA NODE BEGIN*************************************************************************************************************************************************************************************
    if addTibia == True:
        tibia = rootNode.addChild("tibia")
        
        #Tibia visual model
        tibiaVisu = tibia.addChild("VisualModel")
        tibiaVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/TIBIA_AND_FIBULA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")
        tibiaVisu.addObject('OglModel', name="TibiaVisualModel", src="@loader", color="white") 

        #Tibia collision model
        tibiaCollision = tibia.addChild('tibiaCollision')
        tibiaCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/TIBIA_AND_FIBULA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") 
        tibiaCollision.addObject('MeshTopology', src="@loader", name="TibiaCollisionModel")
        tibiaCollision.addObject('MechanicalObject', src="@loader", name="TibiaState")
        tibiaCollision.addObject('TriangleCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        tibiaCollision.addObject('LineCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        tibiaCollision.addObject('PointCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
    #TIBIA NODE END***************************************************************************************************************************************************************************************
    


    #MEDIAL PORTAL NODE BEGIN*************************************************************************************************************************************************************************************
    if addPortal == True:
        medialPortal = rootNode.addChild('medialPortal')

        #Portal visual model
        portalVisual = medialPortal.addChild('portalVisualModel')
        #portalVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/torus.obj", rotation="0 180 90", translation="-2.5 -1 0.5", scale3d="0.2 0.2 0.2") #, handleSeams="1")
        portalVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/torus.obj", rotation="0 180 90", translation="-65 -28 0", scale3d="4.5 4.5 4.5") #, handleSeams="1")
        portalVisual.addObject('OglModel', name="portalVisualModel", src="@loader", color="0.6 0.5 0.4 0.7") 

        #Portal collision model
        portalCollision = medialPortal.addChild('portalCollisionModel')
        #portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="0 180 90", translation="-2.5 -1 0.5", scale3d="0.2 0.2 0.2") #triangulate="true", scale=50)
        portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="0 180 90", translation="-65 -28 0", scale3d="4.5 4.5 4.5")
        portalCollision.addObject('MeshTopology', src="@loader", name="portalCollisionModel")
        portalCollision.addObject('MechanicalObject', src="@loader", name="portalState")
        portalCollision.addObject('TriangleCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        portalCollision.addObject('LineCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        portalCollision.addObject('PointCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
    #MEDIAL PORTAL NODE END***************************************************************************************************************************************************************************************

    #LATERAL PORTAL NODE BEGIN*************************************************************************************************************************************************************************************
    if addPortal == True:
        lateralPortal = rootNode.addChild('lateralPortal')

        #Portal visual model
        portalVisual = lateralPortal.addChild('portalVisualModel')
        portalVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/torus.obj", rotation="30 180 90", translation="-60 -28 -38", scale3d="4.5 4.5 4.5") 
        portalVisual.addObject('OglModel', name="portalVisualModel", src="@loader", color="0.6 0.5 0.4 0.7") 

        #Portal collision model
        portalCollision = lateralPortal.addChild('portalCollisionModel')
        portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="30 180 90", translation="-60 -28 -38", scale3d="4.5 4.5 4.5")   
        portalCollision.addObject('MeshTopology', src="@loader", name="portalCollisionModel")
        portalCollision.addObject('MechanicalObject', src="@loader", name="portalState")
        portalCollision.addObject('TriangleCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        portalCollision.addObject('LineCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        portalCollision.addObject('PointCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
    #LATERAL PORTAL NODE END***************************************************************************************************************************************************************************************


    #SKIN NODE BEGIN*************************************************************************************************************************************************************************************
    if addSkin == True:
        skin = rootNode.addChild('skin')

        #Skin visual model
        skinVisual = skin.addChild('skinVisualModel')
        skinVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/SKIN2.obj",  rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") 
        skinVisual.addObject('OglModel', name="skinVisualModel", src="@loader", color="0.6 0.5 0.4 0.3") 
    #SKIN NODE

        
    #ACL NODE BEGIN**************************************************************************************************************************************************************************
    if addACL == True:    
        ACL = rootNode.addChild('ACL')
        #ACL.gravity = [0, -9.81, 0]
        ACL.addObject('EulerImplicitSolver', name="cg_odesolver")
        ACL.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        #ACL.addObject('MeshGmshLoader', name="meshLoaderACL", filename="mesh/ACL_computational_model.msh", rotation="0 180 60", translation="0.5 0.2 0", scale3d="0.5 0.5 0.5" )
        ACL.addObject('MeshGmshLoader', name="meshLoaderACL", filename="mesh/ACL_computational_model.msh", rotation="0 180 60", translation="6.36 4.54 0", scale3d="11.36 11.36 11.36" )
        ACL.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoaderACL")
        ACL.addObject('MechanicalObject', name="ACLDofs", src="@meshLoaderACL")
        ACL.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        ACL.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@ACLDofs")
        #ACL.addObject('MeshMatrixMass', name="mass", massDensity="10.9e-03", topology="@topo", geometryState="@ACLDofs")
        #ACL.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        ACL.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        ACL.addObject('PrecomputedConstraintCorrection', recompute="true")
        ACL.addObject('FixedConstraint', name="FixedConstraint", indices="2 3 10 11 12")

        #ACL visual model
        ACLVisual = ACL.addChild('ACL Visual Model') 
        ACLVisual.addObject('MeshOBJLoader', name="meshLoaderACL2", filename="mesh/ACL_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #ACLVisual.addObject('MeshOBJLoader', name="meshLoaderACL2", filename="mesh/ACL_.obj", rotation="0 180 30", translation="0 0 0", scale3d="11.36 11.36 11.36", handleSeams="1")
        ACLVisual.addObject('OglModel', name="VisualModel", src="@meshLoaderACL2", color="1.0 0.2 0.2 1.0")
        ACLVisual.addObject('BarycentricMapping', name="visual mapping", input="@../ACLDofs", output="@VisualModel")

        #ACL collision model (extracts triangular mesh from computation model)
        ACLcollision = ACL.addChild('ACL Collision Model')
        ACLcollision.addObject('TriangleSetTopologyContainer', name="Container")
        ACLcollision.addObject('TriangleSetTopologyModifier', name="Modifier")
        ACLcollision.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        ACLcollision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="1")

    #ACL NODE END****************************************************************************************************************************************************************************

    
    #PCL NODE BEGIN**************************************************************************************************************************************************************************
    if addPCL == True:
        PCL = rootNode.addChild('PCL')
        #PCL.gravity = [0, -9.81e-03, 0]
        PCL.addObject('EulerImplicitSolver', name="cg_odesolver")
        PCL.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        #PCL.addObject('MeshGmshLoader', name="meshLoaderPCL", filename="mesh/PCL_ComputationModel.msh", rotation="0 180 30", translation="0 0 0", scale3d="0.5 0.5 0.5" )
        PCL.addObject('MeshGmshLoader', name="meshLoaderPCL", filename="mesh/PCL_ComputationModel.msh", rotation="0 180 30", translation="-3 2 5", scale3d="11.36 11.36 11.36" )
        PCL.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoaderPCL")
        PCL.addObject('MechanicalObject', name="PCLDofs", src="@meshLoaderPCL")
        PCL.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        PCL.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@PCLDofs") 
        #PCL.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        PCL.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        PCL.addObject('PrecomputedConstraintCorrection', recompute="true")
        PCL.addObject('FixedConstraint', name="FixedConstraint", indices="1 2 3 6 19 30 33 38 63 86 101")

        #PCL visual model
        PCLVisual = PCL.addChild('PCL Visual Model') # Check if gravity and tags="Visual" should be added here
        PCLVisual.addObject('MeshOBJLoader', name="meshLoaderPCL2", filename="mesh/PCL_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #PCLVisual.addObject('MeshOBJLoader', name="meshLoaderPCL2", filename="mesh/PCL.obj", rotation="0 180 30", translation="0 0 0", scale3d="11.36 11.36 11.36", handleSeams="1")
        PCLVisual.addObject('OglModel', name="VisualModel", src="@meshLoaderPCL2", color="1.0 0.2 0.2 1.0")
        PCLVisual.addObject('BarycentricMapping', name="visual mapping", input="@../PCLDofs", output="@VisualModel")

        #PCL collision model (extracts triangular mesh from computation model)
        PCLcollision = PCL.addChild('PCL Collision Model')
        PCLcollision.addObject('TriangleSetTopologyContainer', name="Container")
        PCLcollision.addObject('TriangleSetTopologyModifier', name="Modifier")
        PCLcollision.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        PCLcollision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="1")

    #ACL NODE END****************************************************************************************************************************************************************************


    #MEDIAL MENISCUS NODE BEGIN**************************************************************************************************************************************************************************
    if addMedialMeniscus == True:
        medialMeniscus = rootNode.addChild('medialMeniscus')
        medialMeniscus.gravity = [0, -9.81e3, 0]
        medialMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        medialMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        #medialMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/medialMeniscusDummy_5.msh", rotation="90 270 -2", translation="-15 -38 35", scale3d = "26 26 26" )
        medialMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/medialMeniscusComputationModelSTP.msh", rotation="90 270 2", translation="-13 -38 35", scale3d = "27 27 27" )
        medialMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        medialMeniscusDOFS = medialMeniscus.addObject('MechanicalObject', name="medialMeniscusDofs", src="@meshLoader")
        medialMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        medialMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@medialMeniscusDofs") 
        #medialMeniscus.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        medialMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="16")
        medialMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")

        #Add boundary conditions (missing collision between bone and meniscus)
        #medialMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="3 30 33 35 38 40 41 42 43 44 45 46 47 50 52 53 55 61 69 70 71 72 73 74 75 76 77 78 79 80 81 85 86 88 89 90 91 92 94 115 116 117 119 121 122 147 148 150")
        medialMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="2 5 7 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87")

        #Medial meniscus visual model
        medialMeniscusVisual = medialMeniscus.addChild('Medial Meniscus Visual Model') # Check if gravity and tags="Visual" should be added here
        #medialMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_MM", filename="mesh/Medial_comparison.obj", rotation="90 270 -2", translation="-2.272 -40.896 31.808", scale3d="22.72 22.72 22.72", handleSeams="1")
        medialMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_MM", filename="mesh/MEDIAL_MENISCUS_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        medialMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_MM", color="1.0 0.2 0.2 1.0")
        medialMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../medialMeniscusDofs", output="@VisualModel")

        #Medial meniscus collision model (extracts triangular mesh from computation model)
        medialMeniscusTriangularSurface = medialMeniscus.addChild('Medial Meniscus Triangular Surface Model')
        medialMeniscusTriangularSurface.addObject('TriangleSetTopologyContainer', name="Container")
        medialMeniscusTriangularSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        medialMeniscusTriangularSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        medialMeniscusTriangularSurface.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")

    #MEDIAL MENISCUS NODE END****************************************************************************************************************************************************************************

    #LATERAL MENISCUS NODE BEGIN*************************************************************************************************************************************************************************
    if addLateralMeniscus == True:
        lateralMeniscus = rootNode.addChild('lateralMeniscus')
        lateralMeniscus.gravity = [0, -9.81e3, 0]
        lateralMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        lateralMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        #lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -1.5 -1.2")
        #lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -31.5 -26", scale3d="26 26 26")
        lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModelSTP.msh", rotation="90 270 -182", translation="0 -31.5 -26", scale3d="26 26 26")
        lateralMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="-0.5 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        lateralMeniscusDofs = lateralMeniscus.addObject('MechanicalObject', name="lateralMeniscusDofs", src="@meshLoader")
        lateralMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        lateralMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@lateralMeniscusDofs") # massDensity="10.9e-03"
        #lateralMeniscus.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="30", computeGlobalMatrix="0")
        lateralMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        lateralMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")

        #Add boundary conditions (missing collision between bone and meniscus)
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="70 72 73 74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 103 104 110 120 121 127 128 129 130 275 274 273 271")
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="1 129 130 132 70 74 271 273 93")
        lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="2 6 7 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 73")

        #Lateral meniscus visual model
        lateralMeniscusVisual = lateralMeniscus.addChild('Lateral Meniscus Visual Model')
        lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/LATERAL_MENISCUS_VISUAL_MODEL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/MeshLateralMeniscius_obj.obj", rotation="90 270 -182", translation="0 -36.352 -29.536", scale3d="22.72 22.72 22.72", handleSeams="1")
        lateralMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_LM", color="1.0 0.2 0.2 1.0")
        lateralMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../lateralMeniscusDofs", output="@VisualModel")

        #Lateral meniscus collision model (extracts triangular mesh from computation model)
        lateralMeniscusTriangularSurface = lateralMeniscus.addChild('Lateral Meniscus Triangular Surface Model')
        lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyContainer', name="Container")
        lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        lateralMeniscusTriangularSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        lateralMeniscusTriangularSurface.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")

    #LATERAL MENISCUS NODE END***************************************************************************************************************************************************************************

    #INJURED LATERAL MENISCUS NODE BEGIN*************************************************************************************************************************************************************************
    if addLateralMeniscusWithTear == True:
        tornLateralMeniscus = rootNode.addChild('tornLateralMeniscus')
        tornLateralMeniscus.gravity = [0, -9.81e3, 0]
        tornLateralMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        tornLateralMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        #lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -1.5 -1.2")
        tornLateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -31.5 -26", scale3d="26 26 26")
        tornLateralMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="-0.5 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        tornLateralMeniscus.addObject('MechanicalObject', name="lateralMeniscusDofs", src="@meshLoader")
        tornLateralMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        tornLateralMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@lateralMeniscusDofs") # massDensity="10.9e-03"
        #tornLateralMeniscus.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="30", computeGlobalMatrix="0")
        tornLateralMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        tornLateralMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")

        #Add boundary conditions (missing collision between bone and meniscus)
        tornLateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="70 72 73 74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 103 104 110 120 121 127 128 129 130 275 274 273 271")
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="1 129 130 132 70 74 271 273 93")

        #Lateral meniscus visual model
        tornLateralMeniscusVisual = tornLateralMeniscus.addChild('Lateral Meniscus Visual Model')
        tornLateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/LATERAL_MENISCUS_WITH_TEAR_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/MeshLateralMeniscius_obj.obj", rotation="90 270 -182", translation="0 -36.352 -29.536", scale3d="22.72 22.72 22.72", handleSeams="1")
        tornLateralMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_LM", color="1.0 0.2 0.2 1.0")
        tornLateralMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../lateralMeniscusDofs", output="@VisualModel")

        #Lateral meniscus collision model (extracts triangular mesh from computation model)
        tornLateralMeniscusCollision = tornLateralMeniscus.addChild('Lateral Meniscus Triangular Surface Model')
        tornLateralMeniscusCollision.addObject('TriangleSetTopologyContainer', name="Container")
        tornLateralMeniscusCollision.addObject('TriangleSetTopologyModifier', name="Modifier")
        tornLateralMeniscusCollision.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        tornLateralMeniscusCollision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")

    #INJURED LATERAL MENISCUS NODE END***************************************************************************************************************************************************************************



    #CARTILAGE NODE BEGIN**************************************************************************************************************************************************************************
    if addCartilage == True:
        cartilage = rootNode.addChild("cartilage")
        
        #Cartilage visual model
        cartilageVisu = cartilage.addChild("VisualModel")
        #cartilageVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/cartilage2.obj", rotation="0 176 0", translation="75 -93 50", scale3d="0.5 0.5 0.5") #Use this for collision purposes
        cartilageVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/CARTILAGE3.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #Only for visualization purposes
        cartilageVisu.addObject('OglModel', name="CartilageVisualModel", src="@loader", color="0.6 0.5 0.4 0.3") #scale3d="50 50 50", color="1 1 1", updateNormals=False)

        #Cartilage collision model
        #cartilageCollision = cartilage.addChild('cartilageCollision')
        #cartilageCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/cartilage2.obj", rotation="0 176 0", translation="75 -93 50", scale3d="0.5 0.5 0.5") #triangulate="true", scale=50)
        #cartilageCollision.addObject('MeshTopology', src="@loader", name="CartilageCollisionModel")
        #cartilageCollision.addObject('MechanicalObject', src="@loader", name="CartilageState")
        #cartilageCollision.addObject('TriangleCollisionModel', contactStiffness="1e-3", simulated="0", moving="0")
        #cartilageCollision.addObject('LineCollisionModel', contactStiffness="1e-3", simulated="0", moving="0")
        #cartilageCollision.addObject('PointCollisionModel', contactStiffness="1e-3", simulated="0", moving="0")
    #CARTILAGE NODE END****************************************************************************************************************************************************************************

    #COLLATERAL LIGAMENTS NODE BEGIN**************************************************************************************************************************************************************************
    if addCollateralLigaments == True:
        collateralLigaments = rootNode.addChild('collateral Ligaments')

        #Skin visual model
        collateralVisual = collateralLigaments.addChild('collateralVisualModel')
        #collateralVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/collateralLigaments.obj", rotation="-85 277 0", translation="-161 -128 -138", scale3d="0.91 0.91 0.91") #, handleSeams="1")
        collateralVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/COLLATERAL_LIGAMENTS.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        collateralVisual.addObject('OglModel', name="collateralVisualModel", src="@loader", color="0.8 0.3 0.3 1.0") 

    #COLLATERAL LIGAMENTS NODE END****************************************************************************************************************************************************************************

    #MUSCLE NODE BEGIN**************************************************************************************************************************************************************************
    if addMuscle == True:
        muscle = rootNode.addChild('Muscle')

        #Skin visual model
        muscleVisual = muscle.addChild('muscleVisualModel')
        #muscleVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/muscle.obj", rotation="-85 277 0", translation="-161 -128 -138", scale3d="1 1 1") #, handleSeams="1")
        muscleVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/muscle2.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        muscleVisual.addObject('OglModel', name="muscleVisualModel", src="@loader", color="0.6 0.3 0.3 1.0") 

    #MUSCLE LIGAMENTS NODE END****************************************************************************************************************************************************************************

    #TENDONDS NODE BEGIN**************************************************************************************************************************************************************************
    if addTendons == True:
        tendons = rootNode.addChild('tendons')

        #Skin visual model
        tendonVisual = muscle.addChild('tendonVisualModel')
        tendonVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/TENDONS.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        tendonVisual.addObject('OglModel', name="tendonVisualModel", src="@loader", color="0.6 0.3 0.3 0.2") 

    #TENNDONS NODE END****************************************************************************************************************************************************************************

    #ARTERIES NODE BEGIN**************************************************************************************************************************************************************************
    if addArteries == True:
        arteries = rootNode.addChild('arteries')

        #Skin visual model
        arteriesVisual = muscle.addChild('arteriesVisualModel')
        arteriesVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/ARTERIES.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        arteriesVisual.addObject('OglModel', name="arteriesVisualModel", src="@loader", color="0.24 0.24 0.60 1.0") 

    #ARTERIES NODE END****************************************************************************************************************************************************************************

    #VEINS NODE BEGIN**************************************************************************************************************************************************************************
    if addVeins == True:
        veins = rootNode.addChild('veins')

        #visual model
        veinsVisual = muscle.addChild('veinsVisualModel')
        veinsVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/VEINS.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        veinsVisual.addObject('OglModel', name="veinsVisualModel", src="@loader", color="0.24 0.24 0.4 0.1") 

    #VEINS NODE END****************************************************************************************************************************************************************************

    #NERVE NODE BEGIN**************************************************************************************************************************************************************************
    if addNerve == True:
        nerve = rootNode.addChild('nerve')

        #visual model
        nerveVisual = muscle.addChild('nerveVisualModel')
        nerveVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/NERVE.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") #, handleSeams="1")
        nerveVisual.addObject('OglModel', name="nerveVisualModel", src="@loader", color="0.24 0.24 0.4 1.0") 

    #NERVE NODE END****************************************************************************************************************************************************************************

    #OMNI NODE BEGIN*************************************************************************************************************************************************************************************
    omni = rootNode.addChild('Omni')
    omniDOFs = omni.addObject('MechanicalObject', name='DOFs', template='Rigid3d', position='@GeomagicDevice.positionDevice')
    #OMNI NODE END*********************************************************************************************************************************************************************************************

    #INSTRUMENT NODE BEGIN*************************************************************************************************************************************************************************************
    instrument = rootNode.addChild('Instrument')
    instrument.addObject('EulerImplicitSolver', name="ODE solver") 
    instrument.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
    #instrument.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1.0e-5", threshold="10.0e-5")
    instrument.addObject('MechanicalObject', name='instrumentState', template='Rigid3d')
    instrument.addObject('UniformMass', name="mass", totalMass="3.5e-05") #[kg] 3.5e-05 Mass of instrument measured to be 35 g. 
    #instrument.addObject('DiagonalMass', name="mass", massDensity="3.5e-05", topology="@topo", geometryState="@instrumentState") #
    if sendToHapticDevice == False:
        instrument.addObject('LCPForceFeedback', activate="true", forceCoef="1.0") #Multiply haptic force by this coefficient to scale force in haptic device. 
    #instrument.addObject('UncoupledConstraintCorrection')
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

    return rootNode


    

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()