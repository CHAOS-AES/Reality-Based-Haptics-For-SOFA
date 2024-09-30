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
import time
from scipy import sparse
from scipy import linalg
from matplotlib import pyplot as plt
import numpy as np
from functools import wraps

from kalmanFilterForce1d import kalmanFilter
from getEmpiricalForces import getProbingForces
from principalDirectionFunction import calculatePrincipalComponents
from nonlinearization import makeNonlinear

#OPTIONS BEGIN******************************************************************************************

# Enable/disable carving
carving = True

# Enable/disable contact force extraction
extractContactForce = True
exportCSV = True

# Select  anatomy to include
addPortal = True
addCuttingPortal = True
addPCL = True
addACL = True
addTibia = True
addFemur = True
addSkin = False
addLateralMeniscus = True
addLateralMeniscusWithTear = False
addMedialMeniscus = True
addCartilage = True 
addCollateralLigaments = True
addMuscle = True
addTendons = False
addVeins = False
addArteries = False
addNerve = False

#Haptic rendering options
useKalmanFilter = True
sendToHapticDevice = True
useInstrumentPosition = True
#useScalpel = True

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
    
    if carving == True:
        #rootNode.addObject('DefaultVisualManagerLoop')
        #rootNode.addObject('DefaultAnimationLoop')
        #rootNode.addObject('FreeMotionAnimationLoop')
        #rootNode.addObject('CollisionPipeline', verbose=False, draw=False)
        #rootNode.addObject('BruteForceBroadPhase')
        #rootNode.addObject('BVHNarrowPhase', name="narrowPhase")  
        #rootNode.addObject('ParallelBruteForceBroadPhase')
        #rootNode.addObject('ParallelBVHNarrowPhase', name="narrowPhase")    
        
        #rootNode.addObject('DefaultContactManager', name="response", response="FrictionContactConstraint") #responseParams="mu=0.1")
        #rootNode.addObject('LocalMinDistance', alarmDistance="3.14", contactDistance="1.14", angleCone="0.0")
        #rootNode.addObject('LocalMinDistance', alarmDistance="3.408", contactDistance="1.136", angleCone="0.0")
        
        #LCP = rootNode.addObject('LCPConstraintSolver', tolerance="0.001", maxIt="1000", computeConstraintForces="True", mu="0.000001")

        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('DefaultPipeline', name="pipeline", depth="6", verbose="0")
        rootNode.addObject('BruteForceBroadPhase')
        rootNode.addObject('BVHNarrowPhase')
        #rootNode.addObject('MinProximityIntersection', name="Proximity", alarmDistance=0.08, contactDistance=0.05, useSurfaceNormals=False)
        rootNode.addObject('DefaultContactManager', name="response", response="FrictionContactConstraint") #responseParams="mu=0.1")
        rootNode.addObject('LocalMinDistance', alarmDistance="3.408", contactDistance="1.136", angleCone="0.0")
        rootNode.addObject('FreeMotionAnimationLoop')
        LCP = rootNode.addObject('LCPConstraintSolver', tolerance="0.001", maxIt="1000", computeConstraintForces="True", mu="0.000001")

        # Add the CarvingManger object, linking the collision pipeline, as well as the collision model of the tool used to carve. The collisions models to be carved are found using the tags: CarvingSurface.
        # the carvingDistance need to be lower than the contactDistance of the collision pipeline. 
        rootNode.addObject('CarvingManager', active=True, carvingDistance = 0.6) # carvingDistance=1.136)
    else:
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('DefaultPipeline', name="pipeline", depth="6", verbose="0")
        rootNode.addObject('BruteForceBroadPhase')
        rootNode.addObject('BVHNarrowPhase')
        rootNode.addObject('DefaultContactManager', name="response", response="FrictionContactConstraint") #responseParams="mu=0.1")
        rootNode.addObject('LocalMinDistance', alarmDistance="3.408", contactDistance="1.136", angleCone="0.0")
        rootNode.addObject('FreeMotionAnimationLoop')
        LCP = rootNode.addObject('LCPConstraintSolver', tolerance="0.001", maxIt="1000", computeConstraintForces="True", mu="0.000001")

    
    #Camera
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')
    #rootNode.addObject('OglSceneFrame', name='sceneFrame', backgroundColor=[0.0, 0.0, 0.0, 1.0])  # RGBA, black
    rootNode.addObject('InteractiveCamera',name='camera') #, position='0 0 500', orientation='0 0 0', fovy='60', znear='1', zfar='1000') 
    rootNode.camera.position.value = [-350,30,-50]

    #Haptics
    rootNode.addObject('RequiredPlugin', name='Geomagic')
    driver = rootNode.addObject('GeomagicDriver', name='GeomagicDevice', deviceName="Default Device", scale="10", drawDeviceFrame="1", positionBase="-150 0 70", drawDevice="0", orientationBase="0 0.707 0 -0.707", maxInputForceFeedback = "3") #, forceFeedBack="@instrument/LCPFF")  #scale attribute only affects position
    


    #FEMUR AND PATELLA NODE BEGIN*************************************************************************************************************************************************************************************
    if addFemur == True:
        femur = rootNode.addChild("femur")
        
        #Femur visual model
        femurVisu = femur.addChild("VisualModel")
        femurVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/FEMUR_AND_PATELLA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")
        femurVisu.addObject('OglModel', name="FemurVisualModel", src="@loader", color="white") 

        #Femur collision model
        #femurCollision = femur.addChild('femurCollision')
        #femurCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/FEMUR_AND_PATELLA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") 
        #femurCollision.addObject('MeshTopology', src="@loader", name="FemurCollisionModel")
        #femurCollision.addObject('MechanicalObject', src="@loader", name="FemurState")
        #femurCollision.addObject('TriangleCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        #femurCollision.addObject('LineCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        #femurCollision.addObject('PointCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
    #FEMUR NODE END***************************************************************************************************************************************************************************************
    
    #TIBIA AND FIBULA NODE BEGIN*************************************************************************************************************************************************************************************
    if addTibia == True:
        tibia = rootNode.addChild("tibia")
        
        #Tibia visual model
        tibiaVisu = tibia.addChild("VisualModel")
        tibiaVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/TIBIA_AND_FIBULA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")
        tibiaVisu.addObject('OglModel', name="TibiaVisualModel", src="@loader", color="white") 

        #Tibia collision model
        #tibiaCollision = tibia.addChild('tibiaCollision')
        #tibiaCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/TIBIA_AND_FIBULA_MEDULLAR.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545") 
        #tibiaCollision.addObject('MeshTopology', src="@loader", name="TibiaCollisionModel")
        #tibiaCollision.addObject('MechanicalObject', src="@loader", name="TibiaState")
        #tibiaCollision.addObject('TriangleCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        #tibiaCollision.addObject('LineCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
        #tibiaCollision.addObject('PointCollisionModel', contactStiffness="4.4e-9", simulated="0", moving="0")
    #TIBIA NODE END***************************************************************************************************************************************************************************************
    
    if carving == True:
        #MEDIAL SKIN NODE BEGIN***************************************************************************************************************************************************************************************
        medialSkin = rootNode.addChild('medialSkin')
        medialSkin.addObject('EulerImplicitSolver',name="cg_odesolver", rayleighStiffness=0.1, rayleighMass=0.1)#, printLog=False, rayleighStiffness=0.1, rayleighMass=0.1)
        medialSkin.addObject('CGLinearSolver', name="linear solver", iterations=25, tolerance=1.0e-9, threshold=1.0e-9)
        medialSkin.addObject('MeshGmshLoader', name="loader", filename="mesh/SKIN_Computation_Model_.msh", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")
        #medialSkin.addObject('MeshGmshLoader', name="loader", filename="mesh/SKIN_Computation_Model_refined.msh", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545")


        medialSkinDofs = medialSkin.addObject('MechanicalObject', template="Vec3d", name="Volume", src="@loader")
        
        medialSkin.addObject('TetrahedronSetTopologyContainer', name="topo", src="@loader")
        medialSkin.addObject('TetrahedronSetTopologyModifier', name="topoMod")
        medialSkin.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        
        medialSkin.addObject('DiagonalMass', massDensity=1.09e-9)
        medialSkin.addObject('FixedConstraint', indices=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 44, 45, 73, 74, 75, 76, 77, 80, 81, 98, 119, 123, 125, 130, 150, 151, 163, 176, 178, 180, 182, 189, 194, 196, 199, 202, 213, 216, 291, 301, 305, 319])
        #medialSkin.addObject('FixedConstraint', indices=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 73, 74, 75, 76, 77, 80, 81, 98, 118, 119, 120, 123, 124, 125, 129, 130, 150, 151, 163, 176, 178, 180, 182, 189, 194, 196, 199, 202, 213, 216, 291, 301, 305, 319])
        medialSkin.addObject('TetrahedralCorotationalFEMForceField', name="CFEM", youngModulus=15, poissonRatio=0.3, method="large")
        
        #medialSkin.addObject('UncoupledConstraintCorrection') #Efficient, inaccurate with coupled constraints
        #medialSkin.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
        #medialSkin.addObject('LinearSolverConstraintCorrection')   #Accurate. Inefficient with many constraints.
        medialSkin.addObject('PrecomputedConstraintCorrection', recompute="true") 

        # Add corresponding surface topology
        medialSkinSurface = medialSkin.addChild('medialSkinSurface')    
        medialSkinSurface.addObject('TriangleSetTopologyContainer', name="Container")
        medialSkinSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        medialSkinSurface.addObject('TriangleSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        medialSkinSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        
        medialSkinSurface.addObject('TriangleCollisionModel', tags="CarvingSurface")

        medialSkinVisual = medialSkinSurface.addChild('VisualModel')
        medialSkinVisual.addObject('OglModel', name="Visual", color="0.6 0.5 0.4 1.0") # material="Default Diffuse 1 0 1 0 1 Ambient 0 1 1 1 1 Specular 1 1 1 0 1 Emissive 0 1 1 0 1 Shininess 1 100")
        medialSkinVisual.addObject('IdentityMapping', input="@../../Volume", output="@Visual")
        #MEDIAL SKIN NODE END***************************************************************************************************************************************************************************************

    #MEDIAL PORTAL NODE BEGIN*************************************************************************************************************************************************************************************
    if addPortal == True:
        medialPortal = rootNode.addChild('medialPortal')

        #Portal visual model
        portalVisual = medialPortal.addChild('portalVisualModel')
        portalVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/torus.obj", rotation="-20 180 120", translation="-80 -28 0", scale3d="4.5 4.5 4.5") #, handleSeams="1")
        portalVisual.addObject('OglModel', name="portalVisualModel", src="@loader", color="0.47 0.96 0.89 0.96") 

        #Portal collision model
        #portalCollision = medialPortal.addChild('portalCollisionModel')
        #portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="0 180 90", translation="-2.5 -1 0.5", scale3d="0.2 0.2 0.2") #triangulate="true", scale=50)
        #portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="0 180 90", translation="-65 -28 0", scale3d="4.5 4.5 4.5")
        #portalCollision.addObject('MeshTopology', src="@loader", name="portalCollisionModel")
        #portalCollision.addObject('MechanicalObject', src="@loader", name="portalState")
        #portalCollision.addObject('TriangleCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        #portalCollision.addObject('LineCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        #portalCollision.addObject('PointCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
    #MEDIAL PORTAL NODE END***************************************************************************************************************************************************************************************

    #LATERAL PORTAL NODE BEGIN*************************************************************************************************************************************************************************************
    if addPortal == True:
        lateralPortal = rootNode.addChild('lateralPortal')

        #Portal visual model
        portalVisual = lateralPortal.addChild('portalVisualModel')
        portalVisual.addObject('MeshOBJLoader', name="loader", filename="mesh/torus.obj", rotation="60 180 110", translation="-60 -28 -45", scale3d="4.5 4.5 4.5") #scale 4.5
        portalVisual.addObject('OglModel', name="portalVisualModel", src="@loader", color="0.47 0.96 0.89 0.96") 

        #Portal collision model
        #portalCollision = lateralPortal.addChild('portalCollisionModel')
        #portalCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/torus_for_collision.obj", triangulate = "true", rotation="30 180 90", translation="-60 -28 -38", scale3d="4.5 4.5 4.5")   
        #portalCollision.addObject('MeshTopology', src="@loader", name="portalCollisionModel")
        #portalCollision.addObject('MechanicalObject', src="@loader", name="portalState")
        #portalCollision.addObject('TriangleCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        #portalCollision.addObject('LineCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
        #portalCollision.addObject('PointCollisionModel', contactStiffness="4.4", simulated="0", moving="0")
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
        # ACL.addObject('EulerImplicitSolver', name="cg_odesolver")
        # ACL.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        # #ACL.addObject('MeshGmshLoader', name="meshLoaderACL", filename="mesh/ACL_computational_model.msh", rotation="0 180 60", translation="0.5 0.2 0", scale3d="0.5 0.5 0.5" )
        # ACL.addObject('MeshGmshLoader', name="meshLoaderACL", filename="mesh/ACL_computational_model.msh", rotation="0 180 60", translation="6.36 4.54 0", scale3d="11.36 11.36 11.36" )
        # ACL.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoaderACL")
        # ACL.addObject('MechanicalObject', name="ACLDofs", src="@meshLoaderACL")
        # ACL.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        # ACL.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@ACLDofs")
        # #ACL.addObject('MeshMatrixMass', name="mass", massDensity="10.9e-03", topology="@topo", geometryState="@ACLDofs")
        # #ACL.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        # ACL.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        # ACL.addObject('PrecomputedConstraintCorrection', recompute="true")
        # ACL.addObject('FixedConstraint', name="FixedConstraint", indices="2 3 10 11 12")

        #ACL visual model
        ACLVisual = ACL.addChild('ACL Visual Model') 
        ACLVisual.addObject('MeshOBJLoader', name="meshLoaderACL2", filename="mesh/ACL_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #ACLVisual.addObject('MeshOBJLoader', name="meshLoaderACL2", filename="mesh/ACL_.obj", rotation="0 180 30", translation="0 0 0", scale3d="11.36 11.36 11.36", handleSeams="1")
        ACLVisual.addObject('OglModel', name="VisualModel", src="@meshLoaderACL2", color="1.0 0.2 0.2 1.0")
        #ACLVisual.addObject('BarycentricMapping', name="visual mapping", input="@../ACLDofs", output="@VisualModel")

        #ACL collision model (extracts triangular mesh from computation model)
        # ACLcollision = ACL.addChild('ACL Collision Model')
        # ACLcollision.addObject('TriangleSetTopologyContainer', name="Container")
        # ACLcollision.addObject('TriangleSetTopologyModifier', name="Modifier")
        # ACLcollision.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        # ACLcollision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="1")

    #ACL NODE END****************************************************************************************************************************************************************************

    
    #PCL NODE BEGIN**************************************************************************************************************************************************************************
    if addPCL == True:
        PCL = rootNode.addChild('PCL')
        #PCL.gravity = [0, -9.81e-03, 0]
        # PCL.addObject('EulerImplicitSolver', name="cg_odesolver")
        # PCL.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        # #PCL.addObject('MeshGmshLoader', name="meshLoaderPCL", filename="mesh/PCL_ComputationModel.msh", rotation="0 180 30", translation="0 0 0", scale3d="0.5 0.5 0.5" )
        # PCL.addObject('MeshGmshLoader', name="meshLoaderPCL", filename="mesh/PCL_ComputationModel.msh", rotation="0 180 30", translation="-3 2 5", scale3d="11.36 11.36 11.36" )
        # PCL.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoaderPCL")
        # PCL.addObject('MechanicalObject', name="PCLDofs", src="@meshLoaderPCL")
        # PCL.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        # PCL.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@PCLDofs") 
        # #PCL.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        # PCL.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="30")
        # PCL.addObject('PrecomputedConstraintCorrection', recompute="true")
        # PCL.addObject('FixedConstraint', name="FixedConstraint", indices="1 2 3 6 19 30 33 38 63 86 101")

        #PCL visual model
        PCLVisual = PCL.addChild('PCL Visual Model') # Check if gravity and tags="Visual" should be added here
        PCLVisual.addObject('MeshOBJLoader', name="meshLoaderPCL2", filename="mesh/PCL_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #PCLVisual.addObject('MeshOBJLoader', name="meshLoaderPCL2", filename="mesh/PCL.obj", rotation="0 180 30", translation="0 0 0", scale3d="11.36 11.36 11.36", handleSeams="1")
        PCLVisual.addObject('OglModel', name="VisualModel", src="@meshLoaderPCL2", color="1.0 0.2 0.2 1.0")
        #PCLVisual.addObject('BarycentricMapping', name="visual mapping", input="@../PCLDofs", output="@VisualModel")

        #PCL collision model (extracts triangular mesh from computation model)
        # PCLcollision = PCL.addChild('PCL Collision Model')
        # PCLcollision.addObject('TriangleSetTopologyContainer', name="Container")
        # PCLcollision.addObject('TriangleSetTopologyModifier', name="Modifier")
        # PCLcollision.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        # PCLcollision.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="1")

    #ACL NODE END****************************************************************************************************************************************************************************


    #MEDIAL MENISCUS NODE BEGIN**************************************************************************************************************************************************************************
    if addMedialMeniscus == True:
        medialMeniscus = rootNode.addChild('medialMeniscus')
        # medialMeniscus.gravity = [0, -9.81e3, 0]
        # medialMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        # medialMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        # #medialMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/medialMeniscusDummy_5.msh", rotation="90 270 -2", translation="-15 -38 35", scale3d = "26 26 26" )
        # medialMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/medialMeniscusComputationModelSTP.msh", rotation="90 270 2", translation="-13 -38 35", scale3d = "27 27 27" )
        # medialMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="0 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        # medialMeniscusDOFS = medialMeniscus.addObject('MechanicalObject', name="medialMeniscusDofs", src="@meshLoader")
        # medialMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        # medialMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@medialMeniscusDofs") 
        # #medialMeniscus.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="150", computeGlobalMatrix="0")
        # medialMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="5")
        # medialMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")

        #Add boundary conditions (missing collision between bone and meniscus)
        #medialMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="3 30 33 35 38 40 41 42 43 44 45 46 47 50 52 53 55 61 69 70 71 72 73 74 75 76 77 78 79 80 81 85 86 88 89 90 91 92 94 115 116 117 119 121 122 147 148 150")
        #medialMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="2 5 7 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87")

        #Medial meniscus visual model
        medialMeniscusVisual = medialMeniscus.addChild('Medial Meniscus Visual Model') # Check if gravity and tags="Visual" should be added here
        #medialMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_MM", filename="mesh/Medial_comparison.obj", rotation="90 270 -2", translation="-2.272 -40.896 31.808", scale3d="22.72 22.72 22.72", handleSeams="1")
        medialMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_MM", filename="mesh/MEDIAL_MENISCUS_VISUAL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        medialMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_MM", color="1.0 0.2 0.2 1.0")
        #medialMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../medialMeniscusDofs", output="@VisualModel")

        #Medial meniscus collision model (extracts triangular mesh from computation model)
        # medialMeniscusTriangularSurface = medialMeniscus.addChild('Medial Meniscus Triangular Surface Model')
        # medialMeniscusTriangularSurface.addObject('TriangleSetTopologyContainer', name="Container")
        # medialMeniscusTriangularSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        # medialMeniscusTriangularSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        # medialMeniscusTriangularSurface.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")

    #MEDIAL MENISCUS NODE END****************************************************************************************************************************************************************************

    #LATERAL MENISCUS NODE BEGIN*************************************************************************************************************************************************************************
    if addLateralMeniscus == True:
        lateralMeniscus = rootNode.addChild('lateralMeniscus')
        # lateralMeniscus.gravity = [0, -9.81e3, 0]
        # lateralMeniscus.addObject('EulerImplicitSolver', name="cg_odesolver")
        # lateralMeniscus.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1e-09", threshold="1e-09")
        # #lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -1.5 -1.2")
        # #lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModel.msh", rotation="90 270 -182", translation="0 -31.5 -26", scale3d="26 26 26")
        # lateralMeniscus.addObject('MeshGmshLoader', name="meshLoader", filename="mesh/lateralMeniscusComputationModelSTP.msh", rotation="90 270 -182", translation="0 -31.5 -26", scale3d="26 26 26")
        # lateralMeniscus.addObject('TetrahedronSetTopologyContainer', name="topo", position="-0.5 0 0    0.542 0.455 0.542 0.455",  src="@meshLoader")
        # lateralMeniscusDOFS = lateralMeniscus.addObject('MechanicalObject', name="lateralMeniscusDofs", src="@meshLoader")
        # lateralMeniscus.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
        # lateralMeniscus.addObject('DiagonalMass', name="mass", massDensity="1.09e-09", topology="@topo", geometryState="@lateralMeniscusDofs") # massDensity="10.9e-03"
        # #lateralMeniscus.addObject('TetrahedronFEMForceField', template="Vec3d", name="FEM", method="large", poissonRatio="0.3", youngModulus="30", computeGlobalMatrix="0")
        # lateralMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="3")
        # lateralMeniscus.addObject('PrecomputedConstraintCorrection', recompute="true")

        #Add boundary conditions (missing collision between bone and meniscus)
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="70 72 73 74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 103 104 110 120 121 127 128 129 130 275 274 273 271")
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="1 129 130 132 70 74 271 273 93")
        #lateralMeniscus.addObject('FixedConstraint', name="FixedConstraint", indices="2 6 7 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 73")

        #Lateral meniscus visual model
        lateralMeniscusVisual = lateralMeniscus.addChild('Lateral Meniscus Visual Model')
        lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/LATERAL_MENISCUS_VISUAL_MODEL.obj", rotation="-85 277 -185", translation="120 105 -80", scale3d="0.0545 0.0545 0.0545", handleSeams="1")
        #lateralMeniscusVisual.addObject('MeshOBJLoader', name="meshLoader_LM", filename="mesh/MeshLateralMeniscius_obj.obj", rotation="90 270 -182", translation="0 -36.352 -29.536", scale3d="22.72 22.72 22.72", handleSeams="1")
        lateralMeniscusVisual.addObject('OglModel', name="VisualModel", src="@meshLoader_LM", color="1.0 0.2 0.2 1.0")
        #lateralMeniscusVisual.addObject('BarycentricMapping', name="visual mapping", input="@../lateralMeniscusDofs", output="@VisualModel")

        #Lateral meniscus collision model (extracts triangular mesh from computation model)
        # lateralMeniscusTriangularSurface = lateralMeniscus.addChild('Lateral Meniscus Triangular Surface Model')
        # lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyContainer', name="Container")
        # lateralMeniscusTriangularSurface.addObject('TriangleSetTopologyModifier', name="Modifier")
        # lateralMeniscusTriangularSurface.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
        # lateralMeniscusTriangularSurface.addObject('TriangleCollisionModel', name="CollisionModel", contactStiffness="0.13")

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
        tornLateralMeniscus.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="FEM", poissonRatio="0.3", youngModulus="3")
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
        cartilageVisu.addObject('OglModel', name="CartilageVisualModel", src="@loader", color="0.6 0.5 0.4 1.0") #scale3d="50 50 50", color="1 1 1", updateNormals=False)

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

    if carving == True:
        #INSTRUMENT NODE BEGIN*************************************************************************************************************************************************************************************
        instrument = rootNode.addChild('Instrument')
        instrument.addObject('EulerImplicitSolver', name="ODE solver") 
        instrument.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
        #instrument.addObject('CGLinearSolver', name="linear solver", iterations="25", tolerance="1.0e-9", threshold="1.0e-9")
        instrumentStates = instrument.addObject('MechanicalObject', name='instrumentState', template='Rigid3d')
        instrument.addObject('UniformMass', name="mass", totalMass="3.5e-05") #[kg] 3.5e-05 Mass of instrument measured to be 35 g. 
        #instrument.addObject('DiagonalMass', name="mass", massDensity="3.5e-05", topology="@topo", geometryState="@instrumentState") #
        if sendToHapticDevice == False:
            instrument.addObject('LCPForceFeedback', name='LCPFF', activate="true", forceCoef="1.0") #Multiply haptic force by this coefficient to scale force in haptic device. 
        #instrument.addObject('UncoupledConstraintCorrection')
        instrument.addObject('LinearSolverConstraintCorrection')
        instrument.addObject('RestShapeSpringsForceField', stiffness="1000000", angularStiffness="1000000", external_rest_shape="@/Omni", points="0")
               
        #Instrument visual model
        instrumentVisu = instrument.addChild('InstrumentVisualModel')
        instrumentVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/scalpel.obj", scale3d="1 1 1", handleSeams="1")
        instrumentVisu.addObject('OglModel', name="InstrumentVisualmodel", src="@loader", color="gray", rx="0", ry="0", rz="180", dy="0", dz="0")
        instrumentVisu.addObject('RigidMapping', name="MM->VM mapping", input="@instrumentState", output="@InstrumentVisualModel") 
                
        #Scalpel blade collision model
        instrumentCollision = instrument.addChild('instrumentCollision')
        instrumentCollision.addObject('MeshOBJLoader', name="loader", filename="mesh/scalpelBlade.obj", scale3d="1 1 1", handleSeams="1")
        instrumentCollision.addObject('MeshTopology', src="@loader")
        instrumentCollision.addObject('MechanicalObject', name="instrumentCollisionState1", src="@loader",rx="0", ry="0", rz="180", dy="0", dz="0")
        instrumentCollision.addObject('SphereCollisionModel', name='ParticleModel', radius=1.5, tags="CarvingTool")
        instrumentCollision.addObject('LineCollisionModel')
        instrumentCollision.addObject('RigidMapping', name="MM->CM mapping", input="@instrumentState", output="@instrumentCollisionState1")

        #INSTRUMENT NODE END**********************************************************************************************************************************************************************************************

        
    else:
        #INSTRUMENT NODE BEGIN*************************************************************************************************************************************************************************************
        instrument = rootNode.addChild('Instrument')
        instrument.addObject('EulerImplicitSolver', name="ODE solver") 
        instrument.addObject('AsyncSparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
        #instrument.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
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

    #Access data
    if extractContactForce == True: 
        rootNode.addObject(MatrixAccessController('MatrixAccessor', name='matrixAccessor', target=medialSkinDofs, target2=LCP, target3=omniDOFs, target4=driver, target5=instrumentStates, rootNode=rootNode))

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

        #Define haptic cutting data
        self.portalData = np.genfromtxt('portalDataSofa.txt', delimiter='\t')
        #self.portalData = np.genfromtxt('portalData_LK1_medial.txt', delimiter=',')
        self.cuttingForceMagnitude = self.portalData[:,2]
        self.tearPoint = 10/3 #mm
        self.cuttingDisplacement = np.linspace(0, self.tearPoint, len(self.cuttingForceMagnitude))

        #self.cuttingPointStart = 0
        self.initialContactForceDirection = np.zeros((1,3))
        self.initialReactionForceVector = np.zeros((1,3))
        #self.cutFinished = False
        self.cutting = False
        self.cuttingTime = 0

        #User experiment metrics
        self.userOvershootStart = np.zeros((1,3))
        self.userOvershoot = np.zeros((1,3))
        self.measureOvershoot = False
        self.measurementTime = 0
        self.initialReactionForceVectorUserExperiment = np.zeros((1,3))

    def onAnimateEndEvent(self, event):
        
        currentTime = time.time()
        dt = currentTime - self.previousTime
        self.previousTime = currentTime
        self.cuttingTime += dt
        self.measurementTime += dt

        #STEP 0: Initialize
        #constraint = self.target.getData("constraint") 
        constraint = self.target.constraint.value
        #constraint2 = self.target5.constraint.value
        forcesNorm = self.target2.constraintForces.value
        instrumentPosition = self.target3.getData("position")
        GeomagicDevice = self.rootNode.GeomagicDevice
        instrumentVelocityX = self.target5.velocity.value[0]
        #dt = self.rootNode.dt.value
        
        #Detect which anatomy is in contact
        #if len(constraint2) > 0:
        #    constraint = constraint2
        #else:
        #    constraint = constraint

        constraintMatrixInline = np.fromstring(constraint, sep='  ')

        pointId = []
        constraintId = []
        constraintDirections = []
        index = 0
        i = 0

        contactforce_x = 0
        contactforce_y = 0
        contactforce_z = 0

        #CALCULATE CONTACT FORCES START*********************************************************************************************************************************
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

            #STEP 5: GO THROUGH LIST OF NBDOFs AND COMPUTE FORCES
            for i in range (nbDofs):
                contactforce_x += forces[i][0]
                contactforce_y += forces[i][1]
                contactforce_z += forces[i][2]

            reactionForceVector = np.array(([contactforce_x, contactforce_y, contactforce_z]))
            reactionForceVector = np.reshape(reactionForceVector, (1,3))

            #STEP 6: DETECT CUTTING
            if not self.cutting and np.linalg.norm(reactionForceVector) > 0.1 and instrumentVelocityX[0] > 0:
                
                self.initialReactionForceVector = reactionForceVector
                
                #Get displacement of contact node
                instrumentPos = instrumentPosition.value #Get instrument position (x,y,z)
                nodePosition = instrumentPos[0, 0:3]
                nodePosition = np.array([nodePosition]) #Convert to array (1x3)
                self.contactPointStart = nodePosition
                contactPointDisplacement = np.zeros((1,3))

                #User experiment measurement
                self.userOvershootStart = nodePosition
                self.initialReactionForceVectorUserExperiment = reactionForceVector
                self.measureOvershoot = True

                [projectedContactPointDisplacement, reactionForceMagnitude, self.initialContactForceDirection] = calculatePrincipalComponents(contactPointDisplacement, self.initialReactionForceVector)   
                #print('Initial contact point position: ' + str(self.contactPointStart) + ' Inital reaction force vector: ' + str(self.initialReactionForceVector) + ' Initial contact force direction: ' + str(self.initialContactForceDirection))
                self.cutting = True

                
        else: #If there is no interaction between object and instrument, there is no reaction force
            contactforce_x = 0
            contactforce_y = 0
            contactforce_z = 0   

            reactionForceVector = np.array(([contactforce_x, contactforce_y, contactforce_z]))
            reactionForceVector = np.reshape(reactionForceVector, (1,3))
        
        #reactionForceVector = reactionForceVector*(1/30) #compensate for haptic device scale
        #print('Reaction Force Vector: ' + str(reactionForceVector))

        #CALCULATE CONTACT FORCES END*********************************************************************************************************************************


        #COMPUTE HAPTIC FORCE*******************************************************************************************************
        if ((self.cutting == True) & (useKalmanFilter == True)):
            
            #Get displacement of contact node
            instrumentPos = instrumentPosition.value #Get instrument position (x,y,z)
            nodePosition = instrumentPos[0, 0:3]
            nodePosition = np.array([nodePosition]) #Convert to array (1x3)

            contactPointDisplacement = nodePosition - self.contactPointStart
            #print('Contact point displacement: ' + str(contactPointDisplacement))

            #Get force magnitude and projected displacement along force vector of simulation reaction force
            [projectedContactPointDisplacement, reactionForceMagnitude, principalDirection] = calculatePrincipalComponents(contactPointDisplacement, self.initialReactionForceVector)   
            #print('Projected contact point displacement: ' + str(projectedContactPointDisplacement))
            
            if ((projectedContactPointDisplacement > self.tearPoint) or (self.cuttingTime > 5)): #Cutting is finished if displacement is greater than 10 mm or time is greater than 5 seconds
                self.cutting = False
                print("Cut is finished.")
                self.cuttingTime = 0
                return

            #Step 2c: Perform interpolation
            #empiricalForceMagnitude = getProbingForces(projectedContactPointDisplacement)
            empiricalForceMagnitude = np.interp(projectedContactPointDisplacement, self.cuttingDisplacement, self.cuttingForceMagnitude)

            #STEP 3: Perform Kalman Filter sensor fusion
            #kfForceMagnitude = kalmanFilter(reactionForceMagnitude, empiricalForceMagnitude, dt)
            kfForceMagnitude = empiricalForceMagnitude

            #Calculate haptic force
            scaleFactor = 1/(10/2) #0.25 #This value must be tuned manually. 
            #kfForceMagnitude = kfForceMagnitude*scaleFactor #Scale force to fit force capacity of haptic device
            hapticForceMagnitude = kfForceMagnitude*scaleFactor #Scale force

            maxHapticForce = 2.99
            if hapticForceMagnitude > maxHapticForce:  #Prevent excessive force on haptic device
                hapticForceMagnitude = maxHapticForce

            hapticForce = np.multiply(hapticForceMagnitude,self.initialContactForceDirection)

            #print('Haptic force: ' + str(hapticForce) + "Shape:"  + str(hapticForce.shape) + 'Data type' + str(hapticForce.dtype))

            #SEND HAPTIC FORCE TO DEVICE*******************************************************************************************************
            if sendToHapticDevice:
                GeomagicDevice.inputForceFeedback[0] = hapticForce[0] *(-1)
                GeomagicDevice.inputForceFeedback[1] = hapticForce[1] *(-1)
                GeomagicDevice.inputForceFeedback[2] = hapticForce[2] *(-1)
                #DEBUG:
                #print('inputForceFeedback: ' + str(GeomagicDevice.inputForceFeedback[0]) + str(GeomagicDevice.inputForceFeedback[1]) + str(GeomagicDevice.inputForceFeedback[2]))
                
        else:
            #self.contactOngoing = False
            contactPointDisplacement = np.zeros((1,3))
            projectedContactPointDisplacement = np.float64(0)
            reactionForceMagnitude = float(0)
            empiricalForceMagnitude = np.float64(0)
            kfForceMagnitude = np.float64(0)
            principalDirection = np.zeros(3)
            
            self.cutting = False
            self.contactPointStart = np.zeros((1,3))

            hapticForce = np.zeros((1,3))
            hapticForceMagnitude = np.float64(0)
            GeomagicDevice.inputForceFeedback[0] = 0
            GeomagicDevice.inputForceFeedback[1] = 0
            GeomagicDevice.inputForceFeedback[2] = 0

        #Save user experiment data
        if self.measureOvershoot:
            #Get displacement of contact node
            experimentInstrumentPos = instrumentPosition.value #Get instrument position (x,y,z)
            experimentInstrumentPos = experimentInstrumentPos[0, 0:3]
            experimentInstrumentPos = np.array([experimentInstrumentPos]) #Convert to array (1x3)

            self.userOvershoot = experimentInstrumentPos - self.userOvershootStart
            #print('User overshoot: ' + str(self.userOvershoot))

            #Get force magnitude and projected displacement along force vector of simulation reaction force
            [projectedUserOvershoot, tempReactionForce, tempDirectionVector] = calculatePrincipalComponents(self.userOvershoot, self.initialReactionForceVectorUserExperiment)   
            hapticForceMagnitudeUserExperiment = np.linalg.norm(hapticForce)
            print('Projected user overshoot [mm]: ' + str(projectedUserOvershoot) + ' Haptic Force [N]: ' + str(hapticForceMagnitudeUserExperiment))

            dataPackage3 = np.array([dt, projectedUserOvershoot, hapticForceMagnitudeUserExperiment])
            dataPackage3 = dataPackage3.reshape(1,-1)
            with open("C:/Users/objel/Documents/userStudyMeniscus/USER_DATA/userExperimentDataPortalCreation2.csv", "a") as csv_file:
                np.savetxt(csv_file, dataPackage3, delimiter=',', newline='\n', fmt='%f')

            if projectedUserOvershoot > 40:
                print('You have cut too far. Please start over.')

            if self.measurementTime > 3:
                self.measureOvershoot = False
                self.measurementTime = 0

                with open("C:/Users/objel/Documents/userStudyMeniscus/USER_DATA/userExperimentDataPortalCreation2.csv", "a") as csv_file:
                    csv_file.write('\nNEW CUT\n')



    

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()