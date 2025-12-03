# Reality-Based Haptics for SOFA
Simulation scenes with implementation examples for conducting user study on meniscus examination and arthroscopic portal creation using the reality based method. Haptic feedback using 3D systems touch haptic device.
More details in paper "Haptic Rendering Using Reality-Based Force Profiles in Surgical Simulation": https://ieeexplore.ieee.org/abstract/document/11006002/ 

![image](https://github.com/user-attachments/assets/337e9c21-9a58-4a6f-82fb-3328a900ed87)

Disclaimer: The code is developed for SOFA v22. Parts of the code may not be compatible with later versions of SOFA.

To install necessary dependencies:
1. Install SOFA: https://www.sofa-framework.org/community/doc/getting-started/build/windows/
2. Install SofaPython3: https://sofapython3.readthedocs.io/en/latest/
3. Install the Geomagic Haptic plugin: https://www.sofa-framework.org/applications/plugins/geomagic-haptics/
4. Install SofaCarving plugin (only needed for portal creation): https://www.sofa-framework.org/community/doc/plugins/usual-plugins/sofacarving/ 


Once SOFA and the plugins above are installed, do the following steps:
- Put the 3D-MODELS in folder "C:/.../*yourSofaDirectory*/src/share/mesh/"
- Put all the Python files in the same folder somewhere on your hard drive, e.g. "C:/.../myPythonScripts/ "
- Open SOFA --> Click File --> Open --> C:/.../myPythonScripts/DEMO_PORTAL.py.py"
- Click "run", and the simulation should be running.

Note that you will need to select the OpenGL renderer from "view" to load the correct camera view.
