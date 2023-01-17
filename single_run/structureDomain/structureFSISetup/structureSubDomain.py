# No license
# ----------

""" FEniCSV2019.1.0+ <-> MUI(mui4py) <-> MUI(C++) <-> OpenFOAMV6+ two way Coupling Code.

 Incompressible Navier-Stokes equations for fluid domain in OpenFOAM
 Structure dynamics equations for structure domain in FEniCS

 structureSubDomain.py is the sub-domain class of the structure code 
 located in the structureFSISetup sub-folder of the case folder

 Last changed: 25-September-2019"""

# BAE-FSI
# structureFSISetup/structureSubDomain.py

__author__ = "W.L"
__email__ = "wendi.liu@stfc.ac.uk"

__copyright__= "Copyright 2019 UK Research and Innovation " \
               "(c) Copyright IBM Corp. 2017, 2019"

# IBM Confidential
# OCO Source Materials
# 5747-SM3
# The source code for this program is not published or otherwise
# divested of its trade secrets, irrespective of what has
# been deposited with the U.S. Copyright Office.

__license__ = "All rights reserved"

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________

from dolfinx import *
import configparser
import numpy as np

#_________________________________________________________________________________________
#
#%% Import configure file
#_________________________________________________________________________________________

config = configparser.ConfigParser()
config.read('./structureFSISetup/structureInputPara.ini')

OBeamXtemp=float(config['GEOMETRY']['OBeamX'])
OBeamYtemp=float(config['GEOMETRY']['OBeamY'])
OBeamZtemp=float(config['GEOMETRY']['OBeamZ'])
XBeamtemp=float(config['GEOMETRY']['XBeam'])
YBeamtemp=float(config['GEOMETRY']['YBeam'])
ZBeamtemp=float(config['GEOMETRY']['ZBeam'])

#_________________________________________________________________________________________
#
#%% Define SubDomains classes
#%% for defining parts of the boundaries and the interior of the domain
#_________________________________________________________________________________________

class SubDomains:
    def Fixed (self , x ):
        return np.isclose(x[0], (OBeamXtemp))

    def Flex (self , x ):
        return np.isclose(x[0], (OBeamXtemp + XBeamtemp))

    def Symmetry (self , x ):
        return np.logical_or(np.logical_or(np.isclose(x[2], (OBeamZtemp + ZBeamtemp)), np.isclose(x[2], (OBeamZtemp))), np.logical_or(np.isclose(x[1], (OBeamYtemp)), np.isclose(x[1], (OBeamYtemp + YBeamtemp))))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#