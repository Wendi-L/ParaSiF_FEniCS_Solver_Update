# No license
# ----------

""" 
    This is a part of the Partitioned Multi-physical Simulation Framework (parMupSiF)

    FEniCSV2019.1.0+ <-> MUI(mui4py) <-> MUI(C++) <-> OpenFOAMV6+ two way Coupling Code.

    Incompressible Navier-Stokes equations for fluid domain in OpenFOAM
    Structure dynamics equations for structure domain in FEniCS

    __init__.py is the dundant init file of the structure code 
    located in the caseSetup/structureDomain/structureFSISetup sub-folder of the case folder

    Last changed: 25-September-2019
"""

# BAE-FSI
# structureFSISetup/__init__.py

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

from structureFSISetup.structureSubDomain import SubDomains
from structureFSISetup.structureBCS import boundaryConditions

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#