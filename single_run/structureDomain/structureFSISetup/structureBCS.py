# No license
# ----------

""" 
    This is a part of the Partitioned Multi-physical Simulation Framework (parMupSiF)

    FEniCSV2019.1.0+ <-> MUI(mui4py) <-> MUI(C++) <-> OpenFOAMV6+ two way Coupling Code.

    Incompressible Navier-Stokes equations for fluid domain in OpenFOAM
    Structure dynamics equations for structure domain in FEniCS

    structureBCS.py is the boundary condition class of the structure code 
    located in the caseSetup/structureDomain/structureFSISetup sub-folder of the case folder

    Last changed: 25-September-2019
"""

# BAE-FSI
# structureFSISetup/structureBCS.py

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
import numpy as np

#_________________________________________________________________________________________
#
#%% Define boundary conditions
#_________________________________________________________________________________________

class boundaryConditions:
    def DirichletMixedBCs(self, MixedVectorFunctionSpace, boundaries, marks):
        #  !! OUTDATED FUNCTION, NEED UPDATED TO FENICS-X !!
        bc1 = DirichletBC(MixedVectorFunctionSpace.sub(0), ((0.0,0.0,0.0)),boundaries, marks)
        bc2 = DirichletBC(MixedVectorFunctionSpace.sub(1), ((0.0,0.0,0.0)),boundaries, marks)
        return bc1, bc2
    def DirichletBCs(self, VectorFunctionSpace, boundary_dofs):
        bc3 = fem.dirichletbc(np.zeros(3),
                              boundary_dofs,
                              V=VectorFunctionSpace)
        return bc3

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#