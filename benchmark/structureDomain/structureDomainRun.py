# No license
# ----------

""" 
    This is a part of the Partitioned Multi-physical Simulation Framework (parMupSiF)

    FEniCSV2019.1.0+ <-> MUI(mui4py) <-> MUI(C++) <-> OpenFOAMV6+ two way Coupling Code.

    Incompressible Navier-Stokes equations for fluid domain in OpenFOAM
    Structure dynamics equations for structure domain in FEniCS

    structureFSIRun.py is the main function of the structure code 
    located in the caseSetup/structureDomain sub-folder of the case folder

    Last changed: 25-September-2019
"""

# BAE-FSI
# structureFSIRun.py

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
#%% Import configure file
#_________________________________________________________________________________________

import configparser

config = configparser.ConfigParser()
config.read('./structureFSISetup/structureInputPara.ini')

#_________________________________________________________________________________________
#
#%% If iMUICoupling, initialise MPI by mpi4py/MUI for parallelised computation
#_________________________________________________________________________________________

if config['MUI'].getboolean('iMUICoupling'):
    import sys
    from mpi4py import MPI
    import mui4py
    import petsc4py
    import os

    # App common world claims
#    LOCAL_COMM_WORLD = mui4py.mpi_split_by_app(argc=0,
#                                         argv=[],
#                                         threadType=-1,
#                                         thread_support=0,
#                                         use_mpi_comm_split=True)
    # MUI parameters
    dimensionMUI = 3
    data_types = {"dispX": mui4py.FLOAT64,
                  "dispY": mui4py.FLOAT64,
                  "dispZ": mui4py.FLOAT64,
                  "forceX": mui4py.FLOAT64,
                  "forceY": mui4py.FLOAT64,
                  "forceZ": mui4py.FLOAT64}
    # MUI interface creation
    domain = "structureDomain"
    config3d = mui4py.Config(dimensionMUI, mui4py.FLOAT64)

#    iface = ["threeDInterface0"]
#    ifaces3d = mui4py.create_unifaces(domain, iface, config3d)
#    ifaces3d["threeDInterface0"].set_data_types(data_types)

    URI = "mpi://structureDomain/threeDInterface0" 
    iface3d = mui4py.Uniface(uri=URI, config=config3d) 
    iface3d.set_data_types(data_types)

    # App common world claims 
    LOCAL_COMM_WORLD = mui4py.mpi_split_by_app()

    # Necessary to avoid hangs at PETSc vector communication
    petsc4py.init(comm=LOCAL_COMM_WORLD)

    # Define local communicator rank
    rank = LOCAL_COMM_WORLD.Get_rank()

    # Define local communicator size
    size = LOCAL_COMM_WORLD.Get_size()

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________

from dolfinx import *
import structureFSISetup
import structureFSISolver

#_________________________________________________________________________________________
#
#%% Create instances for sub-domains and boundary condition
#_________________________________________________________________________________________

# Create sub-domain instances
subDomains = structureFSISetup.structureSubDomain.SubDomains()
# Create boundary condition instances
BCs = structureFSISetup.structureBCS.boundaryConditions()

#_________________________________________________________________________________________
#
#%% Create solver instances
#_________________________________________________________________________________________

solver = structureFSISolver.structureFSISolver.StructureFSISolver(config, subDomains, BCs)

#_________________________________________________________________________________________
#
#%% Solving
#_________________________________________________________________________________________

if config['MUI'].getboolean('iMUICoupling'):
    solver.solve(LOCAL_COMM_WORLD, iface3d)
else:
    solver.solve()

#_________________________________________________________________________________________
#
#%% Ploting
#_________________________________________________________________________________________

import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
xBench = []
yBench = []
xCpp = []
yCpp = []

for i in np.arange(0.1, 100.1, 0.1):
  x.append(i)
for line in open('structureResults/tip-displacementY_0.txt', 'r'):
    lines = [i for i in line.split()]
    y.append(float(lines[0]))

for line in open('dataInput/Slone_et_al.txt', 'r'):
    lines = [i for i in line.split()]
    xBench.append(float(lines[0]))
    yBench.append(float(lines[1]))

plt.title("Y-Disp Compare")
plt.xlabel('Time [s]')
plt.ylabel('Y-Disp [m]')
plt.plot(xBench, yBench, label = 'Slone et al. 2003', marker= 'o', linestyle='None', c = 'b')
plt.plot(x, y, label = 'Present FEniCS Output', linestyle='-', c = 'g')
plt.xticks(np.arange(0, 101, step=20))
plt.yticks(np.arange(-0.15, 0.16, step=0.05))
plt.legend(loc='upper right')
plt.savefig('../result_compare.png')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#