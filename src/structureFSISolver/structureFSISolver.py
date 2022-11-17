"""
    Parallel Partitioned Multi-physical Simulation Framework (ParaSiF)

    Copyright (C) 2021 Engineering and Environment Group, Scientific
    Computing Department, Science and Technology Facilities Council,
    UK Research and Innovation. All rights reserved.

    This code is licensed under the GNU General Public License version 3

    ** GNU General Public License, version 3 **

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    *********************************************************************

    @file structureFSISolver.py

    @author W. Liu

    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework provides FEniCS v2019.1.0 <-> MUI v1.2 <-> OpenFOAM v6
    two-way coupling.

    Incompressible Navier-Stokes equations for fluid domain in OpenFOAM
    Structure dynamics equations for structure domain in FEniCS.

    The core solver class of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________

from dolfin import *
from mpi4py import MPI
import configparser
import math
import numpy as np
import os
import datetime
import socket
import sys
import structureFSISolver

import structureFSISolver.functions.cfgPrsFn
import structureFSISolver.functions.lameParm
import structureFSISolver.functions.meshBoundarySubdomian
import structureFSISolver.functions.meshMotion
import structureFSISolver.functions.compilerOpt
import structureFSISolver.functions.utility
import structureFSISolver.functions.checkpointLogCtrl
import structureFSISolver.functions.couplingMUIFn
import structureFSISolver.functions.DOFCoordMapping
import structureFSISolver.functions.facetAreas
import structureFSISolver.functions.facetAreas
import structureFSISolver.functions.timeMarching
import structureFSISolver.solvers.linearElasticSolver
import structureFSISolver.solvers.hyperElasticSolver

#_________________________________________________________________________________________
#
#%% Main Structure Solver Class
#_________________________________________________________________________________________

class StructureFSISolver(structureFSISolver.functions.cfgPrsFn.readData,
                         structureFSISolver.functions.meshBoundarySubdomian.meshBoundarySubdomian,
                         structureFSISolver.functions.timeMarching.timeMarching,
                         structureFSISolver.functions.lameParm.lameParm,
                         structureFSISolver.functions.meshMotion.meshMotion,
                         structureFSISolver.functions.compilerOpt.compilerOpt,
                         structureFSISolver.functions.utility.utility,
                         structureFSISolver.functions.checkpointLogCtrl.checkpointLogCtrl,
                         structureFSISolver.functions.couplingMUIFn.couplingMUIFn,
                         structureFSISolver.functions.DOFCoordMapping.DOFCoordMapping,
                         structureFSISolver.functions.facetAreas.facetAreas,
                         structureFSISolver.solvers.linearElasticSolver.linearElastic,
                         structureFSISolver.solvers.hyperElasticSolver.hyperElastic):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Solver initialize
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self,
                 Configure,
                 FixedSubdomain,
                 FlexSubdomain,
                 SymmetrySubdomain,
                 DirichletBoundaryConditions):

        #===========================================
        #%% Obtain files and instances
        #===========================================

        # Obtain configure file
        self.cfg = Configure
        # Obtain fixed sub-domain instances
        self.fixedSDomain = FixedSubdomain
        # Obtain flex sub-domain instances
        self.flexSDomain = FlexSubdomain
        # Obtain symmetry sub-domain instances
        self.symmetrySDomain = SymmetrySubdomain
        # Obtain dirichlet boundary condition instances
        self.dirichletBCs = DirichletBoundaryConditions

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Main solver function
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def solve(self):
        #===========================================
        #%% Setup the wall clock
        #===========================================

        # create an instance of the TicToc wall clock class
        wallClock = structureFSISolver.tictoc.TicToc()
        # Starts the wall clock
        wallClock.tic()

        #===========================================
        #%% Initialise MPI by mpi4py/MUI for
        #%%   parallelised computation
        #===========================================

        self.MUI_Init()

        #===========================================
        #%% Set target folder
        #===========================================

        # Folder directory
        if self.iAbspath():
            self.outputFolderPath = os.path.abspath(self.outputFolderName())
            self.inputFolderPath = os.path.abspath(self.inputFolderName())
        else:
            self.outputFolderPath = self.outputFolderName()
            self.inputFolderPath = self.inputFolderName()

        #===========================================
        #%% Print log information
        #===========================================

        self.Pre_Solving_Log()

        #===========================================
        #%% Set form compiler options
        #===========================================

        self.Set_Compiler_Options()

        #===========================================
        #%% Time marching parameters define
        #===========================================

        self.Time_Marching_Parameters()
        self.Time_Marching_Log()

        #===========================================
        #%% Call solvers
        #===========================================

        if self.solving_method() == 'STVK':
            self.hyperElasticSolve()
        elif self.solving_method() == 'MCK':
            self.linearElasticSolve()
        else:
            sys.exit("{FENICS} Error, solving method not recognised")
        #===========================================
        #%% Calculate wall time
        #===========================================

        # Finish the wall clock
        simtime = wallClock.toc()
        self.Post_Solving_Log(simtime)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#