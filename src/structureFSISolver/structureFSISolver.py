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
import datetime
import math
import numpy as np
import os
import petsc4py
import socket
import sys
import structureFSISolver
#import cfsil4py
import mui4py

import structureFSISolver.cfgPrsFn
import structureFSISolver.lameParm
import structureFSISolver.solvers.linearElasticSolver

#_________________________________________________________________________________________
#
#%% Main Structure Solver Class
#_________________________________________________________________________________________
    
class StructureFSISolver(structureFSISolver.cfgPrsFn.readData,
                         structureFSISolver.lameParm.lameParm,
                         structureFSISolver.solvers.linearElasticSolver.linearElastic):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Solver initialize
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(   self, 
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

        #===========================================
        #%% Debug mode on/off switch 
        #===========================================

        # F-Switch off the debug level codes (if any); T-Switch on the debug level codes (if any).
        self.iDebug = self.cfg['LOGGING'].getboolean('iDebug')
        # F-Switch off the Quiet; T-Switch on the Quiet.
        self.iQuiet = self.cfg['LOGGING'].getboolean('iQuiet')

        #===========================================
        #%% MUI switches & parameters 
        #===========================================
        # F-Switch off the MUI coupling function; T-Switch on the MUI coupling function.
        self.iMUICoupling = self.cfg['MUI'].getboolean('iMUICoupling')
        # F-Switch off the MUI Multidomain function; T-Switch on the MUI Multidomain function.
        self.iMultidomain = self.cfg['MUI'].getboolean('iMultidomain')
        # F-Switch off the MUI Fetch Moment; T-Switch on the MUI Fetch Moment.
        self.iMUIFetchMoment = self.cfg['MUI'].getboolean('iMUIFetchMoment')
        # F-Switch off the MUI Fetch; T-Switch on the MUI Fetch.
        self.iMUIFetchValue = self.cfg['MUI'].getboolean('iMUIFetchValue')
        # F-Use normal fetch function; T-Use fetch_many function.
        self.iMUIFetchMany = self.cfg['MUI'].getboolean('iMUIFetchMany')
        # F-Use Fetch Traction function; T-Use Fetch Force function.
        self.iMUIFetchForce = self.cfg['MUI'].getboolean('iMUIFetchForce')
        # F-Use nearest neighbout spatial sampler; T-Use RBF spatial sampler.
        self.iUseRBF = self.cfg['MUI'].getboolean('iUseRBF')
        # F-The mesh is not evenly spaced; T-The mesh is evenly spaced.
        self.iLoadAreaList = self.cfg['MUI'].getboolean('iLoadAreaList')
        # Factor for area list calculation (float)
        self.areaListFactor = float(self.cfg['MUI']['areaListFactor'])
        # F-Use normal push function; T-Use push_many function.
        self.iMUIPushMany = self.cfg['MUI'].getboolean('iMUIPushMany')
        # F-Not push X; T-Push X.
        self.iPushX = self.cfg['MUI'].getboolean('iPushX')
        # F-Not push Y; T-Push Y.
        self.iPushY = self.cfg['MUI'].getboolean('iPushY')
        # F-Not push Z; T-Push Z.
        self.iPushZ = self.cfg['MUI'].getboolean('iPushZ')
        # Spatial sampler search radius (float)
        self.rMUIFetcher = float(self.cfg['MUI']['rMUIFetcher'])
        # F-Use RBF spatial sampler consistent mode; T-Use RBF spatial sampler conservative mode.
        self.iConservative = self.cfg['MUI'].getboolean('iConservative')
        # RBF spatial sampler cutoff value (float)
        self.cutoffRBF = float(self.cfg['MUI']['cutoffRBF'])
        # F-The RBF matrix will write to file; T-The RBF matrix will read from file.
        self.iReadMatrix = self.cfg['MUI'].getboolean('iReadMatrix')
        # RBF fetch area extend values (float)
        self.fetchExtendRBF = float(self.cfg['MUI']['fetchExtendRBF'])
        # F-Switch off the RBF spatial sampler polynomial terms; T-Switch on the RBF spatial sampler polynomial terms.
        self.iPolynomial = self.cfg['MUI'].getboolean('iPolynomial')
        # Select of basis functions of the RBF spatial sampler (integer)
        self.basisFunc = int(self.cfg['MUI']['basisFunc'])
        # F-Switch off the RBF spatial sampler smooth function; T-Switch on the RBF spatial sampler smooth function.
        self.iSmoothFunc = self.cfg['MUI'].getboolean('iSmoothFunc')
        # Numbers of time steps to forget for MUI push (integer)
        self.forgetTStepsMUI = int(self.cfg['MUI']['forgetTStepsMUI'])
        # ipushLimitMUI
        self.ipushLimitMUI = self.cfg['MUI'].getboolean('ipushLimitMUI')
        # F-Serial FSI coupling mode; T-Parallel FSI coupling mode.
        self.iparallelFSICoupling = self.cfg['MUI'].getboolean('iparallelFSICoupling')
        # Initial under relaxation factor for IQNILS (float)
        self.initUndRelxCpl = float(self.cfg['MUI']['initUndRelxCpl'])
        # Maximum under relaxation factor for IQNILS (float)
        self.undRelxCplMax = float(self.cfg['MUI']['undRelxCplMax'])
        # Iterations for Aitken cycles for IQNILS (integer)
        self.aitkenIterationN = int(self.cfg['MUI']['aitkenIterationN'])
        # F-Using local Alpha for IQNILS; T-Using global Alpha for IQNILS.
        self.globalAlphaInput = self.cfg['MUI'].getboolean('globalAlphaInput')

        #===========================================
        #%% Global solver define
        #===========================================

        # define the solving Method; "STVK" "MCK"
        self.solving_method = self.cfg['SOLVER']['solving_method']
        # define the linear solver; "LU" "LinearVariational"
        self.linear_solver = self.cfg['SOLVER']['linear_solver']
        # define the non-linear solver; "snes" "newton"
        self.nonlinear_solver = self.cfg['SOLVER']['nonlinear_solver']
        # define the linear solver for the problem
        self.prbsolver = self.cfg['SOLVER']['prbsolver']
        # define the solver for project between domains
        self.prjsolver = self.cfg['SOLVER']['prjsolver']
        # define the pre-conditioner for the problem
        self.prbpreconditioner = self.cfg['SOLVER']['prbpreconditioner']
        # define the line search for the snes solver
        self.lineSearch = self.cfg['SOLVER']['lineSearch']
        # define the relative tolerance for the problem
        self.prbRelative_tolerance = float(self.cfg['SOLVER']['prbRelative_tolerance'])
        # define the absolute tolerance for the problem
        self.prbAbsolute_tolerance = float(self.cfg['SOLVER']['prbAbsolute_tolerance'])
        # define the maximum iterations for the problem
        self.prbMaximum_iterations = int(self.cfg['SOLVER']['prbMaximum_iterations'])
        # define the relaxation parameter for the problem
        self.prbRelaxation_parameter = float(self.cfg['SOLVER']['prbRelaxation_parameter'])
        # define the representation of the compiler
        self.compRepresentation = self.cfg['SOLVER']['compRepresentation']
        # switch on the C++ code optimization
        self.cppOptimize = self.cfg['SOLVER'].getboolean('cppOptimize')
        # switch on optimization of the compiler
        self.optimize = self.cfg['SOLVER'].getboolean('optimize')
        # switch on extrapolation WARRING: Please set it 'FALSE' for Parallel Interpolation
        self.allow_extrapolation = self.cfg['SOLVER'].getboolean('allow_extrapolation')
        # Ghost cell mode: "shared_facet"; "shared_vertex"; "none"
        self.ghost_mode = self.cfg['SOLVER']['ghost_mode']
        # switch on error of non convergence
        self.error_on_nonconvergence = self.cfg['SOLVER'].getboolean('error_on_nonconvergence')
        # define the maximum iterations for the krylov solver
        self.krylov_maximum_iterations = int(self.cfg['SOLVER']['krylov_maximum_iterations'])
        # define the relative tolerance for the krylov solver
        self.krylov_prbRelative_tolerance = float(self.cfg['SOLVER']['krylov_prbRelative_tolerance'])
        # define the absolute tolerance for the krylov solver
        self.krylov_prbAbsolute_tolerance = float(self.cfg['SOLVER']['krylov_prbAbsolute_tolerance'])
        # switch on monitor convergence for the krylov solver
        self.monitor_convergence = self.cfg['SOLVER'].getboolean('monitor_convergence')
        # switch on nonzero initial guess for the krylov solver
        self.nonzero_initial_guess = self.cfg['SOLVER'].getboolean('nonzero_initial_guess')
        # switch on report for the krylov solver
        self.show_report = self.cfg['SOLVER'].getboolean('show_report')

        #===========================================
        #%% Global degree orders
        #===========================================

        # Function space degree order
        self.deg_fun_spc = int(self.cfg['ORDER']['deg_fun_spc'])
        # Expression degree (if any)
        self.deg_exp = int(self.cfg['ORDER']['deg_exp'])

        #===========================================
        #%% Target folder input
        #===========================================

        # F-Input/output folder directories are relative paths; T-Input/output folder directories are absolute paths.
        self.iAbspath = self.cfg['FOLDER'].getboolean('iAbspath')
        self.outputFolderName = self.cfg['FOLDER']['outputFolderName']
        self.inputFolderName = self.cfg['FOLDER']['inputFolderName']

        #===========================================
        #%% Solid mechanical parameters input
        #===========================================

        # Density of solid [kg/m^3]
        self.rho_s = float(self.cfg['MECHANICAL']['rho_s'])
        # Poisson ratio [-]
        self.nu_s = float(self.cfg['MECHANICAL']['nu_s'])

        #===========================================
        #%% Solid body external forces input
        #===========================================

        # Body external forces in x-axis direction [N/m^3]
        self.bForExtX = float(self.cfg['EXTFORCE']['bForExtX'])
        # Body external forces in y-axis direction [N/m^3]
        self.bForExtY = float(self.cfg['EXTFORCE']['bForExtY'])
        # Body external forces in z-axis direction [N/m^3]
        self.bForExtZ = float(self.cfg['EXTFORCE']['bForExtZ'])
        # Surface external forces in x-axis direction [N/m^2]
        self.sForExtX = float(self.cfg['EXTFORCE']['sForExtX'])
        # Surface external forces in y-axis direction [N/m^2]
        self.sForExtY = float(self.cfg['EXTFORCE']['sForExtY'])
        # Surface external forces in z-axis direction [N/m^2]
        self.sForExtZ = float(self.cfg['EXTFORCE']['sForExtZ'])
        # Surface external forces end time [s]
        self.sForExtEndTime = float(self.cfg['EXTFORCE']['sForExtEndTime'])

        #===========================================
        #%% Time marching parameter input
        #===========================================

        # End time [s]
        self.T = float(self.cfg['TIME']['T'])
        # Time step size [s]
        self.dt = float(self.cfg['TIME']['dt'])
        # Numbers of sub-iterations (integer) [-]
        self.num_sub_iteration = int(self.cfg['TIME']['num_sub_iteration'])
        # F-Run from initial time step; T-Continue run based on previous results.
        self.iContinueRun = self.cfg['TIME'].getboolean('iContinueRun')
        # F-Run from initial time; T-Run from a different time.
        self.iResetStartTime = self.cfg['TIME'].getboolean('iResetStartTime')
        # New start time (when iResetStartTime = True) [s]
        self.newStartTime = float(self.cfg['TIME']['newStartTime'])
        # F-sub-iteration remains the same; T-change the sub-iteration number.
        self.iChangeSubIter = self.cfg['TIME'].getboolean('iChangeSubIter')
        # Time to change the sub-iteration [s]
        self.TChangeSubIter = float(self.cfg['TIME']['TChangeSubIter'])
        # New numbers of sub-iterations (integer) [-]
        self.num_sub_iteration_new = int(self.cfg['TIME']['num_sub_iteration_new'])
        
        #===========================================
        #%% Time marching accurate control
        #===========================================

        # One-step theta value, valid only on STVK solver
        self.thetaOS = float(self.cfg['TIMEMARCHCOEF']['thetaOS'])
        # Rayleigh damping coefficients, valid only on MCK solver
        self.alpha_rdc = float(self.cfg['TIMEMARCHCOEF']['alpha_rdc'])
        self.beta_rdc = float(self.cfg['TIMEMARCHCOEF']['beta_rdc'])
        # Generalized-alpha method parameters, valid only on MCK solver
        # alpha_m_gam <= alpha_f_gam <= 0.5 for a better performance
        # Suggested values for alpha_m_gam: 0.0 or 0.4
        # Suggested values for alpha_f_gam: 0.0 or 0.2
        self.alpha_m_gam = float(self.cfg['TIMEMARCHCOEF']['alpha_m_gam'])
        self.alpha_f_gam = float(self.cfg['TIMEMARCHCOEF']['alpha_f_gam'])

        #===========================================
        #%% Post-processing parameter input
        #===========================================

        # Output file intervals (integer) [-]
        self.output_interval = int(self.cfg['POSTPROCESS']['output_interval'])
        # X-axis coordinate of the monitoring point [m]
        self.pointMoniX = float(self.cfg['POSTPROCESS']['pointMoniX'])
        # Y-axis coordinate of the monitoring point [m]
        self.pointMoniY = float(self.cfg['POSTPROCESS']['pointMoniY'])
        # Z-axis coordinate of the monitoring point [m]
        self.pointMoniZ = float(self.cfg['POSTPROCESS']['pointMoniZ'])
        # X-axis coordinate of the monitoring point [m]
        self.pointMoniXb = float(self.cfg['POSTPROCESS']['pointMoniXb'])
        # Y-axis coordinate of the monitoring point [m]
        self.pointMoniYb = float(self.cfg['POSTPROCESS']['pointMoniYb'])
        # Z-axis coordinate of the monitoring point [m]
        self.pointMoniZb = float(self.cfg['POSTPROCESS']['pointMoniZb'])

        #===========================================
        #%% Solid Model dimension input
        #===========================================

        # x coordinate of the original point of the beam [m]
        self.OBeamX = float(self.cfg['GEOMETRY']['OBeamX'])
        # y coordinate of the original point of the beam [m]
        self.OBeamY = float(self.cfg['GEOMETRY']['OBeamY'])
        # z coordinate of the original point of the beam [m]
        self.OBeamZ = float(self.cfg['GEOMETRY']['OBeamZ'])
        # length of the beam [m]
        self.XBeam = float(self.cfg['GEOMETRY']['XBeam'])
        # width of the beam [m]
        self.YBeam = float(self.cfg['GEOMETRY']['YBeam'])
        # thick of the beam [m]
        self.ZBeam = float(self.cfg['GEOMETRY']['ZBeam'])

        #===========================================
        #%% Solid calculation selection
        #===========================================

        # F-Generate mesh; T-Load mesh from file.
        self.iMeshLoad = self.cfg['CALMODE'].getboolean('iMeshLoad')
        # F-Linear Hooke's law; T-Non-linear St. Vernant-Kirchhoff material model.
        self.iNonLinearMethod = self.cfg['CALMODE'].getboolean('iNonLinearMethod')
        # F-The HDF5 File Export function closed; T-The HDF5 File Export function opened.
        self.iHDF5FileExport = self.cfg['CALMODE'].getboolean('iHDF5FileExport')
        # F-Load mesh from HDF5 file; T-Load mesh from XML file (when iMeshLoad = T).
        self.iLoadXML = self.cfg['CALMODE'].getboolean('iLoadXML')
        # F-Do not show the generated mesh; T-Show the generated mesh interactively.
        self.iInteractiveMeshShow = self.cfg['CALMODE'].getboolean('iInteractiveMeshShow')
        # F-The HDF5 Mesh Export function closed; T-The HDF5 Mesh Export function opened (when iHDF5FileExport = T).
        self.iHDF5MeshExport = self.cfg['CALMODE'].getboolean('iHDF5MeshExport')
        # F-The HDF5 Subdomains Export function closed; T-The HDF5 Subdomains Export function opened (when iHDF5FileExport = T).
        self.iHDF5SubdomainsExport = self.cfg['CALMODE'].getboolean('iHDF5SubdomainsExport')
        # F-The HDF5 Boundaries Export function closed; T-The HDF5 Boundaries Export function opened (when iHDF5FileExport = T).
        self.iHDF5BoundariesExport = self.cfg['CALMODE'].getboolean('iHDF5BoundariesExport')
        # F-The Subdomains Import function closed; T-The Subdomains Import function opened.
        self.iSubdomainsImport = self.cfg['CALMODE'].getboolean('iSubdomainsImport')
        # F-The Boundaries Import function closed; T-The Boundaries Import function opened.
        self.iBoundariesImport = self.cfg['CALMODE'].getboolean('iBoundariesImport')
        # F-The txt export of time list and max displacement closed; T-The txt export of time list and max displacement opened.
        self.iExporttxt = self.cfg['CALMODE'].getboolean('iExporttxt')
        # F-Apply uniform traction force; T-Apply non-uniform traction force.
        self.iNonUniTraction = self.cfg['CALMODE'].getboolean('iNonUniTraction')
        # F-The gravitational force not included; T-The gravitational force included.
        self.iGravForce = self.cfg['CALMODE'].getboolean('iGravForce')

        #===========================================
        #%% Solid Mesh numbers input
        #===========================================

        # cell numbers along the length of the beam, valid when iMeshLoad=False (integer) [-]
        self.XMesh = int(self.cfg['MESH']['XMesh'])
        # cell numbers along the width of the beam, valid when iMeshLoad=False (integer) [-]
        self.YMesh = int(self.cfg['MESH']['YMesh'])
        # cell numbers along the thick of the beam, valid when iMeshLoad=False (integer) [-]
        self.ZMesh = int(self.cfg['MESH']['ZMesh'])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Initialize MPI by mpi4py/MUI for parallelized computation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def MPI_Init (self):

        if self.iMUICoupling:
            # App common world claims 
            self.LOCAL_COMM_WORLD = mui4py.mpi_split_by_app()
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

            iface = ["threeDInterface0"]
            self.ifaces3d = mui4py.create_unifaces(domain, iface, config3d)
            self.ifaces3d["threeDInterface0"].set_data_types(data_types)

            # Necessary to avoid hangs at PETSc vector communication
            petsc4py.init(comm=self.LOCAL_COMM_WORLD)

        else:
            # App common world claims    
            self.LOCAL_COMM_WORLD = MPI.COMM_WORLD

        # Define local communicator rank
        self.rank = self.LOCAL_COMM_WORLD.Get_rank()

        # Define local communicator size
        self.size = self.LOCAL_COMM_WORLD.Get_size()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define MUI samplers and commit ZERO step
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def MUI_Sampler_Define( self,
                            function_space,
                            grid_dimension,
                            dofs_fetch_list,
                            dofs_push_list,
                            xyz_fetch,
                            Total_Time_Steps):

        if self.iMUICoupling:
            synchronised=False

            dofs_to_xyz = self.dofs_to_xyz(function_space, grid_dimension)

            send_min_X = sys.float_info.max
            send_min_Y = sys.float_info.max
            send_min_Z = sys.float_info.max

            send_max_X = -sys.float_info.max
            send_max_Y = -sys.float_info.max
            send_max_Z = -sys.float_info.max

            for i, p in enumerate(dofs_push_list):
                if (dofs_to_xyz[p][0] < send_min_X):
                    send_min_X = dofs_to_xyz[p][0]

                if (dofs_to_xyz[p][1] < send_min_Y):
                    send_min_Y = dofs_to_xyz[p][1]

                if (dofs_to_xyz[p][2] < send_min_Z):
                    send_min_Z = dofs_to_xyz[p][2]

                if (dofs_to_xyz[p][0] > send_max_X):
                    send_max_X = dofs_to_xyz[p][0]

                if (dofs_to_xyz[p][1] > send_max_Y):
                    send_max_Y = dofs_to_xyz[p][1]

                if (dofs_to_xyz[p][2] > send_max_Z):
                    send_max_Z = dofs_to_xyz[p][2]

            if (send_max_X < send_min_X):
                print("{** FENICS ERROR **} send_max_X: ", send_max_X, " smaller than send_min_X: ", send_min_X, " at rank: ", self.rank)

            if (send_max_Y < send_min_Y):
                print("{** FENICS ERROR **} send_max_Y: ", send_max_Y, " smaller than send_min_Y: ", send_min_Y, " at rank: ", self.rank)

            if (send_max_Z < send_min_Z):
                print("{** FENICS ERROR **} send_max_Z: ", send_max_Z, " smaller than send_min_Z: ", send_min_Z, " at rank: ", self.rank)

            if (len(dofs_push_list)!=0):
                # Set up sending span
                span_push = mui4py.geometry.Box([send_min_X, send_min_Y, send_min_Z],
                                                [send_max_X, send_max_Y, send_max_Z])

                # Announce the MUI send span
                self.ifaces3d["threeDInterface0"].announce_send_span(0, Total_Time_Steps*self.num_sub_iteration, span_push, synchronised)

                print("{FENICS} at rank: ", self.rank, " send_max_X: ", send_max_X, " send_min_X: ", send_min_X)
                print("{FENICS} at rank: ", self.rank, " send_max_Y: ", send_max_Y, " send_min_Y: ", send_min_Y)
                print("{FENICS} at rank: ", self.rank, " send_max_Z: ", send_max_Z, " send_min_Z: ", send_min_Z)

            else:
                # Announce the MUI send span
                #self.ifaces3d["threeDInterface0"].announce_send_disable()
                pass

            recv_min_X = sys.float_info.max
            recv_min_Y = sys.float_info.max
            recv_min_Z = sys.float_info.max

            recv_max_X = -sys.float_info.max
            recv_max_Y = -sys.float_info.max
            recv_max_Z = -sys.float_info.max

            # Declare list to store mui::point3d
            point3dList = []
            point3dGlobalID = []

            for i, p in enumerate(dofs_fetch_list):
                if (dofs_to_xyz[p][0] < recv_min_X):
                    recv_min_X = dofs_to_xyz[p][0]

                if (dofs_to_xyz[p][1] < recv_min_Y):
                    recv_min_Y = dofs_to_xyz[p][1]

                if (dofs_to_xyz[p][2] < recv_min_Z):
                    recv_min_Z = dofs_to_xyz[p][2]

                if (dofs_to_xyz[p][0] > recv_max_X):
                    recv_max_X = dofs_to_xyz[p][0]

                if (dofs_to_xyz[p][1] > recv_max_Y):
                    recv_max_Y = dofs_to_xyz[p][1]

                if (dofs_to_xyz[p][2] > recv_max_Z):
                    recv_max_Z = dofs_to_xyz[p][2]

                point_fetch = self.ifaces3d["threeDInterface0"].Point([dofs_to_xyz[p][0],
                                                                        dofs_to_xyz[p][1],
                                                                        dofs_to_xyz[p][2]])

                point_ID = -999
                for ii, pp in enumerate(xyz_fetch):
                    if (pp[0] == dofs_to_xyz[p][0]):
                        if (pp[1] == dofs_to_xyz[p][1]):
                            if (pp[2] == dofs_to_xyz[p][2]):
                                point_ID = ii
                                break

                if (point_ID<0):
                    print("{** FENICS ERROR **} cannot find point: ", dofs_to_xyz[p][0],
                                                                        dofs_to_xyz[p][1],
                                                                        dofs_to_xyz[p][2],
                                                                        " in Global xyz fetch list")
                point3dList.append(point_fetch)
                point3dGlobalID.append(point_ID)

            if (recv_max_X < recv_min_X):
                print("{** FENICS ERROR **} recv_max_X: ", recv_max_X, " smaller than recv_min_X: ", recv_min_X, " at rank: ", self.rank)

            if (recv_max_Y < recv_min_Y):
                print("{** FENICS ERROR **} recv_max_Y: ", recv_max_Y, " smaller than recv_min_Y: ", recv_min_Y, " at rank: ", self.rank)

            if (recv_max_Z < recv_min_Z):
                print("{** FENICS ERROR **} recv_max_Z: ", recv_max_Z, " smaller than recv_min_Z: ", recv_min_Z, " at rank: ", self.rank)

            if (len(dofs_fetch_list)!=0):
                # Set up receiving span
                span_fetch = mui4py.geometry.Box([recv_min_X, recv_min_Y, recv_min_Z],
                                                 [recv_max_X, recv_max_Y, recv_max_Z])

                # Announce the MUI receive span
                self.ifaces3d["threeDInterface0"].announce_recv_span(0, Total_Time_Steps*self.num_sub_iteration*10, span_fetch, synchronised)

                print("{FENICS} at rank: ", self.rank, " recv_max_X: ", recv_max_X, " recv_min_X: ", recv_min_X)
                print("{FENICS} at rank: ", self.rank, " recv_max_Y: ", recv_max_Y, " recv_min_Y: ", recv_min_Y)
                print("{FENICS} at rank: ", self.rank, " recv_max_Z: ", recv_max_Z, " recv_min_Z: ", recv_min_Z)

            else:
                # Announce the MUI receive span
                #self.ifaces3d["threeDInterface0"].announce_recv_disable()
                pass

            # Spatial/temporal samplers
            if self.rank == 0: print ("{FENICS} Defining MUI samplers ...   ", end="", flush=True)

            fileAddress=self.outputFolderName + '/RBFMatrix/' + str(self.rank)
            os.makedirs(fileAddress)

            if (self.iReadMatrix):
                print ("{FENICS} Reading RBF matrix from ", self.rank)
                sourcefileAddress=self.inputFolderName + '/RBFMatrix'

                # search line number of the pointID
                numberOfFolders = 0
                with open(sourcefileAddress +'/partitionSize.dat', 'r') as f_psr:
                    print ("{FENICS} open partitionSize from ", self.rank)
                    for line in f_psr:
                        numberOfFolders = int(line)
                f_psr.close()
                print ("{FENICS} Number of RBF subfolders: ", numberOfFolders, " from ", self.rank)

                numberOfCols=-99
                for i, point_IDs in enumerate(point3dGlobalID):
                    # search line number of the pointID
                    iFolder=0
                    while iFolder < numberOfFolders:
                        line_number = -1
                        result_line_number = -99
                        result_folder_number = -99
                        with open(sourcefileAddress+'/'+str(iFolder)+'/pointID.dat', 'r') as f_pid:
                            for line in f_pid:
                                line_number += 1
                                if str(point_IDs) in line:
                                    result_line_number = line_number
                                    result_folder_number = iFolder
                                    break
                        f_pid.close()
                        iFolder += 1
                        if (result_folder_number >= 0):
                            break

                    if (result_line_number < 0):
                        print ("{** FENICS ERROR **} Cannot find Point ID: ", point_ID, " in file")
                    # Get the line in H matrix and copy to local file
                    with open(sourcefileAddress+'/'+str(result_folder_number)+'/Hmatrix.dat', 'r') as f_h:
                        for i, line in enumerate(f_h):
                            if i == (result_line_number+6):
                                with open(fileAddress+'/Hmatrix.dat', 'a') as f_h_result:
                                    if line[-1] == '\n':
                                        f_h_result.write(line)
                                    else:
                                        f_h_result.write(line+'\n')
                                    if (numberOfCols<0):
                                        numberOfCols=len(line.split(","))
                                f_h_result.close()
                            elif i > (result_line_number+6):
                                break
                    f_h.close()

                with open(fileAddress+'/matrixSize.dat', 'w') as f_size:
                    f_size.write(str(numberOfCols)+","+str(len(point3dGlobalID))+",0,0,"+str(len(point3dGlobalID))+","+str(numberOfCols))
                f_size.close()

            else:
                if self.rank == 0:
                    with open(self.outputFolderName + '/RBFMatrix'+'/partitionSize.dat', 'w') as f_ps:
                        f_ps.write("%i\n" % self.size)

            # Best practice suggestion: for a better performance on the RBF method, always switch on the smoothFunc when structure Dofs are more than
            #                           fluid points; Tune the rMUIFetcher to receive a reasonable totForce_Fetch value; Tune the areaListFactor to
            #                           ensure totForce_Fetch and Total_Force_on_structure are the same.
            self.t_sampler = mui4py.ChronoSamplerExact()

            self.s_sampler = mui4py.SamplerRbf(self.rMUIFetcher,
                                                    point3dList,
                                                    self.basisFunc,
                                                    self.iConservative,
                                                    self.iPolynomial,
                                                    self.iSmoothFunc,
                                                    self.iReadMatrix,
                                                    fileAddress,
                                                    self.cutoffRBF)

            with open(fileAddress+'/pointID.dat', 'w') as f_pid:
                for pid in point3dGlobalID:
                    f_pid.write("%i\n" % pid)

            # Commit ZERO step
            self.ifaces3d["threeDInterface0"].commit(0)
            if self.rank == 0: print ("{FENICS} Commit ZERO step")
        else:
            pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Solid Mesh input/generation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Mesh_Generation(self):
        if self.iContinueRun:
            # Restart simulation
            if self.iMeshLoad:
                # Load mesh from file
                if self.iLoadXML:
                    # Load mesh from XML file
                    if self.rank == 0: print ("{FENICS} Loading XML mesh ...   ")
                    mesh = Mesh(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/Structure_FEniCS.xml")
                    if self.rank == 0: print ("{FENICS} Done with loading XML mesh")
                else:
                    # Load mesh from HDF5 file
                    if self.rank == 0: print ("{FENICS} Loading HDF5 mesh ...   ")
                    # commit off due to the hanging in FEniCS-v2019.1.0
                    #mesh = Mesh()
                    # generate a dummy mesh and overwrite it by the HDF5 read-in data.
                    mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(0, 0, 0),
                       Point(1, 1, 1),
                       10, 10, 10)
                    hdfInTemp = HDF5File(mesh.mpi_comm(), self.inputFolderPath + "/checkpointData.h5", "r")
                    hdfInTemp.read(mesh, "/mesh", False)
                    hdfInTemp.close()
                    del hdfInTemp
                    if self.rank == 0: print ("{FENICS} Done with loading HDF5 mesh")
            else:
                # Generate mesh
                if self.rank == 0: print ("{FENICS} Generating mesh ...   ")
                mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(self.OBeamX, self.OBeamY, self.OBeamZ),
                       Point((self.OBeamX+self.XBeam), (self.OBeamY+self.YBeam), (self.OBeamZ+self.ZBeam)),
                       self.XMesh, self.YMesh, self.ZMesh)
                if self.rank == 0: print ("{FENICS} Done with generating mesh")
        else:
            # Simulation from zero
            if self.iMeshLoad:
                # Load mesh from file
                if self.iLoadXML:
                    if self.rank == 0: print ("{FENICS} Loading XML mesh ...   ")
                    mesh = Mesh(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/Structure_FEniCS.xml")
                    if self.rank == 0: print ("{FENICS} Done with loading XML mesh")
                else:
                    if self.rank == 0: print ("{FENICS} Loading HDF5 mesh ...   ")
                    # commit off due to the hanging in FEniCS-v2019.1.0
                    #mesh = Mesh()
                    # generate a dummy mesh and overwrite it by the HDF5 read-in data.
                    mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(0, 0, 0),
                       Point(1, 1, 1),
                       10, 10, 10)
                    hdfInTemp = HDF5File(mesh.mpi_comm(), self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                    hdfInTemp.read(mesh, "/mesh", False)
                    hdfInTemp.close()
                    del hdfInTemp
                    if self.rank == 0: print ("{FENICS} Done with loading HDF5 mesh")
            else:
                # Generate mesh
                if self.rank == 0: print ("{FENICS} Generating mesh ...   ")
                mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(self.OBeamX, self.OBeamY, self.OBeamZ),
                       Point((self.OBeamX+self.XBeam), (self.OBeamY+self.YBeam), (self.OBeamZ+self.ZBeam)),
                       self.XMesh, self.YMesh, self.ZMesh)
                if self.rank == 0: print ("{FENICS} Done with generating mesh")

        if self.iHDF5FileExport and self.iHDF5MeshExport:
            if self.rank == 0: print ("{FENICS} Exporting HDF5 mesh ...   ", end="", flush=True)
            hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "w")
            hdfOutTemp.write(mesh, "/mesh")
            hdfOutTemp.close()
            del hdfOutTemp               
            if self.rank == 0: print ("Done")

        if self.iInteractiveMeshShow:
            if self.rank == 0: print ("{FENICS} Interactive Mesh Show ...", end="", flush=True)
            import matplotlib.pyplot as plt
            plt.figure()
            p = plot(mesh, title = "Mesh plot")        
            plt.show()
            if self.rank == 0: print ("Done")

        return mesh

    def Mesh_Original_Generation(self, mesh):
        if self.iContinueRun:
            # Restart simulation
            if self.iMeshLoad:
                # Load mesh from file
                hdfInTemp = HDF5File(mesh.mpi_comm(), self.inputFolderPath + "/checkpointData.h5", "r")
                mesh_original = Mesh(mesh)                    # Store original mesh
                hdfInTemp.read(mesh_original, "/meshOri", False)
                hdfInTemp.close()
                del hdfInTemp
            else:
                # Generate mesh
                mesh_original = Mesh(mesh)                    # Store original mesh
        else:
            mesh_original = Mesh(mesh)                    # Store original mesh

        return mesh_original

    def Get_Grid_Dimension(self, mesh):

        grid_dimension = mesh.geometry().dim()            # Geometry dimensions

        return grid_dimension

    def Get_Face_Narmal(self, mesh):

        face_narmal = FacetNormal(mesh)                   # Face normal vector

        return face_narmal

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define SubDomains and boundaries
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Boundaries_Generation_Fixed_Flex_Sym (self,
                                              mesh,
                                              grid_dimension,
                                              VectorFunctionSpace):

        #===========================================
        #%% Define SubDomains
        #===========================================

        if self.iMeshLoad and self.iSubdomainsImport:
            if self.iLoadXML:
                if self.rank == 0: print ("{FENICS} Loading XML subdomains ...   ", end="", flush=True)
                self.subdomains = MeshFunction("size_t", mesh, self.inputFolderPath + "/Structure_FEniCS_physical_region.xml")
            else:
                if self.rank == 0: print ("{FENICS} Loading HDF5 subdomains ...   ", end="", flush=True)
                self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
                hdfInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                hdfInTemp.read(self.subdomains, "/subdomains")
                hdfInTemp.close()
                del hdfInTemp

            if self.rank == 0: print ("Done")

        else:
            if self.rank == 0: print ("{FENICS} Creating subdomains ...   ", end="", flush=True)

            # Initialize sub-domain instances
            fixed       =  self.fixedSDomain
            flex        =  self.flexSDomain
            symmetry    =  self.symmetrySDomain

            if self.rank == 0: print ("Done")

        if self.iHDF5FileExport and self.iHDF5SubdomainsExport:
            if self.rank == 0: print ("{FENICS} Exporting HDF5 subdomains ...   ", end="", flush=True) 
            self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
            hdfOutTemp.write(self.subdomains, "/subdomains")
            hdfOutTemp.close()
            del hdfOutTemp
            if self.rank == 0: print ("Done")

        #===========================================
        #%% Define and mark mesh boundaries
        #===========================================

        if self.iMeshLoad and self.iBoundariesImport:
            if self.iLoadXML:
                if self.rank == 0: print ("{FENICS} Loading XML boundaries ...   ", end="", flush=True)
                boundaries = MeshFunction("size_t", mesh, self.inputFolderPath + "/Structure_FEniCS_facet_region.xml")
            else:
                if self.rank == 0: print ("{FENICS} Loading HDF5 boundaries ...   ", end="", flush=True)
                boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
                hdfInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                hdfInTemp.read(boundaries, "/boundaries")
                hdfInTemp.close()
                del hdfInTemp
            if self.rank == 0: print ("Done")

        else:
            if self.rank == 0: print ("{FENICS} Creating boundaries ...   ", end="", flush=True)

            boundaries = MeshFunction("size_t",mesh,grid_dimension-1)

            boundaries.set_all(0)
            fixed.mark(boundaries,1)
            flex.mark(boundaries,2)
            symmetry.mark(boundaries,3)
            if self.rank == 0: print ("Done")

        if self.iHDF5FileExport and self.iHDF5BoundariesExport: 
            if self.rank == 0: print ("{FENICS} Exporting HDF5 boundaries ...   ", end="", flush=True)
            hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
            hdfOutTemp.write(boundaries, "/boundaries")
            hdfOutTemp.close()
            del hdfOutTemp
            if self.rank == 0: print ("Done")

        if self.rank == 0: 
            print ("\n")
            print ("{FENICS} Structure Mesh Info: ")
            print ("{FENICS} Dofs: ",VectorFunctionSpace.dim())
            print ("{FENICS} Cells:", mesh.num_cells())
            print ("{FENICS} geometry dimension: ",grid_dimension)
            print ("\n")

        return boundaries

    def Get_ds(self, mesh, boundaries):

        return Measure("ds", domain=mesh, subdomain_data=boundaries)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Time marching parameters define
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Time_Marching_Parameters(self):
        if self.iContinueRun:
            if True:
                hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
                self.Start_Time = hdf5checkpointDataInTemp.attributes("/ud/vector_0")["timestamp"] # Start time [s]
                self.Start_Time += self.dt                                                         # Start time [s]
                hdf5checkpointDataInTemp.close()
                del hdf5checkpointDataInTemp                                                  # Delete HDF5File object, closing file
                self.Time_Steps = math.ceil((self.T - self.Start_Time)/self.dt)                         # Time steps [-]
            else:
                self.Start_Time = self.dt                                                          # Start time [s]
                self.Time_Steps = math.ceil(self.T/self.dt)                                        # Time steps [-]
        else:
            if (self.iResetStartTime):
                self.Start_Time = self.newStartTime+self.dt                                        # Start time [s]
                self.Time_Steps = math.ceil((self.T - self.Start_Time)/self.dt)                         # Time steps [-]
            else:
                self.Start_Time = self.dt                                                          # Start time [s]
                self.Time_Steps = math.ceil(self.T/self.dt)                                        # Time steps [-]
        self.Start_Number_Sub_Iteration = 1                                                    # Initialize sub-iterations counter

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define DOFs extract function 
    #%% and 
    #%% DOFs-Coordinates mapping function
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_subdomain_dofs( self, 
                            MeshFunction, 
                            VectorFunctionSpace, 
                            boundary):
        # Helper function to extract dofs from a subdomain.
        # This is only done once if the mesh do not change.
        # In this StructureFSISolver, we use this function 
        #     to extract the dofs from the original mesh.
        u = Function(VectorFunctionSpace)
        bc = DirichletBC(VectorFunctionSpace, Constant(1.0), MeshFunction, boundary)
        bc.apply(u.vector())
        return np.where(u.vector()==1.0)[0]

    def dofs_to_xyz(self, FunctionSpace, dimension):
        # Convert dofs to coordinates
        return FunctionSpace.tabulate_dof_coordinates().reshape((-1, dimension))

    def dofs_list(self,
                  MeshFunction,
                  FunctionSpace,
                  boundary):

        return list(self.get_subdomain_dofs(MeshFunction, FunctionSpace, boundary))

    def xyz_np(self,
               dofs_list,
               FunctionSpace,
               dimension):
        xyz_np = np.zeros((len(dofs_list), dimension))
        for i, p in enumerate(dofs_list):
            xyz_np[i] = self.dofs_to_xyz(FunctionSpace, dimension)[p]
        return xyz_np

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define facet areas
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def facets_area_list(   self, 
                            MPI_COMM_WORLD, 
                            mesh, 
                            FunctionSpace, 
                            boundary, 
                            dofs_fetch_list, 
                            dimension,
                            temp_vec_function):

        areatotal = 0.0
        areatotal_local = 0.0
        cell2dofs = FunctionSpace.dofmap().cell_dofs
        ones = np.array([1,1,1])

        dpc_help_number = 0
        if (self.deg_fun_spc != 1):
            for i in range(2, self.deg_fun_spc+1):
                dpc_help_number += i
        dofs_Per_Cell=3+dpc_help_number+(self.deg_fun_spc-1)

        for f in facets(mesh):
            if boundary[f.index()] == 2:
                x_array = np.array([])
                y_array = np.array([])
                z_array = np.array([])

                for vertices in f.entities(dimension-3):
                # Currently, this calculation only supports triangular boundary mesh
                    vertex_coor=Vertex(mesh, vertices).point()
                    x_array = np.append(x_array, vertex_coor.x())
                    y_array = np.append(y_array, vertex_coor.y())
                    z_array = np.append(z_array, vertex_coor.z())

                row1=np.array([x_array, y_array, ones])
                row2=np.array([y_array, z_array, ones])
                row3=np.array([z_array, x_array, ones])
                det1=np.linalg.det(row1)
                det2=np.linalg.det(row2)
                det3=np.linalg.det(row3)
                area = 0.5*math.sqrt(det1*det1 + det2*det2 + det3*det3)*self.areaListFactor
                for c in cells(f):
                    c_dofs = cell2dofs(c.index())
                    d_list=[]
                    d_num = 0
                    for i, p in enumerate(c_dofs):
                        if p in dofs_fetch_list:
                            d_list.append(p)
                            d_num+=1
                    if (len(d_list)!=0):
                        for ii, pp in enumerate(d_list):
                            # temp_vec_function[pp] += area/d_num
                            temp_vec_function[pp] += area/dofs_Per_Cell

        for iii, ppp in enumerate(temp_vec_function):
            areatotal += temp_vec_function[iii]

        if (self.rank == 0) and self.iDebug:
            print("Total area of MUI fetched surface= ", areatotal, " m^2")

        return temp_vec_function

    def facets_area_define(self,
                        meshOri,
                        QOri,
                        boundariesOri,
                        dofs_fetch_list,
                        gdimOri):

            self.areaf= Function(QOri)             # Function for facet area

            self.areaf_vec = self.areaf.vector().get_local()

            if (not self.iLoadAreaList):
                if self.rank == 0: print ("{FENICS} facet area calculating")

                self.areaf_vec = self.facets_area_list(  self.LOCAL_COMM_WORLD,
                                                    meshOri,
                                                    QOri,
                                                    boundariesOri,
                                                    dofs_fetch_list,
                                                    gdimOri,
                                                    self.areaf_vec)

                # Apply the facet area vectors
                self.areaf.vector().set_local(self.areaf_vec)
                self.areaf.vector().apply("insert")
                if (self.iHDF5FileExport) and (self.iHDF5MeshExport):
                    hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                else:
                    hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "w")
                hdfOutTemp.write(self.areaf, "/areaf")
                hdfOutTemp.close()

            else:

                hdf5meshAreaDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                hdf5meshAreaDataInTemp.read(self.areaf, "/areaf/vector_0")
                hdf5meshAreaDataInTemp.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define MUI Fetch and Push 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def f_p_steps ( self, 
                    Current_Time_Step, 
                    current_Sub_Iteration):
        # MUI calculate fetch/push iteration steps
        return ((Current_Time_Step - 1) * self.num_sub_iteration + current_Sub_Iteration)

    def forget_p_steps (    self, 
                            Current_Time_Step, 
                            current_Sub_Iteration):
        # MUI calculate forget iteration steps
        return ((Current_Time_Step - self.forgetTStepsMUI - 1) * self.num_sub_iteration + current_Sub_Iteration)

    def MUI_Fetch ( self,
                    dofs_to_xyz, 
                    dofs_fetch_list, 
                    Temporal_sampler,
                    Spatial_sampler,
                    total_Sub_Iteration, 
                    temp_vec_function, 
                    facet_area_vec_function):

        totForceX = 0.0
        totForceY = 0.0
        totForceZ = 0.0
        temp_vec_function_temp = temp_vec_function

        if self.iparallelFSICoupling:
            fetch_iteration = total_Sub_Iteration-1
        else:
            fetch_iteration = total_Sub_Iteration

        if (fetch_iteration >= 0):
            if self.iMUIFetchMany:
                temp_vec_function_temp[0::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceX", 
                                        dofs_to_xyz,
                                        fetch_iteration,
                                        Spatial_sampler,
                                        Temporal_sampler)
                temp_vec_function_temp[1::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceY", 
                                        dofs_to_xyz,
                                        fetch_iteration,
                                        Spatial_sampler,
                                        Temporal_sampler)
                temp_vec_function_temp[2::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceZ", 
                                        dofs_to_xyz,
                                        fetch_iteration,
                                        Spatial_sampler,
                                        Temporal_sampler)

                for i, p in enumerate(dofs_fetch_list):
                    if self.iparallelFSICoupling:
                        temp_vec_function[0::3][p] += (temp_vec_function_temp[0::3][p] - temp_vec_function[0::3][p])*self.initUndRelxCpl
                        temp_vec_function[1::3][p] += (temp_vec_function_temp[1::3][p] - temp_vec_function[1::3][p])*self.initUndRelxCpl
                        temp_vec_function[2::3][p] += (temp_vec_function_temp[2::3][p] - temp_vec_function[2::3][p])*self.initUndRelxCpl
                    else:
                        temp_vec_function[0::3][p] = temp_vec_function_temp[0::3][p]
                        temp_vec_function[1::3][p] = temp_vec_function_temp[1::3][p]
                        temp_vec_function[2::3][p] = temp_vec_function_temp[2::3][p]

                    totForceX += temp_vec_function[0::3][p]
                    totForceY += temp_vec_function[1::3][p]
                    totForceZ += temp_vec_function[2::3][p]

                    if (facet_area_vec_function[p] == 0):
                        temp_vec_function[0::3][p] = 0.
                        temp_vec_function[1::3][p] = 0.
                        temp_vec_function[2::3][p] = 0.
                    else:
                        temp_vec_function[0::3][p] /= facet_area_vec_function[p]
                        temp_vec_function[1::3][p] /= facet_area_vec_function[p]
                        temp_vec_function[2::3][p] /= facet_area_vec_function[p]

            else:
                if (fetch_iteration >= 0):
                    for i, p in enumerate(dofs_fetch_list):
                        temp_vec_function_temp[0::3][p] = self.ifaces3d["threeDInterface0"].fetch("forceX",
                                                    dofs_to_xyz[i], 
                                                    fetch_iteration,
                                                    Spatial_sampler,
                                                    Temporal_sampler)

                        temp_vec_function_temp[1::3][p] = self.ifaces3d["threeDInterface0"].fetch("forceY", 
                                                    dofs_to_xyz[i], 
                                                    fetch_iteration,
                                                    Spatial_sampler,
                                                    Temporal_sampler)

                        temp_vec_function_temp[2::3][p] = self.ifaces3d["threeDInterface0"].fetch("forceZ", 
                                                    dofs_to_xyz[i], 
                                                    fetch_iteration,
                                                    Spatial_sampler,
                                                    Temporal_sampler)

                        if self.iparallelFSICoupling:
                            temp_vec_function[0::3][p] += (temp_vec_function_temp[0::3][p] - temp_vec_function[0::3][p])*self.initUndRelxCpl
                            temp_vec_function[1::3][p] += (temp_vec_function_temp[1::3][p] - temp_vec_function[1::3][p])*self.initUndRelxCpl
                            temp_vec_function[2::3][p] += (temp_vec_function_temp[2::3][p] - temp_vec_function[2::3][p])*self.initUndRelxCpl
                        else:
                            temp_vec_function[0::3][p] = temp_vec_function_temp[0::3][p]
                            temp_vec_function[1::3][p] = temp_vec_function_temp[1::3][p]
                            temp_vec_function[2::3][p] = temp_vec_function_temp[2::3][p]

                        totForceX += temp_vec_function[0::3][p]
                        totForceY += temp_vec_function[1::3][p]
                        totForceZ += temp_vec_function[2::3][p]

                        temp_vec_function[0::3][p] /= facet_area_vec_function[p]
                        temp_vec_function[1::3][p] /= facet_area_vec_function[p]
                        temp_vec_function[2::3][p] /= facet_area_vec_function[p]

                    if self.iDebug:
                        print ("{FENICS**} totForce Apply: ", totForceX, "; ",totForceY, "; ",totForceZ, 
                                "; at iteration: ", fetch_iteration, " at rank: ", self.rank)

        return temp_vec_function

    def MUI_Push(   self,
                    dofs_to_xyz, 
                    dofs_push, 
                    displacement_function, 
                    total_Sub_Iteration):

        d_vec_x = displacement_function.vector().get_local()[0::3]
        d_vec_y = displacement_function.vector().get_local()[1::3]
        d_vec_z = displacement_function.vector().get_local()[2::3]

        if self.iMUIPushMany:
            if self.iPushX:
                self.ifaces3d["threeDInterface0"].push_many("dispX", dofs_to_xyz,
                                                            (d_vec_x[dofs_push]))
            if self.iPushY:
                self.ifaces3d["threeDInterface0"].push_many("dispY", dofs_to_xyz,
                                                        (d_vec_y[dofs_push]))
            if self.iPushZ:
                self.ifaces3d["threeDInterface0"].push_many("dispZ", dofs_to_xyz,
                                                        (d_vec_z[dofs_push]))
            a = self.ifaces3d["threeDInterface0"].commit(total_Sub_Iteration)

        else:
            if self.iPushX:
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].push("dispX", dofs_to_xyz[i],
                                                            (d_vec_x[p]))
            if self.iPushY:
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].push("dispY", dofs_to_xyz[i],
                                                            (d_vec_y[p]))
            if self.iPushZ:
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].push("dispZ", dofs_to_xyz[i],
                                                            (d_vec_z[p]))
            a = self.ifaces3d["threeDInterface0"].commit(total_Sub_Iteration)

        if (self.rank == 0) and self.iDebug:
            print ('{FENICS} MUI commit step: ',total_Sub_Iteration)

        if ((total_Sub_Iteration-self.forgetTStepsMUI) > 0):
            a = self.ifaces3d["threeDInterface0"].forget(total_Sub_Iteration-self.forgetTStepsMUI)
            self.ifaces3d["threeDInterface0"].set_memory(self.forgetTStepsMUI)
            if (self.rank == 0) and self.iDebug:
                print ('{FENICS} MUI forget step: ',(total_Sub_Iteration-self.forgetTStepsMUI))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define directional vectors
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def X_direction_vector(self):
        # Directional vector in x-axis direction
       return Constant((1.0, 0.0, 0.0))
    def Y_direction_vector(self):
        # Directional vector in y-axis direction
       return Constant((0.0, 1.0, 0.0))
    def Z_direction_vector(self):
        # Directional vector in z-axis direction
       return Constant((0.0, 0.0, 1.0))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Solid gravitational/body forces define
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def b_for (self):

        b_for_ext = Constant((self.bForExtX, self.bForExtY, self.bForExtZ)) # Body external forces [N/m^3]

        if self.iGravForce:
            g_force = Constant((0.0, (self.rho_s * (-9.81)), 0.0))          # Gravitational force [N/m^3]
        else:
            g_force = Constant((0.0, (0.0 * (-9.81)), 0.0))                 # Gravitational force [N/m^3]

        return (b_for_ext + g_force)                                        # Body total forces [N/m^3]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Lame parameters
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Define the Lamé's first parameter
    def lamda_s (self):
        return (2.0*(self.mu_s())*self.nu_s/(1.0-2.0*self.nu_s))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Generalized-alpha method functions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Define the acceleration at the present time step
    def Acceleration_March_Term_One(self, 
                                    displacement_function, 
                                    displacement_previous_function, 
                                    velocity_previous_function):
        return (2* (displacement_function - displacement_previous_function - 
                (self.dt*velocity_previous_function))/
                (self.dt**2))

    def Acceleration_March_Term_Two(self, 
                                    acceleration_previous_function,
                                    beta_gam):
        return ((1-(2*beta_gam)) * acceleration_previous_function)

    def Acceleration_March_Term_Three(self, beta_gam):
        return (1 / (2*beta_gam))

    def AMCK (  self, 
                displacement_function, 
                displacement_previous_function, 
                velocity_previous_function,
                acceleration_previous_function,
                beta_gam):
        return (self.Acceleration_March_Term_Three(beta_gam) * 
                (self.Acceleration_March_Term_One(displacement_function,
                displacement_previous_function,velocity_previous_function) - 
                self.Acceleration_March_Term_Two(acceleration_previous_function,beta_gam)))

    # Define the velocity at the present time step
    def Velocity_March_Term_One(self, 
                                acceleration_previous_function,
                                gamma_gam):
        return ((1-gamma_gam)*acceleration_previous_function * self.dt)

    def Velocity_March_Term_Two(self, 
                                acceleration_function,
                                gamma_gam):
        return (acceleration_function * gamma_gam * self.dt)

    def UMCK (  self, 
                acceleration_function, 
                velocity_previous_function, 
                acceleration_previous_function, 
                gamma_gam):
        return (self.Velocity_March_Term_One(acceleration_previous_function,gamma_gam) + 
                self.Velocity_March_Term_Two(acceleration_function,gamma_gam) + 
                velocity_previous_function)

    # define the calculation of intermediate averages based on generalized alpha method
    def Generalized_Alpha_Weights ( self,
                                    present_function,
                                    previous_function,
                                    weights):
        return (weights * previous_function + \
               (1-weights) * present_function)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Stress, force gradient and its
    #%% determination functions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def I (self, grid_dimension):
        # Define the Identity matrix
        return (Identity(grid_dimension))       

    def F_ (self, displacement_function, grid_dimension):
        # Define the deformation gradient
        return (self.I(grid_dimension)+nabla_grad(displacement_function))

    def J_ (self, displacement_function, grid_dimension):
        # Define the determinant of the deformation gradient
        return det(self.F_(displacement_function,grid_dimension))

    def C (self, displacement_function, grid_dimension):
        # Define the right Cauchy-Green strain tensor
        return ((self.F_(displacement_function,grid_dimension).T)*
                self.F_(displacement_function,grid_dimension)) 

    def E (self, displacement_function, grid_dimension):
        # Define the non-linear Lagrangian Green strain tensor
        return (0.5*(self.C(displacement_function,grid_dimension)-self.I(grid_dimension))) 

    def epsilon (self, displacement_function, grid_dimension):
        # Define the linear Lagrangian Green strain tensor
        return (0.5*(nabla_grad(displacement_function)+
                (nabla_grad(displacement_function).T))) 

    def Piola_Kirchhoff_sec(self, displacement_function, strain_tensor, grid_dimension):
        # Define the Second Piola-Kirchhoff stress tensor by the constitutive law 
        #   of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation). 
        #   Valid for large deformations but small strain.
        return (self.lamda_s()*tr(strain_tensor(displacement_function,grid_dimension))*
                self.I(grid_dimension)+2.0*self.mu_s()*
                strain_tensor(displacement_function,grid_dimension))

    def cauchy_stress (self, displacement_function, strain_tensor, grid_dimension):
        # Define the Cauchy stress tensor
        return ((1/self.J_(displacement_function,grid_dimension))*
                (self.F_(displacement_function,grid_dimension))*
                (self.Piola_Kirchhoff_sec(displacement_function,strain_tensor,grid_dimension))*
                (self.F_(displacement_function,grid_dimension).T))

    def Piola_Kirchhoff_fst(self, displacement_function, grid_dimension):
        # Define the First Piola-Kirchhoff stress tensor by the constitutive law 
        #   of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation). 
        #   Valid for large deformations but small strain.
        return (self.J_(displacement_function,grid_dimension)*
                self.cauchy_stress(displacement_function,self.E,grid_dimension)*
                inv(self.F_(displacement_function,grid_dimension).T))

    def Hooke_stress(self, displacement_function, grid_dimension):
        # Define the First Piola-Kirchhoff stress tensor by Hooke's law (linear relation). 
        #   Valid for small-scale deformations only.
        return (self.J_(displacement_function,grid_dimension)*
                self.cauchy_stress(displacement_function,self.epsilon,grid_dimension)*
                inv(self.F_(displacement_function,grid_dimension).T))

    def elastic_stress (self, displacement_function, grid_dimension):
        # Define the elastic stress tensor
        return (2.0*self.mu_s()*sym(grad(displacement_function))+ 
                self.lamda_s()*tr(sym(grad(displacement_function)))*self.I(grid_dimension))

    def Traction_Define(self,
                        VectorFunctionSpace):
        if self.iNonUniTraction:
            if self.rank == 0: print ("{FENICS} Non-uniform traction applied")
            self.tF_apply = Function(VectorFunctionSpace)
            self.tF_apply_vec = self.tF_apply.vector().get_local()

        else:
            if self.rank == 0: print ("{FENICS} Uniform traction applied")
            self.tF_magnitude = Constant(0.0 *self.X_direction_vector() +
                                    0.0 *self.Y_direction_vector() +
                                    0.0 *self.Z_direction_vector() )
            self.tF_apply = self.tF_magnitude

    def Traction_Assign(self,
                        xyz_fetch,
                        dofs_fetch_list,
                        t_sampler,
                        s_sampler,
                        t_sub_it,
                        areaf_vec):
        # Assign traction forces at present time step
        if self.iNonUniTraction:
            if len(xyz_fetch)!=0:
                # Execute only when there are DoFs need to exchange data in this rank.
                self.tF_apply_vec = self.MUI_Fetch( xyz_fetch,
                                                    dofs_fetch_list,
                                                    t_sampler,
                                                    s_sampler,
                                                    t_sub_it,
                                                    self.tF_apply_vec,
                                                    areaf_vec)

            if (self.iMUIFetchValue) and (not ((self.iContinueRun) and (n_steps == 1))):
                # Apply traction components. These calls do parallel communication
                self.tF_apply.vector().set_local(self.tF_apply_vec)
                self.tF_apply.vector().apply("insert")
            else:
                # Do not apply the fetched value, i.e. one-way coupling
                pass

        else:
            if self.rank == 0: print ("{FENICS} Assigning uniform traction forces at present time step ...   ",
                                    end="", flush=True)
            if t <= sForExtEndTime:
                self.tF_magnitude.assign((Constant((self.sForExtX)/(self.YBeam*self.ZBeam))*self.X_direction_vector()) +
                                    (Constant((self.sForExtY)/(self.XBeam*self.ZBeam))*self.Y_direction_vector()) +
                                    (Constant((self.sForExtZ)/(self.XBeam*self.YBeam))*self.Z_direction_vector()))
            else:
                self.tF_magnitude.assign(Constant((0.0)))
            if self.rank == 0:
                print ("Done")




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Set form compiler options
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Set_Compiler_Options(self):
        parameters["allow_extrapolation"] = self.allow_extrapolation
        parameters["form_compiler"]["optimize"] = self.optimize
        parameters["form_compiler"]["cpp_optimize"] = self.cppOptimize
        parameters["form_compiler"]["representation"] = self.compRepresentation
        parameters["ghost_mode"] = self.ghost_mode
        parameters["mesh_partitioner"] = "SCOTCH"
        parameters["partitioning_approach"] = "PARTITION"
        info(parameters, False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% ALE Move Mesh
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Move_Mesh ( self, 
                    VectorFunctionSpace, 
                    displacement_function, 
                    displacement_function_previous, 
                    mesh):
        dOffset = Function(VectorFunctionSpace)
        # Calculate offset of the displacement
        dOffset.vector()[:] = displacement_function.vector().get_local()- \
                              displacement_function_previous.vector().get_local()
        # Move the mesh by ALE function
        ALE.move(mesh, dOffset)
        # A cached search tree needs to be explicitly updated after 
        #   moving/deforming a mesh and before evaluating any function (even by assembler) 
        #   without cell argument
        mesh.bounding_box_tree().build(mesh)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Print log information
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Pre_Solving_Log(self, MPI_COMM_WORLD):

        if self.rank == 0: 
            print ("\n")
            print ("{FENICS} ********** STRUCTURAL-ELASTICITY SIMULATION BEGIN **********")
            print ("\n")
            if self.iDebug:
                print ("{FENICS} ### !!! DEBUG LEVEL ON !!! ###")
                print ("\n")

            if self.iMUICoupling:
                print ("{FENICS} ### !!! MUI COUPLING ON !!! ###")
                print ("\n")
                
            print ("{FENICS} Current Date and Time: ", datetime.datetime.now())
            print ("{FENICS} System Host Name: ", socket.gethostbyaddr(socket.gethostname())[0])
            print ("\n")

            print ("{FENICS} Solver info: ")
            if self.solving_method == 'STVK':
                print ("{FENICS} Solver for the problem: ", self.prbsolver)
                print ("{FENICS} Solver for project between domains: ", self.prjsolver)
                print ("{FENICS} Pre-conditioner for the problem: ", self.prbpreconditioner)
                print ("{FENICS} Relative tolerance: ", self.prbRelative_tolerance)
                print ("{FENICS} Absolute tolerance: ", self.prbAbsolute_tolerance)
                print ("{FENICS} Maximum iterations: ", self.prbMaximum_iterations)
                print ("{FENICS} Relaxation parameter: ", self.prbRelaxation_parameter)
                print ("{FENICS} Representation of the compiler: ", self.compRepresentation)
                print ("{FENICS} C++ code optimization: ", self.cppOptimize)
                print ("{FENICS} optimization of the compiler: ", self.optimize)
                print ("{FENICS} Extrapolation: ", self.allow_extrapolation)
                print ("{FENICS} Ghost cell mode: ", self.ghost_mode)
                print ("{FENICS} Error of non convergence: ", self.error_on_nonconvergence)
            elif self.solving_method == 'MCK':
                print ("{FENICS} Solver for the problem: ", self.prbsolver)
                print ("{FENICS} Solver for project between domains: ", self.prjsolver)
            print ("\n")

            print ("{FENICS} Input parameters: ")
            print ("{FENICS} E: ", self.E_s(), "[Pa]")
            print ("{FENICS} rho: ", self.rho_s, "[kg/m^3]")
            print ("{FENICS} nu: ", self.nu_s, "[-]")
            print ("\n")
        else:
            pass

    def Time_Marching_Log(self):
        if self.rank == 0: 
            print ("\n")
            print ("{FENICS} Total time: ", self.T, " [s]")
            print ("{FENICS} Time step size: ", self.dt, " [s]")
            print ("{FENICS} Time steps: ", self.Start_Time, " [-]")
            print ("{FENICS} Start time: ", self.Time_Steps, " [s]")
            print ("{FENICS} Numbers of sub-iterations: ", self.num_sub_iteration, " [-]")
            print ("\n")

    def print_Disp (self, MPI_COMM_WORLD, displacement_function):
        # Compute and print the displacement of monitored point
        d_DispSum = np.zeros(3)
        d_tempDenominator  = np.array([ self.size, 
                                        self.size, 
                                        self.size])
        MPI_COMM_WORLD.Reduce((displacement_function(
                                Point(self.pointMoniX,self.pointMoniY,self.pointMoniZ))),
                                d_DispSum,op=MPI.SUM,root=0)
        d_Disp = np.divide(d_DispSum,d_tempDenominator)
        if self.rank == 0: 
            print ("{FENICS} Monitored point deflection [m]: ", d_Disp)

    def Export_Disp_txt(    self, 
                            MPI_COMM_WORLD, 
                            displacement_function, 
                            OutputFolderPath):
        if self.iExporttxt:
            pointMoniDispSum = np.zeros(3)
            tempDenominator  = np.array([self.size, 
                                         self.size, 
                                         self.size])
            MPI_COMM_WORLD.Reduce((displacement_function(
                                    Point(self.pointMoniX,self.pointMoniY,self.pointMoniZ))),
                                    pointMoniDispSum,op=MPI.SUM,root=0)
            pointMoniDisp = np.divide(pointMoniDispSum,tempDenominator)

            pointMoniDispSum_b = np.zeros(3)
            tempDenominator_b  = np.array([self.size, 
                                         self.size, 
                                         self.size])
            MPI_COMM_WORLD.Reduce((displacement_function(
                                    Point(self.pointMoniXb,self.pointMoniYb,self.pointMoniZb))),
                                    pointMoniDispSum_b,op=MPI.SUM,root=0)
            pointMoniDisp_b = np.divide(pointMoniDispSum_b,tempDenominator_b)

            for irank in range(self.size):
                if self.rank == irank:
                    ftxt_dispX = open(OutputFolderPath + "/tip-displacementX_" + str(irank)+ ".txt", "a")
                    ftxt_dispX.write(str(pointMoniDisp[0]))
                    ftxt_dispX.write("\n")
                    ftxt_dispX.close

                    ftxt_dispY = open(OutputFolderPath + "/tip-displacementY_" + str(irank)+ ".txt", "a")
                    ftxt_dispY.write(str(pointMoniDisp[1]))
                    ftxt_dispY.write("\n")
                    ftxt_dispY.close

                    ftxt_dispZ = open(OutputFolderPath + "/tip-displacementZ_" + str(irank)+ ".txt", "a")
                    ftxt_dispZ.write(str(pointMoniDisp[2]))
                    ftxt_dispZ.write("\n")
                    ftxt_dispZ.close
                    
                    ftxt_dispXb = open(OutputFolderPath + "/tip-displacementXb_" + str(irank)+ ".txt", "a")
                    ftxt_dispXb.write(str(pointMoniDisp_b[0]))
                    ftxt_dispXb.write("\n")
                    ftxt_dispXb.close

                    ftxt_dispYb = open(OutputFolderPath + "/tip-displacementYb_" + str(irank)+ ".txt", "a")
                    ftxt_dispYb.write(str(pointMoniDisp_b[1]))
                    ftxt_dispYb.write("\n")
                    ftxt_dispYb.close

                    ftxt_dispZb = open(OutputFolderPath + "/tip-displacementZb_" + str(irank)+ ".txt", "a")
                    ftxt_dispZb.write(str(pointMoniDisp_b[2]))
                    ftxt_dispZb.write("\n")
                    ftxt_dispZb.close

    def Export_Disp_vtk(    self, 
                            MPI_COMM_WORLD, 
                            Current_Time_Step, 
                            current_time, 
                            mesh, 
                            grid_dimension, 
                            VectorFunctionSpace, 
                            traction_function, 
                            displacement_function, 
                            stress_file, 
                            disp_file, 
                            traction_file):
        # Export post-processing files
        if ((self.rank == 0) and self.iDebug):
            print ("\n")
            print ("{FENICS} time steps: ", Current_Time_Step, 
                    " output_interval: ", self.output_interval, 
                    " %: ", (Current_Time_Step % self.output_interval))

        if (Current_Time_Step % self.output_interval) == 0:
            if self.rank == 0: 
                print ("\n")
                print ("{FENICS} Export files at ", current_time, " [s] ...   ", end="", flush=True)

            # Compute stress
            Vsig = TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc)
            sig = Function(Vsig, name="Stress")
            if self.iNonLinearMethod:
                sig.assign(project(self.Piola_Kirchhoff_sec(
                            displacement_function,self.E,grid_dimension), 
                            Vsig, solver_type=self.prjsolver,
                            form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                            "representation": self.compRepresentation}))
            else:
                sig.assign(project(self.Piola_Kirchhoff_sec(
                            displacement_function,self.epsilon,grid_dimension), 
                            Vsig, solver_type=self.prjsolver,
                            form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                            "representation": self.compRepresentation}))
            # Save stress solution to file
            sig.rename('Piola Kirchhoff sec Stress', 'stress')
            stress_file << (sig, float(current_time))

            # Save displacement solution to file
            displacement_function.rename('Displacement', 'disp')
            disp_file << (displacement_function, float(current_time))

            # Compute traction
            traction = Function(VectorFunctionSpace, name="Traction")
            traction.assign(project(traction_function, 
                                    VectorFunctionSpace, 
                                    solver_type=self.prjsolver,
                                    form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                                    "representation": self.compRepresentation}))
            # Save traction solution to file
            traction.rename('traction', 'trac')
            traction_file << (traction, float(current_time))
            if self.rank == 0: print ("Done")
        else:
            pass

    def Post_Solving_Log(self, MPI_COMM_WORLD, simtime):

        if self.rank == 0:
            print ("\n")
            print ("{FENICS} Current Date and Time: ", datetime.datetime.now())
            print ("\n")
            print ("{FENICS} Total Simulation time: %g [s]" % simtime)
            print ("\n")
            print ("{FENICS} ********** STRUCTURAL-ELASTICITY SIMULATION COMPLETED **********")

    def Create_Post_Process_Files(self):

        if self.rank == 0: print ("{FENICS} Preparing post-process files ...   ", end="", flush=True)

        self.disp_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/displacement.pvd")
        self.stress_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/stress.pvd")
        self.traction_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/surface_traction_structure.pvd")

        if self.rank == 0: print ("Done")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Setup checkpoint file
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Checkpoint_Output(  self, 
                            current_time, 
                            mesh, 
                            ud_Functions_previous, 
                            d0mck_Functions_previous, 
                            u0mck_Functions_previous, 
                            a_Function_previous, 
                            ud_Functions, 
                            dmck_Function, 
                            t_Function, 
                            File_Exists=True):

        if File_Exists:
            import os
            if self.rank == 0:
                os.remove(self.outputFolderPath + "/checkpointData_" + str(current_time) +".h5")
            self.LOCAL_COMM_WORLD.Barrier()
        else:
            pass

        hdf5checkpointDataOut = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/checkpointData_" + str(current_time) +".h5", "w")
        hdf5checkpointDataOut.write(mesh, "/mesh")
        hdf5checkpointDataOut.write(ud_Functions_previous, "/u0d0", current_time)
        hdf5checkpointDataOut.write(d0mck_Functions_previous, "/d0mck", current_time)
        hdf5checkpointDataOut.write(u0mck_Functions_previous, "/u0mck", current_time)
        hdf5checkpointDataOut.write(a_Function_previous, "/a0mck", current_time)
        hdf5checkpointDataOut.write(ud_Functions, "/ud", current_time)
        hdf5checkpointDataOut.write(dmck_Function, "/dmck", current_time)
        hdf5checkpointDataOut.write(t_Function, "/sigma_s", current_time)
        hdf5checkpointDataOut.write(self.areaf, "/areaf")
        hdf5checkpointDataOut.close()
        # Delete HDF5File object, closing file
        del hdf5checkpointDataOut

    def Load_Functions_Continue_Run(self,
                                    u0d0,
                                    d0mck,
                                    u0mck,
                                    a0mck,
                                    ud,
                                    dmck,
                                    sigma_s):
        if self.iContinueRun:
            hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
            hdf5checkpointDataInTemp.read(u0d0, "/u0d0/vector_0")
            hdf5checkpointDataInTemp.read(d0mck, "/d0mck/vector_0")
            hdf5checkpointDataInTemp.read(u0mck, "/u0mck/vector_0")
            hdf5checkpointDataInTemp.read(a0mck, "/a0mck/vector_0")
            hdf5checkpointDataInTemp.read(ud, "/ud/vector_0")
            hdf5checkpointDataInTemp.read(dmck, "/dmck/vector_0")
            hdf5checkpointDataInTemp.read(sigma_s, "/sigma_s/vector_0")
            hdf5checkpointDataInTemp.close()
            # Delete HDF5File object, closing file
            del hdf5checkpointDataInTemp
        else:
            pass

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
        #%% Initialize MPI by mpi4py/MUI for parallelized computation
        #===========================================

        self.MPI_Init()

        #===========================================
        #%% Set target folder
        #===========================================

        # Folder directory
        if self.iAbspath:
            self.outputFolderPath = os.path.abspath(self.outputFolderName)
            self.inputFolderPath = os.path.abspath(self.inputFolderName)
        else:
            self.outputFolderPath = self.outputFolderName
            self.inputFolderPath = self.inputFolderName

        #===========================================
        #%% Print log information
        #===========================================

        self.Pre_Solving_Log(self.LOCAL_COMM_WORLD)

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

        if self.solving_method == 'STVK':
            pass
        elif self.solving_method == 'MCK':
            self.linearElasticSolve()

        #===========================================
        #%% Calculate wall time
        #===========================================

        # Finish the wall clock
        simtime = wallClock.toc()

        self.Post_Solving_Log(self.LOCAL_COMM_WORLD, simtime)

    # def Solver_OLD(self):
        # #===========================================
        # #%% Setup the wall clock
        # #===========================================

        # # create an instance of the TicToc wall clock class
        # wallClock = structureFSISolver.tictoc.TicToc()
        # # Starts the wall clock
        # wallClock.tic()

        # #===========================================
        # #%% Initialize MPI by mpi4py/MUI for parallelized computation
        # #===========================================

        # if self.iMUICoupling:
            # #import mui4py
            # MPI_COMM_WORLD, ifaces3d = self.MPI_Init()
        # else:
            # MPI_COMM_WORLD = self.MPI_Init()

        # rank = self.rank
        # rank_size = self.size

        # #===========================================
        # #%% Set target folder
        # #===========================================

        # # Folder directory
        # if self.iAbspath:
            # outputFolderPath = os.path.abspath(self.outputFolderName)
            # inputFolderPath = os.path.abspath(self.inputFolderName)
        # else:
            # outputFolderPath = self.outputFolderName
            # inputFolderPath = self.inputFolderName

        # #===========================================
        # #%% Print log information
        # #===========================================

        # self.Pre_Solving_Log(MPI_COMM_WORLD)

        # #===========================================
        # #%% Set form compiler options
        # #===========================================

        # self.Set_Compiler_Options()

        # #===========================================
        # #%% Time marching parameters define
        # #===========================================

        # t, t_step, i_sub_it = \
            # self.Time_Marching_Parameters(MPI_COMM_WORLD, inputFolderPath)

        # self.Time_Marching_Log(MPI_COMM_WORLD, t, t_step)

        # #===========================================
        # #%% Solid Mesh input/generation
        # #===========================================

        # mesh, meshOri, gdim, gdimOri, N = \
            # self.Mesh_Generation(MPI_COMM_WORLD, inputFolderPath, outputFolderPath)

        # #===========================================
        # #%% Define coefficients
        # #===========================================

        # # Time step constants
        # k = Constant(self.dt)
        # # Time lists
        # times = []
        # t_sub_it = 0

        # # One-step theta value
        # theta = Constant(self.thetaOS)

        # # Rayleigh damping coefficients
        # alpha_rdc = Constant(self.alpha_rdc)
        # beta_rdc  = Constant(self.beta_rdc)

        # # Generalized-alpha method parameters
        # # alpha_m_gam >= alpha_f_gam >= 0.5 for a better performance
        # alpha_m_gam = Constant(self.alpha_m_gam)
        # alpha_f_gam = Constant(self.alpha_f_gam)
        # gamma_gam   = Constant((1./2.) + alpha_m_gam - alpha_f_gam)
        # beta_gam    = Constant((1./4.) * (gamma_gam + (1./2.))**2)

        # sync = False

        # if rank == 0:
            # print ("\n")
            # print ("{FENICS} One-step theta: ", float(theta))
            # print ("\n")

        # #===========================================
        # #%% Define function spaces
        # #===========================================

        # if rank == 0: print ("{FENICS} Creating function spaces ...   ")

        # V_ele     =     VectorElement("Lagrange", mesh.ufl_cell(), self.deg_fun_spc) # Displacement & Velocity Vector element

        # QOri      =     FunctionSpace(meshOri, "Lagrange", self.deg_fun_spc)         # Function space by original mesh
        # #SOOri     =     FunctionSpace(meshOri, "Lagrange", 1)                           # Function space with 1st order
        # SO        =     FunctionSpace(mesh, "Lagrange", self.deg_fun_spc)            # Function space with updated mesh
        # #VS        =     VectorFunctionSpace(mesh, "Lagrange", 1)                     # Vector function space with 1st order
        # V         =     VectorFunctionSpace(mesh, "Lagrange", self.deg_fun_spc)      # Vector function space
        # VV        =     FunctionSpace(mesh, MixedElement([V_ele, V_ele]))            # Mixed (Velocity (w) & displacement (d)) function space
        # T_s_space =     TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc)      # Define nth order structure function spaces

        # if rank == 0: print ("{FENICS} Done with creating function spaces")

        # #===========================================
        # #%% Define functions, test functions and trail functions
        # #===========================================

        # if rank == 0: print ("{FENICS} Creating functions, test functions and trail functions ...   ", end="", flush=True)

        # # Trial functions
        # du, dd = TrialFunctions(VV)     # Trial functions for velocity and displacement
        # ddmck = TrialFunction(V)        # Trial function for displacement by MCK solving method

        # # Test functions
        # psi, phi = TestFunctions(VV)    # Test functions for velocity and displacement
        # chi = TestFunction(V)           # Test function for displacement by MCK solving method
        
        # # Functions at present time step
        # ud = Function(VV)               # Functions for velocity and displacement
        # u, d = split(ud)                # Split velocity and displacement functions
        # dmck = Function(V)              # Function for displacement by MCK solving method


        # # Functions at previous time step
        # u0d0 = Function(VV)             # Functions for velocity and displacement
        # u0, d0 = split(u0d0)            # Split velocity and displacement functions
        # d0mck = Function(V)             # Function for displacement by MCK solving method
        # u0mck = Function(V)             # Function for velocity by MCK solving method
        # a0mck = Function(V)             # Function for acceleration by MCK solving method

        # # Define structure traction
        # sigma_s = Function(T_s_space)   # Structure traction normal to structure

        # #dfst = Function(VS)             # Function for displacement with 1st order
        # areaf= Function(QOri)             # Function for facet area

        # if self.iContinueRun:
            # hdf5checkpointDataInTemp = HDF5File(MPI_COMM_WORLD, inputFolderPath + "/checkpointData.h5", "r")
            # hdf5checkpointDataInTemp.read(u0d0, "/u0d0/vector_0")
            # hdf5checkpointDataInTemp.read(d0mck, "/d0mck/vector_0")
            # hdf5checkpointDataInTemp.read(u0mck, "/u0mck/vector_0")
            # hdf5checkpointDataInTemp.read(a0mck, "/a0mck/vector_0")
            # hdf5checkpointDataInTemp.read(ud, "/ud/vector_0")
            # hdf5checkpointDataInTemp.read(dmck, "/dmck/vector_0")
            # hdf5checkpointDataInTemp.read(sigma_s, "/sigma_s/vector_0")
            # #hdf5checkpointDataInTemp.read(areaf, "/areaf/vector_0")
            # hdf5checkpointDataInTemp.close()
            # # Delete HDF5File object, closing file
            # del hdf5checkpointDataInTemp
        # else:
            # pass

        # if rank == 0: print ("Done")

        # #===========================================
        # #%% Define traction forces
        # #===========================================

        # if self.iNonUniTraction:
            # if rank == 0: print ("{FENICS} Non-uniform traction applied")
            # tF_apply = Function(V)
            # tF_apply_vec = tF_apply.vector().get_local()

            # if self.iMUIFetchForce:
                # force_dof_apply = Function(V)
                # force_dof_apply_vec = force_dof_apply.vector().get_local()
        # else:
            # if rank == 0: print ("{FENICS} Uniform traction applied")
            # tF_magnitude = Constant(-(0.0)/(self.YBeam*self.ZBeam))
            # tF_apply = tF_magnitude*self.X_direction_vector()

        # #===========================================
        # #%% Define SubDomains and boundaries
        # #===========================================

        # boundaries, boundariesOri, ds = \
            # self.SubDomains_Boundaries_Generation(  MPI_COMM_WORLD, 
                                                    # mesh, 
                                                    # meshOri, 
                                                    # gdim, 
                                                    # gdimOri, 
                                                    # V, 
                                                    # inputFolderPath, 
                                                    # outputFolderPath)

        # #===========================================
        # #%% Define boundary conditions
        # #===========================================

        # if rank == 0: print ("{FENICS} Creating 3D boundary conditions ...   ", end="", flush=True)
        # if self.solving_method == 'STVK':
            # bc1,bc2 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
            # #bc1 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
            # #!!!!->   
            # #bc3 = DirichletBC(VV.sub(0).sub(1), (0.0),boundaries, 8)
            # #bc4 = DirichletBC(VV.sub(1).sub(1), (0.0),boundaries, 8)
            # #!!!!<-  
            # #bcs = [bc1,bc2,bc3,bc4]
            # bcs = [bc1,bc2]
        # elif self.solving_method == 'MCK':
            # bc1 = self.dirichletBCs.DirichletBCs(V,boundaries,1)
        # #!!!!->   
            # #bc2 = DirichletBC(V.sub(1), (0.0),boundaries, 8)
            # #bc3 = DirichletBC(V.sub(0), (0.0),boundaries, 8)
            # #bc2 = DirichletBC(V, ((0.0,0.0,0.0)),boundaries, 8)
        # #!!!!<-    
            # #bcs = [bc1,bc2]
            # bcs = [bc1]
        # if rank == 0: print ("Done")

        # #===========================================
        # #%% Define DOFs and Coordinates mapping
        # #===========================================  

        # dofs_to_xyz = self.dofs_to_xyz(QOri, gdimOri)

        # dofs_fetch, dofs_fetch_list, xyz_fetch = \
            # self.dofs_fetch_list(boundariesOri, QOri, 2, gdimOri)
        # dofs_push, dofs_push_list, xyz_push = \
            # self.dofs_push_list(boundariesOri, QOri, 2, gdimOri)

        # xyz_fetch_list =  list(xyz_fetch)
        # xyz_fetch_list_total_group = []
        # #print ("{FEniCS***} out: len(dofs_fetch_list): ", len(dofs_fetch_list), " len(dofs_push_list): ", len(dofs_push_list))

        # #===========================================
        # #%% Define facet areas
        # #===========================================

        # areaf_vec = areaf.vector().get_local()

        # if (self.iMUIFetchForce):
            # if (self.iUseRBF):

                # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # local = np.zeros(1)
                # total = np.zeros(1)

                # local[0] = len(xyz_fetch_list)
                # total[0] = 0

                # xyz_fetch_list_flat = [item for sublist in xyz_fetch_list for item in sublist]

                # # use MPI to get the totals 
                # MPI_COMM_WORLD.Reduce(local,total,op = MPI.SUM,root = 0)
                # MPI_COMM_WORLD.Bcast(total, root=0)

                # xyz_fetch_list_total_flat = np.empty(int(total[0]*3), dtype=np.float64)

                # xyz_fetch_list_total = MPI_COMM_WORLD.gather(xyz_fetch_list_flat, root = 0)
                # if MPI_COMM_WORLD.rank == 0:
                    # xyz_fetch_list_total_flat = np.asarray([item for sublist in xyz_fetch_list_total for item in sublist])

                # MPI_COMM_WORLD.Bcast(xyz_fetch_list_total_flat, root=0)
                # xyz_fetch_list_total_group = [ xyz_fetch_list_total_flat.tolist()[i:i+3]
                                                    # for i in range(0, len(xyz_fetch_list_total_flat.tolist()), 3) ]

                # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # if (not self.iContinueRun):
                # if (not self.iLoadAreaList):
                    # if rank == 0: print ("{FENICS} facet area calculating")

                    # areaf_vec = self.facets_area_list(  MPI_COMM_WORLD,
                                                        # meshOri,
                                                        # QOri,
                                                        # boundariesOri,
                                                        # dofs_fetch_list,
                                                        # gdimOri,
                                                        # areaf_vec)

                    # # Apply the facet area vectors
                    # areaf.vector().set_local(areaf_vec)
                    # areaf.vector().apply("insert")
                    # if (self.iHDF5FileExport) and (self.iHDF5MeshExport):
                        # hdfOutTemp = HDF5File(MPI_COMM_WORLD, outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                    # else:
                        # hdfOutTemp = HDF5File(MPI_COMM_WORLD, outputFolderPath + "/mesh_boundary_and_values.h5", "w")
                    # hdfOutTemp.write(areaf, "/areaf")
                    # hdfOutTemp.close()

                # else:

                    # hdf5meshAreaDataInTemp = HDF5File(MPI_COMM_WORLD, inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                    # hdf5meshAreaDataInTemp.read(areaf, "/areaf/vector_0")
                    # hdf5meshAreaDataInTemp.close()

            # else:
                # if (not self.iLoadAreaList):
                    # if rank == 0: print ("{FENICS} facet area calculating")

                    # areaf_vec = self.facets_area_list(  MPI_COMM_WORLD,
                                                        # meshOri,
                                                        # QOri,
                                                        # boundariesOri,
                                                        # dofs_fetch_list,
                                                        # gdimOri,
                                                        # areaf_vec)

                    # # Apply the facet area vectors
                    # areaf.vector().set_local(areaf_vec)
                    # areaf.vector().apply("insert")
                    # if (self.iHDF5FileExport) and (self.iHDF5MeshExport):
                        # hdfOutTemp = HDF5File(MPI_COMM_WORLD, outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                    # else:
                        # hdfOutTemp = HDF5File(MPI_COMM_WORLD, outputFolderPath + "/mesh_boundary_and_values.h5", "w")
                    # hdfOutTemp.write(areaf, "/areaf")
                    # hdfOutTemp.close()

                # else:

                    # hdf5meshAreaDataInTemp = HDF5File(MPI_COMM_WORLD, inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                    # hdf5meshAreaDataInTemp.read(areaf, "/areaf/vector_0")
                    # hdf5meshAreaDataInTemp.close()

        # #===========================================
        # #%% Prepare post-process files
        # #===========================================

        # if rank == 0: print ("{FENICS} Preparing post-process files ...   ", end="", flush=True)

        # disp_file = File(MPI_COMM_WORLD, outputFolderPath + "/displacement.pvd")
        # stress_file = File(MPI_COMM_WORLD, outputFolderPath + "/stress.pvd")
        # traction_file = File(MPI_COMM_WORLD, outputFolderPath + "/surface_traction_structure.pvd")

        # if rank == 0: print ("Done")

        # #===========================================
        # #%% Define the variational FORM 
        # #%% and 
        # #%% Jacobin functions of structure
        # #===========================================

        # if self.solving_method == 'STVK':

            # if rank == 0: print ("{FENICS} Defining variational FORM and Jacobin functions ...   ", end="", flush=True)

            # # Define the traction terms of the structure variational form
            # tF = dot(self.F_(d,gdim).T, tF_apply)
            # tF_ = dot(self.F_(d0,gdim).T, tF_apply)

            # # Define the transient terms of the structure variational form
            # Form_s_T = (1/k)*self.rho_s*inner((u-u0), psi)*dx
            # Form_s_T += (1/k)*inner((d-d0), phi)*dx

            # # Define the stress terms and convection of the structure variational form
            # if self.iNonLinearMethod:
                # if rank == 0: print ("{FENICS} [Defining non-linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by the constitutive law of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation). Valid for large deformations but small strain] ...   ", end="", flush=True)
                # Form_s_SC = inner(theta * self.Piola_Kirchhoff_fst(d,gdim) + (1 - theta) * 
                            # self.Piola_Kirchhoff_fst(d0,gdim), grad(psi)) * dx
                # Form_s_SC -= inner(theta*u + (1-theta)*u0, phi ) * dx
            # else:
                # if rank == 0: print ("{FENICS} [Defining linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by Hooke's law (linear relation). Valid for small-scale deformations only] ...   ", end="", flush=True)
                # Form_s_SC = inner(theta * self.Hooke_stress(d,gdim) + (1 - theta) * 
                            # self.Hooke_stress(d0,gdim), grad(psi)) * dx
                # Form_s_SC -= inner(theta*u + (1-theta)*u0, phi ) * dx

            # # Define the body forces and surface tractions terms of the structure variational form
            # Form_s_ET = -( theta * self.J_(d,gdim) * inner( (self.b_for()), psi ) + 
                        # ( 1 - theta ) * self.J_(d0,gdim) * inner( (self.b_for()), psi ) ) * dx
            # Form_s_ET -= ( theta * self.J_(d,gdim) * inner( tF, psi ) + 
                        # ( 1 - theta ) * self.J_(d0,gdim) * inner( tF_, psi ) ) * ds(2)
            # Form_s_ET -= ( theta * self.J_(d,gdim) * inner( inv(self.F_(d,gdim)) * sigma_s * N, psi )+ 
                        # ( 1 - theta ) * (self.J_(d0,gdim)) * inner(inv(self.F_(d0,gdim)) * sigma_s * N, psi )) * ds(2)

            # # Define the final form of the structure variational form
            # Form_s = Form_s_T + Form_s_SC + Form_s_ET

            # # Make functional into a vector function
            # #Form_s = action(Form_s, ud)

            # Jaco = derivative(Form_s, ud) # Define Jacobin functions

            # if rank == 0: print ("Done")

        # elif self.solving_method == 'MCK':
            # if rank == 0: print ("{FENICS} Defining variational FORM functions ...   ", end="", flush=True)
            # # Define the traction terms of the structure variational form
            # tF = dot(chi, tF_apply)

            # Form_s_Update_Acce = self.AMCK (ddmck, 
                                            # d0mck, 
                                            # u0mck,
                                            # a0mck,
                                            # beta_gam)
            
            # Form_s_Update_velo = self.UMCK (Form_s_Update_Acce, 
                                            # u0mck, 
                                            # a0mck, 
                                            # gamma_gam)

            # Form_s_Ga_Acce = self.Generalized_Alpha_Weights(Form_s_Update_Acce,a0mck,alpha_m_gam)
            # Form_s_Ga_velo = self.Generalized_Alpha_Weights(Form_s_Update_velo,u0mck,alpha_f_gam)
            # Form_s_Ga_disp = self.Generalized_Alpha_Weights(ddmck,d0mck,alpha_f_gam)
            # Form_s_M_Matrix = self.rho_s * inner(Form_s_Ga_Acce, chi) * dx
            # Form_s_M_for_C_Matrix = self.rho_s * inner(Form_s_Ga_velo, chi) * dx
            # Form_s_K_Matrix = inner(self.elastic_stress(Form_s_Ga_disp,gdim), sym(grad(chi))) * dx
            # Form_s_K_for_C_Matrix = inner(self.elastic_stress(Form_s_Ga_velo,gdim), sym(grad(chi))) * dx
            # Form_s_C_Matrix = alpha_rdc * Form_s_M_for_C_Matrix + beta_rdc * Form_s_K_for_C_Matrix
            # Form_s_F_Ext = tF * ds(2)

            # Form_s = Form_s_M_Matrix + Form_s_C_Matrix + Form_s_K_Matrix - Form_s_F_Ext

            # Bilinear_Form = lhs(Form_s)
            # Linear_Form   = rhs(Form_s)

            # if rank == 0: print ("Done")

        # #===========================================
        # #%% Initialize solver
        # #===========================================

        # if self.solving_method == 'STVK':

            # problem = NonlinearVariationalProblem(Form_s, ud, bcs=bcs, J=Jaco)
            # solver = NonlinearVariationalSolver(problem)

            # info(solver.parameters, False)
            # if self.nonlinear_solver == "newton":
                # solver.parameters["nonlinear_solver"]= self.nonlinear_solver
                # solver.parameters["newton_solver"]["absolute_tolerance"] = self.prbAbsolute_tolerance
                # solver.parameters["newton_solver"]["relative_tolerance"] = self.prbRelative_tolerance
                # solver.parameters["newton_solver"]["maximum_iterations"] = self.prbMaximum_iterations
                # solver.parameters["newton_solver"]["relaxation_parameter"] = self.prbRelaxation_parameter
                # solver.parameters["newton_solver"]["linear_solver"] = self.prbsolver
                # solver.parameters["newton_solver"]["preconditioner"] = self.prbpreconditioner
                # solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                # solver.parameters["newton_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                # solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                # solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                # solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess
                # solver.parameters["newton_solver"]["krylov_solver"]['error_on_nonconvergence'] = self.error_on_nonconvergence
            # elif self.nonlinear_solver == "snes":
                # solver.parameters['nonlinear_solver'] = self.nonlinear_solver
                # solver.parameters['snes_solver']['line_search'] = self.lineSearch
                # solver.parameters['snes_solver']['linear_solver'] = self.prbsolver
                # solver.parameters['snes_solver']['preconditioner'] = self.prbpreconditioner
                # solver.parameters['snes_solver']['absolute_tolerance'] = self.prbAbsolute_tolerance
                # solver.parameters['snes_solver']['relative_tolerance'] = self.prbRelative_tolerance
                # solver.parameters['snes_solver']['maximum_iterations'] = self.prbMaximum_iterations
                # solver.parameters['snes_solver']['report'] = self.show_report
                # solver.parameters['snes_solver']['error_on_nonconvergence'] = self.error_on_nonconvergence
                # solver.parameters["snes_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                # solver.parameters["snes_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                # solver.parameters["snes_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                # solver.parameters["snes_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                # solver.parameters["snes_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess
            # else:
                # sys.exit("{FENICS} Error, nonlinear solver value not recognized")

        # elif self.solving_method == 'MCK':

            # if self.linear_solver == 'LU':
                # Bilinear_Assemble, Linear_Assemble = assemble_system(Bilinear_Form, Linear_Form, bcs)
                # solver = LUSolver(Bilinear_Assemble, "mumps")
                # solver.parameters["symmetric"] = True

            # elif self.linear_solver == 'LinearVariational':
                # problem = LinearVariationalProblem(Bilinear_Form, Linear_Form, dmck, bcs)
                # solver = LinearVariationalSolver(problem)
                # # Set linear solver parameters
                # solver.parameters["linear_solver"] = self.prbsolver
                # solver.parameters["preconditioner"] = self.prbpreconditioner
                # solver.parameters["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                # solver.parameters["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                # solver.parameters["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                # solver.parameters["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                # solver.parameters["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess

        # #===========================================
        # #%% Setup checkpoint data
        # #===========================================

        # self.Checkpoint_Output( MPI_COMM_WORLD, 
                                # outputFolderPath, 
                                # (t-self.dt), 
                                # mesh, 
                                # meshOri, 
                                # u0d0, 
                                # d0mck, 
                                # u0mck, 
                                # a0mck, 
                                # ud, 
                                # dmck, 
                                # sigma_s, 
                                # areaf, 
                                # False)

        # #===========================================
        # #%% Define MUI samplers and commit ZERO step
        # #===========================================

        # if self.iMUICoupling:
            # if self.iUseRBF:
                # if self.iMUIFetchForce:
                    # if self.iparallelFSICoupling:
                        # t_sampler, s_sampler = \
                            # self.MUI_Sampler_RBF_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # dofs_to_xyz,
                                                    # dofs_push_list,
                                                    # dofs_to_xyz,
                                                    # xyz_fetch_list_total_group,
                                                    # t_step)
                    # else:
                        # t_sampler, s_sampler = \
                            # self.MUI_Sampler_RBF_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # dofs_to_xyz,
                                                    # dofs_push_list,
                                                    # dofs_to_xyz,
                                                    # xyz_fetch_list_total_group,
                                                    # t_step)
                # else:
                    # if self.iparallelFSICoupling:
                        # t_sampler, s_sampler = \
                            # self.MUI_Sampler_RBF_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # dofs_to_xyz,
                                                    # dofs_push_list,
                                                    # dofs_to_xyz,
                                                    # xyz_fetch_list_total_group,
                                                    # t_step)
                    # else:
                        # t_sampler, s_sampler = \
                            # self.MUI_Sampler_RBF_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # dofs_to_xyz,
                                                    # dofs_push_list,
                                                    # dofs_to_xyz,
                                                    # xyz_fetch_list_total_group,
                                                    # t_step)
            # else:
                # if self.iMUIFetchForce:
                    # t_sampler, s_sampler = \
                            # self.MUI_Sampler_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # xyz_fetch,
                                                    # dofs_push_list,
                                                    # xyz_push,
                                                    # t_step)
                # else:
                    # t_sampler, s_sampler = \
                            # self.MUI_Sampler_Define(MPI_COMM_WORLD, 
                                                    # ifaces3d,
                                                    # dofs_fetch_list,
                                                    # xyz_fetch,
                                                    # dofs_push_list,
                                                    # xyz_push,
                                                    # t_step)
            
        # else:
            # pass

        # if self.iExporttxt: self.Time_Txt_Export_init(MPI_COMM_WORLD, outputFolderPath)

        # # Finish the wall clock on fetch Time
        # fetchTime = 0.0
        # forceVecProjTime = 0.0
        # linearAssembleTime = 0.0
        # bcApplyTime = 0.0
        # solverSolveTime = 0.0
        # totalForceCalTime = 0.0
        # printDispTime = 0.0
        # pushTime = 0.0
        # moveMeshTime = 0.0
        # dispVTKExpTime = 0.0
        # dispTxtExpTime = 0.0
        # checkpointExpTime = 0.0
        # assignOldFuncSpaceTime = 0.0
        # simtimePerIter = 0.0
        # simtimePerStep = 0.0

        # #===========================================
        # #%% Define time loops
        # #===========================================

        # # Time-stepping
        # while t <= self.T:

            # # create an instance of the TicToc wall clock class
            # wallClockPerStep = structureFSISolver.tictoc.TicToc()
            # # Starts the wall clock
            # wallClockPerStep.tic()

            # # Update time list    
            # times.append(t)
            # n_steps = len(times)

            # if rank == 0: 
                # print ("\n")
                # print ("\n")
                # print ("{FENICS} Time: ", t)
                # print ("{FENICS} Time Steps: ", n_steps)

            # if (self.iChangeSubIter):
                # if (t >= self.TChangeSubIter):
                    # present_num_sub_iteration = self.num_sub_iteration_new
                # else:
                    # present_num_sub_iteration = self.num_sub_iteration
            # else:
                # present_num_sub_iteration = self.num_sub_iteration

            # # Sub-iteration for coupling
            # while i_sub_it <= present_num_sub_iteration:

                # t_sub_it += 1

                # if rank == 0: 
                    # print ("\n")
                    # print ("{FENICS} sub-iteration: ", i_sub_it)
                    # print ("{FENICS} total sub-iterations to now: ", t_sub_it)

                # # create an instance of the TicToc wall clock class
                # wallClockPerIter = structureFSISolver.tictoc.TicToc()
                # # Starts the wall clock
                # wallClockPerIter.tic()

                # # create an instance of the TicToc wall clock class
                # wallClockFetchTime = structureFSISolver.tictoc.TicToc()
                # wallClockForceVecProj = structureFSISolver.tictoc.TicToc()
                # wallClockLinearAssemble = structureFSISolver.tictoc.TicToc()
                # wallClockBCApply = structureFSISolver.tictoc.TicToc()
                # wallClockSolverSolve = structureFSISolver.tictoc.TicToc()
                # wallClockTotalForceCal = structureFSISolver.tictoc.TicToc()
                # wallClockPrintDisp = structureFSISolver.tictoc.TicToc()
                # wallClockPush = structureFSISolver.tictoc.TicToc()
                # wallClockMoveMesh = structureFSISolver.tictoc.TicToc()
                # wallClockDispVTKExp = structureFSISolver.tictoc.TicToc()
                # wallClockDispTxtExp = structureFSISolver.tictoc.TicToc()
                # wallClockCheckpointExp = structureFSISolver.tictoc.TicToc()
                # wallClockAssignOldFuncSpace = structureFSISolver.tictoc.TicToc()

                # # Assign traction forces at present time step
                # if self.iNonUniTraction:
                    # if self.iMUICoupling:
                        # if self.iMUIFetchForce:
                            # # Starts the wall clock
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # wallClockFetchTime.tic()
                            # if len(xyz_fetch)!=0:
                                # if self.iparallelFSICoupling:
                                    # if self.iUseRBF:
                                        # if self.iMUIFetchMoment:
                                            # tF_apply_vec, mom_x, mom_y, mom_z = \
                                                # self.MUI_Parallel_FSI_RBF_Fetch( MPI_COMM_WORLD,
                                                                            # ifaces3d,
                                                                            # xyz_fetch,
                                                                            # dofs_fetch_list,
                                                                            # t_sampler,
                                                                            # s_sampler,
                                                                            # n_steps,
                                                                            # i_sub_it,
                                                                            # t_sub_it,
                                                                            # tF_apply_vec,
                                                                            # areaf_vec,
                                                                            # outputFolderPath)

                                        # else:
                                            # tF_apply_vec = self.MUI_Parallel_FSI_RBF_Fetch(  MPI_COMM_WORLD, 
                                                                            # ifaces3d, 
                                                                            # xyz_fetch, 
                                                                            # dofs_fetch_list, 
                                                                            # t_sampler, 
                                                                            # s_sampler, 
                                                                            # n_steps,
                                                                            # i_sub_it,
                                                                            # t_sub_it, 
                                                                            # tF_apply_vec, 
                                                                            # areaf_vec,
                                                                            # outputFolderPath)
                                            # if len(dofs_fetch_list) != 0:
                                                # temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                            # if self.iMultidomain:

                                                # for i, p in enumerate(dofs_fetch_list):

                                                    # if t <= 0.5:
                                                        # tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                            # else:

                                                # for i, p in enumerate(dofs_fetch_list):

                                                    # if t <= 0.5:
                                                        # tF_apply_vec[0::3][p] += ((t*((self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[1::3][p] += ((t*((self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[2::3][p] += ((t*((self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)

                                    # else:
                                        # #nearest neighbour
                                        # if self.iMUIFetchMoment:
                                            # tF_apply_vec, mom_x, mom_y, mom_z = \
                                                # self.MUI_Parallel_FSI_Fetch( MPI_COMM_WORLD,
                                                                            # ifaces3d,
                                                                            # xyz_fetch,
                                                                            # dofs_fetch_list,
                                                                            # t_sampler,
                                                                            # s_sampler,
                                                                            # n_steps,
                                                                            # i_sub_it,
                                                                            # t_sub_it,
                                                                            # tF_apply_vec,
                                                                            # force_dof_apply_vec,
                                                                            # areaf_vec,
                                                                            # outputFolderPath)

                                        # else:
                                            # tF_apply_vec = self.MUI_Parallel_FSI_Fetch(  MPI_COMM_WORLD, 
                                                                            # ifaces3d, 
                                                                            # xyz_fetch, 
                                                                            # dofs_fetch_list, 
                                                                            # t_sampler, 
                                                                            # s_sampler, 
                                                                            # n_steps,
                                                                            # i_sub_it,
                                                                            # t_sub_it, 
                                                                            # tF_apply_vec, 
                                                                            # force_dof_apply_vec, 
                                                                            # areaf_vec,
                                                                            # outputFolderPath)

                                            # #ifaces3d.barrier(float(t_sub_it))
                                            
                                            # if len(dofs_fetch_list) != 0:
                                                # temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                            # if self.iMultidomain:

                                                # for i, p in enumerate(dofs_fetch_list):

                                                    # if t <= 0.5:
                                                        # tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                            # else:

                                                # for i, p in enumerate(dofs_fetch_list):

                                                    # if t <= 0.5:
                                                        # tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        # tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                # else:
                                    # if self.iMUIFetchMoment:
                                        # tF_apply_vec, mom_x, mom_y, mom_z = \
                                            # self.MUI_Fetch( MPI_COMM_WORLD, 
                                                            # ifaces3d, 
                                                            # xyz_fetch, 
                                                            # dofs_fetch_list, 
                                                            # t_sampler, 
                                                            # s_sampler, 
                                                            # n_steps, 
                                                            # t_sub_it, 
                                                            # tF_apply_vec, 
                                                            # areaf_vec)
                                    # else:
                                        # tF_apply_vec = self.MUI_Fetch(  MPI_COMM_WORLD, 
                                                                        # ifaces3d, 
                                                                        # xyz_fetch, 
                                                                        # dofs_fetch_list, 
                                                                        # t_sampler, 
                                                                        # s_sampler, 
                                                                        # n_steps, 
                                                                        # t_sub_it, 
                                                                        # tF_apply_vec, 
                                                                        # areaf_vec)
                                        # if len(dofs_fetch_list) != 0:
                                            # temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                        # if self.iMultidomain:

                                            # for i, p in enumerate(dofs_fetch_list):

                                                # if t <= 0.5:
                                                    # tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    # tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    # tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                        # else:

                                            # for i, p in enumerate(dofs_fetch_list):

                                                # if t <= 0.5:
                                                    # tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    # tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    # tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)

                                
                            # # Finish the wall clock on fetch Time
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # fetchTime = wallClockFetchTime.toc()

                            # # Starts the wall clock
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # wallClockForceVecProj.tic()
                            # if (self.iMUIFetchValue) and (not ((self.iContinueRun) and (n_steps == 1))):
                                # # Apply traction components. These calls do parallel communication
                                # tF_apply.vector().set_local(tF_apply_vec)
                                # tF_apply.vector().apply("insert")
                                # # tF_apply = project(dfst, V)
                                # # tF_apply.assign(project(dfst, 
                                    # # V, 
                                    # # solver_type=self.prjsolver,
                                    # # form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                                    # # "representation": self.compRepresentation}))
                            # else:
                                # # do not apply the fetched value, i.e. one-way coupling
                                # pass

                            # # Finish the wall clock on force vector project
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # forceVecProjTime = wallClockForceVecProj.toc()

                        # else:
                            # # Starts the wall clock
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # wallClockFetchTime.tic()
                            # if self.iparallelFSICoupling:
                                # if self.iMUIFetchMoment:
                                    # tF_apply_vec, mom_x, mom_y, mom_z = \
                                    # self.MUI_Parallel_FSI_Fetch( MPI_COMM_WORLD,
                                                                    # ifaces3d,
                                                                    # xyz_fetch,
                                                                    # dofs_fetch_list,
                                                                    # t_sampler,
                                                                    # s_sampler,
                                                                    # n_steps,
                                                                    # i_sub_it,
                                                                    # t_sub_it,
                                                                    # tF_apply_vec,
                                                                    # force_dof_apply_vec,
                                                                    # areaf_vec,
                                                                    # outputFolderPath)
                                # else:
                                    # tF_apply_vec = self.MUI_Parallel_FSI_Fetch( MPI_COMM_WORLD,
                                                                    # ifaces3d,
                                                                    # xyz_fetch,
                                                                    # dofs_fetch_list,
                                                                    # t_sampler,
                                                                    # s_sampler,
                                                                    # n_steps,
                                                                    # i_sub_it,
                                                                    # t_sub_it,
                                                                    # tF_apply_vec,
                                                                    # force_dof_apply_vec,
                                                                    # areaf_vec,
                                                                    # outputFolderPath)
                            # else:
                                # if self.iMUIFetchMoment:
                                    # tF_apply_vec, mom_x, mom_y, mom_z = \
                                        # self.MUI_Fetch( MPI_COMM_WORLD, 
                                                        # ifaces3d, 
                                                        # xyz_fetch, 
                                                        # dofs_fetch_list, 
                                                        # t_sampler, 
                                                        # s_sampler, 
                                                        # n_steps, 
                                                        # t_sub_it, 
                                                        # tF_apply_vec, 
                                                        # areaf_vec)
                                # else:
                                    # tF_apply_vec = self.MUI_Fetch(  MPI_COMM_WORLD, 
                                                                    # ifaces3d, 
                                                                    # xyz_fetch, 
                                                                    # dofs_fetch_list, 
                                                                    # t_sampler, 
                                                                    # s_sampler, 
                                                                    # n_steps, 
                                                                    # t_sub_it, 
                                                                    # tF_apply_vec, 
                                                                    # areaf_vec)
                            # # Finish the wall clock on fetch Time
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # fetchTime = wallClockFetchTime.toc()

                            # # Starts the wall clock
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # wallClockForceVecProj.tic()
                            # if self.iMUIFetchValue:
                                # # Apply traction components. These calls do parallel communication
                                # tF_apply.vector().set_local(tF_apply_vec)
                                # tF_apply.vector().apply("insert")
                            # else:
                                # # do not apply the fetched value, i.e. one-way coupling
                                # pass
                            # # Finish the wall clock on force vector project
                            # if (sync == True): MPI_COMM_WORLD.Barrier()
                            # forceVecProjTime = wallClockForceVecProj.toc()

                    # else:
                        # if rank == 0: print ("{FENICS} Assigning traction forces at present time step ...   ", 
                                                # end="", flush=True)

                        # temp_area_pernode = (self.YBeam*self.ZBeam)/len(dofs_fetch_list)
                        # print ("temp_area_pernode", temp_area_pernode)
                        # for i, p in enumerate(dofs_fetch_list):

                            # if t <= 7.0:
                                # tF_apply_vec[1::3][p] = ((t*(-(500.0/len(dofs_fetch_list)))/7.0)/temp_area_pernode)
                            # else:
                                # tF_apply_vec[1::3][p] = (0.0)

                            # tF_apply_vec[0::3][p] = 0.0
                            # tF_apply_vec[2::3][p] = 0.0
                            # print ("tF_apply_vec[1::3][p]", tF_apply_vec[1::3][p])
                        
                        # # Apply traction components. These calls do parallel communication
                        # tF_apply.vector().set_local(tF_apply_vec)
                        # tF_apply.vector().apply("insert")
                        # # tF_apply = project(dfst, V)
                        # # tF_apply.assign(project(dfst, 
                            # # V, 
                            # # solver_type=self.prjsolver,
                            # # form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                            # # "representation": self.compRepresentation}))
                        # # # Apply traction components. These calls do parallel communication
                        # # tF_apply.vector().set_local(tF_apply_vec)
                        # # tF_apply.vector().apply("insert")

                # else:
                    # if rank == 0: print ("{FENICS} Assigning traction forces at present time step ...   ", 
                                            # end="", flush=True)
                    # if t <= 7.0:
                        # tF_magnitude.assign(Constant((t*(-500.0)/7.0)/(self.YBeam*self.ZBeam)))
                    # else:
                        # tF_magnitude.assign(Constant((0.0)/(self.YBeam*self.ZBeam)))
                    # if rank == 0:
                        # print ("Done")
                        # #if self.iDebug: print ("{FENICS} tF_magnitude: ", tF_magnitude(0))

                # if (not ((self.iContinueRun) and (n_steps == 1))):
                    # if self.solving_method == 'MCK':
                        # # Starts the wall clock
                        # if (sync == True): MPI_COMM_WORLD.Barrier()
                        # wallClockLinearAssemble.tic()
                        # # Assemble linear form
                        # if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            # pass
                        # else:
                            # Linear_Assemble = assemble(Linear_Form)
                        # # Finish the wall clock on linear assemble
                        # if (sync == True): MPI_COMM_WORLD.Barrier()
                        # linearAssembleTime = wallClockLinearAssemble.toc()
                        # #bcs.apply(Linear_Assemble)
                        # #!!!!!->
                        # # Starts the wall clock
                        # if (sync == True): MPI_COMM_WORLD.Barrier()
                        # wallClockBCApply.tic()
                        # if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            # pass
                        # else:
                            # [bc.apply(Linear_Assemble) for bc in bcs]
                        # # Finish the wall clock on bc apply
                        # if (sync == True): MPI_COMM_WORLD.Barrier()
                        # bcApplyTime = wallClockBCApply.toc()
                        # #!!!!!<-
                    # # Solving the structure functions inside the time loop
                    # # Starts the wall clock
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # wallClockSolverSolve.tic()
                    # if (self.solving_method == 'MCK') and (self.linear_solver == 'LU'):
                        # solver.solve(Bilinear_Assemble, dmck.vector(), Linear_Assemble)
                    # else:
                        # if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            # pass
                        # else:
                            # solver.solve()
                    # # Finish the wall clock on solver solve
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # solverSolveTime = wallClockSolverSolve.toc()
                    # # Starts the wall clock
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # wallClockTotalForceCal.tic()
                    # if self.solving_method == 'MCK':
                        # force_X = dot(tF_apply, self.X_direction_vector())*ds(2)
                        # force_Y = dot(tF_apply, self.Y_direction_vector())*ds(2)
                        # force_Z = dot(tF_apply, self.Z_direction_vector())*ds(2)
                    # else:
                        # force_X = dot(tF, self.X_direction_vector())*ds(2)
                        # force_Y = dot(tF, self.Y_direction_vector())*ds(2)
                        # force_Z = dot(tF, self.Z_direction_vector())*ds(2)

                    # f_X_a = assemble(force_X)
                    # f_Y_a = assemble(force_Y)
                    # f_Z_a = assemble(force_Z)

                    # print ("{FENICS} Total Force_X on structure: ", f_X_a, " at rank ", rank)
                    # print ("{FENICS} Total Force_Y on structure: ", f_Y_a, " at rank ", rank)
                    # print ("{FENICS} Total Force_Z on structure: ", f_Z_a, " at rank ", rank)
                    # # Finish the wall clock on total force calculate
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # totalForceCalTime = wallClockTotalForceCal.toc()
                # else:
                    # pass

                # if self.solving_method == 'STVK':
                    # # Split function spaces
                    # u,d = ud.split(True)

                    # # Compute and print the displacement of monitored point
                    # self.print_Disp (MPI_COMM_WORLD, d)

                    # # MUI Push internal points and commit current steps
                    # if (self.iMUICoupling) and (len(xyz_push)!=0):
                        # self.MUI_Push(  MPI_COMM_WORLD, 
                                        # ifaces3d, 
                                        # xyz_push, 
                                        # dofs_push_list, 
                                        # d, 
                                        # t_sub_it)
                    # else:
                        # pass

                # elif self.solving_method == 'MCK':
                    # # Starts the wall clock
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # wallClockPrintDisp.tic()
                    # # Compute and print the displacement of monitored point
                    # self.print_Disp (MPI_COMM_WORLD, dmck)
                    # # Finish the wall clock on print disp
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # printDispTime = wallClockPrintDisp.toc()
                    # # Starts the wall clock
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # wallClockPush.tic()
                    # # MUI Push internal points and commit current steps
                    # if (self.iMUICoupling) and (len(xyz_push)!=0):
                        # self.MUI_Push(  MPI_COMM_WORLD, 
                                        # ifaces3d, 
                                        # xyz_push, 
                                        # dofs_push_list, 
                                        # dmck, 
                                        # t_sub_it)

                    # else:
                        # pass
                    # # Finish the wall clock on push
                    # if (sync == True): MPI_COMM_WORLD.Barrier()
                    # pushTime = wallClockPush.toc()
                # # Finish the wall clock on total sim time Per iter
                # simtimePerIter = wallClockPerIter.toc()

                # if self.iExporttxt: self.Time_Txt_Export(MPI_COMM_WORLD, 
                                                        # t,
                                                        # n_steps,
                                                        # i_sub_it,
                                                        # fetchTime,
                                                        # forceVecProjTime,
                                                        # linearAssembleTime,
                                                        # bcApplyTime,
                                                        # solverSolveTime,
                                                        # totalForceCalTime,
                                                        # printDispTime,
                                                        # pushTime,
                                                        # moveMeshTime,
                                                        # dispVTKExpTime,
                                                        # dispTxtExpTime,
                                                        # checkpointExpTime,
                                                        # assignOldFuncSpaceTime,
                                                        # simtimePerIter,
                                                        # simtimePerStep,
                                                        # outputFolderPath)

                # # Move to the next sub-iteration
                # i_sub_it += 1

            # if self.solving_method == 'STVK':
                # # Split function spaces
                # u,d = ud.split(True)
                # u0,d0 = u0d0.split(True)

                # self.Move_Mesh(V, d, d0, mesh)

                # if (not (self.iQuiet)):
                    # self.Export_Disp_vtk(   MPI_COMM_WORLD, 
                                            # n_steps, 
                                            # t, 
                                            # mesh, 
                                            # gdim, 
                                            # V, 
                                            # tF, 
                                            # d, 
                                            # stress_file, 
                                            # disp_file, 
                                            # traction_file)

                    # self.Export_Disp_txt(   MPI_COMM_WORLD, 
                                            # d, 
                                            # outputFolderPath)

                    # self.Checkpoint_Output( MPI_COMM_WORLD, 
                                            # outputFolderPath, 
                                            # t, 
                                            # mesh, 
                                            # meshOri, 
                                            # u0d0, 
                                            # d0mck, 
                                            # u0mck, 
                                            # a0mck, 
                                            # ud, 
                                            # dmck, 
                                            # sigma_s, 
                                            # areaf, 
                                            # True)

            # elif self.solving_method == 'MCK':
                # # Starts the wall clock
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # wallClockMoveMesh.tic()
                # self.Move_Mesh(V, dmck, d0mck, mesh)
                # # Finish the wall clock on move mesh
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # moveMeshTime = wallClockMoveMesh.toc()
                # # Starts the wall clock
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # wallClockDispVTKExp.tic()
                # if (not (self.iQuiet)):
                    # self.Export_Disp_vtk(   MPI_COMM_WORLD, 
                                            # n_steps, 
                                            # t, 
                                            # mesh, 
                                            # gdim, 
                                            # V, 
                                            # tF_apply, 
                                            # dmck, 
                                            # stress_file, 
                                            # disp_file, 
                                            # traction_file)
                # # Finish the wall clock on disp VTK export
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # dispVTKExpTime = wallClockDispVTKExp.toc()
                # # Starts the wall clock
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # wallClockDispTxtExp.tic()
                # if (not (self.iQuiet)):
                    # self.Export_Disp_txt(   MPI_COMM_WORLD, 
                                            # dmck, 
                                            # outputFolderPath)
                # # Finish the wall clock on disp txt export
                # if (sync == True): MPI_COMM_WORLD.Barrier()
                # dispTxtExpTime = wallClockDispTxtExp.toc()
            # # Starts the wall clock
            # if (sync == True): MPI_COMM_WORLD.Barrier()
            # wallClockCheckpointExp.tic()
            # if (not (self.iQuiet)):
                # self.Checkpoint_Output( MPI_COMM_WORLD, 
                                        # outputFolderPath, 
                                        # t, 
                                        # mesh, 
                                        # meshOri, 
                                        # u0d0, 
                                        # d0mck, 
                                        # u0mck, 
                                        # a0mck, 
                                        # ud, 
                                        # dmck, 
                                        # sigma_s, 
                                        # areaf, 
                                        # True)
            # # Finish the wall clock on checkpoint export
            # if (sync == True): MPI_COMM_WORLD.Barrier()
            # checkpointExpTime = wallClockCheckpointExp.toc()
            # # Starts the wall clock
            # if (sync == True): MPI_COMM_WORLD.Barrier()
            # wallClockAssignOldFuncSpace.tic()
            # # Assign the old function spaces
            # if self.solving_method == 'STVK':
                # u0d0.assign(ud)

            # elif self.solving_method == 'MCK':
                # if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                    # pass
                # else:
                    # amck = self.AMCK (  dmck.vector(),
                                        # d0mck.vector(),
                                        # u0mck.vector(),
                                        # a0mck.vector(),
                                        # beta_gam)

                    # umck = self.UMCK (  amck,
                                        # u0mck.vector(),
                                        # a0mck.vector(),
                                        # gamma_gam)

                    # a0mck.vector()[:] = amck
                    # u0mck.vector()[:] = umck
                    # d0mck.vector()[:] = dmck.vector()
            # # Finish the wall clock on assign old function space
            # if (sync == True): MPI_COMM_WORLD.Barrier()
            # assignOldFuncSpaceTime = wallClockAssignOldFuncSpace.toc()
            # # Move to next time step
            # i_sub_it = 1
            # t += self.dt
            # # Finish the wall clock
            # simtimePerStep = wallClockPerStep.toc()
            # if self.rank == 0:
                # print ("\n")
                # print ("{FENICS} Simulation time per step: %g [s] at timestep: %i" % (simtimePerStep, n_steps))
            
        # #===========================================
        # #%% Calculate wall time
        # #===========================================

        # # Wait for the other solver
        # ifaces3d["threeDInterface0"].barrier(t_sub_it)

        # # Finish the wall clock
        # simtime = wallClock.toc()

        # self.Post_Solving_Log(MPI_COMM_WORLD, simtime)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#