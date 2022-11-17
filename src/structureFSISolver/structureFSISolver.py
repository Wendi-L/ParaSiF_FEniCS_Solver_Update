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
import structureFSISolver.solvers.hyperElasticSolver

#_________________________________________________________________________________________
#
#%% Main Structure Solver Class
#_________________________________________________________________________________________

class StructureFSISolver(structureFSISolver.cfgPrsFn.readData,
                         structureFSISolver.lameParm.lameParm,
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
    #%% Initialize MPI by mpi4py/MUI for parallelized computation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def MUI_Init(self):
        if self.iMUICoupling():
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

    def MUI_Sampler_Define(self,
                           function_space,
                           grid_dimension,
                           dofs_fetch_list,
                           dofs_push_list,
                           xyz_fetch,
                           Total_Time_Steps):

        if self.iMUICoupling():
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
                self.ifaces3d["threeDInterface0"].announce_send_span(0, Total_Time_Steps*self.num_sub_iteration(), span_push, synchronised)

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
                self.ifaces3d["threeDInterface0"].announce_recv_span(0, Total_Time_Steps*self.num_sub_iteration()*10, span_fetch, synchronised)

                print("{FENICS} at rank: ", self.rank, " recv_max_X: ", recv_max_X, " recv_min_X: ", recv_min_X)
                print("{FENICS} at rank: ", self.rank, " recv_max_Y: ", recv_max_Y, " recv_min_Y: ", recv_min_Y)
                print("{FENICS} at rank: ", self.rank, " recv_max_Z: ", recv_max_Z, " recv_min_Z: ", recv_min_Z)

            else:
                # Announce the MUI receive span
                #self.ifaces3d["threeDInterface0"].announce_recv_disable()
                pass

            # Spatial/temporal samplers
            if self.rank == 0: print ("{FENICS} Defining MUI samplers ...   ", end="", flush=True)

            fileAddress=self.outputFolderName() + '/RBFMatrix/' + str(self.rank)
            os.makedirs(fileAddress)

            if (self.iReadMatrix()):
                print ("{FENICS} Reading RBF matrix from ", self.rank)
                sourcefileAddress=self.inputFolderName() + '/RBFMatrix'

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
                    with open(self.outputFolderName() + '/RBFMatrix'+'/partitionSize.dat', 'w') as f_ps:
                        f_ps.write("%i\n" % self.size)

            # Best practice suggestion: for a better performance on the RBF method, always switch on the smoothFunc when structure Dofs are more than
            #                           fluid points; Tune the rMUIFetcher to receive a reasonable totForce_Fetch value; Tune the areaListFactor to
            #                           ensure totForce_Fetch and Total_Force_on_structure are the same.
            self.t_sampler = mui4py.ChronoSamplerExact()

            self.s_sampler = mui4py.SamplerRbf(self.rMUIFetcher(),
                                               point3dList,
                                               self.basisFunc(),
                                               self.iConservative(),
                                               self.iPolynomial(),
                                               self.iSmoothFunc(),
                                               self.iReadMatrix(),
                                               fileAddress,
                                               self.cutoffRBF())

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
        if self.iContinueRun():
            # Restart simulation
            if self.iMeshLoad():
                # Load mesh from file
                if self.iLoadXML():
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
                mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(self.OBeamX(), self.OBeamY(), self.OBeamZ()),
                       Point((self.OBeamX()+self.XBeam()), (self.OBeamY()+self.YBeam()), (self.OBeamZ()+self.ZBeam())),
                       self.XMesh(), self.YMesh(), self.ZMesh())
                if self.rank == 0: print ("{FENICS} Done with generating mesh")
        else:
            # Simulation from zero
            if self.iMeshLoad():
                # Load mesh from file
                if self.iLoadXML():
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
                mesh = BoxMesh(self.LOCAL_COMM_WORLD, Point(self.OBeamX(), self.OBeamY(), self.OBeamZ()),
                       Point((self.OBeamX()+self.XBeam()), (self.OBeamY()+self.YBeam()), (self.OBeamZ()+self.ZBeam())),
                       self.XMesh(), self.YMesh(), self.ZMesh())
                if self.rank == 0: print ("{FENICS} Done with generating mesh")

        if self.iHDF5FileExport() and self.iHDF5MeshExport():
            if self.rank == 0: print ("{FENICS} Exporting HDF5 mesh ...   ", end="", flush=True)
            hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "w")
            hdfOutTemp.write(mesh, "/mesh")
            hdfOutTemp.close()
            del hdfOutTemp
            if self.rank == 0: print ("Done")

        if self.iInteractiveMeshShow():
            if self.rank == 0: print ("{FENICS} Interactive Mesh Show ...", end="", flush=True)
            import matplotlib.pyplot as plt
            plt.figure()
            p = plot(mesh, title = "Mesh plot")
            plt.show()
            if self.rank == 0: print ("Done")

        return mesh

    def Get_Grid_Dimension(self, mesh):
        # Geometry dimensions
        grid_dimension = mesh.geometry().dim()
        return grid_dimension

    def Get_Face_Narmal(self, mesh):
        # Face normal vector
        face_narmal = FacetNormal(mesh)
        return face_narmal

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define SubDomains and boundaries
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Boundaries_Generation_Fixed_Flex_Sym (self, mesh, grid_dimension, VectorFunctionSpace):

        #Define SubDomains
        if self.iMeshLoad() and self.iSubdomainsImport():
            if self.iLoadXML():
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

        if self.iHDF5FileExport() and self.iHDF5SubdomainsExport():
            if self.rank == 0: print ("{FENICS} Exporting HDF5 subdomains ...   ", end="", flush=True) 
            self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
            hdfOutTemp.write(self.subdomains, "/subdomains")
            hdfOutTemp.close()
            del hdfOutTemp
            if self.rank == 0: print ("Done")

        #Define and mark mesh boundaries
        if self.iMeshLoad() and self.iBoundariesImport():
            if self.iLoadXML():
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

        if self.iHDF5FileExport() and self.iHDF5BoundariesExport():
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
        if self.iContinueRun():
            hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
            # Read start time [s]
            self.Start_Time = self.dt() + hdf5checkpointDataInTemp.attributes("/ud/vector_0")["timestamp"]
            # Calculate time steps [-]
            self.Time_Steps = math.ceil((self.T() - self.Start_Time)/self.dt())
            # Close file and delete HDF5File object
            hdf5checkpointDataInTemp.close()
            del hdf5checkpointDataInTemp
        else:
            if self.iResetStartTime():
                # Reset start time [s]
                self.Start_Time = self.dt() + self.newStartTime()
                # Calculate time steps [-]
                self.Time_Steps = math.ceil((self.T() - self.Start_Time)/self.dt())
            else:
                # Set start time [s]
                self.Start_Time = self.dt()
                # Calculate time steps [-]
                self.Time_Steps = math.ceil(self.T()/self.dt())
        # Initialise sub-iterations counter
        self.Start_Number_Sub_Iteration = 1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define DOFs extract function
    #%% and
    #%% DOFs-Coordinates mapping function
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_subdomain_dofs(self, MeshFunction, VectorFunctionSpace, boundary):
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

    def dofs_list(self, MeshFunction, FunctionSpace, boundary):
        return list(self.get_subdomain_dofs(MeshFunction, FunctionSpace, boundary))

    def xyz_np(self, dofs_list, FunctionSpace, dimension):
        xyz_np = np.zeros((len(dofs_list), dimension))
        for i, p in enumerate(dofs_list):
            xyz_np[i] = self.dofs_to_xyz(FunctionSpace, dimension)[p]
        return xyz_np

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define facet areas
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def facets_area_list_calculation(self,
                                     mesh,
                                     FunctionSpace,
                                     boundary,
                                     dofs_fetch_list,
                                     dimension):
        areatotal = 0.0
        areatotal_local = 0.0
        cell2dofs = FunctionSpace.dofmap().cell_dofs
        ones = np.array([1,1,1])

        dpc_help_number = 0
        if (self.deg_fun_spc() != 1):
            for i in range(2, self.deg_fun_spc()+1):
                dpc_help_number += i
        dofs_Per_Cell=3+dpc_help_number+(self.deg_fun_spc()-1)

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
                area = 0.5*math.sqrt(det1*det1 + det2*det2 + det3*det3)*self.areaListFactor()
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
                            # self.areaf_vec[pp] += area/d_num
                            self.areaf_vec[pp] += area/dofs_Per_Cell

        for iii, ppp in enumerate(self.areaf_vec):
            areatotal += self.areaf_vec[iii]

        if (self.rank == 0) and self.iDebug():
            print("Total area of MUI fetched surface= ", areatotal, " m^2")

    def facets_area_define(self,
                           mesh,
                           Q,
                           boundaries,
                           dofs_fetch_list,
                           gdim):
            # Define function for facet area
            self.areaf= Function(Q)
            self.areaf_vec = self.areaf.vector().get_local()

            if self.iLoadAreaList():
                hdf5meshAreaDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                hdf5meshAreaDataInTemp.read(self.areaf, "/areaf/vector_0")
                hdf5meshAreaDataInTemp.close()
            else:
                if self.rank == 0: print ("{FENICS} facet area calculating")
                # Calculate function for facet area
                self.facets_area_list_calculation(mesh, Q, boundaries, dofs_fetch_list, gdim)
                # Apply the facet area vectors
                self.areaf.vector().set_local(self.areaf_vec)
                self.areaf.vector().apply("insert")
                # Facet area vectors I/O
                if (self.iHDF5FileExport()) and (self.iHDF5MeshExport()):
                    hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                    hdfOutTemp.write(self.areaf, "/areaf")
                    hdfOutTemp.close()
                else:
                    pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define MUI Fetch and Push
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def MUI_Fetch(self, dofs_to_xyz, dofs_fetch_list, total_Sub_Iteration):
        totForceX = 0.0
        totForceY = 0.0
        totForceZ = 0.0
        temp_vec_function_temp = self.tF_apply_vec

        if self.iparallelFSICoupling():
            fetch_iteration = total_Sub_Iteration-1
        else:
            fetch_iteration = total_Sub_Iteration

        if (fetch_iteration >= 0):
            if self.iMUIFetchMany():
                temp_vec_function_temp[0::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceX",
                                       dofs_to_xyz,
                                       fetch_iteration,
                                       self.s_sampler,
                                       self.t_sampler)
                temp_vec_function_temp[1::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceY",
                                       dofs_to_xyz,
                                       fetch_iteration,
                                       self.s_sampler,
                                       self.t_sampler)
                temp_vec_function_temp[2::3][dofs_fetch_list] = self.ifaces3d["threeDInterface0"].\
                            fetch_many("forceZ",
                                       dofs_to_xyz,
                                       fetch_iteration,
                                       self.s_sampler,
                                       self.t_sampler)

                for i, p in enumerate(dofs_fetch_list):
                    if self.iparallelFSICoupling():
                        self.tF_apply_vec[0::3][p] += (temp_vec_function_temp[0::3][p] - \
                                                       self.tF_apply_vec[0::3][p])*self.undRelxCpl()
                        self.tF_apply_vec[1::3][p] += (temp_vec_function_temp[1::3][p] - \
                                                       self.tF_apply_vec[1::3][p])*self.undRelxCpl()
                        self.tF_apply_vec[2::3][p] += (temp_vec_function_temp[2::3][p] - \
                                                       self.tF_apply_vec[2::3][p])*self.undRelxCpl()
                    else:
                        self.tF_apply_vec[0::3][p] = temp_vec_function_temp[0::3][p]
                        self.tF_apply_vec[1::3][p] = temp_vec_function_temp[1::3][p]
                        self.tF_apply_vec[2::3][p] = temp_vec_function_temp[2::3][p]

                    totForceX += self.tF_apply_vec[0::3][p]
                    totForceY += self.tF_apply_vec[1::3][p]
                    totForceZ += self.tF_apply_vec[2::3][p]

                    if (self.areaf_vec[p] == 0):
                        self.tF_apply_vec[0::3][p] = 0.
                        self.tF_apply_vec[1::3][p] = 0.
                        self.tF_apply_vec[2::3][p] = 0.
                    else:
                        self.tF_apply_vec[0::3][p] /= self.areaf_vec[p]
                        self.tF_apply_vec[1::3][p] /= self.areaf_vec[p]
                        self.tF_apply_vec[2::3][p] /= self.areaf_vec[p]

            else:
                if (fetch_iteration >= 0):
                    for i, p in enumerate(dofs_fetch_list):
                        temp_vec_function_temp[0::3][p] = self.ifaces3d["threeDInterface0"].\
                                    fetch("forceX",
                                          dofs_to_xyz[i],
                                          fetch_iteration,
                                          self.s_sampler,
                                          self.t_sampler)

                        temp_vec_function_temp[1::3][p] = self.ifaces3d["threeDInterface0"].\
                                    fetch("forceY",
                                          dofs_to_xyz[i],
                                          fetch_iteration,
                                          self.s_sampler,
                                          self.t_sampler)

                        temp_vec_function_temp[2::3][p] = self.ifaces3d["threeDInterface0"].\
                                    fetch("forceZ",
                                          dofs_to_xyz[i],
                                          fetch_iteration,
                                          self.s_sampler,
                                          self.t_sampler)

                        if self.iparallelFSICoupling():
                            self.tF_apply_vec[0::3][p] += (temp_vec_function_temp[0::3][p] - \
                                                           self.tF_apply_vec[0::3][p])*self.undRelxCpl()
                            self.tF_apply_vec[1::3][p] += (temp_vec_function_temp[1::3][p] - \
                                                           self.tF_apply_vec[1::3][p])*self.undRelxCpl()
                            self.tF_apply_vec[2::3][p] += (temp_vec_function_temp[2::3][p] - \
                                                           self.tF_apply_vec[2::3][p])*self.undRelxCpl()
                        else:
                            self.tF_apply_vec[0::3][p] = temp_vec_function_temp[0::3][p]
                            self.tF_apply_vec[1::3][p] = temp_vec_function_temp[1::3][p]
                            self.tF_apply_vec[2::3][p] = temp_vec_function_temp[2::3][p]

                        totForceX += self.tF_apply_vec[0::3][p]
                        totForceY += self.tF_apply_vec[1::3][p]
                        totForceZ += self.tF_apply_vec[2::3][p]

                        self.tF_apply_vec[0::3][p] /= self.areaf_vec[p]
                        self.tF_apply_vec[1::3][p] /= self.areaf_vec[p]
                        self.tF_apply_vec[2::3][p] /= self.areaf_vec[p]

                    if self.iDebug():
                        print ("{FENICS**} totForce Apply: ", totForceX, "; ",totForceY, "; ",totForceZ,
                                "; at iteration: ", fetch_iteration, " at rank: ", self.rank)

    def MUI_Push(self, dofs_to_xyz, dofs_push, displacement_function, total_Sub_Iteration):
        d_vec_x = displacement_function.vector().get_local()[0::3]
        d_vec_y = displacement_function.vector().get_local()[1::3]
        d_vec_z = displacement_function.vector().get_local()[2::3]

        if self.iMUIPushMany():
            if self.iPushX():
                self.ifaces3d["threeDInterface0"].\
                            push_many("dispX", dofs_to_xyz, (d_vec_x[dofs_push]))
            if self.iPushY():
                self.ifaces3d["threeDInterface0"].\
                            push_many("dispY", dofs_to_xyz, (d_vec_y[dofs_push]))
            if self.iPushZ():
                self.ifaces3d["threeDInterface0"].\
                            push_many("dispZ", dofs_to_xyz, (d_vec_z[dofs_push]))

            a = self.ifaces3d["threeDInterface0"].\
                            commit(total_Sub_Iteration)
        else:
            if self.iPushX():
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].\
                            push("dispX", dofs_to_xyz[i], (d_vec_x[p]))
            if self.iPushY():
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].\
                            push("dispY", dofs_to_xyz[i], (d_vec_y[p]))
            if self.iPushZ():
                for i, p in enumerate(dofs_push):
                    self.ifaces3d["threeDInterface0"].\
                            push("dispZ", dofs_to_xyz[i], (d_vec_z[p]))

            a = self.ifaces3d["threeDInterface0"].\
                            commit(total_Sub_Iteration)

        if (self.rank == 0) and self.iDebug():
            print ('{FENICS} MUI commit step: ',total_Sub_Iteration)

        if ((total_Sub_Iteration-self.forgetTStepsMUI()) > 0):
            a = self.ifaces3d["threeDInterface0"].\
                            forget(total_Sub_Iteration-self.forgetTStepsMUI())
            self.ifaces3d["threeDInterface0"].\
                            set_memory(self.forgetTStepsMUI())
            if (self.rank == 0) and self.iDebug():
                print ('{FENICS} MUI forget step: ',(total_Sub_Iteration-self.forgetTStepsMUI()))

    def MUI_Commit_only(self, total_Sub_Iteration):
        a = self.ifaces3d["threeDInterface0"].\
                            commit(total_Sub_Iteration)

        if (self.rank == 0) and self.iDebug():
            print ('{FENICS} MUI commit step: ',total_Sub_Iteration)

        if ((total_Sub_Iteration-self.forgetTStepsMUI()) > 0):
            a = self.ifaces3d["threeDInterface0"].\
                            forget(total_Sub_Iteration-self.forgetTStepsMUI())
            self.ifaces3d["threeDInterface0"].\
                            set_memory(self.forgetTStepsMUI())
            if (self.rank == 0) and self.iDebug():
                print ('{FENICS} MUI forget step: ',(total_Sub_Iteration-self.forgetTStepsMUI()))

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
        # Body external forces define [N/m^3]
        b_for_ext = Constant((self.bForExtX(), self.bForExtY(), self.bForExtZ()))
        # Gravitational force define [N/m^3]
        if self.iGravForce():
            g_force = Constant((0.0, (self.rho_s() * (-9.81)), 0.0))
        else:
            g_force = Constant((0.0, (0.0 * (-9.81)), 0.0))
        return (b_for_ext + g_force)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Lame parameters
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Define the Lamé's first parameter
    def lamda_s (self):
        return (2.0*(self.mu_s())*self.nu_s()/(1.0-2.0*self.nu_s()))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Generalised-alpha method functions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Define the acceleration at the present time step
    def Acceleration_March_Term_One(self,
                                    displacement_function,
                                    displacement_previous_function,
                                    velocity_previous_function):
        return (2 * (displacement_function - displacement_previous_function -
                     (self.dt() * velocity_previous_function))/
                     (self.dt()**2))

    def Acceleration_March_Term_Two(self,
                                    acceleration_previous_function,
                                    beta_gam):
        return ((1 - (2 * beta_gam)) * acceleration_previous_function)

    def Acceleration_March_Term_Three(self, beta_gam):
        return (1 / (2 * beta_gam))

    def AMCK(self,
             displacement_function,
             displacement_previous_function,
             velocity_previous_function,
             acceleration_previous_function,
             beta_gam):
        return (self.Acceleration_March_Term_Three(beta_gam) *
                    (self.Acceleration_March_Term_One(displacement_function,
                    displacement_previous_function,
                    velocity_previous_function) -
                    self.Acceleration_March_Term_Two(acceleration_previous_function,beta_gam)))

    # Define the velocity at the present time step
    def Velocity_March_Term_One(self,
                                acceleration_previous_function,
                                gamma_gam):
        return ((1 - gamma_gam) * acceleration_previous_function * self.dt())

    def Velocity_March_Term_Two(self,
                                acceleration_function,
                                gamma_gam):
        return (acceleration_function * gamma_gam * self.dt())

    def UMCK(self,
             acceleration_function,
             velocity_previous_function,
             acceleration_previous_function,
             gamma_gam):
        return (self.Velocity_March_Term_One(acceleration_previous_function, gamma_gam) +
                self.Velocity_March_Term_Two(acceleration_function, gamma_gam) +
                velocity_previous_function)

    # define the calculation of intermediate averages based on generalized alpha method
    def Generalized_Alpha_Weights(self,
                                  present_function,
                                  previous_function,
                                  weights):
        return (weights * previous_function + (1 - weights) * present_function)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Stress, force gradient and its
    #%% determination functions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def I(self, grid_dimension):
        # Define the Identity matrix
        return (Identity(grid_dimension))

    def F_(self, displacement_function, grid_dimension):
        # Define the deformation gradient
        return (self.I(grid_dimension) + nabla_grad(displacement_function))

    def J_(self, displacement_function, grid_dimension):
        # Define the determinant of the deformation gradient
        return det(self.F_(displacement_function,grid_dimension))

    def C(self, displacement_function, grid_dimension):
        # Define the right Cauchy-Green strain tensor
        return ((self.F_(displacement_function, grid_dimension).T) *
                self.F_(displacement_function, grid_dimension))

    def E(self, displacement_function, grid_dimension):
        # Define the non-linear Lagrangian Green strain tensor
        return (0.5 * (self.C(displacement_function, grid_dimension) - self.I(grid_dimension)))

    def epsilon(self, displacement_function, grid_dimension):
        # Define the linear Lagrangian Green strain tensor
        return (0.5 * (nabla_grad(displacement_function) + (nabla_grad(displacement_function).T)))

    def Piola_Kirchhoff_sec(self, displacement_function, strain_tensor, grid_dimension):
        # Define the Second Piola-Kirchhoff stress tensor by the constitutive law
        #   of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation).
        #   Valid for large deformations but small strain.
        return (self.lamda_s() * tr(strain_tensor(displacement_function, grid_dimension)) *
                self.I(grid_dimension) + 2.0 * self.mu_s() *
                strain_tensor(displacement_function, grid_dimension))

    def cauchy_stress(self, displacement_function, strain_tensor, grid_dimension):
        # Define the Cauchy stress tensor
        return ((1 / self.J_(displacement_function, grid_dimension)) *
                (self.F_(displacement_function, grid_dimension)) *
                (self.Piola_Kirchhoff_sec(displacement_function, strain_tensor,grid_dimension)) *
                (self.F_(displacement_function, grid_dimension).T))

    def Piola_Kirchhoff_fst(self, displacement_function, grid_dimension):
        # Define the First Piola-Kirchhoff stress tensor by the constitutive law
        #   of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation).
        #   Valid for large deformations but small strain.
        return (self.J_(displacement_function, grid_dimension) *
                self.cauchy_stress(displacement_function, self.E,grid_dimension) *
                inv(self.F_(displacement_function, grid_dimension).T))

    def Hooke_stress(self, displacement_function, grid_dimension):
        # Define the First Piola-Kirchhoff stress tensor by Hooke's law (linear relation).
        #   Valid for small-scale deformations only.
        return (self.J_(displacement_function, grid_dimension) *
                self.cauchy_stress(displacement_function, self.epsilon, grid_dimension) *
                inv(self.F_(displacement_function, grid_dimension).T))

    def elastic_stress(self, displacement_function, grid_dimension):
        # Define the elastic stress tensor
        return (2.0 * self.mu_s() * sym(grad(displacement_function)) +
                self.lamda_s() * tr(sym(grad(displacement_function))) * self.I(grid_dimension))

    def Traction_Define(self, VectorFunctionSpace):
        if self.iNonUniTraction():
            if self.rank == 0: print ("{FENICS} Non-uniform traction applied")
            self.tF_apply = Function(VectorFunctionSpace)
            self.tF_apply_vec = self.tF_apply.vector().get_local()
        else:
            if self.rank == 0: print ("{FENICS} Uniform traction applied")
            self.tF_magnitude = Constant(0.0 *self.X_direction_vector() +
                                    0.0 *self.Y_direction_vector() +
                                    0.0 *self.Z_direction_vector())
            self.tF_apply = self.tF_magnitude

    def Traction_Assign(self, xyz_fetch, dofs_fetch_list, t_sub_it, n_steps):
        # Assign traction forces at present time step
        if self.iNonUniTraction():
            if len(xyz_fetch)!=0:
                # Execute only when there are DoFs need to exchange data in this rank.
                self.MUI_Fetch(xyz_fetch, dofs_fetch_list, t_sub_it)
            if (self.iMUIFetchValue()) and (not ((self.iContinueRun()) and (n_steps == 1))):
                # Apply traction components. These calls do parallel communication
                self.tF_apply.vector().set_local(self.tF_apply_vec)
                self.tF_apply.vector().apply("insert")
            else:
                # Do not apply the fetched value, i.e. one-way coupling
                pass
        else:
            if self.rank == 0: print ("{FENICS} Assigning uniform traction forces at present time step ...   ",
                                    end="", flush=True)
            if (t <= self.sForExtEndTime()):
                self.tF_magnitude.assign((Constant((self.sForExtX()) /
                                                   (self.YBeam() * self.ZBeam())) *
                                                   self.X_direction_vector()) +
                                                   (Constant((self.sForExtY()) /
                                                   (self.XBeam()*self.ZBeam())) *
                                                   self.Y_direction_vector()) +
                                                   (Constant((self.sForExtZ()) /
                                                   (self.XBeam()*self.YBeam())) *
                                                   self.Z_direction_vector()))
            else:
                self.tF_magnitude.assign(Constant((0.0)))
            if self.rank == 0:
                print ("Done")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Set form compiler options
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Set_Compiler_Options(self):
        parameters["allow_extrapolation"] = self.allow_extrapolation()
        parameters["form_compiler"]["optimize"] = self.optimize()
        parameters["form_compiler"]["cpp_optimize"] = self.cppOptimize()
        parameters["form_compiler"]["representation"] = self.compRepresentation()
        parameters["ghost_mode"] = self.ghost_mode()
        parameters["mesh_partitioner"] = "SCOTCH"
        parameters["partitioning_approach"] = "PARTITION"
        info(parameters, False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% ALE Move Mesh
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def Move_Mesh(self,
                  VectorFunctionSpace,
                  displacement_function,
                  displacement_function_previous,
                  mesh):
        dOffset = Function(VectorFunctionSpace)
        # Calculate offset of the displacement
        dOffset.vector()[:] = displacement_function.vector().get_local() - \
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

    def Pre_Solving_Log(self):
        if self.rank == 0:
            print ("\n")
            print ("{FENICS} ********** STRUCTURAL-ELASTICITY SIMULATION BEGIN **********")
            print ("\n")
            if self.iDebug():
                print ("{FENICS} ### !!! DEBUG LEVEL ON !!! ###")
                print ("\n")

            if self.iMUICoupling():
                print ("{FENICS} ### !!! MUI COUPLING ON !!! ###")
                print ("\n")

            print ("{FENICS} Current Date and Time: ", datetime.datetime.now())
            print ("{FENICS} System Host Name: ", socket.gethostbyaddr(socket.gethostname())[0])
            print ("\n")

            print ("{FENICS} Solver info: ")
            if self.solving_method() == 'STVK':
                print ("{FENICS} Solver for the problem: ", self.prbsolver())
                print ("{FENICS} Solver for project between domains: ", self.prjsolver())
                print ("{FENICS} Pre-conditioner for the problem: ", self.prbpreconditioner())
                print ("{FENICS} Relative tolerance: ", self.prbRelative_tolerance())
                print ("{FENICS} Absolute tolerance: ", self.prbAbsolute_tolerance())
                print ("{FENICS} Maximum iterations: ", self.prbMaximum_iterations())
                print ("{FENICS} Relaxation parameter: ", self.prbRelaxation_parameter())
                print ("{FENICS} Representation of the compiler: ", self.compRepresentation())
                print ("{FENICS} C++ code optimization: ", self.cppOptimize())
                print ("{FENICS} optimization of the compiler: ", self.optimize())
                print ("{FENICS} Extrapolation: ", self.allow_extrapolation())
                print ("{FENICS} Ghost cell mode: ", self.ghost_mode())
                print ("{FENICS} Error of non convergence: ", self.error_on_nonconvergence())
            elif self.solving_method() == 'MCK':
                print ("{FENICS} Solver for the problem: ", self.prbsolver())
                print ("{FENICS} Solver for project between domains: ", self.prjsolver())
            print ("\n")

            print ("{FENICS} Input parameters: ")
            print ("{FENICS} E: ", self.E_s(), "[Pa]")
            print ("{FENICS} rho: ", self.rho_s(), "[kg/m^3]")
            print ("{FENICS} nu: ", self.nu_s(), "[-]")
            print ("\n")
        else:
            pass

    def Time_Marching_Log(self):
        if self.rank == 0: 
            print ("\n")
            print ("{FENICS} Total time: ", self.T(), " [s]")
            print ("{FENICS} Time step size: ", self.dt(), " [s]")
            print ("{FENICS} Time steps: ", self.Start_Time, " [-]")
            print ("{FENICS} Start time: ", self.Time_Steps, " [s]")
            print ("{FENICS} Numbers of sub-iterations: ", self.num_sub_iteration(), " [-]")
            print ("\n")

    def print_Disp (self, displacement_function):
        # Compute and print the displacement of monitored point
        d_DispSum = np.zeros(3)
        d_tempDenominator  = np.array([ self.size,
                                        self.size,
                                        self.size])
        self.LOCAL_COMM_WORLD.Reduce((displacement_function(
                                    Point(self.pointMoniX(),self.pointMoniY(),self.pointMoniZ()))),
                                    d_DispSum,op=MPI.SUM,root=0)
        d_Disp = np.divide(d_DispSum,d_tempDenominator)
        if self.rank == 0:
            print ("{FENICS} Monitored point deflection [m]: ", d_Disp)

    def Export_Disp_txt(self, displacement_function):
        if self.iExporttxt():
            pointMoniDispSum = np.zeros(3)
            tempDenominator  = np.array([self.size,
                                         self.size,
                                         self.size])
            self.LOCAL_COMM_WORLD.Reduce((displacement_function(
                                    Point(self.pointMoniX(),self.pointMoniY(),self.pointMoniZ()))),
                                    pointMoniDispSum,op=MPI.SUM,root=0)
            pointMoniDisp = np.divide(pointMoniDispSum,tempDenominator)

            pointMoniDispSum_b = np.zeros(3)
            tempDenominator_b  = np.array([self.size,
                                         self.size,
                                         self.size])
            self.LOCAL_COMM_WORLD.Reduce((displacement_function(
                                    Point(self.pointMoniXb(),self.pointMoniYb(),self.pointMoniZb()))),
                                    pointMoniDispSum_b,op=MPI.SUM,root=0)
            pointMoniDisp_b = np.divide(pointMoniDispSum_b,tempDenominator_b)

            for irank in range(self.size):
                if self.rank == irank:
                    ftxt_dispX = open(self.outputFolderPath + "/tip-displacementX_" + str(irank)+ ".txt", "a")
                    ftxt_dispX.write(str(pointMoniDisp[0]))
                    ftxt_dispX.write("\n")
                    ftxt_dispX.close

                    ftxt_dispY = open(self.outputFolderPath + "/tip-displacementY_" + str(irank)+ ".txt", "a")
                    ftxt_dispY.write(str(pointMoniDisp[1]))
                    ftxt_dispY.write("\n")
                    ftxt_dispY.close

                    ftxt_dispZ = open(self.outputFolderPath + "/tip-displacementZ_" + str(irank)+ ".txt", "a")
                    ftxt_dispZ.write(str(pointMoniDisp[2]))
                    ftxt_dispZ.write("\n")
                    ftxt_dispZ.close

                    ftxt_dispXb = open(self.outputFolderPath + "/tip-displacementXb_" + str(irank)+ ".txt", "a")
                    ftxt_dispXb.write(str(pointMoniDisp_b[0]))
                    ftxt_dispXb.write("\n")
                    ftxt_dispXb.close

                    ftxt_dispYb = open(self.outputFolderPath + "/tip-displacementYb_" + str(irank)+ ".txt", "a")
                    ftxt_dispYb.write(str(pointMoniDisp_b[1]))
                    ftxt_dispYb.write("\n")
                    ftxt_dispYb.close

                    ftxt_dispZb = open(self.outputFolderPath + "/tip-displacementZb_" + str(irank)+ ".txt", "a")
                    ftxt_dispZb.write(str(pointMoniDisp_b[2]))
                    ftxt_dispZb.write("\n")
                    ftxt_dispZb.close

    def Export_Disp_vtk(self,
                        Current_Time_Step,
                        current_time,
                        mesh,
                        grid_dimension,
                        VectorFunctionSpace,
                        displacement_function):
        # Export post-processing files
        if ((self.rank == 0) and self.iDebug()):
            print ("\n")
            print ("{FENICS} time steps: ", Current_Time_Step,
                    " output_interval: ", self.output_interval(),
                    " %: ", (Current_Time_Step % self.output_interval()))

        if (Current_Time_Step % self.output_interval()) == 0:
            if self.rank == 0:
                print ("\n")
                print ("{FENICS} Export files at ", current_time, " [s] ...   ", end="", flush=True)

            # Compute stress
            Vsig = TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc())
            sig = Function(Vsig, name="Stress")
            if self.iNonLinearMethod():
                sig.assign(project(self.Piola_Kirchhoff_sec(
                            displacement_function,self.E,grid_dimension),
                            Vsig, solver_type=self.prjsolver(),
                            form_compiler_parameters={"cpp_optimize": self.cppOptimize(),
                            "representation": self.compRepresentation()}))
            else:
                sig.assign(project(self.Piola_Kirchhoff_sec(
                            displacement_function,self.epsilon,grid_dimension),
                            Vsig, solver_type=self.prjsolver(),
                            form_compiler_parameters={"cpp_optimize": self.cppOptimize(),
                            "representation": self.compRepresentation()}))
            # Save stress solution to file
            sig.rename('Piola Kirchhoff sec Stress', 'stress')
            self.stress_file << (sig, float(current_time))

            # Save displacement solution to file
            displacement_function.rename('Displacement', 'disp')
            self.disp_file << (displacement_function, float(current_time))

            # Compute traction
            traction = Function(VectorFunctionSpace, name="Traction")
            traction.assign(project(self.tF_apply,
                                    VectorFunctionSpace,
                                    solver_type=self.prjsolver(),
                                    form_compiler_parameters={"cpp_optimize": self.cppOptimize(),
                                    "representation": self.compRepresentation()}))
            # Save traction solution to file
            traction.rename('traction', 'trac')
            self.traction_file << (traction, float(current_time))
            if self.rank == 0: print ("Done")
        else:
            pass

    def Post_Solving_Log(self, simtime):
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

    def Checkpoint_Output_Linear(self,
                                 current_time,
                                 mesh,
                                 d0mck_Functions_previous,
                                 u0mck_Functions_previous,
                                 a_Function_previous,
                                 dmck_Function,
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
        hdf5checkpointDataOut.write(u0mck_Functions_previous, "/u0mck", current_time)
        hdf5checkpointDataOut.write(d0mck_Functions_previous, "/d0mck", current_time)
        hdf5checkpointDataOut.write(a_Function_previous, "/a0mck", current_time)
        hdf5checkpointDataOut.write(dmck_Function, "/dmck", current_time)
        hdf5checkpointDataOut.write(self.areaf, "/areaf")
        hdf5checkpointDataOut.close()
        # Delete HDF5File object, closing file
        del hdf5checkpointDataOut

    def Load_Functions_Continue_Run_Linear(self,
                                           d0mck,
                                           u0mck,
                                           a0mck,
                                           dmck):
        if self.iContinueRun():
            hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
            hdf5checkpointDataInTemp.read(d0mck, "/d0mck/vector_0")
            hdf5checkpointDataInTemp.read(u0mck, "/u0mck/vector_0")
            hdf5checkpointDataInTemp.read(a0mck, "/a0mck/vector_0")
            hdf5checkpointDataInTemp.read(dmck, "/dmck/vector_0")
            hdf5checkpointDataInTemp.close()
            # Delete HDF5File object, closing file
            del hdf5checkpointDataInTemp
        else:
            pass

    def Checkpoint_Output_Nonlinear(self,
                                    current_time,
                                    mesh,
                                    ud_Functions_previous,
                                    ud_Functions,
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
        hdf5checkpointDataOut.write(ud_Functions, "/ud", current_time)
        hdf5checkpointDataOut.write(t_Function, "/sigma_s", current_time)
        hdf5checkpointDataOut.write(self.areaf, "/areaf")
        hdf5checkpointDataOut.close()
        # Delete HDF5File object, closing file
        del hdf5checkpointDataOut

    def Load_Functions_Continue_Run_Nonlinear(self, u0d0, ud, sigma_s):
        if self.iContinueRun():
            hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
            hdf5checkpointDataInTemp.read(u0d0, "/u0d0/vector_0")
            hdf5checkpointDataInTemp.read(ud, "/ud/vector_0")
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