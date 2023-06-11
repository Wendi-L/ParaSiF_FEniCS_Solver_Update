""" 
    Parallel Partitioned Multi-physical Simulation Framework (ParaSiF)

    Copyright (C) 2022 Engineering and Environment Group, Scientific 
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
    
    @file couplingMUIFn.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    MUI related coupling functions.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
import sys
import mui4py
from mpi4py import MPI
import petsc4py
import os

class couplingMUIFn:

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
                           xyz_push,
                           xyz_fetch,
                           dmck,
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

            # Best practice suggestion: for a better performance on the RBF method, always switch on the smoothFunc when structure Dofs are more than
            #                           fluid points; Tune the rMUIFetcher to receive a reasonable totForce_Fetch value; Tune the areaListFactor to
            #                           ensure totForce_Fetch and Total_Force_on_structure are the same.
            if self.iMUIFetchMode == 0:
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

            elif self.iMUIFetchMode == 1:
                self.s_sampler = mui4py.SamplerExact()
            else:
                self.s_sampler = mui4py.SamplerPseudoNearestNeighbor(self.rMUIFetcher())

            self.t_sampler = mui4py.ChronoSamplerExact()

            # Commit ZERO step
            self.MUI_Push(xyz_push, dofs_push_list, dmck, 0)
            if self.rank == 0: print ("{FENICS} Commit ZERO step")
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

        if (((total_Sub_Iteration-self.forgetTStepsMUI()) > 0) and (self.forgetTStepsMUI != 0)):
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

        if (((total_Sub_Iteration-self.forgetTStepsMUI()) > 0) and (self.forgetTStepsMUI != 0)):
            a = self.ifaces3d["threeDInterface0"].\
                            forget(total_Sub_Iteration-self.forgetTStepsMUI())
            self.ifaces3d["threeDInterface0"].\
                            set_memory(self.forgetTStepsMUI())
            if (self.rank == 0) and self.iDebug():
                print ('{FENICS} MUI forget step: ',(total_Sub_Iteration-self.forgetTStepsMUI()))

