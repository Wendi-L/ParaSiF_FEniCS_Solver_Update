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
import os
import numpy as np
from mpi4py import MPI
import structureFSISolver

class linearElastic:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Main solver function
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def linearElasticSolve(self):

        #===========================================
        #%% Initialize MPI by mpi4py/MUI for parallelized computation
        #===========================================

        rank = self.MPI_Get_Rank()
        rank_size = self.MPI_Get_Size(self.LOCAL_COMM_WORLD)

        #===========================================
        #%% Time marching parameters define
        #===========================================

        t, t_step, i_sub_it = \
            self.Time_Marching_Parameters(self.LOCAL_COMM_WORLD, self.inputFolderPath)

        self.Time_Marching_Log(self.LOCAL_COMM_WORLD, t, t_step)

        #===========================================
        #%% Solid Mesh input/generation
        #===========================================

        mesh, meshOri, gdim, gdimOri, N = \
            self.Mesh_Generation(self.LOCAL_COMM_WORLD, self.inputFolderPath, self.outputFolderPath)

        #===========================================
        #%% Define coefficients
        #===========================================

        # Time step constants
        k = Constant(self.dt)
        # Time lists
        times = []
        t_sub_it = 0

        # One-step theta value
        theta = Constant(self.thetaOS)

        # Rayleigh damping coefficients
        alpha_rdc = Constant(self.alpha_rdc)
        beta_rdc  = Constant(self.beta_rdc)

        # Generalized-alpha method parameters
        # alpha_m_gam >= alpha_f_gam >= 0.5 for a better performance
        alpha_m_gam = Constant(self.alpha_m_gam)
        alpha_f_gam = Constant(self.alpha_f_gam)
        gamma_gam   = Constant((1./2.) + alpha_m_gam - alpha_f_gam)
        beta_gam    = Constant((1./4.) * (gamma_gam + (1./2.))**2)

        sync = False

        if rank == 0:
            print ("\n")
            print ("{FENICS} One-step theta: ", float(theta))
            print ("\n")

        #===========================================
        #%% Define function spaces
        #===========================================

        if rank == 0: print ("{FENICS} Creating function spaces ...   ")

        V_ele     =     VectorElement("Lagrange", mesh.ufl_cell(), self.deg_fun_spc) # Displacement & Velocity Vector element

        QOri      =     FunctionSpace(meshOri, "Lagrange", self.deg_fun_spc)         # Function space by original mesh
        #SOOri     =     FunctionSpace(meshOri, "Lagrange", 1)                           # Function space with 1st order
        SO        =     FunctionSpace(mesh, "Lagrange", self.deg_fun_spc)            # Function space with updated mesh
        #VS        =     VectorFunctionSpace(mesh, "Lagrange", 1)                     # Vector function space with 1st order
        V         =     VectorFunctionSpace(mesh, "Lagrange", self.deg_fun_spc)      # Vector function space
        VV        =     FunctionSpace(mesh, MixedElement([V_ele, V_ele]))            # Mixed (Velocity (w) & displacement (d)) function space
        T_s_space =     TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc)      # Define nth order structure function spaces

        if rank == 0: print ("{FENICS} Done with creating function spaces")

        #===========================================
        #%% Define functions, test functions and trail functions
        #===========================================

        if rank == 0: print ("{FENICS} Creating functions, test functions and trail functions ...   ", end="", flush=True)

        # Trial functions
        du, dd = TrialFunctions(VV)     # Trial functions for velocity and displacement
        ddmck = TrialFunction(V)        # Trial function for displacement by MCK solving method

        # Test functions
        psi, phi = TestFunctions(VV)    # Test functions for velocity and displacement
        chi = TestFunction(V)           # Test function for displacement by MCK solving method
        
        # Functions at present time step
        ud = Function(VV)               # Functions for velocity and displacement
        u, d = split(ud)                # Split velocity and displacement functions
        dmck = Function(V)              # Function for displacement by MCK solving method


        # Functions at previous time step
        u0d0 = Function(VV)             # Functions for velocity and displacement
        u0, d0 = split(u0d0)            # Split velocity and displacement functions
        d0mck = Function(V)             # Function for displacement by MCK solving method
        u0mck = Function(V)             # Function for velocity by MCK solving method
        a0mck = Function(V)             # Function for acceleration by MCK solving method

        # Define structure traction
        sigma_s = Function(T_s_space)   # Structure traction normal to structure

        #dfst = Function(VS)             # Function for displacement with 1st order
        areaf= Function(QOri)             # Function for facet area

        if self.iContinueRun:
            hdf5checkpointDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/checkpointData.h5", "r")
            hdf5checkpointDataInTemp.read(u0d0, "/u0d0/vector_0")
            hdf5checkpointDataInTemp.read(d0mck, "/d0mck/vector_0")
            hdf5checkpointDataInTemp.read(u0mck, "/u0mck/vector_0")
            hdf5checkpointDataInTemp.read(a0mck, "/a0mck/vector_0")
            hdf5checkpointDataInTemp.read(ud, "/ud/vector_0")
            hdf5checkpointDataInTemp.read(dmck, "/dmck/vector_0")
            hdf5checkpointDataInTemp.read(sigma_s, "/sigma_s/vector_0")
            #hdf5checkpointDataInTemp.read(areaf, "/areaf/vector_0")
            hdf5checkpointDataInTemp.close()
            # Delete HDF5File object, closing file
            del hdf5checkpointDataInTemp
        else:
            pass

        if rank == 0: print ("Done")

        #===========================================
        #%% Define traction forces
        #===========================================

        if self.iNonUniTraction:
            if rank == 0: print ("{FENICS} Non-uniform traction applied")
            tF_apply = Function(V)
            tF_apply_vec = tF_apply.vector().get_local()

            if self.iMUIFetchForce:
                force_dof_apply = Function(V)
                force_dof_apply_vec = force_dof_apply.vector().get_local()
        else:
            if rank == 0: print ("{FENICS} Uniform traction applied")
            tF_magnitude = Constant(-(0.0)/(self.YBeam*self.ZBeam))
            tF_apply = tF_magnitude*self.X_direction_vector()

        #===========================================
        #%% Define SubDomains and boundaries
        #===========================================

        boundaries, boundariesOri, ds = \
            self.SubDomains_Boundaries_Generation(  self.LOCAL_COMM_WORLD, 
                                                    mesh, 
                                                    meshOri, 
                                                    gdim, 
                                                    gdimOri, 
                                                    V, 
                                                    self.inputFolderPath, 
                                                    self.outputFolderPath)

        #===========================================
        #%% Define boundary conditions
        #===========================================

        if rank == 0: print ("{FENICS} Creating 3D boundary conditions ...   ", end="", flush=True)
        if self.solving_method == 'STVK':
            bc1,bc2 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
            #bc1 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
            #!!!!->   
            #bc3 = DirichletBC(VV.sub(0).sub(1), (0.0),boundaries, 8)
            #bc4 = DirichletBC(VV.sub(1).sub(1), (0.0),boundaries, 8)
            #!!!!<-  
            #bcs = [bc1,bc2,bc3,bc4]
            bcs = [bc1,bc2]
        elif self.solving_method == 'MCK':
            bc1 = self.dirichletBCs.DirichletBCs(V,boundaries,1)
        #!!!!->   
            #bc2 = DirichletBC(V.sub(1), (0.0),boundaries, 8)
            #bc3 = DirichletBC(V.sub(0), (0.0),boundaries, 8)
            #bc2 = DirichletBC(V, ((0.0,0.0,0.0)),boundaries, 8)
        #!!!!<-    
            #bcs = [bc1,bc2]
            bcs = [bc1]
        if rank == 0: print ("Done")

        #===========================================
        #%% Define DOFs and Coordinates mapping
        #===========================================  

        dofs_to_xyz = self.dofs_to_xyz(QOri, gdimOri)

        dofs_fetch, dofs_fetch_list, xyz_fetch = \
            self.dofs_fetch_list(boundariesOri, QOri, 2, gdimOri)
        dofs_push, dofs_push_list, xyz_push = \
            self.dofs_push_list(boundariesOri, QOri, 2, gdimOri)

        xyz_fetch_list =  list(xyz_fetch)
        xyz_fetch_list_total_group = []
        #print ("{FEniCS***} out: len(dofs_fetch_list): ", len(dofs_fetch_list), " len(dofs_push_list): ", len(dofs_push_list))

        #===========================================
        #%% Define facet areas
        #===========================================

        areaf_vec = areaf.vector().get_local()

        if (self.iMUIFetchForce):
            if (self.iUseRBF):

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                local = np.zeros(1)
                total = np.zeros(1)

                local[0] = len(xyz_fetch_list)
                total[0] = 0

                xyz_fetch_list_flat = [item for sublist in xyz_fetch_list for item in sublist]

                # use MPI to get the totals 
                self.LOCAL_COMM_WORLD.Reduce(local,total,op = MPI.SUM,root = 0)
                self.LOCAL_COMM_WORLD.Bcast(total, root=0)

                xyz_fetch_list_total_flat = np.empty(int(total[0]*3), dtype=np.float64)

                xyz_fetch_list_total = self.LOCAL_COMM_WORLD.gather(xyz_fetch_list_flat, root = 0)
                if self.LOCAL_COMM_WORLD.rank == 0:
                    xyz_fetch_list_total_flat = np.asarray([item for sublist in xyz_fetch_list_total for item in sublist])

                self.LOCAL_COMM_WORLD.Bcast(xyz_fetch_list_total_flat, root=0)
                xyz_fetch_list_total_group = [ xyz_fetch_list_total_flat.tolist()[i:i+3]
                                                    for i in range(0, len(xyz_fetch_list_total_flat.tolist()), 3) ]

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if (not self.iContinueRun):
                if (not self.iLoadAreaList):
                    if rank == 0: print ("{FENICS} facet area calculating")

                    areaf_vec = self.facets_area_list(  self.LOCAL_COMM_WORLD,
                                                        meshOri,
                                                        QOri,
                                                        boundariesOri,
                                                        dofs_fetch_list,
                                                        gdimOri,
                                                        areaf_vec)

                    # Apply the facet area vectors
                    areaf.vector().set_local(areaf_vec)
                    areaf.vector().apply("insert")
                    if (self.iHDF5FileExport) and (self.iHDF5MeshExport):
                        hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                    else:
                        hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "w")
                    hdfOutTemp.write(areaf, "/areaf")
                    hdfOutTemp.close()

                else:

                    hdf5meshAreaDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                    hdf5meshAreaDataInTemp.read(areaf, "/areaf/vector_0")
                    hdf5meshAreaDataInTemp.close()

            else:
                if (not self.iLoadAreaList):
                    if rank == 0: print ("{FENICS} facet area calculating")

                    areaf_vec = self.facets_area_list(  self.LOCAL_COMM_WORLD,
                                                        meshOri,
                                                        QOri,
                                                        boundariesOri,
                                                        dofs_fetch_list,
                                                        gdimOri,
                                                        areaf_vec)

                    # Apply the facet area vectors
                    areaf.vector().set_local(areaf_vec)
                    areaf.vector().apply("insert")
                    if (self.iHDF5FileExport) and (self.iHDF5MeshExport):
                        hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "a")
                    else:
                        hdfOutTemp = HDF5File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/mesh_boundary_and_values.h5", "w")
                    hdfOutTemp.write(areaf, "/areaf")
                    hdfOutTemp.close()

                else:

                    hdf5meshAreaDataInTemp = HDF5File(self.LOCAL_COMM_WORLD, self.inputFolderPath + "/mesh_boundary_and_values.h5", "r")
                    hdf5meshAreaDataInTemp.read(areaf, "/areaf/vector_0")
                    hdf5meshAreaDataInTemp.close()

        #===========================================
        #%% Prepare post-process files
        #===========================================

        if rank == 0: print ("{FENICS} Preparing post-process files ...   ", end="", flush=True)

        disp_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/displacement.pvd")
        stress_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/stress.pvd")
        traction_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/surface_traction_structure.pvd")

        if rank == 0: print ("Done")

        #===========================================
        #%% Define the variational FORM 
        #%% and 
        #%% Jacobin functions of structure
        #===========================================

        if self.solving_method == 'STVK':

            if rank == 0: print ("{FENICS} Defining variational FORM and Jacobin functions ...   ", end="", flush=True)

            # Define the traction terms of the structure variational form
            tF = dot(self.F_(d,gdim).T, tF_apply)
            tF_ = dot(self.F_(d0,gdim).T, tF_apply)

            # Define the transient terms of the structure variational form
            Form_s_T = (1/k)*self.rho_s*inner((u-u0), psi)*dx
            Form_s_T += (1/k)*inner((d-d0), phi)*dx

            # Define the stress terms and convection of the structure variational form
            if self.iNonLinearMethod:
                if rank == 0: print ("{FENICS} [Defining non-linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by the constitutive law of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation). Valid for large deformations but small strain] ...   ", end="", flush=True)
                Form_s_SC = inner(theta * self.Piola_Kirchhoff_fst(d,gdim) + (1 - theta) * 
                            self.Piola_Kirchhoff_fst(d0,gdim), grad(psi)) * dx
                Form_s_SC -= inner(theta*u + (1-theta)*u0, phi ) * dx
            else:
                if rank == 0: print ("{FENICS} [Defining linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by Hooke's law (linear relation). Valid for small-scale deformations only] ...   ", end="", flush=True)
                Form_s_SC = inner(theta * self.Hooke_stress(d,gdim) + (1 - theta) * 
                            self.Hooke_stress(d0,gdim), grad(psi)) * dx
                Form_s_SC -= inner(theta*u + (1-theta)*u0, phi ) * dx

            # Define the body forces and surface tractions terms of the structure variational form
            Form_s_ET = -( theta * self.J_(d,gdim) * inner( (self.b_for()), psi ) + 
                        ( 1 - theta ) * self.J_(d0,gdim) * inner( (self.b_for()), psi ) ) * dx
            Form_s_ET -= ( theta * self.J_(d,gdim) * inner( tF, psi ) + 
                        ( 1 - theta ) * self.J_(d0,gdim) * inner( tF_, psi ) ) * ds(2)
            Form_s_ET -= ( theta * self.J_(d,gdim) * inner( inv(self.F_(d,gdim)) * sigma_s * N, psi )+ 
                        ( 1 - theta ) * (self.J_(d0,gdim)) * inner(inv(self.F_(d0,gdim)) * sigma_s * N, psi )) * ds(2)

            # Define the final form of the structure variational form
            Form_s = Form_s_T + Form_s_SC + Form_s_ET

            # Make functional into a vector function
            #Form_s = action(Form_s, ud)

            Jaco = derivative(Form_s, ud) # Define Jacobin functions

            if rank == 0: print ("Done")

        elif self.solving_method == 'MCK':
            if rank == 0: print ("{FENICS} Defining variational FORM functions ...   ", end="", flush=True)
            # Define the traction terms of the structure variational form
            tF = dot(chi, tF_apply)

            Form_s_Update_Acce = self.AMCK (ddmck, 
                                            d0mck, 
                                            u0mck,
                                            a0mck,
                                            beta_gam)
            
            Form_s_Update_velo = self.UMCK (Form_s_Update_Acce, 
                                            u0mck, 
                                            a0mck, 
                                            gamma_gam)

            Form_s_Ga_Acce = self.Generalized_Alpha_Weights(Form_s_Update_Acce,a0mck,alpha_m_gam)
            Form_s_Ga_velo = self.Generalized_Alpha_Weights(Form_s_Update_velo,u0mck,alpha_f_gam)
            Form_s_Ga_disp = self.Generalized_Alpha_Weights(ddmck,d0mck,alpha_f_gam)
            Form_s_M_Matrix = self.rho_s * inner(Form_s_Ga_Acce, chi) * dx
            Form_s_M_for_C_Matrix = self.rho_s * inner(Form_s_Ga_velo, chi) * dx
            Form_s_K_Matrix = inner(self.elastic_stress(Form_s_Ga_disp,gdim), sym(grad(chi))) * dx
            Form_s_K_for_C_Matrix = inner(self.elastic_stress(Form_s_Ga_velo,gdim), sym(grad(chi))) * dx
            Form_s_C_Matrix = alpha_rdc * Form_s_M_for_C_Matrix + beta_rdc * Form_s_K_for_C_Matrix
            Form_s_F_Ext = tF * ds(2)

            Form_s = Form_s_M_Matrix + Form_s_C_Matrix + Form_s_K_Matrix - Form_s_F_Ext

            Bilinear_Form = lhs(Form_s)
            Linear_Form   = rhs(Form_s)

            if rank == 0: print ("Done")

        #===========================================
        #%% Initialize solver
        #===========================================

        if self.solving_method == 'STVK':

            problem = NonlinearVariationalProblem(Form_s, ud, bcs=bcs, J=Jaco)
            solver = NonlinearVariationalSolver(problem)

            info(solver.parameters, False)
            if self.nonlinear_solver == "newton":
                solver.parameters["nonlinear_solver"]= self.nonlinear_solver
                solver.parameters["newton_solver"]["absolute_tolerance"] = self.prbAbsolute_tolerance
                solver.parameters["newton_solver"]["relative_tolerance"] = self.prbRelative_tolerance
                solver.parameters["newton_solver"]["maximum_iterations"] = self.prbMaximum_iterations
                solver.parameters["newton_solver"]["relaxation_parameter"] = self.prbRelaxation_parameter
                solver.parameters["newton_solver"]["linear_solver"] = self.prbsolver
                solver.parameters["newton_solver"]["preconditioner"] = self.prbpreconditioner
                solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                solver.parameters["newton_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess
                solver.parameters["newton_solver"]["krylov_solver"]['error_on_nonconvergence'] = self.error_on_nonconvergence
            elif self.nonlinear_solver == "snes":
                solver.parameters['nonlinear_solver'] = self.nonlinear_solver
                solver.parameters['snes_solver']['line_search'] = self.lineSearch
                solver.parameters['snes_solver']['linear_solver'] = self.prbsolver
                solver.parameters['snes_solver']['preconditioner'] = self.prbpreconditioner
                solver.parameters['snes_solver']['absolute_tolerance'] = self.prbAbsolute_tolerance
                solver.parameters['snes_solver']['relative_tolerance'] = self.prbRelative_tolerance
                solver.parameters['snes_solver']['maximum_iterations'] = self.prbMaximum_iterations
                solver.parameters['snes_solver']['report'] = self.show_report
                solver.parameters['snes_solver']['error_on_nonconvergence'] = self.error_on_nonconvergence
                solver.parameters["snes_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                solver.parameters["snes_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                solver.parameters["snes_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                solver.parameters["snes_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                solver.parameters["snes_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess
            else:
                sys.exit("{FENICS} Error, nonlinear solver value not recognized")

        elif self.solving_method == 'MCK':

            if self.linear_solver == 'LU':
                Bilinear_Assemble, Linear_Assemble = assemble_system(Bilinear_Form, Linear_Form, bcs)
                solver = LUSolver(Bilinear_Assemble, "mumps")
                solver.parameters["symmetric"] = True

            elif self.linear_solver == 'LinearVariational':
                problem = LinearVariationalProblem(Bilinear_Form, Linear_Form, dmck, bcs)
                solver = LinearVariationalSolver(problem)
                # Set linear solver parameters
                solver.parameters["linear_solver"] = self.prbsolver
                solver.parameters["preconditioner"] = self.prbpreconditioner
                solver.parameters["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance
                solver.parameters["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance
                solver.parameters["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations
                solver.parameters["krylov_solver"]["monitor_convergence"] = self.monitor_convergence
                solver.parameters["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess

        #===========================================
        #%% Setup checkpoint data
        #===========================================

        self.Checkpoint_Output( self.LOCAL_COMM_WORLD, 
                                self.outputFolderPath, 
                                (t-self.dt), 
                                mesh, 
                                meshOri, 
                                u0d0, 
                                d0mck, 
                                u0mck, 
                                a0mck, 
                                ud, 
                                dmck, 
                                sigma_s, 
                                areaf, 
                                False)

        #===========================================
        #%% Define MUI samplers and commit ZERO step
        #===========================================

        if self.iMUICoupling:
            if self.iUseRBF:
                if self.iMUIFetchForce:
                    if self.iparallelFSICoupling:
                        t_sampler, s_sampler = \
                            self.MUI_Sampler_RBF_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    dofs_to_xyz,
                                                    dofs_push_list,
                                                    dofs_to_xyz,
                                                    xyz_fetch_list_total_group,
                                                    t_step)
                    else:
                        t_sampler, s_sampler = \
                            self.MUI_Sampler_RBF_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    dofs_to_xyz,
                                                    dofs_push_list,
                                                    dofs_to_xyz,
                                                    xyz_fetch_list_total_group,
                                                    t_step)
                else:
                    if self.iparallelFSICoupling:
                        t_sampler, s_sampler = \
                            self.MUI_Sampler_RBF_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    dofs_to_xyz,
                                                    dofs_push_list,
                                                    dofs_to_xyz,
                                                    xyz_fetch_list_total_group,
                                                    t_step)
                    else:
                        t_sampler, s_sampler = \
                            self.MUI_Sampler_RBF_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    dofs_to_xyz,
                                                    dofs_push_list,
                                                    dofs_to_xyz,
                                                    xyz_fetch_list_total_group,
                                                    t_step)
            else:
                if self.iMUIFetchForce:
                    t_sampler, s_sampler = \
                            self.MUI_Sampler_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    xyz_fetch,
                                                    dofs_push_list,
                                                    xyz_push,
                                                    t_step)
                else:
                    t_sampler, s_sampler = \
                            self.MUI_Sampler_Define(self.LOCAL_COMM_WORLD, 
                                                    self.ifaces3d,
                                                    dofs_fetch_list,
                                                    xyz_fetch,
                                                    dofs_push_list,
                                                    xyz_push,
                                                    t_step)
            
        else:
            pass

        if self.iExporttxt: self.Time_Txt_Export_init(self.LOCAL_COMM_WORLD, self.outputFolderPath)

        # Finish the wall clock on fetch Time
        fetchTime = 0.0
        forceVecProjTime = 0.0
        linearAssembleTime = 0.0
        bcApplyTime = 0.0
        solverSolveTime = 0.0
        totalForceCalTime = 0.0
        printDispTime = 0.0
        pushTime = 0.0
        moveMeshTime = 0.0
        dispVTKExpTime = 0.0
        dispTxtExpTime = 0.0
        checkpointExpTime = 0.0
        assignOldFuncSpaceTime = 0.0
        simtimePerIter = 0.0
        simtimePerStep = 0.0

        #===========================================
        #%% Define time loops
        #===========================================

        # Time-stepping
        while t <= self.T:

            # create an instance of the TicToc wall clock class
            wallClockPerStep = structureFSISolver.tictoc.TicToc()
            # Starts the wall clock
            wallClockPerStep.tic()

            # Update time list    
            times.append(t)
            n_steps = len(times)

            if rank == 0: 
                print ("\n")
                print ("\n")
                print ("{FENICS} Time: ", t)
                print ("{FENICS} Time Steps: ", n_steps)

            if (self.iChangeSubIter):
                if (t >= self.TChangeSubIter):
                    present_num_sub_iteration = self.num_sub_iteration_new
                else:
                    present_num_sub_iteration = self.num_sub_iteration
            else:
                present_num_sub_iteration = self.num_sub_iteration

            # Sub-iteration for coupling
            while i_sub_it <= present_num_sub_iteration:

                t_sub_it += 1

                if rank == 0: 
                    print ("\n")
                    print ("{FENICS} sub-iteration: ", i_sub_it)
                    print ("{FENICS} total sub-iterations to now: ", t_sub_it)

                # create an instance of the TicToc wall clock class
                wallClockPerIter = structureFSISolver.tictoc.TicToc()
                # Starts the wall clock
                wallClockPerIter.tic()

                # create an instance of the TicToc wall clock class
                wallClockFetchTime = structureFSISolver.tictoc.TicToc()
                wallClockForceVecProj = structureFSISolver.tictoc.TicToc()
                wallClockLinearAssemble = structureFSISolver.tictoc.TicToc()
                wallClockBCApply = structureFSISolver.tictoc.TicToc()
                wallClockSolverSolve = structureFSISolver.tictoc.TicToc()
                wallClockTotalForceCal = structureFSISolver.tictoc.TicToc()
                wallClockPrintDisp = structureFSISolver.tictoc.TicToc()
                wallClockPush = structureFSISolver.tictoc.TicToc()
                wallClockMoveMesh = structureFSISolver.tictoc.TicToc()
                wallClockDispVTKExp = structureFSISolver.tictoc.TicToc()
                wallClockDispTxtExp = structureFSISolver.tictoc.TicToc()
                wallClockCheckpointExp = structureFSISolver.tictoc.TicToc()
                wallClockAssignOldFuncSpace = structureFSISolver.tictoc.TicToc()

                # Assign traction forces at present time step
                if self.iNonUniTraction:
                    if self.iMUICoupling:
                        if self.iMUIFetchForce:
                            # Starts the wall clock
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            wallClockFetchTime.tic()
                            if len(xyz_fetch)!=0:
                                if self.iparallelFSICoupling:
                                    if self.iUseRBF:
                                        if self.iMUIFetchMoment:
                                            tF_apply_vec, mom_x, mom_y, mom_z = \
                                                self.MUI_Parallel_FSI_RBF_Fetch( self.LOCAL_COMM_WORLD,
                                                                            self.ifaces3d,
                                                                            xyz_fetch,
                                                                            dofs_fetch_list,
                                                                            t_sampler,
                                                                            s_sampler,
                                                                            n_steps,
                                                                            i_sub_it,
                                                                            t_sub_it,
                                                                            tF_apply_vec,
                                                                            areaf_vec,
                                                                            self.outputFolderPath)

                                        else:
                                            tF_apply_vec = self.MUI_Parallel_FSI_RBF_Fetch(  self.LOCAL_COMM_WORLD, 
                                                                            self.ifaces3d, 
                                                                            xyz_fetch, 
                                                                            dofs_fetch_list, 
                                                                            t_sampler, 
                                                                            s_sampler, 
                                                                            n_steps,
                                                                            i_sub_it,
                                                                            t_sub_it, 
                                                                            tF_apply_vec, 
                                                                            areaf_vec,
                                                                            self.outputFolderPath)
                                            if len(dofs_fetch_list) != 0:
                                                temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                            if self.iMultidomain:

                                                for i, p in enumerate(dofs_fetch_list):

                                                    if t <= 0.5:
                                                        tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                            else:

                                                for i, p in enumerate(dofs_fetch_list):

                                                    if t <= 0.5:
                                                        tF_apply_vec[0::3][p] += ((t*((self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[1::3][p] += ((t*((self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[2::3][p] += ((t*((self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)

                                    else:
                                        #nearest neighbour
                                        if self.iMUIFetchMoment:
                                            tF_apply_vec, mom_x, mom_y, mom_z = \
                                                self.MUI_Parallel_FSI_Fetch( self.LOCAL_COMM_WORLD,
                                                                            self.ifaces3d,
                                                                            xyz_fetch,
                                                                            dofs_fetch_list,
                                                                            t_sampler,
                                                                            s_sampler,
                                                                            n_steps,
                                                                            i_sub_it,
                                                                            t_sub_it,
                                                                            tF_apply_vec,
                                                                            force_dof_apply_vec,
                                                                            areaf_vec,
                                                                            self.outputFolderPath)

                                        else:
                                            tF_apply_vec = self.MUI_Parallel_FSI_Fetch(  self.LOCAL_COMM_WORLD, 
                                                                            self.ifaces3d, 
                                                                            xyz_fetch, 
                                                                            dofs_fetch_list, 
                                                                            t_sampler, 
                                                                            s_sampler, 
                                                                            n_steps,
                                                                            i_sub_it,
                                                                            t_sub_it, 
                                                                            tF_apply_vec, 
                                                                            force_dof_apply_vec, 
                                                                            areaf_vec,
                                                                            self.outputFolderPath)

                                            #self.ifaces3d.barrier(float(t_sub_it))
                                            
                                            if len(dofs_fetch_list) != 0:
                                                temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                            if self.iMultidomain:

                                                for i, p in enumerate(dofs_fetch_list):

                                                    if t <= 0.5:
                                                        tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                            else:

                                                for i, p in enumerate(dofs_fetch_list):

                                                    if t <= 0.5:
                                                        tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                        tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                else:
                                    if self.iMUIFetchMoment:
                                        tF_apply_vec, mom_x, mom_y, mom_z = \
                                            self.MUI_Fetch( self.LOCAL_COMM_WORLD, 
                                                            self.ifaces3d, 
                                                            xyz_fetch, 
                                                            dofs_fetch_list, 
                                                            t_sampler, 
                                                            s_sampler, 
                                                            n_steps, 
                                                            t_sub_it, 
                                                            tF_apply_vec, 
                                                            areaf_vec)
                                    else:
                                        tF_apply_vec = self.MUI_Fetch(  self.LOCAL_COMM_WORLD, 
                                                                        self.ifaces3d, 
                                                                        xyz_fetch, 
                                                                        dofs_fetch_list, 
                                                                        t_sampler, 
                                                                        s_sampler, 
                                                                        n_steps, 
                                                                        t_sub_it, 
                                                                        tF_apply_vec, 
                                                                        areaf_vec)
                                        if len(dofs_fetch_list) != 0:
                                            temp_area_pernode = (self.XBeam*self.ZBeam)/len(dofs_fetch_list)

                                        if self.iMultidomain:

                                            for i, p in enumerate(dofs_fetch_list):

                                                if t <= 0.5:
                                                    tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                        else:

                                            for i, p in enumerate(dofs_fetch_list):

                                                if t <= 0.5:
                                                    tF_apply_vec[0::3][p] += ((t*(+(self.bForExtX/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    tF_apply_vec[1::3][p] += ((t*(+(self.bForExtY/len(dofs_fetch_list)))/0.5)/temp_area_pernode)
                                                    tF_apply_vec[2::3][p] += ((t*(+(self.bForExtZ/len(dofs_fetch_list)))/0.5)/temp_area_pernode)

                                
                            # Finish the wall clock on fetch Time
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            fetchTime = wallClockFetchTime.toc()

                            # Starts the wall clock
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            wallClockForceVecProj.tic()
                            if (self.iMUIFetchValue) and (not ((self.iContinueRun) and (n_steps == 1))):
                                # Apply traction components. These calls do parallel communication
                                tF_apply.vector().set_local(tF_apply_vec)
                                tF_apply.vector().apply("insert")
                                # tF_apply = project(dfst, V)
                                # tF_apply.assign(project(dfst, 
                                    # V, 
                                    # solver_type=self.prjsolver,
                                    # form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                                    # "representation": self.compRepresentation}))
                            else:
                                # do not apply the fetched value, i.e. one-way coupling
                                pass

                            # Finish the wall clock on force vector project
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            forceVecProjTime = wallClockForceVecProj.toc()

                        else:
                            # Starts the wall clock
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            wallClockFetchTime.tic()
                            if self.iparallelFSICoupling:
                                if self.iMUIFetchMoment:
                                    tF_apply_vec, mom_x, mom_y, mom_z = \
                                    self.MUI_Parallel_FSI_Fetch( self.LOCAL_COMM_WORLD,
                                                                    self.ifaces3d,
                                                                    xyz_fetch,
                                                                    dofs_fetch_list,
                                                                    t_sampler,
                                                                    s_sampler,
                                                                    n_steps,
                                                                    i_sub_it,
                                                                    t_sub_it,
                                                                    tF_apply_vec,
                                                                    force_dof_apply_vec,
                                                                    areaf_vec,
                                                                    self.outputFolderPath)
                                else:
                                    tF_apply_vec = self.MUI_Parallel_FSI_Fetch( self.LOCAL_COMM_WORLD,
                                                                    self.ifaces3d,
                                                                    xyz_fetch,
                                                                    dofs_fetch_list,
                                                                    t_sampler,
                                                                    s_sampler,
                                                                    n_steps,
                                                                    i_sub_it,
                                                                    t_sub_it,
                                                                    tF_apply_vec,
                                                                    force_dof_apply_vec,
                                                                    areaf_vec,
                                                                    self.outputFolderPath)
                            else:
                                if self.iMUIFetchMoment:
                                    tF_apply_vec, mom_x, mom_y, mom_z = \
                                        self.MUI_Fetch( self.LOCAL_COMM_WORLD, 
                                                        self.ifaces3d, 
                                                        xyz_fetch, 
                                                        dofs_fetch_list, 
                                                        t_sampler, 
                                                        s_sampler, 
                                                        n_steps, 
                                                        t_sub_it, 
                                                        tF_apply_vec, 
                                                        areaf_vec)
                                else:
                                    tF_apply_vec = self.MUI_Fetch(  self.LOCAL_COMM_WORLD, 
                                                                    self.ifaces3d, 
                                                                    xyz_fetch, 
                                                                    dofs_fetch_list, 
                                                                    t_sampler, 
                                                                    s_sampler, 
                                                                    n_steps, 
                                                                    t_sub_it, 
                                                                    tF_apply_vec, 
                                                                    areaf_vec)
                            # Finish the wall clock on fetch Time
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            fetchTime = wallClockFetchTime.toc()

                            # Starts the wall clock
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            wallClockForceVecProj.tic()
                            if self.iMUIFetchValue:
                                # Apply traction components. These calls do parallel communication
                                tF_apply.vector().set_local(tF_apply_vec)
                                tF_apply.vector().apply("insert")
                            else:
                                # do not apply the fetched value, i.e. one-way coupling
                                pass
                            # Finish the wall clock on force vector project
                            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                            forceVecProjTime = wallClockForceVecProj.toc()

                    else:
                        if rank == 0: print ("{FENICS} Assigning traction forces at present time step ...   ", 
                                                end="", flush=True)

                        temp_area_pernode = (self.YBeam*self.ZBeam)/len(dofs_fetch_list)
                        print ("temp_area_pernode", temp_area_pernode)
                        for i, p in enumerate(dofs_fetch_list):

                            if t <= 7.0:
                                tF_apply_vec[1::3][p] = ((t*(-(500.0/len(dofs_fetch_list)))/7.0)/temp_area_pernode)
                            else:
                                tF_apply_vec[1::3][p] = (0.0)

                            tF_apply_vec[0::3][p] = 0.0
                            tF_apply_vec[2::3][p] = 0.0
                            print ("tF_apply_vec[1::3][p]", tF_apply_vec[1::3][p])
                        
                        # Apply traction components. These calls do parallel communication
                        tF_apply.vector().set_local(tF_apply_vec)
                        tF_apply.vector().apply("insert")
                        # tF_apply = project(dfst, V)
                        # tF_apply.assign(project(dfst, 
                            # V, 
                            # solver_type=self.prjsolver,
                            # form_compiler_parameters={"cpp_optimize": self.cppOptimize, 
                            # "representation": self.compRepresentation}))
                        # # Apply traction components. These calls do parallel communication
                        # tF_apply.vector().set_local(tF_apply_vec)
                        # tF_apply.vector().apply("insert")

                else:
                    if rank == 0: print ("{FENICS} Assigning traction forces at present time step ...   ", 
                                            end="", flush=True)
                    if t <= 7.0:
                        tF_magnitude.assign(Constant((t*(-500.0)/7.0)/(self.YBeam*self.ZBeam)))
                    else:
                        tF_magnitude.assign(Constant((0.0)/(self.YBeam*self.ZBeam)))
                    if rank == 0:
                        print ("Done")
                        #if self.iDebug: print ("{FENICS} tF_magnitude: ", tF_magnitude(0))

                if (not ((self.iContinueRun) and (n_steps == 1))):
                    if self.solving_method == 'MCK':
                        # Starts the wall clock
                        if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                        wallClockLinearAssemble.tic()
                        # Assemble linear form
                        if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            pass
                        else:
                            Linear_Assemble = assemble(Linear_Form)
                        # Finish the wall clock on linear assemble
                        if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                        linearAssembleTime = wallClockLinearAssemble.toc()
                        #bcs.apply(Linear_Assemble)
                        #!!!!!->
                        # Starts the wall clock
                        if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                        wallClockBCApply.tic()
                        if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            pass
                        else:
                            [bc.apply(Linear_Assemble) for bc in bcs]
                        # Finish the wall clock on bc apply
                        if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                        bcApplyTime = wallClockBCApply.toc()
                        #!!!!!<-
                    # Solving the structure functions inside the time loop
                    # Starts the wall clock
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    wallClockSolverSolve.tic()
                    if (self.solving_method == 'MCK') and (self.linear_solver == 'LU'):
                        solver.solve(Bilinear_Assemble, dmck.vector(), Linear_Assemble)
                    else:
                        if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                            pass
                        else:
                            solver.solve()
                    # Finish the wall clock on solver solve
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    solverSolveTime = wallClockSolverSolve.toc()
                    # Starts the wall clock
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    wallClockTotalForceCal.tic()
                    if self.solving_method == 'MCK':
                        force_X = dot(tF_apply, self.X_direction_vector())*ds(2)
                        force_Y = dot(tF_apply, self.Y_direction_vector())*ds(2)
                        force_Z = dot(tF_apply, self.Z_direction_vector())*ds(2)
                    else:
                        force_X = dot(tF, self.X_direction_vector())*ds(2)
                        force_Y = dot(tF, self.Y_direction_vector())*ds(2)
                        force_Z = dot(tF, self.Z_direction_vector())*ds(2)

                    f_X_a = assemble(force_X)
                    f_Y_a = assemble(force_Y)
                    f_Z_a = assemble(force_Z)

                    print ("{FENICS} Total Force_X on structure: ", f_X_a, " at rank ", rank)
                    print ("{FENICS} Total Force_Y on structure: ", f_Y_a, " at rank ", rank)
                    print ("{FENICS} Total Force_Z on structure: ", f_Z_a, " at rank ", rank)
                    # Finish the wall clock on total force calculate
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    totalForceCalTime = wallClockTotalForceCal.toc()
                else:
                    pass

                if self.solving_method == 'STVK':
                    # Split function spaces
                    u,d = ud.split(True)

                    # Compute and print the displacement of monitored point
                    self.print_Disp (self.LOCAL_COMM_WORLD, d)

                    # MUI Push internal points and commit current steps
                    if (self.iMUICoupling) and (len(xyz_push)!=0):
                        self.MUI_Push(  self.LOCAL_COMM_WORLD, 
                                        self.ifaces3d, 
                                        xyz_push, 
                                        dofs_push_list, 
                                        d, 
                                        t_sub_it)
                    else:
                        pass

                elif self.solving_method == 'MCK':
                    # Starts the wall clock
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    wallClockPrintDisp.tic()
                    # Compute and print the displacement of monitored point
                    self.print_Disp (self.LOCAL_COMM_WORLD, dmck)
                    # Finish the wall clock on print disp
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    printDispTime = wallClockPrintDisp.toc()
                    # Starts the wall clock
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    wallClockPush.tic()
                    # MUI Push internal points and commit current steps
                    if (self.iMUICoupling) and (len(xyz_push)!=0):
                        self.MUI_Push(  self.LOCAL_COMM_WORLD, 
                                        self.ifaces3d, 
                                        xyz_push, 
                                        dofs_push_list, 
                                        dmck, 
                                        t_sub_it)

                    else:
                        pass
                    # Finish the wall clock on push
                    if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                    pushTime = wallClockPush.toc()
                # Finish the wall clock on total sim time Per iter
                simtimePerIter = wallClockPerIter.toc()

                if self.iExporttxt: self.Time_Txt_Export(self.LOCAL_COMM_WORLD, 
                                                        t,
                                                        n_steps,
                                                        i_sub_it,
                                                        fetchTime,
                                                        forceVecProjTime,
                                                        linearAssembleTime,
                                                        bcApplyTime,
                                                        solverSolveTime,
                                                        totalForceCalTime,
                                                        printDispTime,
                                                        pushTime,
                                                        moveMeshTime,
                                                        dispVTKExpTime,
                                                        dispTxtExpTime,
                                                        checkpointExpTime,
                                                        assignOldFuncSpaceTime,
                                                        simtimePerIter,
                                                        simtimePerStep,
                                                        self.outputFolderPath)

                # Move to the next sub-iteration
                i_sub_it += 1

            if self.solving_method == 'STVK':
                # Split function spaces
                u,d = ud.split(True)
                u0,d0 = u0d0.split(True)

                self.Move_Mesh(V, d, d0, mesh)

                if (not (self.iQuiet)):
                    self.Export_Disp_vtk(   self.LOCAL_COMM_WORLD, 
                                            n_steps, 
                                            t, 
                                            mesh, 
                                            gdim, 
                                            V, 
                                            tF, 
                                            d, 
                                            stress_file, 
                                            disp_file, 
                                            traction_file)

                    self.Export_Disp_txt(   self.LOCAL_COMM_WORLD, 
                                            d, 
                                            self.outputFolderPath)

                    self.Checkpoint_Output( self.LOCAL_COMM_WORLD, 
                                            self.outputFolderPath, 
                                            t, 
                                            mesh, 
                                            meshOri, 
                                            u0d0, 
                                            d0mck, 
                                            u0mck, 
                                            a0mck, 
                                            ud, 
                                            dmck, 
                                            sigma_s, 
                                            areaf, 
                                            True)

            elif self.solving_method == 'MCK':
                # Starts the wall clock
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                wallClockMoveMesh.tic()
                self.Move_Mesh(V, dmck, d0mck, mesh)
                # Finish the wall clock on move mesh
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                moveMeshTime = wallClockMoveMesh.toc()
                # Starts the wall clock
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                wallClockDispVTKExp.tic()
                if (not (self.iQuiet)):
                    self.Export_Disp_vtk(   self.LOCAL_COMM_WORLD, 
                                            n_steps, 
                                            t, 
                                            mesh, 
                                            gdim, 
                                            V, 
                                            tF_apply, 
                                            dmck, 
                                            stress_file, 
                                            disp_file, 
                                            traction_file)
                # Finish the wall clock on disp VTK export
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                dispVTKExpTime = wallClockDispVTKExp.toc()
                # Starts the wall clock
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                wallClockDispTxtExp.tic()
                if (not (self.iQuiet)):
                    self.Export_Disp_txt(   self.LOCAL_COMM_WORLD, 
                                            dmck, 
                                            self.outputFolderPath)
                # Finish the wall clock on disp txt export
                if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
                dispTxtExpTime = wallClockDispTxtExp.toc()
            # Starts the wall clock
            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
            wallClockCheckpointExp.tic()
            if (not (self.iQuiet)):
                self.Checkpoint_Output( self.LOCAL_COMM_WORLD, 
                                        self.outputFolderPath, 
                                        t, 
                                        mesh, 
                                        meshOri, 
                                        u0d0, 
                                        d0mck, 
                                        u0mck, 
                                        a0mck, 
                                        ud, 
                                        dmck, 
                                        sigma_s, 
                                        areaf, 
                                        True)
            # Finish the wall clock on checkpoint export
            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
            checkpointExpTime = wallClockCheckpointExp.toc()
            # Starts the wall clock
            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
            wallClockAssignOldFuncSpace.tic()
            # Assign the old function spaces
            if self.solving_method == 'STVK':
                u0d0.assign(ud)

            elif self.solving_method == 'MCK':
                if ((self.iQuiet) and (self.iMUIFetchValue == False) and (self.iUseRBF == False)):
                    pass
                else:
                    amck = self.AMCK (  dmck.vector(),
                                        d0mck.vector(),
                                        u0mck.vector(),
                                        a0mck.vector(),
                                        beta_gam)

                    umck = self.UMCK (  amck,
                                        u0mck.vector(),
                                        a0mck.vector(),
                                        gamma_gam)

                    a0mck.vector()[:] = amck
                    u0mck.vector()[:] = umck
                    d0mck.vector()[:] = dmck.vector()
            # Finish the wall clock on assign old function space
            if (sync == True): self.LOCAL_COMM_WORLD.Barrier()
            assignOldFuncSpaceTime = wallClockAssignOldFuncSpace.toc()
            # Move to next time step
            i_sub_it = 1
            t += self.dt
            # Finish the wall clock
            simtimePerStep = wallClockPerStep.toc()
            if self.MPI_Get_Rank() == 0:
                print ("\n")
                print ("{FENICS} Simulation time per step: %g [s] at timestep: %i" % (simtimePerStep, n_steps))
            
        #===========================================
        #%% Calculate wall time
        #===========================================

        # Wait for the other solver
        self.ifaces3d["threeDInterface0"].barrier(t_sub_it)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#