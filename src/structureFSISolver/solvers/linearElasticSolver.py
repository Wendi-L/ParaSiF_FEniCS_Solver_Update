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

    @file linearElastic.py

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
        #%% Time marching parameters define
        #===========================================

        t        = self.Start_Time
        t_step   = self.Time_Steps
        i_sub_it = self.Start_Number_Sub_Iteration

        #===========================================
        #%% Solid Mesh input/generation
        #===========================================

        mesh = self.Mesh_Generation()
        gdim = self.Get_Grid_Dimension(mesh)
        N    = self.Get_Face_Narmal(mesh)

        #===========================================
        #%% Define coefficients
        #===========================================

        # Time lists
        times    = []
        t_sub_it = 0

        # Rayleigh damping coefficients
        alpha_rdc = Constant(self.alpha_rdc())
        beta_rdc  = Constant(self.beta_rdc())

        # Generalized-alpha method parameters
        # alpha_m_gam >= alpha_f_gam >= 0.5 for a better performance
        alpha_m_gam = Constant(self.alpha_m_gam())
        alpha_f_gam = Constant(self.alpha_f_gam())
        gamma_gam   = Constant((1./2.) + alpha_m_gam - alpha_f_gam)
        beta_gam    = Constant((1./4.) * (gamma_gam + (1./2.))**2)

        #===========================================
        #%% Define function spaces
        #===========================================

        if self.rank == 0: print ("{FENICS} Creating function spaces ...   ")
        Q         =     FunctionSpace(mesh, "Lagrange", self.deg_fun_spc())            # Function space with updated mesh
        V         =     VectorFunctionSpace(mesh, "Lagrange", self.deg_fun_spc())

        if self.rank == 0: print ("{FENICS} Done with creating function spaces")

        #=======================================================
        #%% Define functions, test functions and trail functions
        #=======================================================

        if self.rank == 0: print ("{FENICS} Creating functions, test functions and trail functions ...   ", end="", flush=True)

        # Trial functions
        ddmck  = TrialFunction(V)        # Trial function for displacement by MCK solving method

        # Test functions
        chi      = TestFunction(V)           # Test function for displacement by MCK solving method

        # Functions at present time step
        dmck = Function(V)              # Function for displacement by MCK solving method

        # Functions at previous time step
        d0mck  = Function(V)             # Function for displacement by MCK solving method
        u0mck  = Function(V)             # Function for velocity by MCK solving method
        a0mck  = Function(V)             # Function for acceleration by MCK solving method

        self.Load_Functions_Continue_Run_Linear(d0mck,u0mck,a0mck,dmck)

        if self.rank == 0: print ("Done")

        #===========================================
        #%% Define traction forces
        #===========================================

        self.Traction_Define(V)

        #===========================================
        #%% Define SubDomains and boundaries
        #===========================================

        boundaries = self.Boundaries_Generation_Fixed_Flex_Sym(mesh, gdim, V)

        ds = self.Get_ds(mesh, boundaries)

        #===========================================
        #%% Define boundary conditions
        #===========================================

        if self.rank == 0: print ("{FENICS} Creating 3D boundary conditions ...   ", end="", flush=True)
        bc1 = self.dirichletBCs.DirichletBCs(V,boundaries,1)
        bcs = [bc1]
        if self.rank == 0: print ("Done")

        #===========================================
        #%% Define DOFs and Coordinates mapping
        #===========================================  

        dofs_fetch_list = self.dofs_list(boundaries, Q, 2)

        xyz_fetch = self.xyz_np(dofs_fetch_list, Q, gdim)

        dofs_push_list = self.dofs_list(boundaries, Q, 2)

        xyz_push = self.xyz_np(dofs_push_list, Q, gdim)

        #===========================================
        #%% Define facet areas
        #===========================================

        self.facets_area_define(mesh, Q, boundaries, dofs_fetch_list, gdim)

        #===========================================
        #%% Prepare post-process files
        #===========================================

        self.Create_Post_Process_Files()

        #===========================================
        #%% Define the variational FORM
        #%% and
        #%% Jacobin functions of structure
        #===========================================

        if self.rank == 0: print ("{FENICS} Defining variational FORM functions ...   ", end="", flush=True)
        # Define the traction terms of the structure variational form
        tF = dot(chi, self.tF_apply)

        Form_s_Update_Acce = self.AMCK (ddmck, d0mck, u0mck, a0mck, beta_gam)
        Form_s_Update_velo = self.UMCK (Form_s_Update_Acce, u0mck, a0mck, gamma_gam)

        Form_s_Ga_Acce = self.Generalized_Alpha_Weights(Form_s_Update_Acce,a0mck,alpha_m_gam)
        Form_s_Ga_velo = self.Generalized_Alpha_Weights(Form_s_Update_velo,u0mck,alpha_f_gam)
        Form_s_Ga_disp = self.Generalized_Alpha_Weights(ddmck,d0mck,alpha_f_gam)
        Form_s_M_Matrix = self.rho_s() * inner(Form_s_Ga_Acce, chi) * dx
        Form_s_M_for_C_Matrix = self.rho_s() * inner(Form_s_Ga_velo, chi) * dx
        Form_s_K_Matrix = inner(self.elastic_stress(Form_s_Ga_disp,gdim), sym(grad(chi))) * dx
        Form_s_K_for_C_Matrix = inner(self.elastic_stress(Form_s_Ga_velo,gdim), sym(grad(chi))) * dx
        Form_s_C_Matrix = alpha_rdc * Form_s_M_for_C_Matrix + beta_rdc * Form_s_K_for_C_Matrix
        Form_s_F_Ext = tF * ds(2)

        Form_s = Form_s_M_Matrix + Form_s_C_Matrix + Form_s_K_Matrix - Form_s_F_Ext

        Bilinear_Form = lhs(Form_s)
        Linear_Form   = rhs(Form_s)

        if self.rank == 0: print ("Done")

        #===========================================
        #%% Initialize solver
        #===========================================

        if self.linear_solver() == 'LU':
            Bilinear_Assemble, Linear_Assemble = assemble_system(Bilinear_Form, Linear_Form, bcs)
            solver = LUSolver(Bilinear_Assemble, "mumps")
            solver.parameters["symmetric"] = True

        elif self.linear_solver() == 'LinearVariational':
            problem = LinearVariationalProblem(Bilinear_Form, Linear_Form, dmck, bcs)
            solver = LinearVariationalSolver(problem)
            # Set linear solver parameters
            solver.parameters["linear_solver"] = self.prbsolver()
            solver.parameters["preconditioner"] = self.prbpreconditioner()
            solver.parameters["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance()
            solver.parameters["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance()
            solver.parameters["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations()
            solver.parameters["krylov_solver"]["monitor_convergence"] = self.monitor_convergence()
            solver.parameters["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess()
        else:
            sys.exit("{FENICS} Error, linear solver value not recognized")

        #===========================================
        #%% Setup checkpoint data
        #===========================================

        self.Checkpoint_Output_Linear((t-self.dt()), mesh, d0mck, u0mck, a0mck, dmck, False)

        #===========================================
        #%% Define MUI samplers and commit ZERO step
        #===========================================

        self.MUI_Sampler_Define(Q, gdim, dofs_fetch_list, dofs_push_list, xyz_push, xyz_fetch, dmck, t_step)

        #===========================================
        #%% Define time loops
        #===========================================

        # Time-stepping
        while t <= self.T():

            # create an instance of the TicToc wall clock class
            wallClockPerStep = structureFSISolver.tictoc.TicToc()
            # Starts the wall clock
            wallClockPerStep.tic()

            # Update time list    
            times.append(t)
            n_steps = len(times)

            if self.rank == 0: 
                print ("\n")
                print ("{FENICS} Time: ", t, " [s]; Time Step Number: ", n_steps)

            # Change number of sub-iterations if needed
            if self.iChangeSubIter():
                if (t >= self.TChangeSubIter()):
                    present_num_sub_iteration = self.num_sub_iteration_new()
                else:
                    present_num_sub_iteration = self.num_sub_iteration()
            else:
                present_num_sub_iteration = self.num_sub_iteration()

            # Sub-iteration for coupling
            while i_sub_it <= present_num_sub_iteration:

                # Increment of total sub-iterations
                t_sub_it += 1

                if self.rank == 0: 
                    print ("\n")
                    print ("{FENICS} Sub-iteration Number: ", i_sub_it, " Total sub-iterations to now: ", t_sub_it)

                # Fetch and assign traction forces at present time step
                self.Traction_Assign(xyz_fetch, dofs_fetch_list, t_sub_it, n_steps, t)

                if (not ((self.iContinueRun()) and (n_steps == 1))):

                    # Assemble linear form
                    Linear_Assemble = assemble(Linear_Form)
                    [bc.apply(Linear_Assemble) for bc in bcs]
                    # Solving the structure functions inside the time loop
                    solver.solve()

                    force_X = dot(self.tF_apply, self.X_direction_vector())*ds(2)
                    force_Y = dot(self.tF_apply, self.Y_direction_vector())*ds(2)
                    force_Z = dot(self.tF_apply, self.Z_direction_vector())*ds(2)

                    f_X_a = assemble(force_X)
                    f_Y_a = assemble(force_Y)
                    f_Z_a = assemble(force_Z)

                    print ("{FENICS} Total Force_X on structure: ", f_X_a, " at self.rank ", self.rank)
                    print ("{FENICS} Total Force_Y on structure: ", f_Y_a, " at self.rank ", self.rank)
                    print ("{FENICS} Total Force_Z on structure: ", f_Z_a, " at self.rank ", self.rank)

                else:
                    pass

                # Compute and print the displacement of monitored point
                self.print_Disp(dmck)

                # MUI Push internal points and commit current steps
                if (self.iMUICoupling()):
                    if (len(xyz_push)!=0):
                        self.MUI_Push(xyz_push, dofs_push_list, dmck, t_sub_it)
                    else:
                        self.MUI_Commit_only(t_sub_it)
                else:
                    pass

                # Increment of sub-iterations
                i_sub_it += 1

            # Mesh motion
            self.Move_Mesh(V, dmck, d0mck, mesh)

            # Data output
            if (not (self.iQuiet())):
                self.Export_Disp_vtk(n_steps, t, mesh, gdim, V, dmck)
                self.Export_Disp_txt(dmck)
                self.Checkpoint_Output_Linear(t, mesh, d0mck, u0mck, a0mck, dmck, False)

            # Function spaces time marching

            amck = self.AMCK (dmck.vector(), d0mck.vector(), u0mck.vector(), a0mck.vector(), beta_gam)
            umck = self.UMCK (amck, u0mck.vector(), a0mck.vector(), gamma_gam)

            a0mck.vector()[:] = amck
            u0mck.vector()[:] = umck
            d0mck.vector()[:] = dmck.vector()

            # Sub-iterator counter reset
            i_sub_it = 1
            # Physical time marching
            t += self.dt()

            # Finish the wall clock
            simtimePerStep = wallClockPerStep.toc()
            if self.rank == 0:
                print ("\n")
                print ("{FENICS} Simulation time per step: %g [s] at timestep: %i" % (simtimePerStep, n_steps))

        #===============================================
        #%% MPI barrier to wait for all solver to finish
        #===============================================

        # Wait for the other solver
        if self.iMUICoupling():
            self.ifaces3d["threeDInterface0"].barrier(t_sub_it)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#