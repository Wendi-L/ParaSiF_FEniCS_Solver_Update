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

    @file hyperElastic.py

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

class hyperElastic:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Main solver function
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def hyperElasticSolve(self):

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

        # Time step constants
        k = Constant(self.dt())

        # Time lists
        times    = []
        t_sub_it = 0

        # One-step theta value
        theta = Constant(self.thetaOS())

        if self.rank == 0:
            print ("\n")
            print ("{FENICS} One-step theta: ", float(theta))
            print ("\n")

        #===========================================
        #%% Define function spaces
        #===========================================

        if self.rank == 0: print ("{FENICS} Creating function spaces ...   ")

        V_ele     =     VectorElement("Lagrange", mesh.ufl_cell(), self.deg_fun_spc()) # Displacement & Velocity Vector element

        Q         =     FunctionSpace(mesh, "Lagrange", self.deg_fun_spc())            # Function space with updated mesh
        VV        =     FunctionSpace(mesh, MixedElement([V_ele, V_ele]))            # Mixed (Velocity (w) & displacement (d)) function space
        V         =     VectorFunctionSpace(mesh, "Lagrange", self.deg_fun_spc())
        T_s_space =     TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc())      # Define nth order structure function spaces

        if self.rank == 0: print ("{FENICS} Done with creating function spaces")

        #=======================================================
        #%% Define functions, test functions and trail functions
        #=======================================================

        if self.rank == 0: print ("{FENICS} Creating functions, test functions and trail functions ...   ", end="", flush=True)

        # Test functions
        psi, phi = TestFunctions(VV)    # Test functions for velocity and displacement

        # Functions at present time step
        ud   = Function(VV)               # Functions for velocity and displacement
        u, d = split(ud)                # Split velocity and displacement functions

        # Functions at previous time step
        u0d0   = Function(VV)             # Functions for velocity and displacement
        u0, d0 = split(u0d0)            # Split velocity and displacement functions

        # Define structure traction
        sigma_s = Function(T_s_space)   # Structure traction normal to structure

        self.Load_Functions_Continue_Run_Nonlinear(u0d0,ud,sigma_s)

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
        bc1,bc2 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
        bcs = [bc1,bc2]
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

        if self.rank == 0: print ("{FENICS} Defining variational FORM and Jacobin functions ...   ", end="", flush=True)

        # Define the traction terms of the structure variational form
        tF = dot(self.F_(d,gdim).T, self.tF_apply)
        tF_ = dot(self.F_(d0,gdim).T, self.tF_apply)

        # Define the transient terms of the structure variational form
        Form_s_T = (1/k)*self.rho_s()*inner((u-u0), psi)*dx
        Form_s_T += (1/k)*inner((d-d0), phi)*dx

        # Define the stress terms and convection of the structure variational form
        if self.iNonLinearMethod():
            if self.rank == 0: print ("{FENICS} [Defining non-linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by the constitutive law of hyper-elastic St. Vernant-Kirchhoff material model (non-linear relation). Valid for large deformations but small strain] ...   ", end="", flush=True)
            Form_s_SC = inner(theta * self.Piola_Kirchhoff_fst(d,gdim) + (1 - theta) *
                        self.Piola_Kirchhoff_fst(d0,gdim), grad(psi)) * dx
            Form_s_SC -= inner(theta*u + (1-theta)*u0, phi ) * dx
        else:
            if self.rank == 0: print ("{FENICS} [Defining linear stress-strain relation: Define the First Piola-Kirchhoff stress tensor by Hooke's law (linear relation). Valid for small-scale deformations only] ...   ", end="", flush=True)
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

        # Define Jacobin functions
        Jaco = derivative(Form_s, ud)

        if self.rank == 0: print ("Done")

        #===========================================
        #%% Initialize solver
        #===========================================

        problem = NonlinearVariationalProblem(Form_s, ud, bcs=bcs, J=Jaco)
        solver = NonlinearVariationalSolver(problem)

        info(solver.parameters, False)
        if self.nonlinear_solver() == "newton":
            solver.parameters["nonlinear_solver"]= self.nonlinear_solver()
            solver.parameters["newton_solver"]["absolute_tolerance"] = self.prbAbsolute_tolerance()
            solver.parameters["newton_solver"]["relative_tolerance"] = self.prbRelative_tolerance()
            solver.parameters["newton_solver"]["maximum_iterations"] = self.prbMaximum_iterations()
            solver.parameters["newton_solver"]["relaxation_parameter"] = self.prbRelaxation_parameter()
            solver.parameters["newton_solver"]["linear_solver"] = self.prbsolver()
            solver.parameters["newton_solver"]["preconditioner"] = self.prbpreconditioner()
            solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance()
            solver.parameters["newton_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance()
            solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations()
            solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence()
            solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess()
            solver.parameters["newton_solver"]["krylov_solver"]['error_on_nonconvergence'] = self.error_on_nonconvergence()
        elif self.nonlinear_solver() == "snes":
            solver.parameters['nonlinear_solver'] = self.nonlinear_solver()
            solver.parameters['snes_solver']['line_search'] = self.lineSearch()
            solver.parameters['snes_solver']['linear_solver'] = self.prbsolver()
            solver.parameters['snes_solver']['preconditioner'] = self.prbpreconditioner()
            solver.parameters['snes_solver']['absolute_tolerance'] = self.prbAbsolute_tolerance()
            solver.parameters['snes_solver']['relative_tolerance'] = self.prbRelative_tolerance()
            solver.parameters['snes_solver']['maximum_iterations'] = self.prbMaximum_iterations()
            solver.parameters['snes_solver']['report'] = self.show_report()
            solver.parameters['snes_solver']['error_on_nonconvergence'] = self.error_on_nonconvergence()
            solver.parameters["snes_solver"]["krylov_solver"]["absolute_tolerance"] = self.krylov_prbAbsolute_tolerance()
            solver.parameters["snes_solver"]["krylov_solver"]["relative_tolerance"] = self.krylov_prbRelative_tolerance()
            solver.parameters["snes_solver"]["krylov_solver"]["maximum_iterations"] = self.krylov_maximum_iterations()
            solver.parameters["snes_solver"]["krylov_solver"]["monitor_convergence"] = self.monitor_convergence()
            solver.parameters["snes_solver"]["krylov_solver"]["nonzero_initial_guess"] = self.nonzero_initial_guess()
        else:
            sys.exit("{FENICS} Error, nonlinear solver value not recognized")

        #===========================================
        #%% Setup checkpoint data
        #===========================================

        self.Checkpoint_Output_Nonlinear((t-self.dt()), mesh, u0d0, ud, sigma_s, False)

        #===========================================
        #%% Define MUI samplers and commit ZERO step
        #===========================================

        self.MUI_Sampler_Define(Q, gdim, dofs_fetch_list, dofs_push_list, xyz_push, xyz_fetch, d, t_step)

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
                    # Solving the structure functions inside the time loop
                    solver.solve()

                    force_X = dot(tF, self.X_direction_vector())*ds(2)
                    force_Y = dot(tF, self.Y_direction_vector())*ds(2)
                    force_Z = dot(tF, self.Z_direction_vector())*ds(2)

                    f_X_a = assemble(force_X)
                    f_Y_a = assemble(force_Y)
                    f_Z_a = assemble(force_Z)

                    print ("{FENICS} Total Force_X on structure: ", f_X_a, " at self.rank ", self.rank)
                    print ("{FENICS} Total Force_Y on structure: ", f_Y_a, " at self.rank ", self.rank)
                    print ("{FENICS} Total Force_Z on structure: ", f_Z_a, " at self.rank ", self.rank)

                else:
                    pass

                # Split function spaces
                u,d = ud.split(True)

                # Compute and print the displacement of monitored point
                self.print_Disp(d)

                # MUI Push internal points and commit current steps
                if (self.iMUICoupling()):
                    if (len(xyz_push)!=0):
                        self.MUI_Push(xyz_push, dofs_push_list, d, t_sub_it)
                    else:
                        self.MUI_Commit_only(t_sub_it)
                else:
                    pass

                # Increment of sub-iterations
                i_sub_it += 1

            # Split function spaces
            u,d = ud.split(True)
            u0,d0 = u0d0.split(True)

            # Mesh motion
            self.Move_Mesh(V, d, d0, mesh)

            # Data output
            if (not (self.iQuiet())):
                self.Export_Disp_vtk(n_steps, t, mesh, gdim, V, d)
                self.Export_Disp_txt(d)
                self.Checkpoint_Output_Nonlinear(t, mesh, u0d0, ud, sigma_s, False)

            # Assign the old function spaces
            u0d0.assign(ud)

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