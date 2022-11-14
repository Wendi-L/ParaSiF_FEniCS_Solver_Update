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
        k = Constant(self.dt)

        # Time lists
        times    = []
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

        if self.rank == 0:
            print ("\n")
            print ("{FENICS} One-step theta: ", float(theta))
            print ("\n")

        #===========================================
        #%% Define function spaces
        #===========================================

        if self.rank == 0: print ("{FENICS} Creating function spaces ...   ")

        V_ele     =     VectorElement("Lagrange", mesh.ufl_cell(), self.deg_fun_spc) # Displacement & Velocity Vector element

        Q         =     FunctionSpace(mesh, "Lagrange", self.deg_fun_spc)            # Function space with updated mesh
        VV        =     FunctionSpace(mesh, MixedElement([V_ele, V_ele]))            # Mixed (Velocity (w) & displacement (d)) function space
        V         =     VectorFunctionSpace(mesh, "Lagrange", self.deg_fun_spc)
        T_s_space =     TensorFunctionSpace(mesh, 'Lagrange', self.deg_fun_spc)      # Define nth order structure function spaces

        if self.rank == 0: print ("{FENICS} Done with creating function spaces")

        #===========================================
        #%% Define functions, test functions and trail functions
        #===========================================

        if self.rank == 0: print ("{FENICS} Creating functions, test functions and trail functions ...   ", end="", flush=True)

        # Trial functions
        du, dd = TrialFunctions(VV)     # Trial functions for velocity and displacement
        ddmck  = TrialFunction(V)        # Trial function for displacement by MCK solving method

        # Test functions
        psi, phi = TestFunctions(VV)    # Test functions for velocity and displacement
        chi      = TestFunction(V)           # Test function for displacement by MCK solving method

        # Functions at present time step
        ud   = Function(VV)               # Functions for velocity and displacement
        u, d = split(ud)                # Split velocity and displacement functions
        dmck = Function(V)              # Function for displacement by MCK solving method

        # Functions at previous time step
        u0d0   = Function(VV)             # Functions for velocity and displacement
        u0, d0 = split(u0d0)            # Split velocity and displacement functions
        d0mck  = Function(V)             # Function for displacement by MCK solving method
        u0mck  = Function(V)             # Function for velocity by MCK solving method
        a0mck  = Function(V)             # Function for acceleration by MCK solving method

        # Define structure traction
        sigma_s = Function(T_s_space)   # Structure traction normal to structure

        self.Load_Functions_Continue_Run(u0d0,d0mck,u0mck,a0mck,ud,dmck,sigma_s)

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
        if self.solving_method == 'STVK':
            bc1,bc2 = self.dirichletBCs.DirichletMixedBCs(VV,boundaries,1)
            bcs = [bc1,bc2]
        elif self.solving_method == 'MCK':
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

        self.facets_area_define(mesh,
                                Q,
                                boundaries,
                                dofs_fetch_list,
                                gdim)

        #===========================================
        #%% Prepare post-process files
        #===========================================

        if self.rank == 0: print ("{FENICS} Preparing post-process files ...   ", end="", flush=True)

        disp_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/displacement.pvd")
        stress_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/stress.pvd")
        traction_file = File(self.LOCAL_COMM_WORLD, self.outputFolderPath + "/surface_traction_structure.pvd")

        if self.rank == 0: print ("Done")

        #===========================================
        #%% Define the variational FORM 
        #%% and 
        #%% Jacobin functions of structure
        #===========================================

        if self.solving_method == 'STVK':

            if self.rank == 0: print ("{FENICS} Defining variational FORM and Jacobin functions ...   ", end="", flush=True)

            # Define the traction terms of the structure variational form
            tF = dot(self.F_(d,gdim).T, self.tF_apply)
            tF_ = dot(self.F_(d0,gdim).T, self.tF_apply)

            # Define the transient terms of the structure variational form
            Form_s_T = (1/k)*self.rho_s*inner((u-u0), psi)*dx
            Form_s_T += (1/k)*inner((d-d0), phi)*dx

            # Define the stress terms and convection of the structure variational form
            if self.iNonLinearMethod:
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

            Jaco = derivative(Form_s, ud) # Define Jacobin functions

            if self.rank == 0: print ("Done")

        elif self.solving_method == 'MCK':
            if self.rank == 0: print ("{FENICS} Defining variational FORM functions ...   ", end="", flush=True)
            # Define the traction terms of the structure variational form
            tF = dot(chi, self.tF_apply)

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

            if self.rank == 0: print ("Done")

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
                                mesh,
                                u0d0,
                                d0mck,
                                u0mck,
                                a0mck,
                                ud,
                                dmck,
                                sigma_s,
                                self.areaf,
                                False)

        #===========================================
        #%% Define MUI samplers and commit ZERO step
        #===========================================

        self.MUI_Sampler_Define(Q,
                                gdim,
                                dofs_fetch_list,
                                dofs_push_list,
                                xyz_fetch,
                                t_step)

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

            if self.rank == 0: 
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

                if self.rank == 0: 
                    print ("\n")
                    print ("{FENICS} sub-iteration: ", i_sub_it)
                    print ("{FENICS} total sub-iterations to now: ", t_sub_it)

                self.Traction_Assign(xyz_fetch,
                                    dofs_fetch_list,
                                    self.t_sampler,
                                    self.s_sampler,
                                    t_sub_it,
                                    self.areaf_vec)

                if (not ((self.iContinueRun) and (n_steps == 1))):
                    if self.solving_method == 'MCK':
                        # Assemble linear form
                        Linear_Assemble = assemble(Linear_Form)
                        [bc.apply(Linear_Assemble) for bc in bcs]
                    # Solving the structure functions inside the time loop
                    solver.solve()

                    if self.solving_method == 'MCK':
                        force_X = dot(self.tF_apply, self.X_direction_vector())*ds(2)
                        force_Y = dot(self.tF_apply, self.Y_direction_vector())*ds(2)
                        force_Z = dot(self.tF_apply, self.Z_direction_vector())*ds(2)
                    else:
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

                if self.solving_method == 'STVK':
                    # Split function spaces
                    u,d = ud.split(True)

                    # Compute and print the displacement of monitored point
                    self.print_Disp (self.LOCAL_COMM_WORLD, d)

                    # MUI Push internal points and commit current steps
                    #if (self.iMUICoupling) and (len(xyz_push)!=0):
                    if (self.iMUICoupling):
                        self.MUI_Push(  xyz_push,
                                        dofs_push_list, 
                                        d, 
                                        t_sub_it)
                    else:
                        pass

                elif self.solving_method == 'MCK':
                    # Compute and print the displacement of monitored point
                    self.print_Disp (self.LOCAL_COMM_WORLD, dmck)
                    # MUI Push internal points and commit current steps
                    #if (self.iMUICoupling) and (len(xyz_push)!=0):
                    if (self.iMUICoupling):
                        self.MUI_Push(  xyz_push,
                                        dofs_push_list, 
                                        dmck, 
                                        t_sub_it)

                    else:
                        pass

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
                                            self.areaf,
                                            True)

            elif self.solving_method == 'MCK':
                self.Move_Mesh(V, dmck, d0mck, mesh)
                if (not (self.iQuiet)):
                    self.Export_Disp_vtk(   self.LOCAL_COMM_WORLD,
                                            n_steps,
                                            t,
                                            mesh,
                                            gdim,
                                            V,
                                            self.tF_apply,
                                            dmck,
                                            stress_file,
                                            disp_file,
                                            traction_file)

                if (not (self.iQuiet)):
                    self.Export_Disp_txt(   self.LOCAL_COMM_WORLD,
                                            dmck,
                                            self.outputFolderPath)

            if (not (self.iQuiet)):
                self.Checkpoint_Output( self.LOCAL_COMM_WORLD,
                                        self.outputFolderPath,
                                        t,
                                        mesh,
                                        mesh,
                                        u0d0,
                                        d0mck,
                                        u0mck,
                                        a0mck,
                                        ud,
                                        dmck,
                                        sigma_s,
                                        self.areaf,
                                        True)

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

            # Move to next time step
            i_sub_it = 1
            t += self.dt
            # Finish the wall clock
            simtimePerStep = wallClockPerStep.toc()
            if self.rank == 0:
                print ("\n")
                print ("{FENICS} Simulation time per step: %g [s] at timestep: %i" % (simtimePerStep, n_steps))

        #===========================================
        #%% Calculate wall time
        #===========================================

        # Wait for the other solver
        self.ifaces3d["threeDInterface0"].barrier(t_sub_it)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#