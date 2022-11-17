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
    
    @file DOFCoordMapping.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    DOF Coord Mapping file of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
from dolfin import *
import numpy as np

class DOFCoordMapping:

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
