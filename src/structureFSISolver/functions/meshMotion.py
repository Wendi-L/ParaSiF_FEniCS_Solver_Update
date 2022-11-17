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
    
    @file meshMotion.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    mesh Motion file of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
from dolfin import *

class meshMotion:

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#