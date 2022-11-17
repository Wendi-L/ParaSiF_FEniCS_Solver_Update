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
    
    @file facetAreas.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    facet Areas file of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
from dolfin import *
import numpy as np
import math

class facetAreas:

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#