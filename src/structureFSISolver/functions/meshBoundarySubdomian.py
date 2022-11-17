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
    
    @file meshBoundarySubdomian.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    mesh boundary and sub-domain related file of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________
from dolfin import *


class meshBoundarySubdomian:

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

