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
    
    @file lameParm.py
    
    @author W. Liu
    
    @brief This is a part of the Parallel Partitioned Multi-physical Simu-
    lation Framework.

    Lame parameters file of the structure code.
    Located in the src/CSM/FEniCS/V2019.1.0/structureFSISolver sub-folder
"""

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________

class lameParm:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #%% Define Lame parameters
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mu_s (self):
    # Define the Lamé's second parameter(shear modulus)
        return (self.E_s()/(2.0*(1.0 + self.nu_s())))

    # Define the Lamé's first parameter
    def lamda_s (self):
        return (2.0*(self.mu_s())*self.nu_s()/(1.0-2.0*self.nu_s()))
