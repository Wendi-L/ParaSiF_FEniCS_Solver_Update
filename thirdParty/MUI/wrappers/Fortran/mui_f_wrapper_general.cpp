/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2021 Y. H. Tang, S. Kudo, X. Bian, Z. Li, G. E. Karniadakis, *
*                    S. M. Longshaw, W. Liu                                  *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file mui_f_wrapper_general.cpp
 * @author S. M. Longshaw (derived from original 3D wrapper by S. Kudo)
 * @date 25 November 2021
 * @brief C interface for Fortran wrapper for general MUI functions with
 *        no associated dimensionality
 */

// Main MUI header include (contains any other needed includes)
#include "../../src/mui.h"
#include "mpi.h"

extern "C" {

// Function to split MPI communicator and return new, local communicator
void mui_mpi_split_by_app_f(MPI_Comm **communicator) {
	*communicator = reinterpret_cast<MPI_Comm*>(mui::mpi_split_by_app());
}

// Function to split MPI communicator and return new, local communicator using threaded MPI init
void mui_mpi_split_by_app_threaded_f(MPI_Comm **communicator, int *argc, char ***argv, int *threadType, int **thread_support) {
	*communicator = reinterpret_cast<MPI_Comm*>(mui::mpi_split_by_app(*argc, *argv, *threadType, *thread_support));
}

void mui_mpi_get_size_f(MPI_Comm *communicator, int *size) {
	MPI_Comm_size(*communicator, size);
}

void mui_mpi_get_rank_f(MPI_Comm *communicator, int *rank) {
	MPI_Comm_rank(*communicator, rank);
}

}
