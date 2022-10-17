# ParaSiF FEniCS Solver Update

## Updates

17th October 2022: Created main branch.

## Overview

This is a working space on the mini-project of ParaSiF FEniCS solver update. 

It includes two parts:
* Code clean-up/optimisation.
This part aims to clean-up and optimise the ParaSiF FEniCS in terms of its ease of read for collebration and long term maintaince. It will based on FEniCS-v2019.1.0.

* FEniCS-v2019 -> FEniCS-X transfer.
This part aims to transfer the code from FEniCS-v2019.1.0 to FEniCS-X-v0.5.1+, by using the updated APIs and new features in FEniCS-X.

## Branching Strategy

* main

Contains the original code and benchmark case during the work. It will contain the final code to be merged into ParaSiF once this piece of work finished.

* FEniCS-v2019

This is the branch on the code clean-up/optimisation activities.

* FEniCS-X

This is the branch on the FEniCS-X transfer. It will be created once code clean-up/optimisation activities done.

## Folder Structure

* src

It contains the source code on the FEniCS elasticity solver.

* benchmark

It contains the benchmark case on the FEniCS elasticity solver, based on test case 10.2. Three-dimensional cantilever of Slone et al. 2003:

Slone, A. K., C. Bailey, and M. Cross. "Dynamic solid mechanics using finite volume methods." Applied mathematical modelling 27.2 (2003): 69-87.

There are two sub-folders: 

** dummyOF/
It contains a simple C++ script as a dummy fluid solver. It's function is to pass the node forces to the structure domain.  

** structureDomain/
It contains the input files of FEniCS solver, includes the main file (structureDomainRun.py), BCs, SubDomains and control parameters. A new subfolder "structureResults" will be generated to collect the results from the FEniCS solver. 

## Install

The dependencies of this code are (correct version of) FEniCS and MUIv1.2.

* Step One: Install FEniCS (FEniCS v2019.1.0 for "FEniCS-v2019" branch and FEniCS-X v0.5.1 for "FEniCS-X" branch).

Following FEniCS homepage (https://fenicsproject.org/) for the installation precedure.

* Step Two: Obtain MUI v1.2 

Clone MUI and checkout v1.2 by

```
git clone https://github.com/MxUI/MUI.git

cd MUI && git checkout 1.2

```

* Step Three: Install MUI Python wrapper by
```
cd wrappers/Python
make USE_RBF=1 INC_EIGEN=/path/to/eigen package
make pip-install
```

After these three steps, the installation has been finished.

## Run the Benchmark case

* Step One: go to the benchmark folder.


* Step Two: Correct addresses.

Open file "Allrun"; In Line 6 & Line 7, correct the addresses on the FEniCS solver source and MUI Python wrapper, respectively.

Open file "dummyOF/Makefile". In Line 7, correct the address on MUI source folder.


* Step Three: Run the case.

Go to the root of this benchmark case and execute the run script by
```
./Allrun
```
To clean-up the results from previous run, execute the clean script before run
```
./Allclean
```

* Step Four: Check results. 

Once the simulation finished, there will be a PNG file generated in the root of this benchmark case. Open it to check the results compared with published results from Slone et al. 2003.