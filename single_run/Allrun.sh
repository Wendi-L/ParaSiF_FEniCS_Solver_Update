#!/bin/sh

# Run from this directory
cd ${0%/*} || exit 1

export PYTHONPATH=$HOME/working/FEniCS_X_transfer/GitHub_repo/src:$PYTHONPATH
export PYTHONPATH=$HOME/working/apps/MUI/MUI-1.2/wrappers/Python:$PYTHONPATH

domainStructure=${PWD}/structureDomain

# Ranks set to each domain
numProcsStructure=1

solverStructure=structureDomainRun.py


# parallel run
mpirun -np ${numProcsStructure} -wdir ${domainStructure} python3 -m mpi4py ${solverStructure}

echo "Done"

# ----------------------------------------------------------------- end-of-file