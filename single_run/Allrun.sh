#!/bin/sh

# Run from this directory
cd ${0%/*} || exit 1

export PYTHONPATH=${PWD}/../src:$PYTHONPATH
export PYTHONPATH=${PWD}/../thirdParty/MUI/wrappers/Python:$PYTHONPATH

domainStructure=${PWD}/structureDomain

# Ranks set to each domain
numProcsStructure=1

solverStructure=structureDomainRun.py


# parallel run
mpirun -np ${numProcsStructure} -wdir ${domainStructure} python3 -m mpi4py ${solverStructure} 2>&1 | tee output.log

echo "Done"

# ----------------------------------------------------------------- end-of-file