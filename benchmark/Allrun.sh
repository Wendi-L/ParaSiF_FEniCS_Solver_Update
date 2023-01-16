#!/bin/sh

# Run from this directory
cd ${0%/*} || exit 1

export PYTHONPATH=${PWD}../src:$PYTHONPATH
export PYTHONPATH=${PWD}../thirdParty/MUI/wrappers/Python:$PYTHONPATH

domainFluid=${PWD}/dummyOF
domainStructure=${PWD}/structureDomain

# Ranks set to each domain
numProcsFluid=1
numProcsStructure=1

solverFluid=./PUSHER_FETCHER_1
solverStructure=structureDomainRun.py

cd ${domainFluid}
make
cd ..

# parallel run
mpirun -np ${numProcsFluid} -wdir ${domainFluid} ${solverFluid} -parallel -coupled :\
       -np ${numProcsStructure} -wdir ${domainStructure} python3 -m mpi4py ${solverStructure}

echo "Done"

# ----------------------------------------------------------------- end-of-file