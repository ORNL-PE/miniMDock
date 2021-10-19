#!/bin/bash

 module use /sw/summit/modulefiles/ums/stf010/Core
 module load llvm/14.0.0-20211016 cuda
 #module load llvm/14.0.0-latest cuda
 #module load llvm/14.0.0-20210909 cuda

 export GPU_PATH=${OLCF_CUDA_ROOT}

 make API=OMPT COMPILER=llvm NUMWI=128

