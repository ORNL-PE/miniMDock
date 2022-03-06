#!/bin/bash

 #LLVM 15 to use LTO features
 module use /sw/summit/modulefiles/ums/stf010/Core
 module load llvm/15.0.0-latest cuda/11.4.2

 #LLVM 14
 #module load llvm/14.0.0-latest cuda

 export GPU_PATH=${OLCF_CUDA_ROOT}

 make API=OMPT COMPILER=llvm NUMWI=64
