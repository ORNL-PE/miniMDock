#!/bin/bash


 module load cuda/10.1.243
 module use /sw/summit/modulefiles/ums/stf010/Core
 #module load llvm/14.0.0-20210819
 module load llvm/13.0.0-latest

 export GPU_PATH=${OLCF_CUDA_ROOT}

 #export GPU_LIBRARY_PATH=/sw/summit/cuda/10.1.243/lib64
 #export GPU_INCLUDE_PATH=/sw/summit/cuda/10.1.243/include

 #module load xl/16.1.1-10

 make API=OMPT COMPILER=llvm

