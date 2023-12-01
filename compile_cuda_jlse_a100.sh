#!/bin/bash


 module use /soft/modulefiles/
 module use /soft/restricted/CNDA/modulefiles/
 module load cuda
 #ml cuda/12.3.0 
 
 export TARGET=80
 export GPU_LIBRARY_PATH=${CUDA_PATH}/lib64
 export GPU_INCLUDE_PATH=${CUDA_PATH}/include

 make API=CUDA

