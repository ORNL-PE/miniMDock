#!/bin/bash


 module load cuda
 export GPU_LIBRARY_PATH=${OLCF_CUDA_ROOT}/lib64
 export GPU_INCLUDE_PATH=${OLCF_CUDA_ROOT}/include

 export FLAVOR=".clang"
 make API=CUDA

