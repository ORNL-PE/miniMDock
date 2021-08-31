#!/bin/bash


 module load cuda llvm/13.0.0-latest
 export GPU_LIBRARY_PATH=${OLCF_CUDA_ROOT}/lib64
 export GPU_INCLUDE_PATH=${OLCF_CUDA_ROOT}/include

 export FLAVOR=".clang"
 make API=CUDA

