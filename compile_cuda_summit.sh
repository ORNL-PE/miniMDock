#!/bin/bash


 module load cuda
 export GPU_LIBRARY_PATH=${CUDAPATH}/lib64
 export GPU_INCLUDE_PATH=${CUDAPATH}/include

 make API=CUDA

