#!/bin/bash


 module load cuda
 module load nvhpc/21.7

 #export GPU_PATH=${OLCF_CUDA_ROOT}

 #export GPU_LIBRARY_PATH=/sw/summit/cuda/10.1.243/lib64
 #export GPU_INCLUDE_PATH=/sw/summit/cuda/10.1.243/include

 #module load xl/16.1.1-10

 make API=OMPT COMPILER=nvhpc

