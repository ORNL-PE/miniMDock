#!/bin/bash


 module load cuda
 export GPU_LIBRARY_PATH=/sw/summit/cuda/10.1.243/lib64
 export GPU_INCLUDE_PATH=/sw/summit/cuda/10.1.243/include

 make API=CUDA

