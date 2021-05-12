#!/bin/bash


 module load cuda10.2
 export GPU_LIBRARY_PATH=/cm/shared/apps/cuda10.2/toolkit/10.2.89/targets/x86_64-linux/lib  
 export GPU_INCLUDE_PATH=/cm/shared/apps/cuda10.2/toolkit/10.2.89/targets/x86_64-linux/include

# export GPU_LIBRARY_PATH=/cm/shared/apps/cuda10.0/toolkit/10.0.130/targets/x86_64-linux/lib   
# export GPU_INCLUDE_PATH=/cm/shared/apps/cuda10.0/toolkit/10.0.130/targets/x86_64-linux/include
 make API=CUDA

