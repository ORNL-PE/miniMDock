#!/bin/bash

 ROCMV=3.7.0
 CARD=hcc

 module load rocm/${ROCMV} cuda10.2/toolkit
 export HIP_PLATFORM=${CARD}
 #export HCC_HOME=/opt/rocm-3.3.0/hcc

 export GPU_LIBRARY_PATH=/opt/rocm-${ROCMV}/lib  
 export GPU_INCLUDE_PATH=/opt/rocm-${ROCMV}/include

 make API=HIP CARD=AMD

