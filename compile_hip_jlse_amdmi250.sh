#!/bin/bash

 module use /soft/modulefiles/
 module use /soft/restricted/CNDA/modulefiles/


 #ml rocm
 #ml rocm/5.7.0 
 ml rocm/5.5.0

 echo ${ROCM_PATH}

 export HIP_PLATFORM=amd
 #export HCC_HOME=/opt/rocm-3.3.0/hcc

 export GPU_LIBRARY_PATH=${ROCM_PATH}/lib  
 export GPU_INCLUDE_PATH=${ROCM_PATH}/include

 make API=HIP CARD=AMD

