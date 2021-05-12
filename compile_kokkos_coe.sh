#!/bin/bash


 ROCMV=3.7.0

 module load rocm/${ROCMV} cuda10.2/toolkit

 export KOKKOS_SRC_DIR=/home/users/coe0179/kokkos
 export KOKKOS_INC_PATH=/home/users/coe0179/kokkos/include/
 export KOKKOS_LIB_PATH=/home/users/coe0179/kokkos/lib64/

 make API=KOKKOS DEVICE=GPU  CARD=AMD

