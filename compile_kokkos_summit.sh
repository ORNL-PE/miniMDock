#!/bin/bash


 module load pgi/19.9 cuda/10.1.243

 export KOKKOS_SRC_DIR=/ccs/home/mathit/kokkos
 export KOKKOS_INC_PATH=/ccs/home/mathit/kokkos/install/include/
 export KOKKOS_LIB_PATH=/ccs/home/mathit/kokkos/install/lib/
 
 make API=KOKKOS DEVICE=GPU CARD=NVIDIA

