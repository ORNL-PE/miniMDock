#!/bin/bash
# Link Kokkos code files for compilation

ln -sf performdocking.h.Kokkos host/inc/performdocking.h
ln -sf performdocking.cpp.Kokkos host/src/performdocking.cpp
#ln -sf dockingparams.hpp host/inc/GpuData.h
