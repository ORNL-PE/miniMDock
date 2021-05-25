#!/bin/bash
# Link OpenMP target code files for compilation

ln -sf performdocking.h.Ompt host/inc/performdocking.h
ln -sf performdocking.cpp.Ompt host/src/performdocking.cpp
ln -sf GpuData.h.Ompt host/inc/GpuData.h
