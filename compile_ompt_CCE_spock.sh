#!/bin/bash
module load PrgEnv-cray craype-accel-amd-gfx908 rocm
make API=OMPT COMPILER=CC

