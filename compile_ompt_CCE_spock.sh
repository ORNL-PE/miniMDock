#!/bin/bash
module load rocm craype-accel-amd-gfx908 cce
make API=OMPT COMPILER=CC

