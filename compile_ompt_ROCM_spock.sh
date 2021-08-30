#!/bin/bash
module load rocm rocm-compiler 
make API=OMPT COMPILER=ROCM

