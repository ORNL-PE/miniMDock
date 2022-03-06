#!/bin/bash
 module use /sw/spock/ums/eiw/modulefiles
 module load rocm
 module load aomp/14.0.1-20220211

 make API=OMPT COMPILER=ROCM NUMWI=64
