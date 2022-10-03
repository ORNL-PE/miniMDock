#!/bin/bash

 #ICPC on arcticus
 module use /soft/restricted/CNDA/modulefiles/
 module load oneapi

 make API=OMPT COMPILER=icpc NUMWI=64
