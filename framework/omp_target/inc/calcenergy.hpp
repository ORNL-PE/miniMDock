
#ifndef CALCENERGY_H
#define CALCENERGY_H

#include "typedefine.h"


#pragma omp declare target
float gpu_calc_energy(
//void gpu_calc_energy(
    float* pGenotype,
 //   float& energy,
    int& run_id,
    float3struct* calc_coords,
    int idx,
    uint32_t blockDim,
    GpuData& cData,
    GpuDockparameters dockpars
);
#pragma omp end declare target

#endif
