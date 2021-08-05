
#ifndef CALCENERGY_H
#define CALCENERGY_H

#include "typedefine.h"
#include "GpuData.h"


#pragma omp declare target
float gpu_calc_energy(
//void gpu_calc_energy(
    float* pGenotype,
 //   float& energy,
    const int& run_id,
    float3struct* calc_coords,
    const int idx,
    uint32_t blockDim,
    GpuData& cData,
    GpuDockparameters& dockpars
);
#pragma omp end declare target

#endif
