
#ifndef CALCENERGY_H
#define CALCENERGY_H

#pragma omp declare target
void gpu_calc_energy(
    float* pGenotype,
    float& energy,
    int& run_id,
    float3* calc_coords,
    int idx,
    uint32_t blockDim,
    GpuData& cData,
    GpuDockparameters dockpars
);
#pragma omp end declare target

#endif
