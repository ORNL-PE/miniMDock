
#ifndef CALCENERGY_H
#define CALCENERGY_H

void gpu_calc_energy(
    float* pGenotype,
    float& energy,
    int& run_id,
    float3* calc_coords,
    float* pFloatAccumulator,
    int idx,
    uint32_t blockDim
);

#endif
