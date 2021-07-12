/*

miniAD is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/
#include "kernels.hpp"
#include "calcenergy.hpp"


void gpu_calc_initpop(	uint32_t pops_by_runs, 
			uint32_t threadsPerBlock, 
			float* pMem_conformations_current, 
			float* pMem_energies_current, 
			GpuData& cData,
			GpuDockparameters dockpars )
{

    #pragma omp target 
    #pragma omp teams distribute parallel for 
    for (int idx = 0; idx < pops_by_runs; idx++)
    {  
        float3struct calc_coords[MAX_NUM_OF_ATOMS];
	
	#pragma omp allocate(calc_coords) allocator(omp_pteam_mem_alloc)   
        float  energy = 0.0f;
        int teamIdx = omp_get_team_num();
        int threadIdx = omp_get_thread_num();
        int teamSize = omp_get_num_threads();
        int run_id = teamIdx / dockpars.pop_size;
        float* pGenotype = pMem_conformations_current + teamIdx * GENOTYPE_LENGTH_IN_GLOBMEM;
	gpu_calc_energy( pGenotype, energy, run_id, calc_coords, threadIdx, teamSize, cData, dockpars );
	
        // Write out final energy
        if (threadIdx == 0){
                pMem_energies_current[teamIdx] = energy;
                cData.pMem_evals_of_new_entities[teamIdx] = 1;
	 }

    }// End for a set of teams

}

