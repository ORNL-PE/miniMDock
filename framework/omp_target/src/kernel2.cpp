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

void gpu_sum_evals(uint32_t blocks, uint32_t threadsPerBlock, GpuData& cData)
{
    int sum_evals = 0;
    int teamIdx = 0;
    #pragma omp target
    #pragma omp teams num_teams(blocks) thread_limit(threadsPerBlock)\
    //reduction(+:sum_evals)
    {
    #pragma omp distribute
    for (int Idx = 0; Idx < cData.dockpars.pop_size; Idx += threadsPerBlock ) {
    
        teamIdx = omp_get_team_num();
        //int sum_evals = 0;
    	int* pEvals_of_new_entities = cData.pMem_evals_of_new_entities + teamIdx * cData.dockpars.pop_size;
        #pragma omp parallel for reduction(+:sum_evals)	
        for (int entity_counter = Idx; entity_counter < cData.dockpars.pop_size; entity_counter++) 
        {
	    sum_evals += pEvals_of_new_entities[entity_counter];
	}
   }
	if(omp_get_thread_num() == 0) cData.pMem_gpu_evals_of_runs[teamIdx] += sum_evals;
   }// End for teams
}
