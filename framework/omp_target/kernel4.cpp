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

#include<kernels.h>

//#define DEBUG_ENERGY_KERNEL4

void gpu_gen_and_eval_newpops(
    uint32_t nblocks,
    uint32_t threadsPerBlock,
    float* pMem_conformations_current,
    float* pMem_energies_current,
    float* pMem_conformations_next,
    float* pMem_energies_next
)
{
    #pragma omp target
    #pragma omp distribute teams num_teams(nblocks) thread_limit(threadsPerBlock)
    for (int blockIdx = 0; blockIdx < nblocks; blockIdx++){
	 float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
	 int parent_candidates[4];
	 float candidate_energies[4];
	 int parents[2];
	 int covr_point[2];
	 float randnums[10];
         float sBestEnergy[32];
         int sBestID[32];
	 float3 calc_coords[MAX_NUM_OF_ATOMS];
         float sFloatAccumulator;
	 #pragma omp parallel for
         
	 int run_id;    
	 int temp_covr_point;
	 float energy;
         int bestID; 

	 // In this case this compute-unit is responsible for elitist selection
	 if ((blockIdx % cData.dockpars.pop_size) == 0) {
         // Find and copy best member of population to position 0
         if (idx < cData.dockpars.pop_size)
         {
            bestID = blockIdx + idx;
            energy = pMem_energies_current[blockIdx + idx];
         }
         else
         {
            bestID = -1;
            energy = FLT_MAX;
         }
        
         // Scan through population (we already picked up a blockDim's worth above so skip)
         int blockDim = threadsPerBlock;
         for (int i = blockIdx + blockDim + idx; i < blockIdx + cData.dockpars.pop_size; i += blockDim)
         {
            float e = pMem_energies_current[i];
            if (e < energy)
            {
                bestID = i;
                energy = e;
            }
        }
        
        // Reduce to shared memory by warp
        int tgx = idx & cData.warpmask;
        WARPMINIMUM2(tgx, energy, bestID);
        int warpID = idx >> cData.warpbits;
        if (tgx == 0)
        {
            sBestID[warpID] = bestID;
            sBestEnergy[warpID] = fminf(MAXENERGY, energy);
        }
        __threadfence();
        __syncthreads();
               
        // Perform final reduction in warp 0
        if (warpID == 0)
        {
            int blocks = blockDim / 32;
            if (tgx < blocks)
            {
                bestID = sBestID[tgx];
                energy = sBestEnergy[tgx];
            }
            else
            {
                bestID = -1;
                energy = FLT_MAX;
            }
            WARPMINIMUM2(tgx, energy, bestID);     
            
            if (tgx == 0)
            {
                pMem_energies_next[blockIdx] = energy;
                cData.pMem_evals_of_new_entities[blockIdx] = 0;
                sBestID[0] = bestID;
            }
        }
        __threadfence();
        __syncthreads();
        
        // Copy best genome to next generation
        int dOffset = blockIdx * GENOTYPE_LENGTH_IN_GLOBMEM;
        int sOffset = sBestID[0] * GENOTYPE_LENGTH_IN_GLOBMEM;
        for (int i = idx ; i < cData.dockpars.num_of_genes; i += blockDim)
        {
            pMem_conformations_next[dOffset + i] = pMem_conformations_current[sOffset + i];
        }
	}
	else
	{
		// Generating the following random numbers: 
		// [0..3] for parent candidates,
		// [4..5] for binary tournaments, [6] for deciding crossover,
		// [7..8] for crossover points, [9] for local search
		for (uint32_t gene_counter = idx;
		     gene_counter < 10;
		     gene_counter += blockDim) {
			 randnums[gene_counter] = gpu_randf(cData.pMem_prng_states);
		}
#if 0
        if ((idx == 0) && (blockIdx == 1))
        {
            printf("%06d ", blockIdx);
            for (int i = 0; i < 10; i++)
                printf("%12.6f ", randnums[i]);
            printf("\n");
        }
#endif
		// Determining run ID
        run_id = blockIdx / cData.dockpars.pop_size;
        __threadfence();
        __syncthreads();


		if (idx < 4)	//it is not ensured that the four candidates will be different...
		{
			parent_candidates[idx]  = (int) (cData.dockpars.pop_size*randnums[idx]); //using randnums[0..3]
			candidate_energies[idx] = pMem_energies_current[run_id*cData.dockpars.pop_size+parent_candidates[idx]];
		}
        __threadfence();
        __syncthreads();

		if (idx < 2) 
		{
			// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (candidate_energies[2*idx] < candidate_energies[2*idx+1])
            {
				if (/*100.0f**/randnums[4+idx] < cData.dockpars.tournament_rate) {		//using randnum[4..5]
					parents[idx] = parent_candidates[2*idx];
				}
				else {
					parents[idx] = parent_candidates[2*idx+1];
				}
            }
			else
            {
				if (/*100.0f**/randnums[4+idx] < cData.dockpars.tournament_rate) {
					parents[idx] = parent_candidates[2*idx+1];
				}
				else {
					parents[idx] = parent_candidates[2*idx];
				}
            }
		}
        __threadfence();
        __syncthreads();

		// Performing crossover
		// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
		// to reduce number of operations in device
		if (/*100.0f**/randnums[6] < cData.dockpars.crossover_rate)	// Using randnums[6]
		{
			if (idx < 2) {
				// Using randnum[7..8]
				covr_point[idx] = (int) ((cData.dockpars.num_of_genes-1)*randnums[7+idx]);
			}
            __threadfence();
            __syncthreads();
			
			// covr_point[0] should store the lower crossover-point
			if (idx == 0) {
				if (covr_point[1] < covr_point[0]) {
					temp_covr_point = covr_point[1];
					covr_point[1]   = covr_point[0];
					covr_point[0]   = temp_covr_point;
				}
			}

            __threadfence();
            __syncthreads();

			for (uint32_t gene_counter = idx;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= blockDim)
			{
				// Two-point crossover
				if (covr_point[0] != covr_point[1]) 
				{
					if ((gene_counter <= covr_point[0]) || (gene_counter > covr_point[1]))
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
				// Single-point crossover
				else
				{									             
					if (gene_counter <= covr_point[0])
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
					else
						offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[1])*GENOTYPE_LENGTH_IN_GLOBMEM+gene_counter];
				}
			}
		}
		else	//no crossover
		{
            for (uint32_t gene_counter = idx;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= blockDim)
            {
                offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter];
            }
		} // End of crossover

        __threadfence();
        __syncthreads();

		// Performing mutation
		for (uint32_t gene_counter = idx;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= blockDim)
		{
			// Notice: dockpars_mutation_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (/*100.0f**/gpu_randf(cData.pMem_prng_states) < cData.dockpars.mutation_rate)
			{
				// Translation genes
				if (gene_counter < 3) {
					offspring_genotype[gene_counter] += cData.dockpars.abs_max_dmov*(2*gpu_randf(cData.pMem_prng_states)-1);
				}
				// Orientation and torsion genes
				else {
					offspring_genotype[gene_counter] += cData.dockpars.abs_max_dang*(2*gpu_randf(cData.pMem_prng_states)-1);
					map_angle(offspring_genotype[gene_counter]);
				}

			}
		} // End of mutation

		// Calculating energy of new offspring
        __threadfence();
        __syncthreads();
        gpu_calc_energy(
            offspring_genotype,
			energy,
			run_id,
			calc_coords,
            &sFloatAccumulator
		);
        
        
        if (idx == 0) {
            pMem_energies_next[blockIdx] = energy;
            cData.pMem_evals_of_new_entities[blockIdx] = 1;

			#if defined (DEBUG_ENERGY_KERNEL4)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", interE, intraE);
			#endif
        }


		// Copying new offspring to next generation
        for (uint32_t gene_counter = idx;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= blockDim)
        {
            pMem_conformations_next[blockIdx * GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter] = offspring_genotype[gene_counter];
        }        
    }
#if 0
    if ((idx == 0) && (blockIdx == 0))
    {
        printf("%06d %16.8f ", blockIdx, pMem_energies_next[blockIdx]);
        for (int i = 0; i < cData.dockpars.num_of_genes; i++)
            printf("%12.6f ", pMem_conformations_next[GENOTYPE_LENGTH_IN_GLOBMEM*blockIdx + i]);
    }
#endif
}


