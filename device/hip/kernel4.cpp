/*

miniMDock is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
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


//#define DEBUG_ENERGY_KERNEL4

__global__ void
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1024 / NUM_OF_THREADS_PER_BLOCK)
gpu_gen_and_eval_newpops_kernel(
    float* pMem_conformations_current,
    float* pMem_energies_current,
    float* pMem_conformations_next,
    float* pMem_energies_next
)
//The GPU global function
{
	// Some OpenCL compilers don't allow declaring 
	// local variables within non-kernel functions.
	// These local variables must be declared in a kernel, 
	// and then passed to non-kernel functions.
	__shared__ float offspring_genotype[ACTUAL_GENOTYPE_LENGTH];
	__shared__ int parent_candidates[4];
	__shared__ float candidate_energies[4];
	__shared__ int parents[2];
	__shared__ int covr_point[2];
	__shared__ float randnums[10];
    __shared__ float sBestEnergy[64];   	 //array of warpSize  cData.warpsize
    __shared__ int sBestID[64];   		 //array of warpSize
	__shared__ float3 calc_coords[MAX_NUM_OF_ATOMS];
    __shared__ float sFloatAccumulator;
	int run_id;    
	int temp_covr_point;
	float energy;
    int bestID; 
	// In this case this compute-unit is responsible for elitist selection
	if ((hipBlockIdx_x % cData.dockpars.pop_size) == 0) {
        // Find and copy best member of population to position 0
        if (hipThreadIdx_x < cData.dockpars.pop_size)
        {
            bestID = hipBlockIdx_x + hipThreadIdx_x;
            energy = pMem_energies_current[hipBlockIdx_x + hipThreadIdx_x];
        }
        else
        {
            bestID = -1;
            energy = FLT_MAX;
        }
        
        // Scan through population (we already picked up a blockDim's worth above so skip)
        for (int i = hipBlockIdx_x + hipBlockDim_x + hipThreadIdx_x; i < hipBlockIdx_x + cData.dockpars.pop_size; i += hipBlockDim_x)
        {
            float e = pMem_energies_current[i];
            if (e < energy)
            {
                bestID = i;
                energy = e;
            }
        }
        
        // Reduce to shared memory by warp
        int tgx = hipThreadIdx_x & cData.warpmask;
        WARPMINIMUM2(tgx, energy, bestID);
        int warpID = hipThreadIdx_x >> cData.warpbits;
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
            int blocks = hipBlockDim_x / cData.warpsize;   //hipWarpSize may work, warp size = 64 on AMD GCN /* cudaDeviceProp props; cudaGetDeviceProperties(&props, deviceID); int w = props.warpSize;*/
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
                pMem_energies_next[hipBlockIdx_x] = energy;
                cData.pMem_evals_of_new_entities[hipBlockIdx_x] = 0;
                sBestID[0] = bestID;
            }
        }
        __threadfence();
        __syncthreads();
        
        // Copy best genome to next generation
        int dOffset = hipBlockIdx_x * GENOTYPE_LENGTH_IN_GLOBMEM;
        int sOffset = sBestID[0] * GENOTYPE_LENGTH_IN_GLOBMEM;
        for (int i = hipThreadIdx_x ; i < cData.dockpars.num_of_genes; i += hipBlockDim_x)
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
		for (uint32_t gene_counter = hipThreadIdx_x;
		     gene_counter < 10;
		     gene_counter += hipBlockDim_x) {
			 randnums[gene_counter] = gpu_randf(cData.pMem_prng_states);
		}
#if 0
        if ((hipThreadIdx_x == 0) && (hipBlockIdx_x == 1))
        {
            printf("%06d ", hipBlockIdx_x);
            for (int i = 0; i < 10; i++)
                printf("%12.6f ", randnums[i]);
            printf("\n");
        }
#endif
		// Determining run ID
        run_id = hipBlockIdx_x / cData.dockpars.pop_size;
        __threadfence();
        __syncthreads();


		if (hipThreadIdx_x < 4)	//it is not ensured that the four candidates will be different...
		{
			parent_candidates[hipThreadIdx_x]  = (int) (cData.dockpars.pop_size*randnums[hipThreadIdx_x]); //using randnums[0..3]
			candidate_energies[hipThreadIdx_x] = pMem_energies_current[run_id*cData.dockpars.pop_size+parent_candidates[hipThreadIdx_x]];
		}
        __threadfence();
        __syncthreads();

		if (hipThreadIdx_x < 2) 
		{
			// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
			if (candidate_energies[2*hipThreadIdx_x] < candidate_energies[2*hipThreadIdx_x+1])
            {
				if (/*100.0f**/randnums[4+hipThreadIdx_x] < cData.dockpars.tournament_rate) {		//using randnum[4..5]
					parents[hipThreadIdx_x] = parent_candidates[2*hipThreadIdx_x];
				}
				else {
					parents[hipThreadIdx_x] = parent_candidates[2*hipThreadIdx_x+1];
				}
            }
			else
            {
				if (/*100.0f**/randnums[4+hipThreadIdx_x] < cData.dockpars.tournament_rate) {
					parents[hipThreadIdx_x] = parent_candidates[2*hipThreadIdx_x+1];
				}
				else {
					parents[hipThreadIdx_x] = parent_candidates[2*hipThreadIdx_x];
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
			if (hipThreadIdx_x < 2) {
				// Using randnum[7..8]
				covr_point[hipThreadIdx_x] = (int) ((cData.dockpars.num_of_genes-1)*randnums[7+hipThreadIdx_x]);
			}
            __threadfence();
            __syncthreads();
			
			// covr_point[0] should store the lower crossover-point
			if (hipThreadIdx_x == 0) {
				if (covr_point[1] < covr_point[0]) {
					temp_covr_point = covr_point[1];
					covr_point[1]   = covr_point[0];
					covr_point[0]   = temp_covr_point;
				}
			}

            __threadfence();
            __syncthreads();

			for (uint32_t gene_counter = hipThreadIdx_x;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= hipBlockDim_x)
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
            for (uint32_t gene_counter = hipThreadIdx_x;
			     gene_counter < cData.dockpars.num_of_genes;
			     gene_counter+= hipBlockDim_x)
            {
                offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter];
            }
		} // End of crossover

        __threadfence();
        __syncthreads();

		// Performing mutation
		for (uint32_t gene_counter = hipThreadIdx_x;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= hipBlockDim_x)
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
        
        
        if (hipThreadIdx_x == 0) {
            pMem_energies_next[hipBlockIdx_x] = energy;
            cData.pMem_evals_of_new_entities[hipBlockIdx_x] = 1;

			#if defined (DEBUG_ENERGY_KERNEL4)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", interE, intraE);
			#endif
        }


		// Copying new offspring to next generation
        for (uint32_t gene_counter = hipThreadIdx_x;
		     gene_counter < cData.dockpars.num_of_genes;
		     gene_counter+= hipBlockDim_x)
        {
            pMem_conformations_next[hipBlockIdx_x * GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter] = offspring_genotype[gene_counter];
        }        
    }
#if 0
    if ((hipThreadIdx_x == 0) && (hipBlockIdx_x == 0))
    {
        printf("%06d %16.8f ", hipBlockIdx_x, pMem_energies_next[hipBlockIdx_x]);
        for (int i = 0; i < cData.dockpars.num_of_genes; i++)
            printf("%12.6f ", pMem_conformations_next[GENOTYPE_LENGTH_IN_GLOBMEM*hipBlockIdx_x + i]);
    }
#endif
}


void gpu_gen_and_eval_newpops(
    uint32_t blocks,
    uint32_t threadsPerBlock,
    float* pMem_conformations_current,
    float* pMem_energies_current,
    float* pMem_conformations_next,
    float* pMem_energies_next
)
{
    hipLaunchKernelGGL(gpu_gen_and_eval_newpops_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, pMem_conformations_current, pMem_energies_current, pMem_conformations_next, pMem_energies_next);
    LAUNCHERROR("gpu_gen_and_eval_newpops_kernel");    
#if 0
    hipError_t status;
    status = hipDeviceSynchronize();
    RTERROR(status, "gpu_gen_and_eval_newpops_kernel");
    status = hipDeviceReset();
    RTERROR(status, "failed to shut down");
    exit(0);
#endif   
}
