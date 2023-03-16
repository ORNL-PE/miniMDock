#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.

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

/* DPCT_ORIG __global__ void
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1024 / NUM_OF_THREADS_PER_BLOCK)
gpu_gen_and_eval_newpops_kernel(
                                float* pMem_conformations_current,
                                float* pMem_energies_current,
                                float* pMem_conformations_next,
                                float* pMem_energies_next
                               )*/
void

gpu_gen_and_eval_newpops_kernel(
                                float* pMem_conformations_current,
                                float* pMem_energies_current,
                                float* pMem_conformations_next,
                                float* pMem_energies_next
                               ,
                                sycl::nd_item<3> item_ct1,
                                GpuData cData,
                                float *offspring_genotype,
                                int *parent_candidates,
                                float *candidate_energies,
                                int *parents,
                                int *covr_point,
                                float *randnums,
                                float *sBestEnergy,
                                int *sBestID,
                                sycl::float3 *calc_coords,
                                float &sFloatAccumulator)
// The GPU global function
{
/* DPCT_ORIG 	__shared__ float  offspring_genotype[ACTUAL_GENOTYPE_LENGTH];*/

/* DPCT_ORIG 	__shared__ int    parent_candidates [4];*/

/* DPCT_ORIG 	__shared__ float  candidate_energies[4];*/

/* DPCT_ORIG 	__shared__ int    parents           [2];*/

/* DPCT_ORIG 	__shared__ int    covr_point        [2];*/

/* DPCT_ORIG 	__shared__ float  randnums          [10];*/

/* DPCT_ORIG 	__shared__ float  sBestEnergy       [32];*/

/* DPCT_ORIG 	__shared__ int    sBestID           [32];*/

/* DPCT_ORIG 	__shared__ float3 calc_coords       [MAX_NUM_OF_ATOMS];*/

/* DPCT_ORIG 	__shared__ float  sFloatAccumulator;*/

        int run_id;
	int temp_covr_point;
	float energy;
	int bestID;

	// In this case this compute-unit is responsible for elitist selection
/* DPCT_ORIG 	if ((blockIdx.x % cData.dockpars.pop_size) == 0) {*/
        if ((item_ct1.get_group(2) % cData.dockpars.pop_size) == 0) {
                // Find and copy best member of population to position 0
/* DPCT_ORIG 		if (threadIdx.x < cData.dockpars.pop_size)*/
                if (item_ct1.get_local_id(2) < cData.dockpars.pop_size)
                {
/* DPCT_ORIG 			bestID = blockIdx.x + threadIdx.x;*/
                        bestID = item_ct1.get_group(2) + item_ct1.get_local_id(2);
/* DPCT_ORIG 			energy = pMem_energies_current[blockIdx.x +
 * threadIdx.x];*/
                        energy =
                            pMem_energies_current[item_ct1.get_group(2) +
                                                  item_ct1.get_local_id(2)];
                }
		else
		{
			bestID = -1;
			energy = FLT_MAX;
		}
		
		// Scan through population (we already picked up a blockDim's worth above so skip)
/* DPCT_ORIG 		for (int i = blockIdx.x + blockDim.x + threadIdx.x; i <
 * blockIdx.x + cData.dockpars.pop_size; i += blockDim.x)*/
                for (int i = item_ct1.get_group(2) +
                             item_ct1.get_local_range(2) +
                             item_ct1.get_local_id(2);
                     i < item_ct1.get_group(2) + cData.dockpars.pop_size;
                     i += item_ct1.get_local_range(2))
                {
			float e = pMem_energies_current[i];
			if (e < energy)
			{
				bestID = i;
				energy = e;
			}
		}
		
		// Reduce to shared memory by warp
/* DPCT_ORIG 		int tgx = threadIdx.x & cData.warpmask;*/
                int tgx = item_ct1.get_local_id(2) & cData.warpmask;
                /*
                DPCT1096:201: The right-most dimension of the work-group used in
                the SYCL kernel that calls this function may be less than "32".
                The function "dpct::select_from_sub_group" may return an
                unexpected result on the CPU device. Modify the size of the
                work-group to ensure that the value of the right-most dimension
                is a multiple of "32".
                */
                WARPMINIMUM2(tgx, energy, bestID);
/* DPCT_ORIG 		int warpID = threadIdx.x >> cData.warpbits;*/
                int warpID = item_ct1.get_local_id(2) >> cData.warpbits;
                if (tgx == 0)
		{
			sBestID[warpID] = bestID;
/* DPCT_ORIG 			sBestEnergy[warpID] = fminf(MAXENERGY,
 * energy);*/
                        sBestEnergy[warpID] = sycl::fmin((float)MAXENERGY, energy);
                }
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:29: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                // Perform final reduction in warp 0
		if (warpID == 0)
		{
/* DPCT_ORIG 			int blocks = blockDim.x / 32;*/
                        int blocks = item_ct1.get_local_range(2) / 32;
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
                        /*
                        DPCT1096:202: The right-most dimension of the work-group
                        used in the SYCL kernel that calls this function may be
                        less than "32". The function
                        "dpct::select_from_sub_group" may return an unexpected
                        result on the CPU device. Modify the size of the
                        work-group to ensure that the value of the right-most
                        dimension is a multiple of "32".
                        */
                        WARPMINIMUM2(tgx, energy, bestID);

                        if (tgx == 0)
			{
/* DPCT_ORIG 				pMem_energies_next[blockIdx.x] =
 * energy;*/
                                pMem_energies_next[item_ct1.get_group(2)] = energy;
/* DPCT_ORIG
 * cData.pMem_evals_of_new_entities[blockIdx.x] = 0;*/
                                cData.pMem_evals_of_new_entities[item_ct1.get_group(2)] = 0;
                                sBestID[0] = bestID;
			}
		}
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:31: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                // Copy best genome to next generation
/* DPCT_ORIG 		int dOffset = blockIdx.x * GENOTYPE_LENGTH_IN_GLOBMEM;*/
                int dOffset = item_ct1.get_group(2) * GENOTYPE_LENGTH_IN_GLOBMEM;
                int sOffset = sBestID[0] * GENOTYPE_LENGTH_IN_GLOBMEM;
/* DPCT_ORIG 		for (int i = threadIdx.x ; i <
 * cData.dockpars.num_of_genes; i += blockDim.x)*/
                for (int i = item_ct1.get_local_id(2);
                     i < cData.dockpars.num_of_genes;
                     i += item_ct1.get_local_range(2))
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
/* DPCT_ORIG 		for (uint32_t gene_counter = threadIdx.x;*/
                for (uint32_t gene_counter = item_ct1.get_local_id(2);
                     gene_counter < 10;
                     /* DPCT_ORIG 		              gene_counter +=
                        blockDim.x)*/
                     gene_counter += item_ct1.get_local_range(2))
                {
                        randnums[gene_counter] = gpu_randf(cData.pMem_prng_states, item_ct1);
                }
#if 0
		if ((threadIdx.x == 0) && (blockIdx.x == 1))
		{
			printf("%06d ", blockIdx.x);
			for (int i = 0; i < 10; i++)
				printf("%12.6f ", randnums[i]);
			printf("\n");
		}
#endif
		// Determining run ID
/* DPCT_ORIG 		run_id = blockIdx.x / cData.dockpars.pop_size;*/
                run_id = item_ct1.get_group(2) / cData.dockpars.pop_size;
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:34: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:35: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

/* DPCT_ORIG 		if (threadIdx.x < 4) */
                if (item_ct1.get_local_id(2) <
                    4) // it is not ensured that the four candidates will be
                       // different...
                {
/* DPCT_ORIG 			parent_candidates[threadIdx.x]  = (int)
 * (cData.dockpars.pop_size*randnums[threadIdx.x]); */
                        parent_candidates[item_ct1.get_local_id(2)] =
                            (int)(cData.dockpars.pop_size *
                                  randnums[item_ct1.get_local_id(
                                      2)]); // using randnums[0..3]
/* DPCT_ORIG 			candidate_energies[threadIdx.x] =
 * pMem_energies_current[run_id*cData.dockpars.pop_size+parent_candidates[threadIdx.x]];*/
                        candidate_energies[item_ct1.get_local_id(2)] =
                            pMem_energies_current
                                [run_id * cData.dockpars.pop_size +
                                 parent_candidates[item_ct1.get_local_id(2)]];
                }
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:36: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

/* DPCT_ORIG 		if (threadIdx.x < 2)*/
                if (item_ct1.get_local_id(2) < 2)
                {
			// Notice: dockpars_tournament_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
/* DPCT_ORIG 			if (candidate_energies[2*threadIdx.x] <
 * candidate_energies[2*threadIdx.x+1])*/
                        if (candidate_energies[2 * item_ct1.get_local_id(2)] <
                            candidate_energies[2 * item_ct1.get_local_id(2) +
                                               1])
                        {
/* DPCT_ORIG 				if (randnums[4+threadIdx.x] <
 * cData.dockpars.tournament_rate) { */
                                if (/*100.0f**/ randnums
                                        [4 + item_ct1.get_local_id(2)] <
                                    cData.dockpars
                                        .tournament_rate) { // using
                                                            // randnum[4..5]
/* DPCT_ORIG 					parents[threadIdx.x] =
 * parent_candidates[2*threadIdx.x];*/
                                        parents[item_ct1.get_local_id(2)] =
                                            parent_candidates
                                                [2 * item_ct1.get_local_id(2)];
                                }
				else {
/* DPCT_ORIG 					parents[threadIdx.x] =
 * parent_candidates[2*threadIdx.x+1];*/
                                        parents[item_ct1.get_local_id(2)] =
                                            parent_candidates
                                                [2 * item_ct1.get_local_id(2) +
                                                 1];
                                }
			}
			else
			{
/* DPCT_ORIG 				if (randnums[4+threadIdx.x] <
 * cData.dockpars.tournament_rate) {*/
                                if (/*100.0f**/ randnums
                                        [4 + item_ct1.get_local_id(2)] <
                                    cData.dockpars.tournament_rate) {
/* DPCT_ORIG 					parents[threadIdx.x] =
 * parent_candidates[2*threadIdx.x+1];*/
                                        parents[item_ct1.get_local_id(2)] =
                                            parent_candidates
                                                [2 * item_ct1.get_local_id(2) +
                                                 1];
                                }
				else {
/* DPCT_ORIG 					parents[threadIdx.x] =
 * parent_candidates[2*threadIdx.x];*/
                                        parents[item_ct1.get_local_id(2)] =
                                            parent_candidates
                                                [2 * item_ct1.get_local_id(2)];
                                }
			}
		}
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:38: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                // Performing crossover
		// Notice: dockpars_crossover_rate was scaled down to [0,1] in host
		// to reduce number of operations in device
		if (/*100.0f**/randnums[6] < cData.dockpars.crossover_rate) // Using randnums[6]
		{
/* DPCT_ORIG 			if (threadIdx.x < 2) {*/
                        if (item_ct1.get_local_id(2) < 2) {
                                // Using randnum[7..8]
/* DPCT_ORIG 				covr_point[threadIdx.x] = (int)
 * ((cData.dockpars.num_of_genes-1)*randnums[7+threadIdx.x]);*/
                                covr_point[item_ct1.get_local_id(2)] =
                                    (int)((cData.dockpars.num_of_genes - 1) *
                                          randnums[7 +
                                                   item_ct1.get_local_id(2)]);
                        }
/* DPCT_ORIG 			__threadfence();*/
                        /*
                        DPCT1078:44: Consider replacing memory_order::acq_rel
                        with memory_order::seq_cst for correctness if strong
                        memory order restrictions are needed.
                        */
                        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 			__syncthreads();*/
                        /*
                        DPCT1065:45: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

                        // covr_point[0] should store the lower crossover-point
/* DPCT_ORIG 			if (threadIdx.x == 0) {*/
                        if (item_ct1.get_local_id(2) == 0) {
                                if (covr_point[1] < covr_point[0]) {
					temp_covr_point = covr_point[1];
					covr_point[1]   = covr_point[0];
					covr_point[0]   = temp_covr_point;
				}
			}

/* DPCT_ORIG 			__threadfence();*/
                        /*
                        DPCT1078:46: Consider replacing memory_order::acq_rel
                        with memory_order::seq_cst for correctness if strong
                        memory order restrictions are needed.
                        */
                        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 			__syncthreads();*/
                        /*
                        DPCT1065:47: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

/* DPCT_ORIG 			for (uint32_t gene_counter =
    * threadIdx.x;*/
                        for (uint32_t gene_counter = item_ct1.get_local_id(2);
                             gene_counter < cData.dockpars.num_of_genes;
                             /* DPCT_ORIG 			              gene_counter+=
                                blockDim.x)*/
                             gene_counter += item_ct1.get_local_range(2))
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
		else //no crossover
		{
/* DPCT_ORIG 			for (uint32_t gene_counter =
    * threadIdx.x;*/
                        for (uint32_t gene_counter = item_ct1.get_local_id(2);
                             gene_counter < cData.dockpars.num_of_genes;
                             /* DPCT_ORIG 			              gene_counter+=
                                blockDim.x)*/
                             gene_counter += item_ct1.get_local_range(2))
                        {
				offspring_genotype[gene_counter] = pMem_conformations_current[(run_id*cData.dockpars.pop_size+parents[0])*GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter];
			}
		} // End of crossover

/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:40: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:41: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                // Performing mutation
/* DPCT_ORIG 		for (uint32_t gene_counter = threadIdx.x;*/
                for (uint32_t gene_counter = item_ct1.get_local_id(2);
                     gene_counter < cData.dockpars.num_of_genes;
                     /* DPCT_ORIG 		              gene_counter+=
                        blockDim.x)*/
                     gene_counter += item_ct1.get_local_range(2))
                {
			// Notice: dockpars_mutation_rate was scaled down to [0,1] in host
			// to reduce number of operations in device
                        if (/*100.0f**/ gpu_randf(cData.pMem_prng_states,
                                                  item_ct1) <
                            cData.dockpars.mutation_rate)
                        {
				// Translation genes
				if (gene_counter < 3) {
                                        offspring_genotype[gene_counter] +=
                                            cData.dockpars.abs_max_dmov *
                                            (2.0f * gpu_randf(
                                                        cData.pMem_prng_states,
                                                        item_ct1) -
                                             1.0f);
                                }
				// Orientation and torsion genes
				else {
                                        offspring_genotype[gene_counter] +=
                                            cData.dockpars.abs_max_dang *
                                            (2.0f * gpu_randf(
                                                        cData.pMem_prng_states,
                                                        item_ct1) -
                                             1.0f);
                                        map_angle(offspring_genotype[gene_counter]);
				}

			}
		} // End of mutation

		// Calculating energy of new offspring
/* DPCT_ORIG 		__threadfence();*/
                /*
                DPCT1078:42: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                gpu_calc_energy(offspring_genotype, energy, run_id, calc_coords,
                                &sFloatAccumulator, item_ct1, cData);

/* DPCT_ORIG 		if (threadIdx.x == 0) {*/
                if (item_ct1.get_local_id(2) == 0) {
/* DPCT_ORIG 			pMem_energies_next[blockIdx.x] =
    * energy;*/
                        pMem_energies_next[item_ct1.get_group(2)] = energy;
/* DPCT_ORIG 			cData.pMem_evals_of_new_entities[blockIdx.x] =
 * 1;*/
                        cData.pMem_evals_of_new_entities[item_ct1.get_group(2)] = 1;

#if defined (DEBUG_ENERGY_KERNEL4)
			printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL4-", "GRIDS", "INTRA", interE, intraE);
			#endif
		}


		// Copying new offspring to next generation
/* DPCT_ORIG 		for (uint32_t gene_counter = threadIdx.x;*/
                for (uint32_t gene_counter = item_ct1.get_local_id(2);
                     gene_counter < cData.dockpars.num_of_genes;
                     /* DPCT_ORIG 		              gene_counter+=
                        blockDim.x)*/
                     gene_counter += item_ct1.get_local_range(2))
                {
/* DPCT_ORIG 			pMem_conformations_next[blockIdx.x *
 * GENOTYPE_LENGTH_IN_GLOBMEM + gene_counter] =
 * offspring_genotype[gene_counter];*/
                        pMem_conformations_next[item_ct1.get_group(2) *
                                                    GENOTYPE_LENGTH_IN_GLOBMEM +
                                                gene_counter] =
                            offspring_genotype[gene_counter];
                }
	}
#if 0
	if ((threadIdx.x == 0) && (blockIdx.x == 0))
	{
		printf("%06d %16.8f ", blockIdx.x, pMem_energies_next[blockIdx.x]);
		for (int i = 0; i < cData.dockpars.num_of_genes; i++)
			printf("%12.6f ", pMem_conformations_next[GENOTYPE_LENGTH_IN_GLOBMEM*blockIdx.x + i]);
	}
#endif
}


void gpu_gen_and_eval_newpops(
                              uint32_t blocks,
                              uint32_t threadsPerBlock,
                              float*   pMem_conformations_current,
                              float*   pMem_energies_current,
                              float*   pMem_conformations_next,
                              float*   pMem_energies_next
                             )
{
/* DPCT_ORIG 	gpu_gen_and_eval_newpops_kernel<<<blocks,
 * threadsPerBlock>>>(pMem_conformations_current, pMem_energies_current,
 * pMem_conformations_next, pMem_energies_next);*/
        /*
        DPCT1049:48: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<GpuData, 0> cData;

                cData.init();

                auto cData_ptr_ct1 = cData.get_ptr();

                /*
                DPCT1101:199: 'ACTUAL_GENOTYPE_LENGTH' expression was replaced
                with a value. Modify the code to use original expression,
                provided in comments, if it is correct.
                */
                sycl::local_accessor<float, 1> offspring_genotype_acc_ct1(
                    sycl::range<1>(63 /*ACTUAL_GENOTYPE_LENGTH*/), cgh);
                sycl::local_accessor<int, 1> parent_candidates_acc_ct1(
                    sycl::range<1>(4), cgh);
                sycl::local_accessor<float, 1> candidate_energies_acc_ct1(
                    sycl::range<1>(4), cgh);
                sycl::local_accessor<int, 1> parents_acc_ct1(sycl::range<1>(2),
                                                             cgh);
                sycl::local_accessor<int, 1> covr_point_acc_ct1(
                    sycl::range<1>(2), cgh);
                sycl::local_accessor<float, 1> randnums_acc_ct1(
                    sycl::range<1>(10), cgh);
                sycl::local_accessor<float, 1> sBestEnergy_acc_ct1(
                    sycl::range<1>(32), cgh);
                sycl::local_accessor<int, 1> sBestID_acc_ct1(sycl::range<1>(32),
                                                             cgh);
                /*
                DPCT1101:200: 'MAX_NUM_OF_ATOMS' expression was replaced with a
                value. Modify the code to use original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<sycl::float3, 1> calc_coords_acc_ct1(
                    sycl::range<1>(256 /*MAX_NUM_OF_ATOMS*/), cgh);
                sycl::local_accessor<float, 0> sFloatAccumulator_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threadsPerBlock),
                                      sycl::range<3>(1, 1, threadsPerBlock)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                                gpu_gen_and_eval_newpops_kernel(
                                    pMem_conformations_current,
                                    pMem_energies_current,
                                    pMem_conformations_next, pMem_energies_next,
                                    item_ct1, *cData_ptr_ct1,
                                    offspring_genotype_acc_ct1.get_pointer(),
                                    parent_candidates_acc_ct1.get_pointer(),
                                    candidate_energies_acc_ct1.get_pointer(),
                                    parents_acc_ct1.get_pointer(),
                                    covr_point_acc_ct1.get_pointer(),
                                    randnums_acc_ct1.get_pointer(),
                                    sBestEnergy_acc_ct1.get_pointer(),
                                    sBestID_acc_ct1.get_pointer(),
                                    calc_coords_acc_ct1.get_pointer(),
                                    sFloatAccumulator_acc_ct1);
                        });
        });
        /*
        DPCT1001:187: The statement could not be removed.
        */
        LAUNCHERROR("gpu_gen_and_eval_newpops_kernel");
#if 0
	cudaError_t status;
	status = cudaDeviceSynchronize();
	RTERROR(status, "gpu_gen_and_eval_newpops_kernel");
	status = cudaDeviceReset();
	RTERROR(status, "failed to shut down");
	exit(0);
#endif
}
