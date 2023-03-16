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

// if defined, new (experimental) SW genotype moves that are dependent
// on nr of atoms and nr of torsions of ligand are used
#define SWAT3 // Third set of Solis-Wets hyperparameters by Andreas Tillack

/* DPCT_ORIG __global__ void
#if (__CUDA_ARCH__ == 750)
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1024 / NUM_OF_THREADS_PER_BLOCK)
#else
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1408 / NUM_OF_THREADS_PER_BLOCK)
#endif
gpu_perform_LS_kernel(
                      float* pMem_conformations_next,
                      float* pMem_energies_next
                     )*/
void
#if (DPCT_COMPATIBILITY_TEMP == 750)
__launch_bounds__(NUM_OF_THREADS_PER_BLOCK, 1024 / NUM_OF_THREADS_PER_BLOCK)
#else

#endif
gpu_perform_LS_kernel(
                      float* pMem_conformations_next,
                      float* pMem_energies_next
                     ,
                      sycl::nd_item<3> item_ct1,
                      GpuData cData,
                      float &rho,
                      int &cons_succ,
                      int &cons_fail,
                      int &iteration_cnt,
                      int &evaluation_cnt,
                      float &offspring_energy,
                      float &sFloatAccumulator,
                      int &entity_id,
                      volatile float *sFloatBuff)
// The GPU global function performs local search on the pre-defined entities of conformations_next.
// The number of blocks which should be started equals to num_of_lsentities*num_of_runs.
// This way the first num_of_lsentities entity of each population will be subjected to local search
// (and each block carries out the algorithm for one entity).
// Since the first entity is always the best one in the current population,
// it is always tested according to the ls probability, and if it not to be
// subjected to local search, the entity with ID num_of_lsentities is selected instead of the first one (with ID 0).
{
/* DPCT_ORIG 	__shared__ float rho;*/

/* DPCT_ORIG 	__shared__ int   cons_succ;*/

/* DPCT_ORIG 	__shared__ int   cons_fail;*/

/* DPCT_ORIG 	__shared__ int   iteration_cnt;*/

/* DPCT_ORIG 	__shared__ int   evaluation_cnt;*/

/* DPCT_ORIG 	__shared__ float offspring_energy;*/

/* DPCT_ORIG 	__shared__ float sFloatAccumulator;*/

/* DPCT_ORIG 	__shared__ int entity_id;*/

/* DPCT_ORIG 	volatile __shared__ float sFloatBuff[3 * MAX_NUM_OF_ATOMS + 4
 * * ACTUAL_GENOTYPE_LENGTH];*/

        float candidate_energy;
	int run_id;
	// Ligand-atom position and partial energies
/* DPCT_ORIG 	float3* calc_coords = (float3*)sFloatBuff;*/
        sycl::float3 *calc_coords = (sycl::float3 *)sFloatBuff;

        // Genotype pointers
	float* genotype_candidate = (float*)(calc_coords + MAX_NUM_OF_ATOMS);
	float* genotype_deviate = (float*)(genotype_candidate + ACTUAL_GENOTYPE_LENGTH);
	float* genotype_bias = (float*)(genotype_deviate + ACTUAL_GENOTYPE_LENGTH);
	float* offspring_genotype = (float*)(genotype_bias + ACTUAL_GENOTYPE_LENGTH);

	// Determining run ID and entity ID
	// Initializing offspring genotype
/* DPCT_ORIG 	run_id = blockIdx.x / cData.dockpars.num_of_lsentities;*/
        run_id = item_ct1.get_group(2) / cData.dockpars.num_of_lsentities;
/* DPCT_ORIG 	if (threadIdx.x == 0)*/
        if (item_ct1.get_local_id(2) == 0)
        {
/* DPCT_ORIG 		entity_id = blockIdx.x %
 * cData.dockpars.num_of_lsentities;*/
                entity_id = item_ct1.get_group(2) % cData.dockpars.num_of_lsentities;
                // Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate,
			// choosing an other entity
                        if (100.0f *
                                gpu_randf(cData.pMem_prng_states, item_ct1) >
                            cData.dockpars.lsearch_rate) {
                                entity_id = cData.dockpars.num_of_lsentities;
			}
		}

		offspring_energy = pMem_energies_next[run_id*cData.dockpars.pop_size+entity_id];
		rho = 1.0f;
		cons_succ = 0;
		cons_fail = 0;
		iteration_cnt = 0;
		evaluation_cnt = 0;
	}
/* DPCT_ORIG 	__syncthreads();*/
        /*
        DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        size_t offset = (run_id * cData.dockpars.pop_size + entity_id) * GENOTYPE_LENGTH_IN_GLOBMEM;
/* DPCT_ORIG 	for (uint32_t gene_counter = threadIdx.x;*/
        for (uint32_t gene_counter = item_ct1.get_local_id(2);
             gene_counter < cData.dockpars.num_of_genes;
             /* DPCT_ORIG 	              gene_counter+= blockDim.x)*/
             gene_counter += item_ct1.get_local_range(2))
        {
		offspring_genotype[gene_counter] = pMem_conformations_next[offset + gene_counter];
		genotype_bias[gene_counter] = 0.0f;
	}
/* DPCT_ORIG 	__syncthreads();*/
        /*
        DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        uint32_t gene_start = (cData.dockpars.true_ligand_atoms==0)*6;

#ifdef SWAT3
/* DPCT_ORIG 	float lig_scale
 * = 1.0f/sqrt((float)cData.dockpars.num_of_atoms);*/
        float lig_scale = 1.0f / sycl::sqrt((float)cData.dockpars.num_of_atoms);
/* DPCT_ORIG 	float gene_scale
 * = 1.0f/sqrt((float)cData.dockpars.num_of_genes);*/
        float gene_scale = 1.0f / sycl::sqrt((float)cData.dockpars.num_of_genes);
#endif
	while ((iteration_cnt < cData.dockpars.max_num_of_iters) && (rho > cData.dockpars.rho_lower_bound))
	{
		// New random deviate
/* DPCT_ORIG 		for (uint32_t gene_counter = threadIdx.x+gene_start;*/
                for (uint32_t gene_counter =
                         item_ct1.get_local_id(2) + gene_start;
                     gene_counter < cData.dockpars.num_of_genes;
                     /* DPCT_ORIG 		              gene_counter+=
                        blockDim.x)*/
                     gene_counter += item_ct1.get_local_range(2))
                {
#ifdef SWAT3
                        genotype_deviate[gene_counter] =
                            rho *
                            (2.0f *
                                 gpu_randf(cData.pMem_prng_states, item_ct1) -
                             1.0f) *
                            (gpu_randf(cData.pMem_prng_states, item_ct1) <
                             gene_scale);

                        // Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= cData.dockpars.base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				if (gene_counter < 6) {
					genotype_deviate[gene_counter] *= cData.dockpars.base_dang_mul_sqrt3 * lig_scale;
				} else {
					genotype_deviate[gene_counter] *= cData.dockpars.base_dang_mul_sqrt3 * gene_scale;
				}
			}
#else
			genotype_deviate[gene_counter] = rho*(2.0f*gpu_randf(cData.pMem_prng_states)-1.0f)*(gpu_randf(cData.pMem_prng_states)<0.3f);

			// Translation genes
			if (gene_counter < 3) {
				genotype_deviate[gene_counter] *= cData.dockpars.base_dmov_mul_sqrt3;
			}
			// Orientation and torsion genes
			else {
				genotype_deviate[gene_counter] *= cData.dockpars.base_dang_mul_sqrt3;
			}
#endif
		}

		// Generating new genotype candidate
/* DPCT_ORIG 		for (uint32_t gene_counter = threadIdx.x+gene_start;*/
                for (uint32_t gene_counter =
                         item_ct1.get_local_id(2) + gene_start;
                     gene_counter < cData.dockpars.num_of_genes;
                     /* DPCT_ORIG 		              gene_counter+=
                        blockDim.x)*/
                     gene_counter += item_ct1.get_local_range(2))
                {
			genotype_candidate[gene_counter] = offspring_genotype[gene_counter] +
			                                   genotype_deviate[gene_counter]   +
			                                   genotype_bias[gene_counter];
		}
		// Evaluating candidate
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                // =================================================================
                gpu_calc_energy(genotype_candidate, candidate_energy, run_id,
                                calc_coords, &sFloatAccumulator, item_ct1,
                                cData);
                // =================================================================
/* DPCT_ORIG 		if (threadIdx.x == 0) {*/
                if (item_ct1.get_local_id(2) == 0) {
                        evaluation_cnt++;
		}
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                if (candidate_energy < offspring_energy)	// If candidate is better, success
		{
/* DPCT_ORIG 			for (uint32_t gene_counter =
 * threadIdx.x+gene_start;*/
                        for (uint32_t gene_counter =
                                 item_ct1.get_local_id(2) + gene_start;
                             gene_counter < cData.dockpars.num_of_genes;
                             /* DPCT_ORIG 			              gene_counter+=
                                blockDim.x)*/
                             gene_counter += item_ct1.get_local_range(2))
                        {
				// Updating offspring_genotype
				offspring_genotype[gene_counter] = genotype_candidate[gene_counter];
				// Updating genotype_bias
				genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] + 0.4f*genotype_deviate[gene_counter];
			}

			// Work-item 0 will overwrite the shared variables
			// used in the previous if condition
/* DPCT_ORIG 			__syncthreads();*/
                        /*
                        DPCT1065:24: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

/* DPCT_ORIG 			if (threadIdx.x == 0)*/
                        if (item_ct1.get_local_id(2) == 0)
                        {
				offspring_energy = candidate_energy;
				cons_succ++;
				cons_fail = 0;
			}
		}
		else // If candidate is worse, check the opposite direction
		{
			// Generating the other genotype candidate
/* DPCT_ORIG 			for (uint32_t gene_counter =
 * threadIdx.x+gene_start;*/
                        for (uint32_t gene_counter =
                                 item_ct1.get_local_id(2) + gene_start;
                             gene_counter < cData.dockpars.num_of_genes;
                             /* DPCT_ORIG 			              gene_counter+=
                                blockDim.x)*/
                             gene_counter += item_ct1.get_local_range(2))
                        {
				genotype_candidate[gene_counter] = offspring_genotype[gene_counter] -
				                                   genotype_deviate[gene_counter] -
				                                   genotype_bias[gene_counter];
			}

			// Evaluating candidate
/* DPCT_ORIG 			__syncthreads();*/
                        /*
                        DPCT1065:25: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

                        // =================================================================
                        gpu_calc_energy(genotype_candidate, candidate_energy,
                                        run_id, calc_coords, &sFloatAccumulator,
                                        item_ct1, cData);
                        // =================================================================

/* DPCT_ORIG 			if (threadIdx.x == 0) {*/
                        if (item_ct1.get_local_id(2) == 0) {
                                evaluation_cnt++;

				#if defined (DEBUG_ENERGY_KERNEL)
				printf("%-18s [%-5s]---{%-5s}   [%-10.8f]---{%-10.8f}\n", "-ENERGY-KERNEL3-", "GRIDS", "INTRA", partial_interE[0], partial_intraE[0]);
				#endif
			}
/* DPCT_ORIG 			__syncthreads();*/
                        /*
                        DPCT1065:26: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

                        if (candidate_energy < offspring_energy) // If candidate is better, success
			{
/* DPCT_ORIG 				for (uint32_t gene_counter =
 * threadIdx.x+gene_start;*/
                                for (uint32_t gene_counter =
                                         item_ct1.get_local_id(2) + gene_start;
                                     gene_counter < cData.dockpars.num_of_genes;
                                     /* DPCT_ORIG
                                        gene_counter+= blockDim.x)*/
                                     gene_counter +=
                                     item_ct1.get_local_range(2))
                                {
					// Updating offspring_genotype
					offspring_genotype[gene_counter] = genotype_candidate[gene_counter];
					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.6f*genotype_bias[gene_counter] - 0.4f*genotype_deviate[gene_counter];
				}

				// Work-item 0 will overwrite the shared variables
				// used in the previous if condition
/* DPCT_ORIG 				__syncthreads();*/
                                /*
                                DPCT1065:27: Consider replacing
                                sycl::nd_item::barrier() with
                                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                                for better performance if there is no access to
                                global memory.
                                */
                                item_ct1.barrier();

/* DPCT_ORIG 				if (threadIdx.x == 0)*/
                                if (item_ct1.get_local_id(2) == 0)
                                {
					offspring_energy = candidate_energy;
					cons_succ++;
					cons_fail = 0;
				}
			}
			else	// Failure in both directions
			{
/* DPCT_ORIG 				for (uint32_t gene_counter =
 * threadIdx.x+gene_start;*/
                                for (uint32_t gene_counter =
                                         item_ct1.get_local_id(2) + gene_start;
                                     gene_counter < cData.dockpars.num_of_genes;
                                     /* DPCT_ORIG
                                        gene_counter+= blockDim.x)*/
                                     gene_counter +=
                                     item_ct1.get_local_range(2))
                                {
					// Updating genotype_bias
					genotype_bias[gene_counter] = 0.5f*genotype_bias[gene_counter];
				}
/* DPCT_ORIG 				if (threadIdx.x == 0)*/
                                if (item_ct1.get_local_id(2) == 0)
                                {
					cons_succ = 0;
					cons_fail++;
				}
			}
		}

		// Changing rho if needed
/* DPCT_ORIG 		if (threadIdx.x == 0)*/
                if (item_ct1.get_local_id(2) == 0)
                {
			iteration_cnt++;
			if (cons_succ >= cData.dockpars.cons_limit)
			{
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			}
			else
				if (cons_fail >= cData.dockpars.cons_limit)
				{
					rho *= LS_CONT_FACTOR;
					cons_fail = 0;
				}
		}
/* DPCT_ORIG 		__syncthreads();*/
                /*
                DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
        }

	// Updating eval counter and energy
/* DPCT_ORIG 	if (threadIdx.x == 0) {*/
        if (item_ct1.get_local_id(2) == 0) {
                cData.pMem_evals_of_new_entities[run_id*cData.dockpars.pop_size+entity_id] += evaluation_cnt;
		pMem_energies_next[run_id*cData.dockpars.pop_size+entity_id] = offspring_energy;
	}

	// Mapping torsion angles and writing out results
	offset = (run_id*cData.dockpars.pop_size+entity_id)*GENOTYPE_LENGTH_IN_GLOBMEM;
/* DPCT_ORIG 	for (uint32_t gene_counter = threadIdx.x+gene_start;*/
        for (uint32_t gene_counter = item_ct1.get_local_id(2) + gene_start;
             gene_counter < cData.dockpars.num_of_genes;
             /* DPCT_ORIG 	              gene_counter+= blockDim.x)*/
             gene_counter += item_ct1.get_local_range(2))
        {
		if (gene_counter >= 3) {
			map_angle(offspring_genotype[gene_counter]);
		}
		pMem_conformations_next[offset + gene_counter] = offspring_genotype[gene_counter];
	}
}


void gpu_perform_LS(
                    uint32_t blocks,
                    uint32_t threads,
                    float*   pMem_conformations_next,
                    float*   pMem_energies_next
                   )
{
/* DPCT_ORIG 	gpu_perform_LS_kernel<<<blocks,
 * threads>>>(pMem_conformations_next, pMem_energies_next);*/
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<GpuData, 0> cData;

                cData.init();

                auto cData_ptr_ct1 = cData.get_ptr();

                sycl::local_accessor<float, 0> rho_acc_ct1(cgh);
                sycl::local_accessor<int, 0> cons_succ_acc_ct1(cgh);
                sycl::local_accessor<int, 0> cons_fail_acc_ct1(cgh);
                sycl::local_accessor<int, 0> iteration_cnt_acc_ct1(cgh);
                sycl::local_accessor<int, 0> evaluation_cnt_acc_ct1(cgh);
                sycl::local_accessor<float, 0> offspring_energy_acc_ct1(cgh);
                sycl::local_accessor<float, 0> sFloatAccumulator_acc_ct1(cgh);
                sycl::local_accessor<int, 0> entity_id_acc_ct1(cgh);
                /*
                DPCT1101:205: '3 * MAX_NUM_OF_ATOMS + 4 *
                ACTUAL_GENOTYPE_LENGTH' expression was replaced with a value.
                Modify the code to use original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<volatile float, 1> sFloatBuff_acc_ct1(sycl::range<
                                                                               1>(
                                                                               1020 /*3 * MAX_NUM_OF_ATOMS + 4 * ACTUAL_GENOTYPE_LENGTH*/),
                                                                           cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                                gpu_perform_LS_kernel(
                                    pMem_conformations_next, pMem_energies_next,
                                    item_ct1, *cData_ptr_ct1, rho_acc_ct1,
                                    cons_succ_acc_ct1, cons_fail_acc_ct1,
                                    iteration_cnt_acc_ct1,
                                    evaluation_cnt_acc_ct1,
                                    offspring_energy_acc_ct1,
                                    sFloatAccumulator_acc_ct1,
                                    entity_id_acc_ct1,
                                    sFloatBuff_acc_ct1.get_pointer());
                        });
        });
        /*
        DPCT1001:186: The statement could not be removed.
        */
        LAUNCHERROR("gpu_perform_LS_kernel");
#if 0
	cudaError_t status;
	status = cudaDeviceSynchronize();
	RTERROR(status, "gpu_perform_LS_kernel");
	status = cudaDeviceReset();
	RTERROR(status, "failed to shut down");
	exit(0);
#endif
}
