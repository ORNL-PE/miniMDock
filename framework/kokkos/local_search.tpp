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
#include "calcenergrad.hpp"
#include "random.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void solis_wets(Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts)
{
	// Outer loop
        int league_size = docking_params.num_of_lsentities * mypars->num_of_runs;

	// Get the size of the shared memory allocation
        size_t shmem_size = Coordinates::shmem_size(docking_params.num_of_atoms) + 2*Genotype::shmem_size(docking_params.num_of_genes) + 2*GenotypeAux::shmem_size(docking_params.num_of_genes)
			  + OneInt::shmem_size() + 2*OneBool::shmem_size() + OneFloat::shmem_size();
	Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ).
                              set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                        KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();
		int team_size = team_member.team_size();
		int run_id = lidx / docking_params.num_of_lsentities;

		// Locally shared: global index in population
		OneInt gpop_idx(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
                Genotype genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_genes);
		Genotype best_genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_genes);
		GenotypeAux genotype_bias(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_genes);
		GenotypeAux genotype_deviate(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_genes);

		// Determine gpop_idx
		if (tidx == 0)
		{
			int entity_id = lidx - run_id * docking_params.num_of_lsentities; // modulus in different form

			// Since entity 0 is the best one due to elitism,
			// it should be subjected to random selection
			if (entity_id == 0) {
				// If entity 0 is not selected according to LS-rate, choosing another entity
				if (100.0f*rand_float(team_member, docking_params) > docking_params.lsearch_rate) {
					entity_id = docking_params.num_of_lsentities; // AT - Should this be (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states))?
				}
			}

			gpop_idx(0) = run_id*docking_params.pop_size+entity_id; // global population index
		}

		team_member.team_barrier();
		// Copy genotype to local shared memory
		copy_genotype(team_member, docking_params.num_of_genes, genotype, next, gpop_idx(0));


		team_member.team_barrier();

		// Initializing best genotype and energy
		float energy = next.energies(gpop_idx(0)); // Dont need to init this since it's overwritten
		float best_energy = energy;
                copy_genotype(team_member, docking_params.num_of_genes, best_genotype, genotype);

		// Initializing variable arrays for solis-wets algorithm
		for(int i = tidx; i < docking_params.num_of_genes; i+= team_size) {
                        genotype_bias[i]=0; // Probably unnecessary since kokkos views are automatically initialized to 0 (not sure if that's the case in scratch though)
                }


		// Initialize iteration controls
		OneBool stay_in_loop(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		stay_in_loop(0)=true;
		OneBool energy_improved(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		energy_improved(0)=false;
		unsigned int iteration_cnt = 0;
		int evaluation_cnt = 0;
		OneFloat rho(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));
		rho(0)  = 1.0f;
		int   cons_succ = 0;
		int   cons_fail = 0;

		team_member.team_barrier();

		// Declare/allocate coordinates for internal use by calc_energy only. Must be outside of loop since there is
		// no way to de/reallocate things in Kokkos team scratch
		Coordinates calc_coords(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_atoms);
		while (stay_in_loop(0)){
			// New random deviate
			float good_dir = 1.0f;
			for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_size) {
				genotype_deviate[gene_cnt] = rho(0)*(2*rand_float(team_member, docking_params)-1)*(rand_float(team_member, docking_params)<0.09f);

				if (gene_cnt < 3) { // Translation genes
					genotype_deviate[gene_cnt] *= docking_params.base_dmov_mul_sqrt3;
				} else { // Orientation and torsion genes
					genotype_deviate[gene_cnt] *= docking_params.base_dang_mul_sqrt3;
				}

				// Generating new genotype candidate
				genotype[gene_cnt] = best_genotype[gene_cnt] +
							(genotype_deviate[gene_cnt] + genotype_bias[gene_cnt]);
			}

			team_member.team_barrier();

			// Calculating energy of candidate, increment #evals, check if energy improved
			energy = calc_energy(team_member, docking_params, consts, calc_coords, genotype, run_id);
			if (tidx == 0 )evaluation_cnt++;
			if ((tidx == 0) && (energy < best_energy)) energy_improved(0)=true;

			team_member.team_barrier();

			if (!energy_improved(0)){ // Candidate is worse, check opposite direction
				good_dir = -1.0;
				for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_size) {
					// Generating other genotype candidate
					genotype[gene_cnt] = best_genotype[gene_cnt] -
								(genotype_deviate[gene_cnt] + genotype_bias[gene_cnt]);
				}

				team_member.team_barrier();

				// Calculating energy of candidate, increment #evals, check if energy improved
				energy = calc_energy(team_member, docking_params, consts, calc_coords, genotype, run_id);
				if(tidx == 0)	evaluation_cnt++;
				if ((tidx == 0) && (energy < best_energy)) energy_improved(0)=true;

				team_member.team_barrier();
			}

			if (energy_improved(0)){ // Success! Candidate is better
				for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_size) {
					// Updating best_genotype
					best_genotype[gene_cnt] = genotype[gene_cnt];

					// Updating genotype_bias
					genotype_bias[gene_cnt] = 0.6f*genotype_bias[gene_cnt] + good_dir*0.4f*genotype_deviate[gene_cnt];
				}

				team_member.team_barrier();
                                if(tidx==0){
				best_energy = energy;
				cons_succ++;
				cons_fail = 0;
				}
				energy_improved(0)=false;
			} else { // Failure in both directions
				for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_size) {
					// Reducing genotype_bias
					genotype_bias[gene_cnt] *= 0.5f;
		 		}
                                if(tidx==0){
				cons_succ = 0;
                                cons_fail++;
				}
			}


			// Iteration controls
			if (tidx == 0) {
				if (cons_succ >= docking_params.cons_limit) {
					rho(0) *= LS_EXP_FACTOR;
					cons_succ = 0;
				} else if (cons_fail >= docking_params.cons_limit) {
					rho(0) *= LS_CONT_FACTOR;
					cons_fail = 0;
				}

				// Updating number of ADADELTA iterations (energy evaluations)
				iteration_cnt++; 
				energy_improved(0)=false; // reset to zero for next loop iteration

				if ((iteration_cnt >= docking_params.max_num_of_iters) || (rho(0) <= docking_params.rho_lower_bound))
					stay_in_loop(0)=false;
			}

			team_member.team_barrier(); // making sure that stay_in_loop(0) is up to date
		}
		// -----------------------------------------------------------------------------

		// Modulo torsion angles
		for (int gene_cnt = tidx; gene_cnt < docking_params.num_of_genes; gene_cnt+= team_size) {
                        if (gene_cnt >=3){
			 while (best_genotype[gene_cnt] >= 360.0f) { best_genotype[gene_cnt] -= 360.0f; }
                         while (best_genotype[gene_cnt] < 0.0f   ) { best_genotype[gene_cnt] += 360.0f; }
			}
		}

                // Copy to global views
                if( tidx == 0 ) {
                        next.energies(gpop_idx(0)) = best_energy;
                        docking_params.evals_of_new_entities(gpop_idx(0)) += evaluation_cnt;
                }

		copy_genotype(team_member, docking_params.num_of_genes, next, gpop_idx(0), best_genotype);

        });
}
