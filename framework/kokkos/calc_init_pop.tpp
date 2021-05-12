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
#include "calcenergy.hpp"
#include "genotype_funcs.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void calc_init_pop(Generation<Device>& current, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts)
{
	// Outer loop over mypars->pop_size * mypars->num_of_runs
        int league_size = mypars->pop_size * mypars->num_of_runs;

	// Get the size of the shared memory allocation
	size_t shmem_size = Coordinates::shmem_size(docking_params.num_of_atoms) + Genotype::shmem_size(docking_params.num_of_genes);
        Kokkos::parallel_for (Kokkos::TeamPolicy<ExSpace> (league_size, NUM_OF_THREADS_PER_BLOCK ).
                              set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                              KOKKOS_LAMBDA (member_type team_member)
        {
                // Get team and league ranks
                int tidx = team_member.team_rank();
                int lidx = team_member.league_rank();

		// Determine which run this team is doing - note this is a floor since integer division
		int run_id = lidx/docking_params.pop_size;

		// Have to declare this outside calc_energy since Solis-Wets has energy calc in a loop
		Coordinates calc_coords(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_atoms);
		
		Genotype genotype(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT),docking_params.num_of_genes);
		copy_genotype(team_member, docking_params.num_of_genes, genotype, current, lidx);

	//	team_member.team_barrier();

		// Get the current energy for each run
		float energy = calc_energy(team_member, docking_params, consts, calc_coords, genotype, run_id);

		// Copy to global views
                if( tidx == 0 ) {
                        current.energies(lidx) = energy;
                        docking_params.evals_of_new_entities(lidx) = 1;
                }
        });
}
