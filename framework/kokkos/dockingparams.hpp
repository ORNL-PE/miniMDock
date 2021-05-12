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
#ifndef DOCKINGPARAMS_HPP
#define DOCKINGPARAMS_HPP
// Docking parameters for Kokkos implementation
template<class Device>
struct DockingParams
{
        char            num_of_atoms;
        char            num_of_atypes;
        int             num_of_intraE_contributors;
        char            gridsize_x;
        char            gridsize_y;
        char            gridsize_z;
	unsigned int    g2;
	unsigned int    g3;
	float           grid_spacing;
	Kokkos::View<float*,Device> fgrids_write; // View for the copy to device
	Kokkos::View<const float*,Device,RandomAccess> fgrids; // View for the kernels (random access)
        int             rotbondlist_length;
        float           coeff_elec;
        float           coeff_desolv;
        Kokkos::View<int*,Device> evals_of_new_entities;
        Kokkos::View<unsigned int*,Device> prng_states;
        int             pop_size;
	float           smooth;
	float           qasp;

	// Used in ADA kernel
        int             num_of_genes;
        float           lsearch_rate;
        unsigned int    num_of_lsentities;
        unsigned int    max_num_of_iters;

	// Used in Solis-Wets kernel
	float base_dmov_mul_sqrt3;
	float base_dang_mul_sqrt3;
	unsigned int cons_limit;
	float rho_lower_bound;

	// Constructor
	DockingParams(const Liganddata& myligand_reference, const Gridinfo* mygrid, const Dockpars* mypars)
		: fgrids_write("fgrids_write", 4 * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2])),
		  evals_of_new_entities("evals_of_new_entities", mypars->pop_size * mypars->num_of_runs),
		  prng_states("prng_states",mypars->pop_size * mypars->num_of_runs * NUM_OF_THREADS_PER_BLOCK)
	{
		// Copy in scalars
		num_of_atoms  = ((char)  myligand_reference.num_of_atoms);
		num_of_atypes = ((char)  myligand_reference.num_of_atypes);
		num_of_intraE_contributors = ((int) myligand_reference.num_of_intraE_contributors);
		gridsize_x    = ((char)  mygrid->size_xyz[0]);
		gridsize_y    = ((char)  mygrid->size_xyz[1]);
		gridsize_z    = ((char)  mygrid->size_xyz[2]);
		g2 = gridsize_x * gridsize_y;
                g3 = gridsize_x * gridsize_y * gridsize_z;

		grid_spacing  = ((float) mygrid->spacing);
		rotbondlist_length = ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
		coeff_elec    = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
		coeff_desolv  = ((float) mypars->coeffs.AD4_coeff_desolv);
		pop_size      = mypars->pop_size;
		qasp            = mypars->qasp;
		smooth          = mypars->smooth;

		num_of_genes  = myligand_reference.num_of_rotbonds + 6;
		lsearch_rate    = mypars->lsearch_rate;
		if (lsearch_rate != 0.0f)
		{
			num_of_lsentities = (unsigned int) (mypars->lsearch_rate/100.0*mypars->pop_size + 0.5);
			max_num_of_iters  = (unsigned int) mypars->max_num_of_iters;

			// Used in Solis-Wets kernel
			base_dmov_mul_sqrt3 = mypars->base_dmov_mul_sqrt3;
			base_dang_mul_sqrt3 = mypars->base_dang_mul_sqrt3;
			cons_limit = (unsigned int) mypars->cons_limit;
			rho_lower_bound = mypars->rho_lower_bound;
		}

		// Point fgrids to fgrids_write
		fgrids = fgrids_write;

		// Create the randomization seeds here and send them to device
		Kokkos::View<unsigned int*,HostType> prng_seeds("prng_seeds",prng_states.extent(0)); // Could be mirror
		for (int i=0; i<prng_seeds.extent(0); i++)
#if defined (REPRO)
			prng_seeds(i) = 1u;
#else
			prng_seeds(i) = genseed(0u);
#endif
                Kokkos::deep_copy(prng_states, prng_seeds);
	}
};

#endif
