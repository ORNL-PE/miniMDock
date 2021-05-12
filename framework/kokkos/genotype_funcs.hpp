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
#ifndef GENOTYPE_FUNCS_HPP
#define GENOTYPE_FUNCS_HPP

#include "common_typedefs.hpp"


// Perhaps these could be replaced with kokkos deep_copies, however it may require
// something sophisticated that isnt worth it unless the speedup is large

// global to local copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, Genotype genotype, const Generation<Device>& generation, int which_pop)
{
	int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, genotype_length),
        		[=] (int& idx) {
        	genotype[idx] = generation.conformations(offset + idx);
        });
}

// local to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, const Generation<Device>& generation, int which_pop, Genotype genotype)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, genotype_length),
                        [=] (int& idx) {
                generation.conformations(offset + idx) = genotype[idx];
        });
}

// local to local copy - note, not a template because Device isnt present.
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, Genotype genotype_copy, Genotype genotype)
{
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, genotype_length),
                        [=] (int& idx) {
                genotype_copy(idx) = genotype[idx];
        });
}

// global to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, const Generation<Device>& generation_copy, int which_pop_copy, const Generation<Device>& generation, int which_pop)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	int offset_copy = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop_copy;
        Kokkos::parallel_for (Kokkos::TeamThreadRange (team_member, genotype_length),
                        [=] (int& idx) {
                generation_copy.conformations(offset_copy + idx) = generation.conformations(offset + idx);
        });
}

#endif
