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
#ifndef GEN_ALGM_HPP
#define GEN_ALGM_HPP

template<class Device>
KOKKOS_INLINE_FUNCTION void perform_elitist_selection(const member_type& team_member, const Generation<Device>& current, const Generation<Device>& next, const DockingParams<Device>& docking_params);

template<class Device>
KOKKOS_INLINE_FUNCTION void crossover(const member_type& team_member, const Generation<Device>& current, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params, const int run_id, TenFloat randnums, TwoInt parents, Genotype offspring_genotype);

template<class Device>
KOKKOS_INLINE_FUNCTION void mutation(const member_type& team_member, const DockingParams<Device>& docking_params, const GeneticParams& genetic_params, Genotype offspring_genotype);

template<class Device>
void gen_alg_eval_new(Generation<Device>& current, Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,GeneticParams& genetic_params,Constants<Device>& consts);


#include "gen_algm.tpp"

#endif
