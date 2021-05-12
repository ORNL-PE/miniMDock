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
#ifndef CALCENERGY_HPP
#define CALCENERGY_HPP

#include "float4struct.hpp"
#include "common_typedefs.hpp"

template<class Device>
KOKKOS_INLINE_FUNCTION void get_atom_pos(const int atom_id, const Conform<Device>& conform, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION void rotate_atoms(const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, Genotype genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_intermolecular_energy(const int atom_id, const DockingParams<Device>& dock_params, const InterIntra<Device>& interintra, const Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_intramolecular_energy(const int contributor_counter, const DockingParams<Device>& dock_params, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_energy(const member_type& team_member, const DockingParams<Device>& docking_params, const Constants<Device>& consts, Coordinates calc_coords, Genotype genotype, int run_id);

#include "calcenergy.tpp"

#endif
