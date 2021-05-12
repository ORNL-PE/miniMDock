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
#ifndef COMMON_TYPEDEFS_HPP
#define COMMON_TYPEDEFS_HPP

#include "float4struct.hpp"

// View type on scratch memory that is used in energy and gradient calculations
// Coordinates of all atoms
typedef Kokkos::View<float4struct*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Coordinates;

// Gradient (inter/intra, xyz, num atoms)
typedef Kokkos::View<float**,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> AtomGradients;

// Genotype
typedef Kokkos::View<float*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Genotype;

// Identical to Genotype, but for auxiliary arrays (e.g. gradient) that arent technically genotypes themselves. To avoid confusion, shouldnt be labeled as a genotype
typedef Kokkos::View<float*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> GenotypeAux;

// Array of length team_size for use in perform_elitist_selection
typedef Kokkos::View<float[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamFloat;
typedef Kokkos::View<int[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamInt;

// Arrays of different fixed sizes (maybe unnecessary but fixed probably performs better so use it if length is known at compile time)
typedef Kokkos::View<bool[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneBool;
typedef Kokkos::View<int[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneInt;
typedef Kokkos::View<int[2],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TwoInt;
typedef Kokkos::View<int[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourInt;
typedef Kokkos::View<float[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneFloat;
typedef Kokkos::View<float[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourFloat;
typedef Kokkos::View<float[10],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TenFloat;

#endif
