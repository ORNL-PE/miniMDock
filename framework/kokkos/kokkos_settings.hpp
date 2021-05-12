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
#ifndef SPACE_SETTINGS_HPP
#define SPACE_SETTINGS_HPP
#include <Kokkos_Core.hpp>

// Declare the memory and execution spaces.

#ifdef USE_OMPTARGET
 using MemSpace = Kokkos::Experimental::OpenMPTargetSpace;
 using ExSpace = Kokkos::Experimental::OpenMPTarget;
#else
 #ifdef USE_GPU
  #ifdef CARD_AMD
   using MemSpace = Kokkos::Experimental::HIPSpace;
   using ExSpace = Kokkos::Experimental::HIP;
  #else
   using MemSpace = Kokkos::CudaSpace;
   using MemSpace_UVM = Kokkos::CudaUVMSpace;
   using ExSpace = Kokkos::Cuda;
   using DeviceType_UVM = Kokkos::Device<ExSpace,MemSpace_UVM>;
  #endif
 #endif
#endif

#ifdef USE_OMP
 using CPUSpace = Kokkos::HostSpace;
 using CPUExec  = Kokkos::OpenMP;
#else
 using CPUSpace = Kokkos::HostSpace;
 using CPUExec = Kokkos::Serial;
#endif

using DeviceType = Kokkos::Device<ExSpace,MemSpace>;

// Designate a CPU-specific Memory and Execution space
using HostType = Kokkos::Device<CPUExec,CPUSpace>;

// TODO: The two typedefs below use ExSpace, which negates the point of templating everything since now
// the kernels will only run in ExSpace. But I don't see a clean way to solve that at the moment - ALS
// Set up member_type for device here so it can be passed as function argument
typedef Kokkos::TeamPolicy<ExSpace>::member_type member_type;

// Set up scratch space (short-term memory for each team)
typedef ExSpace::scratch_memory_space ScratchSpace;

// Set up unmanaged kokkos views to wrap around C-style arrays for deep copies
typedef Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FloatView1D;
typedef Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> IntView1D;
typedef Kokkos::View<unsigned int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnsignedIntView1D;

// Short hand for RandomAccess memory
using RandomAccess = Kokkos::MemoryTraits<Kokkos::RandomAccess>;
#endif
