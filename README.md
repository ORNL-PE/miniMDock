# miniMDock 

A GPU based portable protein-ligand docking code, can be used for virtual drug discovery compound screens based on a molecular recognition model, that analysis a three-dimensional model of an interaction between a protein and a small molecule (ligand). The key calculation, namely, the Lamarckian genetic algorithm (LGA) combined with a local search  using a Solis-Wets based random optimization algorithm, are implemented using performance portable programming models (Kokkos) that enables porting to different emerging systems with minimal efforts, vendor-specific programming models such as CUDA for NVIDIA GPUs, and Heterogeneous-computing Interface for Portability (HIP) versions that can be applicable for AMD and NVIDIA GPUs.
The miniapp is developed based on [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU), can be used to evaluate the performance and portability of GPU-accelerated particle-grid based protein ligand docking programs on different computer architectures, and targets NVIDIA and AMD GPUs. The miniMDock is available under GNU GPL version 2 and/or LGPL version 2.1 licenses.

## Programming Base (API/ Model) 

* CUDA 
* HIP 
* Kokkos  

## Compilation 
### Compilation in General

```zsh 
make DEVICE=<TYPE> API=<SUPPORT> CARD=<VENDOR> NUMWI=<NWI>  
``` 
|  Prameters | Description            | Values                                             |
|:----------:|:----------------------:|:--------------------------------------------------:|
| `<TYPE>`   | Accelerator chosen     | `CPU`, `GPU`                                       |
| `<SUPPORT>`   | API chosen     | `CUDA`, `HIP`, `KOKKOS`                                       |
| `<VENDOR>`   | GPU vendor/type         | `NVIDIA`, `AMD`                                    |
| `<NWI>`    | work-group size | `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256` |


After successful compilation, the host binary **autodock_&lt;api&gt;_&lt;card&gt;_&lt;N&gt;wi** is placed under [bin](./bin).

| Binary-name portion | Description            | Values                                            |
|:-------------------:|:----------------------:|:-------------------------------------------------:|
| **&lt;api&gt;**    | Accelerator chosen     | `cuda` , `hip`, `kokkos`                                     |
| **&lt;card&gt;**    | Accelerator chosen     | `nv`, `amd`                                   |
| **&lt;N&gt;**       | work-group size | `1`, `2`, `4`, `8`,`16`, `32`, `64`, `128`, `256` |


### Programming model and platform based compilation
Steps to build the CUDA, HIP, and Kokkos versions of miniMDock, 
```diff 
+ CUDA version 
```
  * Prerequisites: [cuda](https://developer.nvidia.com/cuda-toolkit) 8.0 or above and at least [gcc](https://gcc.gnu.org/) 5.3 (choose based on cuda version).
  * Build: Define the environment varables GPU_LIBRARY_PATH and GPU_INCLUDE_PATH before build it.
  * Example: Build on summit. 
  ```zsh
  module load cuda 
  export GPU_LIBRARY_PATH=/sw/summit/cuda/10.1.243/lib64 
  export GPU_INCLUDE_PATH=/sw/summit/cuda/10.1.243/include 
  make API=CUDA 
  ```
 #
```diff 
+ HIP version 
```
  * Prerequisites: [rocm](https://github.com/RadeonOpenCompute/ROCm) 3.7 or above and at least [gcc](https://gcc.gnu.org/) 7.2 (choose based on rocm version).
  * Build: Define the environment varables HIP_PLATFORM, GPU_LIBRARY_PATH and GPU_INCLUDE_PATH before build it.
  * Example: Build on Frontier Center of Excellence (CoE).  
  ```zsh
  ROCMV=3.7.0 
  PLATFORM=hcc 
  module load rocm/${ROCMV}  
  export HIP_PLATFORM=${PLATFORM} 
  export GPU_LIBRARY_PATH=/opt/rocm-${ROCMV}/lib 
  export GPU_INCLUDE_PATH=/opt/rocm-${ROCMV}/include 
  make API=HIP CARD=AMD 
  ```
#
```diff 
+ Kokkos version 
```
  * Prerequisites: Built [kokkos](https://github.com/kokkos/kokkos) with the specific backend, [rocm](https://github.com/RadeonOpenCompute/ROCm) 3.7 or above, [cuda](https://developer.nvidia.com/cuda-toolkit) 8.0 or above, and [gcc](https://gcc.gnu.org/) 6.1 or above.
  * Build: Define the environment varables KOKKOS_SRC_DIR, KOKKOS_LIB_PATH and KOKKOS_INC_PATH before build it. 
  * Example: Build kokkos with HIP backend version on Frontier Center of Excellence (CoE). 
  ```zsh
  ROCMV=3.7.0 
  module load rocm/${ROCMV} cuda10.2/toolkit 
  export KOKKOS_SRC_DIR=/home/users/coe0179/kokkos 
  export KOKKOS_INC_PATH=/home/users/coe0179/kokkos/include/ 
  export KOKKOS_LIB_PATH=/home/users/coe0179/kokkos/lib64/ 
  make API=KOKKOS DEVICE=GPU  CARD=AMD
  ```
#
## Running
### Basic command 

```zsh 
./bin/autodock_<api>_<card>_<N>wi \ 
-lfile <ligand>.pdbqt \ 
-nrun <nruns> 
``` 

| Mandatory options | Description   | Value                     |
|:-----------------:|:-------------:|:-------------------------:|
| -lfile            |Ligand file    |&lt;ligand&gt;.pdbqt       |
| -nrun		    | # LGA runs    |&lt;nruns&gt;              |	

### Example 

```zsh 
./bin/autodock_hip_nv_64wi \ 
-lfile ./input/7cpa/7cpa_ligand.pdbqt  \ 
-nrun 10 
``` 
## References 
miniMDock development, discussion, and benchmark have been published in 2020 IEEE/ACM International Workshop on P3HPC. Please cite these references if you use miniMDock in your research: <br/>
<a id="1">[1]</a> M. Thavappiragasam, A. Scheinberg, W. Elwasif, O. Hernandez and A. Sedova, "Performance Portability of Molecular Docking Miniapp On Leadership Computing Platforms," 2020 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC), GA, USA, 2020, pp. 36-44, doi: 10.1109/P3HPC51967.2020.00009.
 

 
