/*

miniMDock is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
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


#include <chrono>
#include <omp.h>

#if defined (N1WI)
	#define KNWI " -DN1WI "
#elif defined (N2WI)
	#define KNWI " -DN2WI "
#elif defined (N4WI)
	#define KNWI " -DN4WI "
#elif defined (N8WI)
	#define KNWI " -DN8WI "
#elif defined (N16WI)
	#define KNWI " -DN16WI "
#elif defined (N32WI)
	#define KNWI " -DN32WI "
#elif defined (N64WI)
	#define KNWI " -DN64WI "
#elif defined (N128WI)
	#define KNWI " -DN128WI "
#elif defined (N256WI)
		#define KNWI " -DN256WI "
#else
	#define KNWI	" -DN64WI "
#endif

#if defined (REPRO)
	#define REP " -DREPRO "
#else
	#define REP " "
#endif


#ifdef __APPLE__
	#define KGDB_GPU	" -g -cl-opt-disable "
#else
	#define KGDB_GPU	" -g -O0 -Werror -cl-opt-disable "
#endif
#define KGDB_CPU	" -g3 -Werror -cl-opt-disable "
// Might work in some (Intel) devices " -g -s " KRNL_FILE

#if defined (DOCK_DEBUG)
	#if defined (CPU_DEVICE)
		#define KGDB KGDB_CPU
	#elif defined (GPU_DEVICE)
		#define KGDB KGDB_GPU
	#endif
#else
	#define KGDB " -cl-mad-enable"
#endif


#define OPT_PROG INC KNWI REP KGDB

#include <vector>

#include "autostop.hpp"
#include "performdocking.h"
#include "correct_grad_axisangle.h"
#include "GpuData.h"
#include "kernels.hpp"

// Device kernels
//void SetKernelsGpuData(GpuData* pData);
//void GetKernelsGpuData(GpuData* pData);
/*
void gpu_calc_initpop(uint32_t blocks, uint32_t threadsPerBlock, float* pConformations_current, float* pEnergies_current);
void gpu_sum_evals(uint32_t blocks, uint32_t threadsPerBlock);
void gpu_gen_and_eval_newpops(
    uint32_t blocks,
    uint32_t threadsPerBlock,
    float* pMem_conformations_current,
    float* pMem_energies_current,
    float* pMem_conformations_next,
    float* pMem_energies_next
);

void gpu_perform_LS(
    uint32_t blocks,
    uint32_t threads,
    float* pMem_conformations_next,
	float* pMem_energies_next
);
*/

template <typename Clock, typename Duration1, typename Duration2>
double elapsed_seconds(std::chrono::time_point<Clock, Duration1> start,
                       std::chrono::time_point<Clock, Duration2> end)
{
  using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
  return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}



inline float average(float* average_sd2_N)
{
	if(average_sd2_N[2]<1.0f)
		return 0.0;
	return average_sd2_N[0]/average_sd2_N[2];
}

inline float stddev(float* average_sd2_N)
{
	if(average_sd2_N[2]<1.0f)
		return 0.0;
	float sq = average_sd2_N[1]*average_sd2_N[2]-average_sd2_N[0]*average_sd2_N[0];
	if((fabs(sq)<=0.000001) || (sq<0.0)) return 0.0;
	return sqrt(sq)/average_sd2_N[2];
}

void copy_map_to_gpu(	GpuTempData& tData,
			std::vector<Map>& all_maps,
			int t,
			int size_of_one_map)
{
//	std::copy();
        memcpy(tData.pMem_fgrids+t*size_of_one_map, all_maps[t].grid.data(), sizeof(float)*size_of_one_map);
//        tData.pMem_fgrids+t*size_of_one_map = all_maps[t].grid.data()
//	#pragma omp target enter data map(to: all_maps[t].grid.data())
}

void setup_gpu_for_docking(GpuData& cData, GpuTempData& tData)
{
    auto const t0 = std::chrono::steady_clock::now();

    // Initialize CUDA
    int device                                      = -1;
    const int gpuCount                                    = omp_get_num_devices();
    if (gpuCount == 0)
    {
        printf("No GPU devices found, exiting.\n");
        exit(-1);
    }
    if (cData.devnum>=gpuCount){
	printf("Error: Requested device %i does not exist (only %i devices available).\n",cData.devnum+1,gpuCount);
	exit(-1);
    }
    if (cData.devnum<0)
	printf("Error: Requested device %i is worng, device number should be >= 0", cData.devnum);
    else
    	#pragma omp target device(cData.devnum)
   
    auto const t1 = std::chrono::steady_clock::now();
    //printf("\nCUDA Setup time %fs\n", elapsed_seconds(t0 ,t1));
	size_t sz_interintra_const	= MAX_NUM_OF_ATOMS*sizeof(float) + 
					  MAX_NUM_OF_ATOMS*sizeof(uint32_t) +
					  MAX_NUM_OF_ATOMS*sizeof(uint32_t);

	size_t sz_intracontrib_const	= 3*MAX_INTRAE_CONTRIBUTORS*sizeof(uint32_t);

	size_t sz_intra_const		= ATYPE_NUM*sizeof(float) + 
					  ATYPE_NUM*sizeof(float) + 
					  ATYPE_NUM*sizeof(unsigned int) + 
					  ATYPE_NUM*sizeof(unsigned int) + 
				          MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*sizeof(float) + 
					  MAX_NUM_OF_ATYPES*sizeof(float);

	size_t sz_rotlist_const		= MAX_NUM_OF_ROTATIONS*sizeof(int);

	size_t sz_conform_const		= 3*MAX_NUM_OF_ATOMS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  4*MAX_NUM_OF_RUNS*sizeof(float);
    // Allocate kernel constant GPU memory
    cData.pKerconst_interintra = (kernelconstant_interintra *) calloc(sz_interintra_const, sizeof(*cData.pKerconst_interintra));
    cData.pKerconst_intracontrib = (kernelconstant_intracontrib *) calloc(sz_intracontrib_const, sizeof(*cData.pKerconst_intracontrib));
    cData.pKerconst_intra = (kernelconstant_intra *) calloc( sz_intra_const, sizeof(*cData.pKerconst_intra));
    cData.pKerconst_rotlist = (kernelconstant_rotlist *) calloc( sz_rotlist_const, sizeof(*cData.pKerconst_rotlist));
    cData.pKerconst_conform = (kernelconstant_conform *) calloc( sz_conform_const, sizeof(*cData.pKerconst_conform));


    // Allocate mem data
    cData.pMem_angle_const = (float*) calloc( 1000 * sizeof(float), sizeof(*cData.pMem_angle_const));
    cData.pMem_dependence_on_theta_const = (float*) calloc( 1000 * sizeof(float), sizeof(*cData.pMem_dependence_on_theta_const));
    cData.pMem_dependence_on_rotangle_const = (float*) calloc( 1000 * sizeof(float), sizeof(*cData.pMem_dependence_on_rotangle_const));
    cData.pMem_rotbonds_const = (int*) calloc( (2*MAX_NUM_OF_ROTBONDS*sizeof(int)), sizeof(*cData.pMem_rotbonds_const));
    cData.pMem_rotbonds_atoms_const = (int*)calloc( MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int), sizeof(*cData.pMem_rotbonds_atoms_const));
    cData.pMem_num_rotating_atoms_per_rotbond_const = (int*) calloc( MAX_NUM_OF_ROTBONDS*sizeof(int), sizeof(*cData.pMem_num_rotating_atoms_per_rotbond_const));

    #pragma omp target enter data map(to: cData.pKerconst_interintra, cData.pKerconst_intracontrib, cData.pKerconst_intra, cData.pKerconst_rotlist, cData.pKerconst_conform, \
     cData.pMem_angle_const, cData.pMem_dependence_on_theta_const, cData.pMem_dependence_on_rotangle_const, cData.pMem_rotbonds_const, cData.pMem_rotbonds_atoms_const, cData.pMem_num_rotating_atoms_per_rotbond_const)
    
    cData.pMem_angle_const = angle;
    cData.pMem_dependence_on_theta_const = dependence_on_theta;
    cData.pMem_dependence_on_rotangle_const = dependence_on_rotangle;

	// Allocate temporary data - JL TODO - Are these sizes correct?
/*    if(cData.preload_gridsize>0){
        status = cudaMalloc((void**)&(tData.pMem_fgrids), cData.preload_gridsize * (sizeof(float)) * (ATYPE_NUM+2));
	RTERROR(status, "pMem_fgrids: failed to allocate GPU memory.\n");
    }*/
    size_t size_populations = MAX_NUM_OF_RUNS * MAX_POPSIZE * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
    tData.pMem_conformations1 = (float*) calloc( size_populations, sizeof(*tData.pMem_conformations1));
    tData.pMem_conformations2 = (float*) calloc( size_populations, sizeof(*tData.pMem_conformations2));
	size_t size_energies = MAX_POPSIZE * MAX_NUM_OF_RUNS * sizeof(float);
    tData.pMem_energies1 = (float*) calloc( size_energies, sizeof(*tData.pMem_energies1));
    tData.pMem_energies2 = (float*) calloc( size_energies, sizeof(*tData.pMem_energies2));
    tData.pMem_evals_of_new_entities = (int*) calloc( MAX_POPSIZE*MAX_NUM_OF_RUNS*sizeof(int), sizeof(*tData.pMem_evals_of_new_entities));

	size_t size_evals_of_runs = MAX_NUM_OF_RUNS*sizeof(int);
    //--- Have to be changed to unified memory
   //#pragma omp requires unified_shared_memory
   { 
    tData.pMem_gpu_evals_of_runs = (int*) calloc( size_evals_of_runs, sizeof(tData.pMem_gpu_evals_of_runs));
    #pragma omp target enter data map(to: tData.pMem_gpu_evals_of_runs)
   }
	size_t blocksPerGridForEachEntity = MAX_POPSIZE * MAX_NUM_OF_RUNS;
	size_t size_prng_seeds = blocksPerGridForEachEntity * NUM_OF_THREADS_PER_BLOCK * sizeof(unsigned int);
    tData.pMem_prng_states = (uint32_t *) calloc(size_prng_seeds, sizeof(tData.pMem_prng_states));
    
    #pragma omp target enter data map(to: tData.pMem_conformations1, tData.pMem_conformations2, tData.pMem_energies1, tData.pMem_energies2, \
     tData.pMem_evals_of_new_entities, tData.pMem_prng_states )

}
void finish_gpu_from_docking(GpuData& cData, GpuTempData& tData)
{
    // Release all Device objects
	// Constant objects
    free(cData.pKerconst_interintra);
    free(cData.pKerconst_intracontrib);
    free(cData.pKerconst_intra);
    free(cData.pKerconst_rotlist);
    free(cData.pKerconst_conform);
    free(cData.pMem_rotbonds_const);
    free(cData.pMem_rotbonds_atoms_const);
    free(cData.pMem_num_rotating_atoms_per_rotbond_const);
    free(cData.pMem_angle_const);
    free(cData.pMem_dependence_on_theta_const);
    free(cData.pMem_dependence_on_rotangle_const);
    
    // Non-constant
    free(tData.pMem_fgrids);
    free(tData.pMem_conformations1);
    free(tData.pMem_conformations2);
    free(tData.pMem_energies1);
    free(tData.pMem_energies2);
    free(tData.pMem_evals_of_new_entities);
    free(tData.pMem_gpu_evals_of_runs);
    free(tData.pMem_prng_states);
}

int docking_with_gpu(   const Gridinfo*  	mygrid,
                        float*      		cpu_floatgrids,
                        Dockpars*   		mypars,
                        const Liganddata*   myligand_init,
                        const Liganddata* 	myxrayligand,
                        const int*        	argc,
                        char**      		argv,
	                SimulationState&	sim_state,
                        GpuData& cData,
			GpuTempData& tData)
/* The function performs the docking algorithm and generates the corresponding result files.
parameter mygrid:
		describes the grid
		filled with get_gridinfo()
parameter cpu_floatgrids:
		points to the memory region containing the grids
		filled with get_gridvalues_f()
parameter mypars:
		describes the docking parameters
		filled with get_commandpars()
parameter myligand_init:
		describes the ligands
		filled with get_liganddata()
parameter myxrayligand:
		describes the xray ligand
		filled with get_xrayliganddata()
parameters argc and argv:
		are the corresponding command line arguments parameter clock_start_program:
		contains the state of the clock tick counter at the beginning of the program
filled with clock() */
{
    auto const t1 = std::chrono::steady_clock::now();

	Liganddata myligand_reference;

	float* cpu_init_populations;
	float* cpu_final_populations;
	unsigned int* cpu_prng_seeds;

	size_t size_floatgrids;
	size_t size_populations;
	size_t size_energies;
	size_t size_prng_seeds;
	size_t size_evals_of_runs;

	int threadsPerBlock;
	int blocksPerGridForEachEntity;
	int blocksPerGridForEachRun;
	int blocksPerGridForEachLSEntity;
	int blocksPerGridForEachGradMinimizerEntity;

	int generation_cnt;
	int i;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking;
	clock_t	clock_stop_docking;

	//setting number of blocks and threads
	threadsPerBlock = NUM_OF_THREADS_PER_BLOCK;
	blocksPerGridForEachEntity = mypars->pop_size * mypars->num_of_runs;
	blocksPerGridForEachRun = mypars->num_of_runs;

	//allocating CPU memory for initial populations
	size_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
	sim_state.cpu_populations.resize(size_populations);
	memset(sim_state.cpu_populations.data(), 0, size_populations);

	//allocating CPU memory for results
	size_energies = mypars->pop_size * mypars->num_of_runs * sizeof(float);
	sim_state.cpu_energies.resize(size_energies);
	cpu_init_populations = sim_state.cpu_populations.data();
	cpu_final_populations = sim_state.cpu_populations.data();

	//allocating memory in CPU for reference orientation angles
	sim_state.cpu_ref_ori_angles.resize(mypars->num_of_runs*3);

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_init_populations, sim_state.cpu_ref_ori_angles.data(), &myligand_reference, mygrid);

	//allocating memory in CPU for pseudorandom number generator seeds and
	//generating them (seed for each thread during GA)
	size_prng_seeds = blocksPerGridForEachEntity * threadsPerBlock * sizeof(unsigned int);
	cpu_prng_seeds = (unsigned int*) malloc(size_prng_seeds);

	LocalRNG r;

	for (i=0; i<blocksPerGridForEachEntity*threadsPerBlock; i++)
		cpu_prng_seeds[i] = r.random_uint();

	//allocating memory in CPU for evaluation counters
	size_evals_of_runs = mypars->num_of_runs*sizeof(int);
	sim_state.cpu_evals_of_runs.resize(size_evals_of_runs);
	memset(sim_state.cpu_evals_of_runs.data(), 0, size_evals_of_runs);

	//preparing the constant data fields for the GPU
	// ----------------------------------------------------------------------
	// The original function does CUDA calls initializing const Kernel data.
	// We create a struct to hold those constants
	// and return them <here> (<here> = where prepare_const_fields_for_gpu() is called),
	// so we can send them to Kernels from <here>, instead of from calcenergy.cpp as originally.
	// ----------------------------------------------------------------------
	// Constant struct

/*
	kernelconstant KerConst;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars, cpu_ref_ori_angles, &KerConst) == 1)
		return 1;
*/
    //GpuData cData; //JL Now pass this in
    	kernelconstant_interintra 	KerConst_interintra;
	kernelconstant_intracontrib 	KerConst_intracontrib;
	kernelconstant_intra 		KerConst_intra;
	kernelconstant_rotlist 	        KerConst_rotlist;
    	kernelconstant_conform 	        KerConst_conform;
	kernelconstant_grads            KerConst_grads;    
    
    
	if (prepare_const_fields_for_gpu(&myligand_reference, mypars, sim_state.cpu_ref_ori_angles.data(), 
					 &KerConst_interintra, 
					 &KerConst_intracontrib, 
					 &KerConst_intra, 
					 &KerConst_rotlist, 
					 &KerConst_conform,
					 &KerConst_grads) == 1) {
		return 1;
	}

	size_t sz_interintra_const	= MAX_NUM_OF_ATOMS*sizeof(float) + 
					  MAX_NUM_OF_ATOMS*sizeof(uint32_t) + 
                                          MAX_NUM_OF_ATOMS*sizeof(uint32_t);

	size_t sz_intracontrib_const	= 3*MAX_INTRAE_CONTRIBUTORS*sizeof(uint32_t);

	size_t sz_intra_const		= ATYPE_NUM*sizeof(float) + 
					  			  ATYPE_NUM*sizeof(float) + 
					  			  ATYPE_NUM*sizeof(unsigned int) + 
					  			  ATYPE_NUM*sizeof(unsigned int) + 
			    			      MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  			  MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float) + 
					  			  MAX_NUM_OF_ATYPES*sizeof(float) + 
					  			  MAX_NUM_OF_ATYPES*sizeof(float);

	size_t sz_rotlist_const		= MAX_NUM_OF_ROTATIONS*sizeof(int);

	size_t sz_conform_const		= 3*MAX_NUM_OF_ATOMS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  3*MAX_NUM_OF_ROTBONDS*sizeof(float) + 
					  4*MAX_NUM_OF_RUNS*sizeof(float);

    // Upload kernel constant data - JL FIXME - Can these be moved once?
    cData.pKerconst_interintra = &KerConst_interintra;
    cData.pKerconst_intracontrib = &KerConst_intracontrib;
    cData.pKerconst_intra = &KerConst_intra;
    cData.pKerconst_rotlist = &KerConst_rotlist;
    cData.pKerconst_conform = &KerConst_conform;

    #pragma omp target enter data map(to: cData.pKerconst_interintra, cData.pKerconst_intracontrib, cData.pKerconst_intra, cData.pKerconst_rotlist, cData.pKerconst_conform)

    //allocating GPU memory for populations, floatgirds,
    //energies, evaluation counters and random number generator states
    size_floatgrids = 4 * (sizeof(float)) * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2]);    
    //if(cData.preload_gridsize==0){
    	tData.pMem_fgrids = (float*) calloc( size_floatgrids, sizeof(tData.pMem_fgrids) );
    //    #pragma omp target enter data map(to: tData.pMem_fgrids)
   // }

    // Flippable pointers
    float* pMem_conformations_current = tData.pMem_conformations1;
    float* pMem_conformations_next = tData.pMem_conformations2;
    float* pMem_energies_current = tData.pMem_energies1;
    float* pMem_energies_next = tData.pMem_energies2;
    
    // Set constant pointers
    cData.pMem_fgrids = tData.pMem_fgrids;
    cData.pMem_evals_of_new_entities = tData.pMem_evals_of_new_entities;
    cData.pMem_gpu_evals_of_runs = tData.pMem_gpu_evals_of_runs;
    cData.pMem_prng_states = tData.pMem_prng_states;
    
    tData.pMem_fgrids = cpu_floatgrids;
    pMem_conformations_current = cpu_init_populations;
    tData.pMem_gpu_evals_of_runs = sim_state.cpu_evals_of_runs.data();
    tData.pMem_prng_states = cpu_prng_seeds;

    // Upload data
    #pragma omp target enter data map(to: tData.pMem_fgrids, pMem_conformations_current, tData.pMem_gpu_evals_of_runs, tData.pMem_prng_states)
    //if(!floatgrids_preloaded){
    //}

	//preparing parameter struct
	cData.dockpars.num_of_atoms                 = myligand_reference.num_of_atoms;
	cData.dockpars.num_of_atypes                = myligand_reference.num_of_atypes;
	cData.dockpars.num_of_map_atypes	    = mygrid->num_of_map_atypes;
	cData.dockpars.num_of_intraE_contributors   = ((int) myligand_reference.num_of_intraE_contributors);
	cData.dockpars.gridsize_x                   = mygrid->size_xyz[0];
	cData.dockpars.gridsize_y                   = mygrid->size_xyz[1];
	cData.dockpars.gridsize_z                   = mygrid->size_xyz[2];
        cData.dockpars.gridsize_x_times_y           = cData.dockpars.gridsize_x * cData.dockpars.gridsize_y; 
        cData.dockpars.gridsize_x_times_y_times_z   = cData.dockpars.gridsize_x * cData.dockpars.gridsize_y * cData.dockpars.gridsize_z;     
	cData.dockpars.grid_spacing                 = ((float) mygrid->spacing);
	cData.dockpars.rotbondlist_length           = ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
	cData.dockpars.coeff_elec                   = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
	cData.dockpars.coeff_desolv                 = ((float) mypars->coeffs.AD4_coeff_desolv);
	cData.dockpars.pop_size                     = mypars->pop_size;
	cData.dockpars.num_of_genes                 = myligand_reference.num_of_rotbonds + 6;
	// Notice: dockpars.tournament_rate, dockpars.crossover_rate, dockpars.mutation_rate
	// were scaled down to [0,1] in host to reduce number of operations in device
	cData.dockpars.tournament_rate              = mypars->tournament_rate/100.0f; 
	cData.dockpars.crossover_rate               = mypars->crossover_rate/100.0f;
	cData.dockpars.mutation_rate                = mypars->mutation_rate/100.f;
	cData.dockpars.abs_max_dang                 = mypars->abs_max_dang;
	cData.dockpars.abs_max_dmov                 = mypars->abs_max_dmov;
	cData.dockpars.qasp 		                = mypars->qasp;
	cData.dockpars.smooth 	                    = mypars->smooth;
	cData.dockpars.lsearch_rate                 = mypars->lsearch_rate;

	if (cData.dockpars.lsearch_rate != 0.0f) 
	{
		cData.dockpars.num_of_lsentities        = (unsigned int) (mypars->lsearch_rate/100.0*mypars->pop_size + 0.5);
		cData.dockpars.rho_lower_bound          = mypars->rho_lower_bound;
		cData.dockpars.base_dmov_mul_sqrt3      = mypars->base_dmov_mul_sqrt3;
		cData.dockpars.base_dang_mul_sqrt3      = mypars->base_dang_mul_sqrt3;
		cData.dockpars.cons_limit               = (unsigned int) mypars->cons_limit;
		cData.dockpars.max_num_of_iters         = (unsigned int) mypars->max_num_of_iters;

		// The number of entities that undergo Solis-Wets minimization,
		blocksPerGridForEachLSEntity = cData.dockpars.num_of_lsentities * mypars->num_of_runs;

		// The number of entities that undergo any gradient-based minimization,
		// by default, it is the same as the number of entities that undergo the Solis-Wets minimizer
		blocksPerGridForEachGradMinimizerEntity = cData.dockpars.num_of_lsentities * mypars->num_of_runs;

		// Enable only for debugging.
		// Only one entity per reach run, undergoes gradient minimization
		//blocksPerGridForEachGradMinimizerEntity = mypars->num_of_runs;
	}

	
	char method_chosen[64]; // 64 chars will be enough for this message as mypars->ls_method is 4 chars at the longest
	if(strcmp(mypars->ls_method, "sw") == 0){
		strcpy(method_chosen,"Solis-Wets (sw)");
	}
	else{
		printf("Error: LS method %s is not (yet) supported in the Cuda version.\n",mypars->ls_method);
		exit(-1);
	}
	printf("Local-search chosen method is: %s\n", (cData.dockpars.lsearch_rate == 0.0f)? "GA" : method_chosen);

	if((mypars->initial_sw_generations>0) && (strcmp(mypars->ls_method, "sw") != 0))
		printf("Using Solis-Wets (sw) for the first %d generations.\n",mypars->initial_sw_generations);


	/*
	printf("dockpars.num_of_intraE_contributors:%u\n", dockpars.num_of_intraE_contributors);
	printf("dockpars.rotbondlist_length:%u\n", dockpars.rotbondlist_length);
	*/

	clock_start_docking = clock();

	//SetKernelsGpuData(&cData);

#ifdef DOCK_DEBUG
	printf("\n");
	// Main while-loop iterarion counter
	unsigned int ite_cnt = 0;
#endif

	/*
	// Added for printing intracontributor_pairs (autodockdevpy)
	for (unsigned int intrapair_cnt=0; 
			  intrapair_cnt<dockpars.num_of_intraE_contributors;
			  intrapair_cnt++) {
		if (intrapair_cnt == 0) {
			printf("%-10s %-10s %-10s\n", "#pair", "#atom1", "#atom2");
		}

		printf ("%-10u %-10u %-10u\n", intrapair_cnt,
					    KerConst.intraE_contributors_const[3*intrapair_cnt],
					    KerConst.intraE_contributors_const[3*intrapair_cnt+1]);
	}
	*/

	// Kernel1
	uint32_t kernel1_gxsize = blocksPerGridForEachEntity;
	uint32_t kernel1_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8lu %10s %4u\n", "K_INIT", "gSize: ", kernel1_gxsize, "lSize: ", kernel1_lxsize); fflush(stdout);
#endif
	// End of Kernel1

	// Kernel2
  	uint32_t kernel2_gxsize = blocksPerGridForEachRun;
  	uint32_t kernel2_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8lu %10s %4u\n", "K_EVAL", "gSize: ", kernel2_gxsize, "lSize: ",  kernel2_lxsize); fflush(stdout);
#endif
	// End of Kernel2

	// Kernel4
  	uint32_t kernel4_gxsize = blocksPerGridForEachEntity;
  	uint32_t kernel4_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("%-25s %10s %8u %10s %4u\n", "K_GA_GENERATION", "gSize: ",  kernel4_gxsize, "lSize: ", kernel4_lxsize); fflush(stdout);
#endif
	// End of Kernel4

    uint32_t kernel3_gxsize, kernel3_lxsize;
	if (cData.dockpars.lsearch_rate != 0.0f) {

		if ((strcmp(mypars->ls_method, "sw") == 0) || (mypars->initial_sw_generations>0)) {
			// Kernel3
			kernel3_gxsize = blocksPerGridForEachLSEntity;
			kernel3_lxsize = threadsPerBlock;
  			#ifdef DOCK_DEBUG
	  		printf("%-25s %10s %8u %10s %4u\n", "K_LS_SOLISWETS", "gSize: ", kernel3_gxsize, "lSize: ", kernel3_lxsize); fflush(stdout);
  			#endif
			// End of Kernel3
		}
	} // End if (dockpars.lsearch_rate != 0.0f)

	// Kernel1
	#ifdef DOCK_DEBUG
		printf("\nExecution starts:\n\n");
		printf("%-25s", "\tK_INIT");fflush(stdout);
        cudaDeviceSynchronize();
	#endif
    gpu_calc_initpop(kernel1_gxsize, kernel1_lxsize, pMem_conformations_current, pMem_energies_current);
	//runKernel1D(command_queue,kernel1,kernel1_gxsize,kernel1_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
        cudaDeviceSynchronize();
		printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel1

	// Kernel2
	#ifdef DOCK_DEBUG
		printf("%-25s", "\tK_EVAL");fflush(stdout);
	#endif
	//runKernel1D(command_queue,kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
    gpu_sum_evals(kernel2_gxsize, kernel2_lxsize);
	#ifdef DOCK_DEBUG
        cudaDeviceSynchronize();
		printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel2
	// ===============================================================================



	#if 0
	generation_cnt = 1;
	#endif
	generation_cnt = 0;
	bool first_time = true;
	float* energies;
	float threshold = 1<<24;
	float threshold_used;
	float thres_stddev = threshold;
	float curr_avg = -(1<<24);
	float curr_std = thres_stddev;
	float prev_avg = 0.0;
	unsigned int roll_count = 0;
	float rolling[4*3];
	float rolling_stddev;
	memset(&rolling[0],0,12*sizeof(float));
	unsigned int bestN = 1;
	unsigned int Ntop = mypars->pop_size;
	unsigned int Ncream = Ntop / 10;
	float delta_energy = 2.0 * thres_stddev / Ntop;
	float overall_best_energy;
	unsigned int avg_arr_size = (Ntop+1)*3;
	float average_sd2_N[avg_arr_size];
	unsigned long total_evals;

	auto const t2 = std::chrono::steady_clock::now();
	printf("\nRest of Setup time %fs\n", elapsed_seconds(t1 ,t2));

	//print progress bar
	AutoStop autostop(mypars->pop_size, mypars->num_of_runs, mypars->stopstd, mypars->as_frequency);
#ifndef DOCK_DEBUG
        if (mypars->autostop)
        {
                autostop.print_intro(mypars->num_of_generations, mypars->num_of_energy_evals);
        }
	else
	{
		printf("\nExecuting docking runs:\n");
		printf("        20%%        40%%       60%%       80%%       100%%\n");
		printf("---------+---------+---------+---------+---------+\n");
	}
#endif
	curr_progress_cnt = 0;

	// -------- Replacing with memory maps! ------------
	while ((progress = check_progress(tData.pMem_gpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
	// -------- Replacing with memory maps! ------------
	{
		if (mypars->autostop) {
                        if (generation_cnt % mypars->as_frequency == 0) {
				#pragma omp target exit data map(from: pMem_energies_current)
				//#pragma omp target exit data map(from: sim_state.cpu_energies.data())
                                if (autostop.check_if_satisfactory(generation_cnt, sim_state.cpu_energies.data(), total_evals))
                                        break; // Exit loop
                        }
		}
		else
		{
#ifdef DOCK_DEBUG
			ite_cnt++;
			printf("\nLGA iteration # %u\n", ite_cnt);
			fflush(stdout);
#endif
			//update progress bar (bar length is 50)
			new_progress_cnt = (int) (progress/2.0+0.5);
			if (new_progress_cnt > 50)
				new_progress_cnt = 50;
			while (curr_progress_cnt < new_progress_cnt) {
				curr_progress_cnt++;
#ifndef DOCK_DEBUG
				printf("*");
#endif
				fflush(stdout);
			}
		}
		// Kernel4
		#ifdef DOCK_DEBUG
			printf("%-25s", "\tK_GA_GENERATION");fflush(stdout);
		#endif
        
		//runKernel1D(command_queue,kernel4,kernel4_gxsize,kernel4_lxsize,&time_start_kernel,&time_end_kernel);
        gpu_gen_and_eval_newpops(kernel4_gxsize, kernel4_lxsize, pMem_conformations_current, pMem_energies_current, pMem_conformations_next, pMem_energies_next);
		#ifdef DOCK_DEBUG
			printf("%15s", " ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel4
		if (cData.dockpars.lsearch_rate != 0.0f) {
			if (strcmp(mypars->ls_method, "sw")  == 0 ) {
				// Kernel3
				#ifdef DOCK_DEBUG
					printf("%-25s", "\tK_LS_SOLISWETS");fflush(stdout);
				#endif
				//runKernel1D(command_queue,kernel3,kernel3_gxsize,kernel3_lxsize,&time_start_kernel,&time_end_kernel);
                gpu_perform_LS(kernel3_gxsize, kernel3_lxsize, pMem_conformations_next, pMem_energies_next);                
				#ifdef DOCK_DEBUG
					printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel3
			}
		} // End if (dockpars.lsearch_rate != 0.0f)
		// -------- Replacing with memory maps! ------------
		// -------- Replacing with memory maps! ------------
		// Kernel2
		#ifdef DOCK_DEBUG
			printf("%-25s", "\tK_EVAL");fflush(stdout);
		#endif
		//runKernel1D(command_queue,kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
        gpu_sum_evals(kernel2_gxsize, kernel2_lxsize);       
        
		#ifdef DOCK_DEBUG
			printf("%15s" ," ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel2
		// ===============================================================================
		// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
		//map_cpu_evals_of_runs = (int*) memMap(command_queue, mem_gpu_evals_of_runs, CL_MAP_READ, size_evals_of_runs);
#else
           #pragma omp target exit data map(from: pMem_gpu_evals_of_runs)
#endif
		// -------- Replacing with memory maps! ------------
		generation_cnt++;
		// ----------------------------------------------------------------------
		// ORIGINAL APPROACH: switching conformation and energy pointers (Probably the best approach, restored)
		// CURRENT APPROACH:  copy data from one buffer to another, pointers are kept the same
		// IMPROVED CURRENT APPROACH
		// Kernel arguments are changed on every iteration
		// No copy from dev glob memory to dev glob memory occurs
		// Use generation_cnt as it evolves with the main loop
		// No need to use tempfloat
		// No performance improvement wrt to "CURRENT APPROACH"

		// Kernel args exchange regions they point to
		// But never two args point to the same region of dev memory
		// NO ALIASING -> use restrict in Kernel
        
        // Flip conformation and energy pointers
        float* pTemp;
        pTemp = pMem_conformations_current;
        pMem_conformations_current = pMem_conformations_next;
        pMem_conformations_next = pTemp;
        pTemp = pMem_energies_current;
        pMem_energies_current = pMem_energies_next;
        pMem_energies_next = pTemp;
               
		// ----------------------------------------------------------------------
		#ifdef DOCK_DEBUG
			printf("\tProgress %.3f %%\n", progress);
			fflush(stdout);
		#endif
	} // End of while-loop

    auto const t3 = std::chrono::steady_clock::now();
    printf("\nDocking time %fs\n", elapsed_seconds(t2, t3));


	clock_stop_docking = clock();
	if (mypars->autostop==0)
	{
		//update progress bar (bar length is 50)mem_num_of_rotatingatoms_per_rotbond_const
		while (curr_progress_cnt < 50) {
			curr_progress_cnt++;
			printf("*");
			fflush(stdout);
		}
	}

	// ===============================================================================
	// Modification based on:
	// http://www.cc.gatech.edu/~vetter/keeneland/tutorial-2012-02-20/08-opencl.pdf
	// ===============================================================================
	//processing results
    #pragma omp target exit data map(from: pMem_conformations_current, pMem_energies_current) 
    //#pragma omp target exit data map(from: cpu_final_populations, sim_state.cpu_energies.data()) 

    //JL TODO - Should the tData arrays be zeroed before the next run?

        // Final autostop statistics output
        if (mypars->autostop) autostop.output_final_stddev(generation_cnt, sim_state.cpu_energies.data(), total_evals);

	printf("\n");
#if defined (DOCK_DEBUG)
	for (int cnt_pop=0;cnt_pop<size_populations/sizeof(float);cnt_pop++)
		printf("total_num_pop: %u, cpu_final_populations[%u]: %f\n",(unsigned int)(size_populations/sizeof(float)),cnt_pop,cpu_final_populations[cnt_pop]);
	for (int cnt_pop=0;cnt_pop<size_energies/sizeof(float);cnt_pop++)
		printf("total_num_energies: %u, cpu_energies[%u]: %f\n",    (unsigned int)(size_energies/sizeof(float)),cnt_pop,sim_state.cpu_energies[cnt_pop]);
#endif

	// Assign simulation results to sim_state
	sim_state.myligand_reference = myligand_reference;
	sim_state.generation_cnt = generation_cnt;
	sim_state.sec_per_run = ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs;
	sim_state.total_evals = total_evals;

    
    
	free(cpu_prng_seeds);

    auto const t4 = std::chrono::steady_clock::now();
    printf("\nShutdown time %fs\n", elapsed_seconds(t3, t4));
	return 0;
}

double check_progress(int* evals_of_runs, int generation_cnt, int max_num_of_evals, int max_num_of_gens, int num_of_runs, unsigned long &total_evals)
//The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
//parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
//the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
//generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
//than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	/*	Stops if every run reached the number of evals or number of generations

	int runs_finished;
	int i;

	runs_finished = 0;
	for (i=0; i<num_of_runs; i++)
		if (evals_of_runs[i] >= max_num_of_evals)
			runs_finished++;

	if ((runs_finished >= num_of_runs) || (generation_cnt >= max_num_of_gens))
		return 1;
	else
		return 0;
        */

	//Stops if the sum of evals of every run reached the sum of the total number of evals

	int i;
	double evals_progress;
	double gens_progress;

	//calculating progress according to number of runs
	total_evals = 0;
	for (i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	//calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}

