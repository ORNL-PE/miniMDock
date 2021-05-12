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

#include <vector>
#include <Kokkos_Core.hpp>

#include "defines.h"
#include "correct_grad_axisangle.h"
#include "autostop.hpp"
#include "performdocking.h"

#include "kokkos_settings.hpp"
#include "dockingparams.hpp"
#include "geneticparams.hpp"
#include "kernelconsts.hpp"
#include "generation.hpp"
#include "prepare_const_fields.hpp"
#include "calc_init_pop.hpp"

#include "local_search.hpp"
#include "gen_algm.hpp"
#include "evals.hpp"
//#include "kdefines.hpp"

// From ./kokkos
/*
#include "kdefines.hpp"
*/

inline void checkpoint(const char* input)
{
#ifdef DOCK_DEBUG
	printf("\n");
	printf(input);
	fflush(stdout);
#endif
}

int docking_with_gpu(const Gridinfo*		mygrid,
		           Kokkos::View<float*,HostType>& fgrids_h,
                           Dockpars*		mypars,
		     const Liganddata*		myligand_init,
		     const Liganddata*		myxrayligand,
    //                       Profile&             profile,
		     const int*			argc,
			   char**		argv)
/* The function performs the docking algorithm and generates the corresponding result files.
parameter mygrid:
		describes the grid
		filled with get_gridinfo()
parameter fgrids_h:
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
		are the corresponding command line arguments parameter */
{
	//------------------------------- SETUP --------------------------------------//

	// Note - Kokkos views initialized to 0 by default
	Kokkos::View<float*,HostType> populations_h(  "populations_h",   mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM);
	Kokkos::View<float*,HostType> energies_h(     "energies_h",      mypars->num_of_runs * mypars->pop_size);
	Kokkos::View<int*,  HostType> evals_of_runs_h("evals_of_runs_h", mypars->num_of_runs);

	std::vector<Ligandresult> cpu_result_ligands(mypars->num_of_runs); // Ligand results
	std::vector<float> cpu_ref_ori_angles(mypars->num_of_runs*3); // Reference orientation angles

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	Liganddata myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, populations_h.data(), cpu_ref_ori_angles.data(), &myligand_reference, mygrid);

	genseed(time(NULL));	//initializing seed generator

	// Initialize GeneticParams (broken out of docking params since they relate to the genetic algorithm, not the docking per se
	GeneticParams genetic_params(mypars);

	// Initialize the objects containing the two alternating generations
	Generation<DeviceType> odd_generation(mypars->pop_size * mypars->num_of_runs);
	Generation<DeviceType> even_generation(mypars->pop_size * mypars->num_of_runs);

	// Odd generation gets the initial population copied in
	Kokkos::deep_copy(odd_generation.conformations, populations_h);

	// Evals of runs on device (for kernel2)
	Kokkos::View<int*,DeviceType> evals_of_runs("evals_of_runs",mypars->num_of_runs);

	// Declare the constant arrays on host and device
	ConstantsW<HostType> consts_h;
	ConstantsW<DeviceType> consts_d;
	Constants<DeviceType> consts;

	// Initialize host constants
	// WARNING - Changes myligand_reference !!! - ALS
	
        if (prepare_const_fields(myligand_reference, mypars, cpu_ref_ori_angles.data(), consts_h) == 1) {
                return 1;
        }
	prepare_axis_correction(angle, dependence_on_theta, dependence_on_rotangle,
                                        consts_h.axis_correction);
       
	// Copy constants to device
	consts_d.deep_copy(consts_h);

	// Set random access constants to consts_d
	consts.set(consts_d);

	// Initialize DockingParams
        DockingParams<DeviceType> docking_params(myligand_reference, mygrid, mypars);
	// Copy grid to device
        Kokkos::deep_copy(docking_params.fgrids_write, fgrids_h);

	// Input check
	 if (strcmp(mypars->ls_method, "sw") == 0) {
		printf("\nUsing Solis-Wets scheme.\n");
	} else {
                printf("\nOnly Solis-Wets scheme is available. Please set -lsmet sw \n\n"); return 1;
        }

	// Get profile for timing
        /*
 	profile.n_evals = mypars->num_of_energy_evals;
        profile.num_atoms = docking_params.num_of_atoms;
        profile.num_rotbonds = myligand_init->num_of_rotbonds;
        */
	//----------------------------- EXECUTION ------------------------------------//
        printf("\nExecution starts:\n\n");
        clock_t clock_start_docking = clock();

	// Autostop / Progress bar
	AutoStop autostop(mypars->pop_size, mypars->num_of_runs, mypars->stopstd, mypars->as_frequency);
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

	// Get the energy of the initial population (formerly kernel1)
	checkpoint("K_INIT");
	calc_init_pop(odd_generation, mypars, docking_params, consts);
	checkpoint(" ... Finished\n");

	// Reduction on the number of evaluations (formerly kernel2)
	checkpoint("K_EVAL");
	sum_evals(mypars, docking_params, evals_of_runs);
	Kokkos::fence();
	checkpoint(" ... Finished\n");

	Kokkos::deep_copy(evals_of_runs_h, evals_of_runs);

	int generation_cnt = 0; // Counter of while loop
	int curr_progress_cnt = 0;
	double progress;
	unsigned long total_evals;
	while ((progress = check_progress(evals_of_runs_h.data(), generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
	{
		if (mypars->autostop) {
			if (generation_cnt % mypars->as_frequency == 0) {
				Kokkos::deep_copy(energies_h,odd_generation.energies);
				if (autostop.check_if_satisfactory(generation_cnt, energies_h.data(), total_evals))
					break; // Exit loop
			}
		} else {
			//update progress bar (bar length is 50)
			int new_progress_cnt = (int) (progress/2.0+0.5);
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

		// Get the next generation via the genetic algorithm (formerly kernel4)
		checkpoint("K_GA_GENERATION");
		if (generation_cnt % 2 == 0) { // Since we need 2 generations at any time, just alternate btw 2 mem allocations
			gen_alg_eval_new(odd_generation, even_generation, mypars, docking_params, genetic_params, consts);
		} else {
			gen_alg_eval_new(even_generation, odd_generation, mypars, docking_params, genetic_params, consts);
		}
		checkpoint(" ... Finished\n");

		// Refine conformations to minimize energies
		if (docking_params.lsearch_rate != 0.0f) {
			if (strcmp(mypars->ls_method, "sw") == 0) {
				// Solis-Wets algorithm
				checkpoint("SOLIS_WETS");
                                if (generation_cnt % 2 == 0){
                                        solis_wets(even_generation, mypars, docking_params, consts);
                                } else {
                                        solis_wets(odd_generation, mypars, docking_params, consts);
                                }
                                checkpoint(" ... Finished\n");
			} else {
				// sd, and fire are NOT SUPPORTED in the Kokkos version (yet)
			}
		}

		// Reduction on the number of evaluations (formerly kernel2)
		checkpoint("K_EVAL");
		sum_evals(mypars, docking_params, evals_of_runs);
		Kokkos::fence();
		checkpoint(" ... Finished\n");

		// Copy evals back to CPU
		Kokkos::deep_copy(evals_of_runs_h, evals_of_runs);

		generation_cnt++;
	}

	clock_t clock_stop_docking = clock();
	if (!mypars->autostop)
	{
		//update progress bar (bar length is 50)mem_num_of_rotatingatoms_per_rotbond_const
		while (curr_progress_cnt < 50) {
			curr_progress_cnt++;
			printf("*");
			fflush(stdout);
		}
	}

	//----------------------------- PROCESSING ------------------------------------//

	// Pull results back to CPU
	if (generation_cnt % 2 == 0) {
		Kokkos::deep_copy(populations_h,odd_generation.conformations);
		Kokkos::deep_copy(energies_h,odd_generation.energies);
	}
	else {
		Kokkos::deep_copy(populations_h,even_generation.conformations);
		Kokkos::deep_copy(energies_h,even_generation.energies);
	}

	// Profiler
	//profile.nev_at_stop = total_evals/mypars->num_of_runs;
	//profile.autostopped = autostop.did_stop();

	// Final autostop statistics output
	if (mypars->autostop) autostop.output_final_stddev(generation_cnt, energies_h.data(), total_evals);

	printf("\n\n");
	// Arrange results and make res files

        for (unsigned long run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
	{
		arrange_result(populations_h.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, energies_h.data()+run_cnt*mypars->pop_size, mypars->pop_size);
		make_resfiles(populations_h.data()+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, 
			      energies_h.data()+run_cnt*mypars->pop_size, 
			      &myligand_reference,
			      myligand_init,
			      myxrayligand, 
			      mypars, 
			      evals_of_runs_h[run_cnt], 
			      generation_cnt, 
			      mygrid, 
			      fgrids_h.data(), 
			      cpu_ref_ori_angles.data()+3*run_cnt, 
			      argc, 
			      argv, 
                              /*1*/0,
			      run_cnt, 
			      &(cpu_result_ligands [run_cnt]));
	}

	// Clustering analysis, generate .dlg output
	clock_t clock_stop_program_before_clustering = clock();
	clusanal_gendlg(cpu_result_ligands.data(), mypars->num_of_runs, myligand_init, mypars,
					 mygrid, argc, argv, (ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs),
					 generation_cnt,(total_evals/mypars->num_of_runs));
	clock_stop_docking = clock();
/*
#if dddefined (DOCK_DEBUG)
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
*/
	return 0;
}

double check_progress(int* evals_of_runs, int generation_cnt, int max_num_of_evals, int max_num_of_gens, int num_of_runs, unsigned long &total_evals)
//The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
//parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
//the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
//generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
//than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	//Stops if the sum of evals of every run reached the sum of the total number of evals

	double evals_progress;
	double gens_progress;

	//calculating progress according to number of runs
	total_evals = 0;
	for (int i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	//calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}
