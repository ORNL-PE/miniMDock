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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "processgrid.h"
#include "processligand.h"
#include "getparameters.h"
#include "performdocking.h"
#include "setup.hpp"
#include "simulation_state.hpp"


#ifdef USE_KOKKOS
 #include <Kokkos_Core.hpp>
 #include "kokkos_settings.hpp"
#else
 #include "GpuData.h"
#endif


#ifndef _WIN32
// Time measurement
#include <sys/time.h>
#endif

template<typename T>
inline double seconds_since(T& time_start)
{
#ifndef _WIN32
	timeval time_end;
	gettimeofday(&time_end,NULL);
        double num_sec     = time_end.tv_sec  - time_start.tv_sec;
        double num_usec    = time_end.tv_usec - time_start.tv_usec;
        return (num_sec + (num_usec/1000000));
#else
	return 0.0;
#endif
}

template<typename T>
inline void start_timer(T& time_start)
{
#ifndef _WIN32
	gettimeofday(&time_start,NULL);
#endif
}

int main(int argc, char* argv[])
{

#ifdef USE_KOKKOS
	Kokkos::initialize();
 	{
#endif
	// Print version info
	printf("AutoDock-GPU version: %s\n", VERSION);
	// Error flag
	int err = 0;
	// Timer initializations
#ifndef _WIN32
	timeval time_start;
	start_timer(time_start);
#else
	// Dummy variables if timers off
	double time_start = 0;
#endif
	double setup_time=0;
	double total_exec_time=0;
	double resulting_time=0;
#ifndef _WIN32
                timeval setup_timer, exec_timer,  resulting_timer;
#else
                double setup_timer, exec_timer, resulting_timer;
#endif

	start_timer(setup_timer);
	// Setup master map set (one for now, nthreads-1 for general case)
	std::vector<Map> all_maps;

	// Objects that are arguments of docking_with_gpu
#ifdef USE_KOKKOS
	Kokkos::View<float*,HostType> floatgrids = Kokkos::View<float*,HostType>("floatgrids0", 0);
#else
	std::vector<float> floatgrids;
	GpuData cData;
	GpuTempData tData;

	cData.devnum=-1;
	// Get device number to run on
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		if (strcmp("-devnum", argv [i]) == 0)
		{
			int tempint;
			sscanf(argv [i+1], "%lu", &tempint);
			if ((tempint >= 1) && (tempint <= 65536))
				cData.devnum = (unsigned long) tempint-1;
			else
				printf("Warning: value of -devnum argument ignored. Value must be an integer between 1 and 65536.\n");
			break;
		}
	}

	setup_gpu_for_docking(cData,tData);
#endif

		//int t_id = 0;
		Dockpars   mypars;
		Liganddata myligand_init;
		Gridinfo   mygrid;
		Liganddata myxrayligand;
	        SimulationState sim_state;
		int i_job = 0;
			// Load files, read inputs, prepare arrays for docking stage
#ifdef USE_KOKKOS
			if (setup(mygrid, floatgrids, mypars, myligand_init, myxrayligand, i_job,  argc, argv) != 0)
#else
			if (setup(all_maps, mygrid, floatgrids, mypars, myligand_init, myxrayligand, i_job, argc, argv) != 0)
#endif
			{
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Keep in setup stage rather than moving to launch stage so a different job will be set up
				printf("\n\nError in setup of the Job \n");
				err = 1;
			} else { // Successful setup
				// Copy preloaded maps to GPU
				setup_time=seconds_since(setup_timer);
			}

			// Starting Docking
	        unsigned int repeats = mypars.num_of_docks;
		for (unsigned int i = 0; i < repeats; ++i){
			int error_in_docking;
			// Critical section to only let one thread access GPU at a time
			{
				printf("\nRunning the Job \n");
				//  start exec timer
	                        start_timer(exec_timer);
				// Dock
#ifdef USE_KOKKOS
				error_in_docking = docking_with_gpu(&(mygrid), floatgrids, &(mypars), &(myligand_init), &(myxrayligand), &argc, argv);
#else
				error_in_docking = docking_with_gpu(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), &argc, argv, sim_state, cData, tData);
#endif
				// End exec timer, start idling timer
				sim_state.exec_time = seconds_since(exec_timer);
			}

			if (error_in_docking!=0){
				// If error encountered: Set error flag to 1; Add to count of finished jobs
				// Set back to setup stage rather than moving to processing stage so a different job will be set up
				printf("\n\nError in docking_with_gpu, stopped the Job \n");
				err = 1;
			} else { // Successful run
#ifndef _WIN32
				total_exec_time+=sim_state.exec_time;
				printf("\nThe Job took %.3f sec for docking\n", sim_state.exec_time);
#endif
			}
		} // End all dockings
#ifndef USE_KOKKOS
			// Post-processing
	                start_timer(resulting_timer);
	                process_result(&(mygrid), floatgrids.data(), &(mypars), &(myligand_init), &(myxrayligand), &argc,argv, sim_state);
	                resulting_time=seconds_since(resulting_timer);

	finish_gpu_from_docking(cData,tData);
#endif

#ifndef _WIN32
	// Total time measurement
	printf("\nSetup time: %.3f sec", setup_time);
	printf("\nReulting time: %.3f sec", resulting_time);
	printf("\nTotal Docking time: %.3f sec", total_exec_time);
	printf("\nTotal Run time: %.3f sec", seconds_since(time_start));
	printf("\nMean Docking time: %.3f sec", total_exec_time/repeats);
#endif

	// Alert user to ligands that failed to complete
	if (err==1){
		printf("\nThe job was not successful.");
	}
#ifdef USE_KOKKOS
	}
	Kokkos::finalize();
#endif
	return 0;
}
