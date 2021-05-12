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
#ifndef GENETICPARAMS_HPP
#define GENETICPARAMS_HPP

// Parameters for genetic algorithm for Kokkos implementation
struct GeneticParams
{
        float           tournament_rate;
        float           crossover_rate;
        float           mutation_rate;
        float           abs_max_dmov;
        float           abs_max_dang;

	// Constructor
	GeneticParams(const Dockpars* mypars)
	{
		// Notice: tournament_rate, crossover_rate, mutation_rate
		// were scaled down to [0,1] in host to reduce number of operations in device
		tournament_rate = mypars->tournament_rate/100.0f;
		crossover_rate  = mypars->crossover_rate/100.0f;
		mutation_rate   = mypars->mutation_rate/100.f;
		abs_max_dang    = mypars->abs_max_dang;
		abs_max_dmov    = mypars->abs_max_dmov;
	}
};

#endif
