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
// TODO - templatize ExSpace - ALS
//template<typename Policy>
void sum_evals(Dockpars* mypars, DockingParams& docking_params, vector<int*> evals_of_runs)
{
        // Outer loop over mypars->num_of_runs
        int league_size = mypars->num_of_runs;
        std::for_each ( std::execution::par_unseq,
			0, league_size,
		        [](int& lidx){

                // Reduce new_entities
                        int sum_evals;
			int offset = lidx*docking_params.pop_size;
	                sum_evals = std::reduce( std::execution::par_unseq,  
		                                 docking_params.evals_of_new_entities.begin() +offset, docking_params.evals_of_new_entities.begin() +offset +docking_params.pop_size, 
					         0, std::plus<int>
                			       );

                        // Add to global view
                        evals_of_runs(lidx) += sum_evals;
                      });
}
