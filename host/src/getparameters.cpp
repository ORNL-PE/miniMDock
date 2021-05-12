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


#include <cstdint>
#include <fstream>
#include <algorithm> 
#include <cctype>
#include <locale>

#include "getparameters.h"
/*
// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}
*/

int get_filenames_and_ADcoeffs(const int* argc,
			       char** argv,
			       Dockpars* mypars)
			       
//The function fills the file name and coeffs fields of mypars parameter
//according to the proper command line arguments.
{
	int i;
	//int ffile_given;
	int  lfile_given;
	long tempint;

	//AutoDock 4 free energy coefficients
	const double coeff_elec_scale_factor = 332.06363;

	//this model assumes the BOUND conformation is the SAME as the UNBOUND, default in AD4.2
	const AD4_free_energy_coeffs coeffs_bound = {0.1662,
						     0.1209,
						     coeff_elec_scale_factor*0.1406,
						     0.1322,
						     0.2983};

	//this model assumes the unbound conformation is EXTENDED, default if AD4.0
	const AD4_free_energy_coeffs coeffs_extended = {0.1560,
						        0.0974,
							coeff_elec_scale_factor*0.1465,
							0.1159,
							0.2744};

	//this model assumes the unbound conformation is COMPACT
	const AD4_free_energy_coeffs coeffs_compact = {0.1641,
						       0.0531,
						       coeff_elec_scale_factor*0.1272,
						       0.0603,
						       0.2272};

	mypars->coeffs = coeffs_bound;	//default coeffs
	mypars->unbound_model = 0;

	//ffile_given = 1;
	lfile_given = 0;

	strcpy(mypars->fldfile, "./input/7cpa/7cpa_protein.maps.fld");
	//strcpy(mypars->fldfile, "./input/3er5/protein.maps.fld");
	//strcpy(mypars->fldfile, "./input/nsc1620/protein.maps.fld");

	for (i=1; i<(*argc)-1; i++)
	{
		//Argument: grid parameter file name.
/*		if (strcmp("-ffile", argv[i]) == 0)
		{
			ffile_given = 1;
			strcpy(mypars->fldfile, argv[i+1]);
		}
*/
		//Argument: ligand pdbqt file name
		if (strcmp("-lfile", argv[i]) == 0)
		{
			lfile_given = 1;
			strcpy(mypars->ligandfile, argv[i+1]);
		}

		//Argument: unbound model to be used.
		//0 means the bound, 1 means the extended, 2 means the compact ...
		//model's free energy coefficients will be used during docking.
		if (strcmp("-ubmod", argv[i]) == 0)
		{
			sscanf(argv[i+1], "%ld", &tempint);

			if (tempint == 0)
			{
				mypars->coeffs = coeffs_bound;
				mypars->unbound_model = 0;
			}
			else
				if (tempint == 1)
				{
					mypars->coeffs = coeffs_extended;
					mypars->unbound_model = 1;
				}
				else
				{
					mypars->coeffs = coeffs_compact;
					mypars->unbound_model = 2;
				}
		}
	}

/*	if (ffile_given == 0 )
	{
		printf("Error: grid fld file was not defined. Use -ffile argument!\n");
		return 1;
	}
*/
	if (lfile_given == 0 )
	{
		printf("Error: ligand pdbqt file was not defined. Use -lfile argument!\n");
		return 1;
	}

	return 0;
}

void get_commandpars(const int* argc,
		         char** argv,
		        double* spacing,
		      Dockpars* mypars)
//The function processes the command line arguments given with the argc and argv parameters,
//and fills the proper fields of mypars according to that. If a parameter was not defined
//in the command line, the default value will be assigned. The mypars' fields will contain
//the data in the same format as it is required for writing it to algorithm defined registers.
{
	int   i;
	long  tempint;
	//float tempfloat;
	int   arg_recognized;

	// ------------------------------------------
	//default values
	mypars->num_of_energy_evals	= 2500000;
	mypars->num_of_generations	= 27000;
	mypars->nev_provided		= false;
//	mypars->use_heuristics		= false;	// Flag if we want to use Diogo's heuristics
//	mypars->heuristics_max		= 50000000;	// Maximum number of evaluations under the heuristics (50M evaluates to 80% at 12.5M evals calculated by heuristics)
	mypars->abs_max_dmov		= 6.0/(*spacing); 	// +/-6A
	mypars->abs_max_dang		= 90; 		// +/- 90°
	mypars->mutation_rate		= 2; 		// 2%
	mypars->crossover_rate		= 80;		// 80%
	mypars->lsearch_rate		= 80;		// 80%

	strcpy(mypars->ls_method, "sw");		// "sw": Solis-Wets. 
							// The following possible methods are not considered here
							// "sd": Steepest-Descent
							// "fire": FIRE, https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf
							// "ad": ADADELTA, https://arxiv.org/abs/1212.5701
							// "adam": ADAM (currently only on Cuda)
	mypars->initial_sw_generations  = 0;
	mypars->smooth			= 0.5f;
	mypars->tournament_rate		= 60;		// 60%
	mypars->rho_lower_bound		= 0.01;		// 0.01
	mypars->base_dmov_mul_sqrt3	= 2.0/(*spacing)*sqrt(3.0);	// 2 A
	mypars->base_dang_mul_sqrt3	= 75.0*sqrt(3.0);		// 75°
	mypars->cons_limit		= 4;		// 4
	mypars->max_num_of_iters	= 300;
	mypars->pop_size		= 150;
	mypars->initpop_gen_or_loadfile	= false;
	mypars->gen_pdbs		= 0;

	mypars->autostop		= 1;
	mypars->as_frequency		= 5;
	mypars->stopstd			= 0.15;
	mypars->num_of_runs		= 1;
	mypars->reflig_en_required	= false;

	mypars->handle_symmetry		= true;
	mypars->gen_finalpop		= false;
	mypars->gen_best		= false;
	strcpy(mypars->resname, "docking");
	mypars->qasp			= 0.01097f;
	mypars->rmsd_tolerance 		= 2.0;			//2 Angstroem
	strcpy(mypars->xrayligandfile, mypars->ligandfile);	// By default xray-ligand file is the same as the randomized input ligand
	mypars->given_xrayligandfile	= false;		// That is, not given (explicitly by the user)
	mypars->num_of_docks = 1;                             // repeat the docking proess num_of_docks times
	// ------------------------------------------

	//overwriting values which were defined as a command line argument
	for (i=1; i<(*argc)-1; i+=2)
	{
		arg_recognized = 0;

		// ---------------------------------
		// MISSING: char fldfile [128]
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		//Argument: name of grid parameter file.
/*		if (strcmp("-ffile", argv [i]) == 0)
			arg_recognized = 1;
*/
		// ---------------------------------
		// MISSING: char ligandfile [128]
		// UPDATED in : get_filenames_and_ADcoeffs()
		// ---------------------------------
		//Argument: name of ligand pdbqt file
		if (strcmp("-lfile", argv [i]) == 0)
			arg_recognized = 1;

		
		//Argument: number of runs. Must be an integer between 1 and 1000.
		//Means the number of required runs
		if (strcmp("-nrun", argv [i]) == 0)
		{
			arg_recognized = 1;
			sscanf(argv [i+1], "%ld", &tempint);

			if ((tempint >= 1) && (tempint <= MAX_NUM_OF_RUNS))
				mypars->num_of_runs = (int) tempint;
			else
				printf("Warning: value of -nrun argument ignored. Value must be an integer between 1 and %d.\n", MAX_NUM_OF_RUNS);
		}
                 if (strcmp("-ndock", argv [i]) == 0){
                        arg_recognized = 1;
                        sscanf(argv [i+1], "%ld", &tempint);

                        if ((tempint >= 1) && (tempint <= MAX_NUM_OF_DOCKS))
                                mypars->num_of_docks = (int) tempint;
                        else
                                printf("Warning: value of -ndock argument ignored. Value must be an integer between 1 and %d.\n", MAX_NUM_OF_DOCKS);
                }

		if (arg_recognized != 1)
			printf("Warning: unknown argument '%s'.\n", argv [i]);
	}

	//validating some settings

	if (mypars->pop_size < mypars->gen_pdbs)
	{
		printf("Warning: value of -npdb argument igonred. Value mustn't be greater than the population size.\n");
		mypars->gen_pdbs = 1;
	}

}

void gen_initpop_and_reflig(Dockpars*       mypars,
			    float*          init_populations,
			    float*          ref_ori_angles,
			    Liganddata*     myligand,
			    const Gridinfo* mygrid)
//The function generates a random initial population
//(or alternatively, it reads from an external file according to mypars),
//and the angles of the reference orientation.
//The parameters mypars, myligand and mygrid describe the current docking.
//The pointers init_population and ref_ori_angles have to point to
//two allocated memory regions with proper size which the function will fill with random values.
//Each contiguous GENOTYPE_LENGTH_IN_GLOBMEM pieces of floats in init_population corresponds to a genotype,
//and each contiguous three pieces of floats in ref_ori_angles corresponds to
//the phi, theta and angle genes of the reference orientation.
//In addition, as part of reference orientation handling,
//the function moves myligand to origo and scales it according to grid spacing.
{
	int entity_id, gene_id;
	int gen_pop;
	//int  gen_seeds;
	FILE* fp;
	int i;
	//float init_orientation[MAX_NUM_OF_ROTBONDS+6];
	double movvec_to_origo[3];

	int pop_size = mypars->pop_size;

    float u1, u2, u3; // to generate random quaternion
    float qw, qx, qy, qz; // random quaternion
    float x, y, z, s; // convert quaternion to angles
    float phi, theta, rotangle;

	//initial population
	gen_pop = 0;

	//Reading initial population from file if only 1 run was requested
	if (mypars->initpop_gen_or_loadfile)
	{
		if (mypars->num_of_runs != 1)
		{
			printf("Warning: more than 1 run was requested. New populations will be generated \ninstead of being loaded from initpop.txt\n");
			gen_pop = 1;
		}
		else
		{
			fp = fopen("initpop.txt","rb"); // fp = fopen("initpop.txt","r");
			if (fp == NULL)
			{
				printf("Warning: can't find initpop.txt. A new population will be generated.\n");
				gen_pop = 1;
			}
			else
			{
				for (entity_id=0; entity_id<pop_size; entity_id++)
					for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
						fscanf(fp, "%f", &(init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]));

				//reading reference orienation angles from file
				fscanf(fp, "%f", &(mypars->ref_ori_angles[0]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[1]));
				fscanf(fp, "%f", &(mypars->ref_ori_angles[2]));

				fclose(fp);
			}
		}
	}
	else
		gen_pop = 1;

	// Local random numbers for thread safety/reproducibility
	LocalRNG r;

	//Generating initial population
	if (gen_pop == 1)
	{
		for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++)
		{
			for (gene_id=0; gene_id<3; gene_id++)
			{
				init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*(mygrid->size_xyz_angstr[gene_id]);
			}
			// generate random quaternion
			u1 = r.random_float();
			u2 = r.random_float();
			u3 = r.random_float();
			qw = sqrt(1.0 - u1) * sin(PI_TIMES_2 * u2);
			qx = sqrt(1.0 - u1) * cos(PI_TIMES_2 * u2);
			qy = sqrt(      u1) * sin(PI_TIMES_2 * u3);
			qz = sqrt(      u1) * cos(PI_TIMES_2 * u3);

			// convert to angle representation
			rotangle = 2.0 * acos(qw);
			s = sqrt(1.0 - (qw * qw));
			if (s < 0.001){ // rotangle too small
				x = qx;
				y = qy;
				z = qz;
			} else {
				x = qx / s;
				y = qy / s;
				z = qz / s;
			}

			theta = acos(z);
			phi = atan2(y, x);

			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = phi / DEG_TO_RAD;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = theta / DEG_TO_RAD;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = rotangle / DEG_TO_RAD;

			//printf("angles = %8.2f, %8.2f, %8.2f\n", phi / DEG_TO_RAD, theta / DEG_TO_RAD, rotangle/DEG_TO_RAD);

			/*
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) myrand() * 360.0;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+3] = (float) myrand() * 360;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+4] = (float) myrand() * 180;
			init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+5] = (float) myrand() * 360;
			*/

			for (gene_id=6; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++) {
				init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = r.random_float()*360;
			}
		}

		//Writing first initial population to initpop.txt
		fp = fopen("initpop.txt", "w");
		if (fp == NULL)
			printf("Warning: can't create initpop.txt.\n");
		else
		{
			for (entity_id=0; entity_id<pop_size; entity_id++)
				for (gene_id=0; gene_id<MAX_NUM_OF_ROTBONDS+6; gene_id++)
					fprintf(fp, "%f ", init_populations[entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]);

			//writing reference orientation angles to initpop.txt
			fprintf(fp, "%f ", mypars->ref_ori_angles[0]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[1]);
			fprintf(fp, "%f ", mypars->ref_ori_angles[2]);

			fclose(fp);
		}
	}

	//genotypes should contain x, y and z genes in grid spacing instead of Angstroms
	//(but was previously generated in Angstroms since fdock does the same)

	for (entity_id=0; entity_id<pop_size*mypars->num_of_runs; entity_id++)
		for (gene_id=0; gene_id<3; gene_id++)
			init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id] = init_populations [entity_id*GENOTYPE_LENGTH_IN_GLOBMEM+gene_id]/mygrid->spacing;

	//changing initial orientation of reference ligand
	/*for (i=0; i<38; i++)
		switch (i)
		{
		case 3: init_orientation [i] = mypars->ref_ori_angles [0];
				break;
		case 4: init_orientation [i] = mypars->ref_ori_angles [1];
				break;
		case 5: init_orientation [i] = mypars->ref_ori_angles [2];
				break;
		default: init_orientation [i] = 0;
		}

	change_conform_f(myligand, init_orientation, 0);*/

	//initial orientation will be calculated during docking,
	//only the required angles are generated here,
	//but the angles possibly read from file are ignored

	for (i=0; i<mypars->num_of_runs; i++)
	{
		// uniform distr.
		// generate random quaternion
		u1 = r.random_float();
		u2 = r.random_float();
		u3 = r.random_float();
		qw = sqrt(1.0 - u1) * sin(PI_TIMES_2 * u2);
		qx = sqrt(1.0 - u1) * cos(PI_TIMES_2 * u2);
		qy = sqrt(      u1) * sin(PI_TIMES_2 * u3);
		qz = sqrt(      u1) * cos(PI_TIMES_2 * u3);

		// convert to angle representation
		rotangle = 2.0 * acos(qw);
		s = sqrt(1.0 - (qw * qw));
		if (s < 0.001){ // rotangle too small
			x = qx;
			y = qy;
			z = qz;
		} else {
			x = qx / s;
			y = qy / s;
			z = qz / s;
		}

		theta = acos(z);
		phi = atan2(y, x);

		ref_ori_angles[3*i]   = phi / DEG_TO_RAD;
		ref_ori_angles[3*i+1] = theta / DEG_TO_RAD;
		ref_ori_angles[3*i+2] = rotangle / DEG_TO_RAD;
	}

	get_movvec_to_origo(myligand, movvec_to_origo);
	move_ligand(myligand, movvec_to_origo);
	scale_ligand(myligand, 1.0/mygrid->spacing);
	get_moving_and_unit_vectors(myligand);

	/*
	printf("ligand: movvec_to_origo: %f %f %f\n", movvec_to_origo[0], movvec_to_origo[1], movvec_to_origo[2]);
	*/

}
