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

#ifndef CONSTDEFNS_HPP
#define CONSTDEFNS_HPP


// Constants used on the device for Kokkos implementation
// Writable and constant versions for each
struct InterIntraW
{
        vector<float[MAX_NUM_OF_ATOMS]> atom_charges_const;
        vector<char[MAX_NUM_OF_ATOMS]>  atom_types_const;

        InterIntraW() : atom_charges_const("atom_charges_const"),
		       atom_types_const("atom_types_const") {};
};

struct InterIntra
{
        vector<const float[MAX_NUM_OF_ATOMS]> atom_charges_const;
        vector<const char[MAX_NUM_OF_ATOMS]>  atom_types_const;

        // Set const random access view 
        void set(InterIntraW interintra_write)
        {
                atom_charges_const = interintra_write.atom_charges_const;
                atom_types_const = interintra_write.atom_types_const;
        };
};

struct IntraContribW
{
        vector<char[3*MAX_INTRAE_CONTRIBUTORS]>  intraE_contributors_const;

	IntraContribW() : intraE_contributors_const("intraE_contributors_const") {};
};

struct IntraContrib
{
        vector<const char[3*MAX_INTRAE_CONTRIBUTORS]>  intraE_contributors_const;

        // Set const random access view
        void set(IntraContribW<Device> intracontrib_write)
        {
                intraE_contributors_const = intracontrib_write.intraE_contributors_const;
        };
};

struct IntraW
{
       vector<float[2*ATYPE_NUM]> reqm_const; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       vector<unsigned int[ATYPE_NUM]> atom1_types_reqm_const;
       vector<unsigned int[ATYPE_NUM]> atom2_types_reqm_const;
       vector<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES]> VWpars_AC_const;
       vector<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES]> VWpars_BD_const;
       vector<float[MAX_NUM_OF_ATYPES]> dspars_S_const;
       vector<float[MAX_NUM_OF_ATYPES]> dspars_V_const;

       IntraW() : reqm_const("reqm_const"),
		 atom1_types_reqm_const("atom1_types_reqm_const"),
		 atom2_types_reqm_const("atom2_types_reqm_const"),
		 VWpars_AC_const("VWpars_AC_const"),
		 VWpars_BD_const("VWpars_BD_const"),
		 dspars_S_const("dspars_S_const"),
		 dspars_V_const("dspars_V_const") {};
};

struct Intra
{
       vector<const float[2*ATYPE_NUM]> reqm_const; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       vector<const unsigned int[ATYPE_NUM]> atom1_types_reqm_const;
       vector<const unsigned int[ATYPE_NUM]> atom2_types_reqm_const;
       vector<const float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES]> VWpars_AC_const;
       vector<const float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES]> VWpars_BD_const;
       vector<const float[MAX_NUM_OF_ATYPES]> dspars_S_const;
       vector<const float[MAX_NUM_OF_ATYPES]> dspars_V_const;

        // Set const random access view
        void set(IntraW intra_write)
        {      
                reqm_const = intra_write.reqm_const;
                atom1_types_reqm_const = intra_write.atom1_types_reqm_const;
                atom2_types_reqm_const = intra_write.atom2_types_reqm_const;
                VWpars_AC_const = intra_write.VWpars_AC_const;
                VWpars_BD_const = intra_write.VWpars_BD_const;
                dspars_S_const = intra_write.dspars_S_const;
                dspars_V_const = intra_write.dspars_V_const;
        };
};

struct RotListW
{
       vector<int[MAX_NUM_OF_ROTATIONS]> rotlist_const;

       RotListW() : rotlist_const("rotlist_const") {};
};

struct RotList
{
       vector<const int[MAX_NUM_OF_ROTATIONS]> rotlist_const;

        // Set const random access view
        void set(RotListW rot_list_write)
        {
                rotlist_const = rot_list_write.rotlist_const;
        };
};

struct ConformW
{
       vector<float[3*MAX_NUM_OF_ATOMS]> ref_coords_const;
       vector<float[3*MAX_NUM_OF_ROTBONDS]> rotbonds_moving_vectors_const;
       vector<float[3*MAX_NUM_OF_ROTBONDS]> rotbonds_unit_vectors_const;
       vector<float[4*MAX_NUM_OF_RUNS]> ref_orientation_quats_const;

       ConformW() : ref_coords_const("ref_coords_const"),
		   rotbonds_moving_vectors_const("rotbonds_moving_vectors_const"),
		   rotbonds_unit_vectors_const("rotbonds_unit_vectors_const"),
		   ref_orientation_quats_const("ref_orientation_quats_const") {};
};

struct Conform
{
       vector<const float[3*MAX_NUM_OF_ATOMS]> ref_coords_const;
       vector<const float[3*MAX_NUM_OF_ROTBONDS]> rotbonds_moving_vectors_const;
       vector<const float[3*MAX_NUM_OF_ROTBONDS]> rotbonds_unit_vectors_const;
       vector<const float[4*MAX_NUM_OF_RUNS]> ref_orientation_quats_const;

        // Set const random access view
        void set(ConformW conform_write)
        {
                ref_coords_const = conform_write.ref_coords_const;
                rotbonds_moving_vectors_const = conform_write.rotbonds_moving_vectors_const;
                rotbonds_unit_vectors_const = conform_write.rotbonds_unit_vectors_const;
                ref_orientation_quats_const = conform_write.ref_orientation_quats_const;
        };
};

struct GradsW
{
        // Added for calculating torsion-related gradients.
        // Passing list of rotbond-atoms ids to the GPU.
        // Contains the same information as processligand.h/Liganddata->rotbonds 
        vector<int[2*MAX_NUM_OF_ROTBONDS]> rotbonds;

        // Contains the same information as processligand.h/Liganddata->atom_rotbonds
        // "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
        // If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
        // it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
        vector<int[MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS]> rotbonds_atoms;
        vector<int[MAX_NUM_OF_ROTBONDS]> num_rotating_atoms_per_rotbond;

        GradsW() : rotbonds("rotbonds"),
		  rotbonds_atoms("rotbonds_atoms"),
		  num_rotating_atoms_per_rotbond("num_rotating_atoms_per_rotbond") {};
};

struct Grads
{
        // Added for calculating torsion-related gradients.
        // Passing list of rotbond-atoms ids to the GPU.
        // Contains the same information as processligand.h/Liganddata->rotbonds 
        vector<const int[2*MAX_NUM_OF_ROTBONDS]> rotbonds;

        // Contains the same information as processligand.h/Liganddata->atom_rotbonds
        // "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
        // If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
        // it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
        vector<const int[MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS]> rotbonds_atoms;
        vector<const int[MAX_NUM_OF_ROTBONDS]> num_rotating_atoms_per_rotbond;

        // Set const random access view
        void set(GradsW grads_write)
        {
                rotbonds = grads_write.rotbonds;
                rotbonds_atoms = grads_write.rotbonds_atoms;
                num_rotating_atoms_per_rotbond = grads_write.num_rotating_atoms_per_rotbond;
        };
};

struct AxisCorrectionW
{
        vector<float[NUM_AXIS_CORRECTION]> angle;
        vector<float[NUM_AXIS_CORRECTION]> dependence_on_theta;
        vector<float[NUM_AXIS_CORRECTION]> dependence_on_rotangle;

        AxisCorrectionW() : angle("angle"),
                           dependence_on_theta("dependence_on_theta"),
                           dependence_on_rotangle("dependence_on_rotangle")     {};
};

struct AxisCorrection
{       
        vector<const float[NUM_AXIS_CORRECTION]> angle;
        vector<const float[NUM_AXIS_CORRECTION]> dependence_on_theta;
        vector<const float[NUM_AXIS_CORRECTION]> dependence_on_rotangle;
        
        // Set const random access view
        void set(AxisCorrectionW axis_correction_write)
        {       
                angle = axis_correction_write.angle;
                dependence_on_theta = axis_correction_write.dependence_on_theta;
                dependence_on_rotangle = axis_correction_write.dependence_on_rotangle;
        };
};

struct ConstantsW
{
        InterIntraW interintra;
        IntraContribW intracontrib;
        IntraW intra;
        RotListW rotlist;
        ConformW conform;
        GradsW grads;
        AxisCorrectionW axis_correction;
};

struct Constants
{
        InterIntra interintra;
        IntraContrib intracontrib;
        Intra intra;
        RotList rotlist;
        Conform conform;
        Grads grads;
        AxisCorrection axis_correction;

        // Set const random access view
        void set(ConstantsW consts_write)
        {
                interintra.set(consts_write.interintra);
                intracontrib.set(consts_write.intracontrib);
                intra.set(consts_write.intra);
                rotlist.set(consts_write.rotlist);
                conform.set(consts_write.conform);
                grads.set(consts_write.grads);
                axis_correction.set(consts_write.axis_correction);
        };
};

// Coordinates of all atoms
typedef vector<float4*> Coordinates;

// Gradient (inter/intra, xyz, num atoms)
typedef vector<float**> AtomGradients;

// Genotype
typedef vector<float*> Genotype;

// Identical to Genotype, but for auxiliary arrays (e.g. gradient) that arent technically genotypes themselves. To avoid confusion, shouldnt be labeled as a genotype
typedef vector<float*> GenotypeAux;

// Array of length team_size for use in perform_elitist_selection
typedef vector<float[NUM_OF_THREADS_PER_BLOCK]> TeamFloat;
typedef vector<int[NUM_OF_THREADS_PER_BLOCK]> TeamInt;

// Arrays of different fixed sizes (maybe unnecessary but fixed probably performs better so use it if length is known at compile time)
typedef vector<bool[1]> OneBool;
typedef vector<int[1]> OneInt;
typedef vector<int[2]> TwoInt;
typedef vector<int[4]> FourInt;
typedef vector<float[1]> OneFloat;
typedef vector<float[4]> FourFloat;
typedef vector<float[10]> TenFloat;

#endif
