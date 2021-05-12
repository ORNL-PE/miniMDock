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
#ifndef KERNELCONSTS_HPP
#define KERNELCONSTS_HPP

//#include "defines.h"

// Constants used on the device for Kokkos implementation
// Writable and constant versions for each
template<class Device>
struct InterIntraW
{
        Kokkos::View<float[MAX_NUM_OF_ATOMS],Device> atom_charges_const;
        Kokkos::View<char[MAX_NUM_OF_ATOMS],Device>  atom_types_const;

        InterIntraW() : atom_charges_const("atom_charges_const"),
		       atom_types_const("atom_types_const") {};

        // Copy from a host version
        void deep_copy(InterIntraW<HostType> interintra_h)
	{
		Kokkos::deep_copy(atom_charges_const,interintra_h.atom_charges_const);
		Kokkos::deep_copy(atom_types_const,interintra_h.atom_types_const);
	};
};

template<class Device>
struct InterIntra
{
        Kokkos::View<const float[MAX_NUM_OF_ATOMS],Device,RandomAccess> atom_charges_const;
        Kokkos::View<const char[MAX_NUM_OF_ATOMS],Device,RandomAccess>  atom_types_const;

        // Set const random access view 
        void set(InterIntraW<Device> interintra_write)
        {
                atom_charges_const = interintra_write.atom_charges_const;
                atom_types_const = interintra_write.atom_types_const;
        };
};

template<class Device>
struct IntraContribW
{
        Kokkos::View<char[3*MAX_INTRAE_CONTRIBUTORS],Device>  intraE_contributors_const;

	IntraContribW() : intraE_contributors_const("intraE_contributors_const") {};

	// Copy from a host version
        void deep_copy(IntraContribW<HostType> intracontrib_h)
        {
                Kokkos::deep_copy(intraE_contributors_const,intracontrib_h.intraE_contributors_const);
        };
};

template<class Device>
struct IntraContrib
{
        Kokkos::View<const char[3*MAX_INTRAE_CONTRIBUTORS],Device,RandomAccess>  intraE_contributors_const;

        // Set const random access view
        void set(IntraContribW<Device> intracontrib_write)
        {
                intraE_contributors_const = intracontrib_write.intraE_contributors_const;
        };
};

template<class Device>
struct IntraW
{
       Kokkos::View<float[2*ATYPE_NUM],Device> reqm_const; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       Kokkos::View<unsigned int[ATYPE_NUM],Device> atom1_types_reqm_const;
       Kokkos::View<unsigned int[ATYPE_NUM],Device> atom2_types_reqm_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device> VWpars_AC_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device> VWpars_BD_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES],Device> dspars_S_const;
       Kokkos::View<float[MAX_NUM_OF_ATYPES],Device> dspars_V_const;

       IntraW() : reqm_const("reqm_const"),
		 atom1_types_reqm_const("atom1_types_reqm_const"),
		 atom2_types_reqm_const("atom2_types_reqm_const"),
		 VWpars_AC_const("VWpars_AC_const"),
		 VWpars_BD_const("VWpars_BD_const"),
		 dspars_S_const("dspars_S_const"),
		 dspars_V_const("dspars_V_const") {};

	// Copy from a host version
        void deep_copy(IntraW<HostType> intra_h)
        {       
                Kokkos::deep_copy(reqm_const, intra_h.reqm_const);
		Kokkos::deep_copy(atom1_types_reqm_const, intra_h.atom1_types_reqm_const);
                Kokkos::deep_copy(atom2_types_reqm_const, intra_h.atom2_types_reqm_const);
                Kokkos::deep_copy(VWpars_AC_const, intra_h.VWpars_AC_const);
                Kokkos::deep_copy(VWpars_BD_const, intra_h.VWpars_BD_const);
                Kokkos::deep_copy(dspars_S_const, intra_h.dspars_S_const);
                Kokkos::deep_copy(dspars_V_const, intra_h.dspars_V_const);
        };
};

template<class Device>
struct Intra
{
       Kokkos::View<const float[2*ATYPE_NUM],Device,RandomAccess> reqm_const; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       Kokkos::View<const unsigned int[ATYPE_NUM],Device,RandomAccess> atom1_types_reqm_const;
       Kokkos::View<const unsigned int[ATYPE_NUM],Device,RandomAccess> atom2_types_reqm_const;
       Kokkos::View<const float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device,RandomAccess> VWpars_AC_const;
       Kokkos::View<const float[MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES],Device,RandomAccess> VWpars_BD_const;
       Kokkos::View<const float[MAX_NUM_OF_ATYPES],Device,RandomAccess> dspars_S_const;
       Kokkos::View<const float[MAX_NUM_OF_ATYPES],Device,RandomAccess> dspars_V_const;

        // Set const random access view
        void set(IntraW<Device> intra_write)
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

template<class Device>
struct RotListW
{
       Kokkos::View<int[MAX_NUM_OF_ROTATIONS],Device> rotlist_const;

       RotListW() : rotlist_const("rotlist_const") {};

	// Copy from a host version
        void deep_copy(RotListW<HostType> rot_list_h)
        {       
                Kokkos::deep_copy(rotlist_const,rot_list_h.rotlist_const);
        };
};

template<class Device>
struct RotList
{
       Kokkos::View<const int[MAX_NUM_OF_ROTATIONS],Device,RandomAccess> rotlist_const;

        // Set const random access view
        void set(RotListW<Device> rot_list_write)
        {
                rotlist_const = rot_list_write.rotlist_const;
        };
};

template<class Device>
struct ConformW
{
       Kokkos::View<float[3*MAX_NUM_OF_ATOMS],Device> ref_coords_const;
       Kokkos::View<float[3*MAX_NUM_OF_ROTBONDS],Device> rotbonds_moving_vectors_const;
       Kokkos::View<float[3*MAX_NUM_OF_ROTBONDS],Device> rotbonds_unit_vectors_const;
       Kokkos::View<float[4*MAX_NUM_OF_RUNS],Device> ref_orientation_quats_const;

       ConformW() : ref_coords_const("ref_coords_const"),
		   rotbonds_moving_vectors_const("rotbonds_moving_vectors_const"),
		   rotbonds_unit_vectors_const("rotbonds_unit_vectors_const"),
		   ref_orientation_quats_const("ref_orientation_quats_const") {};

	// Copy from a host version
        void deep_copy(ConformW<HostType> conform_h)
        {
                Kokkos::deep_copy(ref_coords_const, conform_h.ref_coords_const);
		Kokkos::deep_copy(rotbonds_moving_vectors_const, conform_h.rotbonds_moving_vectors_const);
		Kokkos::deep_copy(rotbonds_unit_vectors_const, conform_h.rotbonds_unit_vectors_const);
		Kokkos::deep_copy(ref_orientation_quats_const, conform_h.ref_orientation_quats_const);
        };
};

template<class Device>
struct Conform
{
       Kokkos::View<const float[3*MAX_NUM_OF_ATOMS],Device,RandomAccess> ref_coords_const;
       Kokkos::View<const float[3*MAX_NUM_OF_ROTBONDS],Device,RandomAccess> rotbonds_moving_vectors_const;
       Kokkos::View<const float[3*MAX_NUM_OF_ROTBONDS],Device,RandomAccess> rotbonds_unit_vectors_const;
       Kokkos::View<const float[4*MAX_NUM_OF_RUNS],Device,RandomAccess> ref_orientation_quats_const;

        // Set const random access view
        void set(ConformW<Device> conform_write)
        {
                ref_coords_const = conform_write.ref_coords_const;
                rotbonds_moving_vectors_const = conform_write.rotbonds_moving_vectors_const;
                rotbonds_unit_vectors_const = conform_write.rotbonds_unit_vectors_const;
                ref_orientation_quats_const = conform_write.ref_orientation_quats_const;
        };
};

template<class Device>
struct GradsW
{
        // Added for calculating torsion-related gradients.
        // Passing list of rotbond-atoms ids to the GPU.
        // Contains the same information as processligand.h/Liganddata->rotbonds 
        Kokkos::View<int[2*MAX_NUM_OF_ROTBONDS],Device> rotbonds;

        // Contains the same information as processligand.h/Liganddata->atom_rotbonds
        // "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
        // If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
        // it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
        Kokkos::View<int[MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS],Device> rotbonds_atoms;
        Kokkos::View<int[MAX_NUM_OF_ROTBONDS],Device> num_rotating_atoms_per_rotbond;

        GradsW() : rotbonds("rotbonds"),
		  rotbonds_atoms("rotbonds_atoms"),
		  num_rotating_atoms_per_rotbond("num_rotating_atoms_per_rotbond") {};

        // Copy from a host version
        void deep_copy(GradsW<HostType> grads_h)
        {
                Kokkos::deep_copy(rotbonds,grads_h.rotbonds);
                Kokkos::deep_copy(rotbonds_atoms,grads_h.rotbonds_atoms);
                Kokkos::deep_copy(num_rotating_atoms_per_rotbond,grads_h.num_rotating_atoms_per_rotbond);
        };
};

template<class Device>
struct Grads
{
        // Added for calculating torsion-related gradients.
        // Passing list of rotbond-atoms ids to the GPU.
        // Contains the same information as processligand.h/Liganddata->rotbonds 
        Kokkos::View<const int[2*MAX_NUM_OF_ROTBONDS],Device,RandomAccess> rotbonds;

        // Contains the same information as processligand.h/Liganddata->atom_rotbonds
        // "atom_rotbonds": array that contains the rotatable bonds - atoms assignment.
        // If the element atom_rotbonds[atom index][rotatable bond index] is equal to 1,
        // it means,that the atom must be rotated if the bond rotates. A 0 means the opposite.
        Kokkos::View<const int[MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS],Device,RandomAccess> rotbonds_atoms;
        Kokkos::View<const int[MAX_NUM_OF_ROTBONDS],Device,RandomAccess> num_rotating_atoms_per_rotbond;

        // Set const random access view
        void set(GradsW<Device> grads_write)
        {
                rotbonds = grads_write.rotbonds;
                rotbonds_atoms = grads_write.rotbonds_atoms;
                num_rotating_atoms_per_rotbond = grads_write.num_rotating_atoms_per_rotbond;
        };
};

template<class Device>
struct AxisCorrectionW
{
        Kokkos::View<float[NUM_AXIS_CORRECTION],Device> angle;
        Kokkos::View<float[NUM_AXIS_CORRECTION],Device> dependence_on_theta;
        Kokkos::View<float[NUM_AXIS_CORRECTION],Device> dependence_on_rotangle;

        AxisCorrectionW() : angle("angle"),
                           dependence_on_theta("dependence_on_theta"),
                           dependence_on_rotangle("dependence_on_rotangle")     {};

        // Copy from a host version
        void deep_copy(AxisCorrectionW<HostType> axis_correction_h)
        {
                Kokkos::deep_copy(angle,axis_correction_h.angle);
                Kokkos::deep_copy(dependence_on_theta,axis_correction_h.dependence_on_theta);
                Kokkos::deep_copy(dependence_on_rotangle,axis_correction_h.dependence_on_rotangle);
        };
};

template<class Device>
struct AxisCorrection
{       
        Kokkos::View<const float[NUM_AXIS_CORRECTION],Device,RandomAccess> angle;
        Kokkos::View<const float[NUM_AXIS_CORRECTION],Device,RandomAccess> dependence_on_theta;
        Kokkos::View<const float[NUM_AXIS_CORRECTION],Device,RandomAccess> dependence_on_rotangle;
        
        // Set const random access view
        void set(AxisCorrectionW<Device> axis_correction_write)
        {       
                angle = axis_correction_write.angle;
                dependence_on_theta = axis_correction_write.dependence_on_theta;
                dependence_on_rotangle = axis_correction_write.dependence_on_rotangle;
        };
};

template<class Device>
struct ConstantsW
{
        InterIntraW<Device> interintra;
        IntraContribW<Device> intracontrib;
        IntraW<Device> intra;
        RotListW<Device> rotlist;
        ConformW<Device> conform;
        GradsW<Device> grads;
        AxisCorrectionW<Device> axis_correction;

	// Copy from a host version
        void deep_copy(ConstantsW<HostType> consts_h)
        {
                interintra.deep_copy(consts_h.interintra);
		intracontrib.deep_copy(consts_h.intracontrib);
		intra.deep_copy(consts_h.intra);
		rotlist.deep_copy(consts_h.rotlist);
		conform.deep_copy(consts_h.conform);
		grads.deep_copy(consts_h.grads);
		axis_correction.deep_copy(consts_h.axis_correction);
        };
};

template<class Device>
struct Constants
{
        InterIntra<Device> interintra;
        IntraContrib<Device> intracontrib;
        Intra<Device> intra;
        RotList<Device> rotlist;
        Conform<Device> conform;
        Grads<Device> grads;
        AxisCorrection<Device> axis_correction;

        // Set const random access view
        void set(ConstantsW<Device> consts_write)
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

#endif
