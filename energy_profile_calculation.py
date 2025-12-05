#!/usr/bin/env python3
"""
Proton Transfer Energy Profile Calculation
==========================================

This script calculates the potential energy surface for proton transfer
from a carboxylic acid group (COOH) to water by performing constrained 
geometry optimizations at fixed O-H distances.

Requirements:
- ASE (Atomic Simulation Environment)
- ORB force field (Orbital Materials)
- NumPy, Matplotlib

Author: [Your name]
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.constraints import Hookean
from ase.optimize import BFGS
from interface.orb import get_orb_calculator


def run_energy_scan(input_structure, oh_indices, distances, spring_constant=100.0,
                   fmax=0.1, max_steps=200, charge=-1, spin=1):
    """
    Perform constrained energy scan along O-H distance.
    
    Parameters
    ----------
    input_structure : str or ase.Atoms
        Initial structure (constrained COOH structure recommended)
    oh_indices : tuple
        (oxygen_index, hydrogen_index) for the O-H bond to scan
    distances : array-like
        Array of O-H distances to scan (in Angstroms)
    spring_constant : float
        Hookean constraint spring constant (eV/Å²)
    fmax : float
        Force convergence criterion (eV/Å)
    max_steps : int
        Maximum optimization steps per distance
    charge : int
        Total system charge
    spin : int
        Spin multiplicity
        
    Returns
    -------
    distances_valid : np.ndarray
        Successfully optimized distances
    energies : np.ndarray
        Total energies at each distance (eV)
    converged : list
        Convergence status for each point
    """
    
    # Load initial structure
    if isinstance(input_structure, str):
        atoms_initial = read(input_structure)
    else:
        atoms_initial = input_structure.copy()
    
    print(f"Initial structure: {len(atoms_initial)} atoms")
    print(f"Scanning {len(distances)} O-H distances from {distances[0]:.2f} to {distances[-1]:.2f} Å")
    print("="*70)
    
    # Setup calculator
    calc = get_orb_calculator("OMOL")
    
    energies = []
    converged_flags = []
    o_idx, h_idx = oh_indices
    
    for i, target_distance in enumerate(distances):
        print(f"[{i+1}/{len(distances)}] Target O-H distance: {target_distance:.3f} Å", end=" ... ")
        
        # Create fresh copy of structure
        atoms = atoms_initial.copy()
        atoms.set_pbc([False, False, False])
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin
        
        # Apply Hookean constraint at target distance
        constraint = Hookean(a1=o_idx, a2=h_idx, rt=target_distance, k=spring_constant)
        atoms.set_constraint(constraint)
        atoms.calc = calc
        
        # Optimize with constraint
        opt = BFGS(atoms, logfile=None)
        
        try:
            opt.run(fmax=fmax, steps=max_steps)
            converged = opt.converged()
            
            # Remove constraint and calculate energy
            atoms_no_constraint = atoms.copy()
            atoms_no_constraint.set_constraint(None)
            atoms_no_constraint.calc = calc
            energy = atoms_no_constraint.get_potential_energy()
            
            energies.append(energy)
            converged_flags.append(converged)
            
            # Check actual distance
            final_pos = atoms.get_positions()
            actual_dist = np.linalg.norm(final_pos[h_idx] - final_pos[o_idx])
            
            status = "✓" if converged else "⚠"
            print(f"{status} E={energy:.4f} eV, actual d={actual_dist:.3f} Å")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            energies.append(np.nan)
            converged_flags.append(False)
    
    print("="*70)
    
    # Filter valid results
    energies = np.array(energies)
    valid_mask = ~np.isnan(energies)
    distances_valid = distances[valid_mask]
    energies_valid = energies[valid_mask]
    
    print(f"Valid points: {len(distances_valid)}/{len(distances)}")
    
    return distances_valid, energies_valid, converged_flags


def plot_energy_profile(distances, energies, output_file='energy_profile.png',
                        reference_distances=None):
    """
    Plot energy profile with optional reference markers.
    
    Parameters
    ----------
    distances : np.ndarray
        O-H distances (Å)
    energies : np.ndarray
        Absolute energies (eV)
    output_file : str
        Output filename for plot
    reference_distances : dict, optional
        Dictionary with 'bonded' and 'transferred' reference distances
    """
    
    # Convert to relative energies
    min_energy = np.min(energies)
    energies_relative = energies - min_energy
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot energy curve
    ax1.plot(distances, energies_relative, 'o-', linewidth=2, markersize=8,
             color='#2E86AB', label='Constrained optimization')
    
    # Add reference markers if provided
    if reference_distances:
        if 'bonded' in reference_distances:
            ax1.axvline(reference_distances['bonded'], color='green', 
                       linestyle='--', alpha=0.7, label='Bonded state (COOH)')
        if 'transferred' in reference_distances:
            ax1.axvline(reference_distances['transferred'], color='red',
                       linestyle='--', alpha=0.7, label='Transferred state (COO⁻)')
    
    ax1.set_xlabel('O-H Distance (Å)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Energy (eV)', fontsize=12, fontweight='bold')
    ax1.set_title('Proton Transfer Energy Profile', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Add secondary y-axis for kJ/mol
    ax2 = ax1.twinx()
    ax2.set_ylabel('Relative Energy (kJ/mol)', fontsize=12, fontweight='bold')
    ax2.set_ylim(ax1.get_ylim()[0]*96.485, ax1.get_ylim()[1]*96.485)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_file}")
    
    return fig


def save_data(distances, energies, output_file='energy_data.csv'):
    """Save energy profile data to CSV file."""
    energies_relative = energies - np.min(energies)
    data = np.column_stack([distances, energies, energies_relative])
    np.savetxt(output_file, data,
               header='O-H_distance_Angstrom,Energy_eV,Relative_Energy_eV',
               delimiter=',', comments='')
    print(f"✓ Data saved: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    INPUT_STRUCTURE = 'graphene_cooh_strong_hookean_optimized.pdb'
    HYDROXYL_O_IDX = 33  # Index of hydroxyl oxygen in COOH group
    H_IDX = 34           # Index of hydrogen to transfer
    
    # Scan parameters
    DISTANCES = np.linspace(0.95, 3.5, 20)  # Å
    SPRING_CONSTANT = 100.0  # eV/Å²
    FMAX = 0.1  # eV/Å
    MAX_STEPS = 200
    
    # System parameters
    CHARGE = -1  # Total system charge (accounting for OH⁻)
    SPIN = 1     # Spin multiplicity
    
    # Output files
    PLOT_FILE = 'proton_transfer_energy_profile.png'
    DATA_FILE = 'proton_transfer_data.csv'
    
    print("\n" + "="*70)
    print("PROTON TRANSFER ENERGY PROFILE CALCULATION")
    print("="*70 + "\n")
    
    # Run energy scan
    distances, energies, converged = run_energy_scan(
        input_structure=INPUT_STRUCTURE,
        oh_indices=(HYDROXYL_O_IDX, H_IDX),
        distances=DISTANCES,
        spring_constant=SPRING_CONSTANT,
        fmax=FMAX,
        max_steps=MAX_STEPS,
        charge=CHARGE,
        spin=SPIN
    )
    
    # Report statistics
    min_energy = np.min(energies)
    max_relative = np.max(energies - min_energy)
    print(f"\nEnergy range: {max_relative:.3f} eV ({max_relative*96.485:.1f} kJ/mol)")
    
    # Plot results
    reference_dists = {
        'bonded': 1.043,      # From constrained structure
        'transferred': 1.608  # From unconstrained structure
    }
    plot_energy_profile(distances, energies, PLOT_FILE, reference_dists)
    
    # Save data
    save_data(distances, energies, DATA_FILE)
    
    print("\n" + "="*70)
    print("CALCULATION COMPLETE")
    print("="*70)
