"""
Modified Poisson-Boltzmann solver with Bikerman steric correction.

Generates all plots for the CO2 uptake / electric field paper:
1. E-field vs concentration (at -200 mV)
2. E-field vs applied potential (at 3 M)
3. Electrostatic potential vs distance
4. Ion concentrations vs distance

The Bikerman steric correction prevents unphysical ion crowding at high
potentials by capping ion density at the close-packing limit.

Usage:
    python make_plots.py
"""

import warnings
import numpy as np
from math import pi
from scipy.integrate import quad, solve_bvp
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---- constants ----
kB   = 1.3806503e-23
elc  = 1.6021765e-19
Avog = 6.0221415e23
eps0 = 8.85418782e-12
T    = 297.15
V_thermal = kB * T / elc  # ~25.6 mV

# Surface potential in dimensionless units (kT/e)
# -200 mV is where OH production kicks in experimentally
SURFACE_POTENTIAL_MV = -200  # mV
SURFACE_POTENTIAL = SURFACE_POTENTIAL_MV * 1e-3 / V_thermal

# Ion size for steric correction (hydrated ion diameter)
ION_SIZE = 3.5e-10  # 3.5 Å


def heaviside_theta(x):
    return np.heaviside(np.asarray(x), 0.0)


def solve_PB_steric(c, surface_potential=SURFACE_POTENTIAL):
    """
    Solve modified PB with Bikerman steric correction.
    
    Parameters
    ----------
    c : float
        Bulk NaCl concentration (M)
    surface_potential : float
        Dimensionless surface potential (kT/e units)
    
    Returns
    -------
    dict with E-field, solution, and diagnostic data
    """
    # bulk densities (1/m^3)
    rhoIonNa = c * Avog * 1000.0
    rhoIonCl = rhoIonNa

    # Maximum packing density
    rho_max = 1.0 / ION_SIZE**3
    nu = rhoIonNa / rho_max  # bulk packing fraction

    # Debye kappa
    kappa = 1.0 / (0.304 / np.sqrt(c) * 1e-9)

    # domain size in Å (10 Debye lengths, but at least 25 Å for plotting)
    Boundary = max((1.0 / kappa) * 1e10 * 10.0, 25.0)

    # --- Cl- image term ---
    ahCl = 2.0e-10  # m

    def W_Cl():
        def integrand(k):
            root = np.sqrt(kappa**2 + k**2)
            num = root * np.cosh(k * ahCl) - k * np.sinh(k * ahCl)
            den = root * (root * np.cosh(k * ahCl) + k * np.sinh(k * ahCl))
            return k * num / den
        res, _ = quad(integrand, 0.0, 10.0 * 1e10, limit=200)
        return elc**2 / (2.0 * 78.3 * 4.0 * pi * eps0) * res / (kB * T)

    WCl = W_Cl()

    def UCl(z):
        z = np.asarray(z, dtype=float)
        out = np.full_like(z, 1000.0)
        pos = z > 0.0
        if np.any(pos):
            zpos = z[pos]
            out[pos] = (30.0 * heaviside_theta(ahCl * 1e10 - zpos) +
                        WCl * (ahCl * 1e10) / zpos * 
                        np.exp(-2.0 * kappa * (zpos * 1e-10 - ahCl)))
        return out

    # --- Na+ image term ---
    ahNa = 2.5e-10  # m

    def W_Na():
        def integrand(k):
            root = np.sqrt(kappa**2 + k**2)
            num = root * np.cosh(k * ahNa) - k * np.sinh(k * ahNa)
            den = root * (root * np.cosh(k * ahNa) + k * np.sinh(k * ahNa))
            return k * num / den
        res, _ = quad(integrand, 0.0, 10.0 * 1e10, limit=200)
        return elc**2 / (2.0 * 78.3 * 4.0 * pi * eps0) * res / (kB * T)

    WNa = W_Na()

    def UNa(z):
        z = np.asarray(z, dtype=float)
        out = np.full_like(z, 1000.0)
        pos = z > 0.0
        if np.any(pos):
            zpos = z[pos]
            out[pos] = (30.0 * heaviside_theta(ahNa * 1e10 - zpos) +
                        WNa * (ahNa * 1e10) / zpos * 
                        np.exp(-2.0 * kappa * (zpos * 1e-10 - ahNa)))
        return out

    # ---- Solve PB equation with steric correction ----
    z0 = 0.1  # Å
    kappa_A = kappa * 1e-10  # kappa in per-Angstrom

    def ode(z, y):
        phi, dphi = y[0], y[1]
        factor = -1.0e-20 * elc**2 / (eps0 * 78.3) / (kB * T)
        
        # Clamp exponents to avoid overflow
        exp_na = np.clip(-UNa(z) - phi, -700, 700)
        exp_cl = np.clip(-UCl(z) + phi, -700, 700)
        
        # Bikerman steric correction
        boltz_na = np.exp(exp_na)
        boltz_cl = np.exp(exp_cl)
        
        # Steric denominator
        denom = 1.0 + nu * (boltz_na + boltz_cl - 2.0)
        denom = np.maximum(denom, 1e-10)
        
        rhs = factor * (rhoIonNa * boltz_na / denom -
                       rhoIonCl * boltz_cl / denom)
        
        return np.vstack((dphi, rhs))

    def bc(ya, yb):
        return np.array([
            ya[0] - surface_potential,
            yb[0] - (-1.0e-4)
        ])

    # mesh with higher density near the surface
    z_mesh = np.unique(np.concatenate([
        np.linspace(z0, 10, 200),
        np.linspace(10.01, Boundary, 400)
    ]))
    
    # Better initial guess: exponential decay
    phi_guess = surface_potential * np.exp(-kappa_A * (z_mesh - z0))
    dphi_guess = -kappa_A * phi_guess
    y_init = np.vstack((phi_guess, dphi_guess))

    sol = solve_bvp(ode, bc, z_mesh, y_init, max_nodes=30000, tol=1e-5, verbose=0)

    if sol.status != 0:
        print(f"Warning (c={c:.3g} M): BVP solver status {sol.status}")

    # E-field at z* = 3 Å
    zSurf = 2.6  # Distance from COO⁻ to nearest water H (from DFT geometry)
    dphi_dz_A = sol.sol(zSurf)[1]
    dphi_dz_m = dphi_dz_A * 1.0e10
    E = -(kB * T / elc) * dphi_dz_m
    
    # Helper functions for ion concentrations (with steric correction)
    def get_ion_conc(z_arr):
        """Return Na+ and Cl- concentrations at given z positions."""
        phi_arr = sol.sol(z_arr)[0]
        exp_na = np.clip(-UNa(z_arr) - phi_arr, -700, 700)
        exp_cl = np.clip(-UCl(z_arr) + phi_arr, -700, 700)
        boltz_na = np.exp(exp_na)
        boltz_cl = np.exp(exp_cl)
        denom = 1.0 + nu * (boltz_na + boltz_cl - 2.0)
        denom = np.maximum(denom, 1e-10)
        
        conc_Na = (rhoIonNa / Avog / 1000.0) * boltz_na / denom
        conc_Cl = (rhoIonCl / Avog / 1000.0) * boltz_cl / denom
        return conc_Na, conc_Cl
    
    return {
        'E': E,
        'sol': sol,
        'z0': z0,
        'Boundary': Boundary,
        'UCl': UCl,
        'UNa': UNa,
        'rhoIonCl': rhoIonCl,
        'rhoIonNa': rhoIonNa,
        'c': c,
        'nu': nu,
        'get_ion_conc': get_ion_conc,
        'surface_potential': surface_potential
    }


if __name__ == "__main__":
    # Concentration range: 0.01 M to 5 M
    exponents = np.arange(-2.0, np.log10(5.0) + 1e-9, 0.15)
    conc_list = 10.0**exponents

    print("=" * 70)
    print(f"Modified PB with Bikerman Steric Correction")
    print(f"Surface potential: {SURFACE_POTENTIAL_MV} mV ({SURFACE_POTENTIAL:.1f} kT/e)")
    print(f"Ion size: {ION_SIZE*1e10:.1f} Å")
    print(f"Concentration range: {conc_list[0]:.3g} to {conc_list[-1]:.3g} M")
    print("=" * 70)

    results = []
    E_list = []
    for c in conc_list:
        print(f"c = {c:.4g} M", end=" ... ")
        res = solve_PB_steric(c)
        results.append(res)
        E_list.append(res['E'])
        print(f"E = {res['E']:.3e} V/m ({np.abs(res['E'])/1e9:.2f} GV/m)")
    E_list = np.array(E_list)

    # Consistent color scheme for all plots
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    z_max = 20.0  # Å - fixed plotting range for all concentrations
    
    # ================================================================
    # Plot 1: E-field vs concentration (matching style with Plot 3)
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.semilogx(conc_list, np.abs(E_list) / 1e9, 'o-', linewidth=2, 
                 markersize=6, color='#2E86AB')
    ax1.set_xlabel("Concentration c (M)", fontsize=12)
    ax1.set_ylabel("|E| at z = 2.6 Å (GV/m)", fontsize=12)
    ax1.set_title(f"Electric field magnitude vs concentration\n"
                  f"Surface potential: ψ₀ = {SURFACE_POTENTIAL_MV/1000:.1f} V", fontsize=12)
    ax1.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    fig1.savefig("Efield_vs_concentration.png", dpi=150)
    print("\nSaved: Efield_vs_concentration.png")
    
    # ================================================================
    # Plot 1b: E-field vs POTENTIAL at 3 M
    # ================================================================
    print("\nComputing E-field vs potential at 3 M...")
    potential_sweep_mV = np.linspace(-50, -1400, 30)
    c_fixed = 3.0  # M
    
    fig1b, ax1b = plt.subplots(figsize=(8, 6))
    
    E_sweep = []
    for psi_mV in potential_sweep_mV:
        phi0 = psi_mV * 1e-3 / V_thermal
        res = solve_PB_steric(c_fixed, phi0)
        E_sweep.append(res['E'])
    E_sweep = np.array(E_sweep)
    
    ax1b.plot(-potential_sweep_mV/1000, np.abs(E_sweep) / 1e9, 'o-', linewidth=2, 
              markersize=6, color='#2E86AB')
    
    ax1b.set_xlabel("Applied Potential |ψ₀| (V)", fontsize=12)
    ax1b.set_ylabel("|E| at z = 2.6 Å (GV/m)", fontsize=12)
    ax1b.set_title(f"Electric field magnitude vs applied potential (c = {c_fixed} M)", fontsize=12)
    ax1b.grid(True, linestyle=":")
    plt.tight_layout()
    fig1b.savefig("Efield_vs_potential.png", dpi=150)
    print("Saved: Efield_vs_potential.png")

    # ================================================================
    # Plot 2: Potential vs distance
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for res, color in zip(results, colors):
        # Always plot to z_max, but only within valid solution range
        z_end = min(res['Boundary'], z_max)
        z_plot = np.linspace(res['z0'], z_end, 500)
        phi_plot = res['sol'].sol(z_plot)[0]
        ax2.plot(z_plot, phi_plot, color=color, linewidth=1.5, 
                 label=f"c = {res['c']:.2g} M")
    
    ax2.set_xlim(0, z_max)
    ax2.set_xlabel("Distance z (Å)", fontsize=12)
    ax2.set_ylabel("Potential φ (kT/e)", fontsize=12)
    ax2.set_title("Electrostatic potential vs distance", fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, linestyle=":")
    plt.tight_layout()
    fig2.savefig("potential_vs_distance.png", dpi=150)
    print("Saved: potential_vs_distance.png")

    # ================================================================
    # Plot 3: Ion concentrations vs distance (matching style with Plot 1)
    # ================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    
    for res, color in zip(results, colors):
        # Always plot to z_max for consistent appearance
        z_end = min(res['Boundary'], z_max)
        z_plot = np.linspace(res['z0'] + 0.5, z_end, 500)
        conc_Na, conc_Cl = res['get_ion_conc'](z_plot)
        
        ax3a.plot(z_plot, conc_Cl, color=color, linewidth=1.5,
                  label=f"c = {res['c']:.2g} M")
        ax3b.plot(z_plot, conc_Na, color=color, linewidth=1.5,
                  label=f"c = {res['c']:.2g} M")
    
    ax3a.set_xlim(0, z_max)
    ax3a.set_xlabel("Distance z (Å)", fontsize=12)
    ax3a.set_ylabel("Cl⁻ concentration (M)", fontsize=12)
    ax3a.set_title("Cl⁻ concentration vs distance", fontsize=12)
    ax3a.legend(loc='best', fontsize=9)
    ax3a.grid(True, linestyle=":")
    
    ax3b.set_xlim(0, z_max)
    ax3b.set_xlabel("Distance z (Å)", fontsize=12)
    ax3b.set_ylabel("Na⁺ concentration (M)", fontsize=12)
    ax3b.set_title("Na⁺ concentration vs distance", fontsize=12)
    ax3b.legend(loc='best', fontsize=9)
    ax3b.grid(True, linestyle=":")
    
    plt.tight_layout()
    fig3.savefig("ion_concentrations_vs_distance.png", dpi=150)
    print("Saved: ion_concentrations_vs_distance.png")

    # ================================================================
    # Log(c) slope analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("Log(c) Slope Analysis")
    print("=" * 70)
    log_c = np.log10(conc_list)
    coeffs = np.polyfit(log_c, np.abs(E_list), 1)
    slope, intercept = coeffs
    print(f"Fitting |E| = A + B*log10(c)")
    print(f"  Slope B = {slope:.3e} V/m per decade")
    print(f"  Intercept A = {intercept:.3e} V/m")
    
    # R² value
    E_fit = np.polyval(coeffs, log_c)
    ss_res = np.sum((np.abs(E_list) - E_fit)**2)
    ss_tot = np.sum((np.abs(E_list) - np.mean(np.abs(E_list)))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  R² = {r2:.4f}")

    # ================================================================
    # Export data to CSV files
    # ================================================================
    import csv
    import os
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. E-field vs concentration
    with open(os.path.join(data_dir, 'Efield_vs_concentration.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E-field vs Concentration Data'])
        writer.writerow([f'Surface potential: {SURFACE_POTENTIAL_MV} mV'])
        writer.writerow([f'Ion size (steric correction): {ION_SIZE*1e10:.1f} Å'])
        writer.writerow([''])
        writer.writerow(['Concentration (M)', 'E-field (V/m)', '|E| (GV/m)'])
        for c, E in zip(conc_list, E_list):
            writer.writerow([f'{c:.4e}', f'{E:.4e}', f'{np.abs(E)/1e9:.4f}'])
    print(f"Saved: {data_dir}/Efield_vs_concentration.csv")
    
    # 2. E-field vs potential
    with open(os.path.join(data_dir, 'Efield_vs_potential.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['E-field vs Applied Potential Data'])
        writer.writerow([f'Concentration: {c_fixed} M'])
        writer.writerow([''])
        writer.writerow(['Potential (mV)', 'Potential (V)', 'E-field (V/m)', '|E| (GV/m)'])
        for psi_mV, E in zip(potential_sweep_mV, E_sweep):
            writer.writerow([f'{psi_mV:.0f}', f'{psi_mV/1000:.3f}', f'{E:.4e}', f'{np.abs(E)/1e9:.4f}'])
    print(f"Saved: {data_dir}/Efield_vs_potential.csv")
    
    # 3. Potential vs distance (representative concentrations)
    with open(os.path.join(data_dir, 'potential_vs_distance.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Electrostatic Potential vs Distance Data'])
        writer.writerow([f'Surface potential: {SURFACE_POTENTIAL_MV} mV ({SURFACE_POTENTIAL:.1f} kT/e)'])
        writer.writerow([''])
        # Header with all concentrations
        header = ['z (Å)'] + [f'φ at c={res["c"]:.2g} M (kT/e)' for res in results]
        writer.writerow(header)
        # Common z grid
        z_common = np.linspace(0.1, 20, 200)
        for z in z_common:
            row = [f'{z:.3f}']
            for res in results:
                if z <= res['Boundary']:
                    phi = res['sol'].sol(z)[0]
                    row.append(f'{phi:.4f}')
                else:
                    row.append('')
            writer.writerow(row)
    print(f"Saved: {data_dir}/potential_vs_distance.csv")
    
    # 4. Ion concentrations vs distance
    with open(os.path.join(data_dir, 'ion_concentrations_vs_distance.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ion Concentrations vs Distance Data'])
        writer.writerow([f'Surface potential: {SURFACE_POTENTIAL_MV} mV'])
        writer.writerow([''])
        # Header
        header = ['z (Å)']
        for res in results:
            header.extend([f'[Cl-] at c={res["c"]:.2g} M (M)', f'[Na+] at c={res["c"]:.2g} M (M)'])
        writer.writerow(header)
        # Common z grid
        z_common = np.linspace(0.6, 20, 200)
        for z in z_common:
            row = [f'{z:.3f}']
            for res in results:
                if z <= res['Boundary']:
                    conc_Na, conc_Cl = res['get_ion_conc'](np.array([z]))
                    row.extend([f'{conc_Cl[0]:.6e}', f'{conc_Na[0]:.6e}'])
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    print(f"Saved: {data_dir}/ion_concentrations_vs_distance.csv")

    plt.show()

