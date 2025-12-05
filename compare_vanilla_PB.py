"""
Compare modified PB (with Stern layer) vs vanilla Gouy-Chapman (no Stern layer).

Shows that:
- Modified PB with Stern layer → log(c) scaling
- Vanilla Gouy-Chapman → √c scaling
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
V_thermal = kB * T / elc

SURFACE_POTENTIAL_MV = -200
SURFACE_POTENTIAL = SURFACE_POTENTIAL_MV * 1e-3 / V_thermal
ION_SIZE = 3.5e-10


def heaviside_theta(x):
    return np.heaviside(np.asarray(x), 0.0)


def solve_PB(c, surface_potential, use_stern_layer=True):
    """
    Solve PB equation.
    
    use_stern_layer=True: Include ion-surface potentials (Stern layer + image charges)
    use_stern_layer=False: Vanilla Gouy-Chapman (ions can approach z=0)
    """
    rhoIonNa = c * Avog * 1000.0
    rhoIonCl = rhoIonNa
    kappa = 1.0 / (0.304 / np.sqrt(c) * 1e-9)
    kappa_A = kappa * 1e-10
    Boundary = max((1.0 / kappa) * 1e10 * 10.0, 25.0)
    
    rho_max = 1.0 / ION_SIZE**3
    nu = rhoIonNa / rho_max

    if use_stern_layer:
        # --- Cl- image term ---
        ahCl = 2.0e-10
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
        ahNa = 2.5e-10
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
    else:
        # Vanilla Gouy-Chapman: no ion-surface interactions
        def UCl(z):
            return np.zeros_like(np.asarray(z, dtype=float))
        def UNa(z):
            return np.zeros_like(np.asarray(z, dtype=float))

    z0 = 0.1

    def ode(z, y):
        phi, dphi = y[0], y[1]
        factor = -1.0e-20 * elc**2 / (eps0 * 78.3) / (kB * T)
        
        exp_na = np.clip(-UNa(z) - phi, -700, 700)
        exp_cl = np.clip(-UCl(z) + phi, -700, 700)
        
        boltz_na = np.exp(exp_na)
        boltz_cl = np.exp(exp_cl)
        
        if use_stern_layer:
            # Bikerman steric correction (only for modified PB)
            denom = 1.0 + nu * (boltz_na + boltz_cl - 2.0)
            denom = np.maximum(denom, 1e-10)
            rhs = factor * (rhoIonNa * boltz_na / denom -
                           rhoIonCl * boltz_cl / denom)
        else:
            # Pure Gouy-Chapman: no steric correction
            rhs = factor * (rhoIonNa * boltz_na - rhoIonCl * boltz_cl)
        
        return np.vstack((dphi, rhs))

    def bc(ya, yb):
        return np.array([ya[0] - surface_potential, yb[0] - (-1.0e-4)])

    z_mesh = np.unique(np.concatenate([
        np.linspace(z0, 10, 200),
        np.linspace(10.01, Boundary, 400)
    ]))
    
    phi_guess = surface_potential * np.exp(-kappa_A * (z_mesh - z0))
    dphi_guess = -kappa_A * phi_guess
    y_init = np.vstack((phi_guess, dphi_guess))

    sol = solve_bvp(ode, bc, z_mesh, y_init, max_nodes=30000, tol=1e-5, verbose=0)

    # E-field at z* = 2.6 Å (COO⁻ to nearest water H distance from DFT)
    zSurf = 2.6
    dphi_dz = sol.sol(zSurf)[1] * 1.0e10
    E = -(kB * T / elc) * dphi_dz
    return E


def gouy_chapman_analytical(c, phi0):
    """
    Analytical Gouy-Chapman surface field: |E| = 2κ(kT/e)sinh(φ₀/2)
    This gives √c scaling.
    """
    kappa = 1.0 / (0.304 / np.sqrt(c) * 1e-9)
    return 2.0 * kappa * V_thermal * np.abs(np.sinh(phi0 / 2.0))


if __name__ == "__main__":
    # Concentration range
    exponents = np.arange(-1.3, np.log10(5.0) + 1e-9, 0.15)
    conc_list = 10.0**exponents
    
    print("Computing E-field for Modified PB (with Stern layer)...")
    E_stern = []
    for c in conc_list:
        E = solve_PB(c, SURFACE_POTENTIAL, use_stern_layer=True)
        E_stern.append(E)
        print(f"  c = {c:.3g} M: E = {np.abs(E)/1e9:.3f} GV/m")
    E_stern = np.array(E_stern)
    
    print("\nComputing E-field for Vanilla Gouy-Chapman (no Stern layer)...")
    E_vanilla = []
    for c in conc_list:
        E = solve_PB(c, SURFACE_POTENTIAL, use_stern_layer=False)
        E_vanilla.append(E)
        print(f"  c = {c:.3g} M: E = {np.abs(E)/1e9:.3f} GV/m")
    E_vanilla = np.array(E_vanilla)
    
    # Analytical Gouy-Chapman for comparison
    E_gc_analytical = gouy_chapman_analytical(conc_list, SURFACE_POTENTIAL)
    
    # ================================================================
    # Plot: Compare the two models (single semi-log plot)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogx(conc_list, np.abs(E_stern) / 1e9, 'o-', linewidth=2, 
                markersize=7, color='#2E86AB', label='Modified PB (Stern layer)')
    ax.semilogx(conc_list, np.abs(E_vanilla) / 1e9, 's-', linewidth=2, 
                markersize=7, color='#E74C3C', label='Gouy-Chapman (no Stern)')
    ax.set_xlabel("Concentration c (M)", fontsize=13)
    ax.set_ylabel("|E| at z = 2.6 Å (GV/m)", fontsize=13)
    ax.set_title("Modified PB shows log(c) scaling; Gouy-Chapman does not", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    fig.savefig("PB_comparison_stern_vs_vanilla.png", dpi=150)
    print("\nSaved: PB_comparison_stern_vs_vanilla.png")
    
    # ================================================================
    # Fit analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("Scaling Analysis")
    print("=" * 60)
    
    log_c = np.log10(conc_list)
    sqrt_c = np.sqrt(conc_list)
    
    # Modified PB: fit to log(c)
    coeffs_stern_log = np.polyfit(log_c, np.abs(E_stern), 1)
    E_fit_stern_log = np.polyval(coeffs_stern_log, log_c)
    ss_res = np.sum((np.abs(E_stern) - E_fit_stern_log)**2)
    ss_tot = np.sum((np.abs(E_stern) - np.mean(np.abs(E_stern)))**2)
    r2_stern_log = 1 - ss_res / ss_tot
    
    # Modified PB: fit to √c
    coeffs_stern_sqrt = np.polyfit(sqrt_c, np.abs(E_stern), 1)
    E_fit_stern_sqrt = np.polyval(coeffs_stern_sqrt, sqrt_c)
    ss_res = np.sum((np.abs(E_stern) - E_fit_stern_sqrt)**2)
    ss_tot = np.sum((np.abs(E_stern) - np.mean(np.abs(E_stern)))**2)
    r2_stern_sqrt = 1 - ss_res / ss_tot
    
    # Vanilla: fit to log(c)
    coeffs_vanilla_log = np.polyfit(log_c, np.abs(E_vanilla), 1)
    E_fit_vanilla_log = np.polyval(coeffs_vanilla_log, log_c)
    ss_res = np.sum((np.abs(E_vanilla) - E_fit_vanilla_log)**2)
    ss_tot = np.sum((np.abs(E_vanilla) - np.mean(np.abs(E_vanilla)))**2)
    r2_vanilla_log = 1 - ss_res / ss_tot
    
    # Vanilla: fit to √c
    coeffs_vanilla_sqrt = np.polyfit(sqrt_c, np.abs(E_vanilla), 1)
    E_fit_vanilla_sqrt = np.polyval(coeffs_vanilla_sqrt, sqrt_c)
    ss_res = np.sum((np.abs(E_vanilla) - E_fit_vanilla_sqrt)**2)
    ss_tot = np.sum((np.abs(E_vanilla) - np.mean(np.abs(E_vanilla)))**2)
    r2_vanilla_sqrt = 1 - ss_res / ss_tot
    
    print("\nModified PB (with Stern layer):")
    print(f"  Fit to log(c): R² = {r2_stern_log:.4f}")
    print(f"  Fit to √c:     R² = {r2_stern_sqrt:.4f}")
    print(f"  → Better fit: {'log(c)' if r2_stern_log > r2_stern_sqrt else '√c'}")
    
    print("\nVanilla Gouy-Chapman (no Stern layer):")
    print(f"  Fit to log(c): R² = {r2_vanilla_log:.4f}")
    print(f"  Fit to √c:     R² = {r2_vanilla_sqrt:.4f}")
    print(f"  → Better fit: {'log(c)' if r2_vanilla_log > r2_vanilla_sqrt else '√c'}")
    
    # ================================================================
    # Export data to CSV
    # ================================================================
    import csv
    import os
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'PB_comparison_stern_vs_vanilla.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Modified PB vs Gouy-Chapman Comparison'])
        writer.writerow([f'Surface potential: {SURFACE_POTENTIAL_MV} mV'])
        writer.writerow(['z* (measurement distance): 2.6 Å'])
        writer.writerow([''])
        writer.writerow(['Model Comparison'])
        writer.writerow(['Concentration (M)', 'Modified PB |E| (GV/m)', 'Gouy-Chapman |E| (GV/m)', 
                         'GC Analytical E(0) (GV/m)'])
        for i, c in enumerate(conc_list):
            writer.writerow([f'{c:.4e}', f'{np.abs(E_stern[i])/1e9:.4f}', 
                           f'{np.abs(E_vanilla[i])/1e9:.4f}', f'{E_gc_analytical[i]/1e9:.4f}'])
        writer.writerow([''])
        writer.writerow(['Scaling Analysis'])
        writer.writerow(['Model', 'R² to log(c)', 'R² to √c', 'Better Fit'])
        writer.writerow(['Modified PB (Stern)', f'{r2_stern_log:.4f}', f'{r2_stern_sqrt:.4f}',
                        'log(c)' if r2_stern_log > r2_stern_sqrt else '√c'])
        writer.writerow(['Gouy-Chapman', f'{r2_vanilla_log:.4f}', f'{r2_vanilla_sqrt:.4f}',
                        'log(c)' if r2_vanilla_log > r2_vanilla_sqrt else '√c'])
    print(f"\nSaved: {data_dir}/PB_comparison_stern_vs_vanilla.csv")
    
    plt.show()

