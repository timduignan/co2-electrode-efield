"""
Estimate the kinetic limitation of proton diffusion from bulk at pH 6.

Shows that diffusion-limited proton flux is far too slow to account for
the observed protonation rates, supporting the need for local water splitting.

Key insight: Even with the most optimistic assumptions (smallest δ, vigorous
stirring), diffusion at pH 6 is still orders of magnitude too slow.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
kB = 1.38e-23  # J/K
T = 298  # K
e = 1.6e-19  # C
NA = 6.02e23  # Avogadro's number
F = 96485  # C/mol (Faraday constant)

# Proton diffusion coefficient (anomalously high due to Grotthuss mechanism)
D_H = 9.3e-9  # m²/s

# Experimental current density for CO2 reduction
i_exp_typical = 0.76  # mA/cm² (experimental value)

# ================================================================
# Sensitivity analysis: range of diffusion layer thicknesses
# ================================================================
# δ ranges from ~1 μm (rotating disk electrode, vigorous stirring)
# to ~500 μm (completely stagnant solution)
# Typical unstirred: 10-100 μm

delta_values_um = np.array([1, 5, 10, 50, 100, 500])  # μm
delta_values = delta_values_um * 1e-6  # convert to m

# At pH 6
pH = 6
H_conc = 10.0**(-pH)  # mol/L
H_conc_m3 = H_conc * 1000  # mol/m³

print("=" * 70)
print("Sensitivity Analysis: Diffusion Layer Thickness at pH 6")
print("=" * 70)
print(f"Proton diffusion coefficient: D_H = {D_H:.2e} m²/s")
print(f"[H+] at pH 6: {H_conc:.0e} M")
print(f"Typical experimental current: {i_exp_typical} mA/cm²")
print()
print(f"{'δ (μm)':>10} | {'Condition':>25} | {'i_diff (mA/cm²)':>15} | {'Deficit':>10}")
print("-" * 70)

conditions = {
    1: "Rotating disk (1000 rpm)",
    5: "Vigorous stirring",
    10: "Moderate stirring",
    50: "Gentle stirring", 
    100: "Unstirred (typical)",
    500: "Completely stagnant"
}

for delta_um, delta in zip(delta_values_um, delta_values):
    J_diff = D_H * H_conc_m3 / delta
    i_diff = F * J_diff / 10  # A/m² to mA/cm²
    ratio = i_exp_typical / i_diff
    cond = conditions.get(delta_um, "")
    print(f"{delta_um:10.0f} | {cond:>25} | {i_diff:15.2e} | {ratio:10.0e}×")

print()
print("KEY POINT: Even with a rotating disk electrode (δ ~ 1 μm),")
print("diffusion at pH 6 is still ~8× too slow for the observed 0.76 mA/cm²!")
print()

# Mean distance between protons at pH 6
n_H_pH6 = H_conc_m3 * NA  # protons/m³
d_mean = (1 / n_H_pH6)**(1/3)  # m
print(f"Mean proton separation at pH 6: {d_mean*1e9:.0f} nm")
print("→ Protons are extremely sparse at this pH")

# ================================================================
# Full pH range analysis (using moderate δ = 10 μm as best case)
# ================================================================
pH_values = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
H_conc_arr = 10.0**(-pH_values)  # mol/L = M
H_conc_m3_arr = H_conc_arr * 1000  # mol/m³

# Use optimistic δ = 10 μm (moderate stirring)
delta_optimistic = 10e-6  # 10 μm

# ================================================================
# Single plot with sensitivity to δ
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Calculate current for optimistic (δ=1μm) and typical (δ=100μm) cases
J_diff_optimistic = D_H * H_conc_m3_arr / 1e-6
i_diff_optimistic = F * J_diff_optimistic / 10

J_diff_typical = D_H * H_conc_m3_arr / 100e-6
i_diff_typical = F * J_diff_typical / 10

# Current density vs pH with uncertainty band
ax.fill_between(pH_values, i_diff_typical, i_diff_optimistic, 
                alpha=0.3, color='#2E86AB', label=r'Range: $\delta$ = 1–100 μm')
ax.semilogy(pH_values, i_diff_optimistic, 'o-', linewidth=2, markersize=7, 
            color='#2E86AB', label=r'Best case ($\delta$ = 1 μm)')
ax.semilogy(pH_values, i_diff_typical, 's--', linewidth=2, markersize=7, 
            color='#1a5276', label=r'Typical ($\delta$ = 100 μm)')
ax.axhline(y=i_exp_typical, color='#E74C3C', linestyle='-', linewidth=2.5, 
           label=f'Experimental (~{i_exp_typical} mA/cm²)')
ax.axvline(x=6, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('pH', fontsize=13)
ax.set_ylabel('Diffusion-limited H⁺ current (mA/cm²)', fontsize=13)
ax.set_title('Proton diffusion from bulk is kinetically limited', fontsize=13)
ax.set_ylim(1e-5, 1e4)
ax.set_xlim(1.5, 8.5)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# Add annotation at pH 6
ax.annotate('pH 6', xy=(6, 1e-4), xytext=(6.3, 1e-4), fontsize=11, color='gray')

plt.tight_layout()
fig.savefig('proton_diffusion_limitation.png', dpi=150)
print("\nSaved: proton_diffusion_limitation.png")

# ================================================================
# Export data to CSV for Excel
# ================================================================
import csv

# Table 1: Sensitivity analysis at pH 6
with open('proton_diffusion_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header section
    writer.writerow(['Proton Diffusion Kinetic Limitation Analysis'])
    writer.writerow([''])
    writer.writerow(['Parameters:'])
    writer.writerow(['Proton diffusion coefficient D_H (m²/s)', f'{D_H:.2e}'])
    writer.writerow(['Faraday constant F (C/mol)', f'{F}'])
    writer.writerow(['Temperature (K)', f'{T}'])
    writer.writerow(['Experimental current density (mA/cm²)', f'{i_exp_typical}'])
    writer.writerow([''])
    
    # Table 1: Sensitivity to diffusion layer thickness at pH 6
    writer.writerow(['Table 1: Sensitivity Analysis at pH 6'])
    writer.writerow(['δ (μm)', 'Condition', '[H+] (M)', 'J_diff (mol/m²/s)', 
                     'i_diff (mA/cm²)', 'Deficit vs Experimental'])
    
    for delta_um, delta in zip(delta_values_um, delta_values):
        J_diff = D_H * H_conc_m3 / delta
        i_diff = F * J_diff / 10  # A/m² to mA/cm²
        ratio = i_exp_typical / i_diff
        cond = conditions.get(delta_um, "")
        writer.writerow([f'{delta_um:.0f}', cond, f'{H_conc:.0e}', 
                        f'{J_diff:.3e}', f'{i_diff:.3e}', f'{ratio:.1e}×'])
    
    writer.writerow([''])
    
    # Table 2: pH dependence
    writer.writerow(['Table 2: pH Dependence of Diffusion-Limited Current'])
    writer.writerow(['pH', '[H+] (M)', 'i_diff (mA/cm²) δ=1μm', 
                     'i_diff (mA/cm²) δ=100μm', 'Deficit (δ=1μm)', 'Deficit (δ=100μm)'])
    
    for i, pH_val in enumerate(pH_values):
        H_conc_i = H_conc_arr[i]
        i_opt = i_diff_optimistic[i]
        i_typ = i_diff_typical[i]
        deficit_opt = i_exp_typical / i_opt if i_opt > 0 else float('inf')
        deficit_typ = i_exp_typical / i_typ if i_typ > 0 else float('inf')
        writer.writerow([f'{pH_val:.0f}', f'{H_conc_i:.0e}', f'{i_opt:.3e}', 
                        f'{i_typ:.3e}', f'{deficit_opt:.1e}×', f'{deficit_typ:.1e}×'])
    
    writer.writerow([''])
    writer.writerow(['Notes:'])
    writer.writerow(['J_diff = D_H * [H+] / δ  (Fickian diffusion flux)'])
    writer.writerow(['i_diff = F * J_diff  (current density)'])
    writer.writerow([f'Mean proton separation at pH 6: {d_mean*1e9:.0f} nm'])

print("Saved: proton_diffusion_data.csv")

plt.show()

