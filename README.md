# Electrode Electric Field Solver

Modified Poisson-Boltzmann solver with Bikerman steric correction for calculating electric fields at electrode-electrolyte interfaces.

## Overview

This code calculates the electric field magnitude at a charged electrode surface in contact with an aqueous electrolyte. The model includes:

- **Bikerman steric correction**: Prevents unphysical ion crowding at high potentials by capping ion density at the close-packing limit
- **Image charge potentials**: Accounts for ion-surface image interactions for both Na⁺ and Cl⁻
- **Concentration and potential sweeps**: Generate E-field as a function of bulk salt concentration or applied potential

## Installation

Requires Python 3 with NumPy, SciPy, and Matplotlib:

```bash
pip install numpy scipy matplotlib
```

## Usage

Generate all plots and data:

```bash
python make_plots.py
```

This produces:
- `Efield_vs_concentration.png` - E-field vs bulk NaCl concentration
- `Efield_vs_potential.png` - E-field vs applied surface potential
- `potential_vs_distance.png` - Electrostatic potential profiles
- `ion_concentrations_vs_distance.png` - Na⁺ and Cl⁻ concentration profiles

Data is exported to CSV files in the `data/` directory.

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Surface potential | -200 mV | Default applied potential |
| Ion size | 3.5 Å | Hydrated ion diameter for steric correction |
| E-field evaluation | z = 2.6 Å | Distance from surface (COO⁻ to nearest water H) |
| Concentration range | 0.01 – 5 M | Bulk NaCl concentration |

## Model Details

The solver uses `scipy.integrate.solve_bvp` to solve the modified Poisson-Boltzmann equation:

$$\frac{d^2\phi}{dz^2} = -\frac{e}{\varepsilon_0 \varepsilon_r} \sum_i z_i c_i^{bulk} \frac{e^{-z_i \phi - U_i(z)}}{1 + \nu(\sum_j e^{-z_j \phi - U_j(z)} - n_{species})}$$

where the denominator implements the Bikerman steric correction and $U_i(z)$ are the image charge potentials.

## Other Scripts

- `compare_vanilla_PB.py` - Comparison between standard PB and Stern layer models
- `proton_diffusion_estimate.py` - Estimates proton diffusion limitations
- `energy_profile_calculation.py` - Proton transfer energy profile calculations

## Citation

If you use this code, please cite: [Add citation]

## License

[Add license]

