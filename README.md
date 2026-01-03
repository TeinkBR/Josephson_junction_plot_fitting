# Diffusive Spin-Triplet Josephson Junction Simulation

## Overview

This package implements a comprehensive simulation framework for **diffusive spin-triplet Josephson junctions** with **antiferromagnetically coupled perpendicular magnetization domains** and **spin-glass interface layers**. The framework is specifically designed to model the experimental observations reported by Li et al. in their study of Nb/Fe/Cr multilayer junctions.

## Key Features

### 1. **Diffusive Transport (Usadel Equations)**
- Appropriate for sputtered Nb with short mean free path (~5 nm)
- Distinguishes between singlet (short-range) and triplet (long-range) transport
- Computes coherence lengths: $\xi_s \approx 0.08$ nm (singlet), $\xi_T \approx 0.8-50$ nm (triplet)

### 2. **Spin-Glass Interface Modeling**
- Models F' layers (0.25 nm each) as frozen spin-glass with random exchange couplings
- Uses **Heisenberg Random Exchange Hamiltonian** for inter-domain frustration
- Implements **Metropolis energy minimization** for field-step evolution
- **History-dependent** magnetization that doesn't fully relax to external field

### 3. **Long-Range Triplet Component (LRTC)**
- LRTC amplitude: $\propto \sin(\theta_L) \sin(\theta_R)$ where $\theta_L$, $\theta_R$ are non-collinearity angles
- Maximum triplet current when magnetization layers are **perpendicular** to bulk Fe
- Decay: $\exp(-d_F/\xi_T)$ showing long-range persistence

### 4. **Spectral Leakage**
- Disorder in F' layers prevents complete suppression expected in clean synthetic antiferromagnets (SAF)
- Explains Li's observation: **no suppression in upper field sweep** despite antiferromagnetic configuration
- Quantified via `leakage_ratio = std(sin theta) / |mean(sin theta)|`

### 5. **Magnetic Field Dependence & Hysteresis**
- Full Fraunhofer pattern calculation with spatially non-uniform $j_c(x,y)$
- Hysteresis from spin-glass memory effects
- Phase shift $\delta$ extracted from residual magnetization

## Project Structure

```
rcsj_sde/
├── rcsj_sde/
│   ├── diffusive_triplet.py          # Module 1: Geometry, materials, spin-glass
│   ├── fraunhofer_diffusive.py       # Module 3: Fraunhofer integration (Usadel kernel)
│   ├── diffusive_rcsj.py             # Module 4: RCSJ dynamics with B-dependent I_c
│   ├── validation_tools.py           # Module 5: Validation & parameter extraction
│   ├── junction.py                   # Base junction class (original)
│   └── ... (other original files)
├── examples/
│   ├── diffusive_complete_workflow.ipynb  # Full simulation tutorial
│   └── ... (other example notebooks)
├── test_simulation.py                # Validation test script
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Installation & Setup

### 1. Create Virtual Environment

```bash
cd rcsj_sde
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install numpy scipy matplotlib scikit-optimize tqdm pandas numba
```

Or use:
```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python3 test_simulation.py
```

Expected output: "[PASS] ALL TESTS PASSED"

## Usage

### Quick Start: Run Complete Workflow

```bash
source venv/bin/activate
jupyter notebook examples/diffusive_complete_workflow.ipynb
```

The notebook demonstrates all stages:
1. Initialize 5-layer junction geometry (S/F'/F/F'/S)
2. Setup magnetic configuration with spin-glass
3. Compute Fraunhofer pattern
4. Extract Li's fit function parameters
5. Analyze spectral leakage and hysteresis

### Programmatic Usage Example

```python
from rcsj_sde.diffusive_triplet import (
    DiffusiveJunctionGeometry, MaterialParameters, MagneticConfiguration, UsadelKernel
)
from rcsj_sde.validation_tools import spectral_leakage_analysis

# Define junction geometry
geometry = DiffusiveJunctionGeometry(
    d_S=100.0,              # Nb superconductor (nm)
    d_F_prime=0.25,         # Spin-glass interface (nm)
    d_F=10.0,               # Bulk Fe (nm)
    junction_length=1000.0,
    junction_width=1000.0,
    n_grid_x=30
)

# Define material parameters
materials = MaterialParameters(
    Delta=1.5,      # Nb gap (meV)
    E_ex=100.0,     # Fe exchange (meV)
    T=2.0           # Temperature (K)
)

# Create magnetic configuration with spin-glass
mag_config = MagneticConfiguration(geometry, n_spins_per_interface=50)

# Evolve during field sweep
B_sweep = [0, 0.01, 0.02, 0.01, 0]
for B in B_sweep:
    mag_config.evolve_field_step(B, n_relax_steps=100, T_eff=0.05)

# Analyze spectral leakage (protects triplet from SAF suppression)
leakage = spectral_leakage_analysis(mag_config)
print(f"Disorder protected: {leakage['disorder_protected']}")
```

## Physical Model

### 5-Layer Stack: S/F'/F/F'/S

| Layer | Material | Thickness | Role |
|-------|----------|-----------|------|
| **S (outer)** | Nb | ~100 nm | Superconductor (BCS gap = 1.5 meV) |
| **F' (left)** | Fe/Cr | 0.25 nm | Spin-mixer with random exchange |
| **F (bulk)** | Fe | ~10-15 nm | Ferromagnet with exchange splitting |
| **F' (right)** | Fe/Cr | 0.25 nm | Spin-mixer with random exchange |

### Transport Kernel (Houzet-Buzdin)

Local supercurrent density from LRTC:

$$j_c(x,y) = j_0 \left[ \sin(\theta_L) \sin(\theta_R) \cdot e^{-d_F/\xi_T} + \cos(\theta_L) \cos(\theta_R) \cdot e^{-d_F/\xi_s} \right]$$

where:
- $\theta_L$, $\theta_R$ = angles between F' and bulk F magnetization
- $\xi_T$, $\xi_s$ = triplet and singlet coherence lengths
- Triplet term dominates due to much longer range

### Fraunhofer Integral

Total critical current from spatial integration:

$$I_c(B) = \left| \int\int j_c(x,y) \cdot e^{i\phi_{flux}(x,y)} dx\, dy \right|$$

where $\phi_{flux} = 2\pi \Phi / \Phi_0$ accounts for external and domain-induced magnetization.

### Li's Empirical Fit Function

$$I_c(\Phi) = I_{c0} \cdot e^{-d_{Nb}/\xi_T} \cdot \left| \frac{J_1\left(\pi(\Phi-\delta)/\Phi_0\right)}{\pi(\Phi-\delta)/\Phi_0} \right|$$

**Extracted Parameters:**
- $I_{c0}$: Maximum critical current
- $xi_T$: Triplet coherence length (meV*nm)
- $\delta$: Phase shift from residual F' magnetization (rad)

## Key Physics Insights

### 1. "Missing Suppression" Anomaly

In a clean synthetic antiferromagnet (SAF), the two F' layers would create destructive interference, completely suppressing the supercurrent. However, Li observes **no suppression** in the upper field sweep.

**Explanation from simulation:**
- Spin-glass disorder creates broad distribution of magnetization angles
- Some regions have strong LRTC signal ($\sin\theta_L \sin\theta_R > 0$)
- Spatial averaging over disorder protects global triplet current: **"spectral leakage"**
- Quantified by `leakage_ratio` parameter

### 2. Hysteresis from Spin-Glass Memory

The F' layers freeze in history-dependent configurations:
- **Up sweep**: Spins partially aligned with $+B$
- **Down sweep**: Spins "lag" behind, partially aligned with previous $+B$
- **Consequence**: $I_c^{up}(B) \neq I_c^{down}(B)$

### 3. Phase Shift from Remanent Magnetization

After field sweep, F' layers retain net magnetization despite nominally antiparallel bulk Fe:
- Creates local magnetic flux contribution
- Shifts Fraunhofer pattern by phase $\delta$
- Directly observable in critical current vs. flux curve

## Validation & Comparison

### Metrics Computed

1. **Fraunhofer Pattern**
   - Period (in units of $\Phi_0$)
   - Contrast: $(I_{c,max} - I_{c,min}) / I_{c,max}$
   - Asymmetry: left-right asymmetry measure

2. **Spectral Leakage**
   - LRTC mean and std
   - Leakage ratio (disorder quantifier)
   - Boolean: disorder-protected or not

3. **Hysteresis**
   - Max difference between up/down sweeps
   - RMS difference
   - Hysteresis loss (area between curves)

4. **Fit Quality**
   - $\chi^2$ against Li's function
   - $R^2$ coefficient of determination
   - Extracted parameters: $I_{c0}$, $\xi_T$, $\delta$

### Comparison with Experiment

The notebook includes code to compare simulation results directly with Li et al. experimental data:

```python
from rcsj_sde.validation_tools import compare_simulation_to_experiment

result = compare_simulation_to_experiment(
    B_range_sim, Ic_sim,
    B_range_exp, Ic_exp,
    junction_area
)

print(f"R^2 = {result['r_squared']:.4f}")
print(f"Extracted xi_T = {result['extracted_params']['xi_T']:.1f} nm")
print(f"Extracted delta = {result['extracted_params']['delta']:.4f} rad")
```

## Computational Performance

| Task | Time | Notes |
|------|------|-------|
| Spin-glass init | ~0.1 s | Random exchange matrix generation |
| Single field step | ~0.5 s | 100 Metropolis steps per point |
| Full field sweep (50 points) | ~25 s | Up and down together |
| Fraunhofer integral | ~2 s | Spatial integration over grid |
| RCSJ I-V single point | ~1 s | 5000 time steps |
| Full I-V map (50 fields, 50 currents) | ~2500 s | ~45 minutes |

**Optimization tips:**
- Use coarser spatial grids (n_grid_x=10-20) for quick exploration
- Use fewer Metropolis steps (n_relax_steps=50) during development
- Parallelize I-V calculations over field values (not yet implemented)

## References

### Core Theory
1. **Houzet & Buzdin (2007)**: "Long range triplet Josephson effect through a ferromagnetic trilayer", *Phys. Rev. B* **76**, 060504(R)
2. **Bergeret, Volkov, Efetov (2005)**: "Proximity effects in superconductor-ferromagnet heterostructures", *Rev. Mod. Phys.* **77**, 1321
3. **Eschrig (2011)**: "Spin-polarized supercurrents for spintronics", *Physics Today* **64** (1), 43

### Experimental Context
4. **Li et al. (2024)**: "Spin-orbit coupling suppression and singlet blocking of spin-triplet Cooper pairs"
5. **Robinson et al. (2010)**: "Controlled Injection of Spin-Triplet Supercurrents", *Science* **329**, 59
6. **Khaire et al. (2010)**: "Observation of Spin-Triplet Superconductivity in Co-Based Josephson Junctions", *PRL* **104**, 137002

### Spin-Glass Physics
7. **Spin-Glass Theory**: Edwards-Anderson Model, Sherrington-Kirkpatrick Model
8. **Random Exchange**: Frustrated magnetic systems with disorder

## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall packages if needed
pip install --upgrade numpy scipy numba
```

### Numerical Issues

**Problem**: LRTC values near zero or negative
- Check: Are F' layers properly initialized with random orientations?
- Verify: Non-collinearity angles should be distributed across [0, pi]

**Problem**: Fraunhofer pattern completely flat
- Check: Is j_c_profile being computed correctly?
- Verify: Coherence length ratios are reasonable

**Problem**: Slow performance
- Use coarser grids: `n_grid_x = 10, n_grid_y = 10`
- Reduce Metropolis steps: `n_relax_steps = 50`
- Reduce field sweep points

## Future Extensions

1. **Dynamical Domain Walls**: Replace Metropolis with LLG dynamics
2. **Spin-Orbit Coupling**: Add Rashba SOC effects on CPR
3. **Shapiro Steps**: AC drive with existing framework
4. **Temperature Dependence**: Full T(xi_T) and T(Delta) dependence
5. **Parallel I-V**: Distribute calculations over CPU cores
6. **Experimental Fitting**: Direct optimization against Li's data

## Contributing

To extend this framework:

1. **New physics**: Add to appropriate module (`diffusive_triplet.py` for geometry/materials, `fraunhofer_diffusive.py` for transport)
2. **New validation metrics**: Add to `validation_tools.py`
3. **New examples**: Create notebook in `examples/`
4. **Testing**: Add test cases to `test_simulation.py`

## License

See LICENSE file in the root directory.

## Contact & Support

For questions about this implementation, refer to:
- The comprehensive tutorial notebook: `examples/diffusive_complete_workflow.ipynb`
- The physical parameter tables in README sections 1-2
- The validation test: `test_simulation.py`

---

**Last Updated**: January 2026  
**Framework Version**: 1.0 (Diffusive Usadel + Spin-Glass)
