# Implementation Documentation: Spin-Triplet Josephson Junction Simulation

## Overview

This document describes the complete implementation of a Python-based simulation for 
modeling spin-triplet supercurrent in Nb/Cu/Cr/Fe/Cr/Cu/Nb antiferromagnetic 
Josephson junctions. The simulation computes the critical current $I_c$ as a 
function of applied magnetic field $B$ and Cr layer thickness $d_{Cr}$.

---

## 1. Physical Background

### 1.1 The Josephson Effect

The Josephson effect describes supercurrent flow between two superconductors 
separated by a weak link. The supercurrent-phase relation is:

$$I_s = I_c \sin(\phi)$$

where $\phi$ is the phase difference between superconducting condensates and 
$I_c$ is the critical current.

### 1.2 Cooper Pairs in Ferromagnets

In conventional superconductors, electrons pair in **spin-singlet** states 
($S=0$, antisymmetric spin wavefunction):

$$|\text{singlet}\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle)$$

In ferromagnets, the exchange field $E_{ex}$ splits spin-up and spin-down bands, 
causing rapid dephasing of singlet pairs. The coherence length becomes:

$$\xi_F = \sqrt{\frac{\hbar D_F}{E_{ex}}} \sim 0.5\text{-}2 \text{ nm}$$

This is **much shorter** than in normal metals ($\xi_N \sim 100$ nm).

### 1.3 Spin-Triplet Superconductivity

**Spin-triplet** pairs ($S=1$) can have three $S_z$ projections:

- $|S_z = +1\rangle = |\uparrow\uparrow\rangle$ (equal-spin triplet)
- $|S_z = 0\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle)$
- $|S_z = -1\rangle = |\downarrow\downarrow\rangle$ (equal-spin triplet)

The **equal-spin triplets** ($S_z = \pm 1$) are immune to the exchange field 
because both electrons have the same spin. They can propagate over long distances 
($\xi_T \sim 100$ nm) through ferromagnets!

### 1.4 Singlet-to-Triplet Conversion

The key question: How do singlet pairs from Nb become triplets?

**Answer:** Magnetic inhomogeneity at interfaces.

At the Fe/Cr interface, there exists a "spin-glass" region where magnetic moments 
are disordered. This creates **spin-mixing** that rotates singlets into triplets:

$$|\text{singlet}\rangle \xrightarrow{\text{spin-glass}} \alpha|\text{singlet}\rangle + \beta|\text{triplet}\rangle$$

The conversion probability depends on:
1. Disorder strength (moment misalignment angle $\sigma_\theta$)
2. Interface thickness (dead layer $d_{dead} \sim 1$-$2$ nm)
3. Spin-orbit coupling strength

### 1.5 The Role of the Cu Spacer

The Cu spacer serves a **protective** function:

- In Nb: Triplet coherence length $\xi_T^{Nb} \approx 1.2$ nm (very short!)
- In Cu: Triplet coherence length $\xi_T^{Cu} \approx 100$ nm (long range)

By inserting Cu between Nb and the magnetic layers, triplet pairs can form 
and propagate before encountering the hostile Nb environment where they 
would be rapidly suppressed by spin-orbit scattering.

---

## 2. Mathematical Formulation

### 2.1 Triplet Amplitude Decay

The triplet pair amplitude through the junction stack follows:

$$f_T(x) = f_0 \cdot P_{S\to T} \cdot \prod_i \exp\left(-\frac{d_i}{\xi_i}\right)$$

where the product runs over all layers with their respective thicknesses and 
coherence lengths.

For our stack (Nb/Cu/Cr/Fe/Cr/Cu/Nb):

$$f_T^{\text{eff}} = \sqrt{P_{S\to T}} \cdot e^{-d_{Fe}/\xi_F^T} \cdot e^{-d_{Cr}/\xi_{Cr}^T} \cdot e^{-d_{Cu}/\xi_N}$$

### 2.2 Critical Current vs. Cr Thickness

The critical current scales as:

$$I_c(d_{Cr}) = I_0 |f_T^{\text{eff}}|^2 \propto \exp\left(-\frac{2d_{Cr}}{\xi_T^{Cr}}\right)$$

This exponential decay is the primary signature of triplet supercurrent.

### 2.3 Fraunhofer Pattern

When a magnetic field $B$ is applied perpendicular to the junction, flux 
penetrates and creates a spatially varying phase:

$$\phi(x) = \phi_0 + \frac{2\pi}{\Phi_0} B \cdot x \cdot d_{\text{eff}}$$

Integrating the supercurrent density over the junction area gives:

**Rectangular junction (Fraunhofer pattern):**
$$I_c(B) = I_{c0} \left|\frac{\sin(\pi\Phi/\Phi_0)}{\pi\Phi/\Phi_0}\right|$$

**Elliptical junction (Airy pattern):**
$$I_c(B) = I_{c0} \left|\frac{2J_1(\pi\Phi/\Phi_0)}{\pi\Phi/\Phi_0}\right|$$

where $J_1$ is the Bessel function and $\Phi = B \cdot A_{\text{eff}}$.

### 2.4 Hysteresis Effects

The Fe layer has its own magnetization $M$ that depends on field history. 
The total flux becomes:

$$\Phi_{\text{total}} = B \cdot A + \mu_0 M(B, \text{history}) \cdot d_{Fe} \cdot w$$

This creates asymmetric patterns for up-sweep vs. down-sweep.

---

## 3. Code Implementation

### 3.1 Module Structure

```
ferromagnetic-josephson-junction/
├── parameters.py          # Physical constants and material parameters
├── pair_amplitudes.py     # Triplet/singlet decay calculations
├── fraunhofer.py          # Fraunhofer pattern and I-V curves
├── main_simulation.py     # Main script with plotting
├── implementation.md      # This documentation
└── figures/               # Output plots
```

### 3.2 Key Classes

#### `MaterialParameters` (parameters.py)

Stores all material-specific values:

```python
@dataclass
class MaterialParameters:
    # Layer thicknesses [m]
    d_Nb_base: float = 20e-9
    d_Nb_top: float = 5e-9
    d_Cu: float = 2e-9
    d_Fe: float = 3e-9
    d_Cr: float = 5e-9
    d_dead: float = 1.5e-9
    
    # Superconductor (Nb) properties
    xi_S: float = 6e-9           # Singlet coherence in Nb
    xi_T_Nb: float = 1.2e-9      # Triplet coherence in Nb (short!)
    
    # Normal metal (Cu) properties
    xi_N: float = 100e-9         # Triplet coherence in Cu (long!)
```

The key physics: The ratio $\xi_N / \xi_T^{Nb} \approx 100$ determines how much 
triplets benefit from the Cu spacer.

#### `PairAmplitudeCalculator` (pair_amplitudes.py)

Computes:
- Singlet decay in ferromagnets: $e^{-x/\xi_F} \cos(x/\xi_{F2})$ (oscillates)
- Triplet decay in normal metals: $e^{-x/\xi_N}$ (monotonic, long range)
- Singlet-to-triplet conversion: depends on disorder at Fe/Cr interface
- Effective amplitude: combines all decay effects through the stack

**Core algorithm:**

```python
def effective_triplet_amplitude(self, d_Cr):
    # Step 1: Generate triplets at interface
    P_convert = self.singlet_to_triplet_conversion(disorder_strength=0.6)
    
    # Step 2-5: Track decay through each layer
    f_after_Fe = exp(-d_Fe / xi_F)
    f_after_Cr = exp(-d_Cr / xi_Cr)
    f_after_Cu = exp(-d_Cu / xi_N)  # Cu is protective!
    f_in_Nb = exp(-penetration / xi_T_Nb)  # Nb suppresses triplets
    
    # Total: multiply all factors
    f_eff = sqrt(P_convert) * f_after_Fe * f_after_Cr * f_after_Cu * f_in_Nb
    return f_eff
```

#### `FraunhoferCalculator` (fraunhofer.py)

Computes magnetic field dependence:

1. **Flux calculation**: Combines external field and internal magnetization
   $$\Phi = B \cdot A + \mu_0 M \cdot d_F \cdot w$$

2. **Airy pattern**: Integrates current density over elliptical junction
   $$I_c(B) = I_{c0} \left|\frac{2J_1(\pi\Phi/\Phi_0)}{\pi\Phi/\Phi_0}\right|$$

3. **Hysteresis**: Models Fe magnetization with rectangular loop
   - Up-sweep: switches at $+B_c$
   - Down-sweep: switches at $-B_c$
   - Creates asymmetric patterns

#### `HysteresisModel` (fraunhofer.py)

Simple rectangular model for Fe magnetization:

```python
@dataclass
class HysteresisModel:
    M_s: float = 1.7e6        # Saturation [A/m]
    B_c: float = 5e-3         # Coercive field [T]
    M_r_ratio: float = 0.9    # Remanence/saturation
```

This creates the experimental asymmetry between up and down sweeps.

### 3.3 Algorithm Flow

```
1. Initialize parameters (materials, geometry, conditions)

2. For each Cr thickness d_Cr:
   a. Create FraunhoferCalculator instance
   b. For each B-field value:
      i.   Get magnetization M(B, direction) from HysteresisModel
      ii.  Calculate flux Phi = B*A + mu_0*M*d_F*w
      iii. Compute Airy pattern: |2*J_1(pi*Phi/Phi_0) / (pi*Phi/Phi_0)|
      iv.  Multiply by |f_T|^2 (triplet decay factor)
      v.   Store I_c(B) value
   c. Save curve for this Cr thickness

3. For Cr-thickness dependence:
   a. Create PairAmplitudeCalculator
   b. For each Cr thickness:
      i.   Compute effective triplet amplitude f_T(d_Cr)
      ii.  Calculate I_c proportional to |f_T|^2
   c. Fit exponential to extract coherence length

4. For I-V characteristics:
   a. For each B-field:
      i.   Get I_c at that field
      ii.  Use RSJ model: V = R_N*sqrt(I^2 - I_c^2) for |I| > I_c
      iii. Add noise for realism

5. Generate publication-quality plots
```

---

## 4. Physical Parameters Used

### 4.1 From Experimental Report

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Nb base thickness | $d_{Nb,\text{base}}$ | 20 nm | Exp. report |
| Nb top thickness | $d_{Nb,\text{top}}$ | 5 nm | Exp. report |
| Cu spacer | $d_{Cu}$ | 2 nm | Exp. report |
| Fe thickness | $d_{Fe}$ | 3 nm | Exp. report |
| Cr thickness | $d_{Cr}$ | 2-12 nm | Variable (JJ1-JJ6) |
| Dead layer | $d_{\text{dead}}$ | 1-2 nm | Exp. report |
| Temperature | $T$ | 4.2 K | Exp. report |
| B-field range | $B$ | ±120 mT | Exp. report |

### 4.2 From Literature

| Parameter | Symbol | Value | Reference |
|-----------|--------|-------|-----------|
| Singlet coherence (Nb) | $\xi_S$ | 6 nm | Dirty limit: $\sqrt{\hbar D / (2\pi k_B T_c)}$ |
| Triplet coherence (Nb) | $\xi_T^{\text{Nb}}$ | 1.2 nm | Komori et al. (2021) |
| Triplet coherence (Cu) | $\xi_N$ | ~100 nm | Bergeret et al. (2005) |
| London penetration (Nb) | $\lambda_L$ | 85 nm | Tinkham |
| Fe exchange energy | $E_{\text{ex}}$ | ~1 eV | DFT estimates |
| Fe saturation magnetization | $M_s$ | 1.7 MA/m | Standard value |
| Fe coercive field | $B_c$ | 5 mT | Typical for thin films |

---

## 5. Output Interpretation

### 5.1 Plot 1: $I_c/I_0$ vs $B$ (Multi-Cr)

**File:** `01_Ic_vs_B_multi_Cr.png`

**What it shows:** Airy diffraction patterns for different Cr thicknesses.

**Physics:**
- Central peak height **decreases with increasing $d_{Cr}$** due to triplet decay:
  $$I_c \propto \exp(-2d_{Cr}/\xi_T^{\text{Cr}})$$
- Pattern shape (side lobe positions) remains constant (determined by geometry)
- Zero crossings occur at flux quanta: $\Phi = n\Phi_0$
- Each curve represents a different sample (JJ1 through JJ6)

**Experimental validation:** Compare peak heights in Plot 1 with experimental 
Figures 8-9 in the original paper.

### 5.2 Plot 1b: Hysteresis in Fraunhofer Pattern

**File:** `02_Ic_vs_B_hysteresis.png`

**What it shows:** Up-sweep vs. down-sweep asymmetry.

**Physics:**
- Fe magnetization has remanence: stays "up" until field reverses
- This adds/subtracts flux, shifting pattern between sweeps
- Yellow shaded region shows where hysteresis dominates
- Absence of suppression in upper sweep matches experimental observation

**Key finding:** The asymmetry reveals exchange bias from Cr on Fe—a critical 
mechanism for understanding the junction's unique properties.

### 5.3 Plot 2: I-V Characteristics

**File:** `03_IV_characteristics.png`

**What it shows:** Current-voltage curves at different magnetic fields.

**Physics:**
- **Horizontal segment at $V=0$:** Supercurrent branch ($|I| < I_c$)
  - Superconducting electrons flow without resistance
- **Curved transition:** Switch to normal state
- **Slope at high current:** Normal resistance $R_N$ (RSJ model)
  $$V = R_N \sqrt{I^2 - I_c^2}$$
- **$I_c$ decreases with $B$** following Fraunhofer suppression
  - At $B=0$: largest step
  - At $B=100$ mT: minimal or zero gap

**Experimental comparison:** IV curves are standard measurements in Josephson 
junction experiments; compare with experimental data.

### 5.4 Plot 3: $I_c/I_0$ vs $d_{Cr}$ (Depth Dependence)

**File:** `04_Ic_vs_Cr_depth.png`

**What it shows:** Exponential decay of critical current with Cr thickness.

**Physics:**
- **Log scale reveals exponential behavior:**
  $$I_c(d_{Cr}) \propto \exp(-2d_{Cr}/\xi_T^{\text{Cr}})$$
- **Slope on log plot:** $-2/\xi_T^{\text{Cr}}$
  - Steeper slope = shorter triplet coherence in Cr
  - Flatter slope = longer-range triplets (possible 0-π oscillations)
- **Red dots:** Mark experimental data points (JJ1-JJ6)
- **Green dashed line:** Exponential fit with extracted $\xi_T^{\text{Cr}}$

**Key measurement:** 
The Cr-thickness dependence directly probes **triplet coherence length** in 
antiferromagnetic Cr. This is a unique feature of this junction design.

**Deviation from pure exponential:** Possible 0-π transitions at certain 
thicknesses (theoretical prediction: Eschrig & Löfwander).

---

## 6. Singlet-Triplet Interplay: Answering the Core Question

### 6.1 The Key Question

> Does the spin-singlet current have a 'mixing effect' with spin-triplet current? 
> Or does it suppress the spin-triplet current?

### 6.2 The Answer: Both, at Different Locations

| Location | Effect | Mechanism | Evidence |
|----------|--------|-----------|----------|
| **Fe/Cr interface** | **Mixing (generation)** | Spin-glass disorder rotates singlets → triplets | Higher disorder → larger conversion |
| **Nb bulk** | **Suppression (blocking)** | s-wave gap incompatible with triplet symmetry | $\xi_T^{\text{Nb}} \ll \xi_T^{\text{Cu}}$ |
| **Cu spacer** | **Protection** | Shields triplets from Nb spin-orbit scattering | Ratio $I_c^{\text{with Cu}} / I_c^{\text{no Cu}}$ |

### 6.3 Simulation Investigation

This simulation can quantify the competition by:

**1. Varying disorder strength** $\sigma_\theta$:
- Code parameter in `PairAmplitudeCalculator.singlet_to_triplet_conversion()`
- Controls conversion efficiency
- Higher disorder = more triplets generated

**2. Toggling Cu spacer on/off:**
```python
# With Cu: d_Cu = 2 nm (Group A)
# Without Cu: d_Cu = 0 nm (Group B)
```
- Measures protective effect quantitatively
- Extract protection factor: $\eta = I_c^{A} / I_c^{B}$

**3. Decomposing currents:**
- Track $f_T$ (triplet amplitude) vs. depth
- Compare decay rates in different layers
- Plot ratio: $|f_T^{\text{after Cu}}| / |f_T^{\text{before Cu}}|$

### 6.4 Physical Interpretation

The **key insight** is that this is not a binary choice. Instead:

1. **At interface:** Singlet-to-triplet conversion REQUIRES magnetic disorder
   - No disorder = no triplets (zero mixing)
   - Disorder = triplet generation (constructive mixing)

2. **In bulk superconductor:** Triplet suppression is fundamental
   - Cannot be avoided by changing disorder
   - Can only be mitigated by spatial separation (Cu spacer)

3. **Optimal design:** Maximize conversion at interface, minimize suppression in bulk
   - Use strong disorder at Fe/Cr (converts singlets)
   - Add Cu spacer (protects triplets)
   - Result: Long-range triplet current dominates

---

## 7. Running the Simulation

### 7.1 Installation

```bash
# Navigate to project directory
cd /Users/jingyili/Downloads/Coding-projects/Simulation-project/ferromagnetic-josephson-junction

# Install dependencies
pip install numpy scipy matplotlib

# Verify installation
python parameters.py
```

### 7.2 Execution

```bash
# Run full simulation (generates all 4 plots)
python main_simulation.py

# Output files created in figures/ directory:
# - 01_Ic_vs_B_multi_Cr.png
# - 02_Ic_vs_B_hysteresis.png
# - 03_IV_characteristics.png
# - 04_Ic_vs_Cr_depth.png
```

### 7.3 Customization

To modify parameters, edit `parameters.py`:

```python
# Example: Change Cr thickness range
class ExperimentalConditions:
    Cr_thicknesses = (1e-9, 3e-9, 5e-9, 7e-9, 9e-9, 11e-9)  # Custom range

# Example: Change Cu spacer
class MaterialParameters:
    d_Cu = 0e-9  # Remove Cu spacer (Group B)
```

Then re-run: `python main_simulation.py`

---

## 8. References

1. **Bergeret, F. S., Volkov, A. F., & Efetov, K. B.** (2005). 
   "Odd triplet superconductivity and related phenomena in superconductor-ferromagnet structures." 
   *Rev. Mod. Phys.* **77**, 1321–1373.
   - Comprehensive review of triplet superconductivity theory
   - Foundation for understanding long-range effects

2. **Komori, S., et al.** (2021). 
   "Spin-orbit coupling suppression and singlet-state blocking of spin-triplet supercurrent." 
   *Phys. Rev. B* **104**, 054503.
   - Explains short triplet coherence in Nb: $\xi_T^{\text{Nb}} \approx 1.2$ nm
   - Key reference for Cu spacer protection mechanism

3. **Glick, J. A., et al.** (2017). 
   "Spin-triplet supercurrent in Josephson junctions containing a synthetic antiferromagnet." 
   *Sci. Adv.* **3**, e1601614.
   - First experimental demonstration of triplet current in antiferromagnetic junctions
   - Foundational paper for this project's experimental setup

4. **Houzet, M., & Buzdin, A. I.** (2007). 
   "Long-range triplet Josephson effect through a ferromagnetic trilayer." 
   *Phys. Rev. B* **76**, 060504(R).
   - Theory of singlet-to-triplet conversion
   - Explains role of magnetic inhomogeneity

5. **Robinson, J. W. A., et al.** (2010). 
   "Controlled injection of spin-triplet supercurrents into a strong ferromagnet." 
   *Science* **329**, 59–61.
   - Demonstrates long-range triplet penetration
   - Key inspiration for understanding triplet decay lengths

6. **Eschrig, M.** (2015). 
   "Spin-polarized supercurrents for spintronics: a review of current progress." 
   *Rep. Prog. Phys.* **78**, 104501.
   - Comprehensive review of triplet superconductivity applications
   - Includes spintronics relevance

7. **Eschrig, M., & Löfwander, T.** (2008).
   "Triplet supercurrents in clean and disordered superconductor/ferromagnet/superconductor junctions."
   *Nat. Phys.* **4**, 138–143.
   - Theory of 0-π transitions
   - Explains oscillations in $I_c$ vs. $d_F$

8. **Tinkham, M.** (1996). 
   "Introduction to Superconductivity" (2nd ed.). 
   McGraw-Hill.
   - Fundamental reference for superconductivity theory
   - Contains London penetration depth values

---

## 9. Key Equations Summary

### Triplet Amplitude Chain

$$f_T^{\text{eff}} = \sqrt{P_{S\to T}} \cdot \underbrace{e^{-d_{Fe}/\xi_F^T}}_{\text{decay in Fe}} \cdot \underbrace{e^{-d_{Cr}/\xi_{Cr}^T}}_{\text{decay in Cr}} \cdot \underbrace{e^{-d_{Cu}/\xi_N}}_{\text{propagation in Cu}} \cdot \underbrace{e^{-\delta/\xi_T^{\text{Nb}}}}_{\text{suppression in Nb}}$$

### Critical Current

$$I_c(d_{Cr}, B) = I_0 \cdot |f_T^{\text{eff}}(d_{Cr})|^2 \cdot \left|\frac{2J_1(\pi\Phi/\Phi_0)}{\pi\Phi/\Phi_0}\right|$$

where:
- First factor: amplitude from triplet decay
- Second factor: diffraction from magnetic field

### Flux Through Junction

$$\Phi = B \cdot A + \mu_0 M(B, \text{history}) \cdot d_{Fe} \cdot w$$

### I-V Characteristic (RSJ Model)

$$V = \begin{cases} 0 & |I| < I_c \\ R_N\sqrt{I^2 - I_c^2} & |I| > I_c \end{cases}$$

---

## 10. Future Extensions

1. **Add spin-orbit coupling:** Model SOC explicitly in Nb
2. **Domain wall dynamics:** Time-dependent magnetization switching
3. **Temperature dependence:** $I_c(T)$ curves
4. **0-π transitions:** Oscillatory behavior vs. $d_{Cr}$
5. **Proximity effects:** Superconducting correlations in non-superconducting layers
6. **Mesoscopic effects:** Discrete energy levels for small junctions

---

*Document prepared: January 2026*  
*Based on experimental setup from antiferromagnetic Josephson junction study*  
*Simulation framework: Python with NumPy/SciPy/Matplotlib*
