"""
Main simulation script: Compare Usadel solver with phenomenological model and experiment
==========================================================================================

This script:
1. Loads 3B_P1 experimental data
2. Runs phenomenological (corrected) simulation
3. Runs Usadel solver simulation
4. Generates comparison plots

Usage:
------
    python run_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import j1
from scipy import optimize, special
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from usadel_wrapper import (
    UsadelSimulator,
    JunctionParams,
    load_experimental_data,
    fit_airy_pattern,
    HAS_CPP,
)

# Physical constants
PHI_0 = 2.067833848e-15  # Wb


# ============================================================================
# Experimental Data Loading (3B_P1)
# ============================================================================

def load_3B_P1_data():
    """Load 3B_P1 experimental data."""
    data_dir = Path(__file__).parent.parent / "H vs Ic"
    
    exp_data = {}
    
    # Switch fields from notebook
    downswitch = -89.581 * 1e-4  # Oe to T
    upswitch = 141.434 * 1e-4    # Oe to T
    
    for name, filename in [
        ('upsweep', '3B_P1_Upsweep_1100B'),
        ('downsweep', '3B_P1_Downsweep_1100B')
    ]:
        filepath = data_dir / filename
        if filepath.exists():
            raw = np.loadtxt(filepath)
            exp_data[name] = {
                'H': raw[:, 0] * 1e-4,    # Column 0: H (Oe -> T)
                'Ic': raw[:, 2] * 1e-3,   # Column 2: Ic (mA -> A)
            }
            print(f"Loaded {name}: {len(raw)} points")
            print(f"  H range: [{raw[:, 0].min():.1f}, {raw[:, 0].max():.1f}] Oe")
            print(f"  Ic range: [{raw[:, 2].min():.4f}, {raw[:, 2].max():.4f}] mA")
    
    return exp_data, {'downswitch': downswitch, 'upswitch': upswitch}


# ============================================================================
# Phenomenological Model (from previous simulation)
# ============================================================================

class PhenomenologicalModel:
    """
    Phenomenological Fraunhofer/Airy pattern with hysteresis.
    
    Based on corrected_simulation.py parameters.
    """
    
    def __init__(self):
        # Junction geometry
        self.width = 2.5e-6       # m (perpendicular to field)
        self.d_barrier = 17e-9    # m (Cu + Cr + Fe + Cr + Cu)
        self.lambda_L = 85e-9     # Nb penetration depth
        
        # =====================================================================
        # ASYMMETRIC HYSTERESIS from 3B_P1 experimental switch fields
        # Downsweep peak at H = -89.58 Oe = -8.96 mT
        # Upsweep peak at H = +141.43 Oe = +14.14 mT
        # =====================================================================
        self.H_switch_up = 14.14e-3     # T (141.43 Oe)
        self.H_switch_down = -8.96e-3   # T (-89.58 Oe)
        self.B_c = 5e-3                 # T - coercive field width
        self.M_s = 1.7e6                # Saturation magnetization [A/m]
        
        # Edge current enhancement
        self.alpha_edge = 0.20          # Boosts side-lobes
        
        # =====================================================================
        # CORRECTED GEOMETRY - tuned to match experimental pattern width
        # Experimental k: k_up ≈ 2040 T⁻¹, k_down ≈ 1674 T⁻¹
        # A_eff = k × Φ_0 / π → A_eff_up ≈ 1.34 μm², A_eff_down ≈ 1.10 μm²
        # =====================================================================
        self.f_focus = 2.5              # Flux focusing (narrow junction)
        
        # Asymmetric A_eff: upsweep has ~22% larger effective area
        self.A_eff_asymmetry = 0.22     # k_up/k_down ≈ 1.22
        
        # Legacy parameters (for backward compatibility)
        self.H_c = 30e-3
        self.H_exchange_bias = 15e-3
        
    @property
    def d_eff(self):
        """Effective magnetic thickness."""
        return self.d_barrier + 2 * self.lambda_L
    
    @property
    def A_eff(self):
        """Base effective area for flux (with focusing)."""
        return self.width * self.d_eff * self.f_focus
    
    def A_eff_sweep(self, sweep_direction):
        """Get sweep-dependent effective area."""
        if sweep_direction == 'up':
            return self.A_eff * (1 + self.A_eff_asymmetry / 2)
        else:
            return self.A_eff * (1 - self.A_eff_asymmetry / 2)
    
    def airy_pattern(self, phi_normalized):
        """Airy pattern |2*J_1(πφ)/(πφ)| with edge enhancement"""
        if np.abs(phi_normalized) < 1e-10:
            base = 1.0
        else:
            arg = np.pi * phi_normalized
            base = np.abs(2 * j1(arg) / arg)
        
        # Edge enhancement: boosts side-lobes
        return base * (1 + self.alpha_edge * (1 - base))
    
    def magnetization(self, B, sweep_direction='up'):
        """Fe magnetization with asymmetric switch fields and smooth tanh."""
        # Smooth tanh hysteresis with asymmetric switch points
        if sweep_direction == 'up':
            # Upsweep: transition at H_switch_up
            M_norm = np.tanh((B - self.H_switch_up) / (self.B_c / 3))
        else:
            # Downsweep: transition at H_switch_down  
            M_norm = -np.tanh((self.H_switch_down - B) / (self.B_c / 3))
        
        return M_norm * self.M_s
    
    def Ic_normalized(self, B, sweep_direction='up'):
        """Compute normalized Ic at field B."""
        # Effective field including magnetization
        M = self.magnetization(B, sweep_direction)
        
        # Magnetization contribution to flux
        MU_0 = 1.25663706212e-6
        f_geometry = self.d_barrier / (2 * self.d_eff)
        B_from_M = MU_0 * M * f_geometry
        
        B_eff = B + B_from_M
        
        # Use sweep-dependent A_eff
        A_eff_current = self.A_eff_sweep(sweep_direction)
        
        # Flux normalized to flux quantum
        phi = B_eff * A_eff_current / PHI_0
        
        return self.airy_pattern(phi)
    
    def compute_Ic_vs_B(self, B_range, sweep_direction='up'):
        """Compute Ic(B) curve."""
        return np.array([self.Ic_normalized(B, sweep_direction) for B in B_range])


# ============================================================================
# Fitting Functions
# ============================================================================

def airy_ic_fit(H, k, m, ic):
    """
    Airy pattern for fitting.
    
    Ic(H) = 2 * Ic0 * |J_1(k*(H + m)) / (k*(H + m))|
    
    Parameters:
    - k: scales the field axis (related to effective area)
    - m: field offset (from magnetization/hysteresis)
    - ic: critical current amplitude
    """
    arg = k * (H + m)
    return 2 * ic * np.where(
        np.abs(arg) < 1e-10,
        0.5,
        np.abs(special.jv(1, arg) / arg)
    )


def fit_experimental_data(exp_data, switches):
    """Fit Airy pattern to experimental data."""
    fits = {}
    
    for name, data in exp_data.items():
        H = data['H']
        Ic = data['Ic']
        
        # Apply switch field mask
        if name == 'downsweep':
            mask = H > switches['downswitch']
        else:
            mask = H < switches['upswitch']
        
        H_fit = H[mask]
        Ic_fit = Ic[mask]
        
        # Initial guess [k, m, Ic0]
        # k ~ π / (field at first zero) ~ π / 0.04 ~ 80 for our geometry
        # But H is in Tesla, so adjust
        p0 = [-3000, 0.005, Ic_fit.max() / 2]  # k in 1/T
        
        try:
            popt, pcov = optimize.curve_fit(
                airy_ic_fit, H_fit, Ic_fit, 
                p0=p0, maxfev=10000
            )
            fits[name] = {
                'params': popt,
                'cov': pcov,
                'H_fit': H_fit,
                'Ic_fit': Ic_fit,
            }
            print(f"\n{name} fit:")
            print(f"  k = {popt[0]:.2f} 1/T")
            print(f"  m = {popt[1]*1e3:.2f} mT")
            print(f"  Ic0 = {popt[2]*1e3:.4f} mA")
        except Exception as e:
            print(f"Fitting {name} failed: {e}")
            fits[name] = None
    
    return fits


# ============================================================================
# Main Comparison Script
# ============================================================================

def run_comparison():
    """Run full comparison: experiment vs phenomenological vs Usadel."""
    
    print("="*70)
    print("JOSEPHSON JUNCTION SIMULATION COMPARISON")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # ----- 1. Load experimental data -----
    print("\n[1] Loading 3B_P1 experimental data...")
    try:
        exp_data, switches = load_3B_P1_data()
    except Exception as e:
        print(f"Failed to load experimental data: {e}")
        exp_data = None
    
    # ----- 2. Fit experimental data -----
    if exp_data:
        print("\n[2] Fitting Airy pattern to experimental data...")
        fits = fit_experimental_data(exp_data, switches)
    
    # ----- 3. Phenomenological model -----
    print("\n[3] Running phenomenological simulation...")
    phenom = PhenomenologicalModel()
    
    B_range = np.linspace(-0.12, 0.12, 500)  # ±120 mT
    
    Ic_phenom_up = phenom.compute_Ic_vs_B(B_range, 'up')
    Ic_phenom_down = phenom.compute_Ic_vs_B(B_range, 'down')
    
    # ----- 4. Usadel solver (if C++ available) -----
    if HAS_CPP:
        print("\n[4] Running semi-analytical S/F/S model...")
        
        # Create junction with parameters tuned to match experiment
        junction = JunctionParams.default_NbCuCrFeCrCuNb(d_Cr=5e-9)
        
        # Tune junction width to match experimental pattern width
        # From fits: k ≈ 1800 1/T → pattern width ~ 2*3.83/1800 = 4.3 mT
        # A_eff = Φ₀ / (k * pattern_halfwidth) ≈ 2.07e-15 / (1800 * 2e-3) ≈ 0.5 μm²
        # If d_eff ≈ 200 nm, then width ≈ 2.5 μm
        junction.width = 2.5e-6  # 2.5 μm
        
        sim = UsadelSimulator(junction)
        
        # Compute Ic at B=0 to check magnitude
        B_test = np.array([0.0])
        Ic_test = sim.compute_Ic_vs_B(B_test, 'up')[0]
        print(f"  Analytical Ic(B=0): {Ic_test*1e6:.2f} μA")
        
        if exp_data:
            exp_Ic_max = max(
                exp_data['upsweep']['Ic'].max(),
                exp_data['downsweep']['Ic'].max()
            )
            print(f"  Experimental Ic_max: {exp_Ic_max*1e6:.2f} μA")
        
        # Use coarser B grid for speed
        B_usadel = np.linspace(-0.12, 0.12, 100)
        
        try:
            Ic_usadel_up = sim.compute_Ic_vs_B(B_usadel, 'up')
            Ic_usadel_down = sim.compute_Ic_vs_B(B_usadel, 'down')
            has_usadel = True
            
            print(f"  Ic range: [{Ic_usadel_up.min()*1e6:.2f}, {Ic_usadel_up.max()*1e6:.2f}] μA")
        except Exception as e:
            print(f"Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            has_usadel = False
    else:
        print("\n[4] Skipping Usadel solver (C++ module not built)")
        has_usadel = False
    
    # ----- 5. Generate comparison plots -----
    print("\n[5] Generating comparison plots...")
    
    # Plot 1: Experimental data with fits
    if exp_data:
        fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, (name, data) in zip([ax1a, ax1b], exp_data.items()):
            H_mT = data['H'] * 1e3  # T to mT
            Ic_mA = data['Ic'] * 1e3  # A to mA
            
            ax.scatter(H_mT, Ic_mA, s=10, alpha=0.6, label='Data')
            
            if fits.get(name):
                H_fine = np.linspace(data['H'].min(), data['H'].max(), 500)
                Ic_fit = airy_ic_fit(H_fine, *fits[name]['params'])
                ax.plot(H_fine * 1e3, Ic_fit * 1e3, 'r-', lw=2, 
                       label=f"Airy fit\nk={fits[name]['params'][0]:.1f}")
            
            ax.set_xlabel('H (mT)', fontsize=12)
            ax.set_ylabel('$I_c$ (mA)', fontsize=12)
            ax.set_title(f'3B_P1 {name.capitalize()}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig1.tight_layout()
        fig1.savefig(output_dir / '12_3B_P1_experimental_fits.png', dpi=150)
        print(f"  Saved: {output_dir / '12_3B_P1_experimental_fits.png'}")
    
    # Plot 2: Phenomenological model
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(B_range * 1e3, Ic_phenom_up, 'b-', lw=2, label='Phenom: Upsweep')
    ax2.plot(B_range * 1e3, Ic_phenom_down, 'r-', lw=2, label='Phenom: Downsweep')
    
    ax2.set_xlabel('B (mT)', fontsize=12)
    ax2.set_ylabel('$I_c / I_{c0}$', fontsize=12)
    ax2.set_title('Phenomenological Simulation (Airy + Hysteresis)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-120, 120])
    
    fig2.tight_layout()
    fig2.savefig(output_dir / '13_phenomenological_simulation.png', dpi=150)
    print(f"  Saved: {output_dir / '13_phenomenological_simulation.png'}")
    
    # Plot 3: Full comparison
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    # Experimental data (normalized)
    if exp_data:
        for name, data in exp_data.items():
            H_mT = data['H'] * 1e3
            Ic_norm = data['Ic'] / data['Ic'].max()
            color = 'blue' if 'up' in name else 'red'
            ax3.scatter(H_mT, Ic_norm, s=10, alpha=0.4, 
                       color=color, label=f'Exp: {name}')
    
    # Phenomenological (already normalized)
    ax3.plot(B_range * 1e3, Ic_phenom_up, 'b-', lw=2, 
            label='Phenom: Upsweep', alpha=0.8)
    ax3.plot(B_range * 1e3, Ic_phenom_down, 'r-', lw=2, 
            label='Phenom: Downsweep', alpha=0.8)
    
    # Usadel (if available)
    if has_usadel:
        Ic_usadel_up_norm = Ic_usadel_up / Ic_usadel_up.max()
        Ic_usadel_down_norm = Ic_usadel_down / Ic_usadel_down.max()
        
        ax3.plot(B_usadel * 1e3, Ic_usadel_up_norm, 'b--', lw=2,
                label='Usadel: Upsweep')
        ax3.plot(B_usadel * 1e3, Ic_usadel_down_norm, 'r--', lw=2,
                label='Usadel: Downsweep')
    
    ax3.set_xlabel('B (mT)', fontsize=12)
    ax3.set_ylabel('$I_c / I_{c,max}$', fontsize=12)
    ax3.set_title('Comparison: Experiment vs Phenomenological vs Usadel', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-120, 120])
    
    fig3.tight_layout()
    fig3.savefig(output_dir / '14_full_comparison.png', dpi=150)
    print(f"  Saved: {output_dir / '14_full_comparison.png'}")
    
    # Plot 3b: Absolute Ic comparison (with Usadel tuned to match)
    if has_usadel and exp_data:
        fig3b, ax3b = plt.subplots(figsize=(12, 7))
        
        # Experimental data (absolute mA)
        for name, data in exp_data.items():
            H_mT = data['H'] * 1e3
            Ic_mA = data['Ic'] * 1e3
            color = 'blue' if 'up' in name else 'red'
            ax3b.scatter(H_mT, Ic_mA, s=10, alpha=0.4, 
                        color=color, label=f'Exp: {name}')
        
        # Usadel (tuned to match)
        ax3b.plot(B_usadel * 1e3, Ic_usadel_up * 1e3, 'b-', lw=2.5,
                 label='Usadel: Upsweep')
        ax3b.plot(B_usadel * 1e3, Ic_usadel_down * 1e3, 'r-', lw=2.5,
                 label='Usadel: Downsweep')
        
        ax3b.set_xlabel('B (mT)', fontsize=12)
        ax3b.set_ylabel('$I_c$ (mA)', fontsize=12)
        ax3b.set_title('Absolute Ic: Experiment vs Usadel (Tuned)', fontsize=14)
        ax3b.legend(loc='upper right', fontsize=10)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_xlim([-120, 120])
        
        fig3b.tight_layout()
        fig3b.savefig(output_dir / '14b_absolute_comparison.png', dpi=150)
        print(f"  Saved: {output_dir / '14b_absolute_comparison.png'}")
    
    # Plot 4: Pattern characteristics comparison
    if exp_data and fits:
        fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # (a) Peak positions
        ax = axes[0]
        labels = []
        peak_positions = []
        
        for name, data in exp_data.items():
            peak_idx = np.argmax(data['Ic'])
            peak_H = data['H'][peak_idx] * 1e3
            labels.append(f'Exp\n{name}')
            peak_positions.append(peak_H)
        
        # Phenomenological peaks
        up_peak = B_range[np.argmax(Ic_phenom_up)] * 1e3
        down_peak = B_range[np.argmax(Ic_phenom_down)] * 1e3
        labels.extend(['Phenom\nup', 'Phenom\ndown'])
        peak_positions.extend([up_peak, down_peak])
        
        colors = ['blue', 'red', 'lightblue', 'lightcoral']
        ax.bar(range(len(labels)), peak_positions, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Peak H (mT)')
        ax.set_title('Peak Position Comparison')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        
        # (b) Pattern width
        ax = axes[1]
        
        # Estimate FWHM from experimental fits
        widths = []
        labels = []
        if fits.get('upsweep'):
            k = fits['upsweep']['params'][0]
            width = 2 * 3.83 / abs(k) * 1e3  # First zero to first zero
            widths.append(width)
            labels.append('Exp up')
        if fits.get('downsweep'):
            k = fits['downsweep']['params'][0]
            width = 2 * 3.83 / abs(k) * 1e3
            widths.append(width)
            labels.append('Exp down')
        
        # Phenomenological width
        phenom_first_zero = phenom.A_eff * 1.22 / PHI_0  # First zero flux
        phenom_width = 2 * 1.22 * PHI_0 / phenom.A_eff * 1e3
        widths.append(phenom_width)
        labels.append('Phenom')
        
        ax.bar(range(len(labels)), widths, color=['blue', 'red', 'green'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Pattern Width (mT)')
        ax.set_title('First Zero to Zero Width')
        
        # (c) Hysteresis shift
        ax = axes[2]
        
        if 'upsweep' in exp_data and 'downsweep' in exp_data:
            up_peak_H = exp_data['upsweep']['H'][np.argmax(exp_data['upsweep']['Ic'])]
            down_peak_H = exp_data['downsweep']['H'][np.argmax(exp_data['downsweep']['Ic'])]
            exp_shift = abs(up_peak_H - down_peak_H) * 1e3
        else:
            exp_shift = 0
        
        phenom_shift = abs(up_peak - down_peak)
        
        ax.bar(['Experimental', 'Phenomenological'], [exp_shift, phenom_shift],
              color=['purple', 'green'])
        ax.set_ylabel('Hysteresis Shift (mT)')
        ax.set_title('Peak-to-Peak Hysteresis')
        
        fig4.tight_layout()
        fig4.savefig(output_dir / '15_pattern_characteristics.png', dpi=150)
        print(f"  Saved: {output_dir / '15_pattern_characteristics.png'}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    if exp_data:
        print("\nKey findings:")
        for name, fit in fits.items():
            if fit:
                k, m, ic = fit['params']
                print(f"  {name}: k={k:.1f}/T, offset={m*1e3:.1f}mT, Ic0={ic*1e3:.3f}mA")
    
    print(f"\nPhenomenological model:")
    print(f"  Effective area: {phenom.A_eff * 1e12:.4f} μm²")
    print(f"  d_eff: {phenom.d_eff * 1e9:.1f} nm")
    print(f"  Hysteresis: H_c={phenom.H_c*1e3:.0f}mT, bias={phenom.H_exchange_bias*1e3:.0f}mT")
    
    if has_usadel:
        print(f"\nUsadel solver: Successfully computed {len(B_usadel)} field points")
    else:
        print(f"\nUsadel solver: Not available (build C++ module first)")
    
    plt.show()


if __name__ == "__main__":
    run_comparison()
