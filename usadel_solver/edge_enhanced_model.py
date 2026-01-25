#!/usr/bin/env python3
"""
Edge-Enhanced Spin-Triplet Josephson Junction Model

This module implements:
1. Edge current enhancement for Airy/Fraunhofer patterns
2. Asymmetric switch fields matching experimental data
3. Smooth tanh hysteresis model
4. Flux focusing and magnetization flux contributions

Based on 3B_P1 experimental data analysis:
- Downsweep switch at H = -89.58 Oe, M↓ → M↑
- Upsweep switch at H = +141.43 Oe, M↑ → M↓

Actual Ic peak positions (from analyze_peaks.py):
- M↑ state (downsweep stable): peak at H = -80.8 Oe
- M↓ state (upsweep stable): peak at H = +131.2 Oe

Pattern center offset: ±106 Oe due to magnetization
Midpoint: +25.2 Oe (slight exchange bias offset)
"""

import numpy as np
from scipy.special import j1
from scipy.signal import find_peaks
from scipy import optimize
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants
HBAR = 1.054571817e-34  # J·s
K_B = 1.380649e-23      # J/K
E_CHARGE = 1.602176634e-19  # C
PHI_0 = 2.067833848e-15  # Wb (flux quantum)
MU_0 = 4 * np.pi * 1e-7  # H/m


class EdgeEnhancedModel:
    """
    Spin-triplet Josephson junction model with edge current enhancement
    and asymmetric hysteresis from exchange bias.
    """
    
    def __init__(
        self,
        # Junction geometry - tuned to match experimental pattern width
        # Experimental k values: k_up ≈ 2040 T⁻¹, k_down ≈ 1674 T⁻¹
        # A_eff = k × Φ₀ / π → A_eff_up ≈ 1.34 μm², A_eff_down ≈ 1.10 μm²
        width: float = 2.5e-6,           # Junction width [m]
        d_barrier: float = 17e-9,        # Physical barrier thickness [m]
        lambda_L: float = 85e-9,         # London penetration depth of Nb [m]
        
        # Superconductor parameters
        Delta_0: float = 1.5e-3,         # Gap at T=0 [meV]
        Tc: float = 9.3,                 # Critical temperature [K]
        T: float = 4.2,                  # Operating temperature [K]
        
        # Triplet transport
        D_triplet: float = 5e-4,         # Effective triplet diffusion [m²/s]
        theta_mix: float = 0.7,          # Spin-mixing angle [rad]
        R_A: float = 60e-12,             # R_N × A product [Ω·m²]
        
        # Edge enhancement
        alpha_edge: float = 0.20,        # Edge current enhancement factor
        
        # Hysteresis parameters (in Tesla)
        H_switch_up: float = 14.14e-3,   # Upsweep switch field [T] (141.43 Oe)
        H_switch_down: float = -8.96e-3, # Downsweep switch field [T] (-89.58 Oe)
        B_c: float = 5e-3,               # Coercive field width [T]
        M_r_ratio: float = 0.95,         # Remanence ratio M_r/M_s
        
        # Flux focusing - tuned to match experimental pattern width
        # f_focus ≈ 2.5 for narrow junction with wide electrodes
        f_focus: float = 2.5,            # Flux focusing factor (base)
        
        # Asymmetric A_eff: different effective areas for up vs down sweep
        # Ratio from experimental k values: k_up/k_down ≈ 1.22
        # This asymmetry arises from remanent magnetization affecting flux
        A_eff_asymmetry: float = 0.22,   # Fractional difference (up is larger)
        
        # Magnetization flux
        M_s: float = 1.7e6,              # Saturation magnetization [A/m]
        d_Fe: float = 1.6e-9,            # Fe layer thickness [m]
        
        # Demagnetization (placeholder)
        N_demag: float = 1.0,            # Demagnetization factor
    ):
        self.width = width
        self.d_barrier = d_barrier
        self.lambda_L = lambda_L
        self.d_eff = d_barrier + 2 * lambda_L  # Effective magnetic thickness
        self.Delta_0 = Delta_0 * E_CHARGE  # Convert to Joules
        self.Tc = Tc
        self.T = T
        self.D_triplet = D_triplet
        self.theta_mix = theta_mix
        self.R_A = R_A
        self.alpha_edge = alpha_edge
        self.H_switch_up = H_switch_up
        self.H_switch_down = H_switch_down
        self.B_c = B_c
        self.M_r_ratio = M_r_ratio
        self.f_focus = f_focus
        self.A_eff_asymmetry = A_eff_asymmetry
        self.M_s = M_s
        self.d_Fe = d_Fe
        self.N_demag = N_demag
        
        # Derived quantities - base A_eff (average of up/down)
        self.A_eff_base = width * self.d_eff * f_focus
        # Asymmetric A_eff for up vs down sweeps
        self.A_eff_up = self.A_eff_base * (1 + A_eff_asymmetry / 2)
        self.A_eff_down = self.A_eff_base * (1 - A_eff_asymmetry / 2)
        self.A_junction = width * width
        self.R_N = R_A / self.A_junction
        
        # Temperature-dependent gap
        if T < Tc:
            self.Delta_T = self.Delta_0 * np.tanh(1.74 * np.sqrt(Tc/T - 1))
        else:
            self.Delta_T = 0.0
            
        # Triplet coherence length
        self.xi_T = np.sqrt(HBAR * D_triplet / (np.pi * K_B * T))
        
        # Triplet efficiency
        self.triplet_efficiency = np.sin(2 * theta_mix)
        
    def _magnetization(self, B: float, sweep_direction: str) -> float:
        """
        Smooth tanh hysteresis model with asymmetric switch fields.
        
        Args:
            B: Applied magnetic field [T]
            sweep_direction: 'up' or 'down'
            
        Returns:
            Normalized magnetization M/M_s in [-1, 1]
        """
        # Apply demagnetization correction
        B_eff = B / self.N_demag
        
        # Asymmetric switch fields from exchange bias
        if sweep_direction == 'up':
            # Upsweep: switch occurs at H_switch_up
            B_switch = self.H_switch_up
            # Before switch: M = -M_r, after switch: M = +M_s
            M = np.tanh((B_eff - B_switch) / (self.B_c / 3))
        else:
            # Downsweep: switch occurs at H_switch_down
            B_switch = self.H_switch_down
            # Before switch: M = +M_r, after switch: M = -M_s
            M = -np.tanh((B_switch - B_eff) / (self.B_c / 3))
            
        return M
    
    def _magnetization_flux(self, M: float) -> float:
        """
        Calculate flux contribution from Fe layer magnetization.
        
        Φ_M = μ_0 × M × d_Fe × w
        
        Note: This is typically small. The main pattern offset comes from
        the exchange bias causing the pattern center to shift.
        """
        return MU_0 * M * self.M_s * self.d_Fe * self.width
    
    def _get_pattern_center_field(self, sweep_direction: str) -> float:
        """
        Get the field offset where the Airy pattern is centered (Φ = 0).
        
        From experimental analysis (analyze_peaks.py):
        - M↑ state (downsweep stable): peak at H = -80.8 Oe
        - M↓ state (upsweep stable): peak at H = +131.2 Oe
        
        The pattern center IS where Ic peaks, so we directly use these values.
        Convert to Tesla: 1 Oe = 1e-4 T
        """
        # Pattern center positions from experimental data
        # These are the H values where the Airy pattern central maximum occurs
        if sweep_direction == 'up':
            # Upsweep stable region (M↓): peak at +131.2 Oe
            return 131.2e-4  # T
        else:
            # Downsweep stable region (M↑): peak at -80.8 Oe
            return -80.8e-4  # T
    
    def _airy_pattern_with_edge(self, phi: float) -> float:
        """
        Airy (circular junction) pattern with edge current enhancement.
        
        Standard Airy: |2 J_1(π Φ/Φ_0) / (π Φ/Φ_0)|
        Edge enhanced: pattern × (1 + α × (1 - pattern))
        
        This boosts side-lobes relative to central peak.
        """
        if abs(phi) < 1e-10:
            base_pattern = 1.0
        else:
            x = np.pi * phi
            base_pattern = np.abs(2 * j1(x) / x)
        
        # Edge enhancement: boosts lower values (side-lobes)
        enhanced = base_pattern * (1 + self.alpha_edge * (1 - base_pattern))
        
        return enhanced
    
    def compute_Ic0(self) -> float:
        """
        Calculate zero-field critical current from Ambegaokar-Baratoff
        with triplet suppression.
        """
        # Suppression from triplet transport through barrier
        # Use d_barrier (physical transport path), not d_eff (magnetic thickness)
        suppression = self.triplet_efficiency**2 * np.exp(-self.d_barrier / self.xi_T)
        
        # Ambegaokar-Baratoff
        Ic0_raw = (np.pi * self.Delta_T / (2 * E_CHARGE * self.R_N)) * \
                  np.tanh(self.Delta_T / (2 * K_B * self.T))
        
        return Ic0_raw * suppression
    
    def compute_Ic_vs_H(
        self,
        H_range_Oe: np.ndarray,
        sweep_direction: str
    ) -> np.ndarray:
        """
        Compute Ic(H) with edge enhancement and asymmetric hysteresis.
        
        The key insight from experimental data:
        - Switch fields (where M flips): -89.6 Oe (down), +141.4 Oe (up)
        - Peak positions (where Ic max): -80.8 Oe (M↑), +131.2 Oe (M↓)
        
        We model this by:
        1. Using pattern center from experimental peak positions
        2. Using switch fields for the hysteresis transition
        3. Applying asymmetric A_eff for correct lobe widths
        
        Args:
            H_range_Oe: Applied field in Oersted
            sweep_direction: 'up' or 'down'
            
        Returns:
            Critical current array [A]
        """
        # Convert Oe to Tesla: 1 Oe = 0.1 mT = 1e-4 T
        B_range = H_range_Oe * 1e-4
        
        Ic0 = self.compute_Ic0()
        Ic = np.zeros_like(B_range, dtype=float)
        
        # Select appropriate A_eff for sweep direction
        if sweep_direction == 'up':
            A_eff = self.A_eff_up
        else:
            A_eff = self.A_eff_down
        
        # Pattern center fields for each magnetization state
        H_center_M_up = -80.8e-4    # T, pattern center when M is UP
        H_center_M_down = 131.2e-4  # T, pattern center when M is DOWN
        
        for i, B in enumerate(B_range):
            # Get magnetization state (smooth transition)
            M_norm = self._magnetization(B, sweep_direction)
            
            # Determine pattern center based on magnetization state
            # M_norm ≈ +1 → M up state → center at H_center_M_up
            # M_norm ≈ -1 → M down state → center at H_center_M_down
            # Smooth interpolation during transition:
            M_weight = (M_norm + 1) / 2  # Maps [-1, +1] to [0, 1]
            H_center = M_weight * H_center_M_up + (1 - M_weight) * H_center_M_down
            
            # Total flux through junction
            # The pattern is centered at H_center, so:
            # Φ = (B - H_center) × A_eff
            Phi_total = (B - H_center) * A_eff
            
            # Normalized flux
            phi = Phi_total / PHI_0
            
            # Airy pattern with edge enhancement
            pattern = self._airy_pattern_with_edge(phi)
            
            Ic[i] = Ic0 * pattern
        
        return Ic


def extract_edge_params_from_data(data_dir: str = None) -> dict:
    """
    Extract edge enhancement factor and switch fields from 3B_P1 data.
    """
    if data_dir is None:
        # Default path
        data_dir = Path(__file__).parent.parent / "H vs Ic"
    else:
        data_dir = Path(data_dir)
    
    # Load experimental data
    downsweep_file = data_dir / "3B_P1_Downsweep_1100B"
    upsweep_file = data_dir / "3B_P1_upsweep_1100B"
    
    if not downsweep_file.exists() or not upsweep_file.exists():
        print(f"Warning: Data files not found in {data_dir}")
        print("Using default parameters based on previous analysis.")
        return {
            'alpha': 0.15,
            'H_switch_down_Oe': -89.581,
            'H_switch_up_Oe': 141.434,
            'Ic_max_down_A': 25.57e-6,
            'Ic_max_up_A': 25.57e-6,
            'lobe_ratio_down': 0.20,
            'lobe_ratio_up': 0.20,
        }
    
    data_down = np.loadtxt(downsweep_file)
    data_up = np.loadtxt(upsweep_file)
    
    H_down, Ic_down = data_down[:, 0], data_down[:, 2]  # Oe, mA
    H_up, Ic_up = data_up[:, 0], data_up[:, 2]
    
    # Known switch fields (where magnetization flips, not where Ic peaks)
    downswitch = -89.581  # Oe
    upswitch = 141.434    # Oe
    
    # Analyze stable regions (after switching)
    down_stable = H_down > downswitch
    H_ds = H_down[down_stable]
    Ic_ds = Ic_down[down_stable]
    
    up_stable = H_up < upswitch
    H_us = H_up[up_stable]
    Ic_us = Ic_up[up_stable]
    
    # Find the actual peak positions (where Ic is maximum)
    H_peak_down = H_ds[np.argmax(Ic_ds)]
    H_peak_up = H_us[np.argmax(Ic_us)]
    
    # Find peaks for side-lobe analysis
    peaks_down, _ = find_peaks(Ic_ds, distance=30, prominence=0.0001)
    peaks_up, _ = find_peaks(Ic_us, distance=30, prominence=0.0001)
    
    # Calculate lobe ratios
    if len(peaks_down) >= 2:
        Ic_peaks_down = Ic_ds[peaks_down]
        sorted_idx = np.argsort(Ic_peaks_down)[::-1]
        lobe_ratio_down = Ic_peaks_down[sorted_idx[1]] / Ic_peaks_down[sorted_idx[0]]
    else:
        lobe_ratio_down = 0.132  # Airy default
    
    if len(peaks_up) >= 2:
        Ic_peaks_up = Ic_us[peaks_up]
        sorted_idx = np.argsort(Ic_peaks_up)[::-1]
        lobe_ratio_up = Ic_peaks_up[sorted_idx[1]] / Ic_peaks_up[sorted_idx[0]]
    else:
        lobe_ratio_up = 0.132
    
    # Estimate edge enhancement
    airy_lobe_ratio = 0.132
    avg_lobe_ratio = (lobe_ratio_down + lobe_ratio_up) / 2
    
    if avg_lobe_ratio > airy_lobe_ratio:
        alpha = (avg_lobe_ratio / airy_lobe_ratio - 1) / (1 - airy_lobe_ratio)
        alpha = max(0, min(0.5, alpha))
    else:
        alpha = 0.15
    
    return {
        'alpha': alpha,
        'H_switch_down_Oe': downswitch,
        'H_switch_up_Oe': upswitch,
        'H_peak_down_Oe': H_peak_down,     # Actual Ic peak position for downsweep
        'H_peak_up_Oe': H_peak_up,         # Actual Ic peak position for upsweep
        'Ic_max_down_A': Ic_ds.max() * 1e-3,  # Convert mA to A
        'Ic_max_up_A': Ic_us.max() * 1e-3,
        'lobe_ratio_down': lobe_ratio_down,
        'lobe_ratio_up': lobe_ratio_up,
    }


def run_comparison(save_plots: bool = True):
    """
    Run comparison between edge-enhanced model and experimental data.
    """
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "H vs Ic"
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Extract parameters from data
    print("=" * 70)
    print("EXTRACTING PARAMETERS FROM 3B_P1 DATA")
    print("=" * 70)
    
    params = extract_edge_params_from_data(data_dir)
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Load experimental data
    downsweep_file = data_dir / "3B_P1_Downsweep_1100B"
    upsweep_file = data_dir / "3B_P1_upsweep_1100B"
    
    if not downsweep_file.exists():
        print(f"\nError: Cannot find {downsweep_file}")
        return
    
    data_down = np.loadtxt(downsweep_file)
    data_up = np.loadtxt(upsweep_file)
    
    H_down, Ic_down = data_down[:, 0], data_down[:, 2] * 1e-3  # Oe, A
    H_up, Ic_up = data_up[:, 0], data_up[:, 2] * 1e-3
    
    # Create model with extracted parameters
    print("\n" + "=" * 70)
    print("CREATING EDGE-ENHANCED MODEL")
    print("=" * 70)
    
    # Scale Ic0 to match experimental maximum
    Ic_exp_max = max(Ic_down.max(), Ic_up.max())
    
    model = EdgeEnhancedModel(
        alpha_edge=params['alpha'],
        H_switch_up=params['H_switch_up_Oe'] * 1e-4,    # Convert Oe to T
        H_switch_down=params['H_switch_down_Oe'] * 1e-4,
        B_c=5e-3,           # 50 Oe coercive width
        f_focus=1.5,        # Flux focusing
    )
    
    # Compute model Ic
    H_fine = np.linspace(-1100, 1100, 500)  # Oe
    
    Ic_model_down = model.compute_Ic_vs_H(H_fine, 'down')
    Ic_model_up = model.compute_Ic_vs_H(H_fine, 'up')
    
    # Scale model to match experiment
    model_max = max(Ic_model_down.max(), Ic_model_up.max())
    scale = Ic_exp_max / model_max
    Ic_model_down *= scale
    Ic_model_up *= scale
    
    print(f"  Model Ic0 (raw): {model.compute_Ic0()*1e6:.2f} μA")
    print(f"  Scale factor: {scale:.2f}")
    print(f"  Model Ic_max (scaled): {max(Ic_model_down.max(), Ic_model_up.max())*1e6:.2f} μA")
    print(f"  Experimental Ic_max: {Ic_exp_max*1e6:.2f} μA")
    print(f"  Edge enhancement α: {params['alpha']:.3f}")
    
    # Plot comparison
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Full comparison
    ax1 = axes[0, 0]
    ax1.scatter(H_down, Ic_down*1e6, s=8, alpha=0.5, color='blue', label='Downsweep Data')
    ax1.scatter(H_up, Ic_up*1e6, s=8, alpha=0.5, color='red', label='Upsweep Data')
    ax1.plot(H_fine, Ic_model_down*1e6, 'navy', lw=2, label='Model (down)')
    ax1.plot(H_fine, Ic_model_up*1e6, 'darkred', lw=2, linestyle='--', label='Model (up)')
    ax1.axvline(params['H_switch_down_Oe'], color='blue', linestyle=':', alpha=0.5)
    ax1.axvline(params['H_switch_up_Oe'], color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('H (Oe)')
    ax1.set_ylabel('Ic (μA)')
    ax1.set_title('Edge-Enhanced Model vs Experiment')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Zoom on central peaks
    ax2 = axes[0, 1]
    zoom_range = 300  # Oe
    down_zoom = (H_down > -zoom_range) & (H_down < zoom_range)
    up_zoom = (H_up > -zoom_range) & (H_up < zoom_range)
    model_zoom = (H_fine > -zoom_range) & (H_fine < zoom_range)
    
    ax2.scatter(H_down[down_zoom], Ic_down[down_zoom]*1e6, s=15, alpha=0.6, color='blue')
    ax2.scatter(H_up[up_zoom], Ic_up[up_zoom]*1e6, s=15, alpha=0.6, color='red')
    ax2.plot(H_fine[model_zoom], Ic_model_down[model_zoom]*1e6, 'navy', lw=2)
    ax2.plot(H_fine[model_zoom], Ic_model_up[model_zoom]*1e6, 'darkred', lw=2, linestyle='--')
    ax2.axvline(params['H_switch_down_Oe'], color='blue', linestyle=':', alpha=0.7,
                label=f'Down switch: {params["H_switch_down_Oe"]:.1f} Oe')
    ax2.axvline(params['H_switch_up_Oe'], color='red', linestyle=':', alpha=0.7,
                label=f'Up switch: {params["H_switch_up_Oe"]:.1f} Oe')
    ax2.set_xlabel('H (Oe)')
    ax2.set_ylabel('Ic (μA)')
    ax2.set_title('Zoom: Central Peak Region')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Hysteresis loop visualization
    ax3 = axes[1, 0]
    H_hyst = np.linspace(-200, 200, 500)
    M_up = [model._magnetization(H*1e-4, 'up') for H in H_hyst]
    M_down = [model._magnetization(H*1e-4, 'down') for H in H_hyst]
    
    ax3.plot(H_hyst, M_up, 'r-', lw=2, label='Upsweep')
    ax3.plot(H_hyst, M_down, 'b-', lw=2, label='Downsweep')
    ax3.axvline(params['H_switch_down_Oe'], color='blue', linestyle=':', alpha=0.5)
    ax3.axvline(params['H_switch_up_Oe'], color='red', linestyle=':', alpha=0.5)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('H (Oe)')
    ax3.set_ylabel('M / M_s')
    ax3.set_title('Magnetization Hysteresis Loop')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Edge enhancement effect
    ax4 = axes[1, 1]
    phi_range = np.linspace(-3, 3, 500)
    
    # Standard Airy
    airy_standard = np.array([
        1.0 if abs(p) < 1e-10 else np.abs(2 * j1(np.pi * p) / (np.pi * p))
        for p in phi_range
    ])
    
    # Edge enhanced
    airy_enhanced = airy_standard * (1 + params['alpha'] * (1 - airy_standard))
    
    ax4.plot(phi_range, airy_standard, 'k-', lw=2, label='Standard Airy')
    ax4.plot(phi_range, airy_enhanced, 'r--', lw=2, 
             label=f'Edge Enhanced (α={params["alpha"]:.2f})')
    ax4.set_xlabel('Φ / Φ₀')
    ax4.set_ylabel('Ic / Ic₀')
    ax4.set_title('Edge Current Enhancement Effect')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Add text box with parameters
    param_text = (
        f"Edge Enhancement Parameters:\n"
        f"  α = {params['alpha']:.3f}\n"
        f"  H_switch_up = {params['H_switch_up_Oe']:.1f} Oe\n"
        f"  H_switch_down = {params['H_switch_down_Oe']:.1f} Oe\n"
        f"  Lobe ratio (down) = {params['lobe_ratio_down']:.3f}\n"
        f"  Lobe ratio (up) = {params['lobe_ratio_up']:.3f}"
    )
    fig.text(0.02, 0.02, param_text, fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8),
             family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if save_plots:
        output_path = figures_dir / "16_edge_enhanced_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    
    plt.show()
    
    # Verify switch field positions
    print("\n" + "=" * 70)
    print("VERIFICATION: PEAK POSITIONS")
    print("=" * 70)
    
    # Find model peak positions
    down_peak_idx = np.argmax(Ic_model_down)
    up_peak_idx = np.argmax(Ic_model_up)
    
    print(f"  Model downsweep peak at H = {H_fine[down_peak_idx]:.1f} Oe")
    print(f"  Expected downsweep peak at H = {params['H_switch_down_Oe']:.1f} Oe")
    print(f"  Model upsweep peak at H = {H_fine[up_peak_idx]:.1f} Oe")
    print(f"  Expected upsweep peak at H = {params['H_switch_up_Oe']:.1f} Oe")
    
    return model, params


if __name__ == "__main__":
    model, params = run_comparison(save_plots=True)
