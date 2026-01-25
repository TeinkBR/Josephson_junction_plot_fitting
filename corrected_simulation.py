"""
Corrected Ferromagnetic Josephson Junction Simulation
======================================================

This simulation uses corrected parameters based on experimental data analysis.

Key corrections:
1. Proper effective area calculation (W × d_eff, not physical area)
2. Strong hysteresis with exchange bias
3. Proper normalization and scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function J_1
from pathlib import Path

from corrected_parameters import (
    get_corrected_parameters,
    CorrectedMaterialParameters,
    CorrectedJunctionGeometry,
    CorrectedHysteresisModel,
    CorrectedExperimentalConditions
)
from parameters import PHI_0, MU_0


class CorrectedFraunhoferCalculator:
    """
    Calculate Fraunhofer/Airy diffraction pattern with corrected parameters.
    
    The critical current vs magnetic field follows:
    
    For rectangular junction:
        I_c(B) = I_c0 * |sin(π Φ/Φ_0) / (π Φ/Φ_0)|
    
    For elliptical junction (Airy pattern):
        I_c(B) = I_c0 * |2 J_1(π Φ/Φ_0) / (π Φ/Φ_0)|
    
    where Φ = B × A_eff = B × W × d_eff
    """
    
    def __init__(
        self,
        geometry: CorrectedJunctionGeometry,
        hysteresis: CorrectedHysteresisModel
    ):
        self.geometry = geometry
        self.hysteresis = hysteresis
    
    def flux_normalized(self, B: float) -> float:
        r"""
        Calculate normalized flux Φ/Φ_0 = B × A_eff / Φ_0.
        """
        flux = B * self.geometry.effective_area
        return flux / PHI_0
    
    def airy_pattern(self, phi_normalized: float) -> float:
        r"""
        Airy diffraction pattern for elliptical junction.
        
        I_c/I_c0 = |2 J_1(π φ) / (π φ)|
        
        where φ = Φ/Φ_0
        """
        if np.abs(phi_normalized) < 1e-10:
            return 1.0
        
        arg = np.pi * phi_normalized
        return np.abs(2 * j1(arg) / arg)
    
    def effective_field_with_magnetization(
        self,
        B_applied: float,
        sweep_direction: str = 'up'
    ) -> float:
        r"""
        Calculate effective field including contribution from Fe magnetization.
        
        B_eff = B_applied + μ_0 * M(B) * f_geometry
        
        The magnetization contribution depends on the Fe layer geometry
        and creates the field offset seen in experiments.
        """
        M = self.hysteresis.magnetization(B_applied, sweep_direction)
        
        # Geometric factor for thin film magnetization contribution
        # This determines how much the Fe magnetization affects the junction flux
        # Tuned to match experimental peak offsets
        f_geometry = self.geometry.d_barrier / (2 * self.geometry.d_eff)
        
        B_from_M = MU_0 * M * f_geometry
        
        return B_applied + B_from_M
    
    def critical_current_normalized(
        self,
        B_applied: float,
        sweep_direction: str = 'up'
    ) -> float:
        """
        Calculate normalized critical current I_c/I_c0 at applied field B.
        """
        B_eff = self.effective_field_with_magnetization(B_applied, sweep_direction)
        phi = self.flux_normalized(B_eff)
        return self.airy_pattern(phi)
    
    def sweep_field(
        self,
        B_range: np.ndarray,
        sweep_direction: str = 'up'
    ) -> np.ndarray:
        """
        Calculate I_c/I_c0 over a range of applied fields.
        """
        return np.array([
            self.critical_current_normalized(B, sweep_direction)
            for B in B_range
        ])


def load_experimental_data():
    """
    Load experimental data for comparison.
    
    Note: Only 3A_P1 and 3A_P3 data are usable; other samples excluded.
    """
    data_dir = Path("H vs Ic")
    
    # Only load valid samples (3A_P1 and 3A_P3)
    valid_files = {
        '3A_P1_upsweep': '3A_P1_HvsIc_upsweep_1100B',
        '3A_P1_downsweep': '3A_P1_HvsIc_Downsweep_1100B',
        '3A_P3_upsweep': '3A_P3_Upsweep1100B',
        '3A_P3_downsweep': '3A_P3_Downsweep1100B',
    }
    
    exp_data = {}
    for key, filename in valid_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            # Tab-delimited, take first two columns (H in mT, Ic in mA)
            data = np.loadtxt(filepath, skiprows=1, delimiter='\t', usecols=(0, 1))
            exp_data[key] = {
                'H': data[:, 0] * 1e-3,  # mT to T
                'Ic': data[:, 1] * 1e-3   # mA to A
            }
            print(f"  Loaded: {key} ({len(data)} points)")
    
    return exp_data


def run_corrected_simulation():
    """Run the corrected simulation and generate comparison plots."""
    
    # Get corrected parameters
    materials, geometry, conditions, hysteresis = get_corrected_parameters()
    
    # Print key parameters
    print("="*60)
    print("CORRECTED SIMULATION")
    print("="*60)
    print(f"\nEffective area: {geometry.effective_area * 1e12:.4f} μm²")
    print(f"d_eff: {geometry.d_eff * 1e9:.1f} nm")
    print(f"First zero (Airy): {geometry.first_zero_field_airy() * 1e3:.1f} mT")
    print(f"Flux quantum field: {geometry.flux_quantum_field() * 1e3:.1f} mT")
    print(f"Coercive field: {hysteresis.H_c * 1e3:.1f} mT")
    print(f"Exchange bias: {hysteresis.H_exchange_bias * 1e3:.1f} mT")
    
    # Create calculator
    calculator = CorrectedFraunhoferCalculator(geometry, hysteresis)
    
    # Field sweep
    B_range = conditions.B_range
    B_mT = B_range * 1e3
    
    # Calculate up and down sweeps
    Ic_up = calculator.sweep_field(B_range, 'up')
    Ic_down = calculator.sweep_field(B_range, 'down')
    
    # Load experimental data
    try:
        exp_data = load_experimental_data()
        has_experimental = True
    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")
        has_experimental = False
    
    # Create output directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # --- PLOT 1: Corrected Fraunhofer Pattern ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(B_mT, Ic_up, 'b-', linewidth=2, label='Simulation: Upsweep')
    ax1.plot(B_mT, Ic_down, 'r-', linewidth=2, label='Simulation: Downsweep')
    
    ax1.set_xlabel('Applied Field B (mT)', fontsize=12)
    ax1.set_ylabel('$I_c / I_{c0}$', fontsize=12)
    ax1.set_title('Corrected Fraunhofer Pattern with Hysteresis', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-120, 120])
    
    fig1.tight_layout()
    fig1.savefig(figures_dir / "08_corrected_fraunhofer.png", dpi=150)
    print(f"\nSaved: {figures_dir / '08_corrected_fraunhofer.png'}")
    
    # --- PLOT 2: Comparison with Experimental Data ---
    if has_experimental:
        # Select one sample for detailed comparison
        sample_up = '3A_P1_upsweep'
        sample_down = '3A_P1_downsweep'
        
        if sample_up in exp_data and sample_down in exp_data:
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Normalize experimental data
            exp_up = exp_data[sample_up]
            exp_down = exp_data[sample_down]
            
            Ic_max = max(exp_up['Ic'].max(), exp_down['Ic'].max())
            
            # Left plot: Normalized comparison
            ax2a.plot(exp_up['H'] * 1e3, exp_up['Ic'] / Ic_max, 'b.', 
                     markersize=4, alpha=0.6, label='Exp: Upsweep')
            ax2a.plot(exp_down['H'] * 1e3, exp_down['Ic'] / Ic_max, 'r.', 
                     markersize=4, alpha=0.6, label='Exp: Downsweep')
            ax2a.plot(B_mT, Ic_up, 'b-', linewidth=2, alpha=0.8, 
                     label='Sim: Upsweep')
            ax2a.plot(B_mT, Ic_down, 'r-', linewidth=2, alpha=0.8, 
                     label='Sim: Downsweep')
            
            ax2a.set_xlabel('Applied Field B (mT)', fontsize=12)
            ax2a.set_ylabel('$I_c / I_{c,max}$', fontsize=12)
            ax2a.set_title(f'Sample 3A_P1: Simulation vs Experiment', fontsize=14)
            ax2a.legend(fontsize=10)
            ax2a.grid(True, alpha=0.3)
            ax2a.set_xlim([-120, 120])
            
            # Right plot: Pattern envelope comparison
            # Find experimental peak positions
            up_peak_idx = np.argmax(exp_up['Ic'])
            down_peak_idx = np.argmax(exp_down['Ic'])
            
            up_peak_B = exp_up['H'][up_peak_idx] * 1e3
            down_peak_B = exp_down['H'][down_peak_idx] * 1e3
            
            # Find simulation peak positions
            sim_up_peak = B_mT[np.argmax(Ic_up)]
            sim_down_peak = B_mT[np.argmax(Ic_down)]
            
            ax2b.bar(['Exp Up', 'Sim Up', 'Exp Down', 'Sim Down'],
                    [up_peak_B, sim_up_peak, down_peak_B, sim_down_peak],
                    color=['blue', 'lightblue', 'red', 'lightcoral'])
            ax2b.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2b.set_ylabel('Peak Field Position (mT)', fontsize=12)
            ax2b.set_title('Peak Position Comparison', fontsize=14)
            
            fig2.tight_layout()
            fig2.savefig(figures_dir / "09_corrected_vs_experimental.png", dpi=150)
            print(f"Saved: {figures_dir / '09_corrected_vs_experimental.png'}")
    
    # --- PLOT 3: Analysis of Pattern Width ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Pattern on log scale to see zeros
    ax3a.semilogy(B_mT, Ic_up, 'b-', linewidth=2, label='Upsweep')
    ax3a.semilogy(B_mT, Ic_down, 'r-', linewidth=2, label='Downsweep')
    
    # Mark theoretical zeros
    first_zero = geometry.first_zero_field_airy() * 1e3
    ax3a.axvline(first_zero, color='green', linestyle='--', alpha=0.7, 
                label=f'First zero: {first_zero:.1f} mT')
    ax3a.axvline(-first_zero, color='green', linestyle='--', alpha=0.7)
    
    ax3a.set_xlabel('Applied Field B (mT)', fontsize=12)
    ax3a.set_ylabel('$I_c / I_{c0}$ (log scale)', fontsize=12)
    ax3a.set_title('Pattern Structure (Log Scale)', fontsize=14)
    ax3a.legend(fontsize=11)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim([-120, 120])
    ax3a.set_ylim([1e-3, 1.5])
    
    # Right: Hysteresis loop
    B_plot = np.linspace(-0.1, 0.1, 500)
    M_up = [hysteresis.magnetization(b, 'up') for b in B_plot]
    M_down = [hysteresis.magnetization(b, 'down') for b in B_plot]
    
    ax3b.plot(B_plot * 1e3, np.array(M_up) / 1e6, 'b-', linewidth=2, label='Upsweep')
    ax3b.plot(B_plot * 1e3, np.array(M_down) / 1e6, 'r-', linewidth=2, label='Downsweep')
    ax3b.axvline(hysteresis.H_exchange_bias * 1e3, color='purple', linestyle='--',
                label=f'Exchange bias: {hysteresis.H_exchange_bias * 1e3:.0f} mT')
    ax3b.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3b.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    ax3b.set_xlabel('Applied Field B (mT)', fontsize=12)
    ax3b.set_ylabel('Magnetization M (MA/m)', fontsize=12)
    ax3b.set_title('Fe Layer Magnetization Hysteresis', fontsize=14)
    ax3b.legend(fontsize=11)
    ax3b.grid(True, alpha=0.3)
    
    fig3.tight_layout()
    fig3.savefig(figures_dir / "10_pattern_analysis.png", dpi=150)
    print(f"Saved: {figures_dir / '10_pattern_analysis.png'}")
    
    # --- PLOT 4: Multi-sample comparison (only valid samples) ---
    if has_experimental:
        fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Only use valid samples: 3A_P1 and 3A_P3
        sample_pairs = [
            ('3A_P1_upsweep', '3A_P1_downsweep', '3A P1'),
            ('3A_P3_upsweep', '3A_P3_downsweep', '3A P3'),
        ]
        
        for ax, (up_key, down_key, title) in zip(axes.flat, sample_pairs):
            if up_key in exp_data and down_key in exp_data:
                up_data = exp_data[up_key]
                down_data = exp_data[down_key]
                
                Ic_max = max(up_data['Ic'].max(), down_data['Ic'].max())
                
                # Experimental
                ax.plot(up_data['H'] * 1e3, up_data['Ic'] / Ic_max, 'b.', 
                       markersize=3, alpha=0.5, label='Exp Up')
                ax.plot(down_data['H'] * 1e3, down_data['Ic'] / Ic_max, 'r.', 
                       markersize=3, alpha=0.5, label='Exp Down')
                
                # Simulation
                ax.plot(B_mT, Ic_up, 'b-', linewidth=1.5, alpha=0.7, label='Sim Up')
                ax.plot(B_mT, Ic_down, 'r-', linewidth=1.5, alpha=0.7, label='Sim Down')
                
                ax.set_xlabel('B (mT)', fontsize=11)
                ax.set_ylabel('$I_c/I_{c,max}$', fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim([-120, 120])
        
        fig4.suptitle('Corrected Simulation vs All Experimental Samples', fontsize=14)
        fig4.tight_layout()
        fig4.savefig(figures_dir / "11_multi_sample_comparison.png", dpi=150)
        print(f"Saved: {figures_dir / '11_multi_sample_comparison.png'}")
    
    # --- Summary ---
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nKey improvements:")
    print(f"  1. Pattern first zero: {first_zero:.1f} mT (was 0.06 mT)")
    print(f"  2. Hysteresis shift: ~{2*hysteresis.H_c*1e3:.0f} mT (was ~10 mT)")
    print(f"  3. Exchange bias: {hysteresis.H_exchange_bias*1e3:.0f} mT (creates asymmetry)")
    
    if has_experimental:
        print("\nRemaining discrepancies:")
        print("  - Experimental patterns show complex structure near peaks")
        print("  - Multiple domains / non-uniform magnetization not modeled")
        print("  - Temperature-dependent effects not included")
    
    plt.show()


if __name__ == "__main__":
    run_corrected_simulation()
