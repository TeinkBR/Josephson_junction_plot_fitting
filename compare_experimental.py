"""
Experimental Data Comparison Script
====================================

This script:
1. Loads experimental H vs Ic data from the "H vs Ic" folder
2. Plots all experimental curves
3. Compares with simulation output
4. Diagnoses discrepancies

Data Format (from experimental files):
- Column 1: H field (Oersted) - needs conversion to mT (1 Oe = 0.1 mT)
- Column 2: I_c (Amperes)
- Columns 3-6: Other measurements (not used initially)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
from pathlib import Path

# Import simulation modules
from parameters import get_default_parameters, PHI_0
from fraunhofer import FraunhoferCalculator


def load_experimental_data(data_dir: str) -> dict:
    """
    Load all experimental data files from the H vs Ic directory.
    
    Parameters
    ----------
    data_dir : str
        Path to the H vs Ic directory
        
    Returns
    -------
    dict
        Dictionary with filename as key and (H, Ic) arrays as values
    """
    data = {}
    
    # Find all data files (exclude .xlsx, .ipynb, and hidden files)
    for filepath in Path(data_dir).iterdir():
        if filepath.is_file() and not filepath.name.startswith('.'):
            if filepath.suffix not in ['.xlsx', '.ipynb']:
                try:
                    # Load data - whitespace separated
                    raw = np.loadtxt(filepath)
                    
                    # Column 1: H field (Oersted)
                    # Column 2: I_c (Amperes)
                    H_Oe = raw[:, 0]
                    Ic_A = raw[:, 1]
                    
                    # Convert H from Oersted to mT (1 Oe = 0.1 mT)
                    H_mT = H_Oe * 0.1
                    
                    # Convert I_c to mA for better readability
                    Ic_mA = Ic_A * 1000
                    
                    data[filepath.name] = {
                        'H_mT': H_mT,
                        'Ic_mA': Ic_mA,
                        'H_Oe': H_Oe,
                        'Ic_A': Ic_A
                    }
                    print(f"Loaded: {filepath.name} ({len(H_Oe)} points)")
                    
                except Exception as e:
                    print(f"Could not load {filepath.name}: {e}")
    
    return data


def plot_experimental_data(data: dict, save_path: str = None):
    """
    Plot all experimental H vs Ic curves as scatter plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Separate by sample group (3A vs 3B) and sweep direction
    groups = {
        '3A_upsweep': [],
        '3A_downsweep': [],
        '3B_upsweep': [],
        '3B_downsweep': []
    }
    
    for name, d in data.items():
        name_lower = name.lower()
        if '3a' in name_lower:
            if 'up' in name_lower:
                groups['3A_upsweep'].append((name, d))
            else:
                groups['3A_downsweep'].append((name, d))
        elif '3b' in name_lower:
            if 'up' in name_lower:
                groups['3B_upsweep'].append((name, d))
            else:
                groups['3B_downsweep'].append((name, d))
    
    # Plot each group
    group_names = ['3A_upsweep', '3A_downsweep', '3B_upsweep', '3B_downsweep']
    titles = ['Group 3A - Upsweep', 'Group 3A - Downsweep', 
              'Group 3B - Upsweep', 'Group 3B - Downsweep']
    
    for ax, group_name, title in zip(axes.flat, group_names, titles):
        curves = groups[group_name]
        colors = cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
        
        for i, (name, d) in enumerate(curves):
            ax.scatter(d['H_mT'], d['Ic_mA'], s=3, alpha=0.7, 
                      color=colors[i], label=name)
        
        ax.set_xlabel('Magnetic Field H [mT]')
        ax.set_ylabel('Critical Current $I_c$ [mA]')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def plot_all_experimental_overlay(data: dict, save_path: str = None):
    """
    Plot all experimental curves overlaid on a single figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color by sweep direction
    for name, d in data.items():
        name_lower = name.lower()
        if 'up' in name_lower:
            color = 'blue'
            marker = 'o'
        else:
            color = 'red'
            marker = 's'
        
        if '3a' in name_lower:
            alpha = 0.7
        else:
            alpha = 0.4
        
        ax.scatter(d['H_mT'], d['Ic_mA'], s=5, alpha=alpha, 
                  color=color, marker=marker, label=name)
    
    ax.set_xlabel('Magnetic Field H [mT]', fontsize=14)
    ax.set_ylabel('Critical Current $I_c$ [mA]', fontsize=14)
    ax.set_title('All Experimental Data: H vs $I_c$\n(Blue=Upsweep, Red=Downsweep)', fontsize=16)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def analyze_experimental_data(data: dict):
    """
    Analyze experimental data to extract key parameters.
    """
    print("\n" + "="*60)
    print("EXPERIMENTAL DATA ANALYSIS")
    print("="*60)
    
    for name, d in data.items():
        H = d['H_mT']
        Ic = d['Ic_mA']
        
        # Find peak (maximum Ic)
        peak_idx = np.argmax(Ic)
        H_peak = H[peak_idx]
        Ic_max = Ic[peak_idx]
        
        # Find first minimum (side lobe)
        # Look for local minima
        center_region = (np.abs(H) < 50)  # Within ±50 mT of center
        
        print(f"\n{name}:")
        print(f"  H range: {H.min():.1f} to {H.max():.1f} mT")
        print(f"  Ic range: {Ic.min():.4f} to {Ic.max():.4f} mA")
        print(f"  Peak at H = {H_peak:.1f} mT, Ic = {Ic_max:.4f} mA")
        print(f"  Peak offset from zero: {H_peak:.1f} mT")


def compare_with_simulation(data: dict, save_path: str = None):
    """
    Compare experimental data with simulation results.
    
    This is the key diagnostic function!
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Load simulation parameters
    mat, geo, cond = get_default_parameters()
    
    # Pick one experimental curve for detailed comparison
    # Use 3A_P1 upsweep as reference
    exp_name = None
    for name in data.keys():
        if '3a' in name.lower() and 'p1' in name.lower() and 'up' in name.lower():
            exp_name = name
            break
    
    if exp_name is None:
        exp_name = list(data.keys())[0]
    
    exp_data = data[exp_name]
    H_exp = exp_data['H_mT']
    Ic_exp = exp_data['Ic_mA']
    
    # Normalize experimental Ic
    Ic_exp_norm = Ic_exp / Ic_exp.max()
    
    # --- Plot 1: Experimental vs Simulation Shape ---
    ax1 = axes[0, 0]
    
    # Generate simulation curve
    calc = FraunhoferCalculator(mat, geo, cond, d_Cr=5e-9)
    B_sim = np.linspace(-0.12, 0.12, 500)  # ±120 mT in Tesla
    Ic_sim = calc.critical_current_vs_B(B_sim, include_hysteresis=False)
    Ic_sim_norm = Ic_sim / Ic_sim.max()
    
    ax1.scatter(H_exp, Ic_exp_norm, s=5, alpha=0.5, color='blue', label='Experimental')
    ax1.plot(B_sim * 1000, Ic_sim_norm, 'r-', linewidth=2, label='Simulation (current)')
    ax1.set_xlabel('H [mT]')
    ax1.set_ylabel('Normalized $I_c/I_{c,max}$')
    ax1.set_title(f'Comparison: {exp_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-120, 120])
    
    # --- Plot 2: Pattern Width Analysis ---
    ax2 = axes[0, 1]
    
    # Find first zeros in experimental data
    center_idx = np.argmax(Ic_exp)
    H_center = H_exp[center_idx]
    
    # Shift to center
    H_shifted = H_exp - H_center
    
    ax2.scatter(H_shifted, Ic_exp_norm, s=5, alpha=0.5, color='blue', label='Exp (centered)')
    ax2.plot(B_sim * 1000, Ic_sim_norm, 'r-', linewidth=2, label='Simulation')
    ax2.set_xlabel('H - H_peak [mT]')
    ax2.set_ylabel('Normalized $I_c/I_{c,max}$')
    ax2.set_title('Centered Pattern Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-100, 100])
    
    # --- Plot 3: Zoom on Central Peak ---
    ax3 = axes[1, 0]
    
    # Zoom to central region
    central_mask = np.abs(H_exp) < 60
    ax3.scatter(H_exp[central_mask], Ic_exp[central_mask], s=10, alpha=0.7, 
               color='blue', label='Experimental')
    
    # Rescale simulation to match experimental amplitude
    Ic_sim_scaled = Ic_sim_norm * Ic_exp.max()
    ax3.plot(B_sim * 1000, Ic_sim_scaled, 'r-', linewidth=2, label='Simulation (scaled)')
    ax3.set_xlabel('H [mT]')
    ax3.set_ylabel('$I_c$ [mA]')
    ax3.set_title('Central Peak Zoom')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-60, 60])
    
    # --- Plot 4: Log Scale to See Side Lobes ---
    ax4 = axes[1, 1]
    
    ax4.semilogy(H_exp, Ic_exp, 'b.', markersize=3, alpha=0.5, label='Experimental')
    ax4.semilogy(B_sim * 1000, Ic_sim_scaled, 'r-', linewidth=2, label='Simulation')
    ax4.set_xlabel('H [mT]')
    ax4.set_ylabel('$I_c$ [mA] (log scale)')
    ax4.set_title('Log Scale - Side Lobe Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-120, 120])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    # --- Quantitative Analysis ---
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON")
    print("="*60)
    
    # 1. Pattern width (first zero crossing)
    # For Airy pattern: first zero at x = 3.83, so Phi = 1.22 * Phi_0
    # For Fraunhofer: first zero at Phi = Phi_0
    
    # Estimate junction area from experimental first zero
    # Find where Ic drops to ~10% of max
    threshold = 0.1 * Ic_exp.max()
    above_threshold = np.where(Ic_exp > threshold)[0]
    if len(above_threshold) > 0:
        H_width = H_exp[above_threshold[-1]] - H_exp[above_threshold[0]]
        print(f"\nExperimental pattern width (at 10% max): {H_width:.1f} mT")
        
        # For Airy pattern, first zero at ~3.83/(pi) = 1.22 flux quanta
        # Phi = B * A = 1.22 * Phi_0
        # A = 1.22 * Phi_0 / B
        B_first_zero = (H_width / 2) * 1e-3  # Convert to Tesla
        A_estimated = 1.22 * PHI_0 / B_first_zero
        print(f"Estimated junction area: {A_estimated * 1e12:.2f} um^2")
        print(f"Simulation junction area: {geo.area * 1e12:.2f} um^2")
        
    # 2. Peak position (should be at H=0 without hysteresis)
    print(f"\nExperimental peak at H = {H_exp[np.argmax(Ic_exp)]:.1f} mT")
    print(f"Simulation peak at B = 0 mT (no offset)")
    
    # 3. Ic magnitude
    print(f"\nExperimental max Ic = {Ic_exp.max():.4f} mA")
    print(f"Simulation gives normalized values")
    
    return fig


def diagnose_problems(data: dict):
    """
    Diagnose specific problems with our simulation based on experimental data.
    """
    print("\n" + "="*60)
    print("DIAGNOSIS: WHAT'S WRONG WITH THE SIMULATION")
    print("="*60)
    
    # Get experimental characteristics
    exp_name = list(data.keys())[0]
    exp_data = data[exp_name]
    H_exp = exp_data['H_mT']
    Ic_exp = exp_data['Ic_mA']
    
    # 1. Check pattern shape
    print("\n1. PATTERN SHAPE ANALYSIS:")
    
    # Calculate experimental pattern width
    peak_idx = np.argmax(Ic_exp)
    H_peak = H_exp[peak_idx]
    Ic_max = Ic_exp[peak_idx]
    
    # Find first minimum on each side
    left_half = Ic_exp[:peak_idx]
    right_half = Ic_exp[peak_idx:]
    
    # Pattern should have minima (nulls) at specific field values
    # For Airy: nulls at 3.83, 7.02, 10.17... (zeros of J_1)
    # For Fraunhofer: nulls at pi, 2pi, 3pi... (zeros of sin)
    
    print(f"   Peak at H = {H_peak:.1f} mT")
    print(f"   Peak Ic = {Ic_max:.4f} mA")
    
    # Check if experimental data shows clear side lobes
    noise_level = np.std(Ic_exp[np.abs(H_exp) > 80])
    signal_at_center = Ic_max
    snr = signal_at_center / noise_level if noise_level > 0 else np.inf
    print(f"   Signal-to-noise ratio: {snr:.1f}")
    
    # 2. Check hysteresis
    print("\n2. HYSTERESIS ANALYSIS:")
    
    # Compare upsweep vs downsweep for same sample
    up_data = None
    down_data = None
    for name, d in data.items():
        if '3a_p1' in name.lower():
            if 'up' in name.lower():
                up_data = d
            else:
                down_data = d
    
    if up_data is not None and down_data is not None:
        # Find peak positions
        H_peak_up = up_data['H_mT'][np.argmax(up_data['Ic_mA'])]
        H_peak_down = down_data['H_mT'][np.argmax(down_data['Ic_mA'])]
        shift = H_peak_up - H_peak_down
        print(f"   Upsweep peak at H = {H_peak_up:.1f} mT")
        print(f"   Downsweep peak at H = {H_peak_down:.1f} mT")
        print(f"   Hysteresis shift = {shift:.1f} mT")
    
    # 3. Identify specific problems
    print("\n3. IDENTIFIED PROBLEMS:")
    
    problems = []
    
    # Problem 1: Pattern too narrow or too wide
    # Expected first zero for elliptical junction with A = 39 um^2
    # Phi = B * A = Phi_0 at B = Phi_0/A = 2.07e-15 / 39e-12 = 53 uT = 0.053 mT
    # This seems too small! Let me recalculate...
    # Actually for first zero of Airy: B = 3.83 * Phi_0 / (pi * A)
    
    mat, geo, cond = get_default_parameters()
    B_first_zero_theory = 3.83 * PHI_0 / (np.pi * geo.area)
    print(f"\n   Theoretical first zero at B = {B_first_zero_theory * 1e3:.2f} mT")
    print(f"   (Using junction area A = {geo.area * 1e12:.2f} um^2)")
    
    # This is WAY too small compared to experimental width of ~60 mT
    # This means our junction area is WRONG!
    
    # Estimate correct area from experimental data
    # If experimental first zero is at ~30 mT:
    B_exp_zero = 30e-3  # Rough estimate from data
    A_correct = 3.83 * PHI_0 / (np.pi * B_exp_zero)
    print(f"\n   To match experimental pattern width:")
    print(f"   Estimated correct area A = {A_correct * 1e12:.4f} um^2")
    print(f"   Current simulation area A = {geo.area * 1e12:.2f} um^2")
    print(f"   --> PROBLEM: Simulation area is {geo.area / A_correct:.0f}x too large!")
    
    problems.append("Junction area in simulation is too large")
    
    # Problem 2: Missing absolute scale
    print(f"\n   Experimental Ic range: {Ic_exp.min():.4f} to {Ic_exp.max():.4f} mA")
    print(f"   Simulation gives only normalized values")
    problems.append("Simulation lacks absolute Ic scale")
    
    # Problem 3: Effective magnetic thickness
    print(f"\n   The pattern width depends on effective magnetic thickness d_eff")
    print(f"   d_eff = d_F + 2*lambda_L (London penetration depths)")
    problems.append("d_eff calculation may be incorrect")
    
    print("\n" + "="*60)
    print("SUMMARY OF REQUIRED CORRECTIONS:")
    print("="*60)
    for i, p in enumerate(problems, 1):
        print(f"   {i}. {p}")
    
    return problems


def main():
    """Main execution function."""
    print("="*60)
    print("EXPERIMENTAL DATA ANALYSIS & SIMULATION COMPARISON")
    print("="*60)
    
    # Define paths
    data_dir = "/Users/jingyili/Downloads/Coding-projects/Simulation-project/ferromagnetic-josephson-junction/H vs Ic"
    fig_dir = "/Users/jingyili/Downloads/Coding-projects/Simulation-project/ferromagnetic-josephson-junction/figures"
    
    # Create Read_data directory for organized data storage
    read_data_dir = "/Users/jingyili/Downloads/Coding-projects/Simulation-project/ferromagnetic-josephson-junction/Read_data"
    os.makedirs(read_data_dir, exist_ok=True)
    
    # Phase 1: Load experimental data
    print("\n--- PHASE 1: Loading Experimental Data ---")
    data = load_experimental_data(data_dir)
    
    if not data:
        print("No data files found!")
        return
    
    # Phase 2: Plot experimental data
    print("\n--- PHASE 2: Plotting Experimental Data ---")
    plot_experimental_data(data, save_path=os.path.join(fig_dir, "05_experimental_data_by_group.png"))
    plot_all_experimental_overlay(data, save_path=os.path.join(fig_dir, "06_experimental_data_overlay.png"))
    
    # Phase 3: Analyze experimental data
    print("\n--- PHASE 3: Analyzing Experimental Data ---")
    analyze_experimental_data(data)
    
    # Phase 4: Compare with simulation
    print("\n--- PHASE 4: Comparison with Simulation ---")
    compare_with_simulation(data, save_path=os.path.join(fig_dir, "07_simulation_vs_experimental.png"))
    
    # Phase 5: Diagnose problems
    print("\n--- PHASE 5: Diagnosis ---")
    problems = diagnose_problems(data)
    
    plt.show()
    
    return data, problems


if __name__ == "__main__":
    data, problems = main()
