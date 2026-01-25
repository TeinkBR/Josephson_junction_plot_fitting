"""
Main Simulation: Spin-Triplet Josephson Junction
=================================================

This script generates the three key plots:

1. I_c/I_0 vs B for different Cr thicknesses (Fraunhofer patterns)
2. I-V characteristics at various magnetic fields
3. I_c/I_0 vs d_Cr (critical current decay with Cr thickness)

Run this script to generate all figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from parameters import get_default_parameters, MaterialParameters, JunctionGeometry, ExperimentalConditions
from pair_amplitudes import PairAmplitudeCalculator
from fraunhofer import FraunhoferCalculator


def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_Ic_vs_B_multi_Cr(
    materials: MaterialParameters,
    geometry: JunctionGeometry,
    conditions: ExperimentalConditions,
    save_path: str = None
):
    """
    Plot 1: I_c/I_0 vs B for different Cr thicknesses.
    
    This shows how the Fraunhofer pattern evolves with Cr layer thickness.
    Thicker Cr → lower overall I_c (more decay) but same pattern shape.
    
    Physics:
    --------
    The critical current at each field is:
    I_c(B, d_Cr) = I_0 * f_triplet^2(d_Cr) * |2*J_1(pi*Phi/Phi_0) / (pi*Phi/Phi_0)|
    
    The first factor gives the overall amplitude decay with d_Cr.
    The second factor (Airy function) gives the field-dependent modulation.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    B_array = conditions.B_range
    Cr_thicknesses = conditions.Cr_thicknesses
    
    # Color map for different Cr thicknesses
    colors = cm.viridis(np.linspace(0.1, 0.9, len(Cr_thicknesses)))
    
    for i, d_Cr in enumerate(Cr_thicknesses):
        calc = FraunhoferCalculator(materials, geometry, conditions, d_Cr=d_Cr)
        
        # Up sweep
        Ic_up = calc.critical_current_vs_B(B_array, sweep_direction='up')
        
        # Plot with distinct color for each Cr thickness
        label = f'$d_{{Cr}} = {d_Cr*1e9:.0f}$ nm'
        ax.plot(B_array * 1e3, Ic_up, '-', color=colors[i], 
                linewidth=2, label=label, alpha=0.9)
    
    ax.set_xlabel(r'Magnetic Field $B$ [mT]', fontsize=14)
    ax.set_ylabel(r'Normalized Critical Current $I_c/I_0$', fontsize=14)
    ax.set_title(r'Fraunhofer Pattern: $I_c/I_0$ vs $B$ for Different Cr Thicknesses', fontsize=16)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([B_array[0]*1e3, B_array[-1]*1e3])
    ax.set_ylim([0, None])
    
    # Add annotation explaining physics
    ax.annotate(
        r'Airy pattern: $I_c \propto \left|\frac{2J_1(\pi\Phi/\Phi_0)}{\pi\Phi/\Phi_0}\right|$',
        xy=(0.02, 0.95), xycoords='axes fraction',
        fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_Ic_vs_B_hysteresis(
    materials: MaterialParameters,
    geometry: JunctionGeometry,
    conditions: ExperimentalConditions,
    d_Cr: float = 5e-9,
    save_path: str = None
):
    """
    Plot 1b: I_c/I_0 vs B showing up-sweep and down-sweep hysteresis.
    
    This demonstrates how the Fe magnetization hysteresis creates
    asymmetry in the Fraunhofer pattern depending on field history.
    
    Physics:
    --------
    The internal magnetization M adds to the flux:
    Phi_total = B*A + mu_0*M*d_F*w
    
    Since M depends on sweep history, the pattern shifts differently
    for up vs down sweeps.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    B_array = conditions.B_range
    calc = FraunhoferCalculator(materials, geometry, conditions, d_Cr=d_Cr)
    
    # Both sweep directions
    Ic_up = calc.critical_current_vs_B(B_array, sweep_direction='up')
    Ic_down = calc.critical_current_vs_B(B_array, sweep_direction='down')
    
    ax.plot(B_array * 1e3, Ic_up, 'b-', linewidth=2, label='Up sweep', alpha=0.8)
    ax.plot(B_array * 1e3, Ic_down, 'r--', linewidth=2, label='Down sweep', alpha=0.8)
    
    # Highlight asymmetry region
    ax.axvspan(-10, 10, alpha=0.1, color='yellow', label='Hysteresis region')
    
    ax.set_xlabel(r'Magnetic Field $B$ [mT]', fontsize=14)
    ax.set_ylabel(r'Normalized Critical Current $I_c/I_0$', fontsize=14)
    ax.set_title(f'Hysteresis in Fraunhofer Pattern ($d_{{Cr}} = {d_Cr*1e9:.0f}$ nm)', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_xlim([B_array[0]*1e3, B_array[-1]*1e3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_IV_characteristics(
    materials: MaterialParameters,
    geometry: JunctionGeometry,
    conditions: ExperimentalConditions,
    d_Cr: float = 5e-9,
    save_path: str = None
):
    """
    Plot 2: I-V characteristics at different magnetic fields.
    
    This shows the superconducting-to-normal transition as current increases.
    At |I| < I_c: zero voltage (supercurrent)
    At |I| > I_c: ohmic behavior (normal current)
    
    Physics:
    --------
    RSJ (Resistively Shunted Junction) model:
    V = 0 for |I| < I_c
    V = R_N*sqrt(I^2 - I_c^2) for |I| > I_c
    
    Different B fields give different I_c, changing the switching current.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    calc = FraunhoferCalculator(materials, geometry, conditions, d_Cr=d_Cr)
    
    # Different magnetic fields
    B_values = [0, 20e-3, 50e-3, 100e-3]
    colors = ['blue', 'green', 'orange', 'red']
    
    for B, color in zip(B_values, colors):
        V, I = calc.generate_IV_characteristic(B=B, noise_level=0.01)
        label = f'$B = {B*1e3:.0f}$ mT'
        ax.plot(V * 1e3, I * 1e6, color=color, linewidth=1.5, label=label, alpha=0.8)
    
    ax.set_xlabel(r'Voltage $V$ [mV]', fontsize=14)
    ax.set_ylabel(r'Current $I$ [$\mu$A]', fontsize=14)
    ax.set_title(f'I-V Characteristics at Different Magnetic Fields ($d_{{Cr}} = {d_Cr*1e9:.0f}$ nm)', fontsize=16)
    ax.legend(loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add annotation
    ax.annotate(
        r'$I_c$ decreases with $B$' + '\n' + r'(Fraunhofer suppression)',
        xy=(0.7, 0.15), xycoords='axes fraction',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_Ic_vs_Cr_depth(
    materials: MaterialParameters,
    save_path: str = None
):
    """
    Plot 3: I_c/I_0 vs Cr thickness (depth dependence).
    
    This shows exponential decay of critical current with increasing
    Cr thickness, reflecting the decay of triplet pair amplitude.
    
    Physics:
    --------
    The triplet amplitude decays through the Cr layer:
    f_T(d_Cr) proportional to exp(-d_Cr/xi_T^Cr)
    
    Since I_c proportional to |f_T|^2:
    I_c(d_Cr) proportional to exp(-2*d_Cr/xi_T^Cr)
    
    The characteristic decay length reveals information about
    the triplet coherence length in the Cr antiferromagnet.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    calc = PairAmplitudeCalculator(materials)
    
    # Cr thickness range
    Cr_range = np.linspace(0.5e-9, 20e-9, 200)
    Ic_norm = calc.critical_current_vs_Cr_thickness(Cr_range)
    
    # Normalize to maximum
    Ic_norm = Ic_norm / Ic_norm.max()
    
    # Plot on log scale to show exponential decay
    ax.semilogy(Cr_range * 1e9, Ic_norm, 'b-', linewidth=2.5, label='Simulation')
    
    # Mark experimental data points (JJ1-JJ6)
    Cr_exp = np.array([2, 4, 6, 8, 10, 12]) * 1e-9
    Ic_exp = calc.critical_current_vs_Cr_thickness(Cr_exp)
    Ic_exp = Ic_exp / Ic_exp.max()
    ax.semilogy(Cr_exp * 1e9, Ic_exp, 'ro', markersize=10, 
                label='Experimental points (JJ1-JJ6)', markeredgecolor='black')
    
    # Fit exponential for decay length extraction
    from scipy.optimize import curve_fit
    def exp_decay(x, a, xi):
        return a * np.exp(-x / xi)
    
    try:
        popt, _ = curve_fit(exp_decay, Cr_range, Ic_norm, p0=[1, 5e-9])
        xi_fit = popt[1]
        ax.semilogy(Cr_range * 1e9, exp_decay(Cr_range, *popt), 'g--', 
                    linewidth=2, alpha=0.7,
                    label=rf'Fit: $\xi_T^{{Cr}} = {xi_fit*1e9:.1f}$ nm')
    except:
        pass
    
    ax.set_xlabel(r'Cr Thickness $d_{Cr}$ [nm]', fontsize=14)
    ax.set_ylabel(r'Normalized Critical Current $I_c/I_0$', fontsize=14)
    ax.set_title(r'Critical Current Decay with Cr Thickness', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 20])
    ax.set_ylim([1e-4, 2])
    
    # Add physics annotation
    ax.annotate(
        r'$I_c \propto \exp(-2d_{Cr}/\xi_T^{Cr})$',
        xy=(0.5, 0.85), xycoords='axes fraction',
        fontsize=13, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def run_full_simulation():
    """
    Run the complete simulation and generate all plots.
    
    This is the main entry point for the simulation.
    """
    print("=" * 60)
    print("Spin-Triplet Josephson Junction Simulation")
    print("Nb/Cu/Cr/Fe/Cr/Cu/Nb Heterostructure")
    print("=" * 60)
    
    # Setup
    setup_plot_style()
    mat, geo, cond = get_default_parameters()
    
    print("\n=== Parameters ===")
    print(f"Junction area: {geo.area * 1e12:.2f} um^2")
    print(f"Fe thickness: {mat.d_Fe * 1e9:.1f} nm")
    print(f"Cu spacer: {mat.d_Cu * 1e9:.1f} nm")
    print(f"Temperature: {cond.temperature} K")
    print(f"B-field range: {cond.B_min*1e3:.0f} to {cond.B_max*1e3:.0f} mT")
    print(f"Cr thicknesses: {[d*1e9 for d in cond.Cr_thicknesses]} nm")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    # Plot 1: Ic vs B for different Cr thicknesses
    print("1. Generating Ic vs B (multi-Cr)...")
    fig1, ax1 = plot_Ic_vs_B_multi_Cr(
        mat, geo, cond, 
        save_path='figures/01_Ic_vs_B_multi_Cr.png'
    )
    
    # Plot 1b: Hysteresis comparison
    print("2. Generating Ic vs B (hysteresis)...")
    fig1b, ax1b = plot_Ic_vs_B_hysteresis(
        mat, geo, cond, d_Cr=5e-9,
        save_path='figures/02_Ic_vs_B_hysteresis.png'
    )
    
    # Plot 2: IV characteristics
    print("3. Generating I-V characteristics...")
    fig2, ax2 = plot_IV_characteristics(
        mat, geo, cond, d_Cr=5e-9,
        save_path='figures/03_IV_characteristics.png'
    )
    
    # Plot 3: Ic vs Cr depth
    print("4. Generating Ic vs Cr depth...")
    fig3, ax3 = plot_Ic_vs_Cr_depth(
        mat,
        save_path='figures/04_Ic_vs_Cr_depth.png'
    )
    
    print("\n=== Simulation Complete ===")
    print("All figures saved to 'figures/' directory")
    
    # Show all plots
    plt.show()
    
    return fig1, fig1b, fig2, fig3


if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Run simulation
    run_full_simulation()
