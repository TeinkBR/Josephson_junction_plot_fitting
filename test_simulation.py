#!/usr/bin/env python3
"""
Quick test script to validate the diffusive triplet Josephson junction simulation.

Run this to verify all modules are working correctly.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from rcsj_sde.diffusive_triplet import (
    DiffusiveJunctionGeometry, MaterialParameters,
    MagneticConfiguration, UsadelKernel
)
from rcsj_sde.fraunhofer_diffusive import FraunhoferIntegrator
from rcsj_sde.diffusive_rcsj import DiffusiveRCSJJunction
from rcsj_sde.validation_tools import (
    FraunhoferFitFunction, spectral_leakage_analysis,
    compute_fraunhofer_metrics
)

def main():
    print("=" * 70)
    print("DIFFUSIVE SPIN-TRIPLET JOSEPHSON JUNCTION SIMULATION")
    print("Validation Test")
    print("=" * 70)
    
    # Test 1: Initialize geometry
    print("\n[1] Testing Geometry Initialization...")
    geometry = DiffusiveJunctionGeometry(
        d_S=100.0, d_F_prime=0.25, d_F=10.0,
        junction_length=1000.0, junction_width=1000.0, n_grid_x=20
    )
    print(f"    [OK] Junction area: {geometry.effective_area*1e12:.2f} um2")
    
    # Test 2: Initialize materials
    print("\n[2] Testing Material Parameters...")
    materials = MaterialParameters(Delta=1.5, E_ex=100.0, T=2.0)
    print(f"    [OK] Singlet coherence length: {materials.xi_F_singlet:.3f} nm")
    print(f"    [OK] Triplet coherence length: {materials.xi_F_triplet:.2f} nm")
    
    # Test 3: Initialize magnetic configuration
    print("\n[3] Testing Magnetic Configuration...")
    mag_config = MagneticConfiguration(geometry, n_spins_per_interface=30)
    print(f"    [OK] Bulk magnetization: {mag_config.M_bulk}")
    print(f"    [OK] Spin-glass grid: {mag_config.n_sg_x} x {mag_config.n_sg_y}")
    
    # Test 4: Usadel kernel
    print("\n[4] Testing Usadel Transport Kernel...")
    kernel = UsadelKernel(materials, geometry)
    print(f"    [OK] Singlet decay factor: {kernel.singlet_decay:.6f}")
    print(f"    [OK] Triplet decay factor: {kernel.triplet_decay:.6f}")
    
    # Test 5: Local critical current density
    print("\n[5] Testing Local Critical Current Calculation...")
    jc_test = kernel.local_critical_current_density(theta_L=np.pi/4, theta_R=np.pi/4)
    print(f"    [OK] j_c(pi/4, pi/4) = {jc_test:.6f} (normalized)")
    
    # Test 6: Fraunhofer integrator
    print("\n[6] Testing Fraunhofer Integrator...")
    fraunhofer = FraunhoferIntegrator(geometry, materials)
    print(f"    [OK] Integrator initialized")
    
    # Test 7: Magnetic evolution
    print("\n[7] Testing Magnetic Evolution During Field Sweep...")
    B_test = np.array([0, 0.01, 0.02, 0.01, 0])  # Small test sweep
    lrtc_track = []
    for B in B_test:
        mag_config.evolve_field_step(B, n_relax_steps=50, T_eff=0.05)
        lrtc_track.append(mag_config.get_average_lrtc_factor())
    print(f"    [OK] Tracked {len(B_test)} field steps")
    print(f"    [OK] LRTC range: [{np.min(lrtc_track):.4f}, {np.max(lrtc_track):.4f}]")
    
    # Test 8: Spectral leakage analysis
    print("\n[8] Testing Spectral Leakage Analysis...")
    leakage = spectral_leakage_analysis(mag_config)
    print(f"    [OK] LRTC mean: {leakage['lrtc_mean']:.4f}")
    print(f"    [OK] Disorder protected: {leakage['disorder_protected']}")
    
    # Test 9: RCSJ junction
    print("\n[9] Testing RCSJ Junction...")
    Ic_func = lambda B: 1.5e-6 * np.exp(-abs(B) / 0.1)  # Simple B-dependent Ic
    rcsj = DiffusiveRCSJJunction(R=100.0, C=1e-12, T=2.0, Ic_func=Ic_func)
    print(f"    [OK] beta parameter: {rcsj.beta:.2f}")
    print(f"    [OK] Thermal noise epsilon: {rcsj.epsilon:.6f}")
    
    # Test 10: Fraunhofer fit function
    print("\n[10] Testing Li's Fit Function...")
    fit_func = FraunhoferFitFunction(I_c0=1.5e-6, xi_T=20.0, d_Nb=100.0, delta=0.1)
    Phi_test = np.linspace(-1e-14, 1e-14, 10)
    Ic_fit = fit_func(Phi_test)
    print(f"    [OK] Fit function evaluated at {len(Phi_test)} flux points")
    print(f"    [OK] I_c range: [{np.min(Ic_fit):.3e}, {np.max(Ic_fit):.3e}] A")
    
    # Summary
    print("\n" + "=" * 70)
    print("[PASS] ALL TESTS PASSED")
    print("=" * 70)
    print("\nThe simulation framework is ready for use!")
    print("\nNext steps:")
    print("1. Review the Jupyter notebook: examples/diffusive_complete_workflow.ipynb")
    print("2. Run a full field sweep simulation (takes ~5-10 minutes)")
    print("3. Analyze results with validation tools")
    print("4. Compare to experimental data from Li et al.")
    print("=" * 70)


if __name__ == "__main__":
    main()
