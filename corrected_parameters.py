"""
Corrected Simulation Parameters
================================

Based on comparison with experimental data, we identified these issues:

1. Junction area is 467x too large → Fraunhofer pattern too narrow
2. Hysteresis is too weak → 58.5 mT shift observed, we had ~10 mT
3. Missing absolute Ic scale
4. Peak offset indicates significant trapped flux / remanent magnetization

This module provides corrected parameters.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from parameters import PHI_0, MU_0, HBAR, E_CHARGE, K_B


@dataclass
class CorrectedMaterialParameters:
    """
    Corrected material parameters based on experimental data analysis.
    
    Key corrections:
    - Junction dimensions derived from Fraunhofer pattern width
    - Hysteresis parameters from sweep asymmetry
    """
    # Layer thicknesses [m] - unchanged from experimental report
    d_Nb_base: float = 20e-9
    d_Nb_top: float = 5e-9
    d_Cu: float = 2e-9
    d_Fe: float = 3e-9
    d_Cr: float = 5e-9
    d_dead: float = 1.5e-9
    
    # Superconductor (Nb) properties
    Tc_Nb: float = 9.3
    Delta_0: float = 1.5e-3
    xi_S: float = 6e-9
    xi_T_Nb: float = 1.2e-9
    
    # Normal metal (Cu) properties
    xi_N: float = 100e-9
    
    # Ferromagnet (Fe) properties
    E_ex_Fe: float = 1.0
    xi_F: float = 0.5e-9
    
    # Antiferromagnet (Cr) properties
    T_Neel: float = 311
    
    # CORRECTED: London penetration depth (affects d_eff)
    lambda_L: float = 85e-9  # Nb London penetration depth


@dataclass
class CorrectedJunctionGeometry:
    """
    Corrected junction geometry based on experimental Fraunhofer pattern.
    
    The pattern width is determined by:
    B_first_zero = 3.83 * Phi_0 / (pi * A_eff)
    
    For Airy pattern (ellipse), A_eff = pi * a * b
    
    From experimental data:
    - Pattern extends ~100 mT on each side
    - First minimum appears around 30-40 mT
    - This indicates much smaller effective area than originally assumed
    
    HOWEVER: The "area" in Fraunhofer is actually:
    A_eff = W * d_eff
    
    where:
    - W = junction width perpendicular to field
    - d_eff = magnetic thickness = d_barrier + 2*lambda_L
    
    This is NOT the physical junction area!
    """
    # Physical dimensions (from lithography)
    physical_length: float = 5e-6      # Along field direction [m]
    physical_width: float = 2.5e-6     # Perpendicular to field [m]
    
    # Effective magnetic thickness
    # d_eff = d_barrier + 2*lambda_L
    # d_barrier = d_Cu + d_Cr + d_Fe + d_Cr + d_Cu
    #           = 2 + 5 + 3 + 5 + 2 = 17 nm
    # lambda_L (Nb) = 85 nm
    # d_eff = 17 nm + 2 * 85 nm = 187 nm
    d_barrier: float = 17e-9
    lambda_L: float = 85e-9
    
    @property
    def d_eff(self) -> float:
        """Effective magnetic thickness for flux calculation."""
        return self.d_barrier + 2 * self.lambda_L
    
    @property
    def effective_area(self) -> float:
        """
        Effective area for Fraunhofer pattern.
        
        A_eff = W * d_eff
        
        This determines the period of the diffraction pattern.
        """
        return self.physical_width * self.d_eff
    
    @property
    def physical_area(self) -> float:
        """Physical junction area (for current density)."""
        return np.pi * self.physical_length * self.physical_width / 4  # Ellipse
    
    def flux_quantum_field(self) -> float:
        """
        Magnetic field corresponding to one flux quantum.
        
        B_phi0 = Phi_0 / A_eff
        """
        return PHI_0 / self.effective_area
    
    def first_zero_field_airy(self) -> float:
        """
        Field at first zero of Airy pattern.
        
        For J_1(x) = 0, first zero at x = 3.832
        So: pi * Phi / Phi_0 = 3.832
        Phi = 3.832 * Phi_0 / pi = 1.22 * Phi_0
        B = 1.22 * Phi_0 / A_eff
        """
        return 1.22 * PHI_0 / self.effective_area


@dataclass
class CorrectedHysteresisModel:
    """
    Corrected hysteresis model based on experimental sweep data.
    
    From analysis:
    - Upsweep peak at H = -46.4 mT
    - Downsweep peak at H = +12.1 mT
    - Total hysteresis shift = 58.5 mT
    
    This indicates:
    - Strong remanent magnetization in Fe
    - Possible exchange bias from Cr antiferromagnet
    - Asymmetric switching behavior
    """
    # Saturation magnetization (Fe)
    M_s: float = 1.7e6           # [A/m]
    
    # Coercive field - CORRECTED based on experimental shift
    H_c: float = 30e-3           # [T] - much larger than before!
    
    # Remanence ratio
    M_r_ratio: float = 0.95      # High remanence
    
    # Exchange bias from Cr (causes asymmetry)
    H_exchange_bias: float = 15e-3   # [T] - shifts entire loop
    
    def magnetization(self, B: float, sweep_direction: str = 'up') -> float:
        """
        Get magnetization at field B with exchange bias.
        
        The exchange bias shifts the hysteresis loop horizontally,
        creating different behavior for up vs down sweeps.
        """
        # Effective field including exchange bias
        B_eff = B - self.H_exchange_bias
        
        if sweep_direction == 'up':
            if B_eff > self.H_c:
                return self.M_s
            elif B_eff < -self.H_c:
                return -self.M_s
            else:
                # Smooth transition (tanh-like instead of linear)
                return self.M_s * np.tanh(2 * B_eff / self.H_c)
        else:  # down sweep
            if B_eff < -self.H_c:
                return -self.M_s
            elif B_eff > self.H_c:
                return self.M_s
            else:
                return -self.M_s * np.tanh(2 * B_eff / self.H_c)


@dataclass
class CorrectedExperimentalConditions:
    """Experimental conditions (unchanged)."""
    temperature: float = 4.2
    B_min: float = -120e-3
    B_max: float = 120e-3
    B_points: int = 500
    Cr_thicknesses: Tuple[float, ...] = (2e-9, 4e-9, 6e-9, 8e-9, 10e-9, 12e-9)
    
    @property
    def B_range(self) -> np.ndarray:
        return np.linspace(self.B_min, self.B_max, self.B_points)


def get_corrected_parameters():
    """Return corrected parameter objects."""
    materials = CorrectedMaterialParameters()
    geometry = CorrectedJunctionGeometry()
    conditions = CorrectedExperimentalConditions()
    hysteresis = CorrectedHysteresisModel()
    
    return materials, geometry, conditions, hysteresis


def print_parameter_comparison():
    """Print comparison of original vs corrected parameters."""
    from parameters import get_default_parameters, JunctionGeometry
    
    orig_mat, orig_geo, orig_cond = get_default_parameters()
    corr_mat, corr_geo, corr_cond, corr_hyst = get_corrected_parameters()
    
    print("="*60)
    print("PARAMETER COMPARISON: Original vs Corrected")
    print("="*60)
    
    print("\n--- Junction Geometry ---")
    print(f"{'Parameter':<30} {'Original':<20} {'Corrected':<20}")
    print("-"*70)
    print(f"{'Physical area [um^2]':<30} {orig_geo.area * 1e12:<20.2f} {corr_geo.physical_area * 1e12:<20.2f}")
    print(f"{'Effective area [um^2]':<30} {orig_geo.area * 1e12:<20.2f} {corr_geo.effective_area * 1e12:<20.4f}")
    print(f"{'d_eff [nm]':<30} {'N/A':<20} {corr_geo.d_eff * 1e9:<20.1f}")
    print(f"{'First zero (Airy) [mT]':<30} {0.06:<20.2f} {corr_geo.first_zero_field_airy() * 1e3:<20.1f}")
    
    print("\n--- Hysteresis ---")
    print(f"{'Parameter':<30} {'Original':<20} {'Corrected':<20}")
    print("-"*70)
    print(f"{'Coercive field H_c [mT]':<30} {5:<20} {corr_hyst.H_c * 1e3:<20.1f}")
    print(f"{'Exchange bias [mT]':<30} {0:<20} {corr_hyst.H_exchange_bias * 1e3:<20.1f}")
    print(f"{'Remanence ratio':<30} {0.9:<20.1f} {corr_hyst.M_r_ratio:<20.2f}")


if __name__ == "__main__":
    print_parameter_comparison()
    
    mat, geo, cond, hyst = get_corrected_parameters()
    print(f"\nCorrected first zero at: {geo.first_zero_field_airy() * 1e3:.1f} mT")
    print(f"Flux quantum field: {geo.flux_quantum_field() * 1e3:.1f} mT")
