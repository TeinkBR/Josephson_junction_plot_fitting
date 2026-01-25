"""
Fraunhofer Pattern Calculation for Josephson Junctions
=======================================================

This module computes the critical current I_c(B) as a function of 
applied magnetic field, accounting for:

1. Elliptical junction geometry → Airy function pattern
2. Magnetic hysteresis → asymmetric up/down sweeps
3. Internal magnetization → phase shifts

Physics:
--------
The supercurrent density through the junction is:

j_s(x,y) = j_c * sin[phi(x,y)]

where the phase includes contributions from both external field and
internal magnetization:

phi(x,y) = phi_0 + (2*pi/Phi_0)*(B_ext + mu_0*M)*x*d_eff

The total critical current is found by integrating over the junction
area and maximizing over phi_0:

I_c(B) = max over phi_0 of |integral of j_c*sin[phi(x,y)] dx dy|

References:
- Barone & Paterno, "Physics and Applications of the Josephson Effect"
- Korucu et al. (elliptical junction theory)
"""

import numpy as np
from scipy.special import jv as bessel_j
from typing import Tuple, Optional
from dataclasses import dataclass

from parameters import (
    MaterialParameters, JunctionGeometry, ExperimentalConditions,
    PHI_0, MU_0, get_default_parameters
)
from pair_amplitudes import PairAmplitudeCalculator


@dataclass
class HysteresisModel:
    """
    Simple rectangular hysteresis model for Fe magnetization.
    
    The Fe layer has a magnetization M that depends on field history:
    - Saturates at +/- M_s for large fields
    - Switches at coercive field B_c
    - Has remanent magnetization M_r at zero field
    
    This creates the asymmetry between up-sweep and down-sweep
    observed in the experiment.
    """
    M_s: float = 1.7e6          # Saturation magnetization [A/m]
    B_c: float = 5e-3           # Coercive field [T]
    M_r_ratio: float = 0.9      # Remanence ratio M_r/M_s
    
    def magnetization(self, B: float, sweep_direction: str = 'up') -> float:
        """
        Get magnetization at field B given sweep history.
        
        Parameters
        ----------
        B : float
            External magnetic field [T]
        sweep_direction : str
            'up' (increasing B) or 'down' (decreasing B)
            
        Returns
        -------
        float
            Magnetization M [A/m]
        """
        if sweep_direction == 'up':
            if B > self.B_c:
                return self.M_s
            elif B < -self.B_c:
                return -self.M_s
            else:
                return self.M_s * self.M_r_ratio * (B / self.B_c)
        else:  # down sweep
            if B < -self.B_c:
                return -self.M_s
            elif B > self.B_c:
                return self.M_s
            else:
                return -self.M_s * self.M_r_ratio * (B / self.B_c)


class FraunhoferCalculator:
    r"""
    Calculates Fraunhofer/Airy diffraction pattern for elliptical Josephson junction.
    
    For an elliptical junction with semi-axes a and b, the critical
    current vs. flux follows the Airy function:
    
    I_c(Phi) = I_c0 * |2*J_1(pi*Phi/Phi_0) / (pi*Phi/Phi_0)|
    
    where Phi = B * A_eff is the flux through the junction.
    """
    
    def __init__(
        self, 
        materials: MaterialParameters,
        geometry: JunctionGeometry,
        conditions: ExperimentalConditions,
        d_Cr: float = None
    ):
        self.mat = materials
        self.geo = geometry
        self.cond = conditions
        self.d_Cr = d_Cr if d_Cr is not None else materials.d_Cr
        
        # Calculate effective magnetic thickness
        lambda_L = 85e-9  # Nb London penetration depth [m]
        self.d_eff = self.mat.d_Fe + 2 * self.d_Cr + 2 * lambda_L
        
        # Initialize pair amplitude calculator
        self.pair_calc = PairAmplitudeCalculator(materials, d_Cr)
        
        # Hysteresis model
        self.hysteresis = HysteresisModel()
        
    def flux_through_junction(self, B: float, M: float = 0) -> float:
        """
        Calculate total magnetic flux through junction.
        
        The flux includes contributions from:
        1. External applied field: Phi_ext = B * A
        2. Internal magnetization: Phi_M = mu_0 * M * d_F * w
        
        Parameters
        ----------
        B : float
            External magnetic field [T]
        M : float
            Internal magnetization [A/m]
            
        Returns
        -------
        float
            Total flux [Wb]
        """
        # External flux through elliptical area
        Phi_ext = B * self.geo.area
        
        # Internal flux from Fe magnetization
        effective_width = 2 * self.geo.semi_minor
        Phi_M = MU_0 * M * self.mat.d_Fe * effective_width
        
        return Phi_ext + Phi_M
    
    def airy_pattern(self, Phi: float) -> float:
        """
        Airy function pattern for elliptical junction.
        
        I_c/I_c0 = |2*J_1(pi*Phi/Phi_0) / (pi*Phi/Phi_0)|
        
        where J_1 is the Bessel function of the first kind.
        
        This replaces the sin(x)/x Fraunhofer pattern for rectangular junctions.
        
        Parameters
        ----------
        Phi : float
            Magnetic flux through junction [Wb]
            
        Returns
        -------
        float
            Normalized critical current I_c/I_c0
        """
        # Normalized flux
        x = np.pi * Phi / PHI_0
        
        if np.abs(x) < 1e-10:
            return 1.0
        else:
            return np.abs(2 * bessel_j(1, x) / x)
    
    def fraunhofer_pattern(self, Phi: float) -> float:
        """
        Standard Fraunhofer pattern (for comparison/rectangular junctions).
        
        I_c/I_c0 = |sin(pi*Phi/Phi_0) / (pi*Phi/Phi_0)|
        
        Parameters
        ----------
        Phi : float
            Magnetic flux [Wb]
            
        Returns
        -------
        float
            Normalized critical current
        """
        x = np.pi * Phi / PHI_0
        
        if np.abs(x) < 1e-10:
            return 1.0
        else:
            return np.abs(np.sin(x) / x)
    
    def critical_current_vs_B(
        self, 
        B_array: np.ndarray,
        include_hysteresis: bool = True,
        sweep_direction: str = 'up',
        use_airy: bool = True
    ) -> np.ndarray:
        """
        Calculate I_c/I_0 as function of applied magnetic field.
        
        Parameters
        ----------
        B_array : np.ndarray
            Array of B-field values [T]
        include_hysteresis : bool
            Whether to include Fe magnetization hysteresis
        sweep_direction : str
            'up' or 'down' for hysteresis
        use_airy : bool
            If True, use Airy pattern (ellipse). If False, use Fraunhofer (rectangle).
            
        Returns
        -------
        np.ndarray
            Normalized critical current at each B-field value
        """
        Ic_norm = np.zeros_like(B_array)
        
        # Base critical current from triplet amplitude (Cr-thickness dependent)
        Ic0_factor = self.pair_calc.effective_triplet_amplitude(self.d_Cr)**2
        
        for i, B in enumerate(B_array):
            # Get magnetization (with or without hysteresis)
            if include_hysteresis:
                M = self.hysteresis.magnetization(B, sweep_direction)
            else:
                M = 0
            
            # Calculate flux
            Phi = self.flux_through_junction(B, M)
            
            # Calculate diffraction pattern
            if use_airy:
                pattern = self.airy_pattern(Phi)
            else:
                pattern = self.fraunhofer_pattern(Phi)
            
            # Total normalized current
            Ic_norm[i] = Ic0_factor * pattern
            
        return Ic_norm
    
    def generate_IV_characteristic(
        self,
        B: float = 0,
        V_range: Tuple[float, float] = (-1e-3, 1e-3),
        n_points: int = 500,
        R_N: float = 1.0,
        noise_level: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V characteristic curve at given magnetic field.
        
        The Josephson junction I-V curve shows:
        - Zero voltage for |I| < I_c (supercurrent branch)
        - Ohmic behavior for |I| > I_c (normal branch)
        
        Using the RSJ (Resistively Shunted Junction) model:
        V = R_N * sqrt(I^2 - I_c^2) for |I| > I_c
        
        Parameters
        ----------
        B : float
            Applied magnetic field [T]
        V_range : tuple
            Voltage range [V]
        n_points : int
            Number of points
        R_N : float
            Normal state resistance [Ohm]
        noise_level : float
            Relative noise amplitude
            
        Returns
        -------
        V : np.ndarray
            Voltage array [V]
        I : np.ndarray
            Current array [A]
        """
        # Get critical current at this field
        Phi = self.flux_through_junction(B, M=0)
        Ic_normalized = self.airy_pattern(Phi)
        
        # Absolute critical current
        Ic0 = 1e-3
        Ic = Ic0 * Ic_normalized * self.pair_calc.effective_triplet_amplitude(self.d_Cr)**2
        
        # Generate current array
        I_max = 3 * max(Ic, 1e-6)
        I_array = np.linspace(-I_max, I_max, n_points)
        V_array = np.zeros_like(I_array)
        
        for i, I in enumerate(I_array):
            if np.abs(I) <= Ic:
                V_array[i] = 0
            else:
                V_array[i] = np.sign(I) * R_N * np.sqrt(I**2 - Ic**2)
        
        # Add realistic noise
        noise = noise_level * R_N * Ic * np.random.randn(n_points)
        V_array += noise
        
        return V_array, I_array


def demo_fraunhofer():
    """Demonstrate Fraunhofer pattern calculation."""
    mat, geo, cond = get_default_parameters()
    
    calc = FraunhoferCalculator(mat, geo, cond, d_Cr=5e-9)
    
    B_array = cond.B_range
    Ic_up = calc.critical_current_vs_B(B_array, sweep_direction='up')
    Ic_down = calc.critical_current_vs_B(B_array, sweep_direction='down')
    
    print("=== Fraunhofer Pattern Demo ===")
    print(f"Max Ic/I0 (up sweep): {Ic_up.max():.4f}")
    print(f"Max Ic/I0 (down sweep): {Ic_down.max():.4f}")
    print(f"Peak position (up): {B_array[np.argmax(Ic_up)]*1e3:.2f} mT")
    print(f"Peak position (down): {B_array[np.argmax(Ic_down)]*1e3:.2f} mT")
    
    return B_array, Ic_up, Ic_down


if __name__ == "__main__":
    demo_fraunhofer()
