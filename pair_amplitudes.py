"""
Cooper Pair Amplitude Calculations
===================================

This module computes the spatial distribution of singlet and triplet
Cooper pair amplitudes through the Nb/Cu/Cr/Fe/Cr/Cu/Nb junction stack.

Physics:
--------
The anomalous Green's function f(x) describes the Cooper pair amplitude.
It satisfies the Usadel equation in the diffusive limit:

D * nabla^2 f + 2i(E + h·sigma) f = 0

where:
- D is the diffusion constant
- E is energy
- h is the exchange field
- sigma are Pauli matrices

References:
- Bergeret et al., Rev. Mod. Phys. 77, 1321 (2005)
- Houzet & Buzdin, Phys. Rev. B 76, 060504 (2007)
"""

import numpy as np
from typing import Tuple
from parameters import MaterialParameters, get_default_parameters


class PairAmplitudeCalculator:
    r"""
    Calculates singlet and triplet Cooper pair amplitudes through the junction.
    
    The junction stack is modeled as regions with different decay characteristics:
    
    Region 1: Nb (left) - Source of singlet pairs
    Region 2: Cu (spacer) - Triplet-friendly, long coherence length
    Region 3: Cr (AFM) - Contains spin-glass interface
    Region 4: Fe (FM) - Strong exchange, singlet→triplet conversion
    Region 5: Cr (AFM) - Contains spin-glass interface  
    Region 6: Cu (spacer) - Triplet-friendly
    Region 7: Nb (right) - Drain
    """
    
    def __init__(self, materials: MaterialParameters, d_Cr: float = None):
        """
        Initialize calculator with material parameters.
        
        Parameters
        ----------
        materials : MaterialParameters
            Material properties for each layer
        d_Cr : float, optional
            Cr layer thickness [m]. If None, uses default from materials.
        """
        self.mat = materials
        self.d_Cr = d_Cr if d_Cr is not None else materials.d_Cr
        
    def singlet_decay_in_ferromagnet(self, x: np.ndarray) -> np.ndarray:
        """
        Singlet pair amplitude decay in ferromagnetic region.
        
        In a ferromagnet, singlet pairs experience the exchange field,
        leading to FFLO-like oscillations:
        
        f_S(x) = f_0 * exp(-x/xi_F1) * cos(x/xi_F2 + phi_0)
        
        The oscillation causes 0-pi transitions at certain thicknesses.
        
        Parameters
        ----------
        x : np.ndarray
            Position within ferromagnet [m]
            
        Returns
        -------
        np.ndarray
            Normalized singlet amplitude |f_S|
        """
        # Decay length (imaginary part of complex coherence length)
        xi_F1 = self.mat.xi_F
        
        # Oscillation length (real part) - typically similar magnitude
        xi_F2 = self.mat.xi_F * 1.2
        
        # Damped oscillation
        decay = np.exp(-x / xi_F1)
        oscillation = np.cos(x / xi_F2)
        
        return np.abs(decay * oscillation)
    
    def triplet_decay_in_normal_metal(self, x: np.ndarray) -> np.ndarray:
        """
        Long-range triplet amplitude decay in normal metal (Cu).
        
        Triplet pairs with S_z = ±1 are not affected by the exchange
        field (their spin is parallel to magnetization), so they decay
        only due to normal depairing:
        
        f_T(x) = f_0 * exp(-x/xi_N)
        
        With xi_N ~ 100 nm >> xi_F ~ 0.5 nm, triplets can
        carry supercurrent through thick ferromagnetic barriers.
        
        Parameters
        ----------
        x : np.ndarray
            Position within normal metal [m]
            
        Returns
        -------
        np.ndarray
            Normalized triplet amplitude |f_T|
        """
        return np.exp(-x / self.mat.xi_N)
    
    def triplet_decay_in_superconductor(self, x: np.ndarray) -> np.ndarray:
        """
        Triplet amplitude decay in superconductor (Nb).
        
        In the s-wave superconductor, triplet pairs are rapidly 
        suppressed due to:
        1. Singlet-state blocking (incompatible symmetry)
        2. Spin-orbit scattering
        
        f_T(x) = f_0 * exp(-x/xi_T^Nb)
        
        With xi_T^Nb ~ 1.2 nm, this is very short range!
        This is why the Cu spacer is crucial - it keeps triplets
        away from the hostile Nb environment.
        
        Parameters
        ----------
        x : np.ndarray
            Position within superconductor [m]
            
        Returns
        -------
        np.ndarray
            Normalized triplet amplitude |f_T|
        """
        return np.exp(-x / self.mat.xi_T_Nb)
    
    def singlet_to_triplet_conversion(self, disorder_strength: float = 0.5) -> float:
        """
        Singlet-to-triplet conversion probability at spin-glass interface.
        
        The conversion occurs at the Fe/Cr interface where magnetic
        moments are disordered ("spin-glass" behavior). The conversion
        probability depends on:
        
        1. Spin-mixing angle theta_mix: rotation of spin quantization axis
        2. Magnetic disorder sigma_theta: angular spread of moments
        3. Interface transparency gamma_B: boundary resistance
        
        The conversion probability can be estimated as:
        
        P_S->T ~= sin^2(theta_mix) * (1 - exp(-d_dead/xi_F))
        
        Parameters
        ----------
        disorder_strength : float
            Dimensionless disorder parameter (0 = ordered, 1 = fully random)
            
        Returns
        -------
        float
            Conversion probability (0 to 1)
        """
        # Effective spin-mixing angle from disorder
        # Higher disorder = more misalignment = more conversion
        theta_mix = disorder_strength * np.pi / 4
        
        # Dead layer contribution
        dead_layer_factor = 1 - np.exp(-self.mat.d_dead / self.mat.xi_F)
        
        # Conversion probability
        P_conversion = np.sin(theta_mix)**2 * dead_layer_factor
        
        return P_conversion
    
    def effective_triplet_amplitude(self, d_Cr: float = None) -> float:
        """
        Calculate effective triplet amplitude reaching the far Nb electrode.
        
        This traces the triplet pair through the full journey:
        
        1. Generated at left Fe/Cr interface (conversion from singlet)
        2. Propagates through Fe (rapid decay, but short distance)
        3. Propagates through right Cr/Cu (longer range)
        4. Enters right Nb (rapid suppression)
        
        The effective amplitude determines I_c:
        I_c proportional to |f_T^eff|^2
        
        Parameters
        ----------
        d_Cr : float, optional
            Cr thickness [m]. If None, uses instance value.
            
        Returns
        -------
        float
            Effective triplet amplitude (normalized)
        """
        if d_Cr is None:
            d_Cr = self.d_Cr
            
        # Step 1: Singlet-to-triplet conversion at interface
        P_convert = self.singlet_to_triplet_conversion(disorder_strength=0.6)
        
        # Step 2: Decay through Fe layer
        xi_T_Fe = self.mat.xi_F * 5
        f_after_Fe = np.exp(-self.mat.d_Fe / xi_T_Fe)
        
        # Step 3: Decay through Cr layer
        xi_T_Cr = 5e-9
        f_after_Cr = np.exp(-d_Cr / xi_T_Cr)
        
        # Step 4: Propagation through Cu spacer (if present)
        if self.mat.d_Cu > 0:
            f_after_Cu = np.exp(-self.mat.d_Cu / self.mat.xi_N)
        else:
            f_after_Cu = 1.0
        
        # Step 5: Entry into Nb (severe suppression)
        penetration_depth = 1e-9
        f_in_Nb = np.exp(-penetration_depth / self.mat.xi_T_Nb)
        
        # Total effective amplitude
        f_eff = np.sqrt(P_convert) * f_after_Fe * f_after_Cr * f_after_Cu * f_in_Nb
        
        return f_eff
    
    def critical_current_vs_Cr_thickness(
        self, 
        Cr_range: np.ndarray
    ) -> np.ndarray:
        """
        Calculate normalized critical current as function of Cr thickness.
        
        The critical current depends on the triplet amplitude reaching
        the far electrode:
        
        I_c(d_Cr) = I_0 |f_T^eff(d_Cr)|^2
        
        We expect exponential decay with possible oscillations from
        0-pi physics.
        
        Parameters
        ----------
        Cr_range : np.ndarray
            Array of Cr thicknesses [m]
            
        Returns
        -------
        np.ndarray
            Normalized critical current I_c/I_0
        """
        Ic_normalized = np.zeros_like(Cr_range)
        
        for i, d_Cr in enumerate(Cr_range):
            f_eff = self.effective_triplet_amplitude(d_Cr)
            Ic_normalized[i] = f_eff**2
            
        return Ic_normalized


def demo_pair_amplitudes():
    """Demonstrate pair amplitude calculations."""
    mat, geo, exp = get_default_parameters()
    calc = PairAmplitudeCalculator(mat)
    
    # Calculate Ic vs Cr thickness
    Cr_range = np.linspace(1e-9, 15e-9, 100)
    Ic_norm = calc.critical_current_vs_Cr_thickness(Cr_range)
    
    print("=== Pair Amplitude Demo ===")
    print(f"Conversion probability: {calc.singlet_to_triplet_conversion():.3f}")
    print(f"Effective amplitude (d_Cr=5nm): {calc.effective_triplet_amplitude(5e-9):.4f}")
    
    return Cr_range, Ic_norm


if __name__ == "__main__":
    Cr_range, Ic_norm = demo_pair_amplitudes()
    print(f"Max Ic/I0: {Ic_norm.max():.4f} at d_Cr = {Cr_range[np.argmax(Ic_norm)]*1e9:.1f} nm")
