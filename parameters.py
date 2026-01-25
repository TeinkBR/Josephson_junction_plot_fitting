"""
Spin-Triplet Josephson Junction Simulation
===========================================
Physical Constants and Experimental Parameters

Based on experimental setup:
- Nb/Cu/Cr/Fe/Cr/Cu/Nb heterostructure
- Elliptical junction geometry
- Temperature: 4.2 K (liquid helium)

References:
- Bergeret et al., Rev. Mod. Phys. 77, 1321 (2005)
- Komori et al., Phys. Rev. B 104, 054503 (2021)
- Glick et al., Sci. Adv. 3, e1601614 (2017)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# =============================================================================
# Fundamental Physical Constants
# =============================================================================

# Planck's constant: $\hbar$ [J·s]
HBAR = 1.054571817e-34

# Electron charge: $e$ [C]
E_CHARGE = 1.602176634e-19

# Boltzmann constant: $k_B$ [J/K]
K_B = 1.380649e-23

# Magnetic flux quantum: $\Phi_0 = h/(2e)$ [Wb]
PHI_0 = 2.067833848e-15

# Vacuum permeability: $\mu_0$ [H/m]
MU_0 = 4 * np.pi * 1e-7


@dataclass
class MaterialParameters:
    r"""
    Material-specific parameters for each layer in the junction stack.
    
    The coherence length determines how far Cooper pairs can propagate:
    $$\xi = \sqrt{\frac{\hbar D}{2\pi k_B T_c}}$$
    where D is the diffusion constant.
    """
    # Layer thicknesses [m]
    d_Nb_base: float = 20e-9      # Bottom Nb electrode
    d_Nb_top: float = 5e-9        # Top Nb electrode
    d_Cu: float = 2e-9            # Cu spacer (Group A) - protects triplets from Nb SOC
    d_Fe: float = 3e-9            # Fe ferromagnetic layer
    d_Cr: float = 5e-9            # Cr antiferromagnetic layer (variable)
    d_dead: float = 1.5e-9        # Spin-glass "dead layer" at Fe/Cr interface
    
    # Superconductor (Nb) properties
    Tc_Nb: float = 9.3            # Critical temperature [K]
    Delta_0: float = 1.5e-3       # Superconducting gap at T=0 [eV]
    xi_S: float = 6e-9            # Singlet coherence length in Nb (dirty limit) [m]
    xi_T_Nb: float = 1.2e-9       # Triplet coherence length in Nb (short due to SOC) [m]
    
    # Normal metal (Cu) properties
    xi_N: float = 100e-9          # Triplet coherence length in Cu [m]
    
    # Ferromagnet (Fe) properties
    E_ex_Fe: float = 1.0          # Exchange energy [eV]
    xi_F: float = 0.5e-9          # Ferromagnetic coherence length [m]
    
    # Antiferromagnet (Cr) properties
    T_Neel: float = 311           # Néel temperature [K]


@dataclass
class JunctionGeometry:
    r"""
    Junction geometry parameters.
    
    The junction has an elliptical cross-section, which modifies the 
    Fraunhofer pattern from sin(x)/x to the Airy function 2J_1(x)/x.
    
    The effective area is:
    $$A_{eff} = \pi a b$$
    where a and b are the semi-major and semi-minor axes.
    """
    # Ellipse parameters [m]
    semi_major: float = 5e-6      # Semi-major axis
    semi_minor: float = 2.5e-6    # Semi-minor axis
    
    @property
    def area(self) -> float:
        """Junction area: A = pi*a*b"""
        return np.pi * self.semi_major * self.semi_minor
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio: a/b"""
        return self.semi_major / self.semi_minor


@dataclass 
class ExperimentalConditions:
    """
    Experimental measurement conditions.
    
    The magnetic field B is swept to generate the Fraunhofer pattern.
    The field range should cover several flux quanta through the junction.
    """
    temperature: float = 4.2      # Measurement temperature [K]
    B_min: float = -120e-3        # Minimum B-field [T]
    B_max: float = 120e-3         # Maximum B-field [T]
    B_points: int = 500           # Number of field points
    
    # Cr thickness range for parameter sweep (samples JJ1-JJ6)
    Cr_thicknesses: Tuple[float, ...] = (2e-9, 4e-9, 6e-9, 8e-9, 10e-9, 12e-9)
    
    @property
    def B_range(self) -> np.ndarray:
        """Generate B-field sweep array"""
        return np.linspace(self.B_min, self.B_max, self.B_points)


# =============================================================================
# Initialize Default Parameters
# =============================================================================

def get_default_parameters():
    """
    Returns default parameter objects based on experimental setup.
    
    These values are extracted from the experimental report on
    Nb/Cu/Cr/Fe/Cr/Cu/Nb antiferromagnetic Josephson junctions.
    """
    materials = MaterialParameters()
    geometry = JunctionGeometry()
    conditions = ExperimentalConditions()
    
    return materials, geometry, conditions


if __name__ == "__main__":
    # Quick test of parameters
    mat, geo, exp = get_default_parameters()
    
    print("=== Junction Parameters ===")
    print(f"Junction area: {geo.area * 1e12:.2f} um^2")
    print(f"Aspect ratio: {geo.aspect_ratio:.1f}")
    print(f"Singlet coherence length (Nb): {mat.xi_S * 1e9:.1f} nm")
    print(f"Triplet coherence length (Cu): {mat.xi_N * 1e9:.1f} nm")
    print(f"B-field range: {exp.B_min*1e3:.0f} to {exp.B_max*1e3:.0f} mT")
    print(f"Cr thicknesses: {[d*1e9 for d in exp.Cr_thicknesses]} nm")
