"""
Module for diffusive spin-triplet Josephson junction simulations.

This module implements the diffusive (Usadel) limit framework for S/F'/F/F'/S
junctions with spin-glass interface layers, based on the Houzet-Buzdin formalism
and targeted at sputtered Nb/Fe/Cr systems with antiferromagnetically coupled
perpendicular magnetization domains.

Key references:
- Houzet & Buzdin, PRB 76, 060504(R) (2007) - Diffusive trilayer CPR
- Bergeret et al., Rev. Mod. Phys. 77, 1321 (2005) - Long-range triplet theory
- Robinson et al., Science 329, 5987 (2010) - Controlled triplet injection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import numba
from scipy import interpolate


# ============================================================================
# MODULE 1: GEOMETRIC & MAGNETIC STRUCTURE
# ============================================================================


@dataclass
class DiffusiveJunctionGeometry:
    """
    Defines the 5-layer S/F'/F/F'/S junction geometry for diffusive transport.

    Parameters
    ----------
    d_S : float
        Superconductor thickness (nm)
    d_F_prime : float
        Spin-mixer interface thickness (nm), default 0.25 nm
    d_F : float
        Bulk ferromagnet thickness (nm)
    junction_length : float
        Junction length in transport direction (nm)
    junction_width : float
        Junction width perpendicular to transport (nm)
    n_grid_x : int
        Number of spatial grid points along junction length
    n_grid_y : int
        Number of spatial grid points along width
    """
    d_S: float = 100.0          # nm
    d_F_prime: float = 0.25     # nm - from Li's experiment
    d_F: float = 10.0           # nm
    junction_length: float = 1000.0  # nm (1 μm)
    junction_width: float = 1000.0   # nm (1 μm)
    n_grid_x: int = 20          # Spatial resolution (coarser for efficiency)
    n_grid_y: int = 20

    def __post_init__(self):
        # Derived quantities
        self.total_F_thickness = 2 * self.d_F_prime + self.d_F  # nm
        self.effective_area = self.junction_length * self.junction_width * 1e-18  # m²
        
        # 2D spatial grids (normalized 0-1)
        self.x_grid = np.linspace(0, 1, self.n_grid_x)
        self.y_grid = np.linspace(0, 1, self.n_grid_y)
        self.dx = 1.0 / (self.n_grid_x - 1) if self.n_grid_x > 1 else 1.0
        self.dy = 1.0 / (self.n_grid_y - 1) if self.n_grid_y > 1 else 1.0
        
        # Create 2D mesh
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')


@dataclass
class MaterialParameters:
    """
    Material parameters for diffusive transport calculation.

    All energies in meV, lengths in nm, diffusion constants in nm²/ps.
    """
    # Superconductor (Nb)
    Delta: float = 1.5              # BCS gap (meV)
    D_S: float = 2.5                # Diffusion constant in S (nm²/ps)
    
    # Bulk Ferromagnet (Fe)
    E_ex: float = 100.0             # Exchange energy (meV)
    D_F: float = 1.0                # Diffusion constant in F (nm²/ps)
    
    # Spin-mixer interface (F')
    D_F_prime: float = 0.5          # Reduced diffusion (more disorder)
    
    # Temperature (K)
    T: float = 2.0
    
    def __post_init__(self):
        # Physical constants
        self.hbar = 0.658           # meV·ps
        self.kB = 0.0862            # meV/K
        self.g_muB = 0.058          # meV/T for electron
        
        # Coherence lengths in diffusive limit
        self.xi_S = np.sqrt(self.hbar * self.D_S / (2 * self.Delta))
        self.xi_F_singlet = np.sqrt(self.hbar * self.D_F / self.E_ex)
        self.xi_F_triplet = np.sqrt(self.hbar * self.D_F / (2 * np.pi * self.kB * self.T))


@dataclass
class SpinGlassLayer:
    """
    Represents a spin-glass F' interface layer with random exchange coupling.

    The layer is FROZEN below T_f, so spins do not thermally relax,
    but can evolve under applied field via Metropolis algorithm.

    Parameters
    ----------
    n_spins : int
        Number of spin sites in the layer
    J_mean : float
        Mean exchange coupling (meV), ~0 for frustrated system
    J_std : float
        Standard deviation of exchange (meV)
    """
    n_spins: int = 50
    J_mean: float = 0.0
    J_std: float = 10.0
    
    # State variables
    theta: np.ndarray = field(default=None, repr=False)
    phi: np.ndarray = field(default=None, repr=False)
    J_matrix: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        # Initialize angles with random orientations
        if self.theta is None:
            self.theta = np.random.uniform(0, np.pi, self.n_spins)
        if self.phi is None:
            self.phi = np.random.uniform(0, 2*np.pi, self.n_spins)
        
        # Generate symmetric random exchange matrix
        if self.J_matrix is None:
            J_upper = np.random.normal(self.J_mean, self.J_std, 
                                       (self.n_spins, self.n_spins))
            self.J_matrix = (J_upper + J_upper.T) / 2
            np.fill_diagonal(self.J_matrix, 0)
    
    def get_magnetization_vector(self) -> np.ndarray:
        """
        Calculate net magnetization direction of the spin-glass layer.

        Returns
        -------
        np.ndarray
            Unit vector (mx, my, mz) representing average magnetization
        """
        mx = np.mean(np.sin(self.theta) * np.cos(self.phi))
        my = np.mean(np.sin(self.theta) * np.sin(self.phi))
        mz = np.mean(np.cos(self.theta))
        
        norm = np.sqrt(mx**2 + my**2 + mz**2)
        if norm > 1e-10:
            return np.array([mx, my, mz]) / norm
        else:
            return np.array([0.0, 0.0, 1.0])
    
    def get_angle_to_bulk(self, bulk_direction: np.ndarray) -> float:
        """
        Calculate angle between F' magnetization and bulk F direction.

        This is the key quantity theta_L or theta_R in the Houzet-Buzdin LRTC formula.

        Parameters
        ----------
        bulk_direction : np.ndarray
            Unit vector of bulk F magnetization

        Returns
        -------
        float
            Angle in radians
        """
        m_Fprime = self.get_magnetization_vector()
        cos_angle = np.clip(np.dot(m_Fprime, bulk_direction), -1, 1)
        return np.arccos(cos_angle)
    
    def compute_energy(self, H_ext: np.ndarray) -> float:
        """
        Compute total energy of spin-glass configuration.

        E = -Σ_ij J_ij S_i · S_j - Σ_i H · S_i

        Parameters
        ----------
        H_ext : np.ndarray
            External field vector (meV units)

        Returns
        -------
        float
            Total energy in meV
        """
        # Spin vectors
        Sx = np.sin(self.theta) * np.cos(self.phi)
        Sy = np.sin(self.theta) * np.sin(self.phi)
        Sz = np.cos(self.theta)
        
        # Exchange energy
        E_ex = 0.0
        for i in range(self.n_spins):
            for j in range(i+1, self.n_spins):
                Si_dot_Sj = (Sx[i]*Sx[j] + Sy[i]*Sy[j] + Sz[i]*Sz[j])
                E_ex -= self.J_matrix[i, j] * Si_dot_Sj
        
        # Zeeman energy
        E_Z = -np.sum(H_ext[0]*Sx + H_ext[1]*Sy + H_ext[2]*Sz)
        
        return E_ex + E_Z
    
    def relax_metropolis(self, H_ext: np.ndarray, n_steps: int = 1000,
                        T_eff: float = 0.1) -> None:
        """
        Relax spin configuration using Metropolis algorithm.

        T_eff is an effective temperature for the algorithm, not physical.
        Use small T_eff for near-ground-state configurations.

        Parameters
        ----------
        H_ext : np.ndarray
            External field vector (meV)
        n_steps : int
            Number of Metropolis steps
        T_eff : float
            Effective temperature (meV) for acceptance
        """
        for _ in range(n_steps):
            # Pick random spin
            i = np.random.randint(self.n_spins)
            theta_old = self.theta[i]
            phi_old = self.phi[i]
            
            # Propose small random rotation
            self.theta[i] += np.random.normal(0, 0.1)
            self.phi[i] += np.random.normal(0, 0.1)
            
            # Enforce periodic boundary conditions
            self.theta[i] = np.clip(self.theta[i], 0, np.pi)
            self.phi[i] = self.phi[i] % (2 * np.pi)
            
            # Energy change
            E_new = self.compute_energy(H_ext)
            self.theta[i], self.phi[i] = theta_old, phi_old
            E_old = self.compute_energy(H_ext)
            
            # Metropolis acceptance
            dE = E_new - E_old
            if dE < 0 or np.random.random() < np.exp(-dE / T_eff):
                self.theta[i], self.phi[i] = self.theta[i] + np.random.normal(0, 0.1), self.phi[i] + np.random.normal(0, 0.1)
                self.theta[i] = np.clip(self.theta[i], 0, np.pi)
                self.phi[i] = self.phi[i] % (2 * np.pi)


class MagneticConfiguration:
    """
    Manages the complete magnetic state of the S/F'/F/F'/S junction.

    Tracks bulk F magnetization and spin-glass configurations for left/right F' layers.
    The key output is the non-collinearity angles theta_L(x,y) and theta_R(x,y).
    """
    
    def __init__(self, geometry: DiffusiveJunctionGeometry,
                 n_spins_per_interface: int = 50):
        """
        Initialize magnetic configuration.

        Parameters
        ----------
        geometry : DiffusiveJunctionGeometry
            Junction geometry specification
        n_spins_per_interface : int
            Number of spins per F' layer
        """
        self.geometry = geometry
        
        # Bulk F magnetization (initially along +z)
        self.M_bulk = np.array([0.0, 0.0, 1.0])
        
        # Coarse grid for spin-glass (smaller than full junction for efficiency)
        self.n_sg_x = min(10, geometry.n_grid_x)
        self.n_sg_y = min(10, geometry.n_grid_y)
        self.sg_x = np.linspace(0, 1, self.n_sg_x)
        self.sg_y = np.linspace(0, 1, self.n_sg_y)
        
        # Create 2D arrays of F' layers
        self.F_prime_left = np.empty((self.n_sg_x, self.n_sg_y), dtype=object)
        self.F_prime_right = np.empty((self.n_sg_x, self.n_sg_y), dtype=object)
        
        for i in range(self.n_sg_x):
            for j in range(self.n_sg_y):
                self.F_prime_left[i, j] = SpinGlassLayer(n_spins=n_spins_per_interface)
                self.F_prime_right[i, j] = SpinGlassLayer(n_spins=n_spins_per_interface)
    
    def get_noncollinearity_angles(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Get non-collinearity angles theta_L and theta_R at grid point (ix, iy).

        Returns
        -------
        Tuple[float, float]
            (theta_L, theta_R) in radians
        """
        theta_L = self.F_prime_left[ix, iy].get_angle_to_bulk(self.M_bulk)
        theta_R = self.F_prime_right[ix, iy].get_angle_to_bulk(self.M_bulk)
        return theta_L, theta_R
    
    def get_noncollinearity_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Interpolate non-collinearity angles at arbitrary position (x, y).

        Parameters
        ----------
        x, y : float
            Position (normalized 0-1)

        Returns
        -------
        Tuple[float, float]
            (theta_L, theta_R) interpolated
        """
        # Find bracketing indices
        ix = np.searchsorted(self.sg_x, x)
        iy = np.searchsorted(self.sg_y, y)
        ix = np.clip(ix, 1, self.n_sg_x - 1)
        iy = np.clip(iy, 1, self.n_sg_y - 1)
        
        # Bilinear interpolation
        x0, x1 = self.sg_x[ix-1], self.sg_x[ix]
        y0, y1 = self.sg_y[iy-1], self.sg_y[iy]
        tx = (x - x0) / (x1 - x0) if x1 != x0 else 0
        ty = (y - y0) / (y1 - y0) if y1 != y0 else 0
        
        theta_L_00, theta_R_00 = self.get_noncollinearity_angles(ix-1, iy-1)
        theta_L_10, theta_R_10 = self.get_noncollinearity_angles(ix, iy-1)
        theta_L_01, theta_R_01 = self.get_noncollinearity_angles(ix-1, iy)
        theta_L_11, theta_R_11 = self.get_noncollinearity_angles(ix, iy)
        
        theta_L = ((1-tx)*(1-ty)*theta_L_00 + tx*(1-ty)*theta_L_10 +
                   (1-tx)*ty*theta_L_01 + tx*ty*theta_L_11)
        theta_R = ((1-tx)*(1-ty)*theta_R_00 + tx*(1-ty)*theta_R_10 +
                   (1-tx)*ty*theta_R_01 + tx*ty*theta_R_11)
        
        return theta_L, theta_R
    
    def evolve_field_step(self, B_ext: float, n_relax_steps: int = 500,
                         T_eff: float = 0.1) -> None:
        """
        Evolve magnetic configuration for one field step.

        CRITICAL: State from previous step is carried forward, implementing
        history-dependent "memory" of frozen spin-glass.

        Parameters
        ----------
        B_ext : float
            External magnetic field (T), along z-axis
        n_relax_steps : int
            Number of Metropolis steps per F' site
        T_eff : float
            Effective temperature (meV) for Metropolis
        """
        # Convert field to energy units (meV)
        g_muB = 0.058  # meV/T
        H_ext = np.array([0.0, 0.0, g_muB * B_ext])
        
        # Update bulk Fe magnetization (simple ferromagnetic behavior)
        H_coercive = 0.01  # T
        if np.abs(B_ext) > H_coercive:
            self.M_bulk = np.array([0.0, 0.0, np.sign(B_ext)])
        
        # Relax spin-glass layers at each spatial point
        # Each site evolves independently, but retains memory from previous step
        for i in range(self.n_sg_x):
            for j in range(self.n_sg_y):
                self.F_prime_left[i, j].relax_metropolis(
                    H_ext, n_steps=n_relax_steps, T_eff=T_eff
                )
                self.F_prime_right[i, j].relax_metropolis(
                    H_ext, n_steps=n_relax_steps, T_eff=T_eff
                )
    
    def get_average_lrtc_factor(self) -> float:
        """
        Compute spatially-averaged LRTC generation factor sin(theta_L)*sin(theta_R).

        Returns
        -------
        float
            Average of sin(theta_L)*sin(theta_R) across junction
        """
        lrtc_sum = 0.0
        for i in range(self.n_sg_x):
            for j in range(self.n_sg_y):
                theta_L, theta_R = self.get_noncollinearity_angles(i, j)
                lrtc_sum += np.sin(theta_L) * np.sin(theta_R)
        return lrtc_sum / (self.n_sg_x * self.n_sg_y)
    
    def spectral_leakage_check(self) -> Dict[str, float]:
        """
        Check for "spectral leakage" - local disorder preventing SAF cancellation.

        Returns
        -------
        dict
            Diagnostic information about spectral leakage
        """
        theta_L_values = []
        theta_R_values = []
        sin_products = []
        
        for i in range(self.n_sg_x):
            for j in range(self.n_sg_y):
                tL, tR = self.get_noncollinearity_angles(i, j)
                theta_L_values.append(tL)
                theta_R_values.append(tR)
                sin_products.append(np.sin(tL) * np.sin(tR))
        
        sin_mean = np.mean(sin_products)
        sin_std = np.std(sin_products)
        
        return {
            'theta_L_mean': np.mean(theta_L_values),
            'theta_L_std': np.std(theta_L_values),
            'theta_R_mean': np.mean(theta_R_values),
            'theta_R_std': np.std(theta_R_values),
            'lrtc_factor_mean': sin_mean,
            'lrtc_factor_std': sin_std,
            'leakage_present': sin_std > 0.1 * np.abs(sin_mean) if sin_mean != 0 else False,
        }


# ============================================================================
# MODULE 2: TRANSPORT KERNEL (USADEL/HOUZET-BUZDIN)
# ============================================================================


class UsadelKernel:
    """
    Computes local supercurrent density using Houzet-Buzdin analytical result
    for S/F'/F/F'/S junctions in diffusive limit.

    The LRTC amplitude is: j_c ~ sin(theta_L)*sin(theta_R)*exp(-d_F/xi_T)

    Reference: Houzet & Buzdin, PRB 76, 060504(R) (2007)
    """
    
    def __init__(self, materials: MaterialParameters, 
                 geometry: DiffusiveJunctionGeometry):
        """
        Initialize Usadel kernel.

        Parameters
        ----------
        materials : MaterialParameters
            Material properties
        geometry : DiffusiveJunctionGeometry
            Layer thicknesses
        """
        self.mat = materials
        self.geo = geometry
        self._compute_decay_factors()
    
    def _compute_decay_factors(self):
        """Pre-compute exponential decay factors for coherence."""
        # Singlet: short-range, strongly suppressed
        self.singlet_decay = np.exp(-self.geo.d_F / self.mat.xi_F_singlet)
        
        # Triplet: long-range, weakly decaying
        self.triplet_decay = np.exp(-self.geo.d_F / self.mat.xi_F_triplet)
        
        # F' layers are thin, assume perfect transmission
        self.Fprime_transmission = 1.0
    
    def local_critical_current_density(self, theta_L: float, theta_R: float,
                                       include_singlet: bool = True) -> float:
        """
        Compute local critical current density j_c based on non-collinearity.

        Following Houzet-Buzdin:
        j_c ~ sin(theta_L)*sin(theta_R)*exp(-d_F/xi_T) [LRTC, dominant]
            + cos(theta_L)*cos(theta_R)*exp(-d_F/xi_S) [singlet, suppressed]

        Parameters
        ----------
        theta_L : float
            Angle between left F' and bulk F (radians)
        theta_R : float
            Angle between right F' and bulk F (radians)
        include_singlet : bool
            Whether to include suppressed singlet term

        Returns
        -------
        float
            Local critical current density (normalized to j_c0)
        """
        # LRTC (triplet) contribution - DOMINANT
        j_triplet = np.sin(theta_L) * np.sin(theta_R) * self.triplet_decay
        
        # Singlet contribution - strongly suppressed
        if include_singlet:
            j_singlet = np.cos(theta_L) * np.cos(theta_R) * self.singlet_decay
        else:
            j_singlet = 0.0
        
        # Both contributions add (can also implement with relative sign)
        j_c = np.abs(j_triplet) + np.abs(j_singlet)
        
        return j_c
    
    def compute_jc_profile(self, mag_config: MagneticConfiguration,
                          include_singlet: bool = True) -> np.ndarray:
        """
        Compute critical current density profile j_c(x,y) across junction.

        Parameters
        ----------
        mag_config : MagneticConfiguration
            Current magnetic state
        include_singlet : bool
            Include suppressed singlet term

        Returns
        -------
        np.ndarray
            Shape (n_grid_x, n_grid_y), j_c at each spatial point
        """
        jc_profile = np.zeros((self.geo.n_grid_x, self.geo.n_grid_y))
        
        for i, x in enumerate(self.geo.x_grid):
            for j, y in enumerate(self.geo.y_grid):
                theta_L, theta_R = mag_config.get_noncollinearity_at_position(x, y)
                jc_profile[i, j] = self.local_critical_current_density(
                    theta_L, theta_R, include_singlet=include_singlet
                )
        
        return jc_profile
    
    def spectral_leakage_diagnostic(self, mag_config: MagneticConfiguration) -> Dict:
        """
        Detailed diagnostic of spectral leakage (protection of triplet from SAF cancellation).

        Returns
        -------
        dict
            Detailed breakdown of angle and LRTC distributions
        """
        return mag_config.spectral_leakage_check()
