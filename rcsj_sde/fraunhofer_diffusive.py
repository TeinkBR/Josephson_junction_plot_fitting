"""
Module 3: Fraunhofer pattern integration for diffusive S/F'/F/F'/S junctions.

Computes the critical current as a function of external magnetic field by:
1. Computing local j_c(x,y) from Usadel kernel
2. Adding phase shifts from external flux and domain magnetization
3. Performing coherent spatial integration
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import interpolate
from tqdm import tqdm

from .diffusive_triplet import (
    DiffusiveJunctionGeometry,
    MaterialParameters,
    MagneticConfiguration,
    UsadelKernel,
)


class FraunhoferIntegrator:
    """
    Computes Fraunhofer pattern by spatial integration of local currents
    with magnetic flux phase shifts.
    """
    
    def __init__(self, geometry: DiffusiveJunctionGeometry,
                 materials: MaterialParameters):
        """
        Initialize Fraunhofer integrator.

        Parameters
        ----------
        geometry : DiffusiveJunctionGeometry
            Junction geometry
        materials : MaterialParameters
            Material parameters
        """
        self.geo = geometry
        self.mat = materials
        self.kernel = UsadelKernel(materials, geometry)
        
        # Physical constants
        self.hbar = materials.hbar  # meV*ps
        self.phi0 = 2.067833848e-15  # Wb (flux quantum), but work in natural units
        self.g_muB = materials.g_muB  # meV/T
    
    def compute_flux_phase(self, B_ext: float, x: float, y: float,
                          mag_config: Optional[MagneticConfiguration] = None) -> float:
        """
        Compute phase accumulated from magnetic flux at position (x,y).

        phi_flux = (2*pi/Phi_0) * [B_ext . effective_area + B_mag . volume]

        For simplicity, we compute the phase shift based on external field
        and an effective magnetization contribution from the frozen spin-glass.

        Parameters
        ----------
        B_ext : float
            External magnetic field (T)
        x, y : float
            Position in junction (normalized 0-1)
        mag_config : MagneticConfiguration, optional
            Magnetic state (if provided, includes mag. domain contribution)

        Returns
        -------
        float
            Phase shift (radians)
        """
        # Phase from external field
        # Flux through effective area = B * A_eff
        # Phase = 2*pi * Phi/Phi_0, but we work in natural units where Phi_0 = 2*pi
        effective_area_local = self.geo.junction_length * self.geo.junction_width * 1e-18  # m²
        flux_quantum = 2.067833848e-15  # Wb
        
        # Flux from external field
        flux_ext = B_ext * effective_area_local  # Wb
        phi_ext = 2 * np.pi * flux_ext / flux_quantum  # radians
        
        # For uniform external field, phase is approximately proportional to B
        # Normalize to typical values
        phi_ext_normalized = phi_ext / (2 * np.pi)  # dimensionless
        
        # Phase from magnetization (if mag_config provided)
        phi_mag = 0.0
        if mag_config is not None:
            # Get average magnetization at this position
            theta_L, theta_R = mag_config.get_noncollinearity_at_position(x, y)
            
            # The frozen magnetization creates residual flux - related to phase shift delta
            # Simplified: use the residual angle as phase contribution
            # More rigorous approach would integrate magnetization over volume
            phi_mag = 0.1 * (theta_L + theta_R) / 2  # empirical relation
        
        return phi_ext_normalized + phi_mag
    
    def compute_critical_current(self, B_ext: float,
                                mag_config: MagneticConfiguration,
                                include_singlet: bool = True) -> float:
        """
        Compute total critical current from Fraunhofer integration.

        I_c(B) = |integral integral j_c(x,y) * exp(i*phi_flux(x,y)) dxdy|

        Parameters
        ----------
        B_ext : float
            External magnetic field (T)
        mag_config : MagneticConfiguration
            Current magnetic state
        include_singlet : bool
            Include suppressed singlet contribution

        Returns
        -------
        float
            Total critical current (normalized to Ic0)
        """
        # Compute j_c profile
        jc_profile = self.kernel.compute_jc_profile(mag_config, include_singlet)
        
        # Compute complex sum of current contributions with phase shifts
        I_real = 0.0
        I_imag = 0.0
        
        for i, x in enumerate(self.geo.x_grid):
            for j, y in enumerate(self.geo.y_grid):
                jc = jc_profile[i, j]
                phi = self.compute_flux_phase(B_ext, x, y, mag_config)
                
                # Add to complex sum
                I_real += jc * np.cos(phi) * self.geo.dx * self.geo.dy
                I_imag += jc * np.sin(phi) * self.geo.dx * self.geo.dy
        
        # Magnitude
        I_c = np.sqrt(I_real**2 + I_imag**2)
        
        return I_c
    
    def compute_fraunhofer_pattern(self, B_range: np.ndarray,
                                  mag_configs: Optional[Dict[float, MagneticConfiguration]] = None,
                                  include_singlet: bool = True,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute critical current vs magnetic field (Fraunhofer pattern).

        Parameters
        ----------
        B_range : np.ndarray
            Array of magnetic field values (T)
        mag_configs : dict, optional
            Pre-computed magnetic configurations for each B value.
            If None, configurations are generated on-the-fly.
        include_singlet : bool
            Include suppressed singlet term
        verbose : bool
            Print progress

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (B_range, I_c(B_range)) - critical currents
        """
        Ic_values = np.zeros_like(B_range)
        
        iterator = tqdm(enumerate(B_range), total=len(B_range)) if verbose else enumerate(B_range)
        
        for idx, B in iterator:
            if mag_configs is not None and B in mag_configs:
                mag_config = mag_configs[B]
            else:
                # Create fresh config - would need to manage evolution elsewhere
                mag_config = MagneticConfiguration(self.geo)
                mag_config.evolve_field_step(B)
            
            Ic_values[idx] = self.compute_critical_current(B, mag_config, include_singlet)
        
        return B_range, Ic_values
    
    def analytical_fraunhofer_singlet(self, B_range: np.ndarray, Ic0: float) -> np.ndarray:
        """
        Standard sinc Fraunhofer pattern for reference (pure singlet).

        I_c(B) = Ic0 * |sinc(pi*Phi/Phi_0)|

        Parameters
        ----------
        B_range : np.ndarray
            Magnetic field values (T)
        Ic0 : float
            Maximum critical current

        Returns
        -------
        np.ndarray
            Fraunhofer pattern
        """
        # Compute normalized flux
        flux_quantum = 2.067833848e-15  # Wb
        effective_area = self.geo.effective_area
        flux = B_range * effective_area  # Wb
        
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = np.pi * flux / flux_quantum
            pattern = np.abs(np.sinc(arg / np.pi))
        
        pattern = np.nan_to_num(pattern, nan=1.0)
        return Ic0 * pattern


class FieldSweepSimulation:
    """
    Full simulation of Fraunhofer pattern during field sweep up and down.
    Manages the history-dependent evolution of spin-glass layers.
    """
    
    def __init__(self, geometry: DiffusiveJunctionGeometry,
                 materials: MaterialParameters):
        """
        Initialize field sweep simulation.

        Parameters
        ----------
        geometry : DiffusiveJunctionGeometry
            Junction geometry
        materials : MaterialParameters
            Material parameters
        """
        self.geo = geometry
        self.mat = materials
        self.integrator = FraunhoferIntegrator(geometry, materials)
        self.mag_config = MagneticConfiguration(geometry)
    
    def run_sweep(self, B_min: float, B_max: float, n_points: int = 50,
                  n_relax_steps: int = 500, T_eff: float = 0.1,
                  equilibrate_initial: bool = True,
                  include_singlet: bool = True,
                  verbose: bool = True) -> Dict:
        """
        Run full field sweep (up and down) with history-dependent evolution.

        Parameters
        ----------
        B_min : float
            Minimum field (T)
        B_max : float
            Maximum field (T)
        n_points : int
            Number of field points per sweep
        n_relax_steps : int
            Metropolis relaxation steps per field point
        T_eff : float
            Effective temperature for Metropolis
        equilibrate_initial : bool
            Fully equilibrate at B_min before starting sweep
        include_singlet : bool
            Include suppressed singlet term
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Results containing B_up, Ic_up, B_down, Ic_down, and diagnostics
        """
        # Initial equilibration at large negative field
        if equilibrate_initial:
            if verbose:
                print("Initial equilibration...")
            for _ in range(5):  # Multiple equilibration sweeps
                self.mag_config.evolve_field_step(B_min, n_relax_steps=n_relax_steps*5, T_eff=T_eff)
        
        # Up sweep
        B_up = np.linspace(B_min, B_max, n_points)
        Ic_up = np.zeros_like(B_up)
        diagnostics_up = []
        
        if verbose:
            print("Up sweep...")
        for idx, B in enumerate(tqdm(B_up, disable=not verbose)):
            self.mag_config.evolve_field_step(B, n_relax_steps=n_relax_steps, T_eff=T_eff)
            Ic_up[idx] = self.integrator.compute_critical_current(
                B, self.mag_config, include_singlet
            )
            diagnostics_up.append(self.mag_config.spectral_leakage_check())
        
        # Down sweep
        B_down = np.linspace(B_max, B_min, n_points)
        Ic_down = np.zeros_like(B_down)
        diagnostics_down = []
        
        if verbose:
            print("Down sweep...")
        for idx, B in enumerate(tqdm(B_down, disable=not verbose)):
            self.mag_config.evolve_field_step(B, n_relax_steps=n_relax_steps, T_eff=T_eff)
            Ic_down[idx] = self.integrator.compute_critical_current(
                B, self.mag_config, include_singlet
            )
            diagnostics_down.append(self.mag_config.spectral_leakage_check())
        
        return {
            'B_up': B_up,
            'Ic_up': Ic_up,
            'B_down': B_down,
            'Ic_down': Ic_down,
            'diagnostics_up': diagnostics_up,
            'diagnostics_down': diagnostics_down,
        }


def extract_li_fit_parameters(B_range: np.ndarray, Ic_data: np.ndarray) -> Dict[str, float]:
    """
    Extract parameters from Li's empirical fit function:
    I_c(Phi) = I_c0 * exp(-d_Nb/xi_T) * |J_1(...) / (...)|

    Parameters
    ----------
    B_range : np.ndarray
        Magnetic field values
    Ic_data : np.ndarray
        Critical current data

    Returns
    -------
    dict
        Fitted parameters: phase_shift, coherence_length, amplitude
    """
    # Find maximum Ic (at B=0 approximately)
    idx_max = np.argmax(Ic_data)
    Ic0_fit = Ic_data[idx_max]
    B_max_fit = B_range[idx_max]
    
    # Estimate phase shift (offset from B=0)
    phase_shift = B_max_fit
    
    # For coherence length, fit to decay - not implemented in detail here
    # Would require more sophisticated fitting
    
    return {
        'Ic0': Ic0_fit,
        'phase_shift': phase_shift,
        'B_at_max': B_max_fit,
    }


# Alias for compatibility
FraunhoferCalculator = FraunhoferIntegrator
