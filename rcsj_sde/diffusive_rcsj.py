"""
Module 4: RCSJ Solver for Diffusive Spin-Triplet Junctions

Integrates the Usadel-derived critical current I_c(B) with the RCSJ
dynamics to simulate voltage-current characteristics under varying
magnetic fields.

References:
- Stewart & McCumber (1968) - RCSJ model
- Houzet & Buzdin (2007) - Diffusive triplet transport
"""

import numpy as np
from typing import Tuple, Optional, Callable
import numba


class DiffusiveRCSJJunction:
    """
    RCSJ junction model with field-dependent critical current from
    Usadel/Houzet-Buzdin calculations.
    
    The key innovation is that I_c is not constant but depends on:
    - Applied magnetic field B_ext
    - History of the field sweep (through spin-glass memory)
    - Local non-collinearity in the S/F'/F/F'/S stack
    
    Parameters
    ----------
    R : float
        Shunt resistance (Ω)
    C : float
        Junction capacitance (F)
    T : float
        Temperature (K)
    Ic_func : Callable
        Function I_c = Ic_func(B_ext) returning critical current
    """
    
    def __init__(self, R: float, C: float, T: float,
                 Ic_func: Optional[Callable] = None):
        """
        Initialize RCSJ junction.
        
        Parameters
        ----------
        R : float
            Shunt resistance (Ω), typical ~100 Ω
        C : float
            Junction capacitance (F), typical ~1 pF
        T : float
            Temperature (K)
        Ic_func : Callable, optional
            Function mapping B_ext (T) to I_c (A)
            If None, uses placeholder constant value
        """
        self.R = R
        self.C = C
        self.T = T
        self.Ic_func = Ic_func if Ic_func is not None else lambda B: 1.5e-6
        
        # Physical constants
        self.hbar = 1.054571817e-34  # J*s
        self.e = 1.602176634e-19     # C
        self.kB = 1.380649e-23       # J/K
        self.hbar_over_2e = self.hbar / (2 * self.e)  # Wb (flux quantum/2pi)
        
        # Derived normalized parameters
        # Stewart-McCumber parameter beta = 2*pi*I_c*R^2*C/Phi_0
        # This will be computed dynamically based on B-dependent Ic
        self.beta = None
        self.epsilon = None  # Thermal noise parameter
        self._update_params()
    
    def _update_params(self, Ic: Optional[float] = None):
        """
        Update normalized RCSJ parameters based on critical current.
        
        Parameters
        ----------
        Ic : float, optional
            Critical current (A). If None, uses maximum from Ic_func.
        """
        if Ic is None:
            Ic = self.Ic_func(0.0)  # Use zero-field value as reference
        
        self.Ic = Ic
        
        # Stewart-McCumber parameter
        # beta = 2*pi * I_c * R^2 * C / Phi_0
        # where Phi_0 = h/(2e) = 2*pi * hbar_over_2e
        phi0 = 2 * np.pi * self.hbar_over_2e
        self.beta = (2 * np.pi * Ic * self.R**2 * self.C) / phi0
        
        # Thermal noise parameter (Gaussian white noise amplitude)
        # epsilon = sigma/beta where sigma^2 = 2*pi * k_B * T / (Phi_0 * I_c * R) (from fluctuation-dissipation)
        sigma_squared = (2 * np.pi * self.kB * self.T) / (phi0 * Ic * self.R)
        self.epsilon = np.sqrt(sigma_squared) / self.beta if self.beta > 0 else 0.0
        
        # Characteristic voltage (for normalization)
        self.V_c = Ic * self.R
        
        # Characteristic time scale
        self.tau_c = phi0 / self.V_c  # Dimensionless time scale
    
    def normalized_current_phase_relation(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute normalized supercurrent from phase difference.
        
        For triplet-dominated transport:
        I_s(phi) = I_c * sin(phi)
        
        (No second harmonic term for pure triplet case)
        
        Parameters
        ----------
        phi : np.ndarray
            Phase difference (radians)
            
        Returns
        -------
        np.ndarray
            Normalized supercurrent i = I_s / I_c
        """
        return np.sin(phi)
    
    def solve_overdamped(self, B_ext: float, I_dc_range: np.ndarray,
                        tau_max: float = 100.0, tau_points: int = 5000,
                        transient_fraction: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve RCSJ equation in overdamped limit (beta*c >> 1).
        
        In overdamped limit, the capacitive term is negligible:
        0 = I_dc - I_s(phi) - (V/R)
        
        This simplifies to: dphi/dtau = V/(I_c * R * omega_c)
        
        Parameters
        ----------
        B_ext : float
            Magnetic field (T)
        I_dc_range : np.ndarray
            DC current bias range (A)
        tau_max : float
            Maximum integration time (normalized units)
        tau_points : int
            Number of time points
        transient_fraction : float
            Fraction of time to discard as transient (0.0-1.0)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (I_dc_range, V_avg) where V_avg is time-averaged voltage
        """
        # Update critical current for this field
        Ic_B = self.Ic_func(B_ext)
        self._update_params(Ic_B)
        
        V = np.zeros_like(I_dc_range, dtype=float)
        tau = np.linspace(0, tau_max, tau_points)
        dt = tau[1] - tau[0]
        n_transient = int(transient_fraction * tau_points)
        
        # Normalized DC current
        i_dc_norm = I_dc_range / Ic_B
        
        for idx, i_dc in enumerate(i_dc_norm):
            # Initial condition
            phi = 0.0
            dphi_dtau = 0.0
            
            # Time evolution
            phi_history = np.zeros(tau_points)
            
            for n in range(tau_points):
                # RCSJ equation: beta * d2phi/dtau2 + dphi/dtau + sin(phi) = i_dc
                # Overdamped: dphi/dtau ~ i_dc - sin(phi)
                
                i_super = np.sin(phi)
                dphi_dtau = i_dc - i_super  # In overdamped limit
                phi += dphi_dtau * dt
                
                # Keep phase in reasonable range
                phi = np.arctan2(np.sin(phi), np.cos(phi))
                phi_history[n] = phi
            
            # Time-averaged voltage (from average dphi/dtau)
            phi_unwrapped = np.unwrap(phi_history)
            dphidt_avg = np.mean(phi_unwrapped[n_transient:] - phi_unwrapped[:-len(phi_unwrapped)+n_transient]) / (dt * (tau_points - n_transient))
            V[idx] = Ic_B * self.R * dphidt_avg / (2 * np.pi)
        
        return I_dc_range, V
    
    def solve_sde(self, B_ext: float, I_dc_range: np.ndarray,
                 tau_max: float = 100.0, tau_points: int = 5000,
                 transient_fraction: float = 0.5,
                 noise_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve full RCSJ stochastic differential equation with thermal noise.
        
        beta * d2phi/dtau2 + dphi/dtau + sin(phi) = i_dc + sigma * dW/dtau
        
        Uses Milstein scheme for better numerical stability.
        
        Parameters
        ----------
        B_ext : float
            Magnetic field (T)
        I_dc_range : np.ndarray
            DC current bias range (A)
        tau_max : float
            Maximum integration time (normalized)
        tau_points : int
            Number of time points
        transient_fraction : float
            Fraction of time to discard as transient
        noise_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (I_dc_range, V_avg) where V_avg is time-averaged voltage
        """
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        # Update critical current
        Ic_B = self.Ic_func(B_ext)
        self._update_params(Ic_B)
        
        V = np.zeros_like(I_dc_range, dtype=float)
        tau = np.linspace(0, tau_max, tau_points)
        dt = tau[1] - tau[0]
        n_transient = int(transient_fraction * tau_points)
        
        i_dc_norm = I_dc_range / Ic_B
        
        for idx, i_dc in enumerate(i_dc_norm):
            # State vector: [phi, dphi/dtau]
            y = np.array([0.0, 0.0])
            y_history = np.zeros((tau_points, 2))
            
            # Generate noise
            dW = np.random.normal(0, np.sqrt(dt), tau_points)
            
            # Milstein integration
            for n in range(tau_points):
                phi, dphi = y
                
                # Drift term
                f = np.array([dphi, (1/self.beta) * (i_dc - np.sin(phi) - dphi)])
                
                # Diffusion term (additive noise on velocity)
                g = np.array([0, self.epsilon])
                
                # Milstein step
                y = y + f * dt + g * dW[n]
                
                # Keep phase wrapped
                y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))
                
                y_history[n] = y
            
            # Time-averaged voltage from average dphi/dtau
            dphidt_avg = np.mean(y_history[n_transient:, 1])
            V[idx] = Ic_B * self.R * dphidt_avg / (2 * np.pi)
        
        return I_dc_range, V
    
    def extract_critical_current_from_iv(self, B_ext: float,
                                         I_dc_range: np.ndarray,
                                         V_measured: np.ndarray,
                                         V_threshold: float = 1e-6) -> float:
        """
        Extract critical current from measured I-V curve.
        
        Critical current is defined as the current where V exceeds threshold.
        
        Parameters
        ----------
        B_ext : float
            Magnetic field (T) - for reference only
        I_dc_range : np.ndarray
            DC current bias values (A)
        V_measured : np.ndarray
            Measured voltage values (V)
        V_threshold : float
            Voltage threshold to define switching (V)
            
        Returns
        -------
        float
            Critical current (A)
        """
        # Find first index where |V| exceeds threshold
        idx = np.where(np.abs(V_measured) > V_threshold)[0]
        
        if len(idx) > 0:
            return I_dc_range[idx[0]]
        else:
            return I_dc_range[-1]  # No switching detected


class FieldSweepIVSimulator:
    """
    Simulate I-V characteristics across a range of magnetic fields.
    
    This coordinator class links the Fraunhofer calculation with the
    RCSJ solver to produce the full B-dependent I-V map.
    """
    
    def __init__(self, junction: DiffusiveRCSJJunction,
                 fraunhofer_func: Callable):
        """
        Parameters
        ----------
        junction : DiffusiveRCSJJunction
            RCSJ junction instance
        fraunhofer_func : Callable
            Function computing Fraunhofer pattern I_c(B)
        """
        self.junction = junction
        self.fraunhofer_func = fraunhofer_func
        
        # Update junction's I_c function to use Fraunhofer result
        self.junction.Ic_func = fraunhofer_func
    
    def simulate_iv_map(self, B_range: np.ndarray,
                       I_dc_range: np.ndarray,
                       solver: str = 'overdamped',
                       tau_max: float = 100.0,
                       tau_points: int = 5000,
                       progress: bool = True) -> dict:
        """
        Compute I-V characteristics for all field values.
        
        Parameters
        ----------
        B_range : np.ndarray
            Magnetic field values (T)
        I_dc_range : np.ndarray
            DC current range (A)
        solver : str
            'overdamped' or 'sde' (with thermal noise)
        tau_max : float
            Integration time (normalized)
        tau_points : int
            Number of time points
        progress : bool
            Whether to show progress bar
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'B_range': Applied magnetic field values
            - 'I_dc_range': DC current values
            - 'V_map': 2D array of voltages V[B_idx, I_idx]
            - 'Ic_extracted': Critical current values at each field
        """
        n_B = len(B_range)
        n_I = len(I_dc_range)
        
        V_map = np.zeros((n_B, n_I))
        Ic_extracted = np.zeros(n_B)
        
        iterator = B_range if not progress else tqdm(B_range, desc='Field sweep')
        
        for b_idx, B in enumerate(iterator):
            if solver == 'overdamped':
                I_dc, V = self.junction.solve_overdamped(B, I_dc_range, tau_max, tau_points)
            elif solver == 'sde':
                I_dc, V = self.junction.solve_sde(B, I_dc_range, tau_max, tau_points)
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            V_map[b_idx, :] = V
            Ic_extracted[b_idx] = self.junction.extract_critical_current_from_iv(B, I_dc, V)
        
        return {
            'B_range': B_range,
            'I_dc_range': I_dc_range,
            'V_map': V_map,
            'Ic_extracted': Ic_extracted
        }


# Numba-compiled helper for fast RCSJ integration
@numba.njit
def _rcsj_overdamped_step(phi: float, dphi: float, i_dc: float,
                          beta: float, dt: float) -> Tuple[float, float]:
    """Single step of overdamped RCSJ integration."""
    i_super = np.sin(phi)
    dphi_new = i_dc - i_super - dphi / beta
    phi_new = phi + dphi_new * dt
    return phi_new, dphi_new
