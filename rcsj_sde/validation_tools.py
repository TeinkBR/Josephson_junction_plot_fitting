"""
Module 5: Validation & Analysis Tools

Utilities for comparing simulation results against Li's experimental
fit function and extracting physical parameters.

References:
- Li et al. (2024) - Experimental data and fit function
- Houzet & Buzdin (2007) - Theoretical predictions
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.optimize import curve_fit
from scipy.special import j1  # Bessel function for Fraunhofer


class FraunhoferFitFunction:
    """
    Li's empirical fit function for the Fraunhofer pattern.
    
    I_c(Phi) = I_{c0} * exp(-d_Nb/xi_T) * |J_1(pi*(Phi-delta)/Phi_0) / (pi*(Phi-delta)/Phi_0)|
    
    Parameters
    ----------
    I_c0 : float
        Maximum critical current (A)
    xi_T : float
        Triplet coherence length (nm)
    d_Nb : float
        Superconductor thickness (nm)
    delta : float
        Phase shift from residual magnetization (rad)
    """
    
    def __init__(self, I_c0: float = 1.5e-6, xi_T: float = 20.0,
                 d_Nb: float = 100.0, delta: float = 0.0):
        self.I_c0 = I_c0
        self.xi_T = xi_T
        self.d_Nb = d_Nb
        self.delta = delta
        
        # Flux quantum
        self.Phi_0 = 2.067833848e-15  # Wb (flux quantum)
    
    def __call__(self, Phi: np.ndarray) -> np.ndarray:
        """
        Evaluate fit function at given flux values.
        
        Parameters
        ----------
        Phi : np.ndarray
            Magnetic flux (Wb)
            
        Returns
        -------
        np.ndarray
            Critical current (A)
        """
        # Normalized flux difference (accounting for phase shift)
        x = np.pi * (Phi - self.delta * self.Phi_0) / self.Phi_0
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            fraunhofer_pattern = np.abs(j1(x) / x)
        fraunhofer_pattern = np.nan_to_num(fraunhofer_pattern, nan=1.0)
        
        # Exponential decay with triplet coherence length
        decay_factor = np.exp(-self.d_Nb / self.xi_T)
        
        return self.I_c0 * decay_factor * fraunhofer_pattern
    
    def to_dict(self) -> dict:
        """Return fit parameters as dictionary."""
        return {
            'I_c0': self.I_c0,
            'xi_T': self.xi_T,
            'd_Nb': self.d_Nb,
            'delta': self.delta
        }
    
    @classmethod
    def from_dict(cls, params: dict):
        """Create instance from parameter dictionary."""
        return cls(**params)


def convert_field_to_flux(B: np.ndarray, junction_area: float) -> np.ndarray:
    """
    Convert magnetic field (T) to flux (Wb).
    
    Phi = B * A
    
    Parameters
    ----------
    B : np.ndarray
        Magnetic field (T)
    junction_area : float
        Effective junction area (m²)
        
    Returns
    -------
    np.ndarray
        Magnetic flux (Wb)
    """
    return B * junction_area


def convert_flux_to_field(Phi: np.ndarray, junction_area: float) -> np.ndarray:
    """
    Convert flux (Wb) to magnetic field (T).
    
    B = Phi / A
    """
    return Phi / junction_area


def fit_fraunhofer_to_simulation(B_range: np.ndarray, Ic_sim: np.ndarray,
                                 junction_area: float,
                                 initial_guess: Optional[Dict] = None) -> FraunhoferFitFunction:
    """
    Fit Li's functional form to simulation data.
    
    Extracts the phase shift delta (from spin-glass magnetization memory)
    and triplet coherence length xi_T.
    
    Parameters
    ----------
    B_range : np.ndarray
        Magnetic field values (T)
    Ic_sim : np.ndarray
        Simulated critical currents (A)
    junction_area : float
        Effective junction area (m²)
    initial_guess : dict, optional
        Initial guess for fit parameters
        
    Returns
    -------
    FraunhoferFitFunction
        Fitted function with extracted parameters
    """
    # Convert to flux
    Phi = convert_field_to_flux(B_range, junction_area)
    
    # Initial guess
    if initial_guess is None:
        I_c0_guess = np.max(Ic_sim)
        xi_T_guess = 20.0  # nm
        d_Nb_guess = 100.0  # nm
        delta_guess = 0.0
    else:
        I_c0_guess = initial_guess.get('I_c0', np.max(Ic_sim))
        xi_T_guess = initial_guess.get('xi_T', 20.0)
        d_Nb_guess = initial_guess.get('d_Nb', 100.0)
        delta_guess = initial_guess.get('delta', 0.0)
    
    # Define function for scipy curve_fit
    def fit_target(Phi, I_c0, xi_T, delta):
        """Fit with fixed d_Nb."""
        fit_func = FraunhoferFitFunction(I_c0=I_c0, xi_T=xi_T,
                                        d_Nb=d_Nb_guess, delta=delta)
        return fit_func(Phi)
    
    # Perform fit
    try:
        popt, _ = curve_fit(
            fit_target, Phi, Ic_sim,
            p0=[I_c0_guess, xi_T_guess, delta_guess],
            maxfev=10000,
            bounds=(
                [1e-8, 1.0, -np.pi],
                [1e-4, 1000.0, np.pi]
            )
        )
        
        I_c0_fit, xi_T_fit, delta_fit = popt
        
    except RuntimeError:
        print("Fit failed, returning initial guess")
        I_c0_fit = I_c0_guess
        xi_T_fit = xi_T_guess
        delta_fit = delta_guess
    
    return FraunhoferFitFunction(I_c0=I_c0_fit, xi_T=xi_T_fit,
                                d_Nb=d_Nb_guess, delta=delta_fit)


def spectral_leakage_analysis(mag_config) -> Dict[str, float]:
    """
    Quantify "spectral leakage" - degree to which spin-glass disorder
    prevents complete suppression in an SAF configuration.
    
    Returns
    -------
    dict
        Diagnostic quantities including:
        - 'lrtc_mean': Average LRTC factor
        - 'lrtc_std': Standard deviation (measure of disorder)
        - 'leakage_ratio': Std/Mean ratio (0 = perfect order, large = disorder)
        - 'disorder_protected': Boolean, whether disorder protects triplet current
    """
    lrtc_mean = mag_config.get_average_lrtc_factor()
    
    # Compute per-point LRTC factors
    lrtc_values = []
    for i in range(mag_config.n_sg_x):
        for j in range(mag_config.n_sg_y):
            theta_L, theta_R = mag_config.get_noncollinearity_angles(i, j)
            lrtc_values.append(np.sin(theta_L) * np.sin(theta_R))
    
    lrtc_std = np.std(lrtc_values)
    
    # Leakage ratio (dimensionless measure of disorder impact)
    leakage_ratio = lrtc_std / (np.abs(lrtc_mean) + 1e-10)
    
    # If disorder is significant compared to mean, it "protects" the current
    disorder_protected = leakage_ratio > 0.2  # Threshold
    
    return {
        'lrtc_mean': float(lrtc_mean),
        'lrtc_std': float(lrtc_std),
        'leakage_ratio': float(leakage_ratio),
        'disorder_protected': bool(disorder_protected),
        'lrtc_values': lrtc_values
    }


def compute_fraunhofer_metrics(B_range: np.ndarray, Ic: np.ndarray,
                              junction_area: float) -> Dict[str, float]:
    """
    Compute standard Fraunhofer pattern metrics.
    
    Parameters
    ----------
    B_range : np.ndarray
        Magnetic field range (T)
    Ic : np.ndarray
        Critical current values (A)
    junction_area : float
        Junction area (m²)
        
    Returns
    -------
    dict
        Metrics including:
        - 'Ic_max': Maximum critical current
        - 'Ic_min': Minimum critical current
        - 'fraunhofer_period': Flux period in units of Phi_0
        - 'fraunhofer_contrast': (Ic_max - Ic_min) / Ic_max
        - 'pattern_symmetry': Measure of left-right asymmetry
    """
    Phi_0 = 2.067833848e-15  # Wb
    
    Ic_max = np.max(Ic)
    Ic_min = np.min(Ic)
    
    # Find flux period (distance between consecutive minima)
    # Convert to flux
    Phi = convert_field_to_flux(B_range, junction_area)
    
    # Find minima
    min_indices = []
    for i in range(1, len(Ic) - 1):
        if Ic[i] < Ic[i-1] and Ic[i] < Ic[i+1]:
            min_indices.append(i)
    
    if len(min_indices) >= 2:
        flux_periods = np.diff([Phi[i] for i in min_indices])
        fraunhofer_period = np.mean(flux_periods) / Phi_0
    else:
        fraunhofer_period = np.nan
    
    # Contrast
    contrast = (Ic_max - Ic_min) / (Ic_max + 1e-10)
    
    # Symmetry (compute asymmetry around midpoint)
    mid_idx = len(Ic) // 2
    left_half = Ic[:mid_idx]
    right_half = Ic[mid_idx:][::-1]
    
    if len(left_half) == len(right_half):
        asymmetry = np.mean(np.abs(left_half - right_half)) / Ic_max
    else:
        asymmetry = np.nan
    
    return {
        'Ic_max': float(Ic_max),
        'Ic_min': float(Ic_min),
        'fraunhofer_period': float(fraunhofer_period),
        'contrast': float(contrast),
        'asymmetry': float(asymmetry)
    }


def compare_simulation_to_experiment(B_range_sim: np.ndarray,
                                     Ic_sim: np.ndarray,
                                     B_range_exp: np.ndarray,
                                     Ic_exp: np.ndarray,
                                     junction_area: float) -> Dict:
    """
    Compare simulation results to experimental data.
    
    Computes goodness-of-fit metrics and extracts fitted parameters.
    
    Parameters
    ----------
    B_range_sim : np.ndarray
        Simulated field range (T)
    Ic_sim : np.ndarray
        Simulated critical current (A)
    B_range_exp : np.ndarray
        Experimental field range (T)
    Ic_exp : np.ndarray
        Experimental critical current (A)
    junction_area : float
        Junction area (m²)
        
    Returns
    -------
    dict
        Comparison results including:
        - 'fit_function': Fitted FraunhoferFitFunction
        - 'chi_squared': χ² metric
        - 'r_squared': R² coefficient
        - 'residuals': Simulation minus experiment (interpolated)
        - 'extracted_params': Dictionary of fitted parameters
    """
    # Fit to simulation data
    fit_func = fit_fraunhofer_to_simulation(B_range_sim, Ic_sim, junction_area)
    
    # Evaluate fit at experimental points (interpolate)
    Ic_fit_at_exp = fit_func(convert_field_to_flux(B_range_exp, junction_area))
    
    # Residuals
    residuals = Ic_exp - Ic_fit_at_exp
    
    # Chi-squared
    chi_sq = np.sum(residuals**2 / (Ic_exp + 1e-12))
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Ic_exp - np.mean(Ic_exp))**2)
    r_sq = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    return {
        'fit_function': fit_func,
        'chi_squared': float(chi_sq),
        'r_squared': float(r_sq),
        'residuals': residuals,
        'extracted_params': fit_func.to_dict()
    }


def hysteresis_analysis(B_range: np.ndarray, Ic_up: np.ndarray,
                       Ic_down: np.ndarray) -> Dict[str, float]:
    """
    Analyze hysteretic behavior in Fraunhofer pattern.
    
    Quantifies the difference between up-sweep and down-sweep.
    
    Parameters
    ----------
    B_range : np.ndarray
        Magnetic field range (T)
    Ic_up : np.ndarray
        Critical current (up sweep)
    Ic_down : np.ndarray
        Critical current (down sweep)
        
    Returns
    -------
    dict
        Hysteresis metrics including:
        - 'hysteresis_loss': Area between up/down curves
        - 'max_difference': Maximum point-wise difference
        - 'rms_difference': RMS difference
        - 'is_hysteretic': Boolean, significant hysteresis present
    """
    # Area between curves (hysteresis loss)
    dB = np.mean(np.diff(B_range))
    hysteresis_loss = np.sum(np.abs(Ic_up - Ic_down)) * dB
    
    # Max difference
    max_diff = np.max(np.abs(Ic_up - Ic_down))
    
    # RMS difference
    rms_diff = np.sqrt(np.mean((Ic_up - Ic_down)**2))
    
    # Threshold for significant hysteresis
    avg_Ic = np.mean((Ic_up + Ic_down) / 2)
    is_hysteretic = max_diff > 0.05 * avg_Ic
    
    return {
        'hysteresis_loss': float(hysteresis_loss),
        'max_difference': float(max_diff),
        'rms_difference': float(rms_diff),
        'is_hysteretic': bool(is_hysteretic)
    }
