import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rcsj_sde.utils import hbar_over_2e
from rcsj_sde.rcsj import rcsj_solver
from rcsj_sde.junction import JosephsonJunction
import copy

def find_critical_current(jj, field_strength, effective_area, delta=0.15, sweep_direction=1, 
                         tau_max=1000, tau_points=10000):
    """
    Find critical current using RCSJ model simulation.
    
    Parameters:
    -----------
    jj : JosephsonJunction
        Josephson junction object
    field_strength : float
        Magnetic field strength (Tesla)
    effective_area : float
        Effective area of the junction (m²)
    delta : float
        Phase shift (fraction of flux quantum)
    sweep_direction : int
        Direction of field sweep: 1 for up sweep, -1 for down sweep
    tau_max : int
        Length of simulation in normalized units
    tau_points : int
        Number of time points in simulation
    
    Returns:
    --------
    float
        Critical current value for the given field
    """
    phi0 = hbar_over_2e * 2 * np.pi  # Flux quantum (Wb)
    flux = field_strength * effective_area
    
    # Calculate phase shift due to field
    phase_shift = np.pi * (flux - delta * phi0) / phi0
    
    # Create time span for simulation
    tspan = np.linspace(0, tau_max, tau_points)
    y0 = np.array([0.0, 0.0])  # Initial phase and phase velocity
    
    # Modified beta parameter for spin-orbit coupling effect
    # Different for up and down sweeps
    beta_modifier = 1.0
    if sweep_direction > 0:  # Up sweep
        beta_modifier = 1.0
    else:  # Down sweep
        beta_modifier = 1.3  # Increased damping for down sweep due to SO coupling
    
    beta = jj.beta * beta_modifier
    epsilon = jj.epsilon / beta_modifier  # Adjust noise term inversely
    
    # Create a modified junction to include phase shift from field
    # We'll modify the CPR parameters based on field
    modified_jj = copy.deepcopy(jj)
    
    # Modify a and b parameters based on field and sweep direction
    if sweep_direction > 0:  # Up sweep
        modified_jj.a = jj.a * np.abs(np.sin(phase_shift) / phase_shift) if phase_shift != 0 else jj.a
        # Standard triplet contribution
        if jj.b > 0:
            phase_shift_half = phase_shift / 2
            modified_jj.b = jj.b * np.abs(np.sin(phase_shift_half) / phase_shift_half) if phase_shift_half != 0 else jj.b
    else:  # Down sweep
        # Enhanced triplet contribution for down sweep due to spin-orbit effect
        modified_jj.a = jj.a * np.abs(np.sin(phase_shift) / phase_shift) if phase_shift != 0 else jj.a
        if jj.b > 0:
            # Spin-orbit coupling modifies the phase for half-integer contribution
            phase_shift_half = phase_shift / 2
            # Enhanced triplet contribution for down sweep (increased b parameter)
            modified_jj.b = 1.5 * jj.b * np.abs(np.sin(phase_shift_half) / phase_shift_half) if phase_shift_half != 0 else 1.5 * jj.b
    
    # Handle NaN cases
    if np.isnan(modified_jj.a):
        modified_jj.a = jj.a * 0.5
    if np.isnan(modified_jj.b):
        modified_jj.b = jj.b * 0.5
        
    # Binary search to find critical current
    i_low = 0.0
    i_high = 2.0 * jj.Ic
    i_critical = 0.0
    tolerance = jj.Ic * 0.01  # 1% tolerance
    
    while (i_high - i_low) > tolerance:
        i_test = (i_low + i_high) / 2
        
        # Run RCSJ simulation with test current
        sol = rcsj_solver(epsilon, beta, i_test/jj.Ic, modified_jj.a, modified_jj.b, y0, tspan)
        
        # Check if junction is in voltage state by measuring average phase velocity
        # Take the second half of the solution to allow for stabilization
        mid_point = len(sol) // 2
        phase_velocity = np.mean(sol[mid_point:, 1])
        
        # If average phase velocity is significant, junction is in voltage state
        if np.abs(phase_velocity) > 0.1:  # Threshold for voltage state
            i_high = i_test
        else:
            i_low = i_test
            i_critical = i_test
    
    return i_critical

def fraunhofer_pattern(jj, B_range, effective_area, d_Nb=20e-9, xi_triplet=5e-9, 
                      delta=0.15, use_rcsj_solver=True, tau_max=1000, tau_points=10000):
    """Simulate Fraunhofer pattern with spin-glass effects and direction-dependent
    spin-orbit coupling using RCSJ model.
    
    Parameters:
    -----------
    jj : JosephsonJunction
        Josephson junction object
    B_range : array
        Magnetic field values to sweep (Tesla)
    effective_area : float
        Effective area of the junction (m²)
    d_Nb : float
        Thickness of Nb layer (m)
    xi_triplet : float
        Triplet coherence length in Nb (m)
    delta : float
        Phase shift as fraction of flux quantum
    use_rcsj_solver : bool
        Whether to use RCSJ solver for accurate critical current (slow) or analytical formula (fast)
    tau_max, tau_points : int
        Simulation parameters when using RCSJ solver
    
    Returns:
    --------
    Ic_up, Ic_down : arrays
        Critical current values for up and down sweeps
    """
    phi0 = hbar_over_2e * 2 * np.pi  # Flux quantum (Wb)
    delta_phi = delta * phi0
    
    # For hysteresis effect, we'll do an up sweep and down sweep
    Ic_up = np.zeros_like(B_range)
    Ic_down = np.zeros_like(B_range)
    
    if use_rcsj_solver:
        # Use RCSJ model to find critical current (more accurate but slower)
        print("Up sweep:")
        for i, B in enumerate(tqdm(B_range)):
            Ic_up[i] = find_critical_current(jj, B, effective_area, delta, 1, tau_max, tau_points) * np.exp(-d_Nb/xi_triplet)
            
        print("Down sweep:")
        for i, B in enumerate(tqdm(B_range[::-1])):
            Ic_down[len(B_range)-i-1] = find_critical_current(jj, B, effective_area, delta, -1, tau_max, tau_points) * np.exp(-d_Nb/xi_triplet)
    else:
        # Use analytical formula (faster but less accurate for complex junctions)
        print("Using analytical formula (fast approximate solution)")
        
        # Up sweep (increasing field)
        for i, B in enumerate(B_range):
            flux = B * effective_area
            with np.errstate(divide='ignore', invalid='ignore'):
                arg = np.pi * (flux - delta_phi) / phi0
                Ic = jj.Ic * jj.a * np.abs(np.sin(arg) / arg) * np.exp(-d_Nb/xi_triplet)
                
                # Add spin-triplet contribution
                if jj.b > 0:
                    arg_half = np.pi * (flux - delta_phi) / (2 * phi0)
                    Ic_triplet = jj.Ic * jj.b * np.abs(np.sin(arg_half) / arg_half) * np.exp(-d_Nb/xi_triplet)
                    Ic += Ic_triplet
            
            Ic_up[i] = Ic if not np.isnan(Ic) else jj.Ic/2 * np.exp(-d_Nb/xi_triplet)
        
        # Down sweep (decreasing field) with enhanced spin-orbit coupling effects
        for i, B in enumerate(B_range[::-1]):
            flux = B * effective_area
            with np.errstate(divide='ignore', invalid='ignore'):
                arg = np.pi * (flux - delta_phi) / phi0
                Ic = jj.Ic * jj.a * np.abs(np.sin(arg) / arg) * np.exp(-d_Nb/xi_triplet)
                
                # Add enhanced spin-triplet contribution for down sweep
                if jj.b > 0:
                    arg_half = np.pi * (flux - delta_phi) / (2 * phi0)
                    # 1.5x stronger triplet contribution for down sweep due to SO coupling
                    Ic_triplet = jj.Ic * (1.5 * jj.b) * np.abs(np.sin(arg_half) / arg_half) * np.exp(-d_Nb/xi_triplet)
                    Ic += Ic_triplet
            
            Ic_down[len(B_range)-i-1] = Ic if not np.isnan(Ic) else jj.Ic/2 * np.exp(-d_Nb/xi_triplet)
    
    return Ic_up, Ic_down


def plot_fraunhofer(B_range, Ic_up, Ic_down, jj, effective_area, filename=None):
    """Plot Fraunhofer pattern with both up and down sweeps.
    
    Parameters:
    -----------
    B_range : array
        Magnetic field values (Tesla)
    Ic_up, Ic_down : arrays
        Critical current values for up and down sweeps
    jj : JosephsonJunction
        Junction object for normalization
    effective_area : float
        Junction effective area (m²)
    filename : str, optional
        If provided, save plot to this file
    """
    phi0 = hbar_over_2e * 2 * np.pi  # Flux quantum (Wb)
    B_max = np.max(np.abs(B_range))
    
    plt.figure(figsize=(10, 6))
    plt.plot(B_range*1e3, Ic_up/jj.Ic, 'b-', label='Upward Sweep', linewidth=1.5)
    plt.plot(B_range*1e3, Ic_down/jj.Ic, 'r--', label='Downward Sweep', linewidth=1.5)
    plt.xlabel('Magnetic Field (mT)', fontsize=12)
    plt.ylabel('Normalized Critical Current $I_c/I_{c0}$', fontsize=12)
    plt.title('Fraunhofer Pattern with Spin-Orbit Coupling and Hysteresis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add secondary axis for flux quantum units
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(np.linspace(-B_max*1e3, B_max*1e3, 7))
    ax2.set_xticklabels([f'{x*effective_area/phi0:.1f}' for x in np.linspace(-B_max, B_max, 7)])
    ax2.set_xlabel(r'$\Phi/\Phi_0$', fontsize=12)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()