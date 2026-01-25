"""Test the analytical model parameters."""
import numpy as np

HBAR = 1.054571817e-34
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19

T = 4.2

# For triplet component, use EFFECTIVE diffusion that accounts for
# the long-range nature of the equal-spin triplet
# In practice, this is material-dependent
D_triplet = 5e-4  # Effective for triplet (higher than singlet in F)

# Triplet coherence length  
xi_T = np.sqrt(HBAR * D_triplet / (np.pi * K_B * T))
print(f'xi_T = {xi_T*1e9:.1f} nm')

# Layer thicknesses
d_Cu = 4e-9
d_Cr = 10e-9  # 2 x 5nm
d_Fe = 3e-9

# Spin mixing - increase for better triplet generation
theta_mix = 0.7  # ~40 degrees
triplet_eff = np.sin(2*theta_mix)
print(f'triplet efficiency = {triplet_eff:.3f}')

xi_N = np.sqrt(HBAR * 1e-2 / (np.pi * K_B * T))
print(f'xi_N = {xi_N*1e9:.1f} nm')

# Suppression factor
suppression = triplet_eff**2 * np.exp(-d_Cu / xi_N) * np.exp(-d_Fe / xi_T) * np.exp(-d_Cr / (0.3*xi_T))
print(f'suppression = {suppression:.6f}')

# Now compute Ic
Delta_T = 1.44e-3 * E_CHARGE
width = 2.5e-6
d_barrier = d_Cu + d_Cr + d_Fe
A_junction = width**2

# Resistance: typical values for such junctions
# R_N * A ~ 1-10 fΩ·m² for transparent interfaces
# Need to match ~25 uA → use R_A that gives this
R_A = 60e-12  # Ω·m² (specific junction resistance-area)
R_N = R_A / A_junction
print(f'R_N = {R_N*1e3:.3f} mΩ')

Ic0_raw = (np.pi * Delta_T / (2 * E_CHARGE * R_N)) * np.tanh(Delta_T / (2 * K_B * T))
print(f'Ic0_raw = {Ic0_raw*1e3:.2f} mA')

Ic0 = Ic0_raw * suppression
print(f'Ic0 = {Ic0*1e6:.2f} uA')

# Target: ~25 uA
print(f'\nTarget Ic0 = 25 uA')
print(f'Current ratio = {Ic0*1e6/25:.2f}')
