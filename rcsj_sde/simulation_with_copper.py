# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from rcsj_sde.junction import JosephsonJunction
from rcsj_sde.fraunhofer import fraunhofer_pattern
from rcsj_sde.utils import hbar_over_2e 


# Define your junction parameters based on your multilayer structure
Ic = 5.7e-07  # Critical current - adjust based on your junction (A)
R = 9         # Normal state resistance - adjust based on your junction (Ohms)
C = 8.8e-12   # Junction capacitance - adjust based on your junction (F)
T = 0.25      # Temperature (K)

# For a complex multilayer junction, you may need to adjust these parameters
# The Nb/Al layers will contribute to the superconducting properties
# The Fe layer introduces magnetic properties that affect the CPR
a = 0.8       # Weight of sin(φ) term in CPR - adjust for magnetic junction
b = 0.2       # Weight of sin(φ/2) term in CPR - adjust for magnetic junction

# Create the junction object
jj = JosephsonJunction(Ic=Ic, a=a, b=b, R=R, C=C, T=T)

# Define magnetic field range
B_max = 120e-3  # Maximum field (Tesla)
B_points = 300
B_range = np.linspace(-B_max, B_max, B_points)

# Define junction effective area
junction_width = 10e-6  # Example width (m)
junction_length = 10e-6  # Example length (m)
lambda_L = 90e-9  # London penetration depth for Nb
d_barrier = 7e-9  # Total barrier thickness (Cu+Cr+Fe+Cr+Cu)
effective_area = junction_width * (2*lambda_L + d_barrier)

# Thickness of Nb layers in your structure
d_Nb = 20e-9  # Middle Nb layer thickness

# Triplet coherence length
xi_triplet = 5e-9  # Typical value for Nb

# Run simulation
Ic_up, Ic_down = fraunhofer_pattern(jj, B_range, effective_area, d_Nb, xi_triplet)

# Plot Fraunhofer pattern
plt.figure(figsize=(10, 6))
plt.plot(B_range*1e3, Ic_up/jj.Ic, 'b-', label='Upward Sweep')
plt.plot(B_range*1e3, Ic_down/jj.Ic, 'r--', label='Downward Sweep')
plt.xlabel('Magnetic Field (mT)')
plt.ylabel('Normalized Critical Current $I_c/I_{c0}$')
plt.title('Fraunhofer Pattern with Spin-Glass Effects')
plt.grid(True)
plt.legend()

# Add secondary axis for flux quantum units
ax2 = plt.gca().twiny()
ax2.set_xlim(plt.gca().get_xlim())
ax2.set_xticks(np.linspace(-B_max*1e3, B_max*1e3, 7))
ax2.set_xticklabels([f'{x/(hbar_over_2e*2*np.pi)*effective_area*1e3:.1f}' for x in np.linspace(-B_max, B_max, 7)])
ax2.set_xlabel(r'$\Phi/\Phi_0$')
plt.show()