#!/usr/bin/env python3
"""Analyze experimental data to find true Ic peak positions."""
import numpy as np
from pathlib import Path

# Load data
data_dir = Path(__file__).parent.parent / 'H vs Ic'

data_down = np.loadtxt(data_dir / '3B_P1_Downsweep_1100B')
data_up = np.loadtxt(data_dir / '3B_P1_upsweep_1100B')

H_down, Ic_down = data_down[:, 0], data_down[:, 2]
H_up, Ic_up = data_up[:, 0], data_up[:, 2]

# Find global max for each sweep
idx_max_down = np.argmax(Ic_down)
idx_max_up = np.argmax(Ic_up)

print('Global peak positions (FULL DATA):')
print(f'  Downsweep: H = {H_down[idx_max_down]:.1f} Oe, Ic = {Ic_down[idx_max_down]*1e3:.2f} μA')
print(f'  Upsweep: H = {H_up[idx_max_up]:.1f} Oe, Ic = {Ic_up[idx_max_up]*1e3:.2f} μA')

# Switch fields from notebook fits
downswitch = -89.581  # Oe
upswitch = 141.434    # Oe

# Find peaks in stable magnetization regions
# Downsweep: after switch (H > -89.6 Oe), magnetization is in UP state
# Upsweep: before switch (H < 141.4 Oe), magnetization is in DOWN state
down_stable = H_down > downswitch
up_stable = H_up < upswitch

H_ds = H_down[down_stable]
Ic_ds = Ic_down[down_stable]
H_us = H_up[up_stable]
Ic_us = Ic_up[up_stable]

idx_peak_ds = np.argmax(Ic_ds)
idx_peak_us = np.argmax(Ic_us)

print('\nPeak positions in STABLE magnetization regions:')
print(f'  Downsweep (H > {downswitch} Oe, M↑): H = {H_ds[idx_peak_ds]:.1f} Oe, Ic = {Ic_ds.max()*1e3:.2f} μA')
print(f'  Upsweep (H < {upswitch} Oe, M↓): H = {H_us[idx_peak_us]:.1f} Oe, Ic = {Ic_us.max()*1e3:.2f} μA')

# Also check peaks before switching (other magnetization state)
down_before = H_down < downswitch  # Before switch, magnetization is DOWN
up_before = H_up > upswitch        # After switch, magnetization is UP

if np.any(down_before):
    H_db = H_down[down_before]
    Ic_db = Ic_down[down_before]
    print(f'  Downsweep (H < {downswitch} Oe, M↓): H = {H_db[np.argmax(Ic_db)]:.1f} Oe, Ic = {Ic_db.max()*1e3:.2f} μA')

if np.any(up_before):
    H_ub = H_up[up_before]
    Ic_ub = Ic_up[up_before]
    print(f'  Upsweep (H > {upswitch} Oe, M↑): H = {H_ub[np.argmax(Ic_ub)]:.1f} Oe, Ic = {Ic_ub.max()*1e3:.2f} μA')

# Calculate pattern center offsets
# The Airy pattern is centered where Φ_total = 0
# Φ_total = Φ_ext + Φ_M = 0 => H = -Φ_M/(μ0 × A_eff)
# So the peak position IS the pattern center

print('\n=== KEY INSIGHT ===')
print('The peak positions tell us where the Airy pattern is centered:')
print(f'  M↑ state (downsweep stable): centered at H = {H_ds[idx_peak_ds]:.1f} Oe')
print(f'  M↓ state (upsweep stable): centered at H = {H_us[idx_peak_us]:.1f} Oe')

# The difference between peak positions = 2 × H_offset from magnetization
H_offset = (H_ds[idx_peak_ds] - H_us[idx_peak_us]) / 2
H_midpoint = (H_ds[idx_peak_ds] + H_us[idx_peak_us]) / 2

print(f'\nPattern shift due to magnetization: ±{abs(H_offset):.1f} Oe')
print(f'Midpoint (H_ext only): {H_midpoint:.1f} Oe')

# What about the switch fields vs peak positions?
print('\n=== SWITCH FIELDS vs PEAK POSITIONS ===')
print(f'Switch fields: {downswitch:.1f} Oe (down), {upswitch:.1f} Oe (up)')
print(f'Peak positions: {H_us[idx_peak_us]:.1f} Oe (M↓), {H_ds[idx_peak_ds]:.1f} Oe (M↑)')
print('Note: Switch fields are where magnetization flips, NOT where Ic peaks!')
