"""
Usadel Solver Package
=====================

Full self-consistent Usadel equation solver for spin-triplet Josephson junctions.

Features:
- C++ accelerated core with OpenMP parallelization
- 2D cross-section geometry (x = depth, y = lateral)
- Explicit spin-flip scattering in Nb
- Full gap self-consistency
- Comparison with experimental data

Usage:
------
>>> from usadel_solver import UsadelSimulator, JunctionParams
>>> junction = JunctionParams.default_NbCuCrFeCrCuNb(d_Cr=5e-9)
>>> sim = UsadelSimulator(junction)
>>> Ic = sim.compute_Ic_vs_B(B_range)
"""

from .usadel_wrapper import (
    UsadelSimulator,
    JunctionParams,
    LayerParams,
    load_experimental_data,
    fit_airy_pattern,
    HAS_CPP,
)

__all__ = [
    'UsadelSimulator',
    'JunctionParams',
    'LayerParams',
    'load_experimental_data',
    'fit_airy_pattern',
    'HAS_CPP',
]

__version__ = '0.1.0'
