# qm-toolkit

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bfadem/qm-toolkit/main?labpath=qm_tutorial.ipynb)

A Python toolkit of commonly used matrices, state vectors, and helper functions
for quantum mechanics. Built for use in upper-division undergraduate physics courses
at Muhlenberg College, and as a personal sandbox for quantum calculations.

## Launch the Tutorial (Students: Start Here)

Click the **Launch Binder** badge above to open the interactive tutorial notebook
in your browser — no installation required.

**Steps:**
1. Click the badge above (or [this link](https://mybinder.org/v2/gh/bfadem/qm-toolkit/main?labpath=qm_tutorial.ipynb))
2. Wait 1–2 minutes for the environment to build (first launch may take longer)
3. When the notebook opens, run cells with **Shift + Enter**

> **Note:** Binder sessions are temporary. If you make changes you want to keep,
> use **File → Download** before closing your browser.

## Contents

### Spin-1/2
| Object | Description |
|---|---|
| `I2` | 2×2 identity matrix |
| `sigma_x, sigma_y, sigma_z` | Pauli matrices |
| `sigma_vec` | List `[sigma_x, sigma_y, sigma_z]` |
| `spin_up, spin_down` | σ_z eigenstates \|+⟩, \|−⟩ |
| `xplus, xminus` | σ_x eigenstates |
| `yplus, yminus` | σ_y eigenstates |
| `sigma_plus, sigma_minus` | Ladder operators σ± = (1/2)(σ_x ± iσ_y) |

### Spin-1  (ℏ = 1)
| Object | Description |
|---|---|
| `I3` | 3×3 identity matrix |
| `jx, jy, jz` | Spin-1 angular momentum matrices |
| `j_vec` | List `[jx, jy, jz]` |
| `jplus, jminus` | Ladder operators j± = jx ± i·jy |
| `jm1, j0, jm_1` | jz eigenstates for m = +1, 0, −1 |

### Photon Polarization
| Object | Description |
|---|---|
| `pol_x, pol_y` | Linear polarization states \|x⟩, \|y⟩ |
| `pol_R, pol_L` | Circular polarization states; convention \|R⟩ = (1/√2)(\|x⟩ + i\|y⟩) |

### Helper Functions
| Function | Description |
|---|---|
| `dagger(A)` | Conjugate transpose A† |
| `comm(A, B)` | Commutator [A, B] = AB − BA |
| `expect(A, psi)` | Expectation value ⟨ψ\|A\|ψ⟩ |
| `norm(psi)` | Norm √⟨ψ\|ψ⟩ |

All objects are provided in two forms:
- **NumPy** (`complex128` ndarray) — for numerical computation
- **SymPy** (suffix `_sym`) — for symbolic manipulation

Helper functions auto-detect which type they receive.

## Usage

Place `qm.py` in your working directory, then:

```python
from qm import *

# Commutator
comm(sigma_x, sigma_y)           # NumPy → 2i·sigma_z
comm(sigma_x_sym, sigma_y_sym)   # SymPy → 2I·sigma_z

# Expectation value
expect(sigma_z, spin_up)         # → 1.0
expect(jz_sym, jm1_sym)          # → 1

# Norm
norm(pol_R)                      # → 1.0

# Dagger
dagger(sigma_y)                  # → sigma_y  (Hermitian)
```

## Requirements

```
numpy
sympy
```

Install with:

```bash
pip install numpy sympy
```

## Files

| File | Description |
|---|---|
| `qm.py` | Main toolkit module |
| `qm_tutorial.ipynb` | Full tutorial notebook (coming soon) |

## Running the built-in verification

```bash
python qm.py
```

This runs all internal consistency checks — eigenvalue equations, commutation
relations, orthonormality, completeness, and helper function tests.

## Author

Dr. Brett Fadem — Physics, Muhlenberg College
