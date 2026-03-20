# qm-toolkit

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bfadem/qm-toolkit/main?labpath=qm_starter.ipynb)

A Python toolkit of commonly used matrices, state vectors, and helper functions
for quantum mechanics. Built for use in upper-division undergraduate physics courses
at Muhlenberg College, and as a personal sandbox for quantum calculations.

## Launch the Starter Notebook (Students: Start Here)

Click the **Launch Binder** badge above to open an interactive workspace in your browser — no installation required.

**Steps:**
1. Click the badge above (or [this link](https://mybinder.org/v2/gh/bfadem/qm-toolkit/main?labpath=qm_starter.ipynb))
2. Wait 1–2 minutes for the environment to build (first launch may take longer)
3. Run the setup cell first, then use the reference and workspace cells below it
4. Run any cell with **Shift + Enter**

> **Note:** Binder sessions are temporary. If you make changes you want to keep,
> use **File → Download** before closing your browser.

## Contents

### Spin-1/2
| Object | Description |
|---|---|
| `I2` | 2×2 identity matrix |
| `sigma_x, sigma_y, sigma_z` | Pauli matrices |
| `sigma_vec` | List `[sigma_x, sigma_y, sigma_z]` |
| `zplus, zminus` | σ_z eigenstates \|+z⟩, \|−z⟩ |
| `xplus, xminus` | σ_x eigenstates \|+x⟩, \|−x⟩ |
| `yplus, yminus` | σ_y eigenstates \|+y⟩, \|−y⟩ |
| `sigma_plus, sigma_minus` | Ladder operators σ± = (1/2)(σ_x ± iσ_y) |

### Spin-1  (ℏ = 1)
| Object | Description |
|---|---|
| `I3` | 3×3 identity matrix |
| `jx, jy, jz` | Spin-1 angular momentum matrices |
| `j_vec` | List `[jx, jy, jz]` |
| `jplus, jminus` | Ladder operators j± = jx ± i·jy |
| `jp1, j0, jm1` | jz eigenstates for m = +1, 0, −1 |

### Photon Polarization
| Object | Description |
|---|---|
| `pol_x, pol_y` | Linear polarization states \|x⟩, \|y⟩ |
| `pol_r, pol_l` | Circular polarization states; convention \|r⟩ = (1/√2)(\|x⟩ + i\|y⟩) |

### Helper Functions
| Function | Description |
|---|---|
| `dagger(A)` | Conjugate transpose A† |
| `comm(A, B)` | Commutator [A, B] = AB − BA |
| `expect(A, psi)` | Expectation value ⟨ψ\|A\|ψ⟩ |
| `norm(psi)` | Norm √⟨ψ\|ψ⟩ |
| `normalize(psi)` | Returns psi / norm(psi) |
| `prob(psi, phi)` | Transition probability \|⟨φ\|ψ⟩\|² |
| `list_objects()` | Print all available matrices and state vectors |
| `list_operations()` | Print all available helper functions |

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
expect(sigma_z, zplus)           # → 1.0
expect(jz_sym, jp1_sym)          # → 1

# Norm
norm(pol_r)                      # → 1.0

# Transition probability
prob(zplus, xplus)               # → 0.5

# Normalize
normalize(np.array([[3],[4]], dtype=complex))  # → [[0.6],[0.8]]

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
| `qm_starter.ipynb` | Clean student workspace (Binder entry point) |
| `qm_tutorial.ipynb` | Full pedagogical tutorial notebook |

## Running the built-in verification

```bash
python qm.py
```

This runs all internal consistency checks — eigenvalue equations, commutation
relations, orthonormality, completeness, and helper function tests.

## Author

Dr. Brett Fadem — Physics, Muhlenberg College
