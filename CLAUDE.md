# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

A Python toolkit of commonly used quantum mechanics matrices, state vectors, and helper functions. Built for upper-division undergraduate physics courses at Muhlenberg College. Deployable via Binder for student use without local installation.

## Running the verification suite

```bash
python qm.py
```

This runs all internal consistency checks (eigenvalue equations, commutation relations, orthonormality, completeness, and helper function tests) defined in the `if __name__ == "__main__"` block.

## Installing dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy==1.26.4`, `sympy==1.12`, `matplotlib==3.8.4`.

## Architecture

Everything lives in a single module, `qm.py`. There are no submodules or packages.

**Dual-form convention:** Every object is defined twice — once as a NumPy `complex128` ndarray and once as a SymPy `Matrix`. NumPy names are plain (`sigma_x`); SymPy names carry the `_sym` suffix (`sigma_x_sym`). Helper functions auto-detect which type they receive via `_is_sympy()` and dispatch accordingly.

**Object categories in `qm.py`:**
- Spin-1/2: Pauli matrices (`sigma_x/y/z`), σ_z eigenstates (`zplus`/`zminus`), σ_x eigenstates (`xplus`/`xminus`), σ_y eigenstates (`yplus`/`yminus`), ladder operators (`sigma_plus`/`sigma_minus`)
- Spin-1: angular momentum matrices (`jx`/`jy`/`jz`), ladder operators (`jplus`/`jminus`), jz eigenstates (`jp1`/`j0`/`jm1` for m=+1, 0, −1)
- Photon polarization: linear (`pol_x`/`pol_y`) and circular (`pol_r`/`pol_l`) states

**Helper functions:** `dagger`, `comm`, `expect`, `norm`, `normalize`, `prob`, `list_objects`, `list_operations`

**Notebooks:**
- `qm_starter.ipynb` — clean student workspace (Binder entry point)
- `qm_tutorial.ipynb` — full tutorial (in progress)

All state vectors are column vectors (shape `(n, 1)`).
