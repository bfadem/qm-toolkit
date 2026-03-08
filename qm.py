"""
qm.py — Quantum Mechanics Toolkit
==================================
A growing collection of commonly used matrices and vectors in quantum mechanics.
Each object is provided in two forms:
  - NumPy (complex128 ndarray)  — for numerical computation
  - SymPy (Matrix)              — for symbolic manipulation

Naming convention:
  <name>       → NumPy version
  <name>_sym   → SymPy version

Current contents
----------------
  Identity (2x2):        I2,        I2_sym
  Pauli matrices:        sigma_x,   sigma_x_sym
                         sigma_y,   sigma_y_sym
                         sigma_z,   sigma_z_sym
  Pauli vector:          sigma_vec  (list: [sigma_x, sigma_y, sigma_z])

  Spin-1/2 eigenstates of σ_z (2×1 column vectors):
    spin_up,    spin_up_sym     |+⟩ = |↑⟩,              eigenvalue +1
    spin_down,  spin_down_sym   |−⟩ = |↓⟩,              eigenvalue −1

  Spin-1/2 eigenstates of σ_x (2×1 column vectors):
    xplus,      xplus_sym       |+x⟩ = (1/√2)(|+⟩ + |−⟩),   eigenvalue +1
    xminus,     xminus_sym      |−x⟩ = (1/√2)(|+⟩ − |−⟩),   eigenvalue −1

  Ladder operators — convention σ± = (1/2)(σ_x ± iσ_y):
    sigma_plus,   sigma_plus_sym    σ₊ = |+⟩⟨−|,  raises |−⟩ → |+⟩
    sigma_minus,  sigma_minus_sym   σ₋ = |−⟩⟨+|,  lowers |+⟩ → |−⟩

── Spin-1  (ℏ = 1 throughout) ────────────────────────────────────────────────
  3×3 Identity:          I3,        I3_sym

  Spin-1 matrices (dimensionless, ℏ = 1):
    jx,   jx_sym        x-component
    jy,   jy_sym        y-component
    jz,   jz_sym        z-component
    j_vec, j_vec_sym    (list: [jx, jy, jz])

  Spin-1 ladder operators  j± = jx ± i·jy:
    jplus,   jplus_sym    j₊ raises m → m+1
    jminus,  jminus_sym   j₋ lowers m → m−1

  Spin-1 eigenstates of jz (3×1 column vectors):
    jm1,    jm1_sym     |j=1, m=+1⟩
    j0,     j0_sym      |j=1, m= 0⟩
    jm_1,   jm_1_sym    |j=1, m=−1⟩

── Photon Polarization  (2×1 column vectors) ─────────────────────────────────
  Linear basis:
    pol_x,  pol_x_sym   |x⟩  — horizontal polarization
    pol_y,  pol_y_sym   |y⟩  — vertical polarization

  Circular basis  (convention: |R⟩ = (1/√2)(|x⟩ + i|y⟩)):
    pol_R,  pol_R_sym   |R⟩  — right circular polarization
    pol_L,  pol_L_sym   |L⟩  — left  circular polarization

── Helper Functions ───────────────────────────────────────────────────────────
  Each function auto-detects NumPy vs SymPy input and dispatches accordingly.

    dagger(A)          — conjugate transpose  A†
    comm(A, B)         — commutator           [A, B] = AB − BA
    expect(A, psi)     — expectation value    ⟨ψ|A|ψ⟩
    norm(psi)          — norm of state        √⟨ψ|ψ⟩
"""

import numpy as np
import sympy as sp

# ── Convenience ────────────────────────────────────────────────────────────────
_i = 1j                          # NumPy imaginary unit
_I = sp.I                        # SymPy imaginary unit

def _is_sympy(A):
    """Return True if A is a SymPy object."""
    return isinstance(A, (sp.MatrixBase, sp.Expr))

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def dagger(A):
    """Conjugate transpose (Hermitian adjoint) A†.

    Works for both NumPy arrays and SymPy matrices.

    Examples
    --------
    >>> dagger(sigma_y)          # NumPy
    >>> dagger(sigma_y_sym)      # SymPy
    """
    if _is_sympy(A):
        return A.H
    return A.conj().T


def comm(A, B, simplify=True):
    """Commutator [A, B] = AB − BA.

    Parameters
    ----------
    A, B     : NumPy ndarray or SymPy Matrix
    simplify : bool (SymPy only) — call sp.simplify on the result

    Examples
    --------
    >>> comm(sigma_x, sigma_y)          # → 2i·sigma_z  (NumPy)
    >>> comm(sigma_x_sym, sigma_y_sym)  # → 2I·sigma_z  (SymPy)
    """
    if _is_sympy(A):
        result = A * B - B * A
        return sp.simplify(result) if simplify else result
    return A @ B - B @ A


def expect(A, psi, simplify=True):
    """Expectation value ⟨ψ|A|ψ⟩.

    Parameters
    ----------
    A        : operator  (NumPy ndarray or SymPy Matrix)
    psi      : state     (column vector, same type as A)
    simplify : bool (SymPy only) — call sp.simplify on the result

    Returns a scalar (NumPy complex or SymPy expression).

    Examples
    --------
    >>> expect(sigma_z, spin_up)          # → 1.0  (NumPy)
    >>> expect(sigma_z_sym, spin_up_sym)  # → 1    (SymPy)
    """
    if _is_sympy(A):
        result = (psi.H * A * psi)[0, 0]
        return sp.simplify(result) if simplify else result
    return (psi.conj().T @ A @ psi)[0, 0]


def norm(psi, simplify=True):
    """Norm of a state vector √⟨ψ|ψ⟩.

    Parameters
    ----------
    psi      : column vector (NumPy ndarray or SymPy Matrix)
    simplify : bool (SymPy only) — call sp.simplify on the result

    Examples
    --------
    >>> norm(spin_up)        # → 1.0   (NumPy)
    >>> norm(xplus_sym)      # → 1     (SymPy)
    """
    if _is_sympy(psi):
        result = sp.sqrt((psi.H * psi)[0, 0])
        return sp.simplify(result) if simplify else result
    return float(np.sqrt((psi.conj().T @ psi)[0, 0].real))

# ══════════════════════════════════════════════════════════════════════════════
# 2×2 Identity
# ══════════════════════════════════════════════════════════════════════════════
I2 = np.eye(2, dtype=complex)

I2_sym = sp.eye(2)

# ══════════════════════════════════════════════════════════════════════════════
# Pauli Matrices
# ══════════════════════════════════════════════════════════════════════════════

# σ_x  (σ_1)
sigma_x = np.array([[0,  1],
                     [1,  0]], dtype=complex)

sigma_x_sym = sp.Matrix([[0, 1],
                          [1, 0]])

# σ_y  (σ_2)
sigma_y = np.array([[0, -_i],
                     [_i,  0]], dtype=complex)

sigma_y_sym = sp.Matrix([[0, -_I],
                          [_I,  0]])

# σ_z  (σ_3)
sigma_z = np.array([[ 1,  0],
                     [ 0, -1]], dtype=complex)

sigma_z_sym = sp.Matrix([[ 1,  0],
                          [ 0, -1]])

# Pauli vector  σ⃗ = [σ_x, σ_y, σ_z]
sigma_vec     = [sigma_x,     sigma_y,     sigma_z]
sigma_vec_sym = [sigma_x_sym, sigma_y_sym, sigma_z_sym]

# ══════════════════════════════════════════════════════════════════════════════
# Spin-1/2 Eigenstates of σ_z  (2×1 column vectors)
# ══════════════════════════════════════════════════════════════════════════════
# |+⟩ = |↑⟩  — spin-up,   σ_z eigenvalue +1
spin_up = np.array([[1],
                    [0]], dtype=complex)

spin_up_sym = sp.Matrix([[1],
                         [0]])

# |−⟩ = |↓⟩  — spin-down, σ_z eigenvalue −1
spin_down = np.array([[0],
                      [1]], dtype=complex)

spin_down_sym = sp.Matrix([[0],
                           [1]])

# ══════════════════════════════════════════════════════════════════════════════
# Spin-1/2 Eigenstates of σ_x  (2×1 column vectors)
# ══════════════════════════════════════════════════════════════════════════════
_r2 = np.sqrt(0.5)                   # 1/√2  (NumPy)
_R2 = 1 / sp.sqrt(2)                 # 1/√2  (SymPy, exact)

# |+x⟩ = (1/√2)(|+⟩ + |−⟩),  σ_x eigenvalue +1
xplus = _r2 * np.array([[1],
                         [1]], dtype=complex)

xplus_sym = _R2 * sp.Matrix([[1],
                              [1]])

# |−x⟩ = (1/√2)(|+⟩ − |−⟩),  σ_x eigenvalue −1
xminus = _r2 * np.array([[ 1],
                          [-1]], dtype=complex)

xminus_sym = _R2 * sp.Matrix([[ 1],
                               [-1]])

# ══════════════════════════════════════════════════════════════════════════════
# Spin-1/2 Eigenstates of σ_y  (2×1 column vectors)
# ══════════════════════════════════════════════════════════════════════════════

# |+y⟩ = (1/√2)(|+⟩ + i|−⟩),  σ_y eigenvalue +1
yplus = _r2 * np.array([[1],
                         [_i]], dtype=complex)

yplus_sym = _R2 * sp.Matrix([[1],
                              [_I]])

# |−y⟩ = (1/√2)(|+⟩ − i|−⟩),  σ_y eigenvalue −1
yminus = _r2 * np.array([[1],
                          [-_i]], dtype=complex)

yminus_sym = _R2 * sp.Matrix([[1],
                               [-_I]])

# ══════════════════════════════════════════════════════════════════════════════
# Ladder Operators  σ± = (1/2)(σ_x ± i σ_y)
# ══════════════════════════════════════════════════════════════════════════════

# σ₊ = (1/2)(σ_x + i σ_y) = |+⟩⟨−|  — raising operator
sigma_plus = 0.5 * (sigma_x + _i * sigma_y)

sigma_plus_sym = sp.Rational(1, 2) * (sigma_x_sym + _I * sigma_y_sym)

# σ₋ = (1/2)(σ_x − i σ_y) = |−⟩⟨+|  — lowering operator
sigma_minus = 0.5 * (sigma_x - _i * sigma_y)

sigma_minus_sym = sp.Rational(1, 2) * (sigma_x_sym - _I * sigma_y_sym)

# ══════════════════════════════════════════════════════════════════════════════
# ── SPIN-1  (ℏ = 1 throughout) ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── 3×3 Identity ──────────────────────────────────────────────────────────────
I3 = np.eye(3, dtype=complex)

I3_sym = sp.eye(3)

# ── Spin-1 Matrices ───────────────────────────────────────────────────────────
# Standard basis ordered |m=+1⟩, |m=0⟩, |m=−1⟩
# Matrix elements from J_i = ⟨m'|J_i|m⟩  with j=1

_r2h = np.sqrt(0.5)          # 1/√2  (NumPy)
_R2h = 1 / sp.sqrt(2)        # 1/√2  (SymPy)

# jx = (1/√2) [[0,1,0],[1,0,1],[0,1,0]]
jx = _r2h * np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=complex)

jx_sym = _R2h * sp.Matrix([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])

# jy = (1/√2) [[0,−i,0],[i,0,−i],[0,i,0]]
jy = _r2h * np.array([[0,  -_i, 0],
                       [_i,  0, -_i],
                       [0,   _i, 0]], dtype=complex)

jy_sym = _R2h * sp.Matrix([[0,  -_I, 0],
                             [_I,  0, -_I],
                             [0,   _I, 0]])

# jz = diag(1, 0, −1)
jz = np.array([[ 1, 0,  0],
               [ 0, 0,  0],
               [ 0, 0, -1]], dtype=complex)

jz_sym = sp.Matrix([[ 1, 0,  0],
                     [ 0, 0,  0],
                     [ 0, 0, -1]])

# Spin-1 vector  j⃗ = [jx, jy, jz]
j_vec     = [jx,     jy,     jz]
j_vec_sym = [jx_sym, jy_sym, jz_sym]

# ── Spin-1 Ladder Operators  j± = jx ± i·jy ──────────────────────────────────

# j₊ = jx + i·jy = √2 [[0,1,0],[0,0,1],[0,0,0]]
jplus = jx + _i * jy

jplus_sym = jx_sym + _I * jy_sym

# j₋ = jx − i·jy = √2 [[0,0,0],[1,0,0],[0,1,0]]
jminus = jx - _i * jy

jminus_sym = jx_sym - _I * jy_sym

# ── Spin-1 Eigenstates of jz  (3×1 column vectors) ───────────────────────────

# |j=1, m=+1⟩
jm1 = np.array([[1],
                [0],
                [0]], dtype=complex)

jm1_sym = sp.Matrix([[1], [0], [0]])

# |j=1, m=0⟩
j0 = np.array([[0],
               [1],
               [0]], dtype=complex)

j0_sym = sp.Matrix([[0], [1], [0]])

# |j=1, m=−1⟩
jm_1 = np.array([[0],
                 [0],
                 [1]], dtype=complex)

jm_1_sym = sp.Matrix([[0], [0], [1]])

# ══════════════════════════════════════════════════════════════════════════════
# ── PHOTON POLARIZATION  (2×1 column vectors) ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Linear Basis ──────────────────────────────────────────────────────────────

# |x⟩ — horizontal (x) linear polarization
pol_x = np.array([[1],
                  [0]], dtype=complex)

pol_x_sym = sp.Matrix([[1], [0]])

# |y⟩ — vertical (y) linear polarization
pol_y = np.array([[0],
                  [1]], dtype=complex)

pol_y_sym = sp.Matrix([[0], [1]])

# ── Circular Basis ────────────────────────────────────────────────────────────
# Convention:  |R⟩ = (1/√2)(|x⟩ + i|y⟩)
#              |L⟩ = (1/√2)(|x⟩ − i|y⟩)

# |R⟩ — right circular polarization
pol_R = _r2 * np.array([[1],
                         [_i]], dtype=complex)

pol_R_sym = _R2 * sp.Matrix([[1], [_I]])

# |L⟩ — left circular polarization
pol_L = _r2 * np.array([[1],
                         [-_i]], dtype=complex)

pol_L_sym = _R2 * sp.Matrix([[1], [-_I]])

# ══════════════════════════════════════════════════════════════════════════════
# Quick verification (runs when the module is executed directly)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Pauli Matrix Verification ===\n")

    labels = ["σ_x", "σ_y", "σ_z"]
    mats   = [sigma_x, sigma_y, sigma_z]

    for label, s in zip(labels, mats):
        sq = s @ s
        is_identity = np.allclose(sq, I2)
        comm_check  = np.allclose(s @ s - s @ s, np.zeros((2,2)))  # trivially true
        print(f"{label}:\n{s}")
        print(f"  {label}² = I?  {is_identity}")
        print()

    # Check commutation: [σ_x, σ_y] = 2i σ_z
    comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
    expected = 2j * sigma_z
    print(f"[σ_x, σ_y] = 2i·σ_z?  {np.allclose(comm_xy, expected)}")

    # Check anti-commutation: {σ_x, σ_y} = 0
    anticomm_xy = sigma_x @ sigma_y + sigma_y @ sigma_x
    print(f"{{σ_x, σ_y}} = 0?       {np.allclose(anticomm_xy, np.zeros((2,2)))}")

    print("\n=== Spin-1/2 Eigenstate Verification ===\n")

    # σ_z |+⟩ = +1 |+⟩
    sz_up = sigma_z @ spin_up
    print(f"σ_z |+⟩ = +|+⟩?  {np.allclose(sz_up, spin_up)}")

    # σ_z |−⟩ = −1 |−⟩
    sz_down = sigma_z @ spin_down
    print(f"σ_z |−⟩ = −|−⟩?  {np.allclose(sz_down, -spin_down)}")

    # Orthonormality
    inner_up_up     = (spin_up.conj().T   @ spin_up)[0, 0]
    inner_down_down = (spin_down.conj().T @ spin_down)[0, 0]
    inner_up_down   = (spin_up.conj().T   @ spin_down)[0, 0]
    print(f"⟨+|+⟩ = 1?  {np.isclose(inner_up_up, 1)}")
    print(f"⟨−|−⟩ = 1?  {np.isclose(inner_down_down, 1)}")
    print(f"⟨+|−⟩ = 0?  {np.isclose(inner_up_down, 0)}")

    # Completeness  |+⟩⟨+| + |−⟩⟨−| = I
    completeness = spin_up @ spin_up.conj().T + spin_down @ spin_down.conj().T
    print(f"|+⟩⟨+| + |−⟩⟨−| = I?  {np.allclose(completeness, I2)}")

    print("\n=== σ_x Eigenstate Verification ===\n")

    # Eigenvalue checks
    print(f"σ_x |+x⟩ = +|+x⟩?  {np.allclose(sigma_x @ xplus,  +xplus)}")
    print(f"σ_x |−x⟩ = −|−x⟩?  {np.allclose(sigma_x @ xminus, -xminus)}")

    # Orthonormality
    inner_xx = (xplus.conj().T @ xminus)[0, 0]
    print(f"⟨+x|−x⟩ = 0?  {np.isclose(inner_xx, 0)}")
    print(f"⟨+x|+x⟩ = 1?  {np.isclose((xplus.conj().T  @ xplus)[0,0],  1)}")
    print(f"⟨−x|−x⟩ = 1?  {np.isclose((xminus.conj().T @ xminus)[0,0], 1)}")

    # Completeness
    comp_x = xplus @ xplus.conj().T + xminus @ xminus.conj().T
    print(f"|+x⟩⟨+x| + |−x⟩⟨−x| = I?  {np.allclose(comp_x, I2)}")

    print("\n=== σ_y Eigenstate Verification ===\n")

    # Eigenvalue checks
    print(f"σ_y |+y⟩ = +|+y⟩?  {np.allclose(sigma_y @ yplus,  +yplus)}")
    print(f"σ_y |−y⟩ = −|−y⟩?  {np.allclose(sigma_y @ yminus, -yminus)}")

    # Orthonormality
    inner_yy = (yplus.conj().T @ yminus)[0, 0]
    print(f"⟨+y|−y⟩ = 0?  {np.isclose(inner_yy, 0)}")
    print(f"⟨+y|+y⟩ = 1?  {np.isclose((yplus.conj().T  @ yplus)[0,0],  1)}")
    print(f"⟨−y|−y⟩ = 1?  {np.isclose((yminus.conj().T @ yminus)[0,0], 1)}")

    # Completeness
    comp_y = yplus @ yplus.conj().T + yminus @ yminus.conj().T
    print(f"|+y⟩⟨+y| + |−y⟩⟨−y| = I?  {np.allclose(comp_y, I2)}")

    print("\n=== Ladder Operator Verification ===\n")

    # Explicit matrix form
    print(f"σ₊ =\n{sigma_plus}\n")
    print(f"σ₋ =\n{sigma_minus}\n")

    # Raising/lowering action
    print(f"σ₊ |−⟩ = |+⟩?  {np.allclose(sigma_plus  @ spin_down, spin_up)}")
    print(f"σ₋ |+⟩ = |−⟩?  {np.allclose(sigma_minus @ spin_up,   spin_down)}")

    # Annihilation
    print(f"σ₊ |+⟩ = 0?    {np.allclose(sigma_plus  @ spin_up,   np.zeros((2,1)))}")
    print(f"σ₋ |−⟩ = 0?    {np.allclose(sigma_minus @ spin_down, np.zeros((2,1)))}")

    # Reconstruction:  σ_x = σ₊ + σ₋,  σ_y = i(σ₋ − σ₊)
    print(f"\nσ₊ + σ₋ = σ_x?          {np.allclose(sigma_plus + sigma_minus, sigma_x)}")
    print(f"i(σ₋ − σ₊) = σ_y?       {np.allclose(1j*(sigma_minus - sigma_plus), sigma_y)}")

    # Commutator [σ₊, σ₋] = σ_z
    comm_pm = sigma_plus @ sigma_minus - sigma_minus @ sigma_plus
    print(f"[σ₊, σ₋] = σ_z?         {np.allclose(comm_pm, sigma_z)}")

    print("\nSymPy:")
    print("sigma_plus_sym  =", sigma_plus_sym)
    print("sigma_minus_sym =", sigma_minus_sym)
    print("σ₊ |−⟩ (sym) =", sigma_plus_sym  * spin_down_sym)
    print("σ₋ |+⟩ (sym) =", sigma_minus_sym * spin_up_sym)

    print("\n=== SymPy versions ===\n")
    print("sigma_x_sym =", sigma_x_sym)
    print("sigma_y_sym =", sigma_y_sym)
    print("sigma_z_sym =", sigma_z_sym)
    print("\nsigma_x_sym² =", sigma_x_sym**2)
    print("\nspin_up_sym   =", spin_up_sym.T,  " (transposed for display)")
    print("spin_down_sym =", spin_down_sym.T, " (transposed for display)")
    print("\nxplus_sym  =", xplus_sym.T,  " (transposed for display)")
    print("xminus_sym =", xminus_sym.T, " (transposed for display)")
    print("\nyplus_sym  =", yplus_sym.T,  " (transposed for display)")
    print("yminus_sym =", yminus_sym.T, " (transposed for display)")
    print("\nσ_z |+⟩  (sym) =", sigma_z_sym * spin_up_sym)
    print("σ_z |−⟩  (sym) =", sigma_z_sym * spin_down_sym)
    print("\nσ_x |+x⟩ (sym) =", sp.simplify(sigma_x_sym * xplus_sym))
    print("σ_x |−x⟩ (sym) =", sp.simplify(sigma_x_sym * xminus_sym))
    print("\nσ_y |+y⟩ (sym) =", sp.simplify(sigma_y_sym * yplus_sym))
    print("σ_y |−y⟩ (sym) =", sp.simplify(sigma_y_sym * yminus_sym))

    # ── Spin-1 Verification ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("=== Spin-1 Matrix Verification ===\n")

    # Commutation relations [jx, jy] = i·jz  (and cyclic)
    comm_jxjy = jx @ jy - jy @ jx
    comm_jyjz = jy @ jz - jz @ jy
    comm_jzjx = jz @ jx - jx @ jz
    print(f"[jx, jy] = i·jz?  {np.allclose(comm_jxjy, _i * jz)}")
    print(f"[jy, jz] = i·jx?  {np.allclose(comm_jyjz, _i * jx)}")
    print(f"[jz, jx] = i·jy?  {np.allclose(comm_jzjx, _i * jy)}")

    # j² = j(j+1)·I = 2·I  for j=1
    j_squared = jx @ jx + jy @ jy + jz @ jz
    print(f"\nj² = 2·I3?        {np.allclose(j_squared, 2 * I3)}")

    print("\n=== Spin-1 Eigenstate Verification ===\n")

    # jz eigenvalues
    print(f"jz |m=+1⟩ = +1·|m=+1⟩?  {np.allclose(jz @ jm1,  +1 * jm1)}")
    print(f"jz |m= 0⟩ =  0·|m= 0⟩?  {np.allclose(jz @ j0,    0 * j0)}")
    print(f"jz |m=−1⟩ = −1·|m=−1⟩?  {np.allclose(jz @ jm_1, -1 * jm_1)}")

    # Ladder action:  j₊|m⟩ = √(j(j+1)−m(m+1)) |m+1⟩
    print(f"\nj₊ |m= 0⟩ = √2·|m=+1⟩?  {np.allclose(jplus  @ j0,   np.sqrt(2) * jm1)}")
    print(f"j₊ |m=−1⟩ = √2·|m= 0⟩?  {np.allclose(jplus  @ jm_1, np.sqrt(2) * j0)}")
    print(f"j₋ |m=+1⟩ = √2·|m= 0⟩?  {np.allclose(jminus @ jm1,  np.sqrt(2) * j0)}")
    print(f"j₋ |m= 0⟩ = √2·|m=−1⟩?  {np.allclose(jminus @ j0,   np.sqrt(2) * jm_1)}")

    # Annihilation at boundaries
    print(f"\nj₊ |m=+1⟩ = 0?           {np.allclose(jplus  @ jm1,  np.zeros((3,1)))}")
    print(f"j₋ |m=−1⟩ = 0?           {np.allclose(jminus @ jm_1, np.zeros((3,1)))}")

    # Orthonormality
    states   = [jm1,  j0,  jm_1]
    labels_j = ["+1", " 0", "−1"]
    print("\nOrthonormality:")
    for i, (si, li) in enumerate(zip(states, labels_j)):
        for sj, lj in zip(states, labels_j):
            val = (si.conj().T @ sj)[0, 0]
            expected_val = 1.0 if si is sj else 0.0
            ok = np.isclose(val, expected_val)
            print(f"  ⟨m={li}|m={lj}⟩ = {expected_val:.0f}?  {ok}")

    # Completeness
    comp_j = jm1 @ jm1.conj().T + j0 @ j0.conj().T + jm_1 @ jm_1.conj().T
    print(f"\n|+1⟩⟨+1| + |0⟩⟨0| + |−1⟩⟨−1| = I3?  {np.allclose(comp_j, I3)}")

    print("\nSymPy — jz eigenstates:")
    print("jz·|m=+1⟩ =", jz_sym * jm1_sym)
    print("jz·|m= 0⟩ =", jz_sym * j0_sym)
    print("jz·|m=−1⟩ =", jz_sym * jm_1_sym)
    print("\nj₊·|m= 0⟩ =", sp.simplify(jplus_sym * j0_sym))
    print("j₋·|m= 0⟩ =", sp.simplify(jminus_sym * j0_sym))

    # ── Photon Polarization Verification ─────────────────────────────────────
    print("\n" + "="*60)
    print("=== Photon Polarization Verification ===\n")

    pol_states  = [pol_x,  pol_y,  pol_R,  pol_L]
    pol_labels  = ["|x⟩",  "|y⟩",  "|R⟩",  "|L⟩"]

    # Normalization
    print("Normalization:")
    for s, l in zip(pol_states, pol_labels):
        val = (s.conj().T @ s)[0, 0]
        print(f"  ⟨{l[1:-1]}|{l[1:-1]}⟩ = 1?  {np.isclose(val, 1)}")

    # Orthogonality within each basis
    print("\nOrthogonality:")
    print(f"  ⟨x|y⟩ = 0?  {np.isclose((pol_x.conj().T @ pol_y)[0,0], 0)}")
    print(f"  ⟨R|L⟩ = 0?  {np.isclose((pol_R.conj().T @ pol_L)[0,0], 0)}")

    # Completeness of each basis
    comp_lin = pol_x @ pol_x.conj().T + pol_y @ pol_y.conj().T
    comp_cir = pol_R @ pol_R.conj().T + pol_L @ pol_L.conj().T
    print(f"\n|x⟩⟨x| + |y⟩⟨y| = I2?  {np.allclose(comp_lin, I2)}")
    print(f"|R⟩⟨R| + |L⟩⟨L| = I2?  {np.allclose(comp_cir, I2)}")

    # Express circular states in linear basis
    print("\nCircular in linear basis:")
    print(f"  |R⟩ = (1/√2)(|x⟩ + i|y⟩)?  "
          f"{np.allclose(pol_R, _r2*(pol_x + _i*pol_y))}")
    print(f"  |L⟩ = (1/√2)(|x⟩ − i|y⟩)?  "
          f"{np.allclose(pol_L, _r2*(pol_x - _i*pol_y))}")

    # Express linear states in circular basis
    print("\nLinear in circular basis:")
    print(f"  |x⟩ = (1/√2)(|R⟩ + |L⟩)?   "
          f"{np.allclose(pol_x, _r2*(pol_R + pol_L))}")
    print(f"  |y⟩ = (i/√2)(|L⟩ − |R⟩)?   "
          f"{np.allclose(pol_y, _r2*_i*(pol_L - pol_R))}")

    print("\nSymPy:")
    print("pol_x_sym =", pol_x_sym.T, " (transposed for display)")
    print("pol_y_sym =", pol_y_sym.T)
    print("pol_R_sym =", pol_R_sym.T)
    print("pol_L_sym =", pol_L_sym.T)
    print("\n⟨R|L⟩ (sym) =", sp.simplify(pol_R_sym.H * pol_L_sym))

    # ── Helper Function Verification ─────────────────────────────────────────
    print("\n" + "="*60)
    print("=== Helper Function Verification ===\n")

    # dagger
    print("dagger(sigma_y) == sigma_y (Hermitian)?  ",
          np.allclose(dagger(sigma_y), sigma_y))
    print("dagger(spin_up) shape:", dagger(spin_up).shape, " (should be (1,2))")

    # comm — NumPy
    print("\nNumPy comm:")
    print(f"  comm(sigma_x, sigma_y) = 2i·sigma_z?  "
          f"{np.allclose(comm(sigma_x, sigma_y), 2j * sigma_z)}")
    print(f"  comm(jx, jy) = i·jz?                  "
          f"{np.allclose(comm(jx, jy), _i * jz)}")

    # comm — SymPy
    print("\nSymPy comm:")
    print(f"  comm(sigma_x_sym, sigma_y_sym) =",
          comm(sigma_x_sym, sigma_y_sym))

    # expect — NumPy
    print("\nNumPy expect:")
    print(f"  ⟨+|σ_z|+⟩ = +1?  {np.isclose(expect(sigma_z, spin_up),   +1)}")
    print(f"  ⟨−|σ_z|−⟩ = −1?  {np.isclose(expect(sigma_z, spin_down), -1)}")
    print(f"  ⟨+|σ_x|+⟩ =  0?  {np.isclose(expect(sigma_x, spin_up),    0)}")

    # expect — SymPy
    print("\nSymPy expect:")
    print(f"  ⟨+|σ_z|+⟩ =", expect(sigma_z_sym, spin_up_sym))
    print(f"  ⟨+x|σ_x|+x⟩ =", expect(sigma_x_sym, xplus_sym))

    # norm — NumPy
    print("\nNumPy norm:")
    for s, l in zip([spin_up, spin_down, xplus, yplus, pol_R],
                    ["|+⟩", "|−⟩", "|+x⟩", "|+y⟩", "|R⟩"]):
        print(f"  norm({l}) = 1?  {np.isclose(norm(s), 1.0)}")

    # norm — SymPy
    print("\nSymPy norm:")
    print(f"  norm(xplus_sym)  =", norm(xplus_sym))
    print(f"  norm(pol_R_sym)  =", norm(pol_R_sym))
