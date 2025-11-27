# steane_decoder.py
# -*- coding: utf-8 -*-
"""
Steane [[7,1,3]] – syndrome-based Pauli decoder.

We assume:
  - CSS Steane with Hx = Hz = H_HAMMING (3x7) as in steane_stabilizers / steane_encoder.
  - Pauli noise: X, Y, Z on physical qubits.
  - Use classical Hamming syndrome:
       s = H * e (mod 2)
    where e is the binary error vector.

Z-type stabilizers detect X-errors.
X-type stabilizers detect Z-errors.

Y = XZ has both X and Z components -> both syndromes non-zero.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from steane_encoder import hamming_generator_matrix  # just to be consistent
from steane_stabilizers import H_HAMMING  # or re-import if you prefer

PauliError = Tuple[int, str]


# ===============================
# 1. Helpers: pattern <-> X/Z bits
# ===============================

def error_pattern_to_xz_bits(pattern: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a length-7 Pauli pattern ['I','X','Y','Z',...] into
    two binary vectors (x,z) of length 7:

      X -> x=1, z=0
      Z -> x=0, z=1
      Y -> x=1, z=1
      I -> x=0, z=0
    """
    n = len(pattern)
    x = np.zeros(n, dtype=int)
    z = np.zeros(n, dtype=int)

    for i, p in enumerate(pattern):
        if p == "X":
            x[i] = 1
        elif p == "Z":
            z[i] = 1
        elif p == "Y":
            x[i] = 1
            z[i] = 1
        elif p == "I":
            pass
        else:
            raise ValueError(f"Unknown Pauli in pattern: {p}")

    return x, z


def xz_bits_to_pattern(x: np.ndarray, z: np.ndarray) -> List[str]:
    """
    Inverse of error_pattern_to_xz_bits.
    """
    if x.shape != z.shape:
        raise ValueError("x and z must have same shape")
    n = len(x)
    pattern: List[str] = []

    for i in range(n):
        xb = int(x[i])
        zb = int(z[i])
        if xb == 0 and zb == 0:
            pattern.append("I")
        elif xb == 1 and zb == 0:
            pattern.append("X")
        elif xb == 0 and zb == 1:
            pattern.append("Z")
        elif xb == 1 and zb == 1:
            pattern.append("Y")
        else:
            raise ValueError("Invalid (x,z) at position {i}")

    return pattern


# ===============================
# 2. Syndrome lookup table
# ===============================

def _build_syndrome_lookup(H: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """
    For the [7,1,3] Steane/Hamming code, each column of H is a
    distinct non-zero 3-bit vector. That 3-bit vector is exactly
    the syndrome of a single-qubit error on that position.

    This builds:
        syndrome (3-tuple) -> qubit index (0..6)
    """
    m, n = H.shape  # 3 x 7
    lookup: Dict[Tuple[int, int, int], int] = {}

    for j in range(n):
        col = H[:, j] % 2
        key = tuple(int(b) for b in col)
        if key == (0, 0, 0):
            continue
        lookup[key] = j

    return lookup


SYNDROME_LOOKUP = _build_syndrome_lookup(H_HAMMING)


# ===============================
# 3. Syndrome computation
# ===============================

def compute_syndromes(x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the X- and Z-error syndromes.

    Z-type stabilizers (from H) see X-errors -> syndrome_X = H * x (mod 2)
    X-type stabilizers see Z-errors         -> syndrome_Z = H * z (mod 2)

    Returns
    -------
    s_X : np.ndarray, shape (3,)
        Syndrome associated to X-errors.
    s_Z : np.ndarray, shape (3,)
        Syndrome associated to Z-errors.
    """
    H = H_HAMMING
    s_X = (H @ x) % 2
    s_Z = (H @ z) % 2
    return s_X, s_Z


# ===============================
# 4. Single-step syndrome decoder
# ===============================

def steane_syndrome_decode(pattern: List[str]) -> List[str]:
    """
    Apply one round of ideal Steane syndrome decoding
    to a 7-qubit Pauli error pattern.

    - Build x,z binary vectors.
    - Compute syndromes s_X (for X-errors) and s_Z (for Z-errors).
    - For each non-zero syndrome, look up the most likely qubit and
      flip the corresponding Pauli component (X or Z).
    - Rebuild the corrected Pauli pattern.

    This corrects any weight-1 X/Z/Y error (distance 3).
    """
    x, z = error_pattern_to_xz_bits(pattern)

    s_X, s_Z = compute_syndromes(x, z)

    # Correct X component (using Z-type stabilizers)
    key_X = tuple(int(b) for b in s_X)
    if key_X != (0, 0, 0) and key_X in SYNDROME_LOOKUP:
        j = SYNDROME_LOOKUP[key_X]
        x[j] ^= 1  # apply X at qubit j

    # Correct Z component (using X-type stabilizers)
    key_Z = tuple(int(b) for b in s_Z)
    if key_Z != (0, 0, 0) and key_Z in SYNDROME_LOOKUP:
        j = SYNDROME_LOOKUP[key_Z]
        z[j] ^= 1  # apply Z at qubit j

    corrected_pattern = xz_bits_to_pattern(x, z)
    return corrected_pattern


