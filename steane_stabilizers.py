# steane_stabilizers.py
# -*- coding: utf-8 -*-
"""
Steane [[7,1,3]] code – stabilizers and CSS structure.

- Define the classical [7,4,3] Hamming parity-check matrix H.
- Build X-type and Z-type stabilizer generators from H (CSS code).
- Provide Qiskit SparsePauliOp objects for use in circuits / checks.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ==========================
# 1. Classical Hamming [7,4,3]
# ==========================

# Standard parity-check matrix for the [7,4,3] Hamming code.
# Each column is the binary representation of 1..7:
# 1 -> 001, 2 -> 010, 3 -> 011, ..., 7 -> 111
# Rows correspond to parity checks.
H_HAMMING = np.array(
    [
        [1, 0, 0, 0, 1, 1, 1],  # check 1
        [0, 1, 0, 1, 0, 1, 1],  # check 2
        [0, 0, 1, 1, 1, 0, 1],  # check 3
    ],
    dtype=int,
)


# ==============================
# 2. CSS structure for Steane
# ==============================

def steane_parity_check_matrices() -> tuple[np.ndarray, np.ndarray]:
    """
    Return (H_x, H_z) for the Steane CSS code.

    For the Steane [[7,1,3]] code we use:
        H_x = H_z = H_HAMMING
    where H_HAMMING is the 3x7 parity-check matrix above.

    Returns
    -------
    Hx : np.ndarray, shape (3, 7)
        Parity-check matrix for X-type stabilizers.
    Hz : np.ndarray, shape (3, 7)
        Parity-check matrix for Z-type stabilizers.
    """
    Hx = H_HAMMING.copy()
    Hz = H_HAMMING.copy()
    return Hx, Hz


# =====================================
# 3. Build stabilizers as Pauli strings
# =====================================

def _row_to_pauli_string(row: np.ndarray, pauli: str) -> str:
    """
    Convert a binary row (length 7) into a Pauli string.

    Parameters
    ----------
    row : np.ndarray
        Binary vector of length 7 (0/1).
    pauli : {"X", "Z"}
        Pauli type to place on positions with 1's.

    Returns
    -------
    s : str
        Pauli string of length 7 over {I, X, Z}.
    """
    s = []
    for bit in row:
        if bit == 1:
            s.append(pauli)
        else:
            s.append("I")
    return "".join(s)


def steane_stabilizer_paulis() -> dict[str, list[SparsePauliOp]]:
    """
    Build the 6 stabilizer generators of the Steane code as SparsePauliOp.

    Using the CSS structure:
    - 3 X-type stabilizers from rows of Hx.
    - 3 Z-type stabilizers from rows of Hz.

    Returns
    -------
    stabs : dict
        {
          "X":  [S1_X, S2_X, S3_X],      # each is SparsePauliOp
          "Z":  [S1_Z, S2_Z, S3_Z],
          "all": [S1_X, S2_X, S3_X, S1_Z, S2_Z, S3_Z]
        }
    """
    Hx, Hz = steane_parity_check_matrices()

    x_stabs = []
    z_stabs = []

    # X-type stabilizers
    for row in Hx:
        pstr = _row_to_pauli_string(row, "X")
        x_stabs.append(SparsePauliOp.from_list([(pstr, 1.0)]))

    # Z-type stabilizers
    for row in Hz:
        pstr = _row_to_pauli_string(row, "Z")
        z_stabs.append(SparsePauliOp.from_list([(pstr, 1.0)]))

    return {
        "X": x_stabs,
        "Z": z_stabs,
        "all": x_stabs + z_stabs,
    }


# ==========================
# 4. Helper pretty-print
# ==========================

def print_steane_stabilizers() -> None:
    """
    Print the 6 stabilizer generators as Pauli strings.
    Useful for quick sanity checks.
    """
    Hx, Hz = steane_parity_check_matrices()

    print("X-type stabilizers:")
    for i, row in enumerate(Hx):
        print(f"S_X[{i}] =", _row_to_pauli_string(row, "X"))

    print("\nZ-type stabilizers:")
    for i, row in enumerate(Hz):
        print(f"S_Z[{i}] =", _row_to_pauli_string(row, "Z"))


if __name__ == "__main__":
    # Simple manual check
    print_steane_stabilizers()


