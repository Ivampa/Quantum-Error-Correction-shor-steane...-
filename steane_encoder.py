# steane_encoder.py
# -*- coding: utf-8 -*-
"""
Steane [[7,1,3]] code – logical states and simple encoders.

- Builds the classical [7,4,3] Hamming code (generator matrix G).
- Generates all classical codewords C = {u G mod 2}.
- Splits them into even-weight (C_even) and odd-weight (C_odd) codewords.
- Defines the logical states:
    |0_L> ∝ sum_{c in C_even} |c>
    |1_L> ∝ sum_{c in C_odd}  |c>
- Provides functions to obtain the Statevectors |0_L>, |1_L>
  and circuits that prepare them from |0000000>.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# =========================================
# 1. Generator matrix G of the [7,4,3] Hamming code
#    (such that H * G^T = 0 mod 2, where H is in steane_stabilizers)
# =========================================

G_HAMMING = np.array(
    [
        [0, 1, 1, 1, 0, 0, 0],  # g1
        [1, 0, 1, 0, 1, 0, 0],  # g2
        [1, 1, 0, 0, 0, 1, 0],  # g3
        [1, 1, 1, 0, 0, 0, 1],  # g4
    ],
    dtype=int,
)


def hamming_generator_matrix() -> np.ndarray:
    """
    Returns the generator matrix G of the [7,4,3] Hamming code.

    Returns
    -------
    G : np.ndarray, shape (4, 7)
    """
    return G_HAMMING.copy()


# =========================================
# 2. Classical codewords of the [7,4,3] Hamming code
# =========================================

def hamming_codewords() -> list[np.ndarray]:
    """
    Generates all codewords of the classical [7,4,3] Hamming code.

    C = { u G (mod 2) : u in F2^4 }  (16 codewords)

    Returns
    -------
    codewords : list of np.ndarray, each of length 7 with values 0/1
    """
    G = hamming_generator_matrix()
    k, n = G.shape  # k = 4, n = 7

    codewords = []
    seen = set()

    for u_int in range(2**k):
        # vector u of 4 bits (u0,...,u3)
        u_bits = np.array([(u_int >> i) & 1 for i in range(k)], dtype=int)
        # codeword = u G (mod 2)
        cw = u_bits @ G % 2  # shape (7,)

        # Avoid duplicates just in case
        key = tuple(int(b) for b in cw)
        if key not in seen:
            seen.add(key)
            codewords.append(cw)

    return codewords


def split_even_odd_codewords(
    codewords: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Splits codewords into even-weight and odd-weight.

    Parameters
    ----------
    codewords : list of np.ndarray
        Binary codewords of length 7.

    Returns
    -------
    (even_cw, odd_cw) : tuple of lists
        even_cw: list of even-weight codewords.
        odd_cw : list of odd-weight codewords.
    """
    even_cw = []
    odd_cw = []

    for cw in codewords:
        if cw.sum() % 2 == 0:
            even_cw.append(cw)
        else:
            odd_cw.append(cw)

    return even_cw, odd_cw


# =========================================
# 3. Logical states |0_L>, |1_L>
# =========================================

def _bits_to_index(bits: np.ndarray) -> int:
    """
    Converts an array of bits (q0...q6) into the integer index used by Qiskit.

    We assume that bits[0] is the most significant qubit (q0),
    i.e., index = b0*2^6 + b1*2^5 + ... + b6*2^0.
    """
    idx = 0
    n = len(bits)
    for i in range(n):
        idx = (idx << 1) | int(bits[i])
    return idx


def logical_state_from_codewords(
    codewords: list[np.ndarray],
    num_qubits: int = 7,
) -> Statevector:
    """
    Builds a Statevector as a uniform superposition of the given codewords.

    |psi> = (1/sqrt(M)) sum_{c in codewords} |c>,

    where |c> is the computational basis state associated with the bitstring c.

    Parameters
    ----------
    codewords : list[np.ndarray]
        List of binary codewords of length num_qubits.
    num_qubits : int
        Number of qubits (default: 7).

    Returns
    -------
    state : Statevector
        Statevector of dimension 2^num_qubits.
    """
    dim = 2**num_qubits
    state = np.zeros(dim, dtype=complex)

    M = len(codewords)
    if M == 0:
        raise ValueError("The list of codewords is empty.")

    amp = 1.0 / np.sqrt(M)

    for cw in codewords:
        idx = _bits_to_index(cw)
        state[idx] += amp

    return Statevector(state)


def steane_logical_codewords() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns the lists of codewords used for |0_L> and |1_L>.

    Simple CSS construction:
      - |0_L> = superposition of all even-weight codewords.
      - |1_L> = superposition of all odd-weight codewords.

    Returns
    -------
    (cw_0L, cw_1L) : tuple of lists
        cw_0L: list of codewords for |0_L>.
        cw_1L: list of codewords for |1_L>.
    """
    codewords = hamming_codewords()
    even_cw, odd_cw = split_even_odd_codewords(codewords)
    return even_cw, odd_cw


def steane_logical_states() -> dict[str, Statevector]:
    """
    Returns the logical states |0_L> and |1_L> as Statevectors.

    Returns
    -------
    states : dict
        {
          "0L": Statevector(|0_L>),
          "1L": Statevector(|1_L>)
        }
    """
    cw_0L, cw_1L = steane_logical_codewords()
    psi_0L = logical_state_from_codewords(cw_0L, num_qubits=7)
    psi_1L = logical_state_from_codewords(cw_1L, num_qubits=7)
    return {"0L": psi_0L, "1L": psi_1L}


# =========================================
# 4. Preparation circuits for |0_L>, |1_L>
# =========================================

def steane_prepare_0L_circuit() -> QuantumCircuit:
    """
    Returns a 7-qubit QuantumCircuit that prepares |0_L>
    from |0000000> using Qiskit's prepare_state.
    """
    states = steane_logical_states()
    psi_0L = states["0L"]

    qc = QuantumCircuit(7, name="prep_0L")
    qc.prepare_state(psi_0L, qc.qubits)
    return qc


def steane_prepare_1L_circuit() -> QuantumCircuit:
    """
    Returns a 7-qubit QuantumCircuit that prepares |1_L>
    from |0000000> using Qiskit's prepare_state.
    """
    states = steane_logical_states()
    psi_1L = states["1L"]

    qc = QuantumCircuit(7, name="prep_1L")
    qc.prepare_state(psi_1L, qc.qubits)
    return qc


if __name__ == "__main__":
    # Small manual test: dimensions and orthogonality
    states = steane_logical_states()
    psi0 = states["0L"]
    psi1 = states["1L"]

    print("Hilbert space dimension:", len(psi0))
    print("<0_L|0_L> =", np.vdot(psi0.data, psi0.data))
    print("<1_L|1_L> =", np.vdot(psi1.data, psi1.data))
    print("<0_L|1_L> =", np.vdot(psi0.data, psi1.data))
