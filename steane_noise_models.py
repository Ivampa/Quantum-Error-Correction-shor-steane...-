# steane_noise_models.py
# -*- coding: utf-8 -*-
"""
Noise models for the Steane [[7,1,3]] code.

Supported (IID single-qubit) Pauli noise:
    * X-only        -> ("X",)
    * Z-only        -> ("Z",)
    * Depolarizing  -> ("X","Y","Z")

This module generates random physical error patterns on 7 qubits
and applies them to a QuantumCircuit.
"""

from __future__ import annotations

import random
from typing import List, Tuple
from qiskit import QuantumCircuit

PauliError = Tuple[int, str]  # (qubit_index, "X"/"Y"/"Z")


def sample_iid_pauli_errors(
    num_qubits: int,
    p: float,
    pauli_set: Tuple[str, ...] = ("X", "Y", "Z"),
) -> List[PauliError]:
    """
    Samples an IID Pauli error pattern on `num_qubits`.

    For each qubit:
      - with probability p, a random Pauli from `pauli_set` is applied
      - with probability (1 - p), no error is applied

    Examples
    --------
    Bit-flip noise only:
        sample_iid_pauli_errors(7, p, ("X",))

    Phase-flip noise only:
        sample_iid_pauli_errors(7, p, ("Z",))

    Depolarizing (Pauli-twirled) noise:
        sample_iid_pauli_errors(7, p, ("X","Y","Z"))

    Parameters
    ----------
    num_qubits : int
        Number of physical qubits (for Steane: 7).
    p : float
        Error probability per qubit.
    pauli_set : tuple of {"X","Y","Z"}
        Allowed Pauli errors.

    Returns
    -------
    errors : list of (int, str)
        List of errors: [(i, "X"), (j, "Z"), ...]
    """
    errors: List[PauliError] = []

    for q in range(num_qubits):
        if random.random() < p:
            pauli = random.choice(pauli_set)
            errors.append((q, pauli))

    return errors


def apply_pauli_errors_to_circuit(
    qc: QuantumCircuit,
    errors: List[PauliError],
) -> None:
    """
    Applies a list of Pauli errors in-place to a Qiskit circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit where the error gates will be inserted.
    errors : list of (int, str)
        List of (qubit_index, "X"/"Y"/"Z").
    """
    for q, pauli in errors:
        if pauli == "X":
            qc.x(q)
        elif pauli == "Y":
            qc.y(q)
        elif pauli == "Z":
            qc.z(q)
        else:
            raise ValueError(f"Unknown Pauli operator: {pauli}")
