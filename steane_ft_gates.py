# steane_ft_gates.py
# -*- coding: utf-8 -*-
"""
Fault-tolerant logical gates for the Steane [[7,1,3]] code.

Implements the standard transversal Clifford gates:
    - Logical Hadamard:      H_L = H^{⊗7}
    - Logical Phase (S gate): S_L = S^{⊗7}
    - Logical CNOT:          CNOT_L = CNOT applied pairwise between code blocks

These functions operate on blocks of 7 physical qubits that encode
one logical qubit in the Steane code.
"""

from __future__ import annotations

from typing import Sequence
from qiskit import QuantumCircuit

STEANE_BLOCK_SIZE = 7


# =========================================
# 1. Single-logical-qubit transversal gates
# =========================================

def apply_logical_H(
    qc: QuantumCircuit,
    block: Sequence[int],
) -> None:
    """
    Apply the transversal logical Hadamard H_L = H^{⊗7}
    on a single Steane code block.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit where the logical operation is applied.
    block : sequence of int
        List or tuple of 7 physical qubit indices that form the Steane block.
        The order must match the encoding convention used in the code.
    """
    if len(block) != STEANE_BLOCK_SIZE:
        raise ValueError(
            f"Steane block must have size {STEANE_BLOCK_SIZE}, "
            f"got {len(block)}."
        )

    for q in block:
        qc.h(q)


def apply_logical_S(
    qc: QuantumCircuit,
    block: Sequence[int],
) -> None:
    """
    Apply the transversal logical phase gate S_L = S^{⊗7}
    on a single Steane code block.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit where the logical operation is applied.
    block : sequence of int
        List or tuple of 7 physical qubit indices that form the Steane block.
    """
    if len(block) != STEANE_BLOCK_SIZE:
        raise ValueError(
            f"Steane block must have size {STEANE_BLOCK_SIZE}, "
            f"got {len(block)}."
        )

    for q in block:
        qc.s(q)


# =========================================
# 2. Two-logical-qubit transversal CNOT
# =========================================

def apply_logical_CNOT(
    qc: QuantumCircuit,
    control_block: Sequence[int],
    target_block: Sequence[int],
) -> None:
    """
    Apply the transversal logical CNOT between two Steane code blocks.

    CNOT_L is implemented as 7 physical CNOTs between corresponding qubits:
        CNOT_L(|ψ_L>_ctrl ⊗ |φ_L>_tgt)
        = ⊗_{i=0..6} CNOT(control_block[i], target_block[i])

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit where the logical operation is applied.
    control_block : sequence of int
        Physical qubit indices (length 7) of the control logical block.
    target_block : sequence of int
        Physical qubit indices (length 7) of the target logical block.
    """
    if len(control_block) != STEANE_BLOCK_SIZE:
        raise ValueError(
            f"Control block must have size {STEANE_BLOCK_SIZE}, "
            f"got {len(control_block)}."
        )
    if len(target_block) != STEANE_BLOCK_SIZE:
        raise ValueError(
            f"Target block must have size {STEANE_BLOCK_SIZE}, "
            f"got {len(target_block)}."
        )

    for qc_ctrl, qc_tgt in zip(control_block, target_block):
        qc.cx(qc_ctrl, qc_tgt)


# =========================================
# 3. Convenience wrappers that build small circuits
# =========================================

def logical_H_circuit() -> QuantumCircuit:
    """
    Return a 7-qubit circuit that applies H_L on a single Steane block.

    Assumes that qubits [0..6] are already in an encoded logical state.
    """
    qc = QuantumCircuit(STEANE_BLOCK_SIZE, name="H_L_transversal")
    apply_logical_H(qc, block=range(STEANE_BLOCK_SIZE))
    return qc


def logical_S_circuit() -> QuantumCircuit:
    """
    Return a 7-qubit circuit that applies S_L on a single Steane block.

    Assumes that qubits [0..6] are already in an encoded logical state.
    """
    qc = QuantumCircuit(STEANE_BLOCK_SIZE, name="S_L_transversal")
    apply_logical_S(qc, block=range(STEANE_BLOCK_SIZE))
    return qc


def logical_CNOT_circuit() -> QuantumCircuit:
    """
    Return a 14-qubit circuit that applies CNOT_L between two Steane blocks.

    Assumes that:
      - qubits [0..6]  encode the control logical qubit
      - qubits [7..13] encode the target logical qubit
    """
    qc = QuantumCircuit(2 * STEANE_BLOCK_SIZE, name="CNOT_L_transversal")
    ctrl_block = range(0, STEANE_BLOCK_SIZE)
    tgt_block = range(STEANE_BLOCK_SIZE, 2 * STEANE_BLOCK_SIZE)
    apply_logical_CNOT(qc, control_block=ctrl_block, target_block=tgt_block)
    return qc


if __name__ == "__main__":
    # Tiny sanity check: draw the transversal gates
    print("H_L transversal:")
    print(logical_H_circuit())

    print("\nS_L transversal:")
    print(logical_S_circuit())

    print("\nCNOT_L transversal:")
    print(logical_CNOT_circuit())

