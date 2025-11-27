# utils_states.py
# -*- coding: utf-8 -*-
"""
Utilities for analyzing Steane logical states.

- State fidelity
- Projection onto the logical subspace span{|0_L>, |1_L>}
- Logical components (alpha, beta) and leakage
- Logical fidelity to target logical states (0L,1L,+L,-L)
- Logical error indicator for Monte Carlo simulations
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

# Import from your local module (adjust if you use a package structure)
from steane_encoder import steane_logical_states


# =========================================
# 1. Basic fidelity between pure states
# =========================================

def state_fidelity(
    psi: Statevector,
    phi: Statevector,
) -> float:
    """
    Compute the fidelity F(psi, phi) = |<psi|phi>|^2 for pure states.

    Parameters
    ----------
    psi : Statevector
        First pure state.
    phi : Statevector
        Second pure state.

    Returns
    -------
    F : float
        Fidelity in [0, 1].
    """
    # Ensure both are Statevector
    psi_sv = Statevector(psi)
    phi_sv = Statevector(phi)

    overlap = np.vdot(psi_sv.data, phi_sv.data)
    return float(np.abs(overlap) ** 2)


# =========================================
# 2. Logical basis and logical + / - states
# =========================================

def logical_basis_states() -> Dict[str, Statevector]:
    """
    Returns the logical basis states and the logical +/- states.

    Returns
    -------
    states : dict
        {
          "0L": |0_L>,
          "1L": |1_L>,
          "+L": (|0_L> + |1_L>)/sqrt(2),
          "-L": (|0_L> - |1_L>)/sqrt(2)
        }
    """
    base = steane_logical_states()
    psi0 = base["0L"]
    psi1 = base["1L"]

    # Build |+_L> and |-_L>
    plus = (psi0.data + psi1.data) / np.sqrt(2.0)
    minus = (psi0.data - psi1.data) / np.sqrt(2.0)

    return {
        "0L": psi0,
        "1L": psi1,
        "+L": Statevector(plus),
        "-L": Statevector(minus),
    }


# =========================================
# 3. Projection onto the logical subspace
# =========================================

def logical_components(
    state: Statevector,
) -> Tuple[complex, complex, float]:
    """
    Project a 7-qubit state onto the logical subspace span{|0_L>, |1_L>}.

    We write:
        |psi> = alpha |0_L> + beta |1_L> + |leakage>,
    where |leakage> is orthogonal to both |0_L>, |1_L>.

    This function returns (alpha, beta, leakage_prob), where
    leakage_prob = 1 - (|alpha|^2 + |beta|^2).

    Parameters
    ----------
    state : Statevector
        Arbitrary 7-qubit state (not necessarily normalized
        if coming from numerical noise; we renormalize).

    Returns
    -------
    alpha : complex
        Coefficient of |0_L>.
    beta : complex
        Coefficient of |1_L>.
    leakage_prob : float
        Probability weight outside the logical subspace.
    """
    # Normalize input state to be safe
    sv = Statevector(state)
    sv = sv / np.sqrt(np.vdot(sv.data, sv.data))

    base = steane_logical_states()
    psi0 = base["0L"]
    psi1 = base["1L"]

    # Ensure logical states are normalized too (they should be)
    psi0 = psi0 / np.sqrt(np.vdot(psi0.data, psi0.data))
    psi1 = psi1 / np.sqrt(np.vdot(psi1.data, psi1.data))

    # Compute overlaps alpha = <0_L|psi>, beta = <1_L|psi>
    alpha = np.vdot(psi0.data, sv.data)
    beta = np.vdot(psi1.data, sv.data)

    prob_logical = np.abs(alpha) ** 2 + np.abs(beta) ** 2
    leakage_prob = float(max(0.0, 1.0 - prob_logical))

    return alpha, beta, leakage_prob


# =========================================
# 4. Logical fidelity to a target logical state
# =========================================

def logical_fidelity_to(
    state: Statevector,
    target_label: str = "0L",
) -> float:
    """
    Compute the fidelity of `state` to a target logical state
    (0L, 1L, +L, -L).

    Parameters
    ----------
    state : Statevector
        7-qubit state to be analyzed.
    target_label : {"0L", "1L", "+L", "-L"}
        Target logical state.

    Returns
    -------
    F : float
        Fidelity in [0,1] with the chosen logical state.
    """
    all_logical = logical_basis_states()
    if target_label not in all_logical:
        raise ValueError(
            f"Unknown logical label '{target_label}'. "
            f"Use one of: {list(all_logical.keys())}"
        )

    target_state = all_logical[target_label]
    return state_fidelity(state, target_state)


# =========================================
# 5. Logical error indicator (for Monte Carlo)
# =========================================

def is_logical_error(
    state: Statevector,
    target_label: str = "0L",
    success_threshold: float = 0.99,
) -> bool:
    """
    Decide whether a run is a logical success or failure,
    based on fidelity to a target logical state.

    By default:
      - "success" if F >= success_threshold
      - "logical error" (failure) if F < success_threshold

    Parameters
    ----------
    state : Statevector
        Final 7-qubit state after encode + noise + correction (+ gates).
    target_label : {"0L", "1L", "+L", "-L"}
        Desired logical state for this experiment.
    success_threshold : float
        Minimum fidelity considered as a successful logical run.

    Returns
    -------
    is_error : bool
        True if the run is considered a logical error, False otherwise.
    """
    F = logical_fidelity_to(state, target_label=target_label)
    return bool(F < success_threshold)


if __name__ == "__main__":
    # Small manual test
    logical = logical_basis_states()
    psi0 = logical["0L"]
    psi1 = logical["1L"]
    plusL = logical["+L"]

    print("F(0L, 0L) =", state_fidelity(psi0, psi0))
    print("F(0L, 1L) =", state_fidelity(psi0, psi1))
    print("F(+L, 0L) =", logical_fidelity_to(plusL, "0L"))

    alpha, beta, leak = logical_components(psi0)
    print("Components of |0_L> itself:")
    print(" alpha =", alpha)
    print(" beta  =", beta)
    print(" leakage =", leak)
