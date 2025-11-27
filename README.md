# Quantum-Error-Correction-shor-steane...-
Codes and simple analysis on qiskit for the Shor and Steane algorithms. 

---

This repository contains a first-principles implementation of the **Shor [[9,1,3]]** and **Steane [[7,1,3]]** quantum error-correcting codes, developed with the aim of understanding, analyzing their behaviour under realistic noise and error models. Future implementations wiwll include QPE, Grover or others.

The current version focuses on an explicit implementation of the Shor code, including:

- Logical states $|0_L\rangle$ and $|1_L\rangle$
- Stabilizer formalism and syndrome extraction
- Error detection and correction (Pauli $X$, $Y$, $Z$)
- Full encode–error–syndrome–recover–decode pipeline
- Monte-Carlo simulations of logical fidelity
- Statistical analysis for multi–error regimes

The Steane [[7,1,3]] code will be added and analysed following the same structure.

- Implementation in **Qiskit circuits** (gates)
- Realistic noise models (depolarizing, dephasing, amplitude damping)
- Experiments on IBM Quantum backends (simulators and hardware)
- Application to small algorithmic examples (QPE, Grover, etc.)
- Comparison with other error-correcting schemes

Code was assisted by AI.

---



