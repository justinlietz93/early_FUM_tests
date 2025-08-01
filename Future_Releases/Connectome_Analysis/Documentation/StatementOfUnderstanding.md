# Statement of Shared Understanding

**Version:** 1.0
**Date:** 2025-07-28

This document serves as the definitive, synchronized understanding between the user and the AI assistant regarding the FUM project. Its purpose is to align our perspectives on the project's goals, the current state of the codebase, and the go-forward plan to prevent further a-strategic work.

---

### Part 1: The Core Goal of the FUM

The fundamental goal of the FUM is not to simply build a large neural network, but to create the conditions for a new kind of intelligence to **emerge**. The FUM is conceptualized as a self-organizing, self-regulating, and self-improving digital organism. The measure of success is not scale, but the demonstration of genuine capability in learning, reasoning, and autonomous self-maintenance. The guiding principle is **emergence over engineering**.

---

### Part 2: The Current State of the Code (`fum/` directory)

The `fum/` directory contains a set of refactored, blueprint-compliant software modules. These modules represent the fundamental "physics" of the FUM universe.

*   **Substrate (`fum/substrate/`):** A data-centric class intended to hold the complete state of the network. **Crucially, we have identified that this substrate is currently incomplete as it is missing key neuron parameters (e.g., `tau_m`, `v_thresh`) required for a functional neuron model.**
*   **Mechanisms (`fum/mechanisms/`):** A suite of stateless functions (`revgsp`, `gdsp`, `ehtp`, `adc`) that correctly implement the learning and homeostatic rules in a maximally efficient, sparse-matrix-compliant manner.
*   **I/O System (`fum/sensors`, `fum/actuators`, `ute`, `utd`):** A modular and blueprint-compliant pipeline for translating external data into spikes (via sensors and the UTE) and translating output spikes into actions (via the UTD and actuators).
*   **Nexus (`fum/nexus/`):** The clean, high-level entry point for an external process to interact with a FUM instance. It is correctly designed to initialize the system and manage its core state.

**Conclusion on Code State:** The individual components are now largely correct, but they are not yet unified into a functioning whole because critical pieces of the substrate and the overarching lifecycle management are missing.

---

### Part 3: The Missing Piece - The Process Logic

Our analysis of `fum_v1_main.py` revealed that our previous understanding of "training" was fundamentally flawed. The FUM is not "trained" in a conventional sense; it is **guided through a lifecycle of self-assembly.** The V1 script revealed the indispensable process logic for this lifecycle:

1.  **Rigorous Validation:** The process must begin with a "pre-flight check" to validate that the initial state of the system is healthy and that the "primordial soup" of input data is sufficiently diverse.
2.  **Integrated Homeostasis:** The core processing loop is not just `stimulus -> learn`. It is a dynamic feedback cycle of `stimulus -> learn -> analyze -> self-regulate`. The various homeostatic mechanisms (structural, intrinsic, synaptic) are not optional; they are the system's metabolism, essential for preventing it from spiraling into chaos or inactivity.
3.  **External Orchestration:** This entire lifecycle is managed by an **external training script**, not by the `FUMNexus` itself. The Nexus is the organism; the training script is the "environment" or "scientist" that guides it.

**Overarching Conclusion:** Our path forward is not to add more disconnected features, but to **build the coherent, V1-inspired training harness** that can shepherd our V2 `fum` components through the necessary lifecycle of validation, guided self-assembly, and homeostatic self-regulation. We must first complete the substrate before this can begin.