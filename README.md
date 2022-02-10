# Quantize-leakage-using-Leakage-Randomized-Benchmarking

Implement Leakage randomized benchmarking (LRB) using Qiskit Experiments in Korean Qiskit Hackathon 2022

-----------
Description
We define transmon qubits by limiting our attention to $|0\rangle$ and $|1\rangle$ states, ignoring all the other excited states. This approach works due to slight anharmonicity of transmons, but there can be always “leakage”, states in the computational space excited to the leakage space, and “seepage”, states in the leakage space relaxed into the computational space.
Such leakage and seepage are significant source of errors because they are usually not considered in quantum error correction. If one wants to correct them in a fault-tolerant way, significant hardware resources have to be dedicated to detect and correct leakage and seepage.
Now that we know how malicious leekage and seepage are, we should see how much leakage and seepage we have in our gate set. Such protocol is introduced in Ref where authors slightly modified standard randomized benchmarking. In this project, we will implement leakage randomized benchmarking in Ref using Qiskit Experiments.

Ref: Quantification and characterization of leakage errors, Wood et al., PRA.
