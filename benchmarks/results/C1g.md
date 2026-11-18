### C1g

| shape | cache | T us | best AITER arm | B us | speedup (min-max) | vs best numerically-passing AITER arm | B37 us | noise % | cell | T num | B num |
|---|---|---:|---|---:|---|---|---:|---:|---|---|---|
| g1 | hot | 15.28 | B1 | 36.16 | 2.37 (2.31-2.39) | B1 2.37 | 36.35 | 3.94 | win | pass | pass |
| g1 | rotating | 15.40 | B1 | 36.02 | 2.34 (2.32-2.34) | B1 2.34 | 36.15 | 1.22 | win | pass | pass |
| g4 | hot | 33.90 | B1 | 65.90 | 1.94 (1.94-1.95) | B1 1.94 | 66.07 | 0.37 | win | pass | pass |
| g4 | rotating | 33.92 | B1 | 65.95 | 1.94 (1.94-1.95) | B1 1.94 | 66.11 | 0.2 | win | pass | pass |
| g8 | hot | 73.21 | B1 | 109.86 | 1.50 (1.50-1.51) | B1 1.50 | 110.11 | 0.79 | win | pass | pass |
| g8 | rotating | 73.29 | B1 | 110.00 | 1.50 (1.50-1.51) | B1 1.50 | 110.22 | 0.39 | win | pass | pass |

hot: uniform geomean speedup vs best AITER arm = 1.904 over 3 shapes

rotating: uniform geomean speedup vs best AITER arm = 1.897 over 3 shapes

calibration (copy GB/s, mm TFLOP/s) per block: n37/n37_b1: 5035, 1449; n37/n37_b2: 5044, 1445; n37/n37_b3: 5036, 1441; n38/n38_b1: 5039, 1450; n38/n38_b2: 5032, 1441; n38/n38_b3: 5012, 1450

errors/unavailable:
- n37 g1 B: unavailable: paged_attention_output_gate_supported: Triton >= 3.8 required for this Gluon dialect, got 3.7.0
- n38 g1 B: unavailable: paged_attention_output_gate_supported: this Gluon build predates shared_memory_descriptor.reinterpret (Triton 3.8.0)
