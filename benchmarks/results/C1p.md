### C1p

| shape | cache | T us | best AITER arm | B us | speedup (min-max) | vs best numerically-passing AITER arm | B37 us | noise % | cell | T num | B num |
|---|---|---:|---|---:|---|---|---:|---:|---|---|---|
| p1024x1 | rotating | 14.25 | B1 | 20.84 | 1.46 (1.46-1.49) | none passes |  | 1.55 | win, T numerics FAIL | FAIL | FAIL |
| p4096x1 | rotating | 71.21 | B1 | 91.48 | 1.28 (1.28-1.30) | none passes |  | 1.54 | win, T numerics FAIL | FAIL | FAIL |
| p4096x4 | rotating | 27.88 | B1 | 31.67 | 1.14 (1.14-1.14) | none passes |  | 0.43 | win, T numerics FAIL | FAIL | FAIL |
| p6143x1 | rotating | 124.65 | B1 | 106.86 | 0.86 (0.86-0.86) | none passes |  | 0.26 | loss, T numerics FAIL | FAIL | FAIL |
| p8192x1 | rotating | 183.05 | B1 | 160.78 | 0.88 (0.88-0.88) | none passes |  | 0.9 | loss, T numerics FAIL | FAIL | FAIL |
| p8192x1+8192 | rotating | 494.79 | B1 | 460.51 | 0.93 (0.93-0.93) | B1 0.93 |  | 0.16 | loss | pass | pass |

rotating: uniform geomean speedup vs best AITER arm = 1.069 over 6 shapes

calibration (copy GB/s, mm TFLOP/s) per block: n38/n38_b1: 5025, 1452; n38/n38_b2: 5043, 1436; n38/n38_b3: 5031, 1446
