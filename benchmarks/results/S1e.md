### S1e

| shape | cache | T us | best AITER arm | B us | speedup (min-max) | vs best numerically-passing AITER arm | B37 us | noise % | cell | T num | B num |
|---|---|---:|---|---:|---|---|---:|---:|---|---|---|
| m4 | hot | 5.61 | B1 | 8.58 | 1.53 (1.52-1.55) | B1 1.53 |  | 3.92 | win | pass | pass |
| m4 | rotating | 8.72 | B1 | 9.60 | 1.10 (1.10-1.11) | B1 1.10 |  | 0.88 | win | pass | pass |
| m8 | hot | 5.89 | B1 | 9.37 | 1.59 (1.59-1.63) | B1 1.59 |  | 3.08 | win | pass | pass |
| m8 | rotating | 8.93 | B1 | 10.43 | 1.17 (1.17-1.18) | B1 1.17 |  | 0.4 | win | pass | pass |
| m16 | hot | 6.44 | B1 | 9.27 | 1.44 (1.44-1.44) | none passes |  | 0.44 | win | pass | FAIL |
| m16 | rotating | 9.61 | B1 | 10.85 | 1.13 (1.12-1.13) | none passes |  | 0.22 | win | pass | FAIL |

hot: weighted geomean speedup vs best AITER arm = 1.518 over 3 shapes

rotating: weighted geomean speedup vs best AITER arm = 1.132 over 3 shapes

calibration (copy GB/s, mm TFLOP/s) per block: n38/n38_b1: 5506, 1435; n38/n38_b2: 5511, 1454; n38/n38_b3: 5503, 1456
