### S1g

| shape | cache | T us | best AITER arm | B us | speedup (min-max) | B37 us | noise % | cell | T num | B num |
|---|---|---:|---|---:|---|---:|---:|---|---|---|
| t4m1 | hot | 5.10 | B1 | 8.33 | 1.63 (1.62-1.68) |  | 3.12 | win | pass | pass |
| t4m1 | rotating | 8.71 | B1 | 8.90 | 1.02 (1.02-1.03) |  | 1.08 | win | pass | pass |
| t4m2 | hot | 4.85 | B1 | 8.65 | 1.78 (1.76-1.79) |  | 4.39 | win | pass | pass |
| t4m2 | rotating | 8.54 | B1 | 9.20 | 1.08 (1.07-1.08) |  | 0.85 | win | pass | pass |
| t4m4 | hot | 5.36 | B1 | 9.02 | 1.68 (1.62-1.75) |  | 6.35 | win | pass | pass |
| t4m4 | rotating | 8.75 | B1 | 9.62 | 1.10 (1.09-1.10) |  | 1.24 | win | pass | pass |
| t4m8 | hot | 5.56 | B1 | 9.83 | 1.77 (1.76-1.80) |  | 2.0 | win | pass | pass |
| t4m8 | rotating | 9.04 | B1 | 10.44 | 1.16 (1.15-1.16) |  | 0.3 | win | pass | pass |
| t4m16 | hot | 6.27 | B1 | 9.19 | 1.47 (1.47-1.48) |  | 2.77 | win | pass | FAIL |
| t4m16 | rotating | 9.01 | B1 | 10.83 | 1.20 (1.20-1.21) |  | 0.28 | win | pass | FAIL |
| t4m32 | hot | 6.09 | B1 | 9.60 | 1.58 (1.57-1.59) |  | 1.02 | win | pass | FAIL |
| t4m32 | rotating | 9.50 | B1 | 11.16 | 1.18 (1.17-1.18) |  | 0.82 | win | pass | FAIL |
| t4m64 | hot | 7.51 | B1 | 10.13 | 1.35 (1.32-1.37) |  | 2.26 | win | pass | FAIL |
| t4m64 | rotating | 11.22 | B1 | 12.46 | 1.11 (1.11-1.11) |  | 0.24 | win | pass | FAIL |
| t4m128 | hot | 10.27 | B1 | 15.74 | 1.53 (1.53-1.54) |  | 0.83 | win | pass | FAIL |
| t4m128 | rotating | 14.29 | B1 | 16.88 | 1.18 (1.17-1.18) |  | 0.78 | win | pass | FAIL |
| t4m256 | hot | 15.06 | B1 | 21.06 | 1.40 (1.39-1.42) |  | 1.34 | win | pass | FAIL |
| t4m256 | rotating | 20.60 | B1 | 23.37 | 1.13 (1.13-1.15) |  | 0.95 | win | pass | FAIL |
| t8m1 | hot | 5.04 | B1 | 8.34 | 1.65 (1.59-1.69) |  | 4.94 | win | pass | pass |
| t8m1 | rotating | 9.36 | B1 | 8.96 | 0.96 (0.95-0.96) |  | 1.23 | loss | pass | pass |
| t8m8 | hot | 5.91 | B1 | 9.39 | 1.59 (1.58-1.61) |  | 1.48 | win | pass | pass |
| t8m8 | rotating | 8.90 | B1 | 10.46 | 1.17 (1.17-1.18) |  | 0.65 | win | pass | pass |
| t8m64 | hot | 7.48 | B1 | 10.13 | 1.35 (1.34-1.36) |  | 1.33 | win | pass | FAIL |
| t8m64 | rotating | 11.37 | B1 | 12.46 | 1.10 (1.09-1.10) |  | 0.26 | win | pass | FAIL |
| t8m256 | hot | 14.59 | B1 | 21.11 | 1.45 (1.44-1.45) |  | 4.3 | win | pass | FAIL |
| t8m256 | rotating | 19.96 | B | 23.39 | 1.17 (1.17-1.18) |  | 1.06 | win | pass | FAIL |
| t8m1024 | rotating | 43.78 | B1 | 56.30 | 1.29 (1.28-1.29) |  | 1.77 | win | pass | pass |
| t8m4193 | rotating | 118.14 | B1 | 161.00 | 1.36 (1.36-1.37) |  | 0.66 | win | pass | pass |
| t8m8192 | rotating | 232.19 | B1 | 248.16 | 1.07 (1.07-1.07) |  | 0.44 | win | pass | pass |
| t8m16384 | rotating | 475.40 | B1 | 429.33 | 0.90 (0.90-0.91) |  | 0.68 | loss | pass | pass |

hot: weighted geomean speedup vs best AITER arm = 1.570 over 9 shapes

rotating: weighted geomean speedup vs best AITER arm = 1.127 over 9 shapes

calibration (copy GB/s, mm TFLOP/s) per block: n38/n38_b1: 5500, 1451; n38/n38_b2: 5498, 1435; n38/n38_b3: 5509, 1460
