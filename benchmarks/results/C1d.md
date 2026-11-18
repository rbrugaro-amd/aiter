### C1d

| shape | cache | T us | best AITER arm | B us | speedup (min-max) | vs best numerically-passing AITER arm | B37 us | noise % | cell | T num | B num |
|---|---|---:|---|---:|---|---|---:|---:|---|---|---|
| m16c16k | hot | 25.18 | B1 | 29.86 | 1.19 (1.18-1.20) | B1 1.19 |  | 1.39 | win | pass | pass |
| m16c16k | rotating | 25.41 | B1 | 31.88 | 1.25 (1.24-1.28) | B1 1.25 |  | 2.53 | win | pass | pass |
| m1c16k | hot | 6.78 | B1 | 10.32 | 1.52 (1.51-1.53) | B1 1.52 |  | 0.98 | win | pass | pass |
| m1c16k | rotating | 7.29 | B1 | 10.49 | 1.44 (1.44-1.45) | B1 1.44 |  | 0.65 | win | pass | pass |
| m1c2k | hot | 6.62 | B1 | 8.74 | 1.32 (1.32-1.33) | B1 1.32 |  | 0.47 | win | pass | pass |
| m1c2k | rotating | 6.62 | B1 | 8.87 | 1.34 (1.32-1.34) | B1 1.34 |  | 1.0 | win | pass | pass |
| m1c64k | hot | 10.84 | B1 | 15.43 | 1.42 (1.41-1.44) | B1 1.42 |  | 1.43 | win | pass | pass |
| m1c64k | rotating | 12.03 | B1 | 15.63 | 1.30 (1.28-1.30) | B1 1.30 |  | 1.11 | win | pass | pass |
| m2c16k | hot | 6.89 | B2 | 10.91 | 1.58 (1.56-1.59) | B2 1.58 |  | 1.6 | win | pass | pass |
| m2c16k | rotating | 7.50 | B2 | 11.72 | 1.56 (1.55-1.56) | B2 1.56 |  | 0.91 | win | pass | pass |
| m32c16k | hot | 47.53 | B2 | 51.58 | 1.08 (1.08-1.09) | B2 1.08 |  | 0.86 | win | pass | pass |
| m32c16k | rotating | 47.95 | B1 | 55.11 | 1.15 (1.14-1.16) | B1 1.15 |  | 1.01 | win | pass | pass |
| m4c16k | hot | 9.28 | B2 | 13.41 | 1.45 (1.44-1.47) | B2 1.45 |  | 2.6 | win | pass | pass |
| m4c16k | rotating | 10.22 | B2 | 15.46 | 1.51 (1.50-1.51) | B2 1.51 |  | 1.02 | win | pass | pass |
| m64c16k | hot | 91.22 | B1 | 93.28 | 1.02 (1.02-1.03) | B1 1.02 |  | 0.54 | win | pass | pass |
| m64c16k | rotating | 92.99 | B1 | 94.62 | 1.02 (1.02-1.02) | B1 1.02 |  | 0.28 | parity | pass | pass |
| m64c2k | hot | 15.56 | B1 | 18.53 | 1.19 (1.19-1.20) | B1 1.19 |  | 0.61 | win | pass | pass |
| m64c2k | rotating | 17.24 | B2 | 20.55 | 1.19 (1.19-1.21) | B2 1.19 |  | 0.99 | win | pass | pass |
| m64c64k | hot | 342.25 | B1 | 345.41 | 1.01 (1.00-1.01) | B1 1.01 |  | 0.63 | parity | pass | pass |
| m64c64k | rotating | 342.40 | B1 | 344.49 | 1.01 (1.00-1.01) | B1 1.01 |  | 0.49 | parity | pass | pass |
| m8c16k | hot | 13.71 | B1 | 19.10 | 1.39 (1.35-1.42) | B1 1.39 |  | 10.72 | win | pass | pass |
| m8c16k | rotating | 15.74 | B1 | 21.27 | 1.35 (1.35-1.37) | B1 1.35 |  | 1.32 | win | pass | pass |
| m8c2k | hot | 5.74 | B1 | 9.44 | 1.64 (1.64-1.66) | B1 1.64 |  | 9.37 | win | pass | pass |
| m8c2k | rotating | 6.33 | B2 | 9.94 | 1.57 (1.53-1.58) | B2 1.57 |  | 6.78 | win | pass | pass |
| m8c64k | hot | 46.51 | B1 | 53.35 | 1.15 (1.14-1.15) | B1 1.15 |  | 0.34 | win | pass | pass |
| m8c64k | rotating | 47.55 | B1 | 55.47 | 1.17 (1.16-1.18) | B1 1.17 |  | 0.87 | win | pass | pass |

hot: uniform geomean speedup vs best AITER arm = 1.289 over 13 shapes

rotating: uniform geomean speedup vs best AITER arm = 1.284 over 13 shapes

calibration (copy GB/s, mm TFLOP/s) per block: n38/n38_b1: 5498, 1445; n38/n38_b2: 5045, 1435; n38/n38_b3: 5020, 1462
