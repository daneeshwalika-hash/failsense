# Outlier analysis - Q1 training cut-offs

117,031 sampled cut-offs (k = 5 per truck), training split only. Overall repair rate (label = repair within the horizon): **0.0276**.
Outlier = above Q3 + 3 x IQR. Nothing was removed or clipped.

## Most skewed features

| feature | skew | p50 | p99 | max |
|---|---|---|---|---|
| 272_9__share | 342.0 | 0 | 4.55e-06 | 1 |
| 272__top_share | 342.0 | 0 | 1.68e-05 | 1 |
| 397_23 | 342.0 | 267 | 1.15e+04 | 3.42e+10 |
| 397_11 | 342.0 | 716 | 2.83e+04 | 3.31e+10 |
| 397_28 | 342.0 | 8.9e+03 | 2.97e+05 | 3.42e+10 |

## Repair rate in outlier vs. normal rows (top 10 by outlier share)

| feature | outlier rows | rate outlier | rate normal | ratio |
|---|---|---|---|---|
| 459_16__share | 25,057 | 0.0337 | 0.0261 | 1.29 |
| 459__top_share | 24,930 | 0.0338 | 0.0261 | 1.30 |
| 459_16 | 22,590 | 0.0378 | 0.0253 | 1.49 |
| 370_0__delta | 20,727 | 0.0317 | 0.0282 | 1.12 |
| 370_0__rate | 20,216 | 0.0315 | 0.0283 | 1.11 |
| 370_0 | 19,847 | 0.0405 | 0.0250 | 1.62 |
| 167_9 | 18,466 | 0.0525 | 0.0229 | 2.29 |
| 167_9__share | 18,350 | 0.0405 | 0.0252 | 1.61 |
| 272_3__share | 17,791 | 0.0393 | 0.0255 | 1.54 |
| 309_0__rate | 15,085 | 0.0485 | 0.0258 | 1.88 |
