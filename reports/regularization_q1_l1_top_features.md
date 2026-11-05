# Regularization - top features of the best L1 models

Note: one-hot specification columns (`cat__*`, kind 'one-hot, NOT scaled') are 0/1 and not standardized, so their coefficients are not on the same scale as the standardized numeric columns; compare them only with each other.

## l1_C0.1

| rank | feature | kind | mean coef | mean abs coef | non-zero in folds |
|---|---|---|---|---|---|
| 1 | num_log1p__397_19 | numeric, log1p + scaled | +2.6036 | 2.6036 | 5/5 |
| 2 | cat__Spec_6_Cat7 | one-hot, NOT scaled | -2.2547 | 2.2547 | 5/5 |
| 3 | num_log1p__835_0 | numeric, log1p + scaled | +2.1380 | 2.1380 | 5/5 |
| 4 | cat__Spec_6_Cat11 | one-hot, NOT scaled | +1.4244 | 1.4244 | 4/5 |
| 5 | num_log1p__397_13 | numeric, log1p + scaled | -1.2748 | 1.2748 | 5/5 |
| 6 | cat__Spec_2_Cat14 | one-hot, NOT scaled | +1.2668 | 1.2668 | 5/5 |
| 7 | num_log1p__459_9 | numeric, log1p + scaled | +1.2190 | 1.2190 | 5/5 |
| 8 | cat__Spec_6_Cat6 | one-hot, NOT scaled | +1.1761 | 1.1761 | 4/5 |
| 9 | cat__Spec_2_Cat16 | one-hot, NOT scaled | +1.1709 | 1.1709 | 4/5 |
| 10 | num_log1p__397_9 | numeric, log1p + scaled | -1.0523 | 1.0523 | 5/5 |

## l1_C0.1_engineered

| rank | feature | kind | mean coef | mean abs coef | non-zero in folds |
|---|---|---|---|---|---|
| 1 | num_log1p__272_3 | numeric, log1p + scaled | -0.8646 | 0.8646 | 5/5 |
| 2 | num_log1p__167_1 | numeric, log1p + scaled | -0.7668 | 0.7668 | 5/5 |
| 3 | num_log1p__100_0 | numeric, log1p + scaled | +0.6289 | 0.6289 | 5/5 |
| 4 | cat__Spec_7_Cat2 | one-hot, NOT scaled | -0.6157 | 0.6157 | 5/5 |
| 5 | num_log1p__397_12 | numeric, log1p + scaled | +0.6129 | 0.6129 | 5/5 |
| 6 | num_log1p__397_10 | numeric, log1p + scaled | +0.5515 | 0.5515 | 5/5 |
| 7 | num_log1p__397_3 | numeric, log1p + scaled | +0.5456 | 0.5456 | 5/5 |
| 8 | num__397_19__share | numeric, scaled | +0.5402 | 0.5402 | 5/5 |
| 9 | num_log1p__397_9 | numeric, log1p + scaled | -0.5333 | 0.5333 | 5/5 |
| 10 | cat__Spec_1_Cat14 | one-hot, NOT scaled | -0.5088 | 0.5088 | 5/5 |
