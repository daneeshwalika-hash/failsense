# Top 10 importances of xgboost

Method: gain-based importance (share of the total gain, mean over the fold models).

**Biased:** impurity and gain importances favour features with many distinct values (counters over one-hot columns) and features used early; they are computed on training data. They are pointers, not explanations: SHAP (06.11) is the proper tool.

| rank | feature | kind | importance | non-zero in folds |
|---|---|---|---|---|
| 1 | num__666_0 | numeric | 0.0725 | 5/5 |
| 2 | num__459_9 | numeric | 0.0511 | 5/5 |
| 3 | num__459_10 | numeric | 0.0424 | 5/5 |
| 4 | num__158_6 | numeric | 0.0310 | 5/5 |
| 5 | num__397_27 | numeric | 0.0144 | 5/5 |
| 6 | cat__Spec_7_Cat4 | one-hot | 0.0133 | 5/5 |
| 7 | cat__Spec_1_Cat1 | one-hot | 0.0110 | 5/5 |
| 8 | num__167_5 | numeric | 0.0108 | 5/5 |
| 9 | num__459_11 | numeric | 0.0105 | 5/5 |
| 10 | cat__Spec_4_Cat0 | one-hot | 0.0102 | 5/5 |
