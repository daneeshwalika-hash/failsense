# Top 10 importances of tree_depth8

Method: impurity-based importance (mean over the fold models).

**Biased:** impurity and gain importances favour features with many distinct values (counters over one-hot columns) and features used early; they are computed on training data. They are pointers, not explanations: SHAP (06.11) is the proper tool.

| rank | feature | kind | importance | non-zero in folds |
|---|---|---|---|---|
| 1 | num__459_9 | numeric | 0.3166 | 5/5 |
| 2 | num__666_0 | numeric | 0.1674 | 5/5 |
| 3 | num__459_10 | numeric | 0.0981 | 2/5 |
| 4 | num__time_step | numeric | 0.0494 | 5/5 |
| 5 | num__459_15 | numeric | 0.0214 | 4/5 |
| 6 | num__837_0 | numeric | 0.0169 | 5/5 |
| 7 | num__397_3 | numeric | 0.0129 | 4/5 |
| 8 | num__397_12 | numeric | 0.0110 | 5/5 |
| 9 | num__459_16 | numeric | 0.0108 | 5/5 |
| 10 | num__397_15 | numeric | 0.0097 | 4/5 |
