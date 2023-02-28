| model_type                   |   title | title and first paragraph   |   title and 5 sentences |   title and 10 sentences |   title and first sentence each paragraph |   raw text |
|:-----------------------------|--------:|:----------------------------|------------------------:|-------------------------:|------------------------------------------:|-----------:|
| KNN                          |   0.309 | 0.371                       |                   0.426 |                    0.417 |                                     0.402 |      0.492 |
| LinearSVM                    |   0.462 | 0.518                       |                   0.528 |                    0.514 |                                     0.465 |      0.503 |
| LogisticRegression           |   0.459 | 0.535                       |                   0.492 |                    0.527 |                                     0.45  |      0.516 |
| LogisticRegressionElasticNet |   0.499 | **0.617**                   |                   0.499 |                    0.532 |                                     0.468 |      0.535 |
| LogisticRegressionLasso      |   0.454 | 0.599                       |                   0.512 |                    0.504 |                                     0.508 |      0.552 |
| LogisticRegressionRidge      |   0.458 | 0.550                       |                   0.493 |                    0.521 |                                     0.451 |      0.52  |
| NaiveBayes                   |   0.493 | 0.566                       |                   0.522 |                    0.511 |                                     0.5   |      0.536 |
| RandomForest                 |   0.523 | 0.562                       |                   0.534 |                    0.542 |                                     0.549 |      0.593 |
| RidgeClassifier              |   0.48  | 0.549                       |                   0.512 |                    0.5   |                                     0.575 |      0.552 |
| SVM                          |   0.49  | 0.528                       |                   0.576 |                    0.557 |                                     0.545 |      0.55  |
| XGBoost                      |   0.436 | 0.555                       |                   0.507 |                    0.547 |                                     0.55  |      0.549 |