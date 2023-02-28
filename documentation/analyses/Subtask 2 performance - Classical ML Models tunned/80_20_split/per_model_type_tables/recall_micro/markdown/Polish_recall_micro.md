| model_type                   |   title |   title and first paragraph |   title and 5 sentences |   title and 10 sentences |   title and first sentence each paragraph | raw text   |
|:-----------------------------|--------:|----------------------------:|------------------------:|-------------------------:|------------------------------------------:|:-----------|
| KNN                          |   0.451 |                       0.427 |                   0.388 |                    0.335 |                                     0.097 | 0.403      |
| LinearSVM                    |   0.558 |                       0.471 |                   0.539 |                    0.558 |                                     0.568 | 0.597      |
| LogisticRegression           |   0.524 |                       0.485 |                   0.563 |                    0.534 |                                     0.544 | 0.655      |
| LogisticRegressionElasticNet |   0.519 |                       0.495 |                   0.553 |                    0.51  |                                     0.573 | 0.641      |
| LogisticRegressionLasso      |   0.35  |                       0.427 |                   0.466 |                    0.519 |                                     0.529 | **0.733**  |
| LogisticRegressionRidge      |   0.578 |                       0.549 |                   0.549 |                    0.544 |                                     0.621 | 0.612      |
| NaiveBayes                   |   0.49  |                       0.481 |                   0.524 |                    0.549 |                                     0.626 | 0.646      |
| RandomForest                 |   0.398 |                       0.539 |                   0.553 |                    0.524 |                                     0.505 | 0.563      |
| RidgeClassifier              |   0.534 |                       0.505 |                   0.51  |                    0.49  |                                     0.578 | 0.558      |
| SVM                          |   0.549 |                       0.539 |                   0.524 |                    0.524 |                                     0.597 | 0.558      |
| XGBoost                      |   0.728 |                       0.67  |                   0.646 |                    0.626 |                                     0.617 | 0.709      |