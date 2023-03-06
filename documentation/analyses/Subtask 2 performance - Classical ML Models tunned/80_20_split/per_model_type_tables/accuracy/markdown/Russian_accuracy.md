| model_type                   |   title |   title and first paragraph | title and 5 sentences   |   title and 10 sentences |   title and first sentence each paragraph | raw text   |
|:-----------------------------|--------:|----------------------------:|:------------------------|-------------------------:|------------------------------------------:|:-----------|
| KNN                          |   0     |                       0.026 | 0.079                   |                    0.105 |                                     0.053 | 0.079      |
| LinearSVM                    |   0     |                       0.053 | 0.053                   |                    0.105 |                                     0.053 | 0.105      |
| LogisticRegression           |   0     |                       0.053 | 0.053                   |                    0     |                                     0.053 | 0.105      |
| LogisticRegressionElasticNet |   0     |                       0.079 | 0.053                   |                    0.079 |                                     0.079 | **0.158**  |
| LogisticRegressionLasso      |   0.026 |                       0.026 | 0.053                   |                    0.053 |                                     0.053 | 0.132      |
| LogisticRegressionRidge      |   0     |                       0.079 | 0.026                   |                    0.079 |                                     0.079 | 0.105      |
| NaiveBayes                   |   0     |                       0.053 | **0.158**               |                    0.132 |                                     0.053 | 0.079      |
| RandomForest                 |   0     |                       0     | 0.000                   |                    0.026 |                                     0.053 | 0.105      |
| RidgeClassifier              |   0.026 |                       0.079 | 0.132                   |                    0.053 |                                     0.053 | 0.079      |
| SVM                          |   0     |                       0.053 | **0.158**               |                    0.053 |                                     0.105 | 0.053      |
| XGBoost                      |   0.026 |                       0     | 0.053                   |                    0.105 |                                     0.132 | **0.158**  |