| model_type                   |   title |   title and first paragraph |   title and 5 sentences |   title and 10 sentences |   title and first sentence each paragraph | raw text   |
|:-----------------------------|--------:|----------------------------:|------------------------:|-------------------------:|------------------------------------------:|:-----------|
| KNN                          |   0.017 |                       0.067 |                   0.05  |                    0.017 |                                     0.067 | 0.117      |
| LinearSVM                    |   0.017 |                       0.083 |                   0.033 |                    0.067 |                                     0.05  | 0.083      |
| LogisticRegression           |   0.067 |                       0.05  |                   0.05  |                    0.033 |                                     0.033 | 0.050      |
| LogisticRegressionElasticNet |   0.033 |                       0.083 |                   0.05  |                    0.067 |                                     0.083 | 0.100      |
| LogisticRegressionLasso      |   0.083 |                       0.033 |                   0.033 |                    0.05  |                                     0.067 | 0.050      |
| LogisticRegressionRidge      |   0.033 |                       0.083 |                   0.033 |                    0.017 |                                     0.067 | 0.100      |
| NaiveBayes                   |   0.033 |                       0.067 |                   0.05  |                    0.067 |                                     0.067 | 0.083      |
| RandomForest                 |   0     |                       0     |                   0.033 |                    0.067 |                                     0.017 | 0.067      |
| RidgeClassifier              |   0.05  |                       0.083 |                   0.067 |                    0.067 |                                     0.083 | **0.150**  |
| SVM                          |   0.033 |                       0.05  |                   0.067 |                    0.067 |                                     0.083 | 0.083      |
| XGBoost                      |   0     |                       0     |                   0.033 |                    0.05  |                                     0     | 0.050      |