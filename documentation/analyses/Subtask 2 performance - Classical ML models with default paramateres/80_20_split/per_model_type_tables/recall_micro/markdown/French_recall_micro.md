| model_type                      |   title |   title and first paragraph |   title and 5 sentences | title and 10 sentences   |   title and first sentence each paragraph | raw text   |
|:--------------------------------|--------:|----------------------------:|------------------------:|:-------------------------|------------------------------------------:|:-----------|
| Binary Relevance kNN            |   0.214 |                       0.175 |                   0.087 | 0.159                    |                                     0.19  | 0.063      |
| ComplementNaiveBayes            |   0.349 |                       0.429 |                   0.365 | 0.349                    |                                     0.365 | 0.381      |
| Dummy Classifier                |   0.563 |                       0.532 |                   0.492 | 0.468                    |                                     0.532 | 0.484      |
| KNN                             |   0.865 |                       0.968 |                   0.992 | **1.000**                |                                     0.659 | **1.000**  |
| LinearSVM                       |   0.254 |                       0.286 |                   0.302 | 0.206                    |                                     0.286 | 0.270      |
| LogisticRegression              |   0.341 |                       0.317 |                   0.286 | 0.206                    |                                     0.294 | 0.286      |
| LogisticRegressionElasticNet    |   0.214 |                       0.333 |                   0.302 | 0.230                    |                                     0.317 | 0.302      |
| LogisticRegressionLasso         |   0.246 |                       0.397 |                   0.325 | 0.270                    |                                     0.389 | 0.421      |
| LogisticRegressionRidge         |   0.357 |                       0.373 |                   0.31  | 0.254                    |                                     0.278 | 0.278      |
| Multilabel k Nearest Neighbours |   0.27  |                       0.333 |                   0.389 | 0.246                    |                                     0.095 | 0.389      |
| NaiveBayes                      |   0.429 |                       0.532 |                   0.524 | 0.563                    |                                     0.571 | 0.690      |
| RandomForest                    |   0.246 |                       0.278 |                   0.238 | 0.246                    |                                     0.278 | 0.325      |
| RidgeClassifier                 |   0.381 |                       0.381 |                   0.317 | 0.246                    |                                     0.278 | 0.294      |
| SVM                             |   0.317 |                       0.063 |                   0.079 | 0.032                    |                                     0.063 | 0.008      |
| XGBoost                         |   0.23  |                       0.381 |                   0.294 | 0.278                    |                                     0.365 | 0.405      |