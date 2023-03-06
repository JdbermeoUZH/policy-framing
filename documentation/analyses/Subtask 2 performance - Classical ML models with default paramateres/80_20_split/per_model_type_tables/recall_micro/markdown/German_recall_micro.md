| model_type                      |   title |   title and first paragraph |   title and 5 sentences | title and 10 sentences   |   title and first sentence each paragraph |   raw text |
|:--------------------------------|--------:|----------------------------:|------------------------:|:-------------------------|------------------------------------------:|-----------:|
| Binary Relevance kNN            |   0.36  |                       0.448 |                   0.36  | 0.285                    |                                     0.326 |      0.174 |
| ComplementNaiveBayes            |   0.448 |                       0.453 |                   0.413 | 0.436                    |                                     0.483 |      0.552 |
| Dummy Classifier                |   0.517 |                       0.5   |                   0.494 | 0.459                    |                                     0.529 |      0.523 |
| KNN                             |   0.686 |                       0.756 |                   0.767 | **0.860**                |                                     0.767 |      0.756 |
| LinearSVM                       |   0.302 |                       0.337 |                   0.355 | 0.360                    |                                     0.43  |      0.442 |
| LogisticRegression              |   0.308 |                       0.378 |                   0.378 | 0.372                    |                                     0.424 |      0.448 |
| LogisticRegressionElasticNet    |   0.25  |                       0.36  |                   0.39  | 0.366                    |                                     0.424 |      0.494 |
| LogisticRegressionLasso         |   0.145 |                       0.366 |                   0.384 | 0.343                    |                                     0.436 |      0.506 |
| LogisticRegressionRidge         |   0.302 |                       0.477 |                   0.459 | 0.448                    |                                     0.453 |      0.436 |
| Multilabel k Nearest Neighbours |   0.459 |                       0.436 |                   0.459 | 0.390                    |                                     0.262 |      0.256 |
| NaiveBayes                      |   0.314 |                       0.605 |                   0.581 | 0.599                    |                                     0.663 |      0.698 |
| RandomForest                    |   0.291 |                       0.576 |                   0.378 | 0.372                    |                                     0.378 |      0.395 |
| RidgeClassifier                 |   0.343 |                       0.494 |                   0.448 | 0.419                    |                                     0.448 |      0.442 |
| SVM                             |   0.483 |                       0.25  |                   0.314 | 0.436                    |                                     0.372 |      0.547 |
| XGBoost                         |   0.279 |                       0.669 |                   0.343 | 0.343                    |                                     0.401 |      0.453 |