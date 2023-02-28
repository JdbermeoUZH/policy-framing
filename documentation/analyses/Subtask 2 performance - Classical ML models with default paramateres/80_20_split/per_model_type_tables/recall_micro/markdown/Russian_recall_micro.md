| model_type                      |   title |   title and first paragraph |   title and 5 sentences | title and 10 sentences   | title and first sentence each paragraph   | raw text   |
|:--------------------------------|--------:|----------------------------:|------------------------:|:-------------------------|:------------------------------------------|:-----------|
| Binary Relevance kNN            |   0.128 |                       0.233 |                   0.221 | 0.233                    | 0.221                                     | 0.221      |
| ComplementNaiveBayes            |   0.453 |                       0.349 |                   0.302 | 0.360                    | 0.302                                     | 0.326      |
| Dummy Classifier                |   0.523 |                       0.477 |                   0.477 | 0.512                    | 0.535                                     | 0.453      |
| KNN                             |   0.488 |                       0.907 |                   0.965 | **1.000**                | **1.000**                                 | **1.000**  |
| LinearSVM                       |   0.302 |                       0.174 |                   0.174 | 0.186                    | 0.244                                     | 0.279      |
| LogisticRegression              |   0.279 |                       0.244 |                   0.209 | 0.198                    | 0.256                                     | 0.244      |
| LogisticRegressionElasticNet    |   0.244 |                       0.186 |                   0.174 | 0.186                    | 0.233                                     | 0.267      |
| LogisticRegressionLasso         |   0.233 |                       0.221 |                   0.186 | 0.326                    | 0.326                                     | 0.395      |
| LogisticRegressionRidge         |   0.349 |                       0.372 |                   0.267 | 0.279                    | 0.256                                     | 0.267      |
| Multilabel k Nearest Neighbours |   0.279 |                       0.326 |                   0.291 | 0.291                    | 0.163                                     | 0.174      |
| NaiveBayes                      |   0.453 |                       0.558 |                   0.558 | 0.570                    | 0.605                                     | 0.663      |
| RandomForest                    |   0.267 |                       0.198 |                   0.151 | 0.209                    | 0.267                                     | 0.256      |
| RidgeClassifier                 |   0.314 |                       0.384 |                   0.267 | 0.279                    | 0.267                                     | 0.267      |
| SVM                             |   0.163 |                       0.012 |                   0.023 | 0.035                    | 0.070                                     | 0.070      |
| XGBoost                         |   0.233 |                       0.233 |                   0.244 | 0.314                    | 0.337                                     | 0.442      |