| model_type                      | model_subtype                 |   title |   title and first paragraph |   title and 5 sentences |   title and 10 sentences |   title and first sentence each paragraph | raw text   |
|:--------------------------------|:------------------------------|--------:|----------------------------:|------------------------:|-------------------------:|------------------------------------------:|:-----------|
| Binary Relevance kNN            | Natively Multilabel           |   0.255 |                       0.328 |                   0.262 |                    0.209 |                                     0.249 | 0.126      |
| ComplementNaiveBayes            | RakelD Partitioning of labels |   0.433 |                       0.467 |                   0.44  |                    0.476 |                                     0.451 | 0.521      |
| Dummy Classifier                | No Upsampling                 |   0.434 |                       0.453 |                   0.449 |                    0.423 |                                     0.474 | 0.443      |
| KNN                             | BorderlineSMOTE               |   0.422 |                       0.508 |                   0.476 |                    0.505 |                                     0.379 | 0.362      |
| KNN                             | No Upsampling                 |   0.315 |                       0.364 |                   0.309 |                    0.269 |                                     0.246 | 0.039      |
| KNN                             | Random Oversampling           |   0.39  |                       0.459 |                   0.407 |                    0.352 |                                     0.326 | 0.070      |
| KNN                             | SMOTE                         |   0.44  |                       0.497 |                   0.447 |                    0.518 |                                     0.443 | 0.407      |
| KNN                             | SVMSMOTE                      |   0     |                       0.513 |                   0.509 |                    0     |                                     0     | 0          |
| LinearSVM                       | BorderlineSMOTE               |   0.338 |                       0.365 |                   0.395 |                    0.382 |                                     0.426 | 0.459      |
| LinearSVM                       | No Upsampling                 |   0.338 |                       0.365 |                   0.395 |                    0.382 |                                     0.426 | 0.459      |
| LinearSVM                       | RakelD Partitioning of labels |   0.333 |                       0.378 |                   0.393 |                    0.395 |                                     0.444 | 0.436      |
| LinearSVM                       | Random Oversampling           |   0.338 |                       0.365 |                   0.395 |                    0.382 |                                     0.426 | 0.459      |
| LinearSVM                       | SMOTE                         |   0.338 |                       0.365 |                   0.395 |                    0.382 |                                     0.426 | 0.459      |
| LogisticRegression              | BorderlineSMOTE               |   0.336 |                       0.364 |                   0.401 |                    0.394 |                                     0.412 | 0.455      |
| LogisticRegression              | No Upsampling                 |   0.34  |                       0.367 |                   0.382 |                    0.337 |                                     0.439 | 0.456      |
| LogisticRegression              | Random Oversampling           |   0.338 |                       0.364 |                   0.392 |                    0.413 |                                     0.407 | 0.466      |
| LogisticRegression              | SMOTE                         |   0.336 |                       0.365 |                   0.392 |                    0.396 |                                     0.416 | 0.461      |
| LogisticRegression              | SVMSMOTE                      |   0.334 |                       0.401 |                   0.409 |                    0.367 |                                     0.441 | 0.460      |
| LogisticRegressionElasticNet    | BorderlineSMOTE               |   0.284 |                       0.392 |                   0.403 |                    0.387 |                                     0.431 | 0.490      |
| LogisticRegressionElasticNet    | No Upsampling                 |   0.284 |                       0.362 |                   0.395 |                    0.396 |                                     0.42  | 0.467      |
| LogisticRegressionElasticNet    | Random Oversampling           |   0.284 |                       0.382 |                   0.409 |                    0.397 |                                     0.429 | 0.491      |
| LogisticRegressionElasticNet    | SMOTE                         |   0.284 |                       0.388 |                   0.406 |                    0.393 |                                     0.426 | 0.488      |
| LogisticRegressionElasticNet    | SVMSMOTE                      |   0.284 |                       0.387 |                   0.419 |                    0.39  |                                     0.429 | 0.472      |
| LogisticRegressionLasso         | BorderlineSMOTE               |   0.202 |                       0.388 |                   0.415 |                    0.381 |                                     0.44  | 0.518      |
| LogisticRegressionLasso         | No Upsampling                 |   0.202 |                       0.388 |                   0.397 |                    0.376 |                                     0.429 | 0.485      |
| LogisticRegressionLasso         | Random Oversampling           |   0.209 |                       0.402 |                   0.408 |                    0.389 |                                     0.466 | 0.516      |
| LogisticRegressionLasso         | SMOTE                         |   0.202 |                       0.394 |                   0.408 |                    0.38  |                                     0.448 | 0.521      |
| LogisticRegressionLasso         | SVMSMOTE                      |   0.217 |                       0.391 |                   0.429 |                    0.374 |                                     0.467 | 0.509      |
| LogisticRegressionRidge         | BorderlineSMOTE               |   0.334 |                       0.479 |                   0.47  |                    0.43  |                                     0.423 | 0.446      |
| LogisticRegressionRidge         | No Upsampling                 |   0.337 |                       0.463 |                   0.451 |                    0.412 |                                     0.417 | 0.436      |
| LogisticRegressionRidge         | RakelD Partitioning of labels |   0.355 |                       0.475 |                   0.397 |                    0.47  |                                     0.397 | 0.417      |
| LogisticRegressionRidge         | Random Oversampling           |   0.341 |                       0.481 |                   0.454 |                    0.437 |                                     0.461 | 0.446      |
| LogisticRegressionRidge         | SMOTE                         |   0.335 |                       0.475 |                   0.46  |                    0.421 |                                     0.431 | 0.446      |
| LogisticRegressionRidge         | SVMSMOTE                      |   0.35  |                       0.45  |                   0.448 |                    0.424 |                                     0.447 | 0.449      |
| Multilabel k Nearest Neighbours | Natively Multilabel           |   0.433 |                       0.437 |                   0.45  |                    0.396 |                                     0.29  | 0.272      |
| NaiveBayes                      | BorderlineSMOTE               |   0.365 |                       0.524 |                   0.556 |                    0.559 |                                     0.544 | 0.579      |
| NaiveBayes                      | No Upsampling                 |   0.333 |                       0.455 |                   0.471 |                    0.476 |                                     0.476 | 0.467      |
| NaiveBayes                      | Random Oversampling           |   0.376 |                       0.555 |                   0.55  |                    0.572 |                                     0.548 | 0.583      |
| NaiveBayes                      | SMOTE                         |   0.358 |                       0.528 |                   0.544 |                    0.558 |                                     0.551 | **0.585**  |
| RandomForest                    | BorderlineSMOTE               |   0.304 |                       0.439 |                   0.362 |                    0.393 |                                     0.395 | 0.415      |
| RandomForest                    | No Upsampling                 |   0.315 |                       0.484 |                   0.378 |                    0.39  |                                     0.393 | 0.423      |
| RandomForest                    | Random Oversampling           |   0.324 |                       0.55  |                   0.409 |                    0.398 |                                     0.418 | 0.427      |
| RandomForest                    | SMOTE                         |   0.3   |                       0.461 |                   0.352 |                    0.393 |                                     0.379 | 0.436      |
| RandomForest                    | SVMSMOTE                      |   0.302 |                       0.469 |                   0.386 |                    0.405 |                                     0.415 | 0.433      |
| RidgeClassifier                 | BorderlineSMOTE               |   0.339 |                       0.486 |                   0.46  |                    0.436 |                                     0.442 | 0.453      |
| RidgeClassifier                 | No Upsampling                 |   0.339 |                       0.486 |                   0.46  |                    0.436 |                                     0.442 | 0.453      |
| RidgeClassifier                 | Random Oversampling           |   0.339 |                       0.486 |                   0.46  |                    0.436 |                                     0.442 | 0.453      |
| RidgeClassifier                 | SMOTE                         |   0.339 |                       0.486 |                   0.46  |                    0.436 |                                     0.442 | 0.453      |
| RidgeClassifier                 | SVMSMOTE                      |   0.347 |                       0.504 |                   0.443 |                    0.431 |                                     0.451 | 0.457      |
| SVM                             | BorderlineSMOTE               |   0.338 |                       0.199 |                   0.184 |                    0.199 |                                     0.212 | 0.271      |
| SVM                             | No Upsampling                 |   0.203 |                       0.165 |                   0.155 |                    0.278 |                                     0.32  | 0.406      |
| SVM                             | RakelD Partitioning of labels |   0.148 |                       0.114 |                   0.056 |                    0.066 |                                     0.133 | 0.249      |
| SVM                             | Random Oversampling           |   0.218 |                       0.159 |                   0.274 |                    0.359 |                                     0.308 | 0.424      |
| SVM                             | SMOTE                         |   0.345 |                       0.196 |                   0.189 |                    0.199 |                                     0.21  | 0.263      |
| SVM                             | SVMSMOTE                      |   0.417 |                       0.199 |                   0.191 |                    0.188 |                                     0.223 | 0.225      |
| XGBoost                         | BorderlineSMOTE               |   0.246 |                       0.501 |                   0.359 |                    0.354 |                                     0.433 | 0.477      |
| XGBoost                         | No Upsampling                 |   0.256 |                       0.531 |                   0.392 |                    0.372 |                                     0.442 | 0.464      |
| XGBoost                         | Random Oversampling           |   0.29  |                       0.543 |                   0.402 |                    0.395 |                                     0.43  | 0.501      |
| XGBoost                         | SMOTE                         |   0.258 |                       0.501 |                   0.355 |                    0.343 |                                     0.427 | 0.466      |
| XGBoost                         | SVMSMOTE                      |   0.252 |                       0.511 |                   0.385 |                    0.376 |                                     0.415 | 0.444      |