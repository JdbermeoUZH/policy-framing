| model_type                      | model_subtype                 | model_name                                   |   title |   title and first paragraph | title and 5 sentences   | title and 10 sentences   |   title and first sentence each paragraph | raw text   |
|:--------------------------------|:------------------------------|:---------------------------------------------|--------:|----------------------------:|:------------------------|:-------------------------|------------------------------------------:|:-----------|
| Binary Relevance kNN            | Natively Multilabel           | BRkNNaClassifier                             |   0.043 |                       0.257 | 0.017                   | 0.004                    |                                     0.022 | 0.000      |
| Binary Relevance kNN            | Natively Multilabel           | BRkNNbClassifier                             |   0.287 |                       0.378 | 0.270                   | 0.287                    |                                     0.213 | 0.174      |
| ComplementNaiveBayes            | RakelD Partitioning of labels | RakelD_ComplementNB                          |   0.474 |                       0.448 | 0.361                   | 0.439                    |                                     0.404 | 0.491      |
| Dummy Classifier                | No Upsampling                 | DummyMostFrequent                            |   0     |                       0     | 0.000                   | 0.000                    |                                     0     | 0.000      |
| Dummy Classifier                | No Upsampling                 | DummyProbSampling                            |   0.252 |                       0.352 | 0.313                   | 0.374                    |                                     0.304 | 0.352      |
| Dummy Classifier                | No Upsampling                 | DummyUniformSampling                         |   0.487 |                       0.5   | 0.461                   | 0.491                    |                                     0.47  | 0.487      |
| KNN                             | BorderlineSMOTE               | kNN_BorderlineSMOTE                          |   0.922 |                       0.852 | **1.000**               | 0.952                    |                                     0.839 | **1.000**  |
| KNN                             | No Upsampling                 | kNN                                          |   0.326 |                       0.174 | 0.052                   | 0.178                    |                                     0.174 | 0.026      |
| KNN                             | Random Oversampling           | kNN_ROS                                      |   0.513 |                       0.443 | 0.461                   | 0.383                    |                                     0.5   | 0.204      |
| KNN                             | SMOTE                         | kNN_SMOTE                                    |   0.917 |                       0.904 | **1.000**               | **1.000**                |                                     0.835 | **1.000**  |
| KNN                             | SVMSMOTE                      | kNN_SVMSMOTE                                 |   0.891 |                       0.761 | **1.000**               | 0                        |                                     0.83  | 0.974      |
| LinearSVM                       | BorderlineSMOTE               | LinearSVM_BorderlineSMOTE                    |   0.365 |                       0.361 | 0.330                   | 0.365                    |                                     0.374 | 0.417      |
| LinearSVM                       | BorderlineSMOTE               | LinearSVM_SVMSMOTE                           |   0.365 |                       0.361 | 0.330                   | 0                        |                                     0.374 | 0.417      |
| LinearSVM                       | No Upsampling                 | LinearSVM                                    |   0.365 |                       0.348 | 0.330                   | 0.365                    |                                     0.374 | 0.417      |
| LinearSVM                       | RakelD Partitioning of labels | RakelD_LineaSVM                              |   0.383 |                       0.352 | 0.330                   | 0.365                    |                                     0.357 | 0.383      |
| LinearSVM                       | Random Oversampling           | LinearSVM_ROS                                |   0.365 |                       0.348 | 0.330                   | 0.365                    |                                     0.374 | 0.417      |
| LinearSVM                       | SMOTE                         | LinearSVM_SMOTE                              |   0.374 |                       0.374 | 0.330                   | 0.365                    |                                     0.374 | 0.417      |
| LogisticRegression              | BorderlineSMOTE               | LogisticRegression_BorderlineSMOTE           |   0.365 |                       0.37  | 0.335                   | 0.361                    |                                     0.378 | 0.422      |
| LogisticRegression              | No Upsampling                 | LogisticRegression                           |   0.361 |                       0.343 | 0.296                   | 0.357                    |                                     0.378 | 0.387      |
| LogisticRegression              | Random Oversampling           | LogisticRegression_ROS                       |   0.365 |                       0.357 | 0.339                   | 0.365                    |                                     0.374 | 0.409      |
| LogisticRegression              | SMOTE                         | LogisticRegression_SMOTE                     |   0.357 |                       0.335 | 0.343                   | 0.374                    |                                     0.37  | 0.404      |
| LogisticRegression              | SVMSMOTE                      | LogisticRegression_SVMSMOTE                  |   0.43  |                       0.361 | 0.348                   | 0.357                    |                                     0.391 | 0.439      |
| LogisticRegressionElasticNet    | BorderlineSMOTE               | LogisticRegressionElasticNet_BorderlineSMOTE |   0.291 |                       0.33  | 0.335                   | 0.343                    |                                     0.422 | 0.387      |
| LogisticRegressionElasticNet    | No Upsampling                 | LogisticRegressionElasticNet                 |   0.287 |                       0.309 | 0.313                   | 0.335                    |                                     0.387 | 0.374      |
| LogisticRegressionElasticNet    | Random Oversampling           | LogisticRegressionElasticNet_ROS             |   0.296 |                       0.33  | 0.339                   | 0.343                    |                                     0.435 | 0.400      |
| LogisticRegressionElasticNet    | SMOTE                         | LogisticRegressionElasticNet_SMOTE           |   0.3   |                       0.33  | 0.339                   | 0.343                    |                                     0.435 | 0.387      |
| LogisticRegressionElasticNet    | SVMSMOTE                      | LogisticRegressionElasticNet_SVMSMOTE        |   0.309 |                       0.357 | 0.339                   | 0.361                    |                                     0.443 | 0.443      |
| LogisticRegressionLasso         | BorderlineSMOTE               | LogisticRegressionLasso_BorderlineSMOTE      |   0.265 |                       0.352 | 0.283                   | 0.361                    |                                     0.404 | 0.430      |
| LogisticRegressionLasso         | No Upsampling                 | LogisticRegressionLasso                      |   0.261 |                       0.343 | 0.278                   | 0.348                    |                                     0.396 | 0.417      |
| LogisticRegressionLasso         | Random Oversampling           | LogisticRegressionLasso_ROS                  |   0.278 |                       0.343 | 0.287                   | 0.374                    |                                     0.43  | 0.457      |
| LogisticRegressionLasso         | SMOTE                         | LogisticRegressionLasso_SMOTE                |   0.261 |                       0.357 | 0.278                   | 0.378                    |                                     0.413 | 0.439      |
| LogisticRegressionLasso         | SVMSMOTE                      | LogisticRegressionLasso_SVMSMOTE             |   0.283 |                       0.361 | 0.348                   | 0.400                    |                                     0.461 | 0.470      |
| LogisticRegressionRidge         | BorderlineSMOTE               | LogisticRegressionRidge_BorderlineSMOTE      |   0.409 |                       0.413 | 0.370                   | 0.400                    |                                     0.361 | 0.409      |
| LogisticRegressionRidge         | No Upsampling                 | LogisticRegressionRidge                      |   0.365 |                       0.387 | 0.357                   | 0.391                    |                                     0.339 | 0.396      |
| LogisticRegressionRidge         | RakelD Partitioning of labels | RakelD_LogisticRegression                    |   0.396 |                       0.343 | 0.357                   | 0.348                    |                                     0.348 | 0.343      |
| LogisticRegressionRidge         | Random Oversampling           | LogisticRegressionRidge_ROS                  |   0.396 |                       0.413 | 0.378                   | 0.409                    |                                     0.357 | 0.404      |
| LogisticRegressionRidge         | SMOTE                         | LogisticRegressionRidge_SMOTE                |   0.396 |                       0.413 | 0.378                   | 0.404                    |                                     0.365 | 0.400      |
| LogisticRegressionRidge         | SVMSMOTE                      | LogisticRegressionRidge_SVMSMOTE             |   0.426 |                       0.378 | 0.357                   | 0.409                    |                                     0.383 | 0.417      |
| Multilabel k Nearest Neighbours | Natively Multilabel           | MLkNN                                        |   0.417 |                       0.439 | 0.396                   | 0.491                    |                                     0.322 | 0.348      |
| NaiveBayes                      | BorderlineSMOTE               | ComplementNaiveBayes_BorderlineSMOTE         |   0.552 |                       0.543 | 0.591                   | 0.596                    |                                     0.648 | 0.717      |
| NaiveBayes                      | BorderlineSMOTE               | NaiveBayes_BorderlineSMOTE                   |   0.561 |                       0.539 | 0.587                   | 0.613                    |                                     0.657 | 0.691      |
| NaiveBayes                      | No Upsampling                 | ComplementNaiveBayes                         |   0.426 |                       0.361 | 0.352                   | 0.357                    |                                     0.391 | 0.348      |
| NaiveBayes                      | No Upsampling                 | NaiveBayes                                   |   0.113 |                       0.143 | 0.161                   | 0.191                    |                                     0.204 | 0.235      |
| NaiveBayes                      | Random Oversampling           | ComplementNaiveBayes_ROS                     |   0.587 |                       0.583 | 0.622                   | 0.643                    |                                     0.691 | 0.717      |
| NaiveBayes                      | Random Oversampling           | NaiveBayes_ROS                               |   0.557 |                       0.622 | 0.609                   | 0.635                    |                                     0.665 | 0.748      |
| NaiveBayes                      | SMOTE                         | ComplementNaiveBayes_SMOTE                   |   0.517 |                       0.543 | 0.591                   | 0.626                    |                                     0.652 | 0.691      |
| NaiveBayes                      | SMOTE                         | NaiveBayes_SMOTE                             |   0.53  |                       0.53  | 0.587                   | 0.613                    |                                     0.648 | 0.700      |
| RandomForest                    | BorderlineSMOTE               | RandomForest_BorderlineSMOTE                 |   0.252 |                       0.37  | 0.361                   | 0.396                    |                                     0.317 | 0.387      |
| RandomForest                    | No Upsampling                 | RandomForest                                 |   0.278 |                       0.343 | 0.361                   | 0.357                    |                                     0.283 | 0.357      |
| RandomForest                    | Random Oversampling           | RandomForest_ROS                             |   0.304 |                       0.439 | 0.391                   | 0.435                    |                                     0.374 | 0.435      |
| RandomForest                    | SMOTE                         | RandomForest_SMOTE                           |   0.265 |                       0.37  | 0.365                   | 0.383                    |                                     0.309 | 0.396      |
| RandomForest                    | SVMSMOTE                      | RandomForest_SVMSMOTE                        |   0.27  |                       0.348 | 0.317                   | 0.400                    |                                     0.283 | 0.404      |
| RidgeClassifier                 | BorderlineSMOTE               | RidgeClassifier_BorderlineSMOTE              |   0.387 |                       0.404 | 0.378                   | 0.404                    |                                     0.365 | 0.404      |
| RidgeClassifier                 | No Upsampling                 | RidgeClassifier                              |   0.383 |                       0.409 | 0.378                   | 0.404                    |                                     0.365 | 0.404      |
| RidgeClassifier                 | Random Oversampling           | RidgeClassifier_ROS                          |   0.383 |                       0.409 | 0.378                   | 0.404                    |                                     0.365 | 0.404      |
| RidgeClassifier                 | SMOTE                         | RidgeClassifier_SMOTE                        |   0.383 |                       0.409 | 0.378                   | 0.404                    |                                     0.365 | 0.404      |
| RidgeClassifier                 | SVMSMOTE                      | RidgeClassifier_SVMSMOTE                     |   0.448 |                       0.396 | 0.370                   | 0.409                    |                                     0.387 | 0.452      |
| SVM                             | BorderlineSMOTE               | SVM_rbf_BorderlineSMOTE                      |   0.152 |                       0.009 | 0.013                   | 0.035                    |                                     0.043 | 0.100      |
| SVM                             | No Upsampling                 | SVM_rbf                                      |   0.109 |                       0.004 | 0.017                   | 0.104                    |                                     0.178 | 0.248      |
| SVM                             | RakelD Partitioning of labels | RakelD_SVM                                   |   0.152 |                       0     | 0.161                   | 0.026                    |                                     0.274 | 0.187      |
| SVM                             | Random Oversampling           | SVM_rbf_ROS                                  |   0.2   |                       0.017 | 0.022                   | 0.174                    |                                     0.209 | 0.335      |
| SVM                             | SMOTE                         | SVM_rbf_SMOTE                                |   0.157 |                       0.009 | 0.013                   | 0.039                    |                                     0.026 | 0.117      |
| SVM                             | SVMSMOTE                      | SVM_rbf_SVMSMOTE                             |   0.3   |                       0.009 | 0.013                   | 0.022                    |                                     0.013 | 0.087      |
| XGBoost                         | BorderlineSMOTE               | XGBoost_narrow_BorderlineSMOTE               |   0.204 |                       0.357 | 0.352                   | 0.370                    |                                     0.352 | 0.396      |
| XGBoost                         | No Upsampling                 | XGBoost_narrow                               |   0.226 |                       0.378 | 0.335                   | 0.378                    |                                     0.374 | 0.409      |
| XGBoost                         | Random Oversampling           | XGBoost_narrow_ROS                           |   0.27  |                       0.435 | 0.400                   | 0.413                    |                                     0.413 | 0.422      |
| XGBoost                         | SMOTE                         | XGBoost_narrow_SMOTE                         |   0.187 |                       0.383 | 0.339                   | 0.383                    |                                     0.361 | 0.443      |
| XGBoost                         | SVMSMOTE                      | XGBoost_narrow_SVMSMOTE                      |   0.222 |                       0.391 | 0.343                   | 0.378                    |                                     0.378 | 0.435      |