| model_type                   | model_subtype       | model_name                                   | title     |   title and first paragraph |   title and 5 sentences |   title and 10 sentences |   title and first sentence each paragraph | raw text   |
|:-----------------------------|:--------------------|:---------------------------------------------|:----------|----------------------------:|------------------------:|-------------------------:|------------------------------------------:|:-----------|
| KNN                          | BorderlineSMOTE     | kNN_BorderlineSMOTE                          | 0.359     |                       0.33  |                   0.398 |                    0.345 |                                     0.097 | 0.403      |
| KNN                          | No Upsampling       | kNN                                          | 0.306     |                       0.18  |                   0.388 |                    0.335 |                                     0.019 | 0.000      |
| KNN                          | Random Oversampling | kNN_ROS                                      | 0.451     |                       0.427 |                   0.32  |                    0.335 |                                     0.019 | 0.000      |
| KNN                          | SMOTE               | kNN_SMOTE                                    | 0.354     |                       0.427 |                   0.388 |                    0.175 |                                     0.097 | 0.403      |
| KNN                          | SVMSMOTE            | kNN_SVMSMOTE                                 | 0.306     |                       0.238 |                   0.32  |                    0.175 |                                     0.097 | 0.097      |
| LinearSVM                    | BorderlineSMOTE     | LinearSVMDual_BorderlineSMOTE                | 0.539     |                       0.466 |                   0.534 |                    0.495 |                                     0.553 | 0.597      |
| LinearSVM                    | BorderlineSMOTE     | LinearSVMDual_SVMSMOTE                       | 0.519     |                       0.466 |                   0.534 |                    0.558 |                                     0.568 | 0.602      |
| LinearSVM                    | No Upsampling       | LinearSVMDual                                | 0.558     |                       0.481 |                   0.539 |                    0.495 |                                     0.558 | 0.597      |
| LinearSVM                    | Random Oversampling | LinearSVMDual_ROS                            | 0.553     |                       0.471 |                   0.519 |                    0.515 |                                     0.573 | 0.583      |
| LinearSVM                    | SMOTE               | LinearSVMDual_SMOTE                          | 0.549     |                       0.456 |                   0.519 |                    0.519 |                                     0.549 | 0.568      |
| LogisticRegression           | BorderlineSMOTE     | LogisticRegression_BorderlineSMOTE           | 0.534     |                       0.485 |                   0.563 |                    0.534 |                                     0.544 | 0.655      |
| LogisticRegression           | No Upsampling       | LogisticRegression                           | 0.524     |                       0.451 |                   0.515 |                    0.481 |                                     0.456 | 0.578      |
| LogisticRegression           | Random Oversampling | LogisticRegression_ROS                       | 0.515     |                       0.451 |                   0.515 |                    0.485 |                                     0.481 | 0.558      |
| LogisticRegression           | SMOTE               | LogisticRegression_SMOTE                     | 0.524     |                       0.456 |                   0.519 |                    0.495 |                                     0.466 | 0.573      |
| LogisticRegression           | SVMSMOTE            | LogisticRegression_SVMSMOTE                  | 0.505     |                       0.456 |                   0.524 |                    0.466 |                                     0.515 | 0.573      |
| LogisticRegressionElasticNet | BorderlineSMOTE     | LogisticRegressionElasticNet_BorderlineSMOTE | 0.505     |                       0.49  |                   0.553 |                    0.51  |                                     0.573 | 0.641      |
| LogisticRegressionElasticNet | No Upsampling       | LogisticRegressionElasticNet                 | 0.461     |                       0.471 |                   0.524 |                    0.495 |                                     0.553 | 0.583      |
| LogisticRegressionElasticNet | Random Oversampling | LogisticRegressionElasticNet_ROS             | 0.500     |                       0.495 |                   0.515 |                    0.49  |                                     0.558 | 0.592      |
| LogisticRegressionElasticNet | SMOTE               | LogisticRegressionElasticNet_SMOTE           | 0.524     |                       0.461 |                   0.529 |                    0.471 |                                     0.553 | 0.583      |
| LogisticRegressionElasticNet | SVMSMOTE            | LogisticRegressionElasticNet_SVMSMOTE        | 0.519     |                       0.481 |                   0.524 |                    0.495 |                                     0.553 | 0.587      |
| LogisticRegressionLasso      | BorderlineSMOTE     | LogisticRegressionLasso_BorderlineSMOTE      | 0.350     |                       0.427 |                   0.374 |                    0.5   |                                     0.529 | 0.578      |
| LogisticRegressionLasso      | No Upsampling       | LogisticRegressionLasso                      | 0.345     |                       0.422 |                   0.5   |                    0.49  |                                     0.549 | **0.733**  |
| LogisticRegressionLasso      | Random Oversampling | LogisticRegressionLasso_ROS                  | 0.345     |                       0.345 |                   0.461 |                    0.49  |                                     0.505 | 0.592      |
| LogisticRegressionLasso      | SMOTE               | LogisticRegressionLasso_SMOTE                | 0.350     |                       0.311 |                   0.466 |                    0.5   |                                     0.5   | 0.583      |
| LogisticRegressionLasso      | SVMSMOTE            | LogisticRegressionLasso_SVMSMOTE             | 0.286     |                       0.272 |                   0.335 |                    0.519 |                                     0.544 | 0.578      |
| LogisticRegressionRidge      | BorderlineSMOTE     | LogisticRegressionRidgeDual_BorderlineSMOTE  | 0.558     |                       0.549 |                   0.549 |                    0.519 |                                     0.607 | 0.612      |
| LogisticRegressionRidge      | No Upsampling       | LogisticRegressionRidgeDual                  | 0.578     |                       0.558 |                   0.549 |                    0.515 |                                     0.621 | 0.607      |
| LogisticRegressionRidge      | Random Oversampling | LogisticRegressionRidgeDual_ROS              | 0.568     |                       0.495 |                   0.534 |                    0.534 |                                     0.583 | 0.602      |
| LogisticRegressionRidge      | SMOTE               | LogisticRegressionRidgeDual_SMOTE            | 0.544     |                       0.485 |                   0.544 |                    0.544 |                                     0.573 | 0.592      |
| LogisticRegressionRidge      | SVMSMOTE            | LogisticRegressionRidgeDual_SVMSMOTE         | 0.515     |                       0.476 |                   0.519 |                    0.495 |                                     0.549 | 0.602      |
| NaiveBayes                   | BorderlineSMOTE     | ComplementNaiveBayes_BorderlineSMOTE         | 0.476     |                       0.495 |                   0.51  |                    0.544 |                                     0.592 | 0.636      |
| NaiveBayes                   | BorderlineSMOTE     | NaiveBayes_BorderlineSMOTE                   | 0.481     |                       0.49  |                   0.49  |                    0.524 |                                     0.626 | 0.646      |
| NaiveBayes                   | No Upsampling       | ComplementNaiveBayes                         | 0.471     |                       0.476 |                   0.515 |                    0.544 |                                     0.534 | 0.641      |
| NaiveBayes                   | No Upsampling       | NaiveBayes                                   | 0.471     |                       0.476 |                   0.519 |                    0.544 |                                     0.573 | 0.641      |
| NaiveBayes                   | Random Oversampling | ComplementNaiveBayes_ROS                     | 0.471     |                       0.481 |                   0.529 |                    0.549 |                                     0.592 | 0.641      |
| NaiveBayes                   | Random Oversampling | NaiveBayes_ROS                               | 0.471     |                       0.481 |                   0.524 |                    0.544 |                                     0.587 | 0.641      |
| NaiveBayes                   | SMOTE               | ComplementNaiveBayes_SMOTE                   | 0.490     |                       0.461 |                   0.524 |                    0.495 |                                     0.587 | 0.631      |
| NaiveBayes                   | SMOTE               | NaiveBayes_SMOTE                             | 0.451     |                       0.471 |                   0.485 |                    0.505 |                                     0.583 | 0.626      |
| NaiveBayes                   | SVMSMOTE            | ComplementNaiveBayes_SVMSMOTE                | 0.471     |                       0.481 |                   0.515 |                    0.544 |                                     0.524 | 0.636      |
| NaiveBayes                   | SVMSMOTE            | NaiveBayes_SVMSMOTE                          | 0.471     |                       0.476 |                   0.515 |                    0.544 |                                     0.524 | 0.641      |
| RandomForest                 | BorderlineSMOTE     | RandomForest_BorderlineSMOTE                 | 0.316     |                       0.461 |                   0.403 |                    0.398 |                                     0.461 | 0.529      |
| RandomForest                 | No Upsampling       | RandomForest                                 | 0.286     |                       0.558 |                   0.553 |                    0.524 |                                     0.505 | 0.563      |
| RandomForest                 | Random Oversampling | RandomForest_ROS                             | 0.335     |                       0.539 |                   0.481 |                    0.456 |                                     0.461 | 0.437      |
| RandomForest                 | SMOTE               | RandomForest_SMOTE                           | 0.141     |                       0.427 |                   0.403 |                    0.417 |                                     0.417 | 0.519      |
| RandomForest                 | SVMSMOTE            | RandomForest_SVMSMOTE                        | 0.398     |                       0.456 |                   0.422 |                    0.417 |                                     0.403 | 0.485      |
| RidgeClassifier              | BorderlineSMOTE     | RidgeClassifier_BorderlineSMOTE              | 0.539     |                       0.471 |                   0.5   |                    0.481 |                                     0.563 | 0.558      |
| RidgeClassifier              | No Upsampling       | RidgeClassifier                              | 0.529     |                       0.471 |                   0.5   |                    0.49  |                                     0.578 | 0.534      |
| RidgeClassifier              | Random Oversampling | RidgeClassifier_ROS                          | 0.534     |                       0.505 |                   0.5   |                    0.461 |                                     0.558 | 0.549      |
| RidgeClassifier              | SMOTE               | RidgeClassifier_SMOTE                        | 0.510     |                       0.466 |                   0.505 |                    0.466 |                                     0.568 | 0.539      |
| RidgeClassifier              | SVMSMOTE            | RidgeClassifier_SVMSMOTE                     | 0.364     |                       0.461 |                   0.51  |                    0.471 |                                     0.534 | 0.544      |
| SVM                          | BorderlineSMOTE     | SVM_rbf_BorderlineSMOTE                      | 0.515     |                       0.413 |                   0.476 |                    0.471 |                                     0.505 | 0.558      |
| SVM                          | BorderlineSMOTE     | SVM_sigmoid_BorderlineSMOTE                  | 0.481     |                       0.519 |                   0.505 |                    0.519 |                                     0.583 | 0.573      |
| SVM                          | No Upsampling       | SVM_rbf                                      | 0.476     |                       0.393 |                   0.519 |                    0.5   |                                     0.549 | 0.515      |
| SVM                          | No Upsampling       | SVM_sigmoid                                  | 0.515     |                       0.505 |                   0.505 |                    0.524 |                                     0.553 | 0.553      |
| SVM                          | Random Oversampling | SVM_rbf_ROS                                  | 0.461     |                       0.461 |                   0.529 |                    0.481 |                                     0.544 | 0.524      |
| SVM                          | Random Oversampling | SVM_sigmoid_ROS                              | 0.490     |                       0.442 |                   0.447 |                    0.485 |                                     0.578 | 0.510      |
| SVM                          | SMOTE               | SVM_rbf_SMOTE                                | 0.476     |                       0.481 |                   0.442 |                    0.485 |                                     0.451 | 0.524      |
| SVM                          | SMOTE               | SVM_sigmoid_SMOTE                            | 0.549     |                       0.447 |                   0.476 |                    0.447 |                                     0.437 | 0.515      |
| SVM                          | SVMSMOTE            | SVM_rbf_SVMSMOTE                             | 0.451     |                       0.388 |                   0.524 |                    0.505 |                                     0.597 | 0.549      |
| SVM                          | SVMSMOTE            | SVM_sigmoid_SVMSMOTE                         | 0.524     |                       0.539 |                   0.481 |                    0.451 |                                     0.602 | 0.563      |
| XGBoost                      | BorderlineSMOTE     | XGBoost_broad_BorderlineSMOTE                | 0.461     |                       0.5   |                   0.534 |                    0.549 |                                     0.553 | 0.587      |
| XGBoost                      | BorderlineSMOTE     | XGBoost_narrow_BorderlineSMOTE               | **0.733** |                       0.631 |                   0.607 |                    0.597 |                                     0.617 | 0.670      |
| XGBoost                      | No Upsampling       | XGBoost_broad                                | 0.481     |                       0.5   |                   0.442 |                    0.447 |                                     0.461 | 0.529      |
| XGBoost                      | No Upsampling       | XGBoost_narrow                               | 0.650     |                       0.67  |                   0.587 |                    0.602 |                                     0.583 | 0.709      |
| XGBoost                      | Random Oversampling | XGBoost_broad_ROS                            | 0.447     |                       0.437 |                   0.422 |                    0.485 |                                     0.553 | 0.539      |
| XGBoost                      | Random Oversampling | XGBoost_narrow_ROS                           | 0.723     |                       0.583 |                   0.563 |                    0.626 |                                     0.568 | 0.655      |
| XGBoost                      | SMOTE               | XGBoost_broad_SMOTE                          | 0.529     |                       0.485 |                   0.51  |                    0.476 |                                     0.617 | 0.529      |
| XGBoost                      | SMOTE               | XGBoost_narrow_SMOTE                         | 0.728     |                       0.597 |                   0.646 |                    0.583 |                                     0.597 | 0.646      |
| XGBoost                      | SVMSMOTE            | XGBoost_broad_SVMSMOTE                       | 0.597     |                       0.466 |                   0.583 |                    0.447 |                                     0.476 | 0.539      |
| XGBoost                      | SVMSMOTE            | XGBoost_narrow_SVMSMOTE                      | 0.728     |                       0.646 |                   0.563 |                    0.578 |                                     0.612 | 0.670      |