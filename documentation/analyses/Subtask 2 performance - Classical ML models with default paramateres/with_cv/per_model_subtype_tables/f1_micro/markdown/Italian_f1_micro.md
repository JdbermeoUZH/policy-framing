| model_type                      | model_subtype                 | title           | title and first paragraph   | title and 5 sentences   | title and 10 sentences   | title and first sentence each paragraph   | raw text            |
|:--------------------------------|:------------------------------|:----------------|:----------------------------|:------------------------|:-------------------------|:------------------------------------------|:--------------------|
| Binary Relevance kNN            | Natively Multilabel           | 0.16 $\pm$ 0.04 | 0.21 $\pm$ 0.01             | 0.12 $\pm$ 0.01         | 0.15 $\pm$ 0.04          | 0.10 $\pm$ 0.00                           | 0.11 $\pm$ 0.01     |
| ComplementNB                    | RakelD Partitioning of labels | 0.41 $\pm$ 0.00 | 0.41 $\pm$ 0.01             | 0.40 $\pm$ 0.01         | 0.41 $\pm$ 0.01          | 0.43 $\pm$ 0.01                           | 0.44 $\pm$ 0.02     |
| Dummy Classifier                | No Upsampling                 | 0.40 $\pm$ 0.02 | 0.37 $\pm$ 0.03             | 0.36 $\pm$ 0.02         | 0.37 $\pm$ 0.04          | 0.38 $\pm$ 0.02                           | 0.38 $\pm$ 0.02     |
| KNN                             | BorderlineSMOTE               | 0.46 $\pm$ 0.03 | 0.42 $\pm$ 0.05             | 0.45 $\pm$ 0.04         | 0.48 $\pm$ 0.02          | 0.41 $\pm$ 0.02                           | 0.47 $\pm$ 0.02     |
| KNN                             | No Upsampling                 | 0.15 $\pm$ 0.06 | 0.22 $\pm$ 0.05             | 0.08 $\pm$ 0.02         | 0.10 $\pm$ 0.04          | 0.13 $\pm$ 0.04                           | 0.04 $\pm$ 0.01     |
| KNN                             | Random Oversampling           | 0.29 $\pm$ 0.04 | 0.35 $\pm$ 0.02             | 0.20 $\pm$ 0.03         | 0.22 $\pm$ 0.06          | 0.25 $\pm$ 0.03                           | 0.13 $\pm$ 0.02     |
| KNN                             | SMOTE                         | 0.46 $\pm$ 0.03 | 0.45 $\pm$ 0.04             | 0.47 $\pm$ 0.02         | 0.48 $\pm$ 0.02          | 0.43 $\pm$ 0.02                           | 0.45 $\pm$ 0.05     |
| KNN                             | SVMSMOTE                      | 0.46 $\pm$ 0.04 | 0                           | 0.44 $\pm$ 0.05         | 0.46 $\pm$ 0.04          | 0                                         | 0.43 $\pm$ 0.08     |
| LinearSVM                       | BorderlineSMOTE               | 0.40 $\pm$ 0.02 | 0.38 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.42 $\pm$ 0.01                           | 0.46 $\pm$ 0.01     |
| LinearSVM                       | No Upsampling                 | 0.40 $\pm$ 0.02 | 0.38 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.42 $\pm$ 0.01                           | 0.46 $\pm$ 0.01     |
| LinearSVM                       | RakelD Partitioning of labels | 0.38 $\pm$ 0.02 | 0.41 $\pm$ 0.01             | 0.38 $\pm$ 0.01         | 0.39 $\pm$ 0.02          | 0.44 $\pm$ 0.01                           | 0.43 $\pm$ 0.03     |
| LinearSVM                       | Random Oversampling           | 0.40 $\pm$ 0.02 | 0.38 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.42 $\pm$ 0.01                           | 0.46 $\pm$ 0.01     |
| LinearSVM                       | SMOTE                         | 0.39 $\pm$ 0.02 | 0.38 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.42 $\pm$ 0.01                           | 0.46 $\pm$ 0.01     |
| LogisticRegression              | BorderlineSMOTE               | 0.40 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.02         | 0.37 $\pm$ 0.01          | 0.43 $\pm$ 0.01                           | 0.46 $\pm$ 0.01     |
| LogisticRegression              | No Upsampling                 | 0.40 $\pm$ 0.02 | 0.40 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.43 $\pm$ 0.02                           | 0.46 $\pm$ 0.02     |
| LogisticRegression              | Random Oversampling           | 0.40 $\pm$ 0.01 | 0.39 $\pm$ 0.02             | 0.37 $\pm$ 0.02         | 0.38 $\pm$ 0.01          | 0.44 $\pm$ 0.02                           | 0.47 $\pm$ 0.01     |
| LogisticRegression              | SMOTE                         | 0.40 $\pm$ 0.02 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.02         | 0.37 $\pm$ 0.01          | 0.44 $\pm$ 0.02                           | 0.46 $\pm$ 0.01     |
| LogisticRegression              | SVMSMOTE                      | 0.40 $\pm$ 0.01 | 0.40 $\pm$ 0.02             | 0.38 $\pm$ 0.02         | 0.40 $\pm$ 0.03          | 0.43 $\pm$ 0.02                           | 0.43 $\pm$ 0.01     |
| LogisticRegressionElasticNet    | BorderlineSMOTE               | 0.39 $\pm$ 0.02 | 0.41 $\pm$ 0.00             | 0.39 $\pm$ 0.02         | 0.43 $\pm$ 0.01          | 0.46 $\pm$ 0.03                           | 0.49 $\pm$ 0.02     |
| LogisticRegressionElasticNet    | No Upsampling                 | 0.38 $\pm$ 0.02 | 0.39 $\pm$ 0.01             | 0.37 $\pm$ 0.01         | 0.41 $\pm$ 0.01          | 0.44 $\pm$ 0.02                           | 0.48 $\pm$ 0.02     |
| LogisticRegressionElasticNet    | Random Oversampling           | 0.39 $\pm$ 0.02 | 0.42 $\pm$ 0.00             | 0.41 $\pm$ 0.01         | 0.44 $\pm$ 0.01          | 0.47 $\pm$ 0.02                           | 0.51 $\pm$ 0.02     |
| LogisticRegressionElasticNet    | SMOTE                         | 0.39 $\pm$ 0.02 | 0.41 $\pm$ 0.01             | 0.39 $\pm$ 0.01         | 0.43 $\pm$ 0.01          | 0.46 $\pm$ 0.03                           | 0.49 $\pm$ 0.02     |
| LogisticRegressionElasticNet    | SVMSMOTE                      | 0.40 $\pm$ 0.03 | 0.42 $\pm$ 0.01             | 0.40 $\pm$ 0.01         | 0.42 $\pm$ 0.00          | 0.46 $\pm$ 0.02                           | 0.49 $\pm$ 0.02     |
| LogisticRegressionLasso         | BorderlineSMOTE               | 0.32 $\pm$ 0.02 | 0.36 $\pm$ 0.01             | 0.35 $\pm$ 0.01         | 0.41 $\pm$ 0.01          | 0.37 $\pm$ 0.01                           | 0.44 $\pm$ 0.00     |
| LogisticRegressionLasso         | No Upsampling                 | 0.30 $\pm$ 0.02 | 0.35 $\pm$ 0.00             | 0.34 $\pm$ 0.01         | 0.39 $\pm$ 0.01          | 0.37 $\pm$ 0.01                           | 0.43 $\pm$ 0.01     |
| LogisticRegressionLasso         | Random Oversampling           | 0.33 $\pm$ 0.03 | 0.37 $\pm$ 0.01             | 0.36 $\pm$ 0.01         | 0.40 $\pm$ 0.01          | 0.39 $\pm$ 0.02                           | 0.45 $\pm$ 0.01     |
| LogisticRegressionLasso         | SMOTE                         | 0.31 $\pm$ 0.02 | 0.36 $\pm$ 0.00             | 0.35 $\pm$ 0.01         | 0.40 $\pm$ 0.01          | 0.38 $\pm$ 0.02                           | 0.45 $\pm$ 0.01     |
| LogisticRegressionLasso         | SVMSMOTE                      | 0.32 $\pm$ 0.03 | 0.38 $\pm$ 0.01             | 0.38 $\pm$ 0.01         | 0.40 $\pm$ 0.02          | 0.40 $\pm$ 0.03                           | 0.45 $\pm$ 0.01     |
| LogisticRegressionRidge         | BorderlineSMOTE               | 0.37 $\pm$ 0.02 | 0.37 $\pm$ 0.01             | 0.35 $\pm$ 0.02         | 0.38 $\pm$ 0.01          | 0.40 $\pm$ 0.02                           | 0.44 $\pm$ 0.01     |
| LogisticRegressionRidge         | No Upsampling                 | 0.35 $\pm$ 0.02 | 0.35 $\pm$ 0.01             | 0.34 $\pm$ 0.02         | 0.37 $\pm$ 0.01          | 0.38 $\pm$ 0.02                           | 0.43 $\pm$ 0.02     |
| LogisticRegressionRidge         | RakelD Partitioning of labels | 0.35 $\pm$ 0.04 | 0.34 $\pm$ 0.01             | 0.34 $\pm$ 0.02         | 0.36 $\pm$ 0.02          | 0.37 $\pm$ 0.01                           | 0.41 $\pm$ 0.01     |
| LogisticRegressionRidge         | Random Oversampling           | 0.37 $\pm$ 0.02 | 0.37 $\pm$ 0.02             | 0.36 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.40 $\pm$ 0.02                           | 0.44 $\pm$ 0.01     |
| LogisticRegressionRidge         | SMOTE                         | 0.37 $\pm$ 0.02 | 0.37 $\pm$ 0.01             | 0.36 $\pm$ 0.02         | 0.38 $\pm$ 0.01          | 0.40 $\pm$ 0.02                           | 0.44 $\pm$ 0.01     |
| LogisticRegressionRidge         | SVMSMOTE                      | 0.38 $\pm$ 0.01 | 0.38 $\pm$ 0.01             | 0.36 $\pm$ 0.02         | 0.37 $\pm$ 0.02          | 0.39 $\pm$ 0.02                           | 0.43 $\pm$ 0.02     |
| Multi-label ARAM                | Natively Multilabel           | 0.08 $\pm$ 0.00 | 0.08 $\pm$ 0.00             | 0.08 $\pm$ 0.00         | 0.08 $\pm$ 0.00          | 0.08 $\pm$ 0.00                           | 0.08 $\pm$ 0.00     |
| Multilabel k Nearest Neighbours | Natively Multilabel           | 0.38 $\pm$ 0.02 | 0.28 $\pm$ 0.05             | 0.31 $\pm$ 0.03         | 0.29 $\pm$ 0.06          | 0.22 $\pm$ 0.08                           | 0.32 $\pm$ 0.02     |
| NaiveBayes                      | BorderlineSMOTE               | 0.46 $\pm$ 0.01 | 0.49 $\pm$ 0.01             | 0.51 $\pm$ 0.02         | 0.54 $\pm$ 0.00          | 0.53 $\pm$ 0.01                           | 0.58 $\pm$ 0.01     |
| NaiveBayes                      | No Upsampling                 | 0.47 $\pm$ 0.03 | 0.47 $\pm$ 0.02             | 0.46 $\pm$ 0.02         | 0.42 $\pm$ 0.02          | 0.43 $\pm$ 0.02                           | 0.39 $\pm$ 0.01     |
| NaiveBayes                      | Random Oversampling           | 0.45 $\pm$ 0.02 | 0.50 $\pm$ 0.01             | 0.53 $\pm$ 0.02         | 0.55 $\pm$ 0.01          | 0.55 $\pm$ 0.01                           | **0.59 $\pm$ 0.00** |
| NaiveBayes                      | SMOTE                         | 0.46 $\pm$ 0.02 | 0.49 $\pm$ 0.01             | 0.52 $\pm$ 0.02         | 0.54 $\pm$ 0.01          | 0.53 $\pm$ 0.01                           | 0.58 $\pm$ 0.00     |
| RandomForest                    | BorderlineSMOTE               | 0.26 $\pm$ 0.03 | 0.28 $\pm$ 0.03             | 0.29 $\pm$ 0.03         | 0.33 $\pm$ 0.03          | 0.32 $\pm$ 0.03                           | 0.39 $\pm$ 0.02     |
| RandomForest                    | No Upsampling                 | 0.23 $\pm$ 0.02 | 0.22 $\pm$ 0.01             | 0.24 $\pm$ 0.02         | 0.29 $\pm$ 0.03          | 0.27 $\pm$ 0.03                           | 0.36 $\pm$ 0.03     |
| RandomForest                    | Random Oversampling           | 0.29 $\pm$ 0.02 | 0.31 $\pm$ 0.02             | 0.31 $\pm$ 0.02         | 0.37 $\pm$ 0.01          | 0.34 $\pm$ 0.01                           | 0.42 $\pm$ 0.02     |
| RandomForest                    | SMOTE                         | 0.26 $\pm$ 0.02 | 0.28 $\pm$ 0.01             | 0.27 $\pm$ 0.03         | 0.33 $\pm$ 0.02          | 0.33 $\pm$ 0.03                           | 0.40 $\pm$ 0.03     |
| RandomForest                    | SVMSMOTE                      | 0.27 $\pm$ 0.03 | 0.26 $\pm$ 0.00             | 0.27 $\pm$ 0.02         | 0.32 $\pm$ 0.03          | 0.31 $\pm$ 0.04                           | 0.40 $\pm$ 0.02     |
| RidgeClassifier                 | BorderlineSMOTE               | 0.37 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.01         | 0.39 $\pm$ 0.01          | 0.40 $\pm$ 0.01                           | 0.44 $\pm$ 0.01     |
| RidgeClassifier                 | No Upsampling                 | 0.37 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.01         | 0.39 $\pm$ 0.01          | 0.40 $\pm$ 0.01                           | 0.44 $\pm$ 0.01     |
| RidgeClassifier                 | Random Oversampling           | 0.37 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.01         | 0.39 $\pm$ 0.01          | 0.40 $\pm$ 0.01                           | 0.44 $\pm$ 0.01     |
| RidgeClassifier                 | SMOTE                         | 0.37 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.36 $\pm$ 0.01         | 0.39 $\pm$ 0.01          | 0.40 $\pm$ 0.01                           | 0.44 $\pm$ 0.01     |
| RidgeClassifier                 | SVMSMOTE                      | 0.39 $\pm$ 0.01 | 0.38 $\pm$ 0.02             | 0.37 $\pm$ 0.02         | 0.37 $\pm$ 0.02          | 0.39 $\pm$ 0.01                           | 0.43 $\pm$ 0.02     |
| SVM                             | BorderlineSMOTE               | 0.22 $\pm$ 0.02 | 0.08 $\pm$ 0.02             | 0.06 $\pm$ 0.03         | 0.05 $\pm$ 0.01          | 0.11 $\pm$ 0.02                           | 0.12 $\pm$ 0.04     |
| SVM                             | No Upsampling                 | 0.09 $\pm$ 0.02 | 0.04 $\pm$ 0.02             | 0.06 $\pm$ 0.01         | 0.10 $\pm$ 0.03          | 0.17 $\pm$ 0.03                           | 0.26 $\pm$ 0.03     |
| SVM                             | RakelD Partitioning of labels | 0.07 $\pm$ 0.03 | 0.04 $\pm$ 0.03             | 0.10 $\pm$ 0.04         | 0.06 $\pm$ 0.07          | 0.17 $\pm$ 0.04                           | 0.15 $\pm$ 0.05     |
| SVM                             | Random Oversampling           | 0.19 $\pm$ 0.03 | 0.08 $\pm$ 0.02             | 0.10 $\pm$ 0.01         | 0.14 $\pm$ 0.05          | 0.23 $\pm$ 0.01                           | 0.29 $\pm$ 0.01     |
| SVM                             | SMOTE                         | 0.23 $\pm$ 0.03 | 0.08 $\pm$ 0.02             | 0.05 $\pm$ 0.02         | 0.05 $\pm$ 0.01          | 0.08 $\pm$ 0.01                           | 0.13 $\pm$ 0.02     |
| SVM                             | SVMSMOTE                      | 0.34 $\pm$ 0.03 | 0.11 $\pm$ 0.02             | 0.07 $\pm$ 0.03         | 0.05 $\pm$ 0.02          | 0.06 $\pm$ 0.03                           | 0.09 $\pm$ 0.05     |
| XGBoost                         | BorderlineSMOTE               | 0.35 $\pm$ 0.01 | 0.37 $\pm$ 0.02             | 0.38 $\pm$ 0.03         | 0.38 $\pm$ 0.02          | 0.40 $\pm$ 0.01                           | 0.45 $\pm$ 0.01     |
| XGBoost                         | No Upsampling                 | 0.32 $\pm$ 0.01 | 0.34 $\pm$ 0.02             | 0.36 $\pm$ 0.03         | 0.37 $\pm$ 0.02          | 0.37 $\pm$ 0.01                           | 0.43 $\pm$ 0.01     |
| XGBoost                         | Random Oversampling           | 0.35 $\pm$ 0.02 | 0.36 $\pm$ 0.01             | 0.38 $\pm$ 0.02         | 0.40 $\pm$ 0.01          | 0.39 $\pm$ 0.01                           | 0.45 $\pm$ 0.01     |
| XGBoost                         | SMOTE                         | 0.34 $\pm$ 0.02 | 0.36 $\pm$ 0.01             | 0.38 $\pm$ 0.03         | 0.39 $\pm$ 0.02          | 0.41 $\pm$ 0.02                           | 0.45 $\pm$ 0.01     |
| XGBoost                         | SVMSMOTE                      | 0.34 $\pm$ 0.02 | 0.36 $\pm$ 0.01             | 0.37 $\pm$ 0.02         | 0.39 $\pm$ 0.01          | 0.39 $\pm$ 0.01                           | 0.44 $\pm$ 0.01     |