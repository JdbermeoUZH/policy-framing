| model_type                   |   title | title and first paragraph   |   title and 5 sentences | title and 10 sentences   |   title and first sentence each paragraph | raw text   |
|:-----------------------------|--------:|:----------------------------|------------------------:|:-------------------------|------------------------------------------:|:-----------|
| KNN                          |   0     | **0.126**                   |                   0.019 | 0.058                    |                                     0.097 | 0.107      |
| LinearSVM                    |   0     | 0.058                       |                   0.087 | 0.049                    |                                     0.117 | 0.068      |
| LogisticRegression           |   0.029 | 0.039                       |                   0.049 | 0.019                    |                                     0.068 | 0.068      |
| LogisticRegressionElasticNet |   0     | 0.058                       |                   0.058 | 0.049                    |                                     0.117 | 0.087      |
| LogisticRegressionLasso      |   0.049 | 0.058                       |                   0.068 | 0.068                    |                                     0.058 | 0.068      |
| LogisticRegressionRidge      |   0.058 | 0.117                       |                   0.107 | 0.078                    |                                     0.087 | 0.058      |
| NaiveBayes                   |   0     | 0                           |                   0.087 | 0.107                    |                                     0.097 | 0.068      |
| RandomForest                 |   0     | 0                           |                   0.117 | 0.049                    |                                     0.117 | 0.117      |
| RidgeClassifier              |   0     | 0.117                       |                   0.117 | **0.126**                |                                     0.087 | 0.078      |
| SVM                          |   0     | 0.087                       |                   0.097 | 0.097                    |                                     0.078 | 0.058      |
| XGBoost                      |   0     | 0.049                       |                   0.039 | 0.107                    |                                     0.078 | **0.126**  |