# Run experiments

Create conda env with: `conda env create -f conda_env.yml`

This project uses two main config files:

  * `config.yaml`: The first and more general of the configurations. It contains settings at the preprocessing, training and metric logging level.
                   Here you choose the dataset language you want to use, list of models, preprocessing parameters and so on. 
  * `training/estimators_config.py`: This file contains a dictionary with the models we are going to fit, the number of iterations during tuning with randomSearch, and the search space of their hyperparameters.

Modify both config files to define how you will run the experiments. You run the experiments with the file: `benchmark_subtask_2.py --config_path config.yaml`

# Check results

Run the command `mlflow ui` from the command line in the root directory of the project. This should launch the MLFlow UI to check the performance of the experiments you run. 