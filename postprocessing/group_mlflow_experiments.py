import os
import argparse

import pandas as pd
from tqdm import tqdm
import mlflow
from mlflow import log_metric, log_param, MlflowClient


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Gather mlruns as single csv')
    parser.add_argument('--df_filepath', type=str, default='./')
    parser.add_argument('--output_dir', type=str, default='./grouped_mlruns')
    parser.add_argument('--grouping_criterion', type=str, default=['unit_of_analysis'], nargs="*")
    parser.add_argument('--f1_micro_threshold', type=float, default=0.6)
    parser.add_argument('--exp_name_prefix', type=str, default='')

    arguments = parser.parse_args()

    return arguments


def group_experiments(runs_df: pd.DataFrame, grouping_cols: tuple[str] = ('unit_of_analysis', ),
                      f1_micro_threshold: float = 0.4, exp_name_prefix: str = ''):

    if 'test_f1_micro_mean' in runs_df.columns:
        runs_df = runs_df[runs_df.test_f1_micro_mean < f1_micro_threshold].copy()
    elif 'test_f1_micro' in runs_df.columns:
        runs_df = runs_df[runs_df.test_f1_micro < f1_micro_threshold].copy()

    for grouping_col_names, df in runs_df.groupby(grouping_cols):
        client = MlflowClient()

        new_exp_name = f"{exp_name_prefix}_all_runs_of_{'_'.join(grouping_cols)}"
        print(new_exp_name)
        new_exp_id = client.create_experiment(new_exp_name)

        try:
            # Add run information to the experiment
            for idx, row in df.iterrows():
                with mlflow.start_run(experiment_id=new_exp_id):
                    for col in df.columns:
                        if 'train' in col or 'test' in col:
                            log_metric(col, row[col])
                        else:
                            log_param(col, row[col])

                mlflow.end_run()
        except Exception as e:
            print(e)
            pass


if __name__ == '__main__':
    args = parse_arguments()
    runs_df = pd.read_csv(args.df_filepath)

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    group_experiments(runs_df=runs_df, grouping_cols=args.grouping_criterion,
                      f1_micro_threshold=args.f1_micro_threshold, exp_name_prefix=args.exp_name_prefix)

