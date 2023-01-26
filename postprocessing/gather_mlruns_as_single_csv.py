import os
import argparse

import pandas as pd
from tqdm import tqdm
from mlflow import MlflowClient


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Gather mlruns as single csv')
    parser.add_argument('--target_dir', type=str, default='./')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--analysis_level', type=str)
    parser.add_argument('--language', type=str)

    arguments = parser.parse_args()

    return arguments


def record_mlflow_runs_as_csv(exp_ids: list[str], analysis_level: str, language: str = '', output_dir='./'):
    rows = []
    output_filepath = os.path.join(output_dir, f'{language}_all_{analysis_level}_results.csv')
    os.makedirs(output_dir, exist_ok=True)

    client = MlflowClient()

    query = f"params.analysis_level = '{analysis_level}'"
    if language != '':
        query += f"and params.language = '{language}'"

    for exp_i, exp_id in tqdm(enumerate(exp_ids)):
        # Retrieve 'model_wide' runs of the experiment
        try:
            runs = client.search_runs(experiment_ids=[exp_id], filter_string=query)

        except Exception as e:
            print(f'problems with exp_id: {exp_id} in position {exp_i}')
            print(e)
            continue

        # Add relevant parameters to the row
        for run in runs:
            row_dict = {'exp_id': exp_id, 'run_uuid': run.info.run_uuid, 'runName': run.data.tags['mlflow.runName'],
                        **run.data.metrics, **run.data.params}
            rows.append(row_dict)

        if exp_i % 100 == 0 and exp_i != 0 and len(rows) > 0:
            pd.DataFrame(rows).set_index(['run_uuid', 'model_name', 'runName']).sort_index().to_csv(output_filepath)

    pd.DataFrame(rows).set_index(['run_uuid', 'model_name', 'runName']).sort_index().to_csv(output_filepath)


if __name__ == '__main__':
    args = parse_arguments()
    os.chdir(args.target_dir)
    exp_ids_ = os.listdir('./mlruns')

    # Print the number of experiments registered
    print(f'A total of {len(exp_ids_)} experiments were recorded')

    # Write all model_wide metrics into a single csv file
    record_mlflow_runs_as_csv(exp_ids=exp_ids_, analysis_level=args.analysis_level, language=args.language,
                              output_dir=args.output_dir)

