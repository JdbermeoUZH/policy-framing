import os
import glob
import yaml
import pprint
import argparse
from typing import Tuple

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EvalPrediction

from utils.constants import LANGUAGES, LABELS


mlb = MultiLabelBinarizer()
mlb.fit([LABELS])


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--analysis_unit', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None, nargs="*")
    parser.add_argument('--single_train_test_split_filepath', type=str, default=None, nargs="*")
    parser.add_argument('--max_length_padding', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--fp16', type=int, default=None)
    parser.add_argument('--load_in_8bit', type=int, default=None)
    parser.add_argument('--minibatch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--truncated', type=int, default=None)
    parser.add_argument('--metrics_output_dir', type=str, default=None, nargs="*")
    parser.add_argument('--fit_only_on_specific_language', type=str, default=None)

    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    if arguments.model_name is not None:
        yaml_config_params['model']['model_name'] = arguments.model_name

    if arguments.fp16 is not None:
        yaml_config_params['model']['fp16'] = arguments.fp16 == 1

    if arguments.load_in_8bit is not None:
        yaml_config_params['model']['load_in_8bit'] = arguments.load_in_8bit == 1

    if arguments.data_dir is not None:
        yaml_config_params['data']['data_dir'] = arguments.data_dir

    if arguments.single_train_test_split_filepath is not None:
        yaml_config_params['preprocessing']['single_train_test_split_filepath'] = arguments.single_train_test_split_filepath

    if arguments.max_length_padding is not None:
        yaml_config_params['preprocessing']['max_length_padding'] = arguments.max_length_padding

    if arguments.analysis_unit is not None:
        yaml_config_params['preprocessing']['analysis_unit'] = arguments.analysis_unit

    if arguments.truncated is not None:
        yaml_config_params['preprocessing']['truncated'] = arguments.truncated == 1

    if arguments.n_epochs is not None:
        yaml_config_params['training']['n_epochs'] = arguments.n_epochs

    if arguments.minibatch_size is not None:
        yaml_config_params['training']['minibatch_size'] = arguments.minibatch_size

    if arguments.gradient_accumulation_steps is not None:
        yaml_config_params['training']['gradient_accumulation_steps'] = arguments.gradient_accumulation_steps

    if arguments.fit_only_on_specific_language is not None:
        yaml_config_params['training']['fit_only_on_specific_language'] = arguments.fit_only_on_specific_language

    if arguments.metrics_output_dir is not None:
        yaml_config_params['output']['metrics_output_dir'] = arguments.metrics_output_dir

    print('command line args:')
    pprint.pprint(arguments)
    print('\n\n')

    print('config args: ')
    pprint.pprint(yaml_config_params)
    print('\n\n')

    return arguments, yaml_config_params


def one_hot_encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels_npy = mlb.transform(df.frames.str.lower().str.split(',')).astype(float)
    df['labels'] = [list(labels_npy[i, :]) for i in range(labels_npy.shape[0])]

    return df


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # finally, compute metrics
    y_true = labels
    metrics = _compute_multi_label_metrics(y_pred, y_true)

    return metrics


def _compute_multi_label_metrics(y_pred, y_true):
    # return as dictionary
    metrics = {
        'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def preprocess_data(examples, unit_of_analysis):
    # take a batch of texts
    text = examples[unit_of_analysis]

    # encode them
    encoding = tokenizer(text, truncation=True)

    # Add their respective labels
    encoding["labels"] = examples['label']

    return encoding


def measure_performance_truncated_dataset(language_: str, metrics_list: list):

        trainer_ = Trainer(
            model,
            training_args,
            train_dataset=encoded_dataset[f"train_{language_}"],
            eval_dataset=encoded_dataset[f"test_{language_}"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        evaluation_results_i = trainer_.evaluate()

        print('\n')

        metrics_list.append({
            'language': language_,
            'unit_of_analysis': preprocessing_config['analysis_unit'],
            'f1_micro': evaluation_results_i['eval_f1_micro'],
            'precision_micro': evaluation_results_i['eval_precision_micro'],
            'recall_micro': evaluation_results_i['eval_recall_micro'],
            'f1_macro': evaluation_results_i['eval_f1_macro'],
            'precision_macro': evaluation_results_i['eval_precision_macro'],
            'recall_macro': evaluation_results_i['eval_recall_macro'],
            'accuracy': evaluation_results_i['eval_accuracy']
        })

        del trainer_


def measure_and_record_metrics(preds_df: pd.DataFrame, true_label_df_: pd.DataFrame, metrics_list: list,
                               language_: str):

    metrics_dict = _compute_multi_label_metrics(
        y_pred=(preds_df > 0.5).astype(int).values, y_true=true_label_df_.astype(int).values)

    metrics_list.append({
        'language': language_,
        'unit_of_analysis': preprocessing_config['analysis_unit'],
        **metrics_dict
    })


def measure_performance_chunked_dataset(language_: str, output_dir_: str):
    # Create DataLoader to iterate for samples of the fold on the intended language
    dataloader = DataLoader(encoded_dataset[f"test_{language_}"],
                            collate_fn=data_collator, batch_size=training_config['minibatch_size'])
    ids_list, true_labels_list, pred_labels_list = [], [], []

    # Evaluate predictions
    for step, batch in enumerate(tqdm(dataloader)):
        ids_batch = batch.pop('id')

        batch.to('cuda')

        with torch.no_grad():
            out = model(**batch)
            pred_labels = torch.sigmoid(out['logits'])

        ids_list.append(ids_batch.to('cpu').numpy())
        pred_labels_list.append(pred_labels.to('cpu').numpy())
        true_labels_list.append(batch['labels'].to('cpu').numpy())

    # Stack the predictions into single dataframe
    preds_df = pd.DataFrame(np.concatenate(pred_labels_list), index=np.concatenate(ids_list)) \
        .sort_index()
    preds_df.index.names = ['id']
    true_label_df = pd.DataFrame(np.concatenate(true_labels_list), index=np.concatenate(ids_list)) \
        .reset_index().drop_duplicates().set_index('index').sort_index()
    true_label_df.index.names = ['id']

    # Get summary predictions of the model
    mean_score_pred_df = preds_df.groupby(level=0).mean()
    majority_voting_df = (preds_df > 0.5).astype(int).groupby(level=0).mean()

    # Store the predicted scores
    pred_scores_dir = os.path.join(output_dir_, 'prediction_scores')
    os.makedirs(pred_scores_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(
        pred_scores_dir, f"{output_config['file_prefix']}_"
                         f"pred_scores_test_fold_{preprocessing_config['analysis_unit']}_{language_}.csv"))

    # Measure the metrics and store them in a dictionary
    measure_and_record_metrics(preds_df=mean_score_pred_df, true_label_df_=true_label_df,
                               metrics_list=metrics['mean_predicted_score'], language_=language_)

    measure_and_record_metrics(preds_df=majority_voting_df, true_label_df_=true_label_df,
                               metrics_list=metrics['majority_voting'], language_=language_)


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()

    dataset_config = config['dataset']
    model_config = config['model']
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    output_config = config['output']

    # Create output directory that will be used to store results
    output_dir = os.path.join(*output_config['metrics_output_dir'], model_config['model_name'])
    os.makedirs(os.path.join(*output_config['metrics_output_dir']), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load Multi-lingual Dataset. It will be stratified per language and label
    dataset = load_from_disk(
        os.path.join(*dataset_config['data_dir'], *preprocessing_config['single_train_test_split_filepath']))

    msg_str = f"Using the dataset: {preprocessing_config['single_train_test_split_filepath']}"
    print(msg_str + '\n' + ''.join(['#'] * len(msg_str)) + '\n')

    if model_config['fine_tune']:
        id2label = {idx: label for idx, label in enumerate(mlb.classes_)}
        label2id = {label: idx for idx, label in enumerate(mlb.classes_)}

        msg_str = f"Fine tuning the model: {model_config['model_name']}"
        print(msg_str + '\n' + ''.join(['#'] * len(msg_str)) + '\n')

        print(f'\t analysis_unit: {preprocessing_config["analysis_unit"]}')
        print(f'\t gradient_accumulation_steps: {training_config["n_epochs"]}')
        print(f'\t minibatch_size: {training_config["minibatch_size"]}')
        print(f'\t n_epochs: {training_config["n_epochs"]}')
        print(f'\t truncated: {preprocessing_config["truncated"]}')

        metrics = {}

        if not preprocessing_config['truncated']:
            # This means the text was split into chunks of certain length
            metrics['mean_predicted_score'] = []
            metrics['majority_voting'] = []
        else:
            metrics['truncated_single_instance'] = []

        # Define tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize/Encode the dataset
        unit_of_analysis = preprocessing_config['analysis_unit']
        if not preprocessing_config['truncated']:
            # This means the text was split into chunks of certain length
            unit_of_analysis += '_chunked'
            col_to_remove_at_inference = [col for col in dataset['train'].column_names if col != 'id']
        else:
            col_to_remove_at_inference = dataset['train'].column_names

        encoded_dataset = dataset.map(
            lambda ex: preprocess_data(ex, unit_of_analysis), batched=True,
            remove_columns=col_to_remove_at_inference)

        # Define Training parameters and Trainer
        training_args = TrainingArguments(
            f"{model_config['model_name']}-{preprocessing_config['analysis_unit']}-sem_eval-task-3-subtask-2",
            learning_rate=2e-5,
            per_device_train_batch_size=training_config['minibatch_size'],
            per_device_eval_batch_size=training_config['minibatch_size'],
            num_train_epochs=training_config['n_epochs'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            weight_decay=0.01,
            load_best_model_at_end=False,
            evaluation_strategy="epoch",
            fp16=model_config['fp16'] if not model_config['load_in_8bit'] else False,
            warmup_ratio=training_config['warmup_ratio']
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_config['model_name'],
            problem_type="multi_label_classification",
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id,
            load_in_8bit=model_config['load_in_8bit'],
            device_map='auto' if model_config['load_in_8bit'] else None
        )

        if tokenizer.pad_token == tokenizer.eos_token:
            model.config.pad_token_id = tokenizer.pad_token_id

        if training_config['fit_only_on_specific_language']:
            trainset_key = f'train_{training_config["fit_only_on_specific_language"]}'
            test_set_key = f'test_{training_config["fit_only_on_specific_language"]}'
        else:
            trainset_key = 'train'
            test_set_key = 'test'

        trainer = Trainer(
            model,
            training_args,
            train_dataset=encoded_dataset[trainset_key],
            eval_dataset=encoded_dataset[test_set_key],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model on each test set individually
        msg_str = "Evaluation metrics for each dataset"
        print(msg_str + '\n' + ''.join(['#'] * len(msg_str)) + '\n')

        for language in LANGUAGES:
            msg_str = f"For the dataset: {language}"
            print(msg_str + '\n' + ''.join(['-'] * len(msg_str)))

            if preprocessing_config['truncated']:
                measure_performance_truncated_dataset(
                    language_=language, metrics_list=metrics['truncated_single_instance']
                )

            else:
                # Measure performance when examples are split into chunks
                measure_performance_chunked_dataset(language_=language, output_dir_=output_dir)

        del model
        del encoded_dataset

        for measurement_type, metrics_list in metrics.items():
            # Save metrics in a csv file
            base_metrics_name = f"{measurement_type}_{model_config['model_name'].replace('/','_')}" \
                                f"-{preprocessing_config['analysis_unit']}_metrics.csv"

            raw_metrics_df = pd.DataFrame(metrics_list).set_index(['language']).sort_index()
            raw_metrics_df.to_csv(os.path.join(output_dir, f"{output_config['file_prefix']}_raw_{base_metrics_name}"))
