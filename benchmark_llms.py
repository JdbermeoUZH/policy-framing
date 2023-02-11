import os
import glob
import yaml
import argparse
from typing import Tuple

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EvalPrediction


LANGUAGES = ('en', 'it', 'fr', 'po', 'ru', 'ge')

LABELS = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
          'policy_prescription_and_evaluation', 'capacity_and_resources', 'economic', 'cultural_identity',
          'health_and_safety', 'quality_of_life', 'legality_constitutionality_and_jurisprudence',
          'political', 'public_opinion', 'external_regulation_and_reputation')

mlb = MultiLabelBinarizer()
mlb.fit([LABELS])

UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--analysis_unit', type=str, default=None)
    parser.add_argument('--max_length_padding', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--fp16', type=int, default=None)
    parser.add_argument('--minibatch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)

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

    if arguments.max_length_padding is not None:
        yaml_config_params['preprocessing']['max_length_padding'] = arguments.max_length_padding

    if arguments.analysis_unit is not None:
        yaml_config_params['preprocessing']['analysis_unit'] = arguments.analysis_unit

    if arguments.n_epochs is not None:
        yaml_config_params['training']['n_epochs'] = arguments.n_epochs

    if arguments.fp16 is not None:
        yaml_config_params['training']['fp16'] = arguments.fp16 == 1

    if arguments.minibatch_size is not None:
        yaml_config_params['training']['minibatch_size'] = arguments.minibatch_size

    if arguments.gradient_accumulation_steps is not None:
        yaml_config_params['training']['gradient_accumulation_steps'] = arguments.gradient_accumulation_steps

    return arguments, yaml_config_params


def one_hot_encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels_npy = mlb.transform(df.frames.str.lower().str.split(',')).astype(float)
    df['labels'] = [list(labels_npy[i, :]) for i in range(labels_npy.shape[0])]

    return df


def join_datasets(data_dir: str, type_of_splits_to_join: str) -> pd.DataFrame:
    df_paths = glob.glob(os.path.join(data_dir, f'*{type_of_splits_to_join}.csv'))
    df = None

    for i, df_path_i in enumerate(df_paths):
        df_i = pd.read_csv(df_path_i, index_col='id')
        df_i['language'] = os.path.basename(df_path_i).split('_')[1]

        if i == 0:
            df = df_i
        else:
            df = pd.concat([df, df_i])

    df = one_hot_encode_labels(df)

    return df


def split_to_train_test_with_iterstrat(df: pd.DataFrame, splits: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:

    mskf = MultilabelStratifiedKFold(n_splits=splits, shuffle=True, random_state=0)

    train_dfs = []
    test_dfs = []

    for language, df_ in df.groupby('language'):
        X = df_[[col for col in df.columns if col not in ['labels', 'frames']]]
        y = df_[[col for col in df.columns if col in ['labels', 'frames']]]

        for train_index, test_index in mskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            train_dfs.append(X_train.join(y_train))
            test_dfs.append(X_test.join(y_test))

            break

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    return train_df, test_df


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
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    # return as dictionary
    metrics = {
        'f1': f1_micro_average,
        'precision': precision,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'recall': recall
    }

    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def load_hf_dataset(single_train_test_split_filepath: str, data_dir_path: str):
    if single_train_test_split_filepath is None or single_train_test_split_filepath.strip() == '':

        # TODO: Update with function used to generate the HF dataset
        all_languages_df = join_datasets(
            data_dir=data_dir_path,
            type_of_splits_to_join=preprocessing_config['data_split_to_use']
        )

        train_df_, test_df_ = split_to_train_test_with_iterstrat(
            all_languages_df, splits=preprocessing_config['n_folds'])

        dataset = DatasetDict({'train': Dataset.from_pandas(train_df_), 'test': Dataset.from_pandas(test_df_)})

        dataset.save_to_disk(os.path.join(data_dir_path, 'multilingual_train_test_ds.hf'))

    else:
        dataset = load_from_disk(
            os.path.join(data_dir_path, single_train_test_split_filepath))

    return dataset


def preprocess_data(examples, unit_of_analysis):
    # take a batch of texts
    text = examples[unit_of_analysis]

    # encode them
    encoding = tokenizer(text, truncation=True)

    # Add their respective labels
    encoding["labels"] = examples['label']

    return encoding


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()

    dataset_config = config['dataset']
    model_config = config['model']
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    output_config = config['output']

    # Load Multi-lingual Dataset. It will be stratified per language and label
    dataset = load_hf_dataset(
        single_train_test_split_filepath=os.path.join(*preprocessing_config['single_train_test_split_filepath']),
        data_dir_path=os.path.join(*dataset_config['data_dir'])
    )

    print('Dataset loaded')

    if model_config['fine_tune']:
        id2label = {idx: label for idx, label in enumerate(mlb.classes_)}
        label2id = {label: idx for idx, label in enumerate(mlb.classes_)}

        msg_str = f"Fine tuning the model: {model_config['model_name']}"
        print(msg_str + '\n' + ''.join(['#'] * len(msg_str)))

        # Define tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config['model_name'],
            problem_type="multi_label_classification",
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        )

        # Define Training parameters and Trainer
        training_args = TrainingArguments(
            f"{model_config['model_name']}-{preprocessing_config['analysis_unit']}-sem_eval-task-3-subtask-2",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=training_config['minibatch_size'],
            per_device_eval_batch_size=training_config['minibatch_size'],
            num_train_epochs=training_config['n_epochs'],
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=training_config['best_metric_to_checkpoint'],
            fp16=training_config['fp16'],
            warmup_ratio=training_config['warmup_ratio'],
            dataloader_num_workers=training_config['dataloader_num_workers']
        )

        # Fit and measure the model performance on each fold of the dataset
        n_folds = max([int(key.split('_')[-1]) for key in dataset.keys() if len(key.split('_')) == 3])

        for fold_i in range(1, n_folds):
            msg_str = f"Fitting fold: {fold_i}"
            print(msg_str + '\n' + ''.join(['-'] * len(msg_str)) + '\n')

            # Tokenize/Encode the dataset
            encoded_dataset = dataset.map(
                lambda ex: preprocess_data(ex, preprocessing_config['data_split_to_use']), batched=True,
                remove_columns=dataset[f'train_fold_{fold_i}'].column_names)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            trainer = Trainer(
                model,
                training_args,
                train_dataset=encoded_dataset[f"train_fold_{fold_i}"],
                eval_dataset=encoded_dataset[f"test_fold_{fold_i}"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            # Train the model
            trainer.train()

            # Evaluate the model on each test set individually
            metrics_ = []
            msg_str = "Evaluation metrics for each dataset"
            print(msg_str + '\n' + ''.join(['#'] * len(msg_str)) + '\n')

            for language in LANGUAGES:
                msg_str = f"For the dataset: {language}"
                print(msg_str + '\n' + ''.join(['-'] * len(msg_str)))

                trainer = Trainer(
                    model,
                    training_args,
                    train_dataset=encoded_dataset[f"train_fold_{fold_i}_{language}"],
                    eval_dataset=encoded_dataset[f"test_fold_{fold_i}_{language}"],
                    tokenizer=tokenizer,
                    # data_collator=data_collator,
                    compute_metrics=compute_metrics
                )

                evaluation_results_i = trainer.evaluate()

                print('\n')

                metrics_.append({
                    'language': language,
                    'unit_of_analysis': preprocessing_config['data_split_to_use'],
                    f'fold_{fold_i}': f'fold_{fold_i}',
                    'f1-mico': evaluation_results_i['eval_f1'],
                    'precision-micro': evaluation_results_i['eval_precision'],
                    'recall-micro': evaluation_results_i['eval_recall'],
                    'roc-auc': evaluation_results_i['eval_roc_auc'],
                    'accuracy': evaluation_results_i['eval_accuracy']
                })

        output_dir = os.path.join(*output_config['metrics_output_dir'], model_config['model_name'])
        os.makedirs(os.path.join(*output_config['metrics_output_dir']), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(
            output_dir,
            f"{output_config['file_prefix']}_raw_{model_config['model_name']}-{preprocessing_config['analysis_unit']}_metrics.csv"
        )
        pd.DataFrame(metrics_).to_csv(metrics_path)

