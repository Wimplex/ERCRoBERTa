import warnings
warnings.filterwarnings('ignore')

import os
import tqdm
import argparse
from typing import Tuple, Iterable
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pandas as pd

from datasets import load_metric, Dataset
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from utils.base import set_seed
from utils.task import Task


class TrainTask(Task):
    def __init__(self, config: DictConfig):
        self.config = config
        self.metric = load_metric('accuracy')
        self.tokenizer = RobertaTokenizer \
            .from_pretrained(**config['task']['tokenizer'])
        self.model = RobertaForSequenceClassification \
            .from_pretrained(**config['task']['model'])


    def _tokenize_function(self, examples):
        return self.tokenizer(examples['utt_context'], padding='max_length', truncation=True)


    def _compute_metrics(self, eval_pred: Tuple[Iterable, Iterable]):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)


    def _preprocess_dataset(self, df_dataset: pd.DataFrame, ds_name: str) -> pd.DataFrame:
        """ Extracts dialog contexts for every utterance. """

        dset_config = self.config['task']['dataset']
        context_size = dset_config['dialogue_context_size']
        sep_token = dset_config['sep_token']
        contexts = []

        df_dataset.sort_values(by=['dia_id', 'utt_id'], inplace=True)

        for i in tqdm.tqdm(range(df_dataset.shape[0]), desc=ds_name):
            current_dialogue_id = df_dataset.iloc[i,:]['dia_id']
            current_context = []

            # Iterate through context
            for j in range(context_size)[::-1]:
                curr_index = i - j

                # Check if index is not out of bounds
                if curr_index >= 0:
                    row = df_dataset.iloc[curr_index,:]

                    # Check if current context's dia_id is the same as original's one
                    if row['dia_id'] == current_dialogue_id:
                        current_context.append(row['translated_utterance'])
            
            current_context = f' {sep_token} '.join(current_context)
            contexts.append(current_context)

        df_dataset['utt_context'] = contexts
        df_dataset.rename(columns={'emo': 'label'}, inplace=True)
        return df_dataset


    def _setup_datasets(self):
        """ Prepares train and eval datasets """

        dset_dict = {'train': None, 'eval': None}
        for base_name in self.config['datasets']:
            ds_config = OmegaConf.load(f'configs/data/{base_name}.yaml')

            # Iterates through partitions in dataset
            for partition in dset_dict.keys():
                partition_csv_path = os.path.join(
                    ds_config['base_path'], ds_config['partitions'][partition])
                df_ds = pd.read_csv(partition_csv_path)

                # Preprocess current DataFrame and convert it to dict form
                df_ds = self._preprocess_dataset(df_ds, ds_name=f"{base_name}/{partition}")
                df_data = df_ds.loc[:,['utt_context', 'label']].to_dict('list')

                # Concatenate to already prepared data
                if dset_dict[partition] is None:
                    dset_dict[partition] = df_data
                else:
                    for k, v in df_data.items():
                        dset_dict[partition][k] += v

        # Convert to batched HF-dataset
        for part in dset_dict.keys():
            dset_dict[part] = Dataset.from_dict(dset_dict[part]).map(
                self._tokenize_function, 
                batched     = True, 
                batch_size  = self.config['task']['dataset']['batch_size']
            ).shuffle()
        return dset_dict


    def _setup_task(self) -> Trainer:
        print("Datasets preparation.")
        datasets = self._setup_datasets()

        training_args = TrainingArguments(do_train=True, **self.config['task']['training_args'])
        print("Training.")
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = datasets['train'],
            eval_dataset = datasets['eval'],
            compute_metrics = self._compute_metrics
        )
        return trainer


    def run(self, random_seed: int):
        set_seed(random_seed)
        trainer = self._setup_task()
        trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/tasks/emo.yaml', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task = TrainTask(config)
    task.run(451)


if __name__ == '__main__':
    main()