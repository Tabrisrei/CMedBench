import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


CHOICES=['A', 'B', 'C', 'D', 
         'E', 'F', 'G', 'H', 
         'I', 'J', 'K', 'L', 
         'M', 'N', 'O', 'P']

@LOAD_DATASET.register_module()
class MMLUproQADataset(BaseDataset):

    @staticmethod
    def load(path: str, category: str):
        path = get_data_path(path)

        dataset = DatasetDict()
        raw_data = []
        
        filename = osp.join(path, 'test-00000-of-00001.parquet')
        dataset = load_dataset('parquet', data_files=filename, split='train')
        # reformat the dataset
        for data in dataset:

            if data['category'] != category:
                continue

            raw_data_item = dict()
            raw_data_item['input'] = data['question']
            raw_data_item['target'] = data['answer']

            num_choices = len(data['options'])
            choices = [CHOICES[i] for i in range(num_choices)]
            for choice, option in zip(choices, data['options']):
                raw_data_item[choice] = option
                
            raw_data.append(raw_data_item)

        dataset = Dataset.from_list(raw_data)
        return dataset
