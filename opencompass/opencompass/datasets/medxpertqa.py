import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedXpertQADataset(BaseDataset):
    """Dataset loader for MedXpertQA dataset in JSONL format."""

    @staticmethod
    def load(path: str, name: str, **kwargs) -> DatasetDict:
        """
        Load the MedXpertQA dataset from JSONL files, filtering by question_type.

        Args:
            path (str): Path to the dataset directory or file.
            name (str): Question type to filter by (only lines with matching question_type are loaded).
            **kwargs: Additional arguments.

        Returns:
            DatasetDict: A dictionary of datasets for each split (e.g., 'test', 'dev').
        """
        path = get_data_path(path)
        dataset = DatasetDict()

        for split in ['test', 'dev']:
            raw_data = []
            filename = osp.join(path, f'{split}.jsonl')

            # Check if file exists
            if not osp.exists(filename):
                continue
            try:
                with open(filename, encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            # Filter by question_type
                            if data.get('question_type') != name:
                                continue
                            item = {
                            'input': data['question'].split('\n')[0],
                            'A': data['options']['A'],
                            'B': data['options']['B'],
                            'C': data['options']['C'],
                            'D': data['options']['D'],
                            'E': data['options']['E'],
                            'F': data['options']['F'],
                            'G': data['options']['G'],
                            'H': data['options']['H'],
                            'I': data['options']['I'],
                            'J': data['options']['J'],
                            'target': data['label'],
                        }
                            raw_data.append(item)
                        except json.JSONDecodeError as e:
                            pass
                        except KeyError as e:
                            pass

                dataset[split] = Dataset.from_list(raw_data)

            except FileNotFoundError:
                pass
            except Exception as e:
                pass

        return dataset




@LOAD_DATASET.register_module()
class MedXpertQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        # label_map = {
        #     1: 'A',
        #     2: 'B',
        #     3: 'C',
        #     4: 'D',
        # }

        dataset = DatasetDict()
        # for split in ['dev', 'train', 'test']:
        for split in ['test', 'dev']:
            raw_data = []
            # if split == 'dev':
            #     filename = osp.join(path, 'JMED.jsonl')
            # elif split == 'test':
            filename = osp.join(path, f'{split}.jsonl')
            # print(f'Loading {filename}')
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    try:
                        raw_data.append({
                            'input': line['question'].split('\n')[0],
                            'A': line['options']['A'],
                            'B': line['options']['B'],
                            'C': line['options']['C'],
                            'D': line['options']['D'],
                            'E': line['options']['E'],
                            'F': line['options']['F'],
                            'G': line['options']['G'],
                            'H': line['options']['H'],
                            'I': line['options']['I'],
                            'J': line['options']['J'],
                            'target': line['label'],
                        })
                    except:
                        print(f"Error processing line: {line}")
                print(f"length of loaded raw_data {split}: {len(raw_data)}")
                dataset[split] = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class MMedXpertQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        # label_map = {
        #     1: 'A',
        #     2: 'B',
        #     3: 'C',
        #     4: 'D',
        # }
        
        dataset = DatasetDict()
        # for split in ['dev', 'train', 'test']:
        for split in ['test', 'dev']:
            raw_data = []
            # if split == 'dev':
            #     filename = osp.join(path, 'JMED.jsonl')
            # elif split == 'test':
            filename = osp.join(path, f'{split}.jsonl')
            # print(f'Loading {filename}')
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    if line.get('question_type') != name:
                        continue
                    try:
                        raw_data.append({
                            'input': line['question'].split('\n')[0],
                            'A': line['options']['A'],
                            'B': line['options']['B'],
                            'C': line['options']['C'],
                            'D': line['options']['D'],
                            'E': line['options']['E'],
                            'F': line['options']['F'],
                            'G': line['options']['G'],
                            'H': line['options']['H'],
                            'I': line['options']['I'],
                            'J': line['options']['J'],
                            'target': line['label'],
                        })
                    except:
                        print(f"Error processing line: {line}")
                print(f"length of loaded raw_data {split}: {len(raw_data)}")
                dataset[split] = Dataset.from_list(raw_data)
        return dataset