from abc import ABC
from typing import Dict, Any
import pandas as pd
import hashlib
import json
import logging
import pandas as pd
from pathlib import Path
import sys
from datasets import Array2D, Image, load_from_disk, Value


class DataSetLoader(ABC):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        self.dataset_path = dataset_path
        self.opt_args = opt_args

    def load_data(self):
        raise NotImplementedError


class CSVDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)

    def load_data(self):
        return pd.read_csv(self.dataset_path, **self.opt_args)


class JSONDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)

    def load_data(self):
        return pd.read_json(self.dataset_path, **self.opt_args)


class FeatherDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)

    def load_data(self):
        return pd.read_feather(self.dataset_path, **self.opt_args)


class ParquetDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)

    def load_data(self):
        return pd.read_parquet(self.dataset_path, **self.opt_args)


class PickleDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)
        print("opt_args", self.opt_args)

    def load_data(self):
        return pd.read_pickle(self.dataset_path, **self.opt_args)


class HuggingFaceDataSetLoader(DataSetLoader):
    def __init__(self, dataset_path: str, opt_args: Dict[str, Any] = None) -> None:
        super().__init__(dataset_path, opt_args)
        self.arrays = set()
        self.images = set()

    def load_data(self):
        dataset = load_from_disk(self.dataset_path)
        for key, feature in dataset.features.items():
            if isinstance(feature, Array2D):
                self.arrays.add(key)
            if isinstance(feature, Image):
                self.images.add(key)

        copy_of_features = dataset.features.copy()
        for feature in self.arrays:
            copy_of_features[feature] = Value(dtype="string", id=None)
        for feature in self.images:
            copy_of_features[feature] = Value(dtype="string", id=None)

        string_dataset = self.dataset.map(
            self._list_to_str,
            batched=True,
            batch_size=100,
            num_proc=1,
            keep_in_memory=True,
            features=copy_of_features,
        )
        return string_dataset.to_pandas()

    def _to_hash(self, encoded_text):
        hash_object = hashlib.md5(encoded_text)
        return hash_object.hexdigest()

    def _list_to_str(self, examples):
        for key in self.arrays:
            examples[key] = [
                self._to_hash("".join(str(value) for value in a_list).encode())
                for a_list in examples[key]
            ]

        for key in self.images:
            examples[key] = [
                self._to_hash(an_image.tobytes()) for an_image in examples[key]
            ]
        return examples
