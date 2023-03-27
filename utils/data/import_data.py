from collections import defaultdict
from typing import Dict, Any
import pandas as pd
from utils.data.load_datasets import (
    CSVDataSetLoader,
    JSONDataSetLoader,
    FeatherDataSetLoader,
    ParquetDataSetLoader,
    PickleDataSetLoader,
    HuggingFaceDataSetLoader,
)


DATASET_TYPES = {
    "csv": CSVDataSetLoader,
    "json": JSONDataSetLoader,
    "feather": FeatherDataSetLoader,
    "parquet": ParquetDataSetLoader,
    "df": PickleDataSetLoader,
    "huggingface": HuggingFaceDataSetLoader,
}


def process_dataset(
    dataset_path: str,
    dataset_type: str,
    opt_args: Dict[str, Any] = None,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:

    _dataset_type = dataset_type.lower()
    if _dataset_type not in DATASET_TYPES:
        raise KeyError(f"Invalid dataset type: {_dataset_type}")

    dataset_loader = DATASET_TYPES[_dataset_type](dataset_path, (opt_args or {}))
    dataset = dataset_loader.load_data()
    if kwargs["max_rows"]:
        dataset = dataset.head(kwargs["max_rows"])
    return dataset
