import os
import sys
from dataclasses import dataclass
from datetime import datetime
from datasets import (
    Dataset,
    DatasetDict
)
from numpy import ndarray
from typing import (
    List, Any, Tuple
)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    get_split_limit,
    DataProcessing
)


_split = [
    "train",
    "validation",
    "test"
]


@dataclass
class HF_Dataset_Params:
    """a dataclass to store the params of the
    following class
    """
    start: List[datetime]
    time_series: List[List[float]]
    feat_static_cat: List[List[Any]]
    feat_dynamic_real: List[List[Any]]
    split_frac: ndarray
    freq: str


class HF_Dataset:
    """Transform a time serie in list type to a
    Hugging Face dataset format
    """
    def __init__(
            self,
            hf_dataset_params: HF_Dataset_Params) -> None:
        assert (len(hf_dataset_params.start) ==
                len(hf_dataset_params.time_series))
        self.start = hf_dataset_params.start
        self.target = hf_dataset_params.time_series
        self.split_frac = hf_dataset_params.split_frac
        self.feat_static_cat = hf_dataset_params.feat_static_cat
        self.feat_dynamic_real = hf_dataset_params.feat_dynamic_real
        self.freq = hf_dataset_params.freq

    def getDataset(self, split_index: int) -> Dataset:
        """Create a dataset for a given partition

        Args:
            split_index (int): index of the partition according
            to the list ["train", "validation", "test"]

        Returns:
            Dataset: HF dataset with 5 fields
        """
        assert split_index <= 2
        n_ts = len(self.target)
        split_limit = [
                    get_split_limit(
                        self.target[i],
                        self.split_frac
                    ) for i in range(n_ts)]
        return Dataset.from_dict(
                {
                    'start': self.start,
                    'target': [
                        self.target[i][:split_limit[i][split_index]]
                        for i in range(n_ts)
                        ],
                    'feat_static_cat': [
                        self.feat_static_cat[i][:split_limit[i][split_index]]
                        for i in range(n_ts)
                        ],
                    'feat_dynamic_real': [
                        self.feat_dynamic_real[i][:split_limit[i][split_index]]
                        for i in range(n_ts)
                        ],
                    'item_id': [f"T{i}" for i in range(n_ts)]
                }
            )

    def getDatasetDict(self) -> DatasetDict:
        """Groups the datasets for the 3 partitions

        Returns:
            DatasetDict: dict of train/valid/test datasets
        """
        return DatasetDict(
            {_split[i]: self.getDataset(i) for i in range(len(_split))}
            )

    def multi_variate_datasets(self) -> Tuple[Dataset]:
        """Converts dataset to a multivariate time
        serie

        Returns:
            Tuple[Dataset]: tuple of multivariate time series
        """
        return DataProcessing(
                    self.getDatasetDict()
                ).multi_variate_format(self.freq)

    def get_num_of_variates(self) -> int:
        """ Get the number of time series in the dataset

        Returns:
            int: number of time series
        """
        return DataProcessing(
                    self.getDatasetDict()
                ).get_num_of_variates()


class XGB_Dataset:
    def __init__(self):
        pass
