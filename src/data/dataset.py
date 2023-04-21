import os
import sys
from datetime import datetime
from datasets import (
    Dataset,
    DatasetDict
)
from numpy import ndarray
from typing import List, Any
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


class HF_Dataset:
    def __init__(
            self,
            start: List[datetime],
            time_series: List[List[float]],
            feat_static_cat: List[List[Any]],
            feat_dynamic_real: List[List[Any]],
            split_frac: ndarray,
            freq: str) -> None:
        assert (len(start) ==
                len(time_series))
        self.start = start
        self.target = time_series
        self.split_frac = split_frac
        self.feat_static_cat = feat_static_cat
        self.feat_dynamic_real = feat_dynamic_real
        self.freq = freq

    def getDataset(self, split_index: int) -> Dataset:
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
                        self.target[i][:split_limit[i][split_index]] for
                        i in range(n_ts)
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
        return DatasetDict(
            {_split[i]: self.getDataset(i) for i in range(len(_split))}
            )

    def multi_variate_datasets(self):
        return DataProcessing(
                    self.getDatasetDict()
                ).multi_variate_format(self.freq)


class XGB_Dataset:
    def __init__(self):
        pass
