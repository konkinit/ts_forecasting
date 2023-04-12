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
    get_split_limit
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
            target: List[List[float]],
            split_frac: ndarray,
            feat_static_cat: List[Any],
            feat_dynamic_real: List[Any],
            item_id: List[str]) -> None:
        self.start = start
        self.target = target
        self.split_frac = split_frac
        self.feat_static_cat = feat_static_cat
        self.feat_dynamic_real = feat_dynamic_real
        self.item_id = item_id

    def getDataset(self, split_index: int) -> Dataset:
        split_limit = get_split_limit(
                        self.target,
                        self.split_frac
                    )
        return Dataset.from_dict(
                {
                    'start': self.start,
                    'target': self.target[:split_limit[split_index]],
                    'feat_static_cat': self.feat_static_cat,
                    'feat_dynamic_real': self.feat_dynamic_real,
                    'item_id': self.item_id
                }
            )

    def getDatasetDict(self) -> DatasetDict:
        return DatasetDict({_split[0]: self.getDataset(0)})
