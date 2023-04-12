import os
import sys
from datetime import datetime
from datasets import (
    Dataset,
    DatasetDict
)
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
            split_frac: List[float],
            feat_static_cat: List[Any],
            feat_dynamic_real: List[Any],
            item_id: List[str]) -> None:
        self.start = start
        self.target = target
        self.feat_static_cat = feat_static_cat
        self.feat_dynamic_real = feat_dynamic_real
        self.item_id = item_id

    def getDataset(self, split: str) -> Dataset:
        split_limit = get_split_limit(self.target,  split)
        return Dataset.from_dict(
                {
                    'start': self.start,
                    'target': self.target[:split_limit],
                    'feat_static_cat': self.feat_static_cat,
                    'feat_dynamic_real': self.feat_dynamic_real,
                    'item_id': self.item_id
                }
            )

    def getDatasetDict(self) -> DatasetDict:
        return DatasetDict({_split[0]: self.getDataset(_split[0])})
