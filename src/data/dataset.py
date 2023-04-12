from datasets import (
    Dataset,
    DatasetDict
)
from typing import List, Any


_split = [
    "train",
    "validation",
    "test"
]


class HF_Dataset:
    def __init__(
            self,
            start: List[Any],
            target: List[Any],
            feat_static_cat: List[Any],
            feat_dynamic_real: List[Any],
            item_id: List[Any]) -> None:
        self.start = start
        self.target = target
        self.feat_static_cat = feat_static_cat
        self.feat_dynamic_real = feat_dynamic_real
        self.item_id = item_id

    def getDataset(self, split: str) -> Dataset:
        return Dataset.from_dict(
                {
                    'start': self.start,
                    'target': self.target,
                    'feat_static_cat': self.feat_static_cat,
                    'feat_dynamic_real': self.feat_dynamic_real,
                    'item_id': self.item_id
                }
            )

    def getDatasetDict(self) -> DatasetDict():
        DatasetDict({_split[0]: self.getDataset(_split[0])})
