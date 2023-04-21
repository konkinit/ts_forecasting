from datasets.dataset_dict import DatasetDict
from datetime import datetime
from functools import (
    lru_cache,
    partial
)
from gluonts.dataset.multivariate_grouper import (
    MultivariateGrouper
)
from numpy import ndarray, array
from pandas import Period
from typing import List


@lru_cache(10_000)
def convert_to_pandas_period(
        date: datetime,
        freq: str) -> Period:
    return Period(date, freq)


def transform_start_field(
        batch,
        freq: str):
    batch["start"] = [convert_to_pandas_period(date, freq)
                      for date in batch["start"]]
    return batch


def get_split_limit(
        raw_serie: List[float],
        split_frac: ndarray):
    assert (
        (split_frac.shape[0] == 3) &
        (split_frac.sum().round(10) == 1.0) &
        (split_frac[0] > split_frac[1]) &
        (split_frac[1] > split_frac[2])
    )
    _split_frac = array([
                    split_frac[0],
                    split_frac[0]+split_frac[1],
                    1
                ])
    return (len(raw_serie)*_split_frac).astype(int)


class DataProcessing:
    def __init__(
            self,
            dataset: DatasetDict):
        self.dataset = dataset.copy()
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]

    def dates_transforming(
                self,
                freq: str) -> None:
        self.train_dataset.set_transform(
            partial(transform_start_field, freq=freq))
        self.test_dataset.set_transform(
            partial(transform_start_field, freq=freq))

    def multi_variate_format(self, freq: str):
        self.dates_transforming(freq)
        num_of_variates = len(self.train_dataset)
        train_grouper = MultivariateGrouper(
                            max_target_dim=num_of_variates
                        )
        test_grouper = MultivariateGrouper(
                            max_target_dim=num_of_variates
                        )
        return (
            train_grouper(self.train_dataset)[0],
            test_grouper(self.test_dataset)[0]
        )
