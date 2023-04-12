import os
import sys
from datasets.dataset_dict import DatasetDict
from functools import partial
from gluonts.dataset.multivariate_grouper import (
    MultivariateGrouper
)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    transform_start_field
)


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
                            max_target_dim=num_of_variates,
                            num_test_dates=(len(self.test_dataset) //
                                            num_of_variates),
                        )
        return (
            train_grouper(self.train_dataset),
            test_grouper(self.test_dataset)
        )
