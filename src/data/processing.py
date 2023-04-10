import os
import sys
import datasets
from functools import partial
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    transform_start_field
)


def get_train_test(dataset):
    return dataset["train"], dataset["test"]


def data_processing(
            train_dataset: datasets.arrow_dataset.Dataset,
            test_dataset: datasets.arrow_dataset.Dataset,
            freq: str) -> None:
    train_dataset.set_transform(
        partial(transform_start_field, freq=freq))
    test_dataset.set_transform(
        partial(transform_start_field, freq=freq))
