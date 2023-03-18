import datasets
from functools import partial
from .utils import transform_start_field


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
