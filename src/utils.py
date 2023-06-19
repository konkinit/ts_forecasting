from datasets.dataset_dict import (
    DatasetDict,
    Dataset
)
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
from typing import List, Tuple


@lru_cache(10_000)
def convert_to_pandas_period(
        date: datetime,
        freq: str) -> Period:
    """Convert a date from datetime format to pandas
    Period format

    Args:
        date (datetime): date to convert
        freq (str): frequency of time serie

    Returns:
        Period: the date converted in Period format
    """
    return Period(date, freq)


def transform_start_field(
        batch: Dataset,
        freq: str) -> Dataset:
    """Transform the start field of dataset to pd.Period
    formt

    Args:
        batch (Dataset): Hugging Face dataset
        freq (str): time serie frequency

    Returns:
        Dataset: dataset with the start field transformed to
        pd.Period format
    """
    batch["start"] = [convert_to_pandas_period(date, freq)
                      for date in batch["start"]]
    return batch


def get_split_limit(
        raw_serie: List[float],
        split_frac: ndarray) -> ndarray:
    """Transformat a list of splitting fraction to
    a list of indexes

    Args:
        raw_serie (List[float]): time serie
        split_frac (ndarray): array of splitting fraction

    Returns:
        ndarray: array of splitting indexes
    """
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
    """A class to process Dict of datasets
    """
    def __init__(
            self,
            dataset: DatasetDict
    ) -> None:
        self.dataset = dataset.copy()
        self.train_dataset = self.dataset["train"]
        self.validation_dataset = self.dataset["validation"]
        self.test_dataset = self.dataset["test"]

    def dates_transforming(
            self,
            freq: str
    ) -> None:
        """Update start to pd.Period in each partition dataset

        Args:
            freq (str): time serie frequency
        """
        self.train_dataset.set_transform(
            partial(transform_start_field, freq=freq))
        self.validation_dataset.set_transform(
            partial(transform_start_field, freq=freq))
        self.test_dataset.set_transform(
            partial(transform_start_field, freq=freq))

    def get_num_of_variates(self) -> int:
        """ Get the number of time series in the dataset

        Returns:
            int: number of time series
        """
        return len(self.train_dataset)

    def formated_dataset(self, freq: str) -> DatasetDict:
        self.dates_transforming(freq)
        return DatasetDict(
            {
                "train": self.train_dataset,
                "validation": self.validation_dataset,
                "test": self.test_dataset
            }
        )

    def multi_variate_format(
            self,
            freq: str
    ) -> Tuple[Dataset]:
        """Since a dataset can be defined with many time series,
        this method converts dataset to a multivariate time
        serie

        Args:
            freq (str): time serie frequency

        Returns:
            Tuple[Dataset]: tuple of multivariate time series
            in HF dataset format
        """
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
            train_grouper(self.train_dataset)[0],
            test_grouper(self.test_dataset)[0]
        )
