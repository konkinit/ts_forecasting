import os
import sys
from datasets import Dataset, DatasetDict
from typing import Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import HF_Dataset_Params
from src.utils import DataProcessing


_split = ["train", "validation", "test"]


class HF_Dataset:
    """Transform a time serie in list type to a
    Hugging Face dataset format
    """

    def __init__(self, hf_dataset_params: HF_Dataset_Params) -> None:
        assert len(hf_dataset_params.start) == len(
            hf_dataset_params.time_series
        )
        self.start = hf_dataset_params.start
        self.target = hf_dataset_params.time_series
        self.feat_static_cat = hf_dataset_params.feat_static_cat
        self.feat_dynamic_real = hf_dataset_params.feat_dynamic_real
        self.freq = hf_dataset_params.freq
        self.prediction_length = hf_dataset_params.prediction_length

    def getDataset(self, split_index: int) -> Dataset:
        """Create a dataset for a given partition

        Args:
            split_index (int): index of the partition
            according to the list
            ["train", "validation", "test"]

        Returns:
            Dataset: HF dataset with 5 fields
        """
        assert split_index <= 2
        n_ts = len(self.target)
        return Dataset.from_dict(
            {
                "start": self.start,
                "target": [
                    self.target[i][
                        : len(self.target[i])
                        - (2 - split_index) * self.prediction_length
                    ]
                    for i in range(n_ts)
                ],
                "feat_static_cat": [
                    self.feat_static_cat[i] for i in range(n_ts)
                ],
                "feat_dynamic_real": [
                    self.feat_dynamic_real[i] for i in range(n_ts)
                ],
                "item_id": [f"T{i}" for i in range(n_ts)],
            }
        )

    def getDatasetDict(self) -> DatasetDict:
        """Groups the datasets for the 3 partitions

        Returns:
            DatasetDict: dict of train/validation/test datasets
        """
        return DatasetDict(
            {split: self.getDataset(idx) for idx, split in enumerate(_split)}
        )

    def getFormattedDatasetDict(self) -> DatasetDict:
        """Get the datasets dict with formatted date

        Returns:
            DatasetDict: formatted datasets dict
        """
        return DataProcessing(
            self.getDatasetDict()
        ).formated_dataset(self.freq)

    def getFeaturesStats(self) -> int:
        """Get the number of time series in the dataset

        Returns:
            int: number of time series
        """
        num_variates = len(self.start)
        num_dynamic_real_features = (
            len(self.feat_dynamic_real[0])
            if self.feat_dynamic_real[0] is not None
            else 0
        )
        num_static_categorical_features = (
            len(self.feat_static_cat[0])
            if self.feat_static_cat[0] is not None
            else 0
        )
        return {
            "input_size": num_variates,
            "num_dynamic_real_features": num_dynamic_real_features,
            "num_static_categorical_features": num_static_categorical_features
        }

    def getMVDatasets(self) -> Tuple[Dataset]:
        """Converts dataset to a multivariate time
        serie

        Returns:
            Tuple[Dataset]: tuple of multivariate time series
        """
        return DataProcessing(
            self.getDatasetDict()
        ).multi_variate_format(self.freq)


class XGB_Dataset:
    def __init__(self):
        pass
