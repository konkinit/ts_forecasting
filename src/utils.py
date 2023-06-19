import torch
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
from gluonts.time_feature import (
    time_features_from_frequency_str
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    # SelectFields,
    # SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.itertools import Cached, Cyclic
from gluonts.transform.sampler import InstanceSampler
from numpy import ndarray, array
from pandas import Period
from transformers import PretrainedConfig
from typing import List, Tuple, Optional, Iterable


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
        split_frac: ndarray
) -> ndarray:
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
    _split_frac = array(
        [
            split_frac[0], split_frac[0]+split_frac[1], 1
        ]
    )
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


def create_transformation(
        freq: str,
        config: PretrainedConfig
) -> Transformation:
    """Create a transformation pipeline

    Args:
        freq (str): the frequency of the time series
        config (PretrainedConfig): the transformer config instance

    Returns:
        Transformation: the transformation stack
    """
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved
            # values) see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series
            # is, sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key
            # FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    """Sample windows from the dataset since the
    entire history of values  cannot be passed to the Transformer due
    to time- and memory constraints.

    Args:
        config (PretrainedConfig): the transformer config instance
        mode (str): splitting mode
        train_sampler (Optional[InstanceSampler], optional):
        _description_. Defaults to None.
        validation_sampler (Optional[InstanceSampler], optional):
        _description_. Defaults to None.

    Returns:
        Transformation: a transformation built with InstzanceSplitter
    """
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length+max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
    config: PretrainedConfig,
    freq: str,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366
    # possible transformed time series) randomly from within the
    # target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_test_dataloader(
    config: PretrainedConfig,
    freq: str,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very
    # last context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(
        transformed_data, is_train=False
    )
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
