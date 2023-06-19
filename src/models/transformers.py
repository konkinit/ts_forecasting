import os
import sys
from datasets import Dataset
from gluonts.time_feature import (
    time_features_from_frequency_str,
    get_lags_for_frequency
)
from typing import Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import TS_Transformer_Params
from src.utils import (
    create_train_dataloader,
    create_test_dataloader
)


class TS_Transformer:
    """Build a time series transtformer from scratch
    """
    def __init__(
            self,
            params: TS_Transformer_Params
    ) -> None:
        # data params :
        self.prediction_length = params.prediction_length
        self.freq = params.freq
        self.dict_features_size = params.dict_features_size
        # transformer params :
        self._transformers = params.list_transformers
        self.dropout = params.dropout
        self.encoder_layers = params.encoder_layers
        assert len(params.embedding_dimension) == params.dict_features_size[
            "num_static_categorical_features"
        ]
        self.embedding_dimension = params.embedding_dimension
        self.decoder_layers = params.decoder_layers
        self.d_model = params.d_model

    def _config(self):
        self.config = self._transformers[0](
            prediction_length=self.prediction_length,
            context_length=2*self.prediction_length,
            lags_sequence=get_lags_for_frequency(self.freq),
            num_time_features=len(
                time_features_from_frequency_str(
                    self.freq
                )
            )+1,
            **self.dict_features_size,
            cardinality=[
                self.dict_features_size["input_size"]
            ],
            embedding_dimension=self.embedding_dimension,
            dropout=self.dropout,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            d_model=self.d_model,
        )

    def _model(self):
        self._config()
        self.model = self._transformers[1](self.config)

    def _transformation(
            self, train_dataset: Dataset, test_dataset: Dataset
    ) -> Tuple:
        train_dataloader = create_train_dataloader(
            config=self.config,
            freq=self.freq,
            data=train_dataset,
            batch_size=256,
            num_batches_per_epoch=100,
        )

        test_dataloader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=test_dataset,
            batch_size=64,
        )
        return (train_dataloader, test_dataloader)
