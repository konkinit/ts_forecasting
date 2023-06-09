import os
import sys
from gluonts.time_feature import (
    time_features_from_frequency_str
)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import TS_Transformer_Params


class TS_Transformer:
    """Build a time series transtformer from scratch
    """
    def __init__(
            self,
            params: TS_Transformer_Params
    ) -> None:
        self._transformers = params.list_transformers
        self.prediction_length = params.prediction_length
        self.num_of_variates = params.num_of_variates
        self.time_features = time_features_from_frequency_str(
            params.freq
        )
        self.dropout = params.dropout
        self.encoder_layers = params.encoder_layers
        self.decoder_layers = params.decoder_layers
        self.d_model = params.d_model

    def model_config(self):
        return self._transformers[0](
            input_size=self.num_of_variates,
            prediction_length=self.prediction_length,
            context_length=self.prediction_length * 2,
            lags_sequence=[1, 24 * 7],
            num_time_features=len(self.time_features) + 1,
            dropout=self.dropout,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            d_model=self.d_model
        )

    def model(self):
        return self.transformers_[1](self.model_config())
