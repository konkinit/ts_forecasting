import os
import sys
from gluonts.time_feature import (
    time_features_from_frequency_str,
    get_lags_for_frequency
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
        # data params :
        self.prediction_length = params.prediction_length
        self.num_of_variates = params.num_of_variates
        self.freq = params.freq
        # transformer params :
        self._transformers = params.list_transformers
        self.dropout = params.dropout
        self.encoder_layers = params.encoder_layers
        self.decoder_layers = params.decoder_layers
        self.d_model = params.d_model

    def _config(self):
        self.config = self._transformers[0](
            prediction_length=self.prediction_length,
            context_length=self.prediction_length * 2,
            input_size=self.num_of_variates,
            lags_sequence=get_lags_for_frequency(self.freq),
            num_time_features=len(
                time_features_from_frequency_str(
                    self.freq
                )
            ) + 1,
            num_dynamic_real_features=0,
            num_static_categorical_features=1,
            num_static_real_features=0,
            cardinality=[self.num_of_variates],
            embedding_dimension=[2],
            dropout=self.dropout,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            d_model=self.d_model
        )

    def _model(self):
        self._config()
        self.model = self._transformers[1](self.config)
