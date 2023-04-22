from dataclasses import dataclass
from gluonts.time_feature import (
    time_features_from_frequency_str
)
from typing import (
    List,
    Any
)


@dataclass(slot=True)
class Time_Series_Transformer_Params:
    transformers_: List[Any]
    num_of_variates: int
    freq: str
    prediction_length: int
    dropout: float
    encoder_layers: int
    decoder_layers: int
    d_model: int


class Time_Series_Transformer:
    """Build a time series transtformer from scratch
    """
    def __init__(
            self,
            params: Time_Series_Transformer_Params) -> None:
        self.transformers_ = params.transformers_
        self.num_of_variates = params.num_of_variates
        self.time_features = time_features_from_frequency_str(
                                params.freq
                            )
        self.prediction_length = params.prediction_length
        self.dropout = params.dropout
        self.encoder_layers = params.encoder_layers
        self.decoder_layers = params.decoder_layers
        self.d_model = params.d_model

    def model_config(self):
        return self.transformer_[0](
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
