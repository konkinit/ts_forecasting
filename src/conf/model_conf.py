from dataclasses import dataclass
from typing import List, Any


@dataclass
class TS_Transformer_Params:
    list_transformers: List[Any]
    num_of_variates: int
    freq: str
    prediction_length: int
    dropout: float
    encoder_layers: int
    decoder_layers: int
    d_model: int
