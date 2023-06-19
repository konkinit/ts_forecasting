from dataclasses import dataclass
from typing import List, Any, Dict


@dataclass
class TS_Transformer_Params:
    prediction_length: int
    freq: str
    dict_features_size: Dict
    list_transformers: List[Any]
    encoder_layers: int
    embedding_dimension: List[int]
    decoder_layers: int
    d_model: int
    dropout: float
