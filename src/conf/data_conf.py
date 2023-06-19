from dataclasses import dataclass
from datetime import datetime
from typing import List, Any


@dataclass
class HF_Dataset_Params:
    """a dataclass to store the params of the
    following class
    """
    start: List[datetime]
    time_series: List[List[float]]
    feat_static_cat: List[List[Any]]
    feat_dynamic_real: List[List[Any]]
    freq: str
    prediction_length: int
