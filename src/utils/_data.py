from datetime import datetime
from functools import lru_cache
from numpy import ndarray, array
from pandas import Period
from typing import List


@lru_cache(10_000)
def convert_to_pandas_period(
        date: datetime,
        freq: str) -> Period:
    return Period(date, freq)


def transform_start_field(
        batch,
        freq: str):
    batch["start"] = [convert_to_pandas_period(date, freq)
                      for date in batch["start"]]
    return batch


def get_split_limit(
        raw_serie: List[float],
        split_frac: ndarray):
    assert (
        len(split_frac) == 3 &
        split_frac.sum() == 1.0 &
        split_frac[0] > split_frac[1] &
        split_frac[1] > split_frac[2]
    )
    _split_frac = array([
                    split_frac[0],
                    split_frac[0]+split_frac[1],
                    1
                ])
    return (len(raw_serie)*_split_frac).astype(int)
