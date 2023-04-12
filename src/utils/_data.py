from datetime import datetime
from functools import lru_cache
from pandas import Period


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
