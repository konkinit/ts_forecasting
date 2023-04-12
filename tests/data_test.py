import os
import sys
import pytest
from pandas import Period
from datetime import datetime
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    convert_to_pandas_period
)


@pytest.mark.parametrize(
    "date, freq, pandas_period",
    [(datetime(1979, 1, 1, 0, 0), "1M", Period('1979-01', 'M'))]
)
def test_convert_to_pandas_period(
        date,
        freq,
        pandas_period):
    assert convert_to_pandas_period(
                date, freq) == pandas_period
