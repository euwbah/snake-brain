from decimal import Decimal
from typing import Union


def coerce_decimal(x: Union[float, Decimal]) -> Decimal:
    if type(x) is float:
        x = Decimal(x)

    return x
