import math
from typing import Union, Optional

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def bound_fn(
    min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]
    ) -> Union[float, int]:
    """ Returns value if between min_val and max_val. Otherwise clamp.
    Args:
        min_val: Union[float,int] minimum clamp value
        max_val: Union[float,int] maximum clamp value
        value:   Union[float,int] value to return if between min/max_value
    
    Returns:
        value if min<value<max. if value < min, returns min. if value > max, returns max.
    """
    return max(min_val, min(max_val, value))