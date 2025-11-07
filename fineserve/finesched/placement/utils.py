import bisect

import numpy as np


def linear_interp(x_val, x_data, y_data):
    """
    Perform linear interpolation to find y_val for a given x_val.
    
    Args:
        x_val: Value to find in x_data
        x_data: Sorted list of x values (increasing order)
        y_data: Corresponding y values
        
    Returns:
        Interpolated y value or None if inputs are invalid
    """
    # Validate inputs
    if len(x_data) == 0 or len(y_data) == 0:
        return None
    if len(x_data) != len(y_data):
        return None
        
    # Handle single point case
    if len(x_data) == 1:
        return y_data[0] / x_data[0] * x_val

    # Find appropriate index
    last_idx = len(x_data) - 1
    if x_val <= x_data[0]:
        idx = 1
    elif x_val >= x_data[last_idx]:
        idx = last_idx
    else:
        idx = bisect.bisect_left(x_data, x_val)
        
    if idx == 0:
        return y_data[0]

    # Perform linear interpolation using polynomial fitting
    p = np.polyfit(x_data[idx - 1:idx + 1], y_data[idx - 1:idx + 1], 1)
    y_val = np.polyval(p, x_val)

    return y_val