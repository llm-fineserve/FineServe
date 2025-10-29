import bisect

import numpy as np


def linear_interp(x_val, x_data, y_data):
    """
    search get y_val by linear interpolation
    :param x_val: value to find
    :param x_data: sorted, increase order
    :param y_data: y_data
    :return:
    """
    if len(x_data) == 0 or len(y_data) == 0:
        return None
    if len(x_data) != len(y_data):
        return None
    if len(x_data) == 1:
        return y_data[0]/x_data[0] * x_val

    last_idx = len(x_data) - 1
    if x_val <= x_data[0]:
        idx = 1
    elif x_val >= x_data[last_idx]:
        idx = last_idx
    else:
        idx = bisect.bisect_left(x_data, x_val)
    if idx == 0:
        return y_data[0]

    # y_val = np.interp(x_val, x_data[idx - 1:idx + 1], y_data[idx - 1:idx + 1])

    p = np.polyfit(x_data[idx - 1:idx + 1], y_data[idx - 1:idx + 1], 1)
    y_val = np.polyval(p, x_val)

    return y_val
