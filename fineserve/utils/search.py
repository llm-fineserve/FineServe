import bisect

import numpy as np


def get_by_linear(x_data, y_data, x_val):
    """
    search get y_val by linear interpolation
    :param x_data: sorted, increase order
    :param y_data: sorted, increase order
    :param x_val: value to find
    :return:
    """
    if not x_data or not y_data:
        return None
    if len(x_data) != len(y_data):
        return None
    if x_val < x_data[0] or x_val > x_data[-1]:
        return None

    idx = bisect.bisect_left(x_data, x_val)
    if idx == 0:
        return y_data[0]

    y_val = np.interp(x_val, x_data[idx - 1:idx + 1], y_data[idx - 1:idx + 1])
    return y_val
