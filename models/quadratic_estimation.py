"""
Network model for interpolating & extrapolating data points from a quadratic equation.

"""
from random import random

from data import Data


def generate_data_quadratic_equation(size: int, min: float, max: float) -> Data:
    """

    Generates sample data fulfilling f(x) = 3x^2 - 4x + 2 where x is the activation of a single input node
    and f(x) is the expected output.

    :param size:
    :param min: minimum value of x
    :param max: maximum value of x
    :return:
    """

    data = []

    for i in range(0, size):
        x = random * (max - min) + min
        data.append(([x], [3 * x ** 2 - 4 * x + 2]))

    return data