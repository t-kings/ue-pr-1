import numpy as np
from helpers import generate_random_in_range


def sphere_function(parameters):
    parameters_array = np.array(list(parameters))
    return np.sum(np.square(parameters_array))


def calculate_step_2_function_value(parameters):
    value = 0.00
    for parameter in parameters:
        try:
            value += round(value + ((parameter + 0.5) ** 2), 2)
        except Exception as e:
            print(parameter)
            print(e)
    return value


def quartic_function(parameters):
    parameters_array = np.array(list(parameters))
    sum_part = np.sum(
        np.arange(1, len(parameters_array) + 1) * np.power(parameters_array, 4)
        + generate_random_in_range(0, 1)
    )
    return sum_part


def schwefel_2_21_function(parameters):
    parameters_array = np.array(list(parameters))
    max_value = np.max(np.abs(parameters_array))
    return max_value


def schwefel_2_22_function(parameters):
    parameters = np.array(list(parameters))
    sum_part = np.sum(np.abs(parameters))
    multiply_part = np.prod(np.abs(parameters))
    return sum_part + multiply_part


def six_hump_camel_back(x1, x2):
    return 4 * x1**2 - 2.1 * x1**4 + (1 / 3) * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4


def rastrigin(x1, x2):
    return (
        20 + x1**2 - 10 * np.cos(2 * np.pi * x1) + x2**2 - 10 * np.cos(2 * np.pi * x2)
    )


def griewank_function(parameters):
    parameters = np.array(list(parameters))
    term1 = np.sum(np.square(parameters)) / 4000
    term2 = np.prod(np.cos(parameters / np.sqrt(np.arange(1, len(parameters) + 1))))
    return term1 - term2 + 1


def branin_function(x, y):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    term1 = a * (y - b * x**2 + c * x - r) ** 2
    term2 = s * (1 - t) * np.cos(x)
    term3 = s

    return term1 + term2 + term3


def ackley_function(parameters):
    x = np.array(list(parameters))
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(np.square(x)) / len(x)))
    term2 = -np.exp(np.sum(np.cos(c * x) / len(x)))
    result = term1 + term2 + a + np.exp(1)
    return result
