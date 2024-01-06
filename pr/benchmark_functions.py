import numpy as np


def calculate_step_2_function_value(
    parameters,
):
    value = 0.00
    for parameter in parameters:
        try:
            value += round(value + ((parameter + 0.5) ** 2), 2)
        except Exception as e:
            print(parameter)
            print(e)
    return value


def sphere_function(parameters):
    return np.sum(np.array(list(parameters)) ** 2)


def rosenbrocks_banana_function(parameters):
    parameters = np.array(list(parameters))
    return np.sum(
        100 * (parameters[1:] - parameters[:-1] ** 2) ** 2 + (1 - parameters[:-1]) ** 2
    )


def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def griewank_function(x):
    term1 = np.sum(x**2) / 4000

    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    return term1 - term2 + 1


def ackley_function(x):
    n = len(x)

    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))

    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x) / n))

    return term1 + term2 + 20 + np.exp(1)


def schwefels_problem(x):
    return np.max(np.abs(x))


def michalewicz_function(x):
    m = 10

    return -np.sum(
        np.sin(x) * np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi) ** (2 * m)
    )


def styblinski_tang_function(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


def zakharov_function(x):
    term1 = np.sum(x**2)

    term2 = (0.5 * np.sum(np.arange(1, len(x) + 1) * x)) ** 2

    term3 = (0.5 * np.sum(np.arange(1, len(x) + 1) * x)) ** 4

    return term1 + term2 + term3


def levy_function(x):
    w = 1 + (x - 1) / 4

    term1 = (np.sin(np.pi * w[0])) ** 2

    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1)) ** 2))

    term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)

    return term1 + term2 + term3
