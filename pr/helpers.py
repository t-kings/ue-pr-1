import random

min_range = -100
max_range = 100


def generate_random_in_range(min_param=min_range, max_param=max_range):
    random_number = random.uniform(min_param, max_param)
    rounded_number = round(random_number, 2)
    return rounded_number
