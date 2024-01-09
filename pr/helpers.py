import random


def generate_random_in_range(min_param=-100, max_param=100):
    random_number = random.uniform(min_param, max_param)
    rounded_number = round(random_number, 2)
    return rounded_number
