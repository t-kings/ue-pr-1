from enum import Enum
from benchmark_functions import (
    sphere_function,
    calculate_step_2_function_value,
    quartic_function,
    schwefel_2_21_function,
    schwefel_2_22_function,
    six_hump_camel_back,
    rastrigin,
    griewank_function,
    branin_function,
    ackley_function,
)
import statistics
import math
from helpers import generate_random_in_range


class AlgorithmsEnum(Enum):
    PSO = "pso"
    MPSO = "mpso"


class BenchmarkFunctionEnum(Enum):
    SPHERE_FUNCTION = "sphere_function"
    CALCULATE_STEP_2_FUNCTION_VALUE = "calculate_step_2_function_value"
    QUARTIC_FUNCTION = "quartic_function"
    SCHWEFEL_2_21_FUNCTION = "schwefel_2_21_function"
    SCHWEFEL_2_22_FUNCTION = "schwefel_2_22_function"
    SIX_HUMP_CAMEL_BACK = "six_hump_camel_back"
    RASTRIGIN = "rastrigin"
    GRIEWANK_FUNCTION = "griewank_function"
    BRANIN_FUNCTION = "branin_function"
    ACKLEY_FUNCTION = "ackley_function"


def calculate_particle_p_best(iteration_particles, previous_iteration_particles):
    for particle, particle_data in iteration_particles.items():
        function_value = particle_data["function_value"]
        previous_particle = previous_iteration_particles[particle]
        previous_p_best = previous_particle.get("p_best", {})
        if function_value < previous_p_best.get("value", float("inf")):
            particle_data["p_best"] = {
                "value": function_value,
                "position": particle_data["position"],
            }
        else:
            particle_data["p_best"] = previous_p_best

    return iteration_particles


def calculate_iteration_g_best(iteration_particles, previous_iteration_g_best):
    min_particle_p_best_value = float("inf")
    min_particle_p_best_position = None
    for particle_data in iteration_particles.values():
        particle_p_best_value = particle_data["p_best"]["value"]
        if particle_p_best_value < min_particle_p_best_value:
            min_particle_p_best_value = particle_p_best_value
            min_particle_p_best_position = particle_data["position"]

    previous_iteration_g_best_value = previous_iteration_g_best.get(
        "value", float("inf")
    )
    if min_particle_p_best_value < previous_iteration_g_best_value:
        return {
            "value": min_particle_p_best_value,
            "position": min_particle_p_best_position,
        }
    return previous_iteration_g_best


def calculate_particle_value_with_algorithm(particles, benchmark_function):
    for particle_data in particles.values():
        res = 0
        if benchmark_function == BenchmarkFunctionEnum.SPHERE_FUNCTION:
            res = sphere_function(particle_data["position"].values())
        elif (
            benchmark_function == BenchmarkFunctionEnum.CALCULATE_STEP_2_FUNCTION_VALUE
        ):
            res = calculate_step_2_function_value(particle_data["position"].values())
        elif benchmark_function == BenchmarkFunctionEnum.QUARTIC_FUNCTION:
            res = quartic_function(particle_data["position"].values())
        elif benchmark_function == BenchmarkFunctionEnum.SCHWEFEL_2_21_FUNCTION:
            res = schwefel_2_21_function(particle_data["position"].values())
        elif benchmark_function == BenchmarkFunctionEnum.SCHWEFEL_2_22_FUNCTION:
            res = schwefel_2_22_function(particle_data["position"].values())
        elif benchmark_function == BenchmarkFunctionEnum.SIX_HUMP_CAMEL_BACK:
            res = six_hump_camel_back(
                particle_data["position"][1], particle_data["position"][2]
            )
        elif benchmark_function == BenchmarkFunctionEnum.RASTRIGIN:
            res = rastrigin(particle_data["position"][1], particle_data["position"][2])
        elif benchmark_function == BenchmarkFunctionEnum.GRIEWANK_FUNCTION:
            res = griewank_function(particle_data["position"].values())
        elif benchmark_function == BenchmarkFunctionEnum.BRANIN_FUNCTION:
            res = branin_function(
                particle_data["position"][1], particle_data["position"][2]
            )
        elif benchmark_function == BenchmarkFunctionEnum.ACKLEY_FUNCTION:
            res = ackley_function(particle_data["position"].values())

        particle_data["function_value"] = round(res, 2)
    return particles


def get_range_per_benchmark_function(benchmark_function):
    ranges = {
        BenchmarkFunctionEnum.SPHERE_FUNCTION: [-100, 100],
        BenchmarkFunctionEnum.CALCULATE_STEP_2_FUNCTION_VALUE: [-10, 10],
        BenchmarkFunctionEnum.QUARTIC_FUNCTION: [-2.56, 2.56],
        BenchmarkFunctionEnum.SCHWEFEL_2_21_FUNCTION: [-100, 100],
        BenchmarkFunctionEnum.SCHWEFEL_2_22_FUNCTION: [-10, 10],
        BenchmarkFunctionEnum.SIX_HUMP_CAMEL_BACK: [-5, 5],
        BenchmarkFunctionEnum.RASTRIGIN: [-50, 50],
        BenchmarkFunctionEnum.GRIEWANK_FUNCTION: [-600, 600],
        BenchmarkFunctionEnum.BRANIN_FUNCTION: [-5, 5],
        BenchmarkFunctionEnum.ACKLEY_FUNCTION: [-32, 32],
    }
    return ranges[benchmark_function]


def generate_initial_particles_with_positions_and_velocities(
    number_of_particles, number_of_coordinates, benchmark_function
):
    particles = {}
    for i in range(number_of_particles):
        position = {}
        velocity = {}
        min_range, max_range = get_range_per_benchmark_function(benchmark_function)
        for coordinate in range(number_of_coordinates):
            random_position = generate_random_in_range(min_range, max_range)
            position[coordinate + 1] = random_position

            random_velocity = generate_random_in_range(min_range, max_range)
            velocity[coordinate + 1] = random_velocity

        particle = {"position": position, "velocity": velocity}
        particles[i + 1] = particle
    return particles


def generate_velocity(
    algorithm, iteration_count, previous_velocity, personal_best, position, global_best
):
    res = 0
    if algorithm == AlgorithmsEnum.MPSO:
        inertia = 0.05
        r1 = generate_random_in_range(0, 1)
        r2 = generate_random_in_range(0, 1)
        c1 = 0.5
        c2 = 2.5
        sum_of_c1_and_c2_average = (c1 + c2) / 2
        change_rate = sum_of_c1_and_c2_average / (
            iteration_count + sum_of_c1_and_c2_average
        )
        c1 = c1 + change_rate
        c2 = c2 - change_rate
        res = round(
            (inertia / iteration_count) * previous_velocity
            + c1 * r1 * (personal_best - position)
            + c2 * r2 * (global_best - position),
            2,
        )
    elif algorithm == AlgorithmsEnum.PSO:
        inertia = 0.05
        particle_influence = 0.5
        global_influence = 2.5
        r1 = generate_random_in_range(0, 1)
        r2 = generate_random_in_range(0, 1)
        res = round(
            (inertia) * previous_velocity
            + particle_influence * r1 * (personal_best - position)
            + global_influence * r2 * (global_best - position),
            2,
        )
    return res


def generate_position(algorithm, velocity, previous_position):
    res = 0
    if algorithm in [AlgorithmsEnum.MPSO, AlgorithmsEnum.PSO]:
        res = velocity + previous_position
    return round(res, 2)


def generate_particles_with_positions_and_velocities_and_algorithm(
    previous_iteration_particles, algorithm, iteration_count, previous_iteration_g_best
):
    particles = {}
    for (
        previous_particle,
        previous_particle_data,
    ) in previous_iteration_particles.items():
        position = {}
        velocity = {}
        for previous_velocity, previous_velocity_data in previous_particle_data[
            "velocity"
        ].items():
            velocity[previous_velocity] = generate_velocity(
                algorithm,
                iteration_count,
                previous_velocity_data,
                previous_particle_data["p_best"]["position"][previous_velocity],
                previous_particle_data["position"][previous_velocity],
                previous_iteration_g_best["position"][previous_velocity],
            )
            position[previous_velocity] = generate_position(
                algorithm,
                velocity[previous_velocity],
                previous_particle_data["position"][previous_velocity],
            )
        particle = {"position": position, "velocity": velocity}
        particles[previous_particle] = particle
    return particles


def get_iteration_value(
    iteration_count, previous_iteration, algorithm, benchmark_function
):
    iteration = {}
    if iteration_count != 1:
        iteration[
            "particles"
        ] = generate_particles_with_positions_and_velocities_and_algorithm(
            previous_iteration["particles"],
            algorithm,
            iteration_count,
            previous_iteration["g_best"],
        )
    else:
        iteration = previous_iteration

    particles = calculate_particle_value_with_algorithm(
        iteration["particles"], benchmark_function
    )

    iteration["particles"] = calculate_particle_p_best(
        particles, previous_iteration["particles"]
    )

    iteration["g_best"] = calculate_iteration_g_best(
        iteration["particles"],
        previous_iteration.get("g_best", {}),
    )
    return iteration


def get_is_stopping_criteria_reached(benchmark_function, g_best):
    stopping_criteria = {
        BenchmarkFunctionEnum.SPHERE_FUNCTION: g_best == 0,
        BenchmarkFunctionEnum.CALCULATE_STEP_2_FUNCTION_VALUE: g_best == 0,
        BenchmarkFunctionEnum.QUARTIC_FUNCTION: g_best == 0,
        BenchmarkFunctionEnum.SCHWEFEL_2_21_FUNCTION: g_best == 0,
        BenchmarkFunctionEnum.SCHWEFEL_2_22_FUNCTION: g_best == 0,
        BenchmarkFunctionEnum.SIX_HUMP_CAMEL_BACK: (0.0898 >= g_best >= -0.7126)
        or (-0.0898 <= g_best <= 0.7126),
        BenchmarkFunctionEnum.RASTRIGIN: g_best == 0,
        BenchmarkFunctionEnum.GRIEWANK_FUNCTION: g_best == 0,
        BenchmarkFunctionEnum.BRANIN_FUNCTION: g_best == 0.398,
        BenchmarkFunctionEnum.ACKLEY_FUNCTION: g_best == 0,
    }
    return stopping_criteria.get(benchmark_function, False)


def get_algorithm_values(algorithm, benchmark_function, particles):
    is_stopping_criteria_reached = False
    iterations = {}
    iteration_count = 1
    while not is_stopping_criteria_reached:
        previous_iteration = iterations.get(
            iteration_count - 1, {"particles": particles}
        )
        iteration = get_iteration_value(
            iteration_count, previous_iteration, algorithm, benchmark_function
        )

        iterations[iteration_count] = iteration
        if (
            get_is_stopping_criteria_reached(
                benchmark_function, iteration["g_best"]["value"]
            )
            or iteration_count == 200
        ):
            is_stopping_criteria_reached = True
        iteration_count += 1
    iteration_g_bests = list(
        map(lambda obj: obj["g_best"]["value"], iterations.values())
    )
    iteration_g_bests_standard_deviation = round(
        statistics.pstdev(iteration_g_bests), 2
    )
    iteration_g_bests_mean = round(statistics.mean(iteration_g_bests), 2)
    return {
        "iteration_count": len(iterations),
        "standard_deviation": iteration_g_bests_standard_deviation,
        "mean": iteration_g_bests_mean,
    }


def calculate_algorithms_with_benchmark_functions(benchmark_function):
    particles = generate_initial_particles_with_positions_and_velocities(
        6, 2, benchmark_function
    )
    mpso_values = get_algorithm_values(
        AlgorithmsEnum.MPSO, benchmark_function, particles
    )
    pso_values = get_algorithm_values(AlgorithmsEnum.PSO, benchmark_function, particles)
    t_test = round(
        (mpso_values["mean"] - pso_values["mean"])
        / math.sqrt(
            ((mpso_values["standard_deviation"] ** 2) / mpso_values["iteration_count"])
            + ((pso_values["standard_deviation"] ** 2) / pso_values["iteration_count"]),
        ),
        2,
    )
    better_algorithm = "mpso" if t_test >= 0 else "pso"
    return {
        "benchmarkFunction": benchmark_function,
        "mpso_values": mpso_values,
        "pso_values": pso_values,
        "t_test": t_test,
        "better_algorithm": better_algorithm,
    }


def custom_particle_swarm_optimization_comparison():
    benchmark_functions = [
        BenchmarkFunctionEnum.SPHERE_FUNCTION,
        BenchmarkFunctionEnum.CALCULATE_STEP_2_FUNCTION_VALUE,
        BenchmarkFunctionEnum.QUARTIC_FUNCTION,
        BenchmarkFunctionEnum.SCHWEFEL_2_21_FUNCTION,
        BenchmarkFunctionEnum.SCHWEFEL_2_22_FUNCTION,
        BenchmarkFunctionEnum.SIX_HUMP_CAMEL_BACK,
        BenchmarkFunctionEnum.RASTRIGIN,
        BenchmarkFunctionEnum.GRIEWANK_FUNCTION,
        BenchmarkFunctionEnum.BRANIN_FUNCTION,
        BenchmarkFunctionEnum.ACKLEY_FUNCTION,
    ]
    benchmark_functions_iterations = {}
    for benchmark_function in benchmark_functions:
        benchmark_functions_iterations[
            benchmark_function
        ] = calculate_algorithms_with_benchmark_functions(benchmark_function)
    return benchmark_functions_iterations


comparison = custom_particle_swarm_optimization_comparison()
