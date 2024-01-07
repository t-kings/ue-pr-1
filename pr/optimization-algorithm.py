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

from helpers import generate_random_in_range


class AlgorithmsEnum(Enum):
    PSO = "pso"
    MPSO = "mpso"


class BenchmarkFunctionEnum(Enum):
    sphere_function = "sphere_function"
    calculate_step_2_function_value = "calculate_step_2_function_value"
    quartic_function = "quartic_function"
    schwefel_2_21_function = "schwefel_2_21_function"
    schwefel_2_22_function = "schwefel_2_22_function"
    six_hump_camel_back = "six_hump_camel_back"
    rastrigin = "rastrigin"
    griewank_function = "griewank_function"
    branin_function = "branin_function"
    ackley_function = "ackley_function"


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


def calculate_iteration_g_best(iteration_particles, previous_Iteration_g_best):
    min_particle_p_best_value = float("inf")
    min_particle_p_best_position = None
    for particle_data in iteration_particles.values():
        particle_p_best_value = particle_data["p_best"]["value"]
        if particle_p_best_value < min_particle_p_best_value:
            min_particle_p_best_value = particle_p_best_value
            min_particle_p_best_position = particle_data["position"]

    previous_Iteration_g_best_value = previous_Iteration_g_best.get(
        "value", float("inf")
    )
    if min_particle_p_best_value < previous_Iteration_g_best_value:
        return {
            "value": min_particle_p_best_value,
            "position": min_particle_p_best_position,
        }
    return previous_Iteration_g_best


def calculate_particle_value_with_algorithm(
    particles, benchmark_function: BenchmarkFunctionEnum
):
    for particle_data in particles.values():
        if benchmark_function == BenchmarkFunctionEnum.sphere_function:
            particle_data["function_value"] = sphere_function(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.calculate_step_2_function_value:
            particle_data["function_value"] = calculate_step_2_function_value(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.quartic_function:
            particle_data["function_value"] = quartic_function(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.schwefel_2_21_function:
            particle_data["function_value"] = schwefel_2_21_function(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.schwefel_2_22_function:
            particle_data["function_value"] = schwefel_2_22_function(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.six_hump_camel_back:
            particle_data["function_value"] = six_hump_camel_back(
                particle_data["position"].values()[0],
                particle_data["position"].values()[1],
            )
        if benchmark_function == BenchmarkFunctionEnum.rastrigin:
            particle_data["function_value"] = rastrigin(
                particle_data["position"].values()[0],
                particle_data["position"].values()[1],
            )
        if benchmark_function == BenchmarkFunctionEnum.griewank_function:
            particle_data["function_value"] = griewank_function(
                particle_data["position"].values()
            )
        if benchmark_function == BenchmarkFunctionEnum.branin_function:
            particle_data["function_value"] = branin_function(
                particle_data["position"].values()[0],
                particle_data["position"].values()[1],
            )
        if benchmark_function == BenchmarkFunctionEnum.ackley_function:
            particle_data["function_value"] = ackley_function(
                particle_data["position"].values()
            )

    return particles


def get_range_per_benchmark_function(benchmarkFunction: BenchmarkFunctionEnum):
    ranges = {
        BenchmarkFunctionEnum.sphere_function: [-100, 100],
        BenchmarkFunctionEnum.calculate_step_2_function_value: [-10, 10],
        BenchmarkFunctionEnum.quartic_function: [-2.56, 2.56],
        BenchmarkFunctionEnum.schwefel_2_21_function: [-100, 100],
        BenchmarkFunctionEnum.schwefel_2_22_function: [-10, 10],
        BenchmarkFunctionEnum.six_hump_camel_back: [-5, 5],
        BenchmarkFunctionEnum.rastrigin: [-50, 50],
        BenchmarkFunctionEnum.griewank_function: [-600, 600],
        BenchmarkFunctionEnum.branin_function: [-5, 5],
        BenchmarkFunctionEnum.ackley_function: [-32, 32],
    }
    return ranges[benchmarkFunction]


def generate_initial_particles_with_positions_and_velocities(
    number_of_particles: int, number_of_coordinates: int, benchmarkFunction
):
    particles = {}
    for i in range(number_of_particles):
        position = {}
        velocity = {}
        min_range, max_range = get_range_per_benchmark_function(benchmarkFunction)
        for coordinate in range(number_of_coordinates):
            random_position = generate_random_in_range()
            position[coordinate + 1] = random_position

            random_velocity = generate_random_in_range(min_range, max_range)
            velocity[coordinate + 1] = random_velocity

        particle = {"position": position, "velocity": velocity}
        particles[i + 1] = particle
    return particles


def generate_velocity(
    algorithm: AlgorithmsEnum,
    iteration_count,
    previous_velocity,
    personal_best,
    position,
    global_best,
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

    if algorithm == AlgorithmsEnum.PSO:
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


def generate_position(
    algorithm: AlgorithmsEnum,
    velocity,
    previous_position,
):
    res = 0
    if algorithm == AlgorithmsEnum.MPSO:
        res = velocity + previous_position
    if algorithm == AlgorithmsEnum.PSO:
        res = velocity + previous_position
    return round(res, 2)


def generate_particles_with_positions_and_velocities_and_algorithm(
    previous_iteration_particles,
    algorithm: AlgorithmsEnum,
    iteration_count,
    previous_iteration_g_best,
):
    particles = {}
    for (
        previous_Particle,
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
        particles[previous_Particle] = particle
    return particles


def get_iteration_value(
    iteration_count,
    previous_Iteration,
    algorithm: AlgorithmsEnum,
    benchmarkFunction: BenchmarkFunctionEnum,
):
    iteration = {}
    if iteration_count != 1:
        iteration[
            "particles"
        ] = generate_particles_with_positions_and_velocities_and_algorithm(
            previous_Iteration["particles"],
            algorithm,
            iteration_count,
            previous_Iteration["g_best"],
        )
    else:
        iteration = previous_Iteration

    particles = calculate_particle_value_with_algorithm(
        iteration["particles"], benchmarkFunction
    )

    iteration["particles"] = calculate_particle_p_best(
        particles, previous_Iteration["particles"]
    )

    iteration["g_best"] = calculate_iteration_g_best(
        iteration["particles"],
        previous_Iteration.get("g_best", {}),
    )
    return iteration


def get_algorithm_values(
    algorithm: AlgorithmsEnum, benchmarkFunction: BenchmarkFunctionEnum
):
    is_stopping_criteria_reached = False
    iterations = {}
    iteration_count = 1
    particles = generate_initial_particles_with_positions_and_velocities(
        6, 2, benchmarkFunction
    )
    while not is_stopping_criteria_reached:
        previous_iteration = iterations.get(
            iteration_count - 1, {"particles": particles}
        )
        iteration = get_iteration_value(
            iteration_count, previous_iteration, algorithm, benchmarkFunction
        )
        iterations[iteration_count] = iteration
        if iteration["g_best"]["value"] == 0 or iteration_count > 100:
            is_stopping_criteria_reached = True
            print(algorithm, benchmarkFunction, "iteration_count", iteration_count)
        iteration_count += 1
    return iterations


def custom_particle_swarm_optimization_comparison():
    benchmarkFunctions = [
        BenchmarkFunctionEnum.sphere_function,
        BenchmarkFunctionEnum.calculate_step_2_function_value,
        BenchmarkFunctionEnum.quartic_function,
        BenchmarkFunctionEnum.schwefel_2_21_function,
        BenchmarkFunctionEnum.schwefel_2_22_function,
        BenchmarkFunctionEnum.six_hump_camel_back,
        BenchmarkFunctionEnum.rastrigin,
        BenchmarkFunctionEnum.griewank_function,
        BenchmarkFunctionEnum.branin_function,
        BenchmarkFunctionEnum.ackley_function,
    ]
    benchmarkFunctionsIterations = {}
    for benchmarkFunction in benchmarkFunctions:
        benchmarkFunctionsIterations[benchmarkFunction] = {
            "pso_values": get_algorithm_values(AlgorithmsEnum.PSO, benchmarkFunction),
            "mpso_values": get_algorithm_values(AlgorithmsEnum.MPSO, benchmarkFunction),
        }

    # print("benchmarkFunctionsIterations")
    # print(benchmarkFunctionsIterations)


custom_particle_swarm_optimization_comparison()
