import numpy as np

# Problem Parameters
def objective_function(traffic_lights):
    return sum((timing - 30) ** 2 for timing in traffic_lights)

# Cuckoo Search Parameters
POP_SIZE = 20
MAX_GENERATIONS = 100
PA = 0.25  # Discovery rate of alien eggs
LOWER_BOUND = 10
UPPER_BOUND = 60

# Initialize population
def initialize_population(size, dim):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (size, dim))

# Levy flight function
def levy_flight(Lambda):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (
        np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2)
    )) ** (1 / Lambda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step[0]

# Replace worst nests with new solutions
def replace_worst(population, fitness):
    worst_index = np.argmax(fitness)
    population[worst_index] = np.random.uniform(LOWER_BOUND, UPPER_BOUND, population.shape[1])
    fitness[worst_index] = objective_function(population[worst_index])

# Cuckoo Search Algorithm
def cuckoo_search(dim):
    population = initialize_population(POP_SIZE, dim)
    fitness = np.array([objective_function(ind) for ind in population])

    best_solution = population[np.argmin(fitness)]
    best_fitness = min(fitness)

    for _ in range(MAX_GENERATIONS):
        for i in range(POP_SIZE):
            new_solution = population[i] + levy_flight(1.5) * np.random.uniform(-1, 1, dim)
            new_solution = np.clip(new_solution, LOWER_BOUND, UPPER_BOUND)
            new_fitness = objective_function(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        for _ in range(int(PA * POP_SIZE)):
            replace_worst(population, fitness)

        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
            best_solution = population[current_best_index]
            best_fitness = fitness[current_best_index]

    return best_solution, best_fitness

# Run the algorithm
if __name__ == "__main__":
    num_traffic_lights = 5  # Number of intersections
    best_solution, best_fitness = cuckoo_search(num_traffic_lights)
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
    print("Ayman Khan-1BM22CS062")