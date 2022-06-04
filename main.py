# P19129 Paraskevi Palioura

import random
import numpy as np
import operator
import matplotlib.pyplot as plt

from Fitness import Fitness

n_cities = 5
n_population = 10
mutation_rate = 0.2
crossover_rate = 0.5

cities = np.array(['1', '2', '3', '4', '5'])

graph = np.array([
    [0, 4, 4, 7, 3],
    [4, 0, 2, 3, 5],
    [4, 2, 0, 2, 3],
    [7, 3, 2, 0, 6],
    [3, 5, 3, 6, 0]
])


# Create the first population set
def generate_population(cities_list, pop_size):
    population_set = []
    for i in range(pop_size):
        ind = cities_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
        ind = np.ndarray.tolist(ind)
        # Checking if individual already exists and creating another one if it does
        for j in range(len(population_set)):
            if np.any(np.all(ind == population_set[j])):
                ind = cities_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
                ind = np.ndarray.tolist(ind)

        population_set.append(ind)
    return np.array(population_set)


# Estimating Fitness
def rank_routes(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness(graph)
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


# Select Individuals for Crossover
def selection(ranked_pop, size):
    selection_results = []
    for i in range(0, int(size)):
        selection_results.append(ranked_pop[i][0])

    return selection_results


def mating_pool(population, selection_results):
    matingpool = []
    for i in range(0, len(selection_results)):
        ind = selection_results[i]
        matingpool.append(population[ind])
    return matingpool


# Ordered Crossover
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    for i in range(start_gene, end_gene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breed_population(matingpool, size):
    new_population_set = []
    length = len(matingpool) - size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, int(size)):
        new_population_set.append(matingpool[i])

    for i in range(0, int(length)):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        new_population_set.append(child)
    return new_population_set


# Mutation
def mutate(individual, rate):
    for swapped in range(len(individual)):
        if random.random() < rate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
        return individual


def mutate_population(population, rate):
    mutated_population = []

    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], rate)
        mutated_population.append(mutated_ind)
    return mutated_population


# Creating next generations
def next_generation(current_gen, size, rate):
    ranked_pop = rank_routes(current_gen)
    selectionResults = selection(ranked_pop, size * crossover_rate)
    matingpool = mating_pool(current_gen, selectionResults)
    children = breed_population(matingpool, size * crossover_rate)
    mutated = mutate_population(current_gen, rate)
    # next gen = current gen  + children + mutated (sort unique) and keep best 10
    next_gen = []
    for i in range(len(children)):
        if type(children[i]) is np.ndarray:
            next_gen.append(np.ndarray.tolist(children[i]))
        else:
            next_gen.append(children[i])

    for i in range(len(mutated)):
        if type(mutated[i]) is np.ndarray:
            next_gen.append(np.ndarray.tolist(mutated[i]))
        else:
            next_gen.append(mutated[i])

    if type(current_gen) is np.ndarray:
        current_gen = np.ndarray.tolist(current_gen)

    for i in range(len(current_gen)):
        next_gen.append(current_gen[i])

    ng = []
    ranked = rank_routes(next_gen)
    ranked = ranked[0:size]  # the n best
    for i in range(0, len(ranked)):
        ranked[i] = ranked[i][0]

    for i in range(0, len(next_gen)):
        if i in ranked:
            ng.append(next_gen[i])
    return ng


# The genetic algorithm
def genetic_algorithm(size, rate):
    pop = generate_population(cities, n_population)
    print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))
    print("Initial best route: " + str(pop[rank_routes(pop)[0][0]]))
    progress = [1 / rank_routes(pop)[0][1]]
    # Repeat until best route is found
    while (1 / rank_routes(pop)[0][1]) != 15.0:
        pop = next_generation(pop, size, rate)
        progress.append(1 / rank_routes(pop)[0][1])

    print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
    print("Final best route: " + str(pop[rank_routes(pop)[0][0]]))
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return best_route


genetic_algorithm(n_population, mutation_rate)
