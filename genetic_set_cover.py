from SetCoveringProblemCreator import *
import random
import time
import math
import csv
import numpy as np
from typing import List, Set, Tuple

#added to ensure reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def calculate_fitness(individual: List[int], subsets: List[Set[int]], universe_size: int) -> float:
    covered = set().union(*[subsets[i] for i, gene in enumerate(individual) if gene])
    num_selected = sum(individual)
    coverage = len(covered)

    if coverage < universe_size:
        return 100 * (coverage / universe_size)  # Partial coverage
    else: 
        return 100 + 10000 * (1 - (num_selected / len(subsets)))  # Full coverage, favor fewer subsets

def tournament_selection(population: List[List[int]], fitness_values: List[float], tournament_size: int) -> List[int]:
    selected = random.sample(range(len(population)), tournament_size)
    return population[max(selected, key=lambda i: fitness_values[i])]

def weighted_random_choices(population: List[List[int]], weights: List[float], k: int) -> List[List[int]]:
    total = sum(weights)
    if total == 0:
        return random.choices(population, k=k)  # If all weights are 0, choose randomly
    return random.choices(population, weights=weights, k=k)

def uniform_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

def reproduce(parent1: List[int], parent2: List[int]) -> List[int]:
    n = len(parent1)
    c = random.randint(1, n - 1)
    return parent1[:c] + parent2[c:]

def crossover_with_fitness_comparison(parent1: List[int], parent2: List[int], subsets: List[Set[int]], universe_size: int) -> List[int]:
    length = len(parent1)
    crossover_point = random.randint(1, length - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child: List[int], mutation_rate: float) -> List[int]:
    return [1 - gene if random.random() < mutation_rate else gene for gene in child]

def single_bit_flip_mutate(child: List[int], mutation_rate: float) -> List[int]:
    # With a probability of mutation_rate, perform a flip
    if random.random() < mutation_rate:
        # Select a random index to flip
        index = random.randint(0, len(child) - 1)
        # Flip the bit at the selected index
        child[index] = 1 - child[index]
    return child

def initialize_population(num_subsets: int, population_size: int) -> List[List[int]]:
    return [
        [random.randint(0, 1) for _ in range(num_subsets)]
        for _ in range(population_size)
    ]

#Textbook implementation without any improvements
def genetic_algorithm_naive(subsets, universe_size, population_size, generations, mutation_rate):
    population = [[random.randint(0, 1) for _ in range(len(subsets))] for _ in range(population_size)]
    
    start_time = time.time()
    best_fitness = 0
    best_individual = None

    for _ in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        
        current_best_fitness = max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
        
        if time.time() - start_time > 40:  # 45 seconds time limit
            break
        
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = weighted_random_choices(population, fitness_values, 2)
            child = reproduce(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    time_taken = time.time() - start_time
    return best_individual, best_fitness, time_taken

#Incorporates all the improvements
def genetic_algorithm_improved(subsets: List[Set[int]], universe_size: int, population_size: int, generations: int, tournament_size: int, mutation_rate: float, elitism_percentage: float) -> Tuple[List[int], float, float]:
    num_subsets = len(subsets)
    population = initialize_population(num_subsets, population_size)
    
    start_time = time.time()
    best_fitness = 0
    best_individual = None
    no_improvement_counter = 0
    no_improvement_limit = 0.3 * generations
    elite_size = max(1, int(elitism_percentage * population_size))  # Ensure at least 1 elite individual

    for generation in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        current_best_fitness = max(fitness_values)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= no_improvement_limit or time.time() - start_time > 40:
            break
        
        # Sort population by fitness (descending order)
        sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
        
        # Keep top few as elite
        elite = sorted_population[:elite_size]
        
        # Generate new population
        new_population = []
        for _ in range(population_size - elite_size):
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)
            child = uniform_crossover(parent1, parent2)
            # child = crossover_with_fitness_comparison(parent1, parent2, subsets, universe_size)
            child = mutate(child, mutation_rate)
            # child = single_bit_flip_mutate(child, mutation_rate)
            new_population.append(child)
        
        # Add elite individuals to the new population
        population = elite + new_population
    
    time_taken = time.time() - start_time
    return best_individual, best_fitness, time_taken

def main():
    roll_no = "2021A7PS2606G"
    set_random_seed(10)
    scp = SetCoveringProblemCreator()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")

    #Best Parameters
    universe_size = 100
    population_size = 1000
    generations = 1000
    mutation_rate = 0.02
    elitism_percentage = 0.2 #between 0 and 1
    tournament_size = 4
    
    best_solution, fitness_value, time_taken = genetic_algorithm_improved(listOfSubsets, universe_size, population_size, generations, tournament_size, mutation_rate, elitism_percentage)
        
    print(f"Roll no: {roll_no}")
    print(f"Number of subsets in scp_test.json file: {len(listOfSubsets)}")
    print("Solution:")
    print(", ".join(f"{i}:{gene}" for i, gene in enumerate(best_solution)))
    print(f"Fitness value of best state: {fitness_value}")
    print(f"Minimum number of subsets that can cover the Universe-set: {sum(best_solution)}")
    print(f"Time taken: {time_taken:.2f} seconds")

if __name__=='__main__':
    main()