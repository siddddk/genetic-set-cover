from SetCoveringProblemCreator import *
import random
import time
import csv
import numpy as np
from typing import List, Set, Tuple

def calculate_fitness(individual: List[int], subsets: List[Set[int]], universe_size: int) -> float:
    covered = set().union(*[subsets[i] for i, gene in enumerate(individual) if gene])
    num_selected = sum(individual)
    coverage = len(covered)
    
    if coverage < universe_size:
        return 100 * (coverage / universe_size)  # Partial coverage
    return 100 + 10000 * (1 - (num_selected / len(subsets)))  # Full coverage, favor fewer subsets

def tournament_selection(population: List[List[int]], fitness_values: List[float], tournament_size: int) -> List[int]:
    selected = random.sample(range(len(population)), tournament_size)
    return population[max(selected, key=lambda i: fitness_values[i])]

def uniform_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

def mutate(child: List[int], mutation_rate: float) -> List[int]:
    return [1 - gene if random.random() < mutation_rate else gene for gene in child]

def initialize_population(num_subsets: int, population_size: int) -> List[List[int]]:
    return [
        [random.randint(0, 1) for _ in range(num_subsets)]
        for _ in range(population_size)
    ]

def genetic_algorithm(subsets: List[Set[int]], universe_size: int, population_size: int, generations: int, mutation_rate: float) -> Tuple[List[int], float, float]:
    num_subsets = len(subsets)
    population = initialize_population(num_subsets, population_size)
    
    start_time = time.time()
    best_fitness = 0
    best_individual = None
    no_improvement_counter = 0
    no_improvement_limit = 0.3 * generations

    for generation in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        current_best_fitness = max(fitness_values)
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= no_improvement_limit or time.time() - start_time > 45:
            print(f"Stopping criteria met at generation {generation}")
            break
        
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitness_values, tournament_size=3)
            parent2 = tournament_selection(population, fitness_values, tournament_size=3)
            child = uniform_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population

        # Elitism: Keep the best individual
        best_index = fitness_values.index(max(fitness_values))
        population[0] = population[best_index]
    
    time_taken = time.time() - start_time
    return best_individual, best_fitness, time_taken

def main():
    roll_no = "2021A7PS2606G"
    scp = SetCoveringProblemCreator()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    # listOfSubsets = scp.Create(usize=100, totalSets=150)
    
    universe_size = 100
    population_size = 400
    generations = 2500
    mutation_rate = 0.01

    best_solution, fitness_value, time_taken = genetic_algorithm(listOfSubsets, universe_size, population_size, generations, mutation_rate)        

    print(f"Roll no: {roll_no}")
    print(f"Number of subsets in scp_test.json file: {len(listOfSubsets)}")
    print("Solution:")
    print(", ".join(f"{i}:{gene}" for i, gene in enumerate(best_solution)))
    print(f"Fitness value of best state: {fitness_value}")
    print(f"Minimum number of subsets that can cover the Universe-set: {sum(best_solution)}")
    print(f"Time taken: {time_taken:.2f} seconds")

if __name__=='__main__':
    main()