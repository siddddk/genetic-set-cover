from SetCoveringProblemCreator import *
import random
import time
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_fitness(individual, subsets, universe_size):
    covered = set()
    num_selected = sum(individual)
    for i, gene in enumerate(individual):
        if gene:
            covered.update(subsets[i])
    
    coverage = len(covered)
    if coverage < universe_size:
        return 100*(coverage / universe_size)  # Partial coverage
    return 100 + 10000*(1 - (num_selected / len(subsets)))  # Full coverage, favor fewer subsets

def weighted_random_choices(population, weights, k):
    total = sum(weights)
    if total == 0:
        return random.choices(population, k=k)  # If all weights are 0, choose randomly
    return random.choices(population, weights=weights, k=k)

def reproduce(parent1, parent2):
    n = len(parent1)
    c = random.randint(1, n - 1)
    return parent1[:c] + parent2[c:]

def mutate(child, mutation_rate):
    return [1 - gene if random.random() < mutation_rate else gene for gene in child]

def genetic_algorithm(subsets, universe_size, population_size, generations, mutation_rate):
    population = [[random.randint(0, 1) for _ in range(len(subsets))] for _ in range(population_size)]
    
    start_time = time.time()
    best_fitness = 0
    best_individual = None
    fitness_over_generations = []

    for _ in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        
        current_best_fitness = max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
        
        if time.time() - start_time > 40:  # 45 seconds time limit
            break
        
        fitness_over_generations.append(best_fitness)
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = weighted_random_choices(population, fitness_values, 2)
            child = reproduce(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    time_taken = time.time() - start_time
    return best_individual, best_fitness, time_taken, fitness_over_generations

def main():
    results = {}
    collection_sizes = [50, 150, 250, 350]
    mean_fitness_final = []  # Store mean best fitness value for each |S|
    std_fitness_final = []   # Store standard deviation of best fitness value for each |S|

    for totalSets in collection_sizes:
        all_fitness_over_generations = []
        for i in range(10):
            scp = SetCoveringProblemCreator()
            listOfSubsets = scp.Create(usize=100, totalSets=totalSets)
            universe_size = 100
            population_size = 50
            generations = 50
            mutation_rate = 0.01
            best_solution, fitness_value, time_taken, fitness_over_generations = genetic_algorithm(listOfSubsets, universe_size, population_size, generations, mutation_rate)
            all_fitness_over_generations.append(fitness_over_generations)
            # print(fitness_over_generations)
            # Plot how the mean best fitness changes over generations for each |S|
                # Convert list of fitness values over generations to a NumPy array for easy calculation
        all_fitness_over_generations = np.array(all_fitness_over_generations)
        
        # Calculate the mean and standard deviation for each generation
        mean_fitness = np.mean(all_fitness_over_generations, axis=0)
        std_fitness = np.std(all_fitness_over_generations, axis=0)

        # Store the mean and standard deviation for the last generation (i.e., the final generation)
        mean_fitness_final.append(mean_fitness[-1])
        std_fitness_final.append(std_fitness[-1])

        # Store the results for plotting
        results[totalSets] = (mean_fitness, std_fitness)

        # Plot mean and standard deviation over generations for each |S|
        plt.figure(figsize=(10, 6))
        generations_range = range(generations)
        
        plt.plot(generations_range, mean_fitness, label=f'Mean Fitness for |S| = {totalSets}')
        plt.fill_between(generations_range, mean_fitness - std_fitness, mean_fitness + std_fitness, 
                         alpha=0.2, label=f'Standard Deviation for |S| = {totalSets}')
        
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(f'Fitness over Generations for |S| = {totalSets}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot as a PNG image
        plt.savefig(f'fitness_over_generations_S_{totalSets}.png')
        plt.close()

    # Plot mean best fitness value and standard deviation for each |S|
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(collection_sizes, mean_fitness_final, yerr=std_fitness_final, fmt='-o', color='b', 
                 label='Mean Best Fitness with Standard Deviation')
    
    plt.xlabel('Collection Size |S|')
    plt.ylabel('Mean Best Fitness After 50 Generations')
    plt.title('Mean Best Fitness for Different Collection Sizes |S|')
    plt.grid(True)

    # Save the plot as a PNG image
    plt.savefig('mean_best_fitness_with_std_for_collection_sizes.png')
    plt.close()

if __name__=='__main__':
    main()