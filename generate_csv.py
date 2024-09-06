from SetCoveringProblemCreator import *
import random
import time
import csv

def calculate_fitness(individual, subsets, universe_size):
    covered = set()
    num_selected = sum(individual)
    for i, gene in enumerate(individual):
        if gene:
            covered.update(subsets[i])
    
    coverage = len(covered)
    if coverage < universe_size:
        return 100*(coverage / universe_size)  # Partial coverage
    return 100 + 100000*(1 - (num_selected / len(subsets)))  # Full coverage, favor fewer subsets

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

    for _ in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        
        current_best_fitness = max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
        
        if time.time() - start_time > 45:  # 45 seconds time limit
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

def main():
    for totalSets in [50, 150, 250, 350]:
        for i in range(50):
            scp = SetCoveringProblemCreator()
            listOfSubsets = scp.Create(usize=100, totalSets=totalSets)
            universe_size = 100
            population_size = 50
            generations = 50
            mutation_rate = 0.001
            best_solution, fitness_value, time_taken = genetic_algorithm(listOfSubsets, universe_size, population_size, generations, mutation_rate)
            with open(f'fitness_values_{totalSets}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fitness_value, sum(best_solution), time_taken])

if __name__=='__main__':
    main()