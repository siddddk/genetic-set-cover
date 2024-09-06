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

# def mutate(child, mutation_rate):
#     # Select a random gene index
#     gene_index = random.randint(0, len(child) - 1)
    
#     # Flip the selected gene with mutation_rate probability
#     if random.random() < mutation_rate:
#         child[gene_index] = 1 - child[gene_index]
    
#     return child

def initialize_population(subsets, universe_size, population_size):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(len(subsets))]
        population.append(individual)
    return population

def genetic_algorithm(subsets, universe_size, population_size, generations, mutation_rate):
    population = initialize_population(subsets, universe_size, population_size)
    
    start_time = time.time()
    best_fitness = 0
    best_individual = None
    no_improvement_counter = 0
    no_improvement_limit = 0.3 * generations

    for _ in range(generations):
        fitness_values = [calculate_fitness(ind, subsets, universe_size) for ind in population]
        current_best_fitness = max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness_values.index(best_fitness)]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= no_improvement_limit:
            print(f"No improvement in {no_improvement_limit} generations. Halting.")
            break
        
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
    roll_no = "2021A7PS2606G"  # Replace with your actual roll number
    scp = SetCoveringProblemCreator()

    # Read the actual problem from scp_test.json
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    # listOfSubsets = scp.Create(usize=100, totalSets=150)
    print(len(listOfSubsets))
    
    
    # Genetic Algorithm parameters
    universe_size = 100
    population_size = 250
    generations = 2000
    mutation_rate = 0.001
    # mutation_rate = 0.9

    best_solution, fitness_value, time_taken = genetic_algorithm(listOfSubsets, universe_size, population_size, generations, mutation_rate)        


    print(f"Roll no : {roll_no}")
    print(f"Number of subsets in scp_test.json file : {len(listOfSubsets)}")
    print("Solution :")
    for i, gene in enumerate(best_solution):
        print(f"{i}:{gene}", end=", ")
    print()
    print(f"Fitness value of best state : {fitness_value}")
    print(f"Minimum number of subsets that can cover the Universe-set : {sum(best_solution)}")
    print(f"Time taken : {time_taken:.2f} seconds")

if __name__=='__main__':
    main()