import numpy as np
from g07 import G07


class EvolutionaryAlgorithm:
    def __init__(self, gene_length, population_size, mutation_rate, generations):
        self.gene_length = gene_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = np.random.uniform(low=-10, high=10, size=(population_size, gene_length))
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate_population(self, fitness_function, generation, heuristic):
        fitness_values = np.array([fitness_function(individual, generation, heuristic) for individual in self.population])
        return fitness_values

    def select_parents(self, fitness_values, tournament_size=3):
        parents = []

        for _ in range(self.population_size):
            tournament_indices = np.random.choice(self.population_size, tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            selected_parent = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(self.population[selected_parent])

        return np.array(parents)

    def crossover(self, parents, crossover_rate=0.8):
        offspring = []

        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, self.gene_length - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspring.extend([child1, child2])

        return np.array(offspring)

    def mutate(self, population):
        mutated_population = population.copy()

        for i in range(self.population_size):
            for j in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    mutated = mutated_population[i, j] + np.random.uniform(-10, 10)
                    mutated_population[i, j] = mutated if -10 <= mutated <= 10 else mutated_population[i, j]
                     
        return mutated_population

 
    def run(self, fitness_function, heuristic):
        for generation in range(self.generations):
                        
            # Evaluate the population
            fitness_values = self.evaluate_population(fitness_function, generation, heuristic)

            # Update the best solution
            min_fitness = np.min(fitness_values)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = self.population[np.argmin(fitness_values)]

            # Select parents
            parents = self.select_parents(fitness_values)

            # Apply crossover
            offspring = self.crossover(parents)

            # Apply mutation
            mutated_offspring = self.mutate(offspring)

            # Replace old population with new population
            self.population = mutated_offspring

        return self.best_solution, self.best_fitness

# Example usage:
def custom_fitness_function(individual, generation, heuristic=False):
    g07 = G07()
    
    # Objective function value
    objective_value = g07.fitness(individual)
    
    # Objective Weight
    objective_weight = 1
    
    # Penalty term
    penalty_value = g07.penalty(individual, heuristic)
    
    
    # Weight for the penalty 
    penalty_weight = 1
    
    # Weight for the generation
    generation_weight = 0
    
    # Combined fitness value (objective + penalty)
    combined_fitness = objective_value * objective_weight + penalty_weight * penalty_value + generation * generation_weight
    
    return combined_fitness

ea = EvolutionaryAlgorithm(gene_length=10, population_size=100, mutation_rate=.2, generations=200)
best_solution, best_fitness = ea.run(custom_fitness_function, heuristic=True)

print("\n\nBest Solution:", best_solution)
print("Best Fitness:", best_fitness)
print("constraints numbers:", G07().constraints(best_solution), end='\n\n')


