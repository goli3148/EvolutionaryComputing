import numpy as np
from typing import List, Callable, Tuple
import random
from matplotlib import pyplot as plt

# BENCHMARK FUNCTIONS DEFINITION
def benchmarkFunction1(individual: List)-> int:
    return (np.sum(individual)) ** 2
def benchmarkFunction2(individual: List)-> int:
    sum = 0
    for index in range(len(individual)):
        if index + 1 == len(individual):break
        sum+= 100 * (individual[index+1] - individual[index]**2)**2 + (individual[index] - 1)**2
    return sum

# GENE DEFINITION
def genes():
    return (0,1)


# CREATE RANDOM POPULATION
def population_create(populationSize: int, geneSize: int) -> List:
    population = []
    for index in range(populationSize):
        individual = [random.choice(genes()) for _ in range(geneSize)]
        population.append(individual)
    return population

# MUTATION
def mutate_individual(individual: List, mutation_rate) -> List:
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_individual.append(random.choice(genes()))
        else:
            mutated_individual.append(gene)
    return mutated_individual

# NORAMLIZATION THE EVALUATION
def normalization(bound: Tuple, individual: List, func: Callable)-> int:
    score = func(individual)
    if score < bound[0]:    return bound[0]
    elif score > bound[1]:  return bound[1]
    else:                   return score
    
# Evaluation
def evaluation(individual: List, func: Callable, bound: Tuple)->int:
    return normalization(bound=bound, func=func, individual=individual)

# EVOLUTIONAR STRATEGY
def ES(func: callable, bound:Tuple, sigma:int=.5):
    # CHEKC IT FOR DIFFRENT SIZE OF GENES
    N = [2, 5, 10, 50]
    generations = 5
    for n in N:
        x,y=ES_SUB(n, generations, func, bound, sigma)
        plt.plot(x,y, label=f'N={n}')
    plt.title(f"ES for sigma={sigma}")
    plt.legend()
    plt.show()   
def ES_SUB(N:int, generations: int, func: callable, bound:Tuple, sigma):
    x,y = [0], [0]
    population = population_create(populationSize=1, geneSize=N)
    individual = population[0]
    for generation in range(generations):
        new_individual = mutate_individual(individual, mutation_rate=sigma)
        new_score = evaluation(individual=new_individual, func=func, bound=bound)
        score = evaluation(individual=individual, func=func, bound=bound)
        if new_score > score:
            individual = new_individual
            y.append(new_score)
        else:
            individual = new_individual
            y.append(new_score)
        x.append(x[-1]+1)
    return x,y

# CHECK FOR DIFFRENT SIGMAS
for _ in range(10,100,10):
    ES(benchmarkFunction2, (-30, 30), sigma=_/100)