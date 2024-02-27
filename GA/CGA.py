import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any, List, Callable
from decimal import *
import random
import copy
import matplotlib.pyplot as plt

class CGA:
    population: NDArray
    parent: NDArray
    
    Meu: int
    L: int
    Pc: int
    Pm: int
    MaxGen: int
    generation: int
    
    fitness_func: Callable
    
    plot: plt
    
    best_cases_generation: List
    worst_cases_generation: List
    ave_cases_generation: List
    
    def __init__(self, fitness_func:Callable) -> None:
        self.fitness_func = fitness_func
        ...
        
    def run(self, plotting=True):
        self.initialization()
        self.generation = 0
        
        while self.generation < self.MaxGen:
            # Main Algorithm
            self.selection()
            self.crossover()
            self.mutation()
            # Gathering Data
            best, worst, ave = self.worst_best_average()
            self.best_cases_generation.append(best)
            self.worst_cases_generation.append(worst)
            self.ave_cases_generation.append(ave)
            # Reset Variables for Next Generation
            self.parent = np.zeros((self.Meu,self.L))
            self.generation += 1
        
        if plotting:
            #plotting results
            self.plotting()
    
    def initialization(self):
        # self.population = (np.random.rand(self.Meu, self.L) > .5) * 1
        self.population = (np.random.rand(self.Meu, self.L) > 1) * 1
        self.parent = np.zeros((self.Meu,self.L))
        
        self.best_cases_generation, self.worst_cases_generation, self.ave_cases_generation = [], [], []
        self.plot = plt
    
    def selection(self) -> None:
        # sigma(function evaluation)
        # this number is to large
        # we use Desimal Module to control this large data
        sum_eva = Decimal(1) 
        for ind in self.population:sum_eva += self.function_evaluation(ind)
        # calculate probabilty for every population
        prob = np.array([[index,self.function_evaluation(self.population[index])/sum_eva] for index in range(len(self.population))], dtype=np.float64)
        # roulette wheel
        for i in range(self.Meu):
            index = int(random.choices(prob[:,0], weights=prob[:,1])[0])
            self.parent[i] = self.population[index].copy()
    
    def crossover(self) -> None:
        for i in range(0, self.Meu, 2):
            # copy two parents in "a" and "b" variables
            a, b = (self.parent[i].copy(), self.parent[i+1].copy())
            r = random.random()
            if r<self.Pc:
                # crossover happens
                cutoff = random.randint(1, self.L-2)
                temp = b[:cutoff].copy()
                b[:cutoff], a[:cutoff] = a[:cutoff], temp 
            self.parent[i], self.parent[i+1] = a, b
        self.population = self.parent.copy()
            
        
    def mutation(self) -> None:
        # Choose individuals 
        for ind in self.population:
            # search all genes
            for index in range(self.L):
                r = random.random()
                if r<self.Pm:
                    # flip the gene with r probibility
                    ind[index] = 1 - ind[index]
    
    def worst_best_average(self, worst_case=None, best_case=None) -> Tuple:
        # best evaluation is worst case from the start
        best    = self.function_evaluation(worst_case) if not worst_case == None \
            else self.function_evaluation(np.zeros((self.L,)))
        # worst evaluation is best case from the start
        worst   = self.function_evaluation(best_case) if not best_case == None \
            else self.function_evaluation(np.ones((self.L,)))
        sum_eva = Decimal(0)
        for ind in self.population:
            ind_eva = self.function_evaluation(ind)
            if ind_eva > best:
                best = copy.deepcopy(ind_eva)
            if ind_eva < worst:
                worst = copy.deepcopy(ind_eva)
            sum_eva += ind_eva
        return best, worst, sum_eva/self.Meu
    
    
    def function_evaluation(self, args) -> int:
        return self.bit_to_int(args)
    
    def bit_to_int(self, bits) -> int:
        bit_string = ''
        for bit in bits: bit_string+=str(int(bit))
        # Convert string of bits to integer
        return int(bit_string, 2)
    
    
    def plotting(self):
        plt.figure()
        self.plot.title("CGA runs:1")
        self.plot.xlabel('generations')
        self.plot.ylabel('fitness')
        
        generations = [i for i in range(self.MaxGen)]
        
        self.plot.plot(generations, self.best_cases_generation, label="best", c='g')
        self.plot.plot(generations, self.worst_cases_generation, label="worst", c='r')
        self.plot.plot(generations, self.ave_cases_generation, label='average', c='b')
        
        plt.legend()
        self.plot.show()


        
        