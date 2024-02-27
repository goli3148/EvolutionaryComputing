from typing import Callable
from CGA import CGA
import matplotlib.pyplot as plt
import numpy as np

class functionEvaluations:
    def function_evaluation1(self, val, old_min, old_max):
        val = self.map_to_range(val, 0, 10, old_min, old_max)
        return val ** 2
    def function_evaluation2(self, val, old_min, old_max):
        val = self.map_to_range(val, -10, 10, old_min, old_max)
        return int(np.abs(np.cos(val) * np.exp(-1*np.abs(val)/5)))
    
    def map_to_range(self, val, new_min, new_max, old_min, old_max) -> int:
        old_range = old_max - old_min
        new_range = new_max - new_min
        scaled = (val - old_min) / old_range
        return int(new_min + (scaled * new_range))


class main(CGA):
        
    def __init__(self, fitness_func: Callable) -> None:
        self.L = 40
        self.Meu = 50
        self.Pc = .1
        self.Pm = .1
        self.MaxGen = 50
        super().__init__(fitness_func)
        
    def function_evaluation(self, args) -> int:
        return self.fitness_func(self.bit_to_int(args), self.bit_to_int(np.zeros((self.L,))), self.bit_to_int(np.ones(self.L,)))

    def runs(self):
        self.runs_iterator = 10
        # Matrix: Each Row shows run index and each coloumn show generation
        self.best_runs = np.zeros((self.runs_iterator, self.MaxGen))
        self.worst_runs = np.zeros((self.runs_iterator, self.MaxGen))
        self.ave_runs = np.zeros((self.runs_iterator, self.MaxGen))
        for i in range(self.runs_iterator):
            self.run(plotting=False)
            self.best_runs[i] = np.array(self.best_cases_generation)
            self.worst_runs[i] = np.array(self.worst_cases_generation)
            self.ave_runs[i] = np.array(self.ave_cases_generation)
        self.plotting_runs()
    def plotting_runs(self):
        plot = plt
        plot.title(f"CGA runs:{self.runs_iterator}")
        self.plot.xlabel('generations')
        self.plot.ylabel('fitness')
        
        generations = [i for i in range(self.MaxGen)]
        
        plot.plot(generations, np.mean(self.best_runs, axis=0), label="best runs", c='g')
        plot.plot(generations, np.mean(self.worst_runs, axis=0), label="worst runs", c='r')
        plot.plot(generations, np.mean(self.ave_runs, axis=0), label='average runs', c='b')
        
        plt.legend()
        plot.show()

cga = main(functionEvaluations().function_evaluation1)
cga.run()
cga.runs()
cga = main(functionEvaluations().function_evaluation2)
cga.run()
cga.runs()