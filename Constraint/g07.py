import numpy as np
class G07:
    # The optimum solution is x* = (2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173,
    #                               0.990654756560493, 1.43057392853463, 1.32164415364306, 9.82872576524495,
    #                               8.2800915887356, 8.3759266477347)
    # where f(x*) = 24.30620906818.

    def __init__(self):
        self.dimension = 10
        self.lower_bounds = np.zeros(self.dimension)
        self.upper_bounds = 10 * np.ones(self.dimension)
        self.g = np.empty(8)

    def fitness(self, x):

        f = x[0]**2 + x[1]**2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10)**2 + \
            4 * (x[3] - 5)**2 + (x[4] - 3)**2 + 2 * (x[5] - 1)**2 + 5 * x[6]**2 + \
            7 * (x[7] - 11)**2 + 2 * (x[8] - 10)**2 + (x[9] - 7)**2 + 45

        return f

    def constraints(self, x):
        self.g[0] = -105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7]
        self.g[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
        self.g[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
        self.g[3] = 3 * (x[0] - 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120
        self.g[4] = 5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40
        self.g[5] = x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
        self.g[6] = 0.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30
        self.g[7] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9]

        return self.g
    
    def penalty(self, x, heuristic=False):
        self.constraints(x)
        pen = 0
        for index,i in enumerate(self.g):
            if heuristic and (index == 0 or index == 2):
                pen = pen+i*4 if i > 0 else pen
            else:
                pen = pen+i if i > 0 else pen
        return pen
    
