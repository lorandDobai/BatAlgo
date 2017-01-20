import numpy as np
import random
import math
from typing import Callable,Tuple
def bat_solve(pop_size: int, gen_nr: int, dim: int,  fn: Callable,
              fqmin : float = 0, fqmax: float = 2 , low_bound: float =-10, up_bound: float =10) ->Tuple[float,np.ndarray]:
    """

    :param pop_size: Population size, i.e. number of bats
    :param gen_nr: Number of generations i.e. iterations
    :param dim: Search space dimensionality
    :param fn: Fitness function
    :param fqmin: Minimum frequency
    :param fqmax: Max Frequency
    :param low_bound: Lower bound of domain
    :param up_bound: Upper bound of domain
    :return: A tuple consisting of the minimum fitness value and it's corresponding point
    """
    ALPHA = GAMMA = 0.9
    #Initialize Vectors
    frequencies = np.zeros((pop_size,),dtype=np.float64)

    velocities = np.zeros((pop_size, dim),dtype=np.float64)

    solutions = np.ndarray((pop_size,dim),dtype=np.float64)
    fitness = np.ndarray((pop_size,),dtype=np.float64)

    #Pulse - Loudness Optimization
    loudness = np.zeros((pop_size,),dtype=np.float64) + np.random.uniform(0.3,0.7,(pop_size))
    pulse_0 = np.ones((pop_size,),dtype=np.float64) *   np.random.uniform(0.3,0.7,(pop_size))
    pulses = np.copy(pulse_0)

    for i in range(pop_size):
        solutions[i] = low_bound + (up_bound - low_bound) * np.random.random((dim,))
        fitness[i] = fn(solutions[i])

    i_min = np.argmin(fitness)
    fn_min, best_sol = fitness[i_min], solutions[i_min]
    decrease_counter = 0
    for t in range(gen_nr):

        for i in range(pop_size):

            frequencies[i] = fqmin + (fqmin-fqmax) * random.uniform(0, 1)
            velocities[i] = velocities[i] + (solutions[i]-best_sol)*frequencies[i]


            if  np.random.rand()  > pulses[i]:
                new_solution = best_sol + 0.001 * np.random.normal(0,1,(dim,))
            else:
                new_solution = solutions[i] + velocities[i]
                clip_bounds(new_solution, low_bound, up_bound)

            fnew = fn(new_solution)

            if fnew <= fitness[i] and np.random.rand() < loudness[i]:
                solutions[i], fitness[i] = new_solution,fnew

            if fnew <= fn_min:
                best_sol,fn_min = new_solution, fnew
                decrease_counter+= 1
                pulses = pulse_0 * (1-np.exp(-GAMMA*decrease_counter))
                loudness = loudness * ALPHA


    return fn_min,best_sol

def clip_bounds(arr, low, top):
    arr[arr < low] = low
    arr[arr > top] = top
    return arr

def sphere(cand):
    """
    Domain in [-10,10]
    Minimum 0 in (0,0,..0)
    :param cand:
    :return:
    """
    return np.sum(cand**2)

def eggcrate(cand):
    """
    Domain = (-2pi,2pi)
    Minimum 0 at (0,0)
    :param cand:
    :return:
    """
    x,y = cand
    return x**2 + y**2 + 25*(math.sin(x)**2 + math.sin(y)**2)

def rosenbrock(cand):
    """
    Rosenbrock benchmark function. Global minimum 0 known at (1,...,1) in for any D >= 3
    + another(-1, 1, 1, ..., 1) for D>3
    Domain = (-2.048,2.018)
    :param cand: solution to be evaluatedx
    :return: rosebrock function value
    """
    result = 0
    for i in range(len(cand)-1):
        x_curr, x_next = cand[i]**2,cand[i+1]
        result += (1-x_curr)**2 + 100 * (x_next - x_curr)**2
    return result



def xyfn(cand):
    """
    xy = 6 and x+y = 5
    Domain = [-10,10]
    :param cand:
    :return:
    """
    result = 0
    result += abs(5-(cand[0]+cand[1]))
    result += abs(6-(cand[0]*cand[1]))
    return result
if __name__ == '__main__':
    np.random.seed(42)
    L = []

    for i in range(30):
        fmin, sol = bat_solve(40, 2000, 7, rosenbrock,fqmin=0,fqmax=2,low_bound=-2.048,up_bound=2.048)
        print("Minimum  {} in point {}".format(fmin, sol))
        L.append(fmin)


    arr = np.array(L)
    print("Average fn minimum = {} +/- {}".format(np.mean(arr), np.std(arr)))
