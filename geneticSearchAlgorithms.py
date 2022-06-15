# %load geneticSearchAlgorithms.py
from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices 
import math 
import random
import time
from itertools import islice 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
from matplotlib import pyplot as plt 

class GASolver: 
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size 
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        self.pop = {}
        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')
        
    def make_initial_pop(self):
        while len(self.pop) < self.N:
            randExp = generate_random_expr(self.params.depth, self.identifiers, self.params)
            if(is_viable_expr(randExp, self.identifiers, self.params)):
                # add random expression to pop dictionary and initialize fitness
                randExpFitness = compute_fitness(randExp, self.identifiers, self.params) 
                self.pop.update({randExp : randExpFitness})
        

                  
#     Please add whatever helper functions you wish.

#     TODO: Implement the genetic algorithm as described in the
#     project instructions.
#     This function need not return anything. However, it should
#     update the fields best_solution_so_far, best_fitness_so_far and
#     population_stats
               
    def run_ga_iterations(self, n_iter=1000):
        # Start time
        start = time.time()
        
        
        # create initial population
        self.make_initial_pop()
        
        # sort population
        self.pop = sorted(self.pop.items(), key=lambda item: item[1], reverse=True)
        
        k = int(self.params.elitism_fraction * self.N)
        
        temp = self.params.temperature
        
        gen_number = 0
        
        while(gen_number < n_iter):
            if(gen_number % 10 == 0):
                print("Generation: ", gen_number)
            
            # Take the top k = self.params.elitism_fraction * self.N
            # and put them in the next generation as elites
            
            next_gen = list(islice(self.pop, k))
            self.pop = dict(self.pop)

            # For remaining N-k expression moving on to the next generation,
            # implement Mutation/Crossover

            
            current_gen_list = list(self.pop.keys())
            current_gen_fitness = list(self.pop.values())
            current_gen_weights = []
            
            # Generates list of weights for current gen
            for i in range(len(current_gen_list)):
                current_gen_weights.append(math.exp((current_gen_fitness[i]/temp)))

            # Loop until we have N-k new expressions
            count = self.N - k
            while(count > 0):
                # Picks random e1, e2 according to weights
                (e1, e2) = random.choices(current_gen_list, weights=current_gen_weights, k=2)

                # Crossover
                (e1_cross, e2_cross) = random_subtree_crossover(e1, e2, copy=True)

                # Mutate e1_cross until viable and add to new generation
                viable1 = False
                while(not viable1):
                    e1_mutation = random_expression_mutation(e1_cross, self.identifiers, self.params, copy=True)
                    if(is_viable_expr(e1_mutation, self.identifiers, self.params)):
                        viable1 = True
                next_gen.append(e1_mutation)
                count = count - 1

                # Mutate e2_cross until viable and add to new generation
                viable2 = False
                while(not viable2):
                    e2_mutation = random_expression_mutation(e2_cross, self.identifiers, self.params, copy=True)
                    if(is_viable_expr(e2_mutation, self.identifiers, self.params)):
                        viable2 = True
                next_gen.append(e2_mutation)
                count = count - 1

            # Put new generation in dictionary with fitness
            next_gen_dict = {}
            for i in next_gen:
                fitness = compute_fitness(i, self.identifiers, self.params) 
                next_gen_dict.update({i : fitness})

            # Sort new generation
            next_gen = sorted(next_gen_dict.items(), key=lambda item: item[1], reverse=True)
            next_gen = dict(next_gen)

            # Update best_solution_so_far and best_fitness_so_far
            best_fit = list(next_gen.values())[0]
            best_solu = list(next_gen.keys())[0] 
            if(best_fit > self.best_fitness_so_far):
                self.best_fitness_so_far = best_fit
                self.best_solution_so_far = best_solu

            # Update the population/add to population_stats
            self.pop = next_gen
            self.population_stats.append(self.best_fitness_so_far)
            
            gen_number = gen_number + 1
        
        # End time
        end = time.time()
        
        # Elapsed time
        elapsed = end-start
        print("Run Time:", elapsed)
            
        
            




## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)


# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
       ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
    solver = GASolver(params,['x'],500)
    solver.run_ga_iterations(100)
    print('Done!')
    print(f'Best solution found: {solver.best_solution_so_far.simplify()}, fitness = {solver.best_fitness_so_far}')
    stats = solver.population_stats
    niters = len(stats)
    plt.plot(range(niters), [st[0] for st in stats] , 'b-')
    plt.show()



