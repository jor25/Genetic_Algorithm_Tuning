# Name: Jordan Le
# Date: 1/7/20
# Description: Developing genetic algorithm to tune neural networks (for deep learning)

import numpy as np
import Model_arch as ma
#import tensorflow as tf
import matplotlib
import random as rand

"""
Genetic algorithm class will manage network tuning.
"""
class Gen_alg:
    def __init__(self):
        self.pop_num = 12               # Population Number
        self.remaining = int(self.pop_num / 2)
        self.gen_num = 5                                    # Number of generations
        self.mutation_rate = .05
        self.population = [DNA(id) for id in range(self.pop_num)]
        self.survivors = np.zeros(self.remaining)           # Survivors per generation

    def fitness(self):      # Take the top half of the population of them
        function = [10, 20, 30, 20, 10]                     # Temp example to make sure this works
        pop_fits = np.zeros(self.pop_num, dtype=int)        # Holds all fitness values
        pop_ind = 0
        f2 = np.asarray(function)

        for pop in self.population:                         # Each individual in the population
            fit = np.where(f2 == pop.hidden_layers)         # Where they match
            fit_score = len(fit[0])
            pop.fit_vals.append(fit_score)      # Each population member has a list of their fitnesses
            pop_fits[pop_ind] = fit_score       # Update the pop's score for specific index
            pop_ind += 1                        # Increment index
            print("ID: {}\tFit Score: {}".format(pop.id, fit_score))

        surv_ind = np.argpartition(pop_fits, -self.remaining)[-self.remaining:]     # Index of the survivors
        print("Survivor indexes: {}\tSurv_fit_vals: {}".format(surv_ind, pop_fits[surv_ind]))



    def cross_over(self):       # Recombine the existing survivors
        pass

    def generation(self):       # Run the GA
        for i in range(self.gen_num):
            print("Gen: {}".format(i))
            self.fitness()


"""
Class for each individual in the population will be built off this information.
This information will later be scrambled with crossover to create new population.
"""
class DNA:
    def __init__(self, id):
        self.id = id                        # The populant's id number
        self.fit_vals = []                  # Accuracy scores to be added
        self.input_layer = 15               # Same number for each input
        self.hidden_layers = [rand.randint(10, 35) for i in range(5)]       # Set various hidden layers randomly
        self.output_layer = 5               # Placeholder output layer



"""
Where all the exciting stuff happens...
"""
if __name__ == "__main__":
    print("Genetic Algorithm to tune Neural Nets")
    GA = Gen_alg()
    print(GA.pop_num)
    for i in range(len(GA.population)):
        print("Pop_Id: {}\nLayers: {}\n".format(GA.population[i].id, GA.population[i].hidden_layers))

    GA.generation()