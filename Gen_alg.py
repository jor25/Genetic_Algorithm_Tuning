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
        self.pop_num = 10
        self.mutation_rate = .05
        self.population = [DNA(id) for id in range(self.pop_num)]

    def fitness(self):      # Take the top half of the population of them
        pass

    def cross_over(self):       # Recombine the existing
        pass


"""
Class for each individual in the population will be built off this information.
This information will later be scrambled with crossover to create new population.
"""
class DNA:
    def __init__(self, id):
        self.id = id                        # The populant's id number
        self.fit_vals = []                  # Accuracy scores to be added
        self.input_layer = 15               # Same number for each input
        self.hidden_layers = [rand.randint(1, 100) for i in range(3)]       # Set various hidden layers randomly
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