# Name: Jordan Le
# Date: 1/7/20
# Description: Developing genetic algorithm to tune neural networks (for deep learning)
# Resouces:
# Collect Top K - https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

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

        # Merge these two together later
        surv_ind = np.argpartition(pop_fits, -self.remaining)[-self.remaining:]     # Index of the survivors
        print("Survivor indexes: {}\tSurv_fit_vals: {}".format(surv_ind, pop_fits[surv_ind]))

        term_ind = np.argpartition(pop_fits, self.remaining)[:self.remaining]     # Index of the terminated
        print("Terminated indexes: {}\tTerm_fit_vals: {}".format(term_ind, pop_fits[term_ind]))

        parents = self.pair_off(surv_ind)
        for i in range(len(parents)):
            print("****** Old_child: {} ******".format(self.population[term_ind[i]].hidden_layers))
            self.cross_over(parents[i], term_ind[i])    # Parent pair and the ID they replace
            print("****** New_child: {} ******".format(self.population[term_ind[i]].hidden_layers))


    def pair_off(self, surv_index):
        # Pair off the survivors
        par_v1 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v1
        par_v2 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v2
        parents = np.concatenate((par_v1, par_v2), axis=0)                                         # Full pop pairs
        print("parv1: {}\nparv2: {}\npar_all: {}".format(par_v1, par_v2, parents))    # Verify
        return parents

    def cross_over(self, parents, ch_id):       # Recombine the existing survivors
        parent_1 = self.population[parents[0]].hidden_layers        # Initialize parent 1 np array
        parent_2 = self.population[parents[1]].hidden_layers        # Initialize parent 2 np array

        # Merge the survivors at some crossover point - randomly?
        split = int(np.random.uniform(low=1, high=len(parent_1) - 1))   # Choose random point between 1 and parent length
        child = np.hstack([parent_1[:split], parent_2[split:]])         # Take first chunk from parent 1, 2nd chunk parent 2

        print("Parent_id's: = {}\nParents are: {}, {}\nChild_id: {}\nThe Child: {}".format(parents, parent_1, parent_2, ch_id, child))

        self.population[ch_id].hidden_layers = child    # Check this works outside...
        #pass

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
        self.history = [id]                 # List of all combinations



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