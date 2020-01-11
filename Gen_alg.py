# Name: Jordan Le
# Date: 1/7/20
# Description: Developing genetic algorithm to tune neural networks (for deep learning)
# Resouces:
# Collect Top K - https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

import numpy as np
import Model_arch as ma
#import tensorflow as tf
import matplotlib.pyplot as plt
import random as rand
import time

"""
Genetic algorithm class will manage network tuning.
"""
class Gen_alg:
    def __init__(self):
        self.pop_num = 100               # Population Number
        self.remaining = int(self.pop_num / 2)
        self.gen_num = 500                                   # Number of generations
        self.mutation_rate = .05                            # How likely are we to mutate
        self.mut_val = .50                                   # How much mutation
        self.chrom_num = 100                                 # How large is the DNA sequence?
        self.population = [DNA(id, self.chrom_num) for id in range(self.pop_num)]
        self.survivors = np.zeros(self.remaining)           # Survivors per generation
        self.func = np.random.choice(35, self.chrom_num)

    def fitness(self):                                      # Take the top half of the population of them
        function = self.func                     # Temp example to make sure this works
        pop_fits = np.zeros(self.pop_num, dtype=int)        # Holds all fitness values
        pop_ind = 0
        f2 = np.asarray(function)

        for pop in self.population:                         # Each individual in the population
            fit = np.where(f2 == pop.hidden_layers)         # Where they match
            fit_score = len(fit[0])
            pop.fit_vals.append(fit_score)      # Each population member has a list of their fitnesses
            pop_fits[pop_ind] = fit_score       # Update the pop's score for specific index
            pop_ind += 1                        # Increment index
            print("ID: {}\tFit Score: {}\tDNA: {}".format(pop.id, fit_score, pop.hidden_layers))

        # Merge these two together later
        surv_ind = np.argpartition(pop_fits, -self.remaining)[-self.remaining:]     # Index of the survivors
        print("Survivor indexes: {}\tSurv_fit_vals: {}".format(surv_ind, pop_fits[surv_ind]))

        term_ind = np.argpartition(pop_fits, self.remaining)[:self.remaining]     # Index of the terminated
        print("Terminated indexes: {}\tTerm_fit_vals: {}".format(term_ind, pop_fits[term_ind]))

        parents = self.pair_off(surv_ind)
        for i in range(len(parents)):
            #print("****** Old_child: {} ******".format(self.population[term_ind[i]].hidden_layers))
            self.cross_over(parents[i], term_ind[i])    # Parent pair and the ID they replace
            #print("****** New_child: {} ******".format(self.population[term_ind[i]].hidden_layers))
        print("Function: {}".format(function))


    def pair_off(self, surv_index):
        # Pair off the survivors
        par_v1 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v1
        par_v2 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v2
        parents = np.concatenate((par_v1, par_v2), axis=0)                                         # Full pop pairs
        #print("parv1: {}\nparv2: {}\npar_all: {}".format(par_v1, par_v2, parents))    # Verify
        return parents

    def cross_over(self, parents, ch_id):       # Recombine the existing survivors
        parent_1 = self.population[parents[0]].hidden_layers        # Initialize parent 1 np array
        parent_2 = self.population[parents[1]].hidden_layers        # Initialize parent 2 np array
        '''
        # Merge the survivors at some crossover point - randomly?
        split = int(np.random.uniform(low=1, high=len(parent_1) - 1))   # Choose random point between 1 and parent length
        child = np.hstack([parent_1[:split], parent_2[split:]])         # Take first chunk from parent 1, 2nd chunk parent 2
        
        #print("Parent_id's: = {}\nParents are: {}, {}\nChild_id: {}\nThe Child: {}".format(parents, parent_1, parent_2, ch_id, child))

        self.population[ch_id].hidden_layers = child    # Check this works outside...
        '''
        # Alternate crossover
        for i in range(len(parent_1)):
            self.population[ch_id].hidden_layers[i] = np.random.choice([parent_1[i], parent_2[i]]) # Choose one of these randomly
        child = self.population[ch_id].hidden_layers

        self.mutation(child, ch_id)
        self.population[ch_id].history.append([parents[0], parents[1]])     # Add parents to history
        #pass

    def mutation(self, child, ch_id):
        # if 1, then we mutate.
        if np.random.choice([0, 1], p=[1 - self.mutation_rate, self.mutation_rate]):

            num_cells = int(np.random.choice(len(child), 1) * self.mutation_rate)       # How many cells to replace, 5%
            mut_ind = np.random.choice(len(child), num_cells, replace=False)  # Choose from these indeces

            for ind in mut_ind:
                self.mut_val = rand.uniform(0, 1)  # Choose how random of a change, percentage
                operation = np.random.choice(['+', '-'])
                if operation == '+':    # Increase by 20%
                    self.population[ch_id].hidden_layers[ind] += int(
                        self.mut_val * self.population[ch_id].hidden_layers[ind])
                    #pass
                else:                   # Decrease by 20%
                    self.population[ch_id].hidden_layers[ind] -= int(
                        self.mut_val * self.population[ch_id].hidden_layers[ind])
                    #pass
            print("MUTATION: {}".format(self.population[ch_id].hidden_layers))


    def generation(self, images):       # Run the GA
        for i in range(self.gen_num):
            print("**** Gen: {} ****".format(i))
            self.fitness()

            #if (i + 1) % 2 == 0:  # display on evens
            graph_display(images, self.population, i, self.mutation_rate)       # Interactive gen Alg display
        #print("Fit Values: {}\nAnscestry: {}".format(self.population[0].fit_vals, self.population[0].history))


"""
Class for each individual in the population will be built off this information.
This information will later be scrambled with crossover to create new population.
"""
class DNA:
    def __init__(self, id, chrom_num):
        self.id = id                        # The populant's id number
        self.num_layers = chrom_num
        self.fit_vals = [0]                  # Accuracy scores to be added
        #self.input_layer = 15               # Same number for each input
        self.hidden_layers = [rand.randint(10, 35) for i in range(self.num_layers)]       # Set various hidden layers randomly
        #self.output_layer = 5               # Placeholder output layer
        self.history = [id]                 # List of all combinations


def graph_display(images, population, gen_num, mut_rate):
    max_fitness = population[0].fit_vals[ np.argmax(population[0].fit_vals) ]

    for i in range(len(images)):    # population length
        images[i].set_ydata(population[i].fit_vals)
        images[i].set_xdata(np.arange(len(population[i].fit_vals)))
        #img1.set_xlim()
        #print("Hello {}".format(i))

        ax = plt.gca()
        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()

        temp_max = population[i].fit_vals[ np.argmax(population[i].fit_vals) ]
        if max_fitness < temp_max:
            max_fitness = temp_max  # Update best score

    # Place the title of the plot with dynamic details
    fig.suptitle('Genetic Algorithm\n'
                 'Max Fitness: {}'
                 '      Generation: {}'
                 '      Pop Size: {}'
                 '      Mutation Rate: {}'.format(max_fitness, gen_num, len(population), mut_rate), fontsize=10)


    # Draw the plots and wait.
    plt.draw()
    plt.pause(1e-15)
    time.sleep(0.1)
    pass

"""
Where all the exciting stuff happens...
"""
if __name__ == "__main__":
    # Initialize Matplotlib figure
    fig = plt.figure()

    print("Genetic Algorithm to tune Neural Nets")
    GA = Gen_alg()
    print(GA.pop_num)
    images = []     # List of matplotlib elements
    for i in range(len(GA.population)):
        print("Pop_Id: {}\nLayers: {}\n".format(GA.population[i].id, GA.population[i].hidden_layers))

        # Initialize all the plots
        temp_img, = plt.plot(GA.population[i].fit_vals, np.arange(len(GA.population[i].fit_vals)))
        images.append(temp_img)


    GA.generation(images)

    # Keep the image around
    plt.show()