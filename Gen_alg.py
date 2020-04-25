# Name: Jordan Le
# Date: 1/7/20
# Description: Developing genetic algorithm to tune neural networks (for deep learning)
# Resouces:
# Collect Top K - https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
# Matplotlib plots - https://matplotlib.org/tutorials/introductory/pyplot.html
# Rescale Matplotlib - https://stackoverflow.com/questions/10984085/automatically-rescale-ylim-and-xlim-in-matplotlib
# Horizontal stack with numpy - https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
# Concatinate numpy - https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
# Random choice numpy - https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html

import numpy as np
import Model_arch as ma
import matplotlib.pyplot as plt
import time
import Dna
from configs import *

"""
Genetic algorithm class will manage network tuning.
"""
class Gen_alg:
    def __init__(self):
        self.pop_num = POP_SIZE                             # Population Number
        self.remaining = int(self.pop_num * .50)            # Number of individuals who survive per generation
        self.removed = self.pop_num - self.remaining        # Number of individuals that need to be remade
        self.gen_num = GEN_NUM                              # Number of generations
        self.mutation_rate = .25                            # How likely are we to mutate
        self.mut_val = .50                                  # How much mutation
        self.chrom_num = 4                                  # How large is the DNA sequence?
        self.potential_mutations = int(self.chrom_num * .5)  # Take 50% of the genes for potential mutation
        self.survivors = np.zeros(self.remaining)           # Survivors per generation
        self.func = np.random.choice(35, self.chrom_num)

        # Initialize the DNA Sequence
        self.pop_dna = [Dna.DNA(id, self.chrom_num) for id in range(self.pop_num)]

        # Initialize datasets - x, y, y_train, x2, y2, y_test - Load the dataset
        self.x_train, self.y_one_hot, self.y_train, self.x_test, self.y2_one_hot, self.y_test = ma.load_dataset()

        # Initialize the models
        if USING_MODELS:
            self.model_pop = [ma.model(self.pop_dna[i]) for i in range(self.pop_num)]

    def fitness(self):                                      # Take the top half of the population of them
        pop_fits = np.zeros(self.pop_num, dtype=int)        # Holds all fitness values

        # USING MODELS
        if USING_MODELS:
            for i in range(self.pop_num):   # Only do training if it hasn't been trained yet - only allow 1 epoch
                if not self.model_pop[i].trained:
                    self.model_pop[i].training(self.x_train, self.y_one_hot)
                accuracy, predictions = self.model_pop[i].calc_fitness(self.x_test, self.y_test)  # Check how I do on training data
                print("Predictions len: {}\nAccuracy: {}%".format(len(predictions), accuracy * 100))
                self.pop_dna[i].fit_vals.append(accuracy)   # Accuracy is how I'm calculating fitness
                pop_fits[i] = accuracy

        # Testing without the models
        else:
            function = self.func  # Temp example to make sure this works
            pop_ind = 0
            f2 = np.asarray(function)
            # Replace the fitness function here
            for pop in self.pop_dna:                         # Each individual in the population
                fit = np.where(f2 == pop.hidden_layers)         # Where they match
                fit_score = len(fit[0])
                pop.fit_vals.append(fit_score)      # Each population member has a list of their fitnesses
                pop_fits[pop_ind] = fit_score       # Update the pop's score for specific index
                pop_ind += 1                        # Increment index
                print("ID: {}\tFit Score: {}\tDNA: {}".format(pop.id, fit_score, pop.hidden_layers))


        # Merge these two together later
        surv_ind = np.argpartition(pop_fits, -self.remaining)[-self.remaining:]     # Index of the survivors

        if self.pop_num == 1:
            term_ind = np.argpartition(pop_fits, self.remaining)[:self.remaining]     # Index of the terminated
        else:
            term_ind = np.argpartition(pop_fits, self.removed)[:self.removed]  # Index of the terminated

        # DEBUG
        print("Survivor indexes: {}\tSurv_fit_vals: {}".format(surv_ind, pop_fits[surv_ind]))
        print("Terminated indexes: {}\tTerm_fit_vals: {}".format(term_ind, pop_fits[term_ind]))

        parents = self.pair_off(surv_ind)
        print("Parents: {}\tParent Shape: {}".format(parents, parents.shape))
        for i in range(len(term_ind)):
            #print("****** Old_child: {} ******".format(self.pop_dna[term_ind[i]].hidden_layers))
            self.cross_over(parents[i % len(parents)], term_ind[i])    # Parent pair and the ID they replace
            #print("****** New_child: {} ******".format(self.pop_dna[term_ind[i]].hidden_layers))
        #print("Function: {}".format(function))


    def pair_off(self, surv_index):
        '''
        Pair off the indexes of the survivors. Randomly select parents to breed with. May need to check for matching.
        Or just assume single parent can repopulate on its own like a frog.
        :param surv_index: All the indexes of the top percentage of individuals
        :return: 2d numpy array of pairs of parents
        '''
        par_v1 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v1
        par_v2 = np.random.choice(surv_index, size=(int(self.remaining/2), 2), replace=False)         # Pair off v2
        parents = np.concatenate((par_v1, par_v2), axis=0)                                         # Full pop pairs
        #print("parv1: {}\nparv2: {}\npar_all: {}".format(par_v1, par_v2, parents))    # Verify
        return parents

    def cross_over(self, parents, ch_id):       # Recombine the existing survivors
        parent_1 = self.pop_dna[parents[0]].hidden_layers        # Initialize parent 1 np array
        parent_2 = self.pop_dna[parents[1]].hidden_layers        # Initialize parent 2 np array
        print("Parent1: {}\nParent2: {}".format(parent_1, parent_2))

        num_crosses = np.random.randint(low=1, high=int((len(parent_1)+1) / 2))     # How many times am I swapping
        #splits = np.random.choice(len(parent_1) - 1, num_crosses, replace=False)    # All the places to split the list
        #parts = np.trim_zeros(np.sort(splits))  # Sort and remove Zero from start
        parts = np.sort( np.random.choice( np.trim_zeros(np.arange(len(parent_1))), num_crosses, replace=False) )
        print("\tnum_crosses: {}\n\tparts: {}".format(num_crosses, parts))

        print("Before Pop[{}]: {}".format(ch_id, self.pop_dna[ch_id].hidden_layers))

        for i, part in enumerate(parts):
            if np.random.choice([1, 2]):  # Select a random parent as parent x
                parent_x = parent_1
                parent_y = parent_2
            else:
                parent_x = parent_2
                parent_y = parent_1

            if i == 0:  # On the first index
                self.pop_dna[ch_id].hidden_layers[0:part] = parent_x[0:part]  # Load the first chunk from parent X
            else:  # All our other cases
                self.pop_dna[ch_id].hidden_layers[parts[i - 1]:part] = parent_x[parts[i - 1]:part]  # Load last point to latest point

            if i == len(parts) - 1:  # First and last index are the same
                self.pop_dna[ch_id].hidden_layers[parts[i]:len(parent_y)] = parent_y[parts[i]:len(parent_y)]

        print("After Pop[{}]: {}".format(ch_id, self.pop_dna[ch_id].hidden_layers))

        self.mutation(ch_id, self.pop_dna[ch_id].hidden_layers)
        self.pop_dna[ch_id].history.append([parents[0], parents[1]])     # Add parents to history

        # Update the models HERE!
        if USING_MODELS:
            self.model_pop[ch_id] = ma.model(self.pop_dna[ch_id])    # Create a new model at this index

    def mutation(self, ch_id, child):
        '''
        Function loops through the child's genes and selects a percentage with a chance to mutate it by some amount.
        Directly modifies the child variable through argument.
        :param ch_id: Integer index of a specific dino brain.
        :param brains: The actual neural network used by the dinos
        :return: N/A
        '''

        # All of the potential indexes to mutate at
        mutation_indexes = np.random.choice(len(child) - 1, self.potential_mutations, replace=False)
        mutated = False
        for i in mutation_indexes:
            # if 1, then we mutate. Random chance of mutation
            if np.random.choice([0, 1], p=[1 - self.mutation_rate, self.mutation_rate]):    # Percent chance of mutation
                child[i] += np.random.choice(np.arange(-5, 5, step=1))                      # Adjust number of layers in weight
                mutated = True

        if mutated:
            print("{}'s MUTATION: {}".format(ch_id, self.pop_dna[ch_id].hidden_layers))  # DEBUG


    def generation(self, images):#, model_pop):       # Run the GA
        for i in range(self.gen_num):
            print("**** Gen: {} ****".format(i))

            # Train the models before calling the fitness on them, this will evaluate how well they did on the dataset.

            self.fitness()

            #if (i + 1) % 2 == 0:  # display on evens

            graph_display(images, self.pop_dna, i, self.mutation_rate)       # Interactive gen Alg display
        #graph_display(images, self.pop_dna, i, self.mutation_rate)           # Not Interactive gen Alg display
        #print("Fit Values: {}\nAnscestry: {}".format(self.pop_dna[0].fit_vals, self.pop_dna[0].history))


def graph_display(images, population, gen_num, mut_rate):
    max_fitness = population[0].fit_vals[ np.argmax(population[0].fit_vals) ]
    pop_len = len(images) #int(len(images)*.05 + 1)   # for speed up - show only 5% of population
    for i in range(pop_len):    # population length
        images[i].set_ydata(population[i].fit_vals)
        images[i].set_xdata(np.arange(len(population[i].fit_vals)))

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


"""
Where all the exciting stuff happens...
"""
if __name__ == "__main__":
    # Initialize Matplotlib figure
    fig = plt.figure()

    print("Genetic Algorithm to tune Neural Nets")
    GA = Gen_alg()      # Initialize Genetic Algorithm object
    print(GA.pop_num)

    images = []     # List of matplotlib elements
    for i in range(len(GA.pop_dna)):
        print("Pop_Id: {}\nLayers: {}\n".format(GA.pop_dna[i].id, GA.pop_dna[i].hidden_layers))

        # Initialize all the plots
        temp_img, = plt.plot(GA.pop_dna[i].fit_vals, np.arange(len(GA.pop_dna[i].fit_vals)))
        images.append(temp_img)

    # Run Generation
    GA.generation(images)

    # Keep the image around
    plt.show()