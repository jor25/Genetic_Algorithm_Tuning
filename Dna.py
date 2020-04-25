# Model DNA, contains the model architecture information of each individual.
# Modifications made here are pushed to the model architecture later on.

from configs import *
import random as rand

"""
Class for each individual in the population will be built off this information.
This information will later be scrambled with crossover to create new population.
"""
class DNA:
    def __init__(self, id, chrom_num):
        self.id = id                        # The individual's id number
        self.num_layers = chrom_num         # Number of layers
        self.fit_vals = [0]                 # Accuracy scores to be added
        self.input_layer = 5                # Same number for each input
        self.hidden_layers = [rand.randint(10, 35) for i in range(self.num_layers)]       # Set various hidden layers randomly
        self.output_layer = NUM_CLASSES     # Placeholder output layer
        self.history = [id]                 # List of all parent combinations through generations