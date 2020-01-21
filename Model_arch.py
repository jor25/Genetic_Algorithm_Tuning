# Name: Jordan Le
# Date: 1/7/20
# Description: Model architecture file for a neural network class.

import Gen_alg as GA
import numpy as np
from operator import add
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import keras.losses as kl

"""
Model class will be the backbone for all the individuals in the population.
"""
class model:
    def __init__(self, dna):
        self.learning_rate = 0.001
        self.model = self.create_network()                        # No initial weights
        #self.model = self.create_network("weight_files/nn_3.hdf5")  # Using my trained weights

    def create_network(self, weights=None):
        # Create my model
        model = Sequential()
        model.add(Dense(output_dim=dna.hidden_layers[0], activation='relu', input_dim=50))    # Input
        for i in range(1, len(dna.hidden_layers)-1):
            model.add(Dense(output_dim=dna.hidden_layers[i], activation='relu'))
        model.add(Dense(output_dim=dna.hidden_layers[-1], activation='softmax'))               # Output
        opt = Adam(self.learning_rate)

        # Compile model
        model.compile(loss=kl.categorical_crossentropy, metrics=['accuracy'], optimizer=opt)

        # Load weights if they're available
        if weights:
            model.load_weights(weights)
            print("model loaded")
        return model