# Name: Jordan Le
# Date: 1/7/20
# Description: Model architecture file for a neural network class.

import numpy as np
from operator import add
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
import keras.losses as kl
from sklearn.metrics import accuracy_score
from keras.datasets import fashion_mnist
from configs import *


"""
Model class will be the backbone for all the individuals in the population.
"""
class model:
    def __init__(self, dna):
        self.mod_id = dna.id
        self.learning_rate = 0.001
        self.input_layer = dna.input_layer
        self.hidden_layers = dna.hidden_layers
        self.output_layer = dna.output_layer
        self.model = self.create_network()  # No initial weights
        self.trained = False    # Flag to allow training
        #self.model = self.create_network("weight_files/nn_3.hdf5")  # Using my trained weights

    def create_network(self, weights=None):
        # Create my model
        model = Sequential()

        # Set up Input layer
        #model.add(Dense(output_dim=self.hidden_layers[0], activation='relu', input_dim=self.input_layer))
        model.add(Conv2D(20, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Flatten())

        # Set up hidden layers
        for i in range(1, len(self.hidden_layers)):
            model.add(Dense(output_dim=self.hidden_layers[i], activation='relu'))           # Add all hidden layers

        # Set up output layer
        model.add(Dense(output_dim=self.output_layer, activation='softmax'))                # Output layer
        opt = Adam(self.learning_rate)      # Set learning rate

        # Compile model
        model.compile(loss=kl.categorical_crossentropy, metrics=['accuracy'], optimizer=opt)

        # Load weights if they're available
        if weights:
            model.load_weights(weights)
            print("model loaded")
        else:
            print("model[{}] created.".format(self.mod_id))

        return model

    def training(self, data, labels):
        # Conduct training after model architectures have been established
        self.model.fit(data, labels, batch_size=128,
                     epochs=1,
                     validation_split=0.2)
        self.trained = True     # Flag is now on

    def calc_fitness(self, data, labels):
        one_hot_predictions = self.model.predict(data)
        #print("onehot_pred: {}\none_hot_shape: {}".format(one_hot_predictions, one_hot_predictions.shape))
        predictions = [np.argmax(predict) for predict in one_hot_predictions]   # Get the value from predictions
        #print("pred: {}\npred_len: {}".format(predictions, len(predictions)))
        return accuracy_score(labels, predictions), predictions     # Accuracy and predicted values


def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # Load the dataset
    x_train, x_test = pre_process_data(x_train, x_test)     # Preprocess x

    x, y = data_prep(x_train, y_train, NUM_CLASSES, IMG_ROWS, IMG_COLS)
    x2, y2 = data_prep(x_test, y_test, NUM_CLASSES, IMG_ROWS, IMG_COLS)


    return x, y, y_train, x2, y2, y_test

def pre_process_data(x_train, x_test):
    # Handles data processing
    # add empty color dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(x_train.shape[1:])    # DEBUG
    return x_train, x_test

def data_prep(data, labels, num_classes, img_rows, img_cols):
    out_y = keras.utils.to_categorical(labels, num_classes)     # Onehots my data

    num_images = data.shape[0]
    x_shaped_array = data.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y
