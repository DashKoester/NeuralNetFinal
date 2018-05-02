''' Implementation of a neural network that classifies
    handwritten digits from the MNIST database. Completed 
    for the Final Project for PSYC 489 at UIUC'''
'__author__' == 'Dashiell Koester'

from imageData import ImageData
import randomGuesses
import perceptron
import backProp
# import numpy as np
# np.set_printoptions(linewidth=125)
# np.set_printoptions(threshold=np.nan)


# Create image daata objects
training_data = ImageData('train-images/train-images.idx3-ubyte', 'train-images/train-labels.idx1-ubyte')
test_data = ImageData('test-images/test-images.idx3-ubyte', 'test-images/test-labels.idx1-ubyte')


# # classify using a random guess
# random_accuracy, random_cm = randomGuesses.random_guesses(test_data.labels)
# print random_accuracy
# print random_cm

# # classify using 10 trained perceptrons
# perceptron_accuracy, perceptron_cm = perceptron.run(training_data,  test_data)
# print perceptron_accuracy
# print perceptron_cm

# classify using a deep neural network
backProp.run(training_data, test_data)


