''' Implementation of a neural network that classifies
    handwritten digits from the MNIST database. Completed 
    for the Final Project for PSYC 489 at UIUC'''
'__author__' == 'Dashiell Koester'

from imageData import ImageData
import perceptron
import numpy as np
np.set_printoptions(linewidth=125)
np.set_printoptions(threshold=np.nan)

# @TODO: Figure out why it's classifying some digits as -1  (it's not surpassing the  -10000 confidence threshold??)


# Create image daata objects
training_data = ImageData('train-images/train-images.idx3-ubyte', 'train-images/train-labels.idx1-ubyte')
test_data = ImageData('test-images/test-images.idx3-ubyte', 'test-images/test-labels.idx1-ubyte')

print test_data.labels

# accuracy, confusionMatrix = perceptron.run_perceptron_test(training_data,  test_data)

# print accuracy
# print confusionMatrix
