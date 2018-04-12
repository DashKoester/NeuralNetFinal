#!usr/bin/env python
__author__ = 'Dashiell Koester and Joey Weng'
''' MP3 for CS440/ECE448 at UIUC: Perceptron Implementation'''

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
# np.set_printoptions(threshold=np.nan)

class Perceptron:
  
  def __init__(self, class_label, bias = .5):
    self.class_label = class_label
    self.weights = np.append(np.random.rand(784), bias)
    # self.weights = np.append(np.zeros(784), bias)
    
    
  def train(self, image_data, eta, epochs):
    training_images = image_data.images
    training_image_labels = image_data.labels
     
    # iterate through each training image
    for i, image in enumerate(training_images):
    
      # check if current image should be in this perceptron's class
      if training_image_labels[i] == self.class_label:
        curr_label = 1.0
      else:
        curr_label = -1.0
      
      # run perceptron through training epochs
      for _ in xrange(epochs):

        # classify with current weights
        y = np.sign(np.dot(image, self.weights))

        # update dem weights
        if y != curr_label:
          for i in xrange(len(self.weights)):
            self.weights[i] += (eta * curr_label * image[i])

    return self.weights

  # Returns confidence
  def test(self, test_image):
    return np.dot(test_image, self.weights)


def find_missclassifications(perceptrons, image_data):
  images = image_data.images
  image_labels = image_data.labels

  # list of image classifications
  classifications = predict(perceptrons, images)

  num_missclassifications = 0
  for i in xrange(len(classifications)):
    if classifications[i] != image_labels[i]:
      num_missclassifications += 1
  return num_missclassifications

def predict(perceptrons, images):
  classifications = []
  # run through all the test images
  for i, test_image in enumerate(images):
    classification = -1
    best_confidence = -100000

    # find the best guess for each test image
    for j in xrange(10):
      confidence = perceptrons[j].test(test_image)
      if confidence > best_confidence:
        best_confidence = confidence
        classification = j
    classifications.append(classification)
    
  return classifications


def run_perceptron_test(training_image_data, test_image_data):

  # create the perceptrons
  perceptrons = []
  for i in xrange(10):
    perceptrons.append(Perceptron(i, 1))
    perceptrons[i].train(training_image_data, .2, 20)

  # plot the training curve (commented out to avoid unnecessary computaion/file writes)
  # eta = 0.1
  # epochs = 1
  # missclassifications = []
  # for epoch in xrange(50):
  #   for i in xrange(10):
  #     perceptrons[i].train(training_image_data, eta, epochs)
  #   missclassifications.append(find_missclassifications(perceptrons, training_image_data))
  # plt.plot(xrange(1, len(missclassifications)+1), missclassifications, marker='o')
  # plt.xlabel('Iterations')
  # plt.ylabel('Misclassifications')
  # plt.savefig('training_curve(0.1Random).png')

  # set up test data
  test_images = test_image_data.images
  test_image_labels = test_image_data.labels

  # list of image classifications
  classifications = predict(perceptrons, test_images)
  print classifications

  # result analysis
  num_correct_classifications = 0.0
  for i in xrange(len(classifications)):
    if classifications[i] == test_image_labels[i]:
      num_correct_classifications += 1
  accuracy = num_correct_classifications / float(len(test_image_labels))

  # confusion matrix
  cm = confusion_matrix(test_image_labels, classifications)
  
  return accuracy, cm

