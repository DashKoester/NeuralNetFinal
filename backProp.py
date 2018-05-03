#!/usr/bin/env python
''' Back-Propagation Neural Network implementation for PSYC489 Final Project '''
__author__ = 'Dashiell Koester'
import numpy as np

from unit import Unit
from connection import Connection


class Backprop_Network:
  ''' Implementation of Back-Propagation Network '''

  def __init__(self):
    self.layers = []
    self.expected_outputs = [[0.02 for _ in xrange(10)] for i in xrange(10)]
    for i in xrange(10):
      self.expected_outputs[i][i] = 0.98
  
  def initialize_network(self, num_inputs, num_outputs, num_hidden_layers = 1):
    # input layer
    self.layers.append([Unit() for _ in xrange(num_inputs)])

    # hidden layers
    ''' using the mean number of nodes in input/output layer to compute
        number of nodes in the hidden layer '''
    for i in xrange(num_hidden_layers):
      self.layers.append([Unit() for _ in xrange((num_inputs + num_outputs) / 2)])
    
    # output layer
    self.layers.append([Unit() for _ in xrange(num_outputs)])

    # connect the layers
    for i in xrange(len(self.layers) - 1):
      for sender in self.layers[i]:
        for recipient in self.layers[i+1]:
          sender.add_connection(recipient, np.random.rand()) # initialize weights randomly
          


  def forward_propagate(self):
    ''' Propagates activations forward through the hidden layers'''

    output =  []

    for layer in self.layers[1:]:
        for unit in layer:
          unit.get_input()
          # print unit.input,
        for unit in layer:
          unit.sigmoid_activation()
          # print unit.activation,
          # get final activations
          if layer is self.layers[-1]:
            output.append(unit.activation)
    
    return output

  
  def backwards_propagate(self, expected_output):
    ''' Propagates errors backwards through the hidden layers '''
    for i, layer in reversed(list(enumerate(self.layers))):

      # hidden layer errors
      if layer is not self.layers[-1]:
        for hidden_unit in layer:
          hidden_unit.calculate_hidden_error()
          
      # output layer errors
      else:
        for j, output_unit in enumerate(layer):
          # print expected_output[j]
          output_unit.calculate_output_error(expected_output[j])
          # print output_unit.error
      
      # update connection deltas
        for unit in layer:
          for connection in unit.outgoing_connections:
            connection.update_deltas(0.2) # using learning rate of 0.2

  # update the connection weights
  def learn(self, learning_rate):
    for layer in self.layers:
      for unit in layer:
        for connection in unit.outgoing_connections:
          connection.weight += connection.delta

      

  # train the network on the training image data
  def train(self, training_image_data, learning_rate, num_epochs):
    # reshape training data
    images = training_image_data.images[:60]
    labels = training_image_data.labels
    # train for given number of epochs
    for epoch in xrange(num_epochs):
      print 'Training epoch %s:' % str(epoch + 1)
      global_error = 0.0
      for i, image in enumerate(images):

        # normalize the data to [0,255]
        image = image / 255.0

        # initialize inputs
        for j in xrange(len(image)):
          self.layers[0][j].activation = image[j]

        # forward propagate activations
        fp_output = self.forward_propagate()

        # expected output
        correct_class = labels[i]
        expected = self.expected_outputs[correct_class]
        
        # backwards propagate error
        self.backwards_propagate(expected)

        # get global error
        for output_unit in self.layers[-1]:
          # print output_unit.error
          global_error += (output_unit.error)
      # print global_error
      global_error = global_error**2 / (epoch+1)
        

      # update the weights
      self.learn(learning_rate)
      print('>epoch=%d, error=%.3f') % (epoch, global_error)
    

  # test the network on the testing image data
  def test(self, test_image_data):
    print 'Testing Network...'

    






def run(training_image_data, test_image_data):
  
  # reshape image data
  training_images = training_image_data.images
  training_labels = training_image_data.labels
  test_images = test_image_data.images
  test_labels = test_image_data.labels

  # get number of features and classes
  num_inputs = len(training_images[0])
  num_outputs = len(set(training_labels))

  # initialize network
  one_layer_network = Backprop_Network()
  one_layer_network.initialize_network(num_inputs, num_outputs)
  # train for 20 epochs with learning rate of  0.2
  one_layer_network.train(training_image_data, 0.2, 1)

  # test the network on the testing data
  # one_layer_network.test(test_image_data)

