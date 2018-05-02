#!/usr/bin/env python
''' Back-Propagation Neural Network implementation for PSYC489 Final Project '''
__author__ = 'Dashiell Koester'
import numpy as np
import math

class Connection:
  ''' Defines a connection in a neural network. Contains a sender node, a recipient node, and a weight. '''
  def __init__(self, sender, recipient, weight):
    self.sender = sender
    self.recipient = recipient
    self.weight = weight
    self.delta = 0.0

  # learning equation for training 
  def update_deltas(self, learning_rate, mu=0.7):
    self.delta += (learning_rate * self.sender.error * self.recipient.activation) + (0.7 * self.delta)




class Unit:
  ''' Defines a node in a neural network.'''
  def __init__(self):
    self.input = 0.0
    self.activation = 0.0
    # self.inhibitory = 0.0
    # self.excitatory = 0.0
    self.error = 0.0
    self.incoming_connections = []
    self.outgoing_connections = []

  # adds a connection with random weight to another unit
  def add_connection(self, recipient, weight):
    newConnection = Connection(self, recipient, weight)
    self.outgoing_connections.append(newConnection)
    recipient.incoming_connections.append(newConnection)

  # gets the total input from all incoming connections
  def get_input(self):
    total_input = 0.0
    for i in self.incoming_connections:
      total_input += i.sender.activation * i.weight
    self.input = total_input

  # sigmoid activation function
  # returns true if activation was updated
  def sigmoid_activation(self):
    original_activation = self.activation
    self.activation = 1 / (1 + math.exp(-self.input))
    return self.activation != original_activation

  # calculates the error for a node in a hidden layer
  def calculate_hidden_error(self):
    sum_error = 0.0
    for connection in self.outgoing_connections:
      sum_error += (connection.recipient.error * connection.weight)
    self.error = sum_error * self.activation * (1 - self.activation)

    # calculates the error for a node in the output layer
  def calculate_output_error(self, expected_value):
    error = expected_value - self.activation
    self.error = error * self.activation * (1 - self.activation)


class Backprop_Network:
  ''' Implementation of Back-Propagation Network '''

  def __init__(self):
    self.layers = []
    self.expected_outputs = [[0 for _ in xrange(10)] for i in xrange(10)]
    for i in xrange(10):
      self.expected_outputs[i][i] = 1.0
  
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

        # don't add connection to bias node in hidden layers
        if i != len(self.layers) - 2:
          for recipient in self.layers[i+1][:-1]:
            sender.add_connection(recipient, np.random.rand())

        # completely connect output layer
        else:
          for recipient in self.layers[i+1]:
            sender.add_connection(recipient, np.random.rand())

    # setup activation for bias nodes in hidden layers
    for hidden_layer in self.layers[1:-1]:
      hidden_layer[-1].activation = 1.0


  def forward_propagate(self):
    ''' Propagates activations forward through the hidden layers'''

    for layer in self.layers[1:]:
      # we don't update activation for bias nodes
      if layer is not self.layers[-1]:
        # update activation synchronously
        for unit in layer[:-1]:
          unit.get_input()
        for unit in layer[:-1]:
          unit.sigmoid_activation()
          # print unit.activation

      # output layer has no bias node, so update all values
      else:
        for unit in layer:
          unit.get_input()
          print unit.input
        for unit in layer:
          unit.sigmoid_activation()
    
    # get final activations
    output =  []
    for unit in self.layers[-1]:
      output.append(unit.activation)
    
    return output

  
  def backwards_propagate(self, expected_output):
    ''' Propagates errors backwards through the hidden layers '''
    for i, layer in reversed(list(enumerate(self.layers))):

      # hidden layer errors
      if layer is not self.layers[-1]:
        for hidden_unit in layer:
          # calculate error for each hidden unit
          hidden_unit.calculate_hidden_error()
          

      # output layer errors
      else:
        for j, output_unit in enumerate(layer):
          output_unit.calculate_output_error(expected_output[j])
          # print output_unit.error
      
    # update connection deltas
    for layer in self.layers:
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
    images = training_image_data.images[:10]
    labels = training_image_data.labels
    # train for given number of epochs
    for epoch in xrange(num_epochs):
      print 'Training epoch %d:' % epoch
      global_error = 0.0
      for i, image in enumerate(images):

        # initialize inputs
        for j in xrange(len(image)):
          self.layers[0][j].activation = image[j]

        # forward propagate activations
        fp_output = self.forward_propagate()
        print fp_output

        # expected output
        correct_class = labels[i]
        expected = self.expected_outputs[correct_class]
        
        # backwards propagate error
        self.backwards_propagate(expected)

        # get global error
        for output_unit in self.layers[-1]:
          global_error += (output_unit.error)**2 

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

