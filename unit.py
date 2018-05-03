from connection import Connection
import math

class Unit:
  ''' Defines a node in a neural network.'''
  def __init__(self):
    self.input = 0.0
    self.activation = 0.0
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
    for connection in self.incoming_connections:
      total_input += connection.sender.activation * connection.weight
    self.input = total_input + 1 

  # Sigmoid activation function
  def sigmoid_activation(self):
    if math.fabs(self.input) > 1: 
      self.input = self.input / len(self.incoming_connections) # normalize values
    self.activation = 1 / (1 + math.exp(-self.input))

  # Leaky ReLU activation function
  def leaky_relu(self):
    self.activation = self.input if self.input > 0 else 0.01*self.input

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
