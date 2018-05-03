class Connection:
  ''' Defines a connection in a neural network. Contains a sender node, a recipient node, and a weight. '''
  def __init__(self, sender, recipient, weight):
    self.sender = sender
    self.recipient = recipient
    self.weight = weight
    self.delta = 0.0

  # learning equation for training 
  def update_deltas(self, learning_rate, mu=0.7):
    self.delta += (learning_rate * self.recipient.error * self.sender.activation) + (0.7 * self.delta)
