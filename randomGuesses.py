import random
from sklearn.metrics import confusion_matrix

def random_guesses(test_data):
  random.seed(1)
  classifications = []
  for _ in xrange(len(test_data)):
      classifications.append(random.randint(0, 9))

  # result analysis
  num_correct_classifications = 0.0
  for i in xrange(len(classifications)):
    if classifications[i] == test_data[i]:
      num_correct_classifications += 1
  accuracy = num_correct_classifications / float(len(test_data))

  # confusion matrix
  cm = confusion_matrix(test_data, classifications)
  return accuracy, cm