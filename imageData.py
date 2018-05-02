import idx2numpy
import numpy as np

class ImageData:

  def __init__(self, image_file, label_file):
    self.images = idx2numpy.convert_from_file(image_file)
    self.labels = idx2numpy.convert_from_file(label_file)

    # append bias node
    images = []
    for i in self.images:
      images.append(np.append(np.ndarray.flatten(i), 1))
    self.images = images

