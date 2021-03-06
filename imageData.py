import idx2numpy
import numpy as np

class ImageData:

  def __init__(self, image_file, label_file, with_bias=False):
    self.images = idx2numpy.convert_from_file(image_file)
    self.labels = idx2numpy.convert_from_file(label_file)

    images = []
    for i in self.images:
      # append bias node
      if with_bias == True:
        images.append(np.append(np.ndarray.flatten(i), 1))
      else:
        images.append(np.ndarray.flatten(i))
    self.images = images
    

