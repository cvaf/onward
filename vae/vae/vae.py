import os, sys
sys.path.append(os.getcwd())

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model

# Basic data
img_shape = (28, 28, 1)
batch_size = 16
# latent_dim 
