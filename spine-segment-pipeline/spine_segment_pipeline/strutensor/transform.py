""" Radon Transform as described in Birkfellner, Wolfgang. Applied Medical Image Processing: A Basic Course. [p. 344] """

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R

