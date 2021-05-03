import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
from tqdm import tqdm
import os
import utils as utl

def sandage(lums9, alp9, phi9, sig9, lst9):
    """
    IR Luminosity function
    ----------------------
    Parameters:
    -----------
    lums9 : numpy.ndarray
        range of luminosities
    alp, phi9, sig9, lst9 : float
        function parameters
    -----------
    return
    -----------
    float :
        number of galaxies in the give range
    """
    ab = 1 + (lums9/lst9)
    bc = np.log10(ab)
    cd = bc**2
    ef = -cd/(2*sig9*sig9)
    gh = np.exp(ef)
    xy = (lums9/lst9)**(1-alp9)
    return xy*phi9*gh

