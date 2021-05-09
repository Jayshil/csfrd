import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import astropy.constants as con
import astropy.units as u
from tqdm import tqdm
import os
import utils as utl
import irlf as irlf

# LF Parameters
zdo = np.array([0.5, 1.5, 2.5, 3.5])
zup = np.array([1.5, 2.5, 3.5, 4.5])
zcen = (zdo + zup)/2

alp, alp_err = np.array([-0.4, -0.4, -0.4, -0.4]), np.zeros(len(zcen))
logl, logl_err = np.array([25.20, 25.40, 25.63, 25.84]), np.array([0.085, 0.03, 0.05, 0.15])
logp, logp_err = np.array([-2.88, -3.03, -3.73, -4.59]), np.array([0.30, 0.075, 0.145, 0.295])

# Lower limit of integration
limit1 = 0.01


# Defining Kappa and the range of luminosities over which we want to perform integration
kap_ir = 4.5*10**(-44)
lums_ir1 = np.logspace(24,27,100)*utl.lam_to_nu(2500000)*1e7
print(lums_ir1)

# Location of the results file
p2 = os.getcwd() + '/Results/'