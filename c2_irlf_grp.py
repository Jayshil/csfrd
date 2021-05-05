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
zdo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
zup = np.array([1.5, 2.5, 3.5, 4.5, 6.0])

alp, alp_err = np.array([1.22, 1.15, 1.08, 1.25, 1.28]), np.array([0.16, 0.145, 0.14, 0.49, 0.47])
logl, logl_err = np.array([11.95, 12.01, 12.12, 11.90, 12.16]), np.array([0.385, 0.395, 0.22, 0.54, 0.805])
logp, logp_err = np.array([-3.44, -3.45, -3.32, -3.43, -3.73]), np.array([0.235, 0.185, 0.145, 0.445, 0.50])

# Lower limit of integration
limit = 1e8*((con.L_sun.to(u.erg/u.s)).value)

