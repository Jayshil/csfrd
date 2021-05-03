import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import astropy.constants as con
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
    float, numpy.ndarray:
        number of galaxies in the give range
    """
    ab = 1 + (lums9/lst9)
    bc = np.log10(ab)
    cd = bc**2
    ef = -cd/(2*sig9*sig9)
    gh = np.exp(ef)
    xy = (lums9/lst9)**(1-alp9)
    return xy*phi9*gh

def lum_den22(lum, lst9, lst9err, phi9, phi9err, sig9, sig9err, alp9, alp9err, limit=0.03):
    """
    Function to calculate luminosity density
    ----------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    lst9, phi9, sig9, alp9 : float
        LF parameters
    lst9err, phi9err, sig9err, alp9err : float
        errors in LF parameters
    limit : float
        lower limit of the intensity
        as a function of L*
        default is 0.03 (from Madau&Dickinson)
    -----------
    return
    -----------
    numpy.ndarray :
        an array of luminosity density
    """
    # Values of Parameters
    lst2 = np.random.normal(lst9, lst9err, 10000)
    phi2 = np.random.normal(phi9, phi9err, 10000)
    alp2 = np.random.normal(alp9, alp9err, 10000)
    sig2 = np.random.normal(sig9, sig9err, 10000)
    # Values of luminosities
    nor_lum = np.logspace(np.log10(limit*lst9), np.max(np.log10(lum)), 100000)
    # Integration array
    rho2 = np.array([])
    # Integration starts
    for i in tqdm(range(10000)):
        if lst2[i] < 0 :#alp2[i] != alp2[i] or lum2[i] != lum2[i] or lum2[i] == 0 or phi2[i] != phi2[i]:
            continue
        else:
            #nor_lum = np.logspace(np.log10(limit*lst9), np.max(np.log10(lum)), 100000)
            nor_sc1 = sandage(lums9=nor_lum, alp9=alp2[i], phi9=phi2[i], sig9=sig2[i], lst9=lst2[i])
            nor_sc = nor_lum*nor_sc1#/phi2[j]
            rho_nor = inte.simps(y=nor_sc, x=np.log10(nor_lum))
            rho2 = np.hstack((rho2, rho_nor))
    #print("\nlength: ")
    #print(len(rho2))
    #print(np.mean(rho2))
    return rho2

def sfrd_w_err(lum, lst9, lst9err, phi9, phi9err, sig9, sig9err, alp9, alp9err, kappa, limit=0.03):
    """
    Function to calculate star formation rate density
    -------------------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    lst9, phi9, sig9, alp9 : float
        LF parameters
    lst9err, phi9err, sig9err, alp9err : float
        errors in LF parameters
    kappa : float
        conversion factor b/w luminosity density and
        star formation rate
    limit : float
        lower limit of the intensity
        as a function of L*
        default is 0.03 (from Madau&Dickinson)
    -----------
    return
    -----------
    float
        mean star formation rate
    float
        error in star formation rate
    """
    lum_den2 = lum_den22(lum, lst9, lst9err, phi9, phi9err, sig9, sig9err, alp9, alp9err, limit)
    kpp1 = kappa
    sfr2 = kpp1*lum_den2
    return np.mean(sfr2), np.std(sfr2)