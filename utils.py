import numpy as np
import scipy.integrate as inte

def m_to_l(m):
    """
    To transform absolute magnitude to
    luminosity in cgs units
    -----------------------
    Parameters:
    -----------
    m : float, numpy.ndarray
        Absolute Magnitude
    -----------
    returns:
    -----------
    l : float, numpy.ndarray
        Luminosity
    """
    d1 = 10*3.0857*10**18
    abc = 4*np.pi*(d1*d1)
    expp = 10**(-0.4*(m-48.6))
    l1 = abc*expp
    return l1

def l_to_m(l):
    """
    To transform luminosity to absolute magnitude
    ---------------------------------------------
    Parameters:
    -----------
    l : float, numpy.ndarray
        luminosity
    -----------
    returns
    -----------
    m : float, numpy.adarray
        Absolute magnitude
    -----------
    """
    d1 = 10*3.0857*10**18
    m2 = l/(4*np.pi*d1*d1)
    m1 = -2.5*np.log10(m2) + 48.6
    return m1



def schechter(lum, phi1, lum1, alpha):
    """
    The Schechter Function
    ----------------------
    Paramters:
    ----------
    lum : float, numpy.ndarray
        input luminosities of the galaxies
    phi1 : float
        normalisation constant
    lum1 : float
        characteristic luminosity
        the 'knee' of the function
    alpha : float
        the faint-end slope of power-law function
    ----------
    returns:
    ----------
    float or numpy.ndarray
        number of galaxies in given luminosity range
    """
    ab = phi1/lum1
    cd = (lum/lum1)**alpha
    expp = np.exp(-(lum/lum1))
    xy = ab*cd*expp
    return xy

def schechter_mag(M, phi1, m1, alpha):
    """
    The Schechter Function
    as described above.
    -------------------
    Parameters:
    -----------
    M : float, or numpy.ndarray
        absolute magnitude of the galaxies
    phi1 : float
        normalisation constant
    m1 : float
        the characteristic absolute magnitude
    alpha : float
        the faint-end slope of power-law function
    -----------
    returns
    -----------
    float or numpy.ndarray
        number of galaxies in given absolute magnitude range
    """
    m2 = 0.4*(m1-M)
    ab = 0.921*phi1
    cd = 10**(m2*(alpha+1))
    ef = np.exp(-10**m2)
    xxy = ab*cd*ef
    return xxy

def log_schechter(lum, lum1, phi1, alpha):
    """
    The Normalised logarithmic Schechter Function
    ---------------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    phi1 : float
        normalisation constant
    lum1 : float
        characteristic luminosity
        the 'knee' of the function
    alpha : float
        the faint-end slope of power law
    -----------
    return
    -----------
    float, numpy.ndarray
        number of galaxies in given bin
    """
    logg = np.log10(lum) - np.log10(lum1)
    ab = np.log(10)*phi1
    bc = 10**((alpha+1)*logg)
    cd = np.exp(-10**logg)
    return ab*bc*cd

def lum_den(lum, lum1, phi1, alpha):
    """
    Function to calculate luminosity density
    ----------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    phi1 : float
        normalisation constant
    lum1 : float
        characteristic luminosity
        the 'knee' of the function
    alpha : float
        the faint-end slope of power law
    -----------
    return
    -----------
    float
        luminosity density
    """
    # To calculate rho(0.001L*)
    nor_lum = np.linspace(0.001*lum1, np.max(lum), 10000)
    nor_sc1 = schechter(nor_lum, lum1=lum1, phi1=phi1, alpha=alpha)
    nor_sc = nor_lum*nor_sc1/phi1
    rho_nor = inte.simps(nor_sc, nor_lum)
    return rho_nor