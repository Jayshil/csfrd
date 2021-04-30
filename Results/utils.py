import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
from tqdm import tqdm
import os

def lam_to_nu(lam):
    """
    Function to compute frequency (in Hz)
    from Wavelength (in A)
    -----------------------
    Parameters:
    -----------
    lam : float
        wavelength in A
    -----------
    returns
    -----------
    float :
        frequency in Hz
    """
    lam1 = lam*10**(-10)
    freq = 299792458/lam1
    return freq

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
    expp = 10**(-0.4*(m+48.6))
    l1 = abc*expp
    return l1

def m_to_l_wave(m, lam):
    """
    To transform absolute magnitude to
    luminosity in cgs units
    -----------------------
    Parameters:
    -----------
    m : float, numpy.ndarray
        Absolute Magnitude
    lam : float
        Wavelength in A
    -----------
    returns:
    -----------
    l : float, numpy.ndarray
        Luminosity
    """
    d1 = 10*3.0857*10**18
    abc = 4*np.pi*(d1*d1)
    expp = 10**(-0.4*(m+48.6))
    l1 = abc*expp*lam_to_nu(lam)
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
    m1 = -2.5*np.log10(m2) - 48.6
    return m1

def log_err(para, err):
    """
    To calculate the log err
    ------------------------
    Parameters:
    -----------
    para : float, numpy.ndarray
        given parameter
    err : float, numpy.ndarray
        error in the given parameter
    -----------
    return
    -----------
    float, numpy.ndarray:
        log10 parameter and error in it
    """
    ab = err/para
    bc = 1/(np.log(10))
    return np.log10(para), ab*bc

#----------------------------------------------------------
#---------Different Schechter functions -------------------
#----------------------------------------------------------

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
    ab = phi1/np.abs(lum1)
    cd = (np.abs(lum/lum1))**alpha
    expp = np.exp(-np.abs(lum/lum1))
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

#---------------------------
# LF for IR luminosities
#---------------------------

def sandage(lums9, alp9, phi9, sig9, lst9):
    ab = 1 + (lums9/lst9)
    bc = np.log10(ab)
    cd = bc**2
    ef = -cd/(2*sig9*sig9)
    gh = np.exp(ef)
    xy = (lums9/lst9)**(1-alp9)
    return xy*phi9*gh

#-----------------------------------------------------------
#-------------- Calculating SFRD ---------------------------
#-----------------------------------------------------------

def lum_den(lum, lum1, phi1, alpha, limit=0.03, Auv=0.0):
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
    limit : float
        lower limit of the intensity
        as a function of L*
        default is 0.03 (from Madau&Dickinson)
    Auv : float
        dust correction (in mag)
        default is 0.0
    -----------
    return
    -----------
    float
        luminosity density
    """
    # Dust correction in luminosity
    l_uv = 10**(0.4*Auv)
    # To calculate rho(0.001L*)
    nor_lum = np.linspace(limit*lum1, np.max(lum), 10000)
    nor_sc1 = schechter(nor_lum, lum1=lum1, phi1=phi1, alpha=alpha)
    nor_sc = nor_lum*nor_sc1#/phi1
    rho_nor = (inte.simps(nor_sc, nor_lum))*l_uv
    return rho_nor

def sfrd_wo_err(lum, lum1, phi1, alpha, kappa, limit=0.03, Auv=0.0):
    """
    Function to calculate star formation rate density
    -------------------------------------------------
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
    kappa : float
        conversion factor
    limit : float
        lower limit of the intensity
        as a function of L*
        default is 0.03 (from Madau & Dickinson)
    Auv : float
        dust correction (in mag)
        default is 0.0
    -----------
    return
    -----------
    float
        star formation rate density
    """
    lum_den2 = lum_den(lum, lum1, phi1, alpha, limit, Auv)
    sfrd2 = kappa*lum_den2
    return sfrd2


def lum_den22(lum, lum1, lum1err, phi1, phi1err, alpha, alphaerr, limit=0.03):
    """
    Function to calculate luminosity density
    ----------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    phi1 : float
        normalisation constant
    phi1err : float
        Error in normalisation constant
    lum1 : float
        characteristic luminosity
        the 'knee' of the function
    lum1err : float
        Error in characteristic luminosity
    alpha : float
        the faint-end slope of power law
    alphaerr : float
        Error in the faint-end slope of power law
    limit : float
        lower limit of the intensity
        as a function of L*
        default is 0.03 (from Madau&Dickinson)s
    -----------
    return
    -----------
    numpy.ndarray :
        an array of luminosity density
    """
    # Values of Parameters
    lum2 = np.random.normal(lum1, lum1err, 10000)
    phi2 = np.random.normal(phi1, phi1err, 10000)
    alp2 = np.random.normal(alpha, alphaerr, 10000)
    # Use only certain precision
    """
    lum2 = np.around(lum22, 5)
    phi2 = np.around(phi22, 5)
    alp2 = np.around(alp22, 5)
    
    f1 = open(os.getcwd() + '/alp_' + str(alpha) + '_' + str(phi1) + '.dat', 'w')
    for i in range(len(alp2)):
        f1.write(str(alp2[i]) + '\t' + str(lum2[i]) + '\t' + str(phi2[i]) + '\n')
    f1.close()
    """
    # Values of luminosities
    nor_lum = np.linspace(limit*lum1, np.max(lum), 100000)
    # Integration array
    rho2 = np.array([])
    # Integration starts
    for i in tqdm(range(10000)):
        if lum2[i] < 0 :#alp2[i] != alp2[i] or lum2[i] != lum2[i] or lum2[i] == 0 or phi2[i] != phi2[i]:
            continue
        else:
            nor_sc1 = schechter(nor_lum, lum1=lum2[i], phi1=phi2[i], alpha=alp2[i])
            nor_sc = nor_lum*nor_sc1#/phi2[j]
            rho_nor = inte.simps(nor_sc, nor_lum)
            rho2 = np.hstack((rho2, rho_nor))
    #print("\nlength: ")
    #print(len(rho2))
    return rho2


def sfrd_w_err(lum, lum1, lum1err, phi1, phi1err, alpha, alphaerr, kappa, limit=0.03):
    """
    Function to calculate luminosity density
    ----------------------------------------
    Parameters:
    -----------
    lum : float, numpy.ndarray
        luminosity range
    phi1 : float
        normalisation constant
    phi1err : float
        Error in normalisation constant
    lum1 : float
        characteristic luminosity
        the 'knee' of the function
    lum1err : float
        Error in characteristic luminosity
    alpha : float
        the faint-end slope of power law
    alphaerr : float
        Error in the faint-end slope of power law
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
    lum_den2 = lum_den22(lum, lum1, lum1err, phi1, phi1err, alpha, alphaerr, limit)
    kpp1 = kappa
    sfr2 = kpp1*lum_den2
    return np.mean(sfr2), np.std(sfr2)


def asymmetric_gausian_distribution(mean, std1, std2, size):
    """
    To compute asymmetric gaussian distribution
    ------------------------------------------
    Parameters:
    -----------
    mean : float
        mean of the distribution
    std1 : float
        positive standard deviation of the distribution
    std2 : float
        negative standard deviation of the distribution
    size : int
        size of the distribution
    -----------
    return
    -----------
    numpy.ndarray
        random asymmetric distribution
    """
    d1 = np.random.normal(mean, std1, 2*size)
    d2 = np.random.normal(mean, std2, 2*size)
    up = d1[d1>mean][:int(size/2)]
    dn = d2[d2<mean][:int(size/2)]
    d3 = np.concatenate((dn, up))
    np.random.shuffle(d3)
    return d3

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """
    get_quantiles function
    ----------------------
    Imported from `exotoolbox`
    by Nestor Espinoza
    see, https://github.com/nespinoza/exotoolbox
    ---------------------------------------------
    DESCRIPTION
        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.
    OUTPUTS
        Median of the parameter,upper credibility bound, lower credibility bound
    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]