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
kap_ir = 0.68*4.5*10**(-44)
lums_ir1 = np.logspace(24,27,100)*utl.lam_to_nu(2500000)*1e7
#print(lums_ir1)

# Location of the results file
p2 = os.getcwd() + '/Results/'

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
    # For L*
    lum9 = np.random.normal(lum1, lum1err, 10000)
    lum2 = (10**lum9)*utl.lam_to_nu(2500000)*1e7
    # For phi*
    phi9 = np.random.normal(phi1, phi1err, 10000)
    phi2 = 10**phi9
    # Alpha
    alp2 = np.random.normal(alpha, alphaerr, 10000)
    # Use only certain precision
    # Values of luminosities
    nor_lum = np.linspace(limit*np.mean(lum2), np.max(lum), 100000)
    # Integration array
    rho2 = np.array([])
    # Integration starts
    for i in tqdm(range(10000)):
        if lum2[i] < 0 :#alp2[i] != alp2[i] or lum2[i] != lum2[i] or lum2[i] == 0 or phi2[i] != phi2[i]:
            continue
        else:
            nor_sc1 = utl.schechter(nor_lum, lum1=lum2[i], phi1=phi2[i], alpha=alp2[i])
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

"""
# Without errors
for i in range(len(zcen)):
    sam = np.logspace(np.log10(limit1), np.max(np.log10(lums_ir1)), 100000)
    lf = irlf.sandage(lums9=sam, alp9=alp[i], phi9=10**logp[i], sig9=sig[i], lst9=(10**logl[i])*(con.L_sun.to(u.erg/u.s).value))
    nor = sam*lf
    rho = inte.simps(y=nor, x=np.log10(sam))
    sfrd = rho*kap_ir
    print('For redshift: ', zcen[i])
    print('SFRD: ', sfrd)
    print('log(SFRD): ', np.log10(sfrd))
"""

# Performing the integration
f33 = open(p2 + 'sfrd_kop_new.dat','w')
f33.write('#Name_of_the_paper\tZ_down\tZ_up\tSFRD\tSFRD_err\n')

for j in range(len(zcen)):
    sfrd_ir, sfrd_err_ir = sfrd_w_err(lum=lums_ir1, lum1=logl[j], lum1err=logl_err[j],\
         phi1=logp[j], phi1err=logp_err[j], alpha=alp[j], alphaerr=alp_err[j], kappa=kap_ir, limit=0.01)
    f33.write('Koprowski_et_al_2017' + '\t' + str(zdo[j]) + '\t' + str(zup[j]) + '\t' + str(sfrd_ir) + '\t' + str(sfrd_err_ir) + '\n')

f33.close()