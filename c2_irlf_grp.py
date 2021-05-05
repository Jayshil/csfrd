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
zcen = (zdo + zup)/2

alp, alp_err = np.array([1.22, 1.15, 1.08, 1.25, 1.28]), np.array([0.16, 0.145, 0.14, 0.49, 0.47])
logl, logl_err = np.array([11.95, 12.01, 12.12, 11.90, 12.16]), np.array([0.385, 0.395, 0.22, 0.54, 0.805])
logp, logp_err = np.array([-3.44, -3.45, -3.32, -3.43, -3.73]), np.array([0.235, 0.185, 0.145, 0.445, 0.50])
sig, sig_err = 0.5*np.ones(len(zcen)), np.zeros(len(zcen))

# Lower limit of integration
limit1 = 1e8*((con.L_sun.to(u.erg/u.s)).value)


# Defining Kappa and the range of luminosities over which we want to perform integration
kap_ir = 4.5*10**(-44)
lums_ir1 = np.logspace(10, 13, 10000)*(con.L_sun.to(u.erg/u.s).value)

# Location of the results file
p2 = os.getcwd() + '/Results/'

def lum_den22(lum, lst9, lst9err, phi9, phi9err, sig9, sig9err, alp9, alp9err, limit):
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
    # For L*
    lst7 = np.random.normal(lst9, lst9err, 10000)
    lst2 = (10**lst7)*((con.L_sun.to(u.erg/u.s)).value)
    print('\nL*')
    print(np.mean(lst2))
    print(np.std(lst2))
    phi7 = np.random.normal(phi9, phi9err, 10000)
    phi2 = 10**phi7
    print('\nphi*')
    print(np.mean(phi2))
    print(np.std(phi2))
    # For alpha and sigma
    alp2 = np.random.normal(alp9, alp9err, 10000)
    sig2 = np.random.normal(sig9, sig9err, 10000)
    # Values of luminosities
    nor_lum = np.logspace(np.log10(limit), np.max(np.log10(lum)), 100000)
    # Integration array
    rho2 = np.array([])
    # Integration starts
    for i in tqdm(range(10000)):
        if lst2[i] < 0 :#alp2[i] != alp2[i] or lum2[i] != lum2[i] or lum2[i] == 0 or phi2[i] != phi2[i]:
            continue
        else:
            #nor_lum = np.logspace(np.log10(limit*lst9), np.max(np.log10(lum)), 100000)
            nor_sc1 = irlf.sandage(lums9=nor_lum, alp9=alp2[i], phi9=phi2[i], sig9=sig2[i], lst9=lst2[i])
            nor_sc = nor_lum*nor_sc1#/phi2[j]
            rho_nor = inte.simps(y=nor_sc, x=np.log10(nor_lum))
            rho2 = np.hstack((rho2, rho_nor))
    #print("\nlength: ")
    #print(len(rho2))
    #print(np.mean(rho2))
    return rho2

def sfrd_w_err(lum, lst9, lst9err, phi9, phi9err, sig9, sig9err, alp9, alp9err, kappa, limit):
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

"""
sfrd_ir, sfrd_err_ir = sfrd_w_err(lum=lums_ir1, lst9=logl[0], lst9err=logl_err[0], \
        phi9=logp[0], phi9err=logp_err[0], sig9=sig[0], sig9err=sig_err[0], alp9=alp[0], \
        alp9err=alp_err[0], kappa=kap_ir, limit=limit1)

print(sfrd_ir)
print(sfrd_err_ir)

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

# Performing the integration
f33 = open(p2 + 'sfrd_grp_new.dat','w')
f33.write('#Name_of_the_paper\tZ_down\tZ_up\tSFRD\tSFRD_err\n')


for j in range(len(zcen)):
    sfrd_ir, sfrd_err_ir = sfrd_w_err(lum=lums_ir1, lst9=logl[j], lst9err=logl_err[j], \
        phi9=logp[j], phi9err=logp_err[j], sig9=sig[j], sig9err=sig_err[j], alp9=alp[j], \
        alp9err=alp_err[j], kappa=kap_ir, limit=limit1)
    f33.write('Gruppioni_et_al_2020' + '\t' + str(zdo[j]) + '\t' + str(zup[j]) + '\t' + str(sfrd_ir) + '\t' + str(sfrd_err_ir) + '\n')

f33.close()