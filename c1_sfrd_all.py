import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as con
import utils as utl
import covariance as cov
import os

# Defining kappa
sol_lum = (con.L_sun*1e7).value
kap_uv = 2.2e-10/sol_lum

# Range of Luminosities (or absolute magnitudes) used
mags_all = np.linspace(-24, -13, 10)
lums_all = utl.m_to_l_wave(mags_all, 1500)

# Location of new data
p1 = os.getcwd() + '/data/New_UV/'

# To save the results
p22 = os.getcwd() + '/Results/Diff_lim'
f22 = open(p22 + 'sfrd_uv_new035.dat', 'w')
f22.write('#Name_of_the_paper\tZ_up\tZ_down\tSFRD\n')

# List of data files
list_uv = os.listdir(p1)

plt.figure(figsize=(16,9))

for i in range(len(list_uv)):
    z1_uv, z2_uv, mst_uv, msterr_uv, phi_uv, phierr_uv, alp_uv, alperr_uv = np.loadtxt(p1 + list_uv[i], usecols=(0,1,2,3,4,5,6,7), unpack=True)
    ppr_n = np.loadtxt(p1 + list_uv[i], usecols=8, dtype=str, unpack=True)
    #
    # This is because some of the data file has only one rows
    # and numpy read them as numpy.float64 object, not as numpy.ndarray
    #
    if type(mst_uv) == np.float64:
        lngth = 1
        z1_uv, z2_uv, mst_uv, msterr_uv, phi_uv, phierr_uv, alp_uv, alperr_uv, ppr_n\
             = np.array([z1_uv]), np.array([z2_uv]), np.array([mst_uv]), np.array([msterr_uv]),\
               np.array([phi_uv]), np.array([phierr_uv]), np.array([alp_uv]), np.array([alperr_uv]), np.array([ppr_n])
    else:
        lngth = len(mst_uv)
    #
    print('-------------------------------------------------------------')
    print('Working on: ' + ppr_n[0])
    print('-------------------------------------------------------------')
    #
    # Calculating SFRD
    #
    sfrd_uv = np.zeros(len(z1_uv))
    sfrd_uv_err = np.zeros(len(z1_uv))
    for j in range(len(z1_uv)):
        # Computing parameters array
        logphi, logphi_err = utl.log_err(phi_uv[j], phierr_uv[j])
        mean_all = np.array([mst_uv[j], logphi, alp_uv[j]])
        err_all = np.array([msterr_uv[j], logphi_err, alperr_uv[j]])
        zcen = (z1_uv[j] + z2_uv[j])/2
        #lst11 = utl.m_to_l_wave(mean_all[0], 1500)
        lt1 = 0.35/kap_uv
        sfr2, sfr2e = cov.sfrd_w_err(lum=lums_all, z=zcen, mean2=mean_all, err2=err_all, kappa=kap_uv, limit=lt1)
        sfrd_uv[j], sfrd_uv_err[j] = sfr2, sfr2e
        f22.write(ppr_n[0] + '\t' + str(z1_uv[j]) + '\t' + str(z2_uv[j]) + '\t' + str(sfr2) + '\t' + str(sfr2e) + '\n')
    #
    # log sfrd and error in it
    log_sfr_uv, log_sfr_uv_err = utl.log_err(sfrd_uv, sfrd_uv_err)
    #
    # Plotting the results
    zcen1 = (z1_uv + z2_uv)/2
    zup, zdown = np.abs(z1_uv - zcen1), np.abs(zcen1-z2_uv)
    plt.errorbar(x=zcen1, xerr=[zup, zdown], y=log_sfr_uv, yerr= log_sfr_uv_err, label=ppr_n[0], fmt='.')


f22.close()

plt.xlabel('Redshift')
plt.ylabel(r'SFRD (in $M_\odot year^{-1} Mpc^{-3}$')
plt.grid()
plt.legend(loc='best')
plt.show()