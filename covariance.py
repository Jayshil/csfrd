import numpy as np
import matplotlib.pyplot as plt
import posdef as pdf
import utils as utl

# Data of covariance matrices
def covz(z):
    """
    Parameters:
    -----------
    z : int
        redshift
    -----------
    return
        covariance matrix
    """
    if z == 2:
        cov_21 = np.array( [[0.00366342, 0.00162317, 0.00252753],\
            [0.00162317, 0.00138304, 0.00097829],\
            [0.00252753, 0.00097829, 0.00117065]])
        cov_z = pdf.nearestPD(cov_21)
    elif z == 3:
        cov_z = np.array([[0.00796234, 0.00422674, 0.00167071],\
            [0.00422674, 0.00283603, 0.00109828],\
            [0.00167071, 0.00109828, 0.00079994]])
    elif z == 4:
        cov_z = np.array([[0.00468796, 0.00329079, 0.00243851],\
            [0.00329079, 0.00297998, 0.00122257],\
            [0.00243851, 0.00122257, 0.00165734]])
    elif z == 5:
        cov_z = np.array([[0.01074832, 0.00780709, 0.00449567],\
            [0.00780709, 0.00640997, 0.00383279],\
            [0.00449567, 0.00383279, 0.00287709]])
    elif z == 6:
        cov_z = np.array([[0.0078469, 0.00737499, 0.00544656],\
            [0.00737499, 0.00767651, 0.00608668],\
            [0.00544656, 0.00608668, 0.00539578]])
    elif z == 7:
        cov_z = np.array([[0.01384952, 0.01546666, 0.01230685],\
            [0.01546666, 0.01874599, 0.01491299],\
            [0.01230685, 0.01491299, 0.01211733]])
    elif z == 8:
        cov_81 = np.array([[0.08437214, 0.08840825, 0.0602166 ],\
            [0.08840825, 0.09631012, 0.08780118],\
            [0.0602166, 0.08780118, 0.04688886]])
        cov_z = pdf.nearestPD(cov_81)
    return cov_z

def covar(z, err):
    """
    Function to find covariance matrix
    at a given redshift
    ----------------------------------
    Parameters:
    -----------
    z : float
        redshift of the target
    err : numpy.ndarray
        array of errors in parameters
        parameters - M*, log phi*, alpha
    -----------
    return
    -----------
    numpy.ndarray :
        covariance matrix
        parameters - M*, log phi*, alpha
    """
    zz = np.around(z)
    covz1 = covz(zz)
    # Correlation matrix
    corr = np.zeros((3,3))
    for i in range(len(3)):
        for j in range(len(3)):
            corr[i][j] = covz1[i][j] / np.sqrt(covz1[i][i] * covz1[j][j])
    # Covariance matrix
    cov_new = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            cov_new[i][j] = corr[i][j] * err[i] * err[j]
    return cov_new


def corr_para(z, means, err, size=10000):
    """
    To generate correlated samples
    ------------------------------
    Parameters
    ----------
    z : float, int
        redshift of the object
    means : numpy.ndarray
        means of the parameters (M*, logphi, alpha)
    err : numpy.ndarray
        uncertainties in the parameters (M*, logphi, alpha)
    size : numpy.ndarray
        size of the returned array
    ----------
    returns
    ----------
    numpy.ndarray:
        correlated samples of L*
    numpy.ndarray:
        correlated samples of phi*
    numpy.ndarray:
        correlated samples of alpha
    """
    covs = covar(z, err)
    samples1 = np.random.multivariate_normal(means, covs, size=size)
    lum_z = utl.m_to_l_wave(samples1.T[0], 1500)
    phi_z = 10**samples1.T[1]
    alp_z = samples1.T[2]
    return lum_z, phi_z, alp_z