import numpy as np
import matplotlib.pyplot as plt
import corner

means_mpa = np.array([-20.93, -2.7721132953863266, -1.69])

cov_mpa = np.array([[0.00468796, 0.00329079, 0.00243851],\
     [0.00329079, 0.00297998, 0.00122257],\
     [0.00243851, 0.00122257, 0.00165734]]) 

samples = np.random.multivariate_normal(means_mpa, cov_mpa, 10000)

lbs = np.array(['M*', 'logPhi*', 'alpha'])

corner.corner(samples, labels=lbs, truths=means_mpa)
plt.show()

mean_4 = np.array([-20.88, -2.705533773838407, -1.64])
err_4 = np.array([0.08, 0.015228426395939087, 0.04])

cov_4 = np.array([[0.00686333, 0.00441759, 0.00248485],\
    [0.00441759, 0.00497807, 0.00171691],\
    [0.00248485, 0.00171691, 0.00160949]])

a01 = cov_mpa[0][1]/np.sqrt(cov_mpa[0][0]*cov_mpa[1][1])
a02 = cov_mpa[0][2]/np.sqrt(cov_mpa[0][0]*cov_mpa[2][2])
a12 = cov_mpa[1][2]/np.sqrt(cov_mpa[1][1]*cov_mpa[2][2])

print(a01)
print(a02)
print(a12)

cov_41 = np.array([[err_4[0]**2, a01*err_4[0]*err_4[1], a02*err_4[0]*err_4[2]],\
    [a01*err_4[0]*err_4[1], err_4[1]**2, a12*err_4[1]*err_4[2]],\
    [a02*err_4[0]*err_4[2], a12*err_4[1]*err_4[2], err_4[2]**2]])

print(cov_41)
print('-----')
print(cov_4)

samples1 = np.random.multivariate_normal(mean_4, cov_41, 10000)
corner.corner(samples1, labels=lbs, truths=mean_4)
plt.show()

samples2 = np.random.multivariate_normal(mean_4, cov_4, 10000)
corner.corner(samples2, labels=lbs, truths=mean_4)
plt.show()