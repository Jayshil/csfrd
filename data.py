import numpy as np
import os

try:
    os.mkdir(os.getcwd() + '/data/')
    p2 = os.getcwd() + '/data/'
except:
    p2 = os.getcwd() + '/data/'

b = True

while b:
    paper = input('Enter the name of the paper: ')
    f1 = open(p2 + 'data_' + paper.replace(' ','') + '.dat', 'w')
    f1.write('#Redshift\t M* \t M*_err\t Phi* \t Phi*_err\t Alpha\t Alpha_err + \t + NameOfPaper + \n')
    a = True
    while a:
        z = input('Enter redshift: ')
        z_err = input('Enter the error in redshift: ')
        mag = input('Enter charactersitic abs magnitude: ')
        mag_err = input('Enter error in M: ')
        phi = input('Enter phi: ')
        phi_err = input('Enter error in phi: ')
        alp = input('Enter alpha: ')
        alp_err = input('Enter the error in alpha: ')
        f1.write(z + '\t' + z_err + '\t' + mag + '\t' + mag_err + '\t' + phi + '\t' + phi_err + '\t' + alp + '\t' + alp_err + '\t' + paper.replace(' ','') + '\n')
        efg = input('Same paper? (/NO): ')
        if efg == 'NO':
            a = False
    f1.close()
    abc = input('Do you want to continue? (/NO): ')
    if abc == 'NO':
        b = False