"""
Author:
	Baher Kher Bek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.stats import linregress


def func(x, a, b, c):
    return (a / (x-b)) + c

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

disparities = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/disparities.npy')
alpha = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/alpha.npy')
popt = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/popt.npy')
InvDisparities = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/InverseDisparities.npy')


x = np.linspace(min(disparities), max(disparities), 90)
Y = func(x, *popt)

plt.subplot(1, 2, 1)
plt.title('Disparity V/S alpha Coefficient', fontdict=font)
plt.xlabel('Disparity', fontdict=font)
plt.ylabel('alpha', fontdict=font)
plt.plot(disparities, alpha, 'o-')
plt.plot(x, Y, 'r')
plt.legend(["Data Collected", "Regression"], loc ="upper right")


plt.subplot(1, 2, 2)
linearfit = linregress(InvDisparities, alpha)
Intercept, slope = linearfit.intercept, linearfit.slope
plt.title('Inverse Disparity V/S alpha Coefficient', fontdict=font)
plt.xlabel('1 / Disparity', fontdict=font)
plt.ylabel('alpha', fontdict=font)
plt.plot(InvDisparities, alpha, 'o-')
plt.plot(InvDisparities, slope*InvDisparities + Intercept, 'r')
plt.legend(["Data Collected", "Regression"], loc ="upper left")
plt.savefig('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/Regression.jpg')
plt.show()
