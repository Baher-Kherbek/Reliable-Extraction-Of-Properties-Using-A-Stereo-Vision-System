"""
Author:
	Baher Kher Bek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np

def func(x, a, b, c):
    return (a / (x-b)) + c

#Curve fitting
disparities = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/disparities.npy')
alpha = np.load('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/alpha.npy')
popt, pcov = curve_fit(func, disparities, alpha, method='dogbox')
np.save('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/popt.npy', popt)
np.save('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/pcov.npy', pcov)
