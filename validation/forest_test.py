#!/bin/python3

# Validate all routines in forest class
# Mainly use results from 
# Strozzi, David J. 
# Vlasov simulations of kinetic enhancement of Raman backscatter in laser fusion plasmas. 
# Diss. Massachusetts Institute of Technology, 2005.

from plasmaforest.core import *
import scipy.constants as sc
from numpy import pi
import astropy.units as u
from plasmapy.formulary.parameters import thermal_speed
import warnings
from plasmapy.utils.exceptions import RelativityWarning

# Disable plasmapy relativity warning
warnings.filterwarnings("ignore", category=RelativityWarning)

# Setup case
ndim = 1 # 1D case
laser_lambda = 351.0e-9 # m
laser_kvac = 2*pi/laser_lambda
laser_omega = sc.c*laser_kvac
nc = sc.epsilon_0*sc.m_e/sc.e**2*laser_omega**2 # m/s
ne = 0.1*nc # plasma at 0.1 nc
Te = 3000 # eV
Te = temperature_energy(Te,'eVtoK')

# Get forest class instance and assert setup
birch = forest(Te,ne,ndim)
print('Temp (K): %0.1f; Density (1/m^3): %0.3e; ndim: %d' % (birch.Te,birch.ne,birch.ndim))
necheck = 9.049e26
Techeck = 3000/8.617333262145e-5
real_assert(birch.ne,necheck,1e23)
real_assert(birch.Te,Techeck,1)

# Calculate quantities and validate
# Electron thermal velocity
birch.get_vthe()
vthe_check = 22.97*1e-6*1e12
print('v_th,e (m/s): %0.3e' % (birch.vthe))
real_assert(birch.vthe,vthe_check,1e4)
print('All tests in forest_test.py complete.\n')
