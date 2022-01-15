#!/bin/python3

# Validate all routines in forest class
# Mainly use results from:
# Strozzi, David J. 
# Vlasov simulations of kinetic enhancement of 
# Raman backscatter in laser fusion plasmas. 
# Diss. Massachusetts Institute of Technology, 2005.

from plasmaforest.core import *
import scipy.constants as sc
from numpy import pi
import numpy as np
import astropy.units as u
from plasmapy.formulary.collisions import Coulomb_logarithm
from plasmapy.utils.exceptions import RelativityWarning
import warnings

# Disable plasmapy relativity warning
warnings.filterwarnings("ignore", category=RelativityWarning)

# Setup case
ndim = 1 # 1D case
laser_lambda = 351.0e-9 # m
laser_kvac = 2*pi/laser_lambda
laser_omega = sc.c*laser_kvac
nc = sc.epsilon_0*sc.m_e/sc.e**2*laser_omega**2 # m/s
ne = 0.1*nc # plasma at 0.1 nc
TeeV = 3000 # eV
Te = temperature_energy(TeeV,'eVtoK')
Z = 1.5
Ti = Te/3
mi = 2.5*sc.m_p

# Get forest class instance and assert setup
birch = forest(Te,ne,ndim,Z,Ti,mi)
print('\nTemp {K}: %0.1f; Density {1/m^3}: %0.3e; ndim: %d' \
    % (birch.Te,birch.ne,birch.ndim))
necheck = 9.049e26
Techeck = 3000/8.617333262145e-5
real_assert(birch.ne,necheck,1e23)
real_assert(birch.Te,Techeck,1e4)

## Calculate quantities and validate
# Electron thermal velocity
birch.get_vthe()
vthe_check = 22.97*1e-6*1e12
print('v_{th,e} [m/s]: %0.3e' % (birch.vthe))
real_assert(birch.vthe,vthe_check,1e4)

# Electron plasma frequency
birch.get_ompe()
ompe_check = 1.697*1e15
print('\omega_{p,e} [1/s]: %0.3e' % (birch.ompe))
real_assert(birch.ompe,ompe_check,1e12)

# Debye length
birch.get_dbyl()
dbyl_check = 13.54e-9
print('\lambda_D [m]: %0.3e' % (birch.dbyl))
real_assert(birch.dbyl,dbyl_check,1e-11)

# Coulomb logarithm
birch.get_coulomb_log(species='ei')
cl_check = 7.88
print('\lambda_{ei}: %0.2f' % (birch.coulomb_log_ei))
real_assert(birch.coulomb_log_ei,cl_check,1e-2)

# Electron-ion collision frequency
birch.get_collision_freq(species='ei')
nu_check = 0.21*1e12
print('\\nu_{ei}: %0.3e' % (birch.collision_freq_ei))
real_assert(birch.collision_freq_ei,nu_check,1e11)

# Final statement
print('All tests in forest_test.py complete.\n')

## Funcions still not validated:
# birch.get_coulomb_log(species=['ee','ii'])
# birch.get_collision_freq(species=['ee','ii'])
