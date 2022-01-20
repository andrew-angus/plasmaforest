#!/bin/python3

from plasmaforest.laser import *
import scipy.constants as sc
import numpy as np
import astropy.units as u
from plasmapy.utils.exceptions import RelativityWarning
import warnings

# Disable plasmapy relativity warning
warnings.filterwarnings("ignore", category=RelativityWarning)

# Setup Strozzi case as in forest_test.py
ndim = 1 # 1D case
laser_lambda = 351.0e-9 # m
laser_kvac = 2*np.pi/laser_lambda
laser_omega = sc.c*laser_kvac
nc = sc.epsilon_0*sc.m_e/sc.e**2*laser_omega**2 # m/s
ne = 0.1*nc # plasma at 0.1 nc
TeeV = 3000 # eV
Te = temperature_energy(TeeV,'eVtoK')
nion = 2
Z = np.array([1,2])
mi = np.array([1,4])*sc.m_p
Ti = np.ones(int(nion))*Te/3
ni = np.ones(int(nion))*ne/3
np.set_printoptions(precision=3)

# Get forest class instance and assert setup
alder = laser_forest(lambda0=laser_lambda,\
    Te=Te,ne=ne,ndim=ndim,nion=nion,Ti=Ti,ni=ni,Z=Z,mi=mi)

# Dispersion
alder.get_kvac()
alder.get_omega0()
alder.get_k0()
print(f'\lambda_0 [m] = {laser_lambda:0.3e}')
print(f'\omega_0 [1/s] = {alder.omega0:0.3e}')
print(f'k_vac [1/m] = {alder.kvac:0.3e}')
print(f'k_0 [1/m] = {alder.k0:0.3e}')
real_assert(alder.omega0,laser_omega,1e12)
real_assert(alder.kvac,laser_kvac,1e4)
k0test = 16.98e6
real_assert(alder.k0,k0test,1e4)

# Critical density
alder.get_nc()
print(f'n_c [1/m^3] = {alder.nc:0.3e}')
real_assert(alder.nc,nc,1e23)

print('All tests in laser_test.py complete.\n')
