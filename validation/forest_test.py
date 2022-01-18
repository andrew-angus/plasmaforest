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
#nion = 1
#Z = np.array([1.5])
#mi = np.array([2.5*sc.m_p])
nion = 2
Z = np.array([1,2])
mi = np.array([1,4])*sc.m_p
Ti = np.ones(int(nion))*Te/3
ni = np.ones(int(nion))*ne/3
np.set_printoptions(precision=3)

# Get forest class instance and assert setup
birch = forest(Te=Te,ne=ne,ndim=ndim,nion=nion,Z=Z,Ti=Ti,ni=ni,mi=mi)
print('\nTe {K}: %0.1f; ne {1/m^3}: %0.3e; ndim: %d' \
    % (birch.Te,birch.ne,birch.ndim))
print('Ti {K}: %0.1f; nion: %d' \
    % (birch.Ti[0],birch.nion))
print('Ions: [H,He]; Z:',Z)
print('ni {1/m^3}:',ni,'; mi {kg}:',mi)

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
print('\lambda_{ei}:',birch.coulomb_log_ei)
real_assert(birch.coulomb_log_ei[0],cl_check,1e-2)

# Electron-ion collision frequency
birch.get_collision_freq(species='ei')
nu_check = 0.21*1e12
print('\\nu_{ei} [1/s]:',birch.collision_freq_ei)
nu_tot = np.sum(birch.collision_freq_ei)
print('\\nu_{ei,tot} [1/s]: %0.3e' % (nu_tot))
real_assert(nu_tot,nu_check,1e11)

# EMW dispersion
k0 = birch.emw_dispersion(arg=laser_omega,target='k')
print('k_{0} [1/m]: %0.3e' % (k0)) 
real_assert(k0,laser_kvac,1e6)
om0 = birch.emw_dispersion(arg=k0,target='omega')
print('\omega_{0} [1/s]: %0.3e' % (om0)) 
real_assert(om0,laser_omega,1e12)
res = birch.emw_dispersion_res(om0,k0)
real_assert(abs(res),0,1e-15)

# EPW fluid dispersion
om1s = 2.086e15
k1 = birch.bohm_gross(arg=om1s,target='k')
print('k_{ek} [1/m]: %0.3e' % (k1)) 
om1 = birch.bohm_gross(arg=k1,target='omega')
print('\omega_{ek} [1/s]: %0.3e' % (om1)) 
real_assert(om1,om1s,1e12)
res = birch.bohm_gross_res(om1,k1)
real_assert(abs(res),0,1e-15)

# Final statement
print('All tests in forest_test.py complete.\n')

## Funcions still not validated againtst known answers:
# birch.get_coulomb_log(species=['ee','ii'])
# birch.get_collision_freq(species=['ee','ii'])
