#!/bin/python3

# Validate all routines in forest class
# Mainly use results from:
# Strozzi, David J. 
# Vlasov simulations of kinetic enhancement of 
# Raman backscatter in laser fusion plasmas. 
# Diss. Massachusetts Institute of Technology, 2005.

from plasmaforest.core import *
from plasmaforest.wave import *
from plasmaforest.laser import *
import scipy.constants as sc
import numpy as np
import astropy.units as u
from plasmapy.utils.exceptions import RelativityWarning
import warnings

# Disable plasmapy relativity warning
warnings.filterwarnings("ignore", category=RelativityWarning)

# Setup case
ndim = 1 # 1D case
lambda0 = 351.0e-9 # m
I0 = 2e18 # W/m^2

# Laser forest instance initially without plasma parameter specification
birch = laser_forest(lambda0,I0,ndim,electrons=False,nion=0)

# Get and verify laser vacuum parameters and critical density
birch.get_kvac()
birch.get_omega0()
birch.get_nc()
kvactest = 2*np.pi/lambda0
omega0test = sc.c*kvactest
nctest = sc.epsilon_0*sc.m_e/sc.e**2*omega0test**2 # m/s
print(f'\nndim = {ndim:d}')
print(f'I_0 [W/m^2] = {I0:0.1e}')
print(f'\lambda_0 [m] = {lambda0:0.3e}')
print(f'k_vac [1/m] = {birch.kvac:0.3e}')
print(f'\omega_0 [1/s] = {birch.omega0:0.3e}')
print(f'n_c [1/m^3] = {birch.nc:0.3e}')
real_assert(birch.omega0,omega0test,1e12)
real_assert(birch.kvac,kvactest,1e4)
real_assert(birch.nc,nctest,1e23)

# Specify electron properties
ne = 0.1*birch.nc # plasma at 0.1 nc
TeeV = 3000 # eV
Te = temperature_energy(TeeV,'eVtoK')
birch.set_electrons(electrons=True,Te=Te,ne=ne)

# Specify ion properties
nion = 2
Z = np.array([1,2])
mi = np.array([1,4])*sc.m_p
Ti = np.ones(int(nion))*Te/3
ni = np.ones(int(nion))*ne/3
birch.set_ions(nion=nion,Ti=Ti,ni=ni,Z=Z,mi=mi)

# Get forest class instance and assert setup
#birch = forest(Te=Te,ne=ne,ndim=ndim,nion=nion,Z=Z,Ti=Ti,ni=ni,mi=mi)
np.set_printoptions(precision=3)
print('Te {K}: %0.1f; ne {1/m^3}: %0.3e' \
    % (birch.Te,birch.ne))
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
birch.get_vth(species='e')
vthe_check = 22.97*1e-6*1e12
print('v_{th,e} [m/s]: %0.3e' % (birch.vthe))
real_assert(birch.vthe,vthe_check,1e4)

# Electron plasma frequency
birch.get_omp(species='e')
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

# Laser wavenumber
birch.get_k0()
print(f'k_0 [1/m] = {birch.k0:0.3e}')
k0test = 16.98e6
real_assert(birch.k0,k0test,1e4)

# EMW dispersion checks
om0 = birch.emw_dispersion(arg=birch.k0,target='omega')
real_assert(om0,birch.omega0,1e12)
res = birch.emw_dispersion_res(birch.omega0,birch.k0)
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
print('All health checks complete. What a happy forest.\n')

## Funcions still not validated againtst known answers:
# birch.get_coulomb_log(species=['ee','ii'])
# birch.get_collision_freq(species=['ee','ii','ie'])
# birch.get_vth(species='i')
# birch.get_omp(species='i')
# All EPW kinetic dispersion functions
