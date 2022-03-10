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
from plasmaforest.srs import *
import scipy.constants as sc
import numpy as np
import astropy.units as u
from plasmapy.utils.exceptions import RelativityWarning
import warnings

# Disable plasmapy relativity warning
warnings.filterwarnings("ignore", category=RelativityWarning)

# Setup case
mode = 'fluid'
relativistic = False
ndim = 1 # 1D case
lambda0 = 351.0e-9 # m
I0 = 2e19 # W/m^2

# Laser forest instance initially without plasma parameter specification
birch = srs_forest(mode,relativistic,lambda0,I0,ndim,electrons=False,nion=0)

# Get and verify laser vacuum parameters and critical density
birch.get_kvac()
birch.get_omega0()
birch.get_nc0()
kvactest = 2*np.pi/lambda0
omega0test = sc.c*kvactest
nctest = sc.epsilon_0*sc.m_e/sc.e**2*omega0test**2 # m/s
print(f'\nndim = {ndim:d}')
print(f'I_0 [W/m^2] = {I0:0.1e}')
print(f'\lambda_0 [m] = {lambda0:0.3e}')
print(f'k_vac [1/m] = {birch.kvac:0.3e}')
print(f'\omega_0 [1/s] = {birch.omega0:0.3e}')
print(f'n_c [1/m^3] = {birch.nc0:0.3e}')
real_assert(birch.omega0,omega0test,1e12)
real_assert(birch.kvac,kvactest,1e4)
real_assert(birch.nc0,nctest,1e23)

# Specify electron properties
ne = 0.1*birch.nc0 # plasma at 0.1 nc
TeeV = 3000 # eV
Te = temperature_energy(TeeV,'eVtoK')
birch.set_electrons(electrons=True,Te=Te,ne=ne)
print('Te [K]: %0.1f; ne [1/m^3]: %0.3e' \
    % (birch.Te,birch.ne))

# Specify ion properties
nion = 2
Z = np.array([1,2])
mi = np.array([1,4])*sc.m_p
Ti = np.ones(int(nion))*Te/3
ni = np.ones(int(nion))*ne/3
birch.set_ions(nion=nion,Ti=Ti,ni=ni,Z=Z,mi=mi)
np.set_printoptions(precision=3)
print('Ti [K]: %0.1f; nion: %d' \
    % (birch.Ti[0],birch.nion))
print('Ions: [H,He]; Z:',Z)
print('ni [1/m^3]:',ni,'; mi {kg}:',mi)

# Electron thermal velocity
birch.get_vth(species='e')
print('v_the [m/s]: %0.3e' % (birch.vthe))
real_assert(birch.vthe/np.sqrt(2),22.97*1e-6*1e12,1e4)

# Electron plasma frequency
birch.get_omp(species='e')
print('\omega_pe [1/s]: %0.3e' % (birch.ompe))
real_assert(birch.ompe,1.697*1e15,1e12)

# Debye length
birch.get_dbyl()
print('\lambda_D [m]: %0.3e' % (birch.dbyl))
real_assert(birch.dbyl,13.54e-9,1e-8)

# Electron spacing and De Broglie wavelength
birch.get_spacing(species='e')
print('e_spacing [m]: %0.3e' % (birch.e_spacing))
real_assert(birch.e_spacing,1.03e-9,1e-11)

# Electron-ion Coulomb logarithm
birch.get_coulomb_log(species='ei')
print('\lambda_ei:',birch.coulomb_log_ei)
real_assert(birch.coulomb_log_ei[0],7.88,1e-2)

# Electron-ion collision frequency
birch.get_collision_freq(species='ei')
print('\\nu_ei [1/s]:',birch.collision_freq_ei)
nu_tot = np.sum(birch.collision_freq_ei)
print('\\nu_ei,tot [1/s]: %0.3e' % (nu_tot))
real_assert(nu_tot,0.21*1e12,1e11)

# Laser wavenumber
birch.get_k0()
print(f'k_0 [1/m] = {birch.k0:0.3e}')
real_assert(birch.k0,16.98e6,1e4)

# EMW dispersion checks
om0 = birch.emw_dispersion(arg=birch.k0,target='omega')
real_assert(om0,birch.omega0,1e12)
res = birch.emw_dispersion_res(birch.omega0,birch.k0)
real_assert(abs(res),0,1e-15)

# EPW fluid dispersion checks
om2s = 2.086e15
k2 = birch.bohm_gross(arg=om2s,target='k')
print('k_2 [1/m]: %0.3e' % (k2)) 
om2 = birch.bohm_gross(arg=k2,target='omega')
print('\omega_2 [1/s]: %0.3e' % (om2)) 
real_assert(om2,om2s,1e12)
res = birch.bohm_gross_res(om2,k2)
real_assert(abs(res),0,1e-15)

# Laser refractive index
birch.get_ri0()
print(f'Laser RI: {birch.ri0:0.3f}')
real_assert(birch.ri0,np.sqrt(0.9),1e-3)

# Plasma E field
birch.get_E0()
print(f'E_0 [V/m]: {birch.E0:0.3e}')
real_assert(birch.E0,0.126*1e12,1e8)

# Laser quiver velocity
birch.get_vos0()
print(f'v_os,0 [m/s]: {birch.vos0:0.3e}')
real_assert(birch.vos0,4.023*1e-6*1e12/np.sqrt(birch.ri0),1e3)

# Laser phase velocity
birch.get_vp0()
print(f'v_p,0 [m/s]: {birch.vp0:0.3e}')
real_assert(birch.vp0,316*1e-6*1e12,1e5)

# Laser group velocity
birch.get_vg0()
print(f'v_g,0 [m/s]: {birch.vg0:0.3e}')
real_assert(birch.vg0,284*1e-6*1e12,1e6)

# Laser collisional damping
birch.get_damping0()
print('\\nu_0 [1/s]:', birch.damping0)
print(f'\\nu_0,tot [1/s]: {np.sum(birch.damping0):0.3e}')
real_assert(np.sum(birch.damping0),1.051e10,1e9)

# Raman collisional damping
birch.get_damping1()
print('\\nu_1 [1/s]:', birch.damping1)
print(f'\\nu_1,tot [1/s]: {np.sum(birch.damping1):0.3e}')
real_assert(np.sum(birch.damping1),2.813e10,2e9)

# SRS fluid resonance matching
birch.resonance_solve()
print(f'fluid \\omega_1 [1/s]: {birch.omega1:0.3e}')
print(f'fluid \\omega_2 [1/s]: {birch.omega2:0.3e}')
print(f'fluid k_1 [1/m]: {birch.k1:0.3e}')
print(f'fluid k_2 [1/m]: {birch.k2:0.3e}')
assert(birch.emw_dispersion_res(birch.omega1,birch.k1) < 1e14)

# EPW collisional damping 
birch.get_cdamping2()
print(f'\\nu_2c [1/s]: {birch.cdamping2}')
print(f'\\nu_2c,tot [1/s]: {np.sum(birch.cdamping2):0.3e}')
real_assert(np.sum(birch.cdamping2),5e10,4e10)

# EPW Landau damping 
birch.get_ldamping2()
print(f'fluid \\nu_2l [1/s]: {birch.ldamping2:0.3e}')
real_assert(birch.ldamping2,64.47e12,2e13)
birch.set_relativistic(True)
birch.get_ldamping2()
print(f'relativistic \\nu_2l [1/s]: {birch.ldamping2:0.3e}')

# Final statement
print('All health checks complete. What a happy forest.\n')
