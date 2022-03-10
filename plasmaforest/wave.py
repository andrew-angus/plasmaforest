#!/bin/python3

import plasmapy as pp
import astropy.units as u
import astropy.constants as ac
import scipy.constants as sc
import numpy as np
from typing import Union,Optional,Tuple
from typeguard import typechecked
from .core import *

# Core class, mainly a wrapper of select plasmapy functionality
# Currently restricted to single-ion species
# Specifying ion parameters optional
# Take inputs in SI units
@typechecked
class wave_forest(forest):
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
  
  # Solve the EMW dispersion relation in a plasma
  def emw_dispersion(self,arg:floats,target:str) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if target == 'omega':
      return np.sqrt(sqr(self.ompe) + sqr(sc.c*arg))
    elif target == 'k':
      return np.sqrt((sqr(arg) - sqr(self.ompe))\
          /sqr(sc.c))
    else:
      raise Exception("target must be one of \'omega\' or \'k\'.")

  # Return residual of emw dispersion relation in dimensionless units for accuracy
  def emw_dispersion_res(self,omega:floats,k:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    kvac = omega/sc.c
    k0 = k/kvac
    ompe0 = self.ompe/omega
    return -1.0+sqr(k0)+sqr(ompe0)

  # Fluid EPW dispersion relation
  def bohm_gross(self,arg:floats,target:str) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    if target == 'omega':
      return np.sqrt(sqr(self.ompe) + prefac*sqr(self.vthe*arg))
    elif target == 'k':
      return np.sqrt((sqr(arg) - sqr(self.ompe))\
          /(prefac*sqr(self.vthe)))
    else:
      raise Exception("target must be one of \'omega\' or \'k\'.")

  # Residual of fluid EPW dispersion relation, dimensionless for accuracy
  def bohm_gross_res(self,omega:floats,k:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = np.sqrt(0.5*gamma)
    return -sqr(omega/self.ompe)+sqr(prefac*self.vthe*k/self.ompe)+1.0

  # Plasma dispersion function
  def Zfun(self,omega:flomplex,k:flomplex,species:str) -> flomplex:
    zeta = self.__zeta__(omega=omega,k=k,species=species)
    Z = pp.dispersion.plasma_dispersion_func(zeta)
    return Z

  # Derivative of plasma dispersion function
  def dZfun(self,omega:flomplex,k:flomplex,species:str) -> flomplex:
    zeta = self.__zeta__(omega=omega,k=k,species=species)
    Z = pp.dispersion.plasma_dispersion_func(zeta)
    dZ = -2*(1+zeta*Z)

  # Calculate zeta for both plasma dispersion function and its derivative
  def __zeta__(self,omega:flomplex,k:flomplex,species:str) -> flomplex:
    if species == 'e':
      if self.vthe is None:
        self.get_vth(species='e')
      a = self.vthe
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      a = self.vthi
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")
    return omega/(k*a)

  # Plasma susceptibility calculated with the plasma dispersion function
  def susceptibility(self,omega:flomplex,k:flomplex,species:str) -> flomplex: 
    dZ = dZfun(omega=omega,k=k,species=species)
    if species == 'e':
      if self.vthe is None:
        self.get_vth(species='e')
      if self.ompe is None:
        self.get_omp(species='e')
      a = self.vthe
      omp = self.ompe
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      if self.ompi is None:
        self.get_omp(species='i')
      omp = self.ompi
      a = self.vthi
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")
      
      return -sqr(omp/(k*a))*dZ

  # Linear kinetic dispersion equation
  def kinetic_dispersion(self,omega:flomplex,k:flomplex,full:Optional[bool]=True) -> flomplex:
    dis = 1 + susceptibility(omega=omega,k=k,species='e')
    if full:
      dis += np.sum(susceptibility(omega=omega,k=k,species='i'))
    return dis

  # Phase velocity of a wave
  def phase_velocity(self,omega:floats,k:floats) -> floats:
    return omega/k

  # EMW group velocity
  def emw_group_velocity(self,omega:floats,k:floats) -> floats:
    return sqr(sc.c)*k/omega

  # EPW bohm-gross group velocity
  def bohm_gross_group_velocity(self,omega:floats,k:floats) -> floats:
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    return prefac*sqr(self.vthe)*k/omega

  # General EMW critical density
  def emw_nc(self,omega:floats) -> floats:
    return sc.epsilon_0*sc.m_e*sqr(omega/sc.e)

  # General EMW collisional damping rate
  # Use Kruer Ch 5
  def emw_damping(self,omega:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    if self.dbyl is None:
      self.get_dbyl()
    if self.coulomb_log_ei is None:
      self.get_coulomb_log(species='ei')
    vth1Drms = self.vthe/np.sqrt(2)
    impact = np.log(np.exp(self.coulomb_log_ei)/self.dbye*(vth1Drms/omega))
    return sqr(self.ompe/omega)/(3*pwr(2*np.pi,3/2))*self.Z \
        /(self.ne+np.sum(self.ni))*pwr(self.ompe/vth1Drms,3)*self.ompe*impact/2
  
  # Refractive index function
  def emw_ri(self,nc:floats) -> floats:
    self.electron_check()
    return np.sqrt(1-self.ne/nc)

  # EMW E field in plasma
  def emw_E(self,I:floats,ri:floats) -> floats:
    return np.sqrt(2*I/(sc.epsilon_0*sc.c*ri))

  # EMW B field in plasma
  def emw_B(self,I:floats,vp:floats) -> floats:
    return np.sqrt(2*I*sc.mu_0/vp)

  # EMW electron quiver velocity
  def emw_vos(self,E:floats,omega:floats) -> floats:
    return sc.e*E/(sc.m_e*omega)

  # EPW collisional damping
  # Calculated according to Rand - Collision Damping of Electron Plasma Waves (1965)
  def epw_coll_damping(self,omega:floats) -> floats:
    # Check for attributes
    if self.coulomb_log_ei is None:
      self.get_coulomb_log(species='ei')
    if self.vthe is None:
      self.get_vth(species='e')
    if self.ompe is None:
      self.get_omp(species='e')

    # Calculate in cgs units
    ecgs = ac.e.gauss.value
    hbarcgs = ac.hbar.cgs.value
    necgs = (self.ne/u.m**3).cgs.value
    nicgs = (self.ni/u.m**3).cgs.value
    mecgs = (sc.m_e*u.kg).cgs.value
    vthecgs = (self.vthe/np.sqrt(2)*u.m/u.s).cgs.value

    # Calculate switch quantities
    omrat = omega/self.ompe
    G = 0.27*omrat-0.091
    regime = sqr(ecgs)/(hbarcgs*vthecgs)
    if regime < 1:
      logfac = np.log(2*mecgs*sqr(vthecgs)/(hbarcgs*self.ompe))-0.442-G
    else:
      logfac = np.log(2*mecgs*pwr(vthecgs,3)/(sqr(ecgs)*self.ompe))-1.077-G
    
    A = 16*np.pi/3*np.sqrt(2*np.pi)*pwr(ecgs,6)*sqr(self.Z)*necgs*nicgs\
        /pwr(mecgs*vthecgs*omega,3)*logfac
    
    return 0.5*A*self.ompe

  def epw_landau_damping(self,omega:floats,k:floats,relativistic:Optional[bool]=False) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')

    # Relativistic calc of Landau damping according to:
    # Bers - Relativistic Landau damping of electron plasma waves 
    # in stimulated Raman scattering (2009)
    if relativistic:
      vth = self.vthe/np.sqrt(2)
      mu = sqr(sc.c/vth)
      N = sc.c*k/omega
      kdb = k*self.dbye
      z0 = mu*np.abs(N)/np.sqrt(sqr(N)-1)
      gamma = np.sqrt(np.pi/8)*omega*pwr(mu,3/2)*np.exp(mu*(1-N/np.sqrt(sqr(N)-1)))\
          /(np.abs(N)*(sqr(N)-1))*(1+2/z0+2/sqr(z0))/(1+6*sqr(kdb)-5/(2*mu))

    # First order approximation of non-relativistic landau damping
    # Calculated according to Swanson - Plasma Waves (2012)
    else:
      """
      vthe1d = self.vthe/np.sqrt(2)
      dk = k*vthe1d/omega
      omega2 = omega*(1+3/2*sqr(dk))
      print(self.ompe,omega,omega2)
      gamma = np.sqrt(np.pi/8)*omega2/pwr(dk,3)\
          *np.exp(-0.5*sqr(omega2/(k*vthe1d))) # lpse
      print(f'{gamma:0.3e}')
      """
      gamma = np.sqrt(np.pi)*sqr(self.ompe*omega)/pwr(k*self.vthe,3)\
          *np.exp(-sqr(omega/(k*self.vthe)))/2

    return gamma
