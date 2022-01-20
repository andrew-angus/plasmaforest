#!/bin/python3

import plasmapy as pp
import astropy.units as u
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
    prefac = gamma/self.ndim
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
    prefac = gamma/self.ndim
    return (-sqr(omega)+prefac*sqr(self.vthe*k))/sqr(self.ompe)+1.0

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
      a = self.vthe*np.sqrt(2/self.ndim)
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      a = self.vthi*np.sqrt(2/self.ndim)
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")
    return omega/k/a

  # Plasma susceptibility calculated with the plasma dispersion function
  def susceptibility(self,omega:flomplex,k:flomplex,species:str) -> flomplex: 
    dZ = dZfun(omega=omega,k=k,species=species)
    if species == 'e':
      if self.vthe is None:
        self.get_vth(species='e')
      if self.ompe is None:
        self.get_omp(species='e')
      a = self.vthe*np.sqrt(2/self.ndim)
      omp = self.ompe
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      if self.ompi is None:
        self.get_omp(species='i')
      omp = self.ompi
      a = self.vthi*np.sqrt(2/self.ndim)
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
  def emw_group_velocity(self,omega,k) -> floats:
    return sqr(sc.c)*k/omega

  # EPW bohm-gross group velocity
  def bohm_gross_group_velocity(self,omega,k) -> floats:
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = gamma/self.ndim
    return prefac*sqr(self.vthe)*k/omega

  # General EMW critical density
  def emw_nc(self,omega:floats) -> floats:
    return sc.epsilon_0*sc.m_e*sqr(omega/sc.e)
  
  # Refractive index function
  def ri(self,nc:floats) -> floats:
    self.electron_check()
    return np.sqrt(1-self.ne/nc)
