#!/bin/python3
from .core import *
from .wave import *
from typeguard import typechecked
from typing import Optional, Union, Dict, TypedDict, Any

# Laser-plasma forest with routines for laser specific quantities
@typechecked
class laser_forest(wave_forest):
  # Initialise using laser vacuum wavelength and intensity
  def __init__(self,lambda0:floats,I0:floats,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.lambda0 = lambda0 # Laser wavelength
    self.I0 = I0 # Laser intensity
    self.nc = None # Critical density
    self.om0 = None # Laser frequency
    self.kvac = None # Laser wavenumber in vacuum
    self.k0 = None # Laser wavenumber in plasma
    self.ib = None # Inverse bremstrah
    self.v_quiver = None # Quiver velocity

  # Update intensity attribute
  def set_intensity(self,I0:floats):
    self.I0 = I0
    # Nullify any intensity-dependent attributes

  # Calculate vacuum wavenumber
  def get_kvac(self):
    self.kvac = 2*np.pi/self.lambda0

  # Calculate vacuum omega
  def get_omega0(self):
    if self.kvac is None:
      self.get_kvac()
    self.omega0 = self.kvac*sc.c

  # Calculate plasma wavenumber
  def get_k0(self):
    if self.omega0 is None:
      self.get_omega0()
    self.k0 = self.emw_dispersion(self.omega0,target='k')

  # General EMW critical density
  def emw_nc(self,omega):
    return sc.epsilon_0*sc.m_e*sqr(omega/sc.e)
  
  # Set laser critical density attribute
  def get_nc(self):
    if self.omega0 is None:
      self.get_omega0()
    self.nc = self.emw_nc(self.omega0)

  # Get collisional damping rate, private general method for inheritance
  def __collisional_damping__(self):
    pass

  # Get laser collisional damping rate
  def get_collisional_damping(self):
    pass

  # Get E field

  # Get quiver velocity

  # Get phase velocity

  # Get group velocity

class srs_forest(laser_forest):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)

# Dimensionless unit conversion class
# Normalisation based on laser wavelength in vacuum
class units:
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
