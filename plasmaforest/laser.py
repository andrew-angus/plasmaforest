#!/bin/python3
from .core import *
#from pytypes import typechecked
from typeguard import typechecked
from typing import Optional, Union, Dict, TypedDict, Any

# Laser-plasma forest with routines for laser specific quantities
class options(TypedDict):
  Te:floats
  ne:floats
  ndim:ints
  nion:ints
  Ti:Optional[np.ndarray]
  ni:Optional[np.ndarray]
  Z:Optional[np.ndarray]
  mi:Optional[np.ndarray]
  
@typechecked
class laser_forest(forest):
  #def __init__(self,lambda0:floats,**kwargs:Union[Dict,Any]):
  def __init__(self,lambda0:floats,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.lambda0 = lambda0 # Laser wavelength
    self.nc = None # Critical density
    self.om0 = None # Laser frequency
    self.kvac = None # Laser wavenumber in vacuum
    self.k0 = None # Laser wavenumber in plasma

  # Calculate vacuum wavenumber
  def get_kvac(self):
    self.kvac = 2*np.pi/self.lambda0

  # Calculate vacuum omega

  # Calculate plasma wavenumber

  # Routine to calculate critical density
  def get_critical_density(self):
    pass

  # Get collisional damping rate, private general method for inheritance
  def __collisional_damping__(self):
    pass

  # Get laser collisional damping rate
  def get_collisional_damping(self):
    pass

class srs_forest(laser_forest):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)

# Dimensionless unit conversion class
# Normalisation based on laser wavelength in vacuum
class units:
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
