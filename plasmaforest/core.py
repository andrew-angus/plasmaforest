#!/bin/python3

import plasmapy as pp
import astropy.units as u

# Core class, mainly a wrapper of select plasmapy functionality
# Take inputs in SI units
class forest:
  # Initialise with physical parameters and dimensionality
  def __init__(self,Te,ne,ndim):
    self.Te = Te
    self.ne = ne
    self.ndim = ndim
    self.vthe = None
    self.ompe = None
    self.dbyl = None

  # Get electron thermal velocity
  def get_vthe(self):
    Te = self.Te * u.K
    vthe = pp.formulary.parameters.thermal_speed(T=Te,particle='e-',\
        ndim=self.ndim,method='rms')
    self.vthe = vthe.value

  # Get electron plasma frequency
  def get_ompe(self):
    ne = self.ne / u.m**3
    ompe = pp.formulary.parameters.plasma_frequency(n=ne,particle='e-')
    self.ompe = ompe.value

  # Get Debye length
  def get_dbyl(self):
    Te = self.Te * u.K
    ne = self.ne / u.m**3
    dbyl = pp.formulary.parameters.Debye_length(T_e=Te,n_e=ne)
    self.dbyl = dbyl.value

# Function for converting between eV and K using astropy
def temperature_energy(T,method=None):
  if method == 'KtoeV':
    T = T * u.K
    TeV = T.to(u.eV,equivalencies=u.temperature_energy())
    return TeV.value
  elif method == 'eVtoK':
    T = T * u.eV
    TK = T.to(u.K,equivalencies=u.temperature_energy())
    return TK.value
  else:
    raise Exception("Error: method argument must be one of \'KtoeV\'" \
        "or \'eVtoK\'.")

# Modified assertion function for real valued arguments
def real_assert(val,valcheck,diff):
  assert(val > valcheck - diff)
  assert(val < valcheck + diff)
