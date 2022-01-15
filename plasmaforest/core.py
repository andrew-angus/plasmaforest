#!/bin/python3

import plasmapy as pp
import astropy.units as u
import scipy.constants as sc
import numpy as np

# Core class, mainly a wrapper of select plasmapy functionality
# Currently restricted to single-ion species
# Specifying ion parameters optional
# Take inputs in SI units
class forest:
  # Initialise with physical parameters and dimensionality
  def __init__(self,Te,ne,ndim,Z=None,Ti=None,mi=None):
    self.Te = Te
    self.ne = ne
    self.ndim = ndim
    self.Z = Z
    self.Ti = Ti
    self.mi = mi
    self.vthe = None
    self.ompe = None
    self.dbyl = None
    self.coulomb_log_ei = None
    self.coulomb_log_ee = None
    self.coulomb_log_ii = None
    if Z is not None:
      self.ni = ne/Z # Quasi-neutrality

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

  # Get coulomb logaritms
  # Calculated using NRL formulary (which uses cgs units)
  def get_coulomb_log(self,species):
    # Check ion parameters specified
    self._exist(self.Z,'self.Z')
    self._exist(self.Ti,'self.Ti')
    self._exist(self.mi,'self.mi')
    self._exist(self.ni,'self.ni')

    # Get everything in cgs units, eV for temperatures
    ne = (self.ne/u.m**3).cgs.value
    ni = (self.ni/u.m**3).cgs.value
    Te = temperature_energy(self.Te,method='KtoeV')
    Ti = temperature_energy(self.Ti,method='KtoeV')
    me = (sc.m_e*u.kg).cgs.value 
    mi = (self.mi*u.kg).cgs.value 
    mui = self.mi/sc.m_p
    Z = self.Z

    ## Coulomb log calcs for different species
    # Electron-ion 
    if species == 'ei' or species == 'ie':
      # Selective quantities
      Ti_mr = Ti*me/mi
      Zscaled = 10*Z**2
      # Evaluation cases
      if Ti_mr < Te and Te < Zscaled:
        cl = 23-np.log(np.sqrt(ne)*Z*np.power(Te,-3/2))
      elif Ti_mr < Zscaled and Zscaled < Te:
        cl = 24-np.log(np.sqrt(ne)/Te)
      elif Te < Ti_mr:
        cl = 16-np.log(np.sqrt(ni)*np.power(Ti,-3/2)*Z**2*mui)
      else:
        raise Exception(\
            "Error: coulomb_log_ei calc does not fit any NRL formulary cases") 
      self.coulomb_log_ei = cl
    # Electron-electron
    elif species == 'ee':
      self.coulomb_log_ee = 23.5-np.log(np.sqrt(ne)*np.power(Te,-5/4))\
          -np.sqrt(1e-5+(np.log(Te)-2)**2/16)
    # Ion-ion
    elif species == 'ii':
      self.coulomb_log_ii = 23-np.log(Z**2/Ti*np.sqrt(2*ni*Z**2/Ti))
    else:
      raise Exception(\
          "Error: species must be one of \'ei\', \'ie\', \'ee\' or \'ii\'")

  # Calculate collision frequency according to NRL formulary
  def get_collision_freq(self,species):
    """
    ne = (self.ne/u.m**3).cgs.value
    Ti = temperature_energy(self.Ti,method='KtoeV')
    me = (sc.m_e*u.kg).cgs.value 
    mi = (self.mi*u.kg).cgs.value 
    mui = self.mi/sc.m_p
    """
    if species == 'ei':
      ni = (self.ni/u.m**3).cgs.value
      Te = temperature_energy(self.Te,method='KtoeV')
      Z = self.Z
      if self.coulomb_log_ei is None:
        self.get_coulomb_log(species='ei')
      self.collision_freq_ei = 3.9e-6*np.power(Te,-3/2)*ni*Z**2\
          *self.coulomb_log_ei 

  def _exist(self,var,varname):
    if var is None:
      raise Exception(f"Error: must set {varname}")

# Function for converting between eV and K using astropy
def temperature_energy(T,method):
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
