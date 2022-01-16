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
  def __init__(self,Te,ne,ndim,nion,Ti=None,ni=None,Z=None,mi=None):
    self.Te = Te # Electron temperature
    self.ne = ne # Electron density
    self.ndim = ndim # Plasma dimensionality
    self.vthe = None # Electron thermal velocity
    self.ompe = None # Electron plasma frequency
    self.dbyl = None # Debye length
    self.coulomb_log_ee = None # Electron-electron coulomb log
    self.collision_freq_ee = None # Electron-electron collision frequency
    self.set_ions(nion=nion,Ti=Ti,ni=ni,Z=Z,mi=mi)

  def set_ions(self,nion,Ti=None,ni=None,Z=None,mi=None):
    self.nion = nion # Number of ion species
    self.Z = Z # Ion charge states
    self.Ti = Ti # Ion temperatures
    self.ni = ni # Ion densities
    self.mi = mi # Ion masses
    self.coulomb_log_ei = None # Electron-ion coulomb logs
    self.coulomb_log_ii = None # Ion-ion coulomb logs
    self.collision_freq_ei = None # Electron-ion collision frequencies
    self.collision_freq_ie = None # Ion-electron collision frequencies
    self.collision_freq_ii = None # Ion-ion collision frequencies

    # Check nion > 0
    if nion < 0 or not isinstance(nion,int):
      raise Exception("nion parameter must be integer >= 0")

    # Check ion parameter specification
    for i in [Z,Ti,ni,mi]:
      if i is None and nion > 0:
        raise Exception("nion > 0 but not all ion parameters specified")
      elif i is not None and nion == 0:
        raise Exception("nion = 0 but ion parameters specified")
      elif nion > 0 and (not isinstance(i,np.ndarray) or len(i) != nion):
        raise Exception("Ion parameters must be numpy arrays of length nion")

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
    if species in ['ei','ie','ii'] and self.nion == 0:
      print('Warning: no ion parameters specified to '\
          + 'calculate ion coulomb logarithm, use set_ions method.')
      return

    # Get everything in cgs units, eV for temperatures
    nrl = self._nrl_collisions(species)
    Z = self.Z

    ## Coulomb log calcs for different species
    # Electron-ion 
    if species == 'ei' or species == 'ie':
      # Selective quantities
      ne,Te,Ti,me,mi,mui,ni = nrl
      cl = np.zeros(self.nion)
      for i in range(self.nion):
        Ti_mr = Ti[i]*me/mi[i]
        Zscaled = 10*np.power(Z[i],2)
        # Evaluation cases
        if Ti_mr < Te and Te < Zscaled:
          cl[i] = 23-np.log(np.sqrt(ne)*Z[i]*np.power(Te,-3/2))
        elif Ti_mr < Zscaled and Zscaled < Te:
          cl[i] = 24-np.log(np.sqrt(ne)/Te)
        elif Te < Ti_mr:
          cl[i] = 16-np.log(np.sqrt(ni[i])*np.power(Ti[i],-3/2)\
              *np.power(Z[i],2)*mui[i])
        else:
          raise Exception(\
              "Error: coulomb_log_ei calc does not fit any NRL formulary cases") 
      self.coulomb_log_ei = cl
    # Electron-electron
    elif species == 'ee':
      ne,Te = nrl
      self.coulomb_log_ee = 23.5-np.log(np.sqrt(ne)*np.power(Te,-5/4))\
          -np.sqrt(1e-5+np.power(np.log(Te)-2,2)/16)
    # Ion-ion
    elif species == 'ii':
      ne,Te,Ti,me,mi,mui,ni = nrl
      self.coulomb_log_ii = 23-np.log(np.power(self.Z,3)/np.power(Ti,3/2)\
          *np.sqrt(2*ni))
    else:
      raise Exception(\
          "Error: species must be one of \'ei\', \'ie\', \'ee\' or \'ii\'")

  # Calculate collision frequency according to NRL formulary
  def get_collision_freq(self,species):
    # Check ion parameters specified if required
    if species in ['ei','ie','ii'] and self.nion == 0:
      print('Warning: no ion parameters specified to '\
          + 'calculate ion collision frequency, use set_ions method.')
      return

    # Get everything in cgs units, eV for temperatures
    nrl = self._nrl_collisions(species)
    Z = self.Z

    # Calculate collision frequencies for each species pair
    if species == 'ei':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ei is None:
        self.get_coulomb_log(species='ei')
      self.collision_freq_ei = 3.9e-6*np.power(Te,-3/2)*ni*np.power(self.Z,2)\
          *self.coulomb_log_ei 
    elif species == 'ee':
      ne,Te = nrl
      if self.coulomb_log_ee is None:
        self.get_coulomb_log(species='ee')
      self.collision_freq_ee = 7.7e-6*np.power(Te,-3/2)*ne\
          *self.coulomb_log_ee 
    elif species == 'ie':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ei is None:
        self.get_coulomb_log(species='ei')
      self.collision_freq_ei = 1.6e-9/mui*np.power(Te,-3/2)*ne\
          *np.power(self.Z,2)*self.coulomb_log_ei 
    elif species == 'ii':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ii is None:
        self.get_coulomb_log(species='ii')
      self.collision_freq_ii = 6.8e-8/np.sqrt(mui)*2*np.power(Ti,-3/2)\
          *ni*np.power(self.Z,4)*self.coulomb_log_ii 

  # Return NRL formulary units for collision quantity calcs
  def _nrl_collisions(self,species):
    ne = (self.ne/u.m**3).cgs.value
    Te = temperature_energy(self.Te,method='KtoeV')
    if species in ['ei','ie','ii']:
      Ti = temperature_energy(self.Ti,method='KtoeV')
      me = (sc.m_e*u.kg).cgs.value 
      mi = (self.mi*u.kg).cgs.value 
      mui = self.mi/sc.m_p
      ni = (self.ni/u.m**3).cgs.value
      return ne,Te,Ti,me,mi,mui,ni
    else:
      return ne,Te

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
