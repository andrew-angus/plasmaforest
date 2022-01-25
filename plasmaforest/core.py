#!/bin/python3

import plasmapy as pp
import astropy.units as u
import scipy.constants as sc
import numpy as np
from typing import Union,Optional,Tuple,Any
from typeguard import typechecked

# Custom types
floats = Union[np.float64,float,np.ndarray]
ints = Union[np.int_,int,np.ndarray]
complexes = Union[np.complex64,complex,np.ndarray]
flint = Union[floats,ints]
flomplex = Union[floats,complexes]

# Core class, common plasma parameters
# Take inputs in SI units
@typechecked
class forest:
  # Initialise with physical parameters and dimensionality
  def __init__(self,ndim:int,electrons:bool,nion:int,\
      Te:Optional[float]=None,ne:Optional[float]=None,\
      Ti:Optional[np.ndarray]=None,ni:Optional[np.ndarray]=None,\
      Z:Optional[np.ndarray]=None,mi:Optional[np.ndarray]=None):
    self.set_ndim(ndim=ndim)
    self.set_electrons(electrons=electrons,Te=Te,ne=ne)
    self.set_ions(nion=nion,Ti=Ti,ni=ni,Z=Z,mi=mi)

  # Set number of dimensions and null dimension-dependent properties
  def set_ndim(self,ndim:int):
    self.ndim = ndim # Plasma dimensionality
    self.vthe = None # Electron RMS thermal velocity
    self.vthi = None # Ion RMS thermal velocity

  # Set ion properties and null any ion-dependent properties
  def set_ions(self,nion:int,Ti:Optional[np.ndarray]=None,\
      ni:Optional[np.ndarray]=None,Z:Optional[np.ndarray]=None,\
      mi:Optional[np.ndarray]=None):
    self.nion = nion # Number of ion species
    self.Z = Z # Ion charge states
    self.Ti = Ti # Ion temperatures
    self.ni = ni # Ion densities
    self.mi = mi # Ion masses
    self.vthi = None # Ion RMS thermal velocities
    self.ompi = None # Ion plasma frequencies
    self.coulomb_log_ei = None # Electron-ion coulomb logs
    self.coulomb_log_ii = None # Ion-ion coulomb logs
    self.collision_freq_ei = None # Electron-ion collision frequencies
    self.collision_freq_ie = None # Ion-electron collision frequencies
    self.collision_freq_ii = None # Ion-ion collision frequencies

    # Check nion >= 0
    if nion < 0:
      raise Exception("nion parameter must be integer >= 0")

    # Check ion parameter specification
    arrs = [Z,Ti,ni,mi]
    dtypes = [np.int_,np.float64,np.float64,np.float64]
    for i in range(len(arrs)):
      if arrs[i] is None and nion > 0:
        raise Exception("nion > 0 but not all ion parameters specified")
      elif arrs[i] is not None and nion == 0:
        raise Exception("nion = 0 but ion parameters specified")
      elif nion > 0 and len(arrs[i]) != nion:
        raise Exception("Ion parameters must be numpy arrays of length nion")
      elif nion > 0:
        dtype_check(arrs[i],dtypes[i])

  def set_electrons(self,electrons:bool,Te:Optional[float]=None,\
      ne:Optional[float]=None):
    self.electrons = electrons # Boolean switch for using electron properties
    self.Te = Te # Electron temperature
    self.ne = ne # Electron density
    self.vthe = None # Electron RMS thermal velocity
    self.ompe = None # Electron plasma frequency
    self.dbyl = None # Debye length
    self.coulomb_log_ee = None # Electron-electron coulomb log
    self.collision_freq_ee = None # Electron-electron collision frequency
    self.e_spacing = None # Inter-electron spacing

    # Check electron parameters specified correctly
    if electrons:
      if self.Te is None:
        raise Exception("Parameter Te must be specified if electrons=True")
      if self.ne is None:
        raise Exception("Parameter ne must be specified if electrons=True")

  # Method to check ion specification
  def ion_check(self):
    if self.nion == 0:
      raise Exception('no ion parameters specified, use set_ions method.')

  # Method to check electron specification
  def electron_check(self):
    if not self.electrons:
      raise Exception('no electron parameters specified, use set_electrons method.')

  # Get RMS thermal velocity
  def get_vth(self,species:str):
    if species == 'e':
      self.electron_check()
      Te = self.Te * u.K
      vthe = pp.formulary.parameters.thermal_speed(T=Te,particle='e-',\
          ndim=self.ndim,method='rms')
      self.vthe = vthe.value
    elif species == 'i':
      self.ion_check()
      self.vthi = np.sqrt(self.ndim*self.Te*sc.k/self.mi)
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")

  # Get plasma frequency
  def get_omp(self,species:str):
    if species == 'e':
      self.electron_check()
      ne = self.ne / u.m**3
      ompe = pp.formulary.parameters.plasma_frequency(n=ne,particle='e-')
      self.ompe = ompe.value
    elif species == 'i':
      self.ion_check()
      self.ompi = np.sqrt(sqr(self.Z*sc.e)*self.ni/(self.mi*sc.epsilon_0))
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")

  # Get Debye length
  def get_dbyl(self):
    self.electron_check()
    Te = self.Te * u.K
    ne = self.ne / u.m**3
    dbyl = pp.formulary.parameters.Debye_length(T_e=Te,n_e=ne)
    self.dbyl = dbyl.value

  # Get electron spacing
  def get_e_spacing(self):
    self.electron_check()
    self.e_spacing = pwr(self.ne,-1/3)

  # Get coulomb logaritms
  # Calculated using NRL formulary (which uses cgs units)
  def get_coulomb_log(self,species:str):
    # Check parameters specified
    if species in ['ei','ie','ii']:
      self.ion_check()
    if species in ['ee','ei','ie']:
      self.electron_check()

    # Get everything in cgs units, eV for temperatures
    nrl = self.__nrl_collisions__(species)
    Z = self.Z

    ## Coulomb log calcs for different species
    # Electron-ion 
    if species == 'ei' or species == 'ie':
      # Selective quantities
      ne,Te,Ti,me,mi,mui,ni = nrl
      cl = np.zeros(self.nion)
      for i in range(self.nion):
        Ti_mr = Ti[i]*me/mi[i]
        Zscaled = 10*sqr(Z[i])
        # Evaluation cases
        if Ti_mr < Te and Te < Zscaled:
          cl[i] = 23-np.log(np.sqrt(ne)*Z[i]*pwr(Te,-3/2))
        elif Ti_mr < Zscaled and Zscaled < Te:
          cl[i] = 24-np.log(np.sqrt(ne)/Te)
        elif Te < Ti_mr:
          cl[i] = 16-np.log(np.sqrt(ni[i])*pwr(Ti[i],-3/2)\
              *sqr(Z[i])*mui[i])
        else:
          raise Exception(\
              "Error: coulomb_log_ei calc does not fit any NRL formulary cases") 
      self.coulomb_log_ei = cl

    # Electron-electron
    elif species == 'ee':
      ne,Te = nrl
      self.coulomb_log_ee = 23.5-np.log(np.sqrt(ne)*pwr(Te,-5/4))\
          -np.sqrt(1e-5+sqr(np.log(Te)-2)/16)
    # Ion-ion
    elif species == 'ii':
      ne,Te,Ti,me,mi,mui,ni = nrl
      uniques = self.nion*(self.nion+1)//2
      self.coulomb_log_ii = np.zeros(uniques)
      for i in range(self.nion):
        for j in range(i+1):
          vid = sym_mtx_to_vec(i,j,self.nion) 
          self.coulomb_log_ii[vid] = 23-np.log(Z[i]*Z[j]*(mui[i]+mui[j])\
              /(mui[i]*Ti[j]+mui[j]*Ti[i])*np.sqrt(ni[i]*sqr(Z[i])\
              /Ti[i]+ni[j]*sqr(Z[j])/Ti[j]))

  # Calculate collision frequency according to NRL formulary
  def get_collision_freq(self,species:str):
    # Check parameters specified
    if species in ['ei','ie','ii']:
      self.ion_check()
    if species in ['ee','ei','ie']:
      self.electron_check()

    # Get everything in cgs units, eV for temperatures
    nrl = self.__nrl_collisions__(species)
    Z = self.Z

    # Calculate collision frequencies for each species pair
    if species == 'ei':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ei is None:
        self.get_coulomb_log(species='ei')
      self.collision_freq_ei = 3.9e-6*pwr(Te,-3/2)*ni*sqr(self.Z)\
          *self.coulomb_log_ei 
    elif species == 'ee':
      ne,Te = nrl
      if self.coulomb_log_ee is None:
        self.get_coulomb_log(species='ee')
      self.collision_freq_ee = 7.7e-6*pwr(Te,-3/2)*ne\
          *self.coulomb_log_ee 
    elif species == 'ie':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ei is None:
        self.get_coulomb_log(species='ei')
      self.collision_freq_ie = 1.6e-9/mui*pwr(Te,-3/2)*ne\
          *sqr(self.Z)*self.coulomb_log_ei 
    elif species == 'ii':
      ne,Te,Ti,me,mi,mui,ni = nrl
      if self.coulomb_log_ii is None:
        self.get_coulomb_log(species='ii')
      self.collision_freq_ii = np.zeros((self.nion,self.nion))
      for i in range(self.nion):
        for j in range(self.nion):
          vid = sym_mtx_to_vec(i,j,self.nion) 
          self.collision_freq_ii[i,j] = 6.8e-8*np.sqrt(mui[j])/mui[i]\
              *(1+mui[j]/mui[i])*pwr(Ti[j],-3/2)\
              *ni[j]*sqr(Z[i]*Z[j])*self.coulomb_log_ii[vid]

  # Return NRL formulary units for collision quantity calcs
  def __nrl_collisions__(self,species:str) -> Tuple:
    ne = (self.ne/u.m**3).cgs.value
    Te = temperature_energy(self.Te,method='KtoeV')
    if species in ['ei','ie','ii']:
      Ti = temperature_energy(self.Ti,method='KtoeV')
      me = (sc.m_e*u.kg).cgs.value 
      mi = (self.mi*u.kg).cgs.value 
      mui = self.mi/sc.m_p
      ni = (self.ni/u.m**3).cgs.value
      return ne,Te,Ti,me,mi,mui,ni
    elif species == 'ee':
      return ne,Te
    else:
      raise Exception("species must be one of [\'ei\',\'ee\',\'ie\',\'ii\'].")

# Function for converting between eV and K using astropy
@typechecked
def temperature_energy(T:floats,method:str) -> floats:
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
@typechecked
def real_assert(val:floats,valcheck:floats,diff:floats,msg:Optional[str]=None):
  if msg is not None:
    assert val > valcheck - diff
    assert val < valcheck + diff
  else:
    assert val > valcheck - diff, msg
    assert val < valcheck + diff, msg

# Symmetric matrix entry represented by unique entry in 1D vector
# This routine returns the vector id for 2D index arguments
@typechecked
def sym_mtx_to_vec(i:ints,j:ints,n:ints) -> ints:
  if i <= j:
    return i*n-(i-1)*i//2+j-i;
  else:
    return j*n-(j-1)*j//2+i-j;

# Reverse to go from vector to symmetric matrix
@typechecked
def vec_to_sym_mtx(i:ints,n:ints) -> Tuple[ints,ints]:
  row = 0
  keyafter = -1
  while i >= keyafter:
    row += 1
    keyafter = row*n-(row-1)*row//2
  row -= 1
  col = n-keyafter+i
  return row,col

# Short numpy power routine
@typechecked
def pwr(arg:flint,power:flint) -> flint:
  return np.power(arg,power)

# Short numpy square
@typechecked
def sqr(arg:flint) -> flint:
  return pwr(arg,2)

# Checks numpy datatype is correct
@typechecked
def dtype_check(arr:np.ndarray,dtype:type):
  assert arr.dtype == dtype, f'numpy array {arr} must be {dtype} dtype'+\
      f' but is {arr.dtype}'

