#!/bin/python3

from .core import *
from .laser import *
from .wave import *
from scipy.optimize import bisect

# SRS forest, three modes: laser, raman, epw
# Currently direct backscatter only
@typechecked
class srs_forest(laser_forest):
  def __init__(self,mode:str,sdl:bool,relativistic:bool,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.set_mode(mode)
    self.set_relativistic(relativistic)
    self.set_strong_damping_limit(sdl)

  # Check mode
  def __mode_check__(self,mode:str):
    if mode not in ['fluid','kinetic']:
      raise Exception('Mode argument must be one of \'fluid\' or \'kinetic\'')
    else:
      self.mode = mode

  # Set mode routine with nullifications
  def set_mode(self,mode:str):
    self.__mode_check__(mode)
    self.omega1 = None
    self.k1 = None
    self.omega2 = None
    self.k2 = None
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None

  def set_relativistic(self,relativistic:bool):
    self.relativistic = relativistic
    self.ldamping2 = None

  def set_strong_damping_limit(self,sdl:bool):
    self.sdl = sdl
    self.gain_coeff = None
    self.growth_rate = None
  # Update nullfications on inherited set routines
  def set_ndim(self,*args,**kwargs):
    super().set_ndim(*args,**kwargs)
    self.omega1 = None
    self.k1 = None
    self.omega2 = None
    self.k2 = None
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None
  def set_electrons(self,*args,**kwargs):
    super().set_electrons(*args,**kwargs)
    self.omega1 = None
    self.k1 = None
    self.omega2 = None
    self.k2 = None
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None
  def set_ions(self,*args,**kwargs):
    super().set_ions(*args,**kwargs)
    self.damping1 = None
    self.cdamping2 = None
  def set_intensity(self,*args,**kwargs):
    super().set_intensity(*args,**kwargs)

  # Set wavenumbers and frequencies manually
  def set_wavenumbers(self,k1:float,k2:float):
    # Set attributes
    self.k1 = k1
    self.k2 = k2
    self.ldamping2 = None
  def set_frequencies(self,omega1:float,omega2:float):
    # Set attributes
    self.omega1 = omega1
    self.omega2 = omega2
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None

  # Get matching wavenumbers and frequencies by either fluid or kinetic dispersion
  def resonance_solve(self,undamped:Optional[bool]=False):
    # Check omega0 and k0 already set
    if self.omega0 is None:
      self.get_omega0()
    if self.k0 is None:
      self.get_k0()

    if self.mode == 'fluid':
      # Solve for EPW wavenumber and set other unknowns bisecting fluid raman dispersion
      self.k2 = bisect(self.__bsrs__,self.k0,2*self.k0) # Look between k0 and 2k0
      self.omega2 = self.bohm_gross(self.k2,target='omega')

    elif self.mode == 'kinetic':
      # Solve similarly to above but replacing bohm-gross with linear kinetic dispersion
      if undamped:
        self.k2 = bisect(self.__bsrs_kinu__,self.k0,2*self.k0)
        self.omega2 = self.undamped_dispersion(self.k2) # Undamped mode
        self.get_ldamping2()
      else:
        self.k2 = bisect(self.__bsrs_kin__,self.k0,2*self.k0)
        omega2 = self.epw_kinetic_dispersion(self.k2,target='omega')
        self.ldamping2 = -np.imag(omega2)
        self.omega2 = np.real(omega2)

    # Lastly set raman quantities by matching conditions
    self.k1 = self.k0 - self.k2
    self.omega1 = self.omega0 - self.omega2


  # Raman dispersion residual from k2
  def __bsrs__(self,k2):
    omega_ek = np.real(self.bohm_gross(k2,target='omega'))
    return self.emw_dispersion_res((self.omega0-omega_ek),(self.k0-k2))
  # Raman dispersion residual from kinetic k2, undamped and natural modes
  def __bsrs_kin__(self,k2):
    omega_ek = np.real(self.epw_kinetic_dispersion(k2,target='omega')) # Natural mode
    return self.emw_dispersion_res((self.omega0-omega_ek),(self.k0-k2))
  def __bsrs_kinu__(self,k2):
    omega_ek = self.undamped_dispersion(k2) # Undamped mode
    return self.emw_dispersion_res((self.omega0-omega_ek),(self.k0-k2))

  # Raman collisional damping
  def get_damping1(self):
    if self.omega1 is None:
      self.resonance_solve()
    self.damping1 = self.emw_damping(self.omega1)

  # EPW collisional damping
  def get_cdamping2(self):
    if self.omega2 is None:
      self.resonance_solve()
    self.cdamping2 = self.epw_coll_damping(self.omega2)

  # EPW Landau damping
  def get_ldamping2(self):
    if self.omega2 is None or self.k2 is None:
      self.resonance_solve()
    self.ldamping2 = self.epw_landau_damping(self.omega2,self.k2,self.mode,self.relativistic)

  def get_gain_coeff(self):
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    if self.k0 is None:
      self.get_k0()
    if self.omega0 is None:
      self.get_omega0()
    if self.omega1 is None or self.omega2 is None \
        or self.k1 is None or self.k2 is None:
      self.resonance_solve()
    K = np.abs(self.k2)*sqr(self.ompe)\
        /np.sqrt(8*sc.m_e*self.ne*self.omega0*self.omega1*self.omega2)
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    Kf = K/sqr(self.ompe)*np.abs(sqr(self.omega2)-prefac*sqr(self.vthe*self.k2))
    if self.mode == 'fluid':
      if self.sdl:
        if self.ldamping2 is None:
          self.get_ldamping2()
        res = self.bohm_gross_res(self.omega2,self.k2)*0.5*self.omega2
        if self.nion > 0:
          if self.cdamping2 is None:
            self.get_cdamping2()
          nu2 = np.sum(self.cdamping2) + self.ldamping2
        else:
          nu2 = self.ldamping2
        nueff = np.sqrt(np.maximum(-sqr(res)+sqr(nu2),0))/(sqr(res)+sqr(nu2))
        self.gain_coeff = 2*sqr(K)*nueff/(pwr(sc.c,4)*np.abs(self.k0*self.k1))
      else:
        self.gain_coeff = 2*Kf/(sqr(c)*self.vthe*np.sqrt(prefac*np.abs(self.k0*self.k1*self.k2)))
    elif self.mode == 'kinetic':
      if self.sdl:
        perm = self.kinetic_permittivity(self.omega2,self.k2,full=False)
        fac = -np.imag(1/perm)
        self.gain_coeff = 4*sqr(K)*self.omega2*fac/\
            (pwr(sc.c,4)*sqr(self.ompe)*np.abs(self.k0*self.k1))
      else:
        raise Exception("Gain coefficient calc for non-SDL kinetic case not implemented.")





