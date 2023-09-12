#!/bin/python3

from .core import *
from .laser import *
from .wave import *
from scipy.optimize import bisect,minimize,minimize_scalar,Bounds,brentq
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import copy
import sys

# SRS forest, three modes: laser, raman, epw
# Currently direct backscatter only
@typechecked
class srs_forest(laser_forest):
  def __init__(self,mode:str,sdl:bool,relativistic:bool,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.set_mode(mode)
    self.set_relativistic(relativistic)
    self.set_strong_damping_limit(sdl)
    self.srs = True

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
    self.vg2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
    self.kappa1 = None
    self.srs = True

  # Set relativistic routine with nullifications
  def set_relativistic(self,relativistic:bool):
    self.relativistic = relativistic
    self.ldamping2 = None
    self.vg2 = None
    self.gamma = None
    self.rosenbluth = None
    self.srs = True

  # Set strong damping limit routine with nullifications
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
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
    self.kappa1 = None
  def set_electrons(self,*args,**kwargs):
    super().set_electrons(*args,**kwargs)
    self.k1 = None
    self.k2 = None
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None
    self.vg1 = None
    self.vg2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
    self.kappa1 = None
    self.srs = True
  def set_ions(self,*args,**kwargs):
    super().set_ions(*args,**kwargs)
    self.damping1 = None
    self.cdamping2 = None
    self.gamma = None
    self.kappa1 = None
  def set_intensity(self,*args,**kwargs):
    super().set_intensity(*args,**kwargs)
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None

  # Set wavenumbers and frequencies manually
  def set_wavenumbers(self,k1:float,k2:float):
    # Set attributes
    self.k1 = k1
    self.k2 = k2
    self.ldamping2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
    self.kappa1 = None
  def set_frequencies(self,omega1:float,omega2:float):
    # Set attributes
    self.omega1 = omega1
    self.omega2 = omega2
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
    self.kappa1 = None

  # Get matching wavenumbers and frequencies by either fluid or kinetic dispersion
  def resonance_solve(self,undamped:Optional[bool]=True):
    # Check omega0 and k0 already set
    if self.omega0 is None:
      self.get_omega0()
    if self.k0 is None:
      self.get_k0()
    if self.nc0 is None:
      self.get_nc0()
    if self.vthe is None:
      self.get_vth('e')

    failed = False
    if self.ne > 0.25*self.nc0:
      failed = True
    else:
      if self.mode == 'fluid':
        # Solve for EPW wavenumber and set other unknowns brentqing fluid raman dispersion
        try:
          self.k2 = brentq(self.__bsrs__,self.k0,2*self.k0,\
              xtol=4*np.finfo(float).eps) # Look between k0 and 2k0
          self.omega2 = self.bohm_gross(self.k2,target='omega')
        except:
          failed = True
      elif self.mode == 'kinetic':
        # Solve similarly to above but replacing bohm-gross with linear kinetic dispersion
        if self.relativistic:
          self.k2 = brentq(self.__bsrs_kinr__,self.k0,2*self.k0,xtol=4*np.finfo(float).eps)
          self.omega2 = self.relativistic_dispersion(self.k2) # Undamped relativistic mode
          self.get_ldamping2()
        else:
          if undamped:
            disbnd = 0.7546*self.ompe/self.vthe
            if self.k0 < disbnd:
              kbnd = np.minimum(2*self.k0,disbnd)
              try:
                self.k2 = brentq(self.__bsrs_kinu__,self.k0,kbnd,xtol=4*np.finfo(float).eps)
                self.omega2 = self.undamped_dispersion(self.k2,target='omega') # Undamped mode
                self.get_ldamping2()
              except:
                failed = True
            else:
              failed = True
          else:
            ubnd = self.omega0-np.real(self.epw_kinetic_dispersion(self.k0,target='omega'))
            lbnd = self.ompe
            if lbnd < ubnd:
              self.k2 = brentq(self.__bsrs_kin__,self.k0,2*self.k0,\
                  xtol=4*np.finfo(float).eps)
              omega2 = self.epw_kinetic_dispersion(self.k2,target='omega')
              self.ldamping2 = -np.imag(omega2)
              self.omega2 = np.real(omega2)
            else:
              failed = True

    # Lastly set raman quantities by matching conditions
    if failed:
      if self.verbose:
        print('resonance solve failed; no resonant srs backscatter at given conditions')
      self.srs = False
      self.omega1 = None; self.omega2 = None
      self.k1 = None; self.k2 = None
    else:
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
    omega_ek = self.undamped_dispersion(k2,target='omega') # Undamped mode
    return self.emw_dispersion_res((self.omega0-omega_ek),(self.k0-k2))
  def __bsrs_kinr__(self,k2):
    omega_ek = self.relativistic_dispersion(k2) # Relativistic dispersion
    return self.emw_dispersion_res((self.omega0-omega_ek),(self.k0-k2))

  def alt_resonance_solve(self):

    if self.omega0 is None:
      self.get_omega0()
    if self.k0 is None:
      self.get_k0()
    if self.nc0 is None:
      self.get_nc0()

    # Set frequency, calculate resulting gain
    def obj_fun(om1):
      self.omega1 = om1[0]
      self.omega2 = self.omega0 - om1[0]
      self.k1 = self.emw_dispersion(om1[0],target='k')
      self.k2 = self.k0 + self.k1
      self.get_gain_coeff()
      return -self.gain_coeff

    failed = False
    if self.ne > 0.25*self.nc0:
      failed = True
    else:
      if self.mode == 'kinetic':
        lbnd = np.maximum(self.ompe,\
            self.omega0-np.real(self.epw_kinetic_dispersion(2*self.k0,target='omega')))
        ubnd = self.omega0-np.real(self.epw_kinetic_dispersion(self.k0,target='omega'))
      else:
        lbnd = self.omega0-self.bohm_gross(2*self.k0,target='omega')
        ubnd = self.omega0-self.bohm_gross(self.k0,target='omega')
      failed = False
      if ubnd-1 < lbnd+1:
        failed = True
      else:
        # Optimise objective function
        k2 = self.k0 + self.omega0/sc.c*np.sqrt(1-2*self.ompe/self.omega0)
        om2 = self.bohm_gross(k2,target='omega')
        om1 = np.minimum(np.maximum(self.omega0 - om2,lbnd+1),ubnd-1)
        bnds = Bounds(lb=lbnd+1,ub=ubnd-1)
        res = minimize(obj_fun,om1,tol=1e-14,bounds=bnds,method='TNC')

    if failed:
      if self.verbose:
        print('resonance solve failed; no resonant srs backscatter at given conditions')
      self.srs = False
      self.omega1 = None; self.omega2 = None
      self.k1 = None; self.k2 = None

  # Raman collisional damping
  def get_damping1(self):
    if self.omega1 is None:
      self.resonance_solve()
    if self.cdampingx is None:
      self.get_cdampingx()
    self.damping1 = self.collisional_damping(self.cdampingx,self.omega1)

  # EPW collisional damping
  def get_cdamping2(self):
    if self.omega2 is None:
      self.resonance_solve()
    if self.cdampingx is None:
      self.get_cdampingx()
    self.cdamping2 = self.collisional_damping(self.cdampingx,self.omega2)

  # EPW Landau damping
  def get_ldamping2(self,force_kinetic:Optional[bool]=False):
    if self.omega2 is None or self.k2 is None:
      self.resonance_solve()
    if self.mode == 'fluid' and force_kinetic:
      if self.relativistic:
        omega = self.relativistic_dispersion(self.k2)
        self.ldamping2 = self.epw_landau_damping(np.real(omega),self.k2,'kinetic',self.relativistic)
      else:
        #omega = self.epw_kinetic_dispersion(self.k2,target='omega')
        #self.ldamping2 = -np.imag(omega)
        omega = self.undamped_dispersion(self.k2,target='omega')
        self.ldamping2 = self.epw_landau_damping(omega,self.k2,'kinetic',self.relativistic)
    else:
      self.ldamping2 = self.epw_landau_damping(self.omega2,self.k2,self.mode,self.relativistic)

  # Raman group velocity
  def get_vg1(self):
    if self.k1 is None:
      self.get_k1()
    self.vg1 = self.emw_group_velocity(self.omega1,self.k1)

  # EPW group velocity
  def get_vg2(self,force_fluid=True):
    if self.k2 is None:
      self.get_k2()
    if self.mode == 'fluid' or force_fluid:
      self.vg2 = self.bohm_gross_group_velocity(self.omega2,self.k2)
    elif self.mode == 'kinetic':
      if self.relativistic:
        raise Exception('Relativistic EPW group velocity calc not implemented')
      else:
        self.vg2 = self.kinetic_group_velocity(self.omega2,self.k2)
    else:
      raise Exception('Mode must be one of fluid or kinetic')

  # Get raman wavenumber
  def get_k1(self):
    if self.omega1 is None:
      self.resonance_solve()
    else:
      self.k1 = self.emw_dispersion(self.omega1,target='k')

  # Get EPW wavenumber
  def get_k2(self,undamped=True):
    if self.omega2 is None:
      self.resonance_solve(undamped)
    else:
      if self.relativistic:
        print('Warning: get_k2 method not implemented correctly for relativistic dispersion')
        self.k2 = self.bohm_gross(self.omega2,target='k')
      if self.mode == 'kinetic':
        if undamped:
          self.k2 = self.undamped_dispersion(self.omega2,target='k')
        else:
          self.k2 = self.epw_kinetic_dispersion(self.omega2,target='k')
      else:
        self.k2 = self.bohm_gross(self.omega2,target='k')

  # Raman critical density
  def get_nc1(self):
    if self.omega1 is None:
      self.resonance_solve()
    self.nc1 = self.emw_nc(self.omega1)

  # Raman spatial damping
  def get_kappa1(self):
    if self.vg1 is None:
      self.get_vg1()
    if self.damping1 is None:
      self.get_damping1()
    self.kappa1 = self.damping1/np.abs(self.vg1)

  # EPW spatial damping
  def get_kappa2(self):
    if self.vg2 is None:
      self.get_vg2()
    if self.damping2 is None:
      self.get_damping2()
    self.kappa2 = self.damping2/np.abs(self.vg2)

  # SRS gain coefficient calculations for various cases
  def get_gain_coeff(self,collisional=True,force_kinetic=False):

    # Check attributes set
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

    if self.srs:

      # Get gain coefficent for each case
      if self.mode == 'fluid':

        # Calculate constants
        K = np.abs(self.k2)*sqr(self.ompe)\
            /np.sqrt(8*sc.m_e*self.ne*self.omega0*self.omega1*self.omega2)
        gamma = (2+self.ndim)/self.ndim
        prefac = 0.5*gamma
        Kf = K/sqr(self.ompe)*np.abs(sqr(self.omega2)-prefac*sqr(self.vthe*self.k2))


        if self.sdl:
          if self.ldamping2 is None:
            self.get_ldamping2(force_kinetic=force_kinetic)
          if self.nion > 0 and collisional:
            if self.cdamping2 is None:
              self.get_cdamping2()
            nu2 = np.sum(self.cdamping2) + self.ldamping2
          else:
            nu2 = self.ldamping2

          res = self.bohm_gross_res(self.omega2,self.k2)*0.5*self.omega2
          nueff = nu2/(sqr(res)+sqr(nu2))
          #self.gain_coeff = 2*sqr(K)*nueff/(pwr(sc.c,4)*np.abs(self.k0*self.k1))
          #self.gain_coeff *= self.omega1*self.omega0
          fac = nueff*sqr(self.ompe)/self.omega2
          self.gain_coeff = sqr(sc.e/(sqr(sc.c)*sc.m_e))/(4*sc.epsilon_0)\
              *fac*sqr(self.k2)/np.abs(self.k0*self.k1)

        else:
          raise Exception('not implemented')
          #self.gain_coeff = 2*Kf/(sqr(sc.c)*self.vthe*np.sqrt(prefac*\
          #    np.abs(self.k0*self.k1*self.k2)))

      elif self.mode == 'kinetic':

        if self.sdl:

          #if self.ldamping2 is None:
          #  self.get_ldamping2()
          if self.nion > 0 and collisional:
            if self.cdamping2 is None:
              self.get_cdamping2()
          #  nu2 = np.sum(self.cdamping2) + self.ldamping2
          #else:
          #  nu2 = self.ldamping2

          if self.relativistic:
            perm = self.relativistic_permittivity(self.omega2,self.k2)
          else:
            try:
              perm = self.kinetic_permittivity(self.omega2,self.k2,full=False)
            except:
              if self.verbose:
                print('Warning: gain coefficient calculation failed')
              perm = 1+0j

          #res = np.real(perm)*sqr(self.ompe)/(2*self.omega2)
          #nueff = nu2/(sqr(res)+sqr(nu2))
          #fac = nueff*sqr(self.ompe)/self.omega2
          #self.gain_coeff = sqr(sc.e/(sqr(sc.c)*sc.m_e))/(4*sc.epsilon_0)\
              #*fac*sqr(self.k2)/np.abs(self.k0*self.k1)
          if collisional:
            perm += 1j*2*np.sum(self.cdamping2)/sqr(self.ompe)*self.omega2
          fac = np.imag(perm)/sqr(np.abs(perm))
          self.gain_coeff = sqr(sc.e/(sqr(sc.c)*sc.m_e))/(2*sc.epsilon_0)\
              *fac*sqr(self.k2)/np.abs(self.k0*self.k1)
        else:
          raise Exception('not implemented')
          #if self.vg2 is None:
          #  self.get_vg2()
          #self.gain_coeff = sqr(self.ompe)*self.k2/(sqr(sc.c)*np.sqrt(2*sc.m_e*\
          #    self.omega2*self.k0*np.abs(self.k1)*self.ne*self.vg2))
    else:
      self.gain_coeff = 0.0

    if np.isnan(self.gain_coeff):
      self.gain_coeff = 0.0
      

  # Get undamped infinite homogeneous growth rate
  def get_gamma0(self):
    if self.ompe is None:
      self.get_omp(species='e')
    if self.k0 is None:
      self.get_k0()
    if self.omega0 is None:
      self.get_omega0()
    if self.k1 is None or self.k2 is None \
        or self.omega1 is None or self.omega2 is None:
      self.resonance_solve()

    #fac = sc.e*self.k2*self.ompe*np.sqrt(1/\
        #(8*self.omega0*self.omega1*self.omega2*sc.epsilon_0*self.k0))/(sc.c*sc.m_e)
    #self.gamma0 = fac*np.sqrt(self.I0)
    self.get_vos0()
    self.gamma0 = self.k2*self.vos0/4*self.ompe/np.sqrt(self.omega1*self.omega2)

  # Get damped infinite homogeneous growth rate
  def get_gamma(self):
    if self.gamma0 is None:
      self.get_gamma0()
    if self.ldamping2 is None:
      self.get_ldamping2()
    if self.cdamping2 is None:
      self.get_cdamping2()
    if self.damping1 is None:
      self.get_damping1()
    nu1 = np.sum(self.damping1)
    nu2 = self.ldamping2 + np.sum(self.cdamping2)
    self.gamma = np.sqrt((nu1-nu2)**2/4+self.gamma0**2)-nu1/2-nu2/2

  # Get Rosenbluth coefficient
  def get_rosenbluth(self,gradn=None,gradT=None,force_kinetic=False,collisional=True):

    # Check relevant attributes assigned
    if self.gamma0 is None:
      self.get_gamma0()
    if self.vg1 is None:
      self.get_vg1()
    if self.vg2 is None:
      self.get_vg2(force_fluid=True)
    if self.vthe is None:
      self.get_vth(species='e')
    if self.ldamping2 is None:
      self.get_ldamping2(force_kinetic=force_kinetic)
    if collisional and self.nion > 0 and self.cdamping2 is None:
      self.get_cdamping2()

    if collisional:
      nu2 = self.ldamping2 + np.sum(self.cdamping2)
    else:
      nu2 = self.ldamping2
    
    if gradn is None and gradT is None:
      raise Exception('must provide one of gradn or gradT')

    # Gradient of wavenumber mismatch
    dkmis = 0.0
    if gradn is not None:
      dkmis += gradn*-0.5*sqr(sc.e)/(sc.m_e*sc.epsilon_0)*\
          (1/(sc.c**2*self.k0)-2/(3*self.vthe**2*self.k2)-1/(sc.c**2*self.k1))
    if gradT is not None:
      dkmis += self.k2/(4*self.Te)*gradT

    # Rosenbluth gain coefficient and resonance region
    self.rosenbluth = 2*np.pi*sqr(self.gamma0)/np.abs(self.vg1*self.vg2*dkmis)
    self.rose_region = 2*nu2/np.abs(self.vg2*dkmis)

  # Do resonance solve by Kruer formula i.e. assume omega2 equals ompe
  def kruer_resonance(self):
    if self.k0 is None:
      self.get_k0()

    self.k1 = self.omega0/sc.c*np.sqrt(1-2*self.ompe/self.omega0)
    self.omega2 = self.ompe
    self.omega1 = self.omega0 - self.omega2
    self.k2 = self.k0 + self.k1

  # Print frequency residual
  def frequency_matching(self,normalised=True):
    try:
      if normalised:
        return 1.0 - (self.omega1 + self.omega2)/self.omega0
      else:
        return self.omega0 - self.omega1 - self.omega2
    except:
      print('error: not all frequency attributes specified')

  # Print frequency residual
  def wavenumber_matching(self,normalised=True):
    try:
      if normalised:
        return 1.0 - (-self.k1 + self.k2)/self.k0
      else:
        return self.k0 + self.k1 - self.k2
    except:
      print('error: not all wavenumber attributes specified')

  # 1D BVP solve with parent forest setting resonance conditions
  def bvp_solve_simple(self,W0in:float,W1in:float,dx:float,gr:float,points=1000,\
      plots=False,pump_depletion=True):

    # Refine cell size and set grid points
    x = np.linspace(0,dx,points)
    
    # Initialise wave action arrays
    W0 = np.ones_like(x)*W0in
    W1 = np.ones_like(x)*W1in

    if pump_depletion:
      # ODE evolution functions
      def Fsrs(xi,Wi):
        W0i, W1i = Wi
        f1 = -gr*W0i*W1i
        f2 = -gr*W0i*W1i
        return np.vstack((f1,f2))
      def bc(ya,yb):
        return np.array([np.abs(ya[0]-W0in),np.abs(yb[1]-W1in)])

      # Solve bvp and convert to intensity for return
      y = np.vstack((W0,W1))
      res = solve_bvp(Fsrs,bc,x,y,tol=1e-3,max_nodes=1e6)
      if not res.success:
        print(res.message)
        raise Exception("Solver failed to find a solution")
      W0 = res.sol(x)[0]
      W1 = res.sol(x)[1]
    else:
      def Fsrs(xi,Wi):
        W1i = Wi
        f1 = -gr*W0in*W1i
        return f1
      def bc(ya,yb):
        return np.array([np.abs(yb[0]-W1in)])

      # Solve bvp and convert to intensity for return
      y = W1[np.newaxis,:]
      res = solve_bvp(Fsrs,bc,x,y,tol=1e-3,max_nodes=1e6)
      W1 = res.sol(x)[0]

    if plots:
      plt.plot(x,W0,label='Pump')
      plt.legend()
      plt.show()
      plt.plot(x,W1,label='Raman')
      plt.legend()
      plt.show()

    return x, W0, W1

  # 1D BVP solve with parent forest setting resonance conditions
  def bvp_solve(self,I1_seed:float,xrange:tuple,nrange:tuple,Trange:tuple,\
      ntype:str,points=1001,\
      plots=False,pump_depletion=True,errhndl=True,force_kinetic=False,\
      cutoff=True,cdamping=False):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('bvp_solve only works in strong damping limit; self.sdl must be True.')

    # Resonance solve on parent forest if not already
    if self.omega2 is None or self.omega1 is None \
        or self.k2 is None or self.k1 is None:
      self.resonance_solve(undamped=True)

    # Establish density profile
    x,n = den_profile(xrange,nrange,ntype,points)
    x,T = den_profile(xrange,Trange,ntype,points)

    # Get gain for Raman seed at each point in space
    gr = np.array([self.__gain__(n[i],self.omega1,Te=T[i],force_kinetic=force_kinetic,\
        collisional=cdamping) \
        for i in range(points)])
    if cdamping:
      cd0 = np.zeros_like(gr)
      cd1 = np.zeros_like(gr)
      cd2 = np.zeros_like(gr)
      for i in range(points):
        cd0[i], cd1[i], cd2[i] = self.__cdamping__(n[i],self.omega1,Te=T[i])
    if cutoff:
      nzeros = np.argwhere(gr != 0.0).flatten()
      x = x[nzeros]
      n = n[nzeros]
      gr = gr[nzeros]
      if cdamping:
        cd0 = cd0[nzeros]
        cd1 = cd1[nzeros]
        cd2 = cd2[nzeros]
    points = len(gr)
    grf = PchipInterpolator(x,gr)
    if cdamping:
      cd0f = PchipInterpolator(x,cd0)
      cd1f = PchipInterpolator(x,cd1)
      cd2f = PchipInterpolator(x,cd2)

    # Initialise wave action arrays
    I0 = np.ones_like(x)*self.I0/self.omega0
    I1 = np.ones_like(x)*I1_seed/self.omega1
    I0bc = I0[0]; I1bc = I1[-1]

    if pump_depletion:
      # ODE evolution functions
      if cdamping:
        def Fsrs(xi,Iin):
          I0i, I1i = Iin
          gri = grf(xi)
          cd0i = cd0f(xi)
          cd1i = cd1f(xi)
          f1 = -gri*I0i*I1i - I0i*cd0i
          f2 = -gri*I0i*I1i + I1i*cd1i
          return np.vstack((f1,f2))
      else:
        def Fsrs(xi,Iin):
          I0i, I1i = Iin
          gri = grf(xi)
          f1 = -gri*I0i*I1i
          return np.vstack((f1,f1))
      def bc(ya,yb):
        return np.array([np.abs(ya[0]-I0bc),np.abs(yb[1]-I1bc)])

      # Solve bvp and convert to intensity for return
      y = np.vstack((I0,I1))
      res = solve_bvp(Fsrs,bc,x,y,tol=1e-3,max_nodes=1e6)
      if not res.success and errhndl:
        print(res.message)
        raise Exception("Solver failed to find a solution")
      elif not res.success and self.verbose:
        print('Warning: solver unsuccesful')
        print(res.message)
      I0 = res.sol(x)[0]*self.omega0
      I1 = res.sol(x)[1]*self.omega1
    else:
      I0cons = I0[0]
      def Fsrs(xi,Iin):
        I1i = Iin
        gri = grf(xi)
        f1 = -gri*I0cons*I1i
        return f1
      def bc(ya,yb):
        return np.array([np.abs(yb[0]-I1bc)])

      # Solve bvp and convert to intensity for return
      y = I1[np.newaxis,:]
      res = solve_bvp(Fsrs,bc,x,y,tol=1e-10,max_nodes=1e5)
      I0 *= self.omega0
      I1 = res.sol(x)[0]*self.omega1

    if plots:
      self.__srs_plots__(x,n,gr,I0,I1)

    return x,n,I0,I1,gr

  # 1D BVP solve with parent forest setting resonance conditions
  def ray_trace_solve(self,xrange:tuple,nrange:tuple,ntype:str, \
      I1_noise:Optional[float]=0.0,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None,points:Optional[int]=101, \
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False,reversal=True):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('bvp_solve only works in strong damping limit; self.sdl must be True.')

    # Establish density profile
    x = np.linspace(xrange[0],xrange[1],points)
    dx = x[1]-x[0]
    cells = points - 1
    xc,n = den_profile(xrange,nrange,ntype,points,centred=True)
    
    # Resonance range
    if absorption:
      grres, om1res, ompe, k0, kappa0, dampingfac, roseg = \
          self.__resonance_range__(n,absorption)
    else:
      grres, om1res, ompe, k0, roseg = self.__resonance_range__(n,absorption)

    # Check seed inputs
    if I1_seed > 1e-10 and om1_seed is None:
      print('Seed Raman frequency not specified, defaulting to mid x-range resonant value')
      om1_seed = om1res[cells//2]
      seed = True
    elif I1_seed > 1e-10:
      seed = True
    else:
      om1_seed = om1res[-1]
      seed = False

    if I1_noise < 1e-153:
      noise = False
    else:
      noise = True
    
    # Initialise cell arrays and seed powers
    I0 = np.zeros_like(x)
    I1 = np.zeros_like(x)
    #I1n = I1_noise/om1res
    om0 = self.omega0
    om1 = np.ones_like(xc)
    gr = np.zeros_like(xc)
    P0 = self.I0/om0
    P1 = I1_seed/om1_seed

    # Raman ray class
    class rray:
      def __init__(self,cid,act,forest):
        self.cid = cid
        self.act = act
        self.forest = forest

    # Ray trace with SRS modelling
    conv = 1; niter = 0
    while (conv > 1e-4 and niter < 100) or niter < 10:
      # Initialisation
      I0old = copy.deepcopy(I0)
      I1old = copy.deepcopy(I1)
      I0[:] = 0.0
      I1[:] = 0.0

      # Add Raman Seed ray to list
      rrays = []
      if seed:
        forest = self.__raman_mode__(n[cells-1],om1_seed)
        rrays.append(rray(cells-1,P1,forest))
      
      # Launch laser ray
      P = P0
      for i in range(cells):

        # Cell update
        I0[i] += P

        # IB
        if absorption:
          P *= np.exp(-2*dx*kappa0[i])

        # SRS
        # Noise signal
        if noise:
          #exch = P*(1-np.exp(-grres[i]*I1n[i]*dx))
          if reversal:
            exch = I0[i]*1e-9*(np.exp(grres[i]*I0[i]*dx)-1)
          else:
            exch = P*(1-np.exp(-grres[i]*I0[i]*1e-9*dx))
          if exch > 1e-153:
            if pump_depletion:
              exch = np.minimum(exch,P)
              P = np.maximum(0.0,P-exch)
            if absorption:
              forest = self.__raman_mode__(n[i],om1res[i])
              forest.cdampingx = np.sum(dampingfac[:,i])
              forest.get_kappa1()
              exch *= np.exp(-2*forest.kappa1*dx)
            forest = self.__raman_mode__(n[i-1],om1res[i])
            rrays.append(rray(i-1,exch,forest))

        # Dominant signal
        if reversal:
          exch = I1old[i]/om1[i]*(np.exp(gr[i]*I0[i]*dx)-1)
        else:
          exch = P*(1-np.exp(-gr[i]*(I1old[i]/om1[i])*dx))
        if exch > 1e-153:
          if pump_depletion:
            exch = np.minimum(exch,P)
            P = np.maximum(0.0,P-exch)
          if absorption:
            forest = self.__raman_mode__(n[i],om1[i])
            forest.cdampingx = np.sum(dampingfac[:,i])
            forest.get_kappa1()
            exch *= np.exp(-2*forest.kappa1*dx)
          forest = self.__raman_mode__(n[i-1],om1[i])
          rrays.append(rray(i-1,exch,forest))


      # Update exit value
      I0[-1] += P

      # Push all raman rays out of domain
      for i in rrays:
        while i.cid >= 0:
          # Cell upate
          I1[i.cid] += i.act*i.forest.omega1
          om1[i.cid] += i.act*sqr(i.forest.omega1)

          # IB
          if absorption:
            i.forest.set_electrons(electrons=True,Te=self.Te,ne=n[i.cid])
            i.forest.ompe = ompe[i.cid]
            i.forest.cdampingx = np.sum(dampingfac[:,i.cid])
            i.forest.get_kappa1()
            i.act *= np.exp(-2*dx*i.forest.kappa1)

          # Propagate
          i.cid -= 1
          
        # Update exit value
        I1[-1] += i.act*i.forest.omega1

      # Update cell arrays for next iteration
      om1 = np.where(I1[:-1] < 1e-153, om1res, om1/I1[:-1])
      gr = np.array([self.__gain__(n[i],om1[i],ompe[i],k0[i]) for i in range(cells)])

      # Calculate convergence condition
      conv = np.sum(np.abs(I0old-I0))+np.sum(np.abs(I1old[:-1]-I1[:-1])/om1)
      niter += 1
      print(f'Iteration: {niter}; Convergence: {conv}')

    # Convert to intensity from wave action
    I0 *= self.omega0

    # Get Raman intensity array in proper order
    tmp = I1[-1]
    I1[1:] = I1[:-1]
    I1[0] = tmp
    I1 += I1_noise

    # Optionally plot
    if plots:
      self.__srs_plots__(x,n,gr,I0,I1,centred=True,xc=xc)

    return x,xc,n,I0,I1,gr

  # Ray trace solver with more flexibility in coordinates, temp and den profiles
  def ray_trace_solve2(self,x:np.ndarray,n:np.ndarray,Te:np.ndarray, \
      Ti:Optional[np.ndarray]=None,noise:Optional[bool]=True,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None, P0:Optional[float]=None,\
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False,geometry:Optional[str]='planar',flip_launch=False,\
      addnoise=True):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('bvp_solve only works in strong damping limit; self.sdl must be True.')

    if flip_launch:
      x = np.flip(x)
      n = np.flip(n)
      Te = np.flip(Te)
      Ti = np.flip(Ti)

    # Cell counts and centers
    points = len(x)
    cells = len(n)
    xc = np.array([(x[i]+x[i+1])/2 for i in range(cells)])

    # Density gradient
    #nlog = np.log(n)
    #fwd = (nlog[1:]-nlog[:-1])/(xc[1:]-xc[:-1])
    fwd = (n[1:]-n[:-1])/(xc[1:]-xc[:-1])
    gradn = np.zeros_like(n)
    gradn[0] = fwd[0]
    gradn[-1] = fwd[-1]
    gradn[1:-1] = (fwd[:-1]+fwd[1:])/2
    gradn = -gradn
    #gradn *= n
    
    # Resonance range
    if absorption:
      grres, om1res, ompe, k0, kappa0, dampingfac, roseg = \
          self.__resonance_range__(n,absorption,Te,Ti,gradn=gradn)
    else:
      grres, om1res, ompe, k0, roseg = self.__resonance_range__(n,absorption,Te,gradn=gradn)

    
    #print(om1res)
    #plt.plot(xc,om1res)
    #plt.show()
    #plt.plot(xc,om1res)
    #plt.show()
    #plt.plot(xc,n)
    #plt.show()

    # Binning arrays for frequency spectrum
    maxom1 = np.max(om1res)
    minom1 = np.min(om1res)
    nbins = cells // 2
    binbnds = np.linspace(minom1,maxom1,nbins+1)
    pwrbin = np.zeros(nbins)
    freqbin = (binbnds[1:]+binbnds[:-1])/2

    # Check seed inputs
    if I1_seed > 1e-10 and om1_seed is None:
      print('Seed Raman frequency not specified, defaulting to mid x-range resonant value')
      om1_seed = om1res[cells//2]
      seed = True
    elif I1_seed > 1e-10:
      seed = True
    else:
      om1_seed = om1res[-1]
      seed = False

    # Cell volumes 
    dr = np.abs(np.diff(x))
    if geometry == 'planar':
      drV = np.ones_like(dr)
    elif geometry == 'cylindrical':
      V = np.array([np.pi*(np.abs(np.abs(x[i])**2-np.abs(x[i+1])**2)) for i in range(points-1)])
      drV = dr/V
    elif geometry == 'spherical':
      V = np.array([4/3*np.pi*(np.abs(x[i]**3-x[i+1]**3)) for i in range(points-1)])
      drV = dr/V
    
    # Initialise cell arrays and seed powers
    I0 = np.zeros_like(xc)
    I1 = np.zeros_like(xc)
    om1res = np.where(om1res > 1e-153, om1res, 0.0)
    om0 = self.omega0
    om1 = np.ones_like(xc)
    gr = np.zeros_like(xc)
    if P0 is None:
      P0 = self.I0
    grres = np.where(roseg*P0*drV/self.omega0 < np.log(1), 0.0, grres)
    #if I1_noise < 1e-153:
    #  P1n = P0*1e-9
    #else:
    #  P1n = I1_noise/drV
    #I1n = np.where(om1res > 1e-153, P1n*drV/np.maximum(om1res,1e-153), 0.0)

    # Laser ray class
    class lray:
      def __init__(self,cid,pwr,dire):
        self.cid = cid
        self.pwr = pwr
        self.dire = dire
    # Raman ray class
    class rray:
      def __init__(self,cid,pwr,dire,forest):
        self.cid = cid
        self.pwr = pwr
        self.dire = dire
        self.forest = forest

    # Ray trace with SRS modelling
    print('starting ray tracing')
    conv = 2; niter = 0; ra_frac = 1.0; ema = 0.0
    while (conv > 1 and niter < 30):
      # Initialisation
      nnzero = 0
      I0old = copy.deepcopy(I0)
      I1old = copy.deepcopy(I1)
      I0[:] = 0.0
      I1[:] = 0.0

      # Add Raman Seed ray to list
      rrays = []
      if seed:
        forest = self.__raman_mode__(n[cells-1],om1_seed,Te[cells-1])
        rrays.append(rray(cells-1,I1_seed/drV[-1],-1,forest))
      
      # Launch laser ray
      if flip_launch:
        l = lray(cells-1,P0,-1)
      else:
        l = lray(0,P0,1)
      while (l.cid < cells and l.cid >= 0):

        # Cell update
        lfac = drV[l.cid]/self.omega0
        Wcell = l.pwr*lfac
        I0[l.cid] += Wcell

        # IB
        if absorption:
          #l.pwr *= np.exp(-2*kappa0[l.cid]*dr[l.cid])
          l.pwr = l.pwr-l.pwr*(1-np.exp(-2*kappa0[l.cid]*dr[l.cid]))
          Wcell = l.pwr*lfac

        # SRS
        # Noise signal
        if noise and grres[l.cid] > 1e-153:
          ramamp = np.minimum(grres[l.cid]*I0[l.cid]*dr[l.cid]\
              ,roseg[l.cid]*I0[l.cid]) # GP replaces this
          ramamp = np.minimum(ramamp,np.log(1e9*om1res[l.cid]/self.omega0))
          if absorption:
            forest = self.__raman_mode__(n[l.cid],om1res[l.cid])
            forest.cdampingx = np.sum(dampingfac[:,l.cid])
            forest.get_kappa1()
            ramabs = 2*np.sum(forest.kappa1)*dr[l.cid]
          else:
            ramabs = 0.0
          # Min amplification threshold
          #cthresh = np.log(1e5)
          cthresh = 0.0
          if ramamp > np.maximum(ramabs,cthresh):
            #exch = Wcell*(1-np.exp(-grres[l.cid]*I0[l.cid]*1e-9*dr[l.cid]))
            exch = I0[l.cid]*1e-9*(np.exp(ramamp)-1)
            pact = exch/drV[l.cid]
            if pump_depletion:
              exchl = np.minimum(pact*self.omega0,l.pwr)
              l.pwr = np.maximum(0.0,l.pwr-exchl)
              Wcell = l.pwr*lfac
            if absorption:
              pact *= np.exp(-ramabs)
            rcid = l.cid-l.dire
            if rcid < cells:
              forest = self.__raman_mode__(n[rcid],om1res[l.cid],Te[rcid])
              rrays.append(rray(rcid,pact*om1res[l.cid],-l.dire,forest))

        # Dominant signal
        if gr[l.cid] > 1e-153:
          ramamp = np.minimum(gr[l.cid]*I0[l.cid]*dr[l.cid],\
              roseg[l.cid]*I0[l.cid])
          ramamp = np.minimum(ramamp,np.log(I0[l.cid]/I1old[l.cid]))
          ramamp = np.maximum(ramamp, 0.0)
          if absorption:
            forest = self.__raman_mode__(n[l.cid],om1[l.cid])
            forest.cdampingx = np.sum(dampingfac[:,l.cid])
            forest.get_kappa1()
            ramabs = 2*forest.kappa1*dr[l.cid]
          else:
            ramabs = 0.0
          # Min amplification threshold
          #cthresh = np.log(10)
          cthresh = 0.0
          if ramamp > np.maximum(ramabs,cthresh):
            exch = I1old[l.cid]*(np.exp(ramamp)-1)
            pact = exch/drV[l.cid]
            if pump_depletion:
              exchl = np.minimum(pact*self.omega0,l.pwr)
              l.pwr = np.maximum(0.0,l.pwr-exchl)
            if absorption:
              pact *= np.exp(-ramabs)
            rcid = l.cid-l.dire
            if rcid < cells:
              forest = self.__raman_mode__(n[rcid],om1[l.cid],Te[rcid])
              rrays.append(rray(rcid,pact*om1[l.cid],-l.dire,forest))

        if l.pwr < 1e-153:
          break

        l.cid += l.dire

      # Update exit value
      #I0[-1] += l.pwr*drV[-1]/self.omega0

      # Push all raman rays out of domain
      for r in rrays:
        while (r.cid < cells and r.cid >= 0):

          if n[r.cid] > r.forest.nc1:
            r.pwr *= 1-ra_frac
            r.dire = -r.dire
          else:
            # Cell update
            rfac = drV[r.cid]/r.forest.omega1
            Wcell = r.pwr*rfac
            I1[r.cid] += Wcell*r.forest.omega1
            om1[r.cid] += Wcell*sqr(r.forest.omega1)

            # IB
            if absorption:
              r.forest.set_electrons(electrons=True,Te=Te[r.cid],ne=n[r.cid])
              r.forest.ompe = ompe[r.cid]
              r.forest.cdampingx = np.sum(dampingfac[:,r.cid])
              r.forest.get_kappa1()
              r.pwr *= np.exp(-2*r.forest.kappa1*dr[r.cid])
              Wcell = r.pwr*rfac

            # Lower power threshold
            if r.pwr < 1e-153:
              break

          # Propagate
          r.cid += r.dire
          
        # Bin ray
        for i in range(nbins):
          if r.forest.omega1 < binbnds[i+1]:
            pwrbin[i] += r.pwr
            break

      # Update cell arrays for next iteration
      for i in range(cells):
        if I0[i] > 1e-153:
          nnzero += 1
        if I1[i] > 1e-153:
          nnzero += 1
          om1[i] /= I1[i]
          gr[i] = self.__gain__(n[i],om1[i],ompe[i],k0[i],Te[i])
          I1[i] /= om1[i]
        else:
          om1[i] = 0.0
          gr[i] = 0.0
          I1[i] = 0.0

      # Calculate convergence condition
      conv = (np.sum(np.abs(I0old-I0))+np.sum(np.abs(I1old-I1)))/nnzero
      emaold = ema
      ema = 0.4*conv + 0.6*ema
      niter += 1
      print(f'Iteration: {niter}; Convergence: {conv}; EMA {ema}')
      #plt.semilogy(xc,I1old,label='old')
      #plt.semilogy(xc,I1,label='old')
      #plt.show()
      if np.abs(emaold-ema) < 1:
        break
      print('')

    # Convert to intensity from wave action
    I0 *= self.omega0
    I1 *= om1 
    if addnoise:
      I1 += I0*1e-9

    # Optionally plot
    #if plots:
    #  self.__srs_plots__(x,n,gr,I0,I1,centred=True,xc=xc)

    return x,xc,n,I0,I1,gr,om1,grres,roseg,freqbin,pwrbin

  def ray_trace_solve3(self,xrange:tuple,nrange:tuple,ntype:str, \
      I1_noise:Optional[float]=0.0,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None,points:Optional[int]=101, \
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False):

    # Establish density profile
    x = np.linspace(xrange[0],xrange[1],points)
    xc,n = den_profile(xrange,nrange,ntype,points,centred=True)
    Te = np.ones_like(n)*self.Te
    if self.Ti is not None:
      Ti = np.ones_like(n)*self.Ti
    else:
      Ti = None
    P0 = self.I0#/self.omega0

    return self.ray_trace_solve2(x,n,Te,Ti,I1_noise,I1_seed,om1_seed,P0,plots,\
        pump_depletion,absorption,'planar')


  # SRS gain function for any density and Raman frequency
  def __gain__(self,ne:float,om1:float,ompe:Optional[float]=None,\
      k0:Optional[float]=None,Te:Optional[float]=None,\
      force_kinetic:Optional[bool]=False,collisional:Optional[bool]=True):
    birch = self.__raman_mode__(ne,om1,Te)
    if k0 is None:
      birch.get_k0()
    else:
      birch.k0 = k0
    if ompe is not None:
      birch.ompe = ompe
    if om1 < birch.ompe:
      birch.gain_coeff = 0.0
    else:
      k1 = -birch.emw_dispersion(om1,target='k')
      birch.set_wavenumbers(k1,birch.k0-k1)
      birch.get_gain_coeff(force_kinetic=force_kinetic,collisional=collisional)
    return birch.gain_coeff

  # Collisional damping for each mode
  def __cdamping__(self,ne:float,om1:float,ompe:Optional[float]=None,\
      k0:Optional[float]=None,Te:Optional[float]=None):
    birch = self.__raman_mode__(ne,om1,Te)
    if k0 is None:
      birch.get_k0()
    else:
      birch.k0 = k0
    if ompe is not None:
      birch.ompe = ompe
    if om1 < birch.ompe:
      birch.get_kappa0()
      birch.kappa1 = 0.0
      birch.get_cdamping2()
    else:
      k1 = -birch.emw_dispersion(om1,target='k')
      birch.set_wavenumbers(k1,birch.k0-k1)
      birch.get_kappa0()
      birch.get_kappa1()
      birch.get_cdamping2()
    return birch.kappa0, birch.kappa1, birch.cdamping2

  # k2calcs
  def k2_calcs(self,ne:float,om1:float):
    birch = self.__raman_mode__(ne,om1,None)
    birch.get_k2()
    return birch.k2

  # Profile calcs
  def profile_calcs(self,ne:float,om1:float):
    birch = self.__raman_mode__(ne,om1,None)
    birch.get_omp('e')
    birch.get_vth('e')
    birch.get_ldamping2()
    birch.get_gain_coeff()
    perm = birch.kinetic_permittivity(birch.omega2,birch.k2)
    dampingfac = np.imag(perm)/np.abs(perm)**2
    return birch.k2, birch.gain_coeff, dampingfac, \
        birch.ompe, birch.vthe, birch.ldamping2

  # Resonance solve across a density range
  def __resonance_range__(self,n:np.ndarray,absorption:Optional[bool]=False,\
      Te:Optional[np.ndarray]=None,Ti:Optional[np.ndarray]=None,\
      gradn:Optional[np.ndarray]=None):
    birches = []
    for i in range(len(n)):
      birches.append(copy.deepcopy(self))
      if Te is None:
        birches[i].set_electrons(electrons=True,Te=self.Te,ne=n[i])
      else:
        birches[i].set_electrons(electrons=True,Te=Te[i],ne=n[i])
      if gradn is not None:
        alder = copy.deepcopy(birches[i])
        alder.mode = 'fluid'
        alder.set_intensity(1)
        alder.resonance_solve()
        alder.get_rosenbluth(gradn=gradn[i])
        birches[i].rosenbluth = alder.rosenbluth*alder.omega0
      birches[i].resonance_solve()
      if birches[i].omega1 is None:
        birches[i].omega1 = 0.0
        birches[i].gain_coeff = 0.0
      else:
        birches[i].get_gain_coeff()
      if absorption:
        if Ti is not None:
          birches[i].set_ions(nion=self.nion,Ti=Ti[i]*np.ones(self.nion),\
              ni=n[i]/self.Z,Z=self.Z,mi=self.mi)
        birches[i].get_kappa0()
    om1res = np.array([i.omega1 for i in birches])
    grres = np.array([i.gain_coeff for i in birches])
    ompe = np.array([i.ompe for i in birches])
    k0 = np.array([i.k0 for i in birches])
    roseg = np.array([i.rosenbluth for i in birches])
    if absorption:
      kappa0 = np.array([np.sum(i.kappa0) for i in birches])
      dampingfac = np.zeros((self.nion,len(kappa0)))
      for i,j in enumerate(birches):
        dampingfac[:,i] = j.cdampingx
      return grres, om1res, ompe, k0, kappa0, dampingfac, roseg
    else:
      return grres, om1res, ompe, k0, roseg

  # Creates copy of parent forest with new raman frequency and reference density
  def __raman_mode__(self,ne,om1,Te:Optional[float]=None):
    forest = copy.deepcopy(self)
    if Te is None:
      forest.set_electrons(electrons=True,ne=ne,Te=self.Te)
    else:
      forest.set_electrons(electrons=True,ne=ne,Te=Te)
    forest.set_frequencies(om1,self.omega0-om1)
    forest.get_k0()
    k1 = forest.emw_dispersion(om1,target='k')
    forest.set_wavenumbers(k1,forest.k0+k1)
    forest.get_nc1()
    return forest

  # Standard plotting routine
  def __srs_plots__(self,x:np.ndarray,n:np.ndarray,gr:np.ndarray, \
                    I0:np.ndarray,I1:np.ndarray, \
                    xc:Optional[np.ndarray]=None,centred:Optional[bool]=False):
    if self.nc0 is None:
      self.get_nc0()
    fig, axs = plt.subplots(2,2,sharex='col',figsize=(12,12/1.618034))
    if centred:
      axs[0,0].plot(xc*1e6,n/self.nc0)
    else:
      axs[0,0].plot(x*1e6,n/self.nc0)
    axs[0,0].set_ylabel('n_e/n_c')
    if centred:
      axs[0,1].plot(xc*1e6,gr)
    else:
      axs[0,1].plot(x*1e6,gr)
    axs[0,1].set_ylabel('Wave Gain [m/J]')
    axs[1,0].semilogy(x*1e6,I0)
    axs[1,0].set_ylabel('I0 [W/m\^2]')
    axs[1,0].set_xlabel('x [um]')
    axs[1,1].semilogy(x*1e6,I1)
    axs[1,1].set_ylabel('I1 [W/m\^2]')
    axs[1,1].set_xlabel('x [um]')
    fig.suptitle(f'Mode: {self.mode}; SDL: {self.sdl}; Relativistic: {self.relativistic}; '\
        +f'\nTe: {self.Te:0.2e} K; I00: {self.I0:0.2e} W/m\^2; lambda0: {self.lambda0:0.2e} m')
    plt.tight_layout()
    plt.show()
