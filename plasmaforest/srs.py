#!/bin/python3

from .core import *
from .laser import *
from .wave import *
from scipy.optimize import bisect
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import copy

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
    self.vg2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None

  # Set relativistic routine with nullifications
  def set_relativistic(self,relativistic:bool):
    self.relativistic = relativistic
    self.ldamping2 = None
    self.vg2 = None
    self.gamma = None
    self.rosenbluth = None

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
  def set_electrons(self,*args,**kwargs):
    super().set_electrons(*args,**kwargs)
    self.omega1 = None
    self.k1 = None
    self.omega2 = None
    self.k2 = None
    self.damping1 = None
    self.cdamping2 = None
    self.ldamping2 = None
    self.vg1 = None
    self.vg2 = None
    self.gamma0 = None
    self.gamma = None
    self.rosenbluth = None
  def set_ions(self,*args,**kwargs):
    super().set_ions(*args,**kwargs)
    self.damping1 = None
    self.cdamping2 = None
    self.gamma = None
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

  # Get matching wavenumbers and frequencies by either fluid or kinetic dispersion
  def resonance_solve(self,undamped:Optional[bool]=True):
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
      if self.relativistic:
        self.k2 = bisect(self.__bsrs_kinr__,self.k0,2*self.k0)
        self.omega2 = self.relativistic_dispersion(self.k2) # Undamped relativistic mode
        self.get_ldamping2()
      else:
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
  def __bsrs_kinr__(self,k2):
    omega_ek = self.relativistic_dispersion(k2) # Relativistic dispersion
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
  def get_ldamping2(self,force_kinetic:Optional[bool]=False):
    if self.omega2 is None or self.k2 is None:
      self.resonance_solve()
    if self.mode == 'fluid' and force_kinetic:
      omega = self.epw_kinetic_dispersion(self.k2,target='omega')
      self.ldamping2 = -np.imag(omega)
    else:
      self.ldamping2 = self.epw_landau_damping(self.omega2,self.k2,self.mode,self.relativistic)

  # Raman group velocity
  def get_vg1(self):
    if self.omega1 is None or self.k1 is None:
      self.resonance_solve()
    self.vg1 = self.emw_group_velocity(self.omega1,self.k1)

  # EPW group velocity
  def get_vg2(self):
    if self.omega2 is None or self.k2 is None:
      self.resonance_solve()
    if self.mode == 'fluid':
      self.vg2 = self.bohm_gross_group_velocity(self.omega2,self.k2)
    elif self.mode == 'kinetic':
      if self.relativistic:
        raise Exception('Relativistic EPW group velocity calc not implemented')
      else:
        self.vg2 = self.kinetic_group_velocity(self.omega2,self.k2)
    else:
      raise Exception('Mode must be one of fluid or kinetic')

  # Get k1 by EMW dispersion
  #def get_k1(self):

  # SRS gain coefficient calculations for various cases
  def get_gain_coeff(self):

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

    # Calculate constants
    K = np.abs(self.k2)*sqr(self.ompe)\
        /np.sqrt(8*sc.m_e*self.ne*self.omega0*self.omega1*self.omega2)
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    Kf = K/sqr(self.ompe)*np.abs(sqr(self.omega2)-prefac*sqr(self.vthe*self.k2))

    # Get gain coefficent for each case
    if self.mode == 'fluid':

      if self.sdl:
        if self.ldamping2 is None:
          self.get_ldamping2(force_kinetic=True)
        if self.nion > 0:
          if self.cdamping2 is None:
            self.get_cdamping2()
          nu2 = np.sum(self.cdamping2) + self.ldamping2
        else:
          nu2 = self.ldamping2

        res = self.bohm_gross_res(self.omega2,self.k2)*0.5*self.omega2
        nueff = nu2/(sqr(res)+sqr(nu2))
        self.gain_coeff = 2*sqr(K)*nueff/(pwr(sc.c,4)*np.abs(self.k0*self.k1))
        self.gain_coeff *= self.omega1*self.omega0

      else:
        self.gain_coeff = 2*Kf/(sqr(sc.c)*self.vthe*np.sqrt(prefac*\
            np.abs(self.k0*self.k1*self.k2)))

    elif self.mode == 'kinetic':

      if self.sdl:

        if self.relativistic:
          perm = self.relativistic_permittivity(self.omega2,self.k2)
        else:
          perm = self.kinetic_permittivity(self.omega2,self.k2,full=False)

        fac = -np.imag(1/perm)
        self.gain_coeff = 4*sqr(K)*self.omega2*fac/\
            (pwr(sc.c,4)*sqr(self.ompe)*np.abs(self.k0*self.k1))
        self.gain_coeff *= self.omega1*self.omega0

      else:
        if self.vg2 is None:
          self.get_vg2()
        self.gain_coeff = sqr(self.ompe)*self.k2/(sqr(sc.c)*np.sqrt(2*sc.m_e*\
            self.omega2*self.k0*np.abs(self.k1)*self.ne*self.vg2))

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

    fac = sc.e*self.k2*self.ompe*np.sqrt(1/\
        (8*self.omega0*self.omega1*self.omega2*sc.epsilon_0*self.k0))/(sc.c*sc.m_e)
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
  def get_rosenbluth(self,gradn):

    # Check relevant attributes assigned
    if self.gamma0 is None:
      self.get_gamma0()
    if self.vg1 is None:
      self.get_vg1()
    if self.vg2 is None:
      self.get_vg2()
    if self.vthe is None:
      self.get_vth(species='e')

    # Gradient of wavenumber mismatch
    dkmis = gradn*-0.5*sqr(sc.e)/(sc.m_e*sc.epsilon_0)*\
        (1/(sc.c**2*self.k0)-2/(3*self.vthe**2*self.k2)-1/(sc.c**2*self.k1))

    # Rosenbluth gain coefficient
    self.rosenbluth = 2*np.pi*sqr(self.gamma0)/np.abs(self.vg1*self.vg2*dkmis)

  # 1D BVP solve with parent forest setting resonance conditions
  def bvp_solve(self,I1_seed:float,xrange:tuple,nrange:tuple,ntype:str,points=101,\
      plots=False,pump_depletion=True):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('bvp_solve only works in strong damping limit; self.sdl must be True.')

    # Establish density profile
    x = np.linspace(xrange[0],xrange[1],points)
    if ntype == 'linear':
      m = (nrange[1]-nrange[0])/(xrange[1]-xrange[0])
      n = np.minimum(nrange[0] + np.maximum(x - xrange[0], 0) * m, nrange[1])
    elif ntype == 'exp':
      dr = abs(xrange[0]-xrange[1])
      Ln = dr/np.log(nrange[1]/nrange[0])
      r = abs(x-xrange[1])
      n = nrange[1]*np.exp(-r/Ln)
    else:
      raise Exception("ntype must be one of \'linear\' or \'exp\'")
    
    # Resonance solve on parent forest if not already
    if self.omega2 is None or self.omega1 is None \
        or self.k2 is None or self.k1 is None:
      self.resonance_solve(undamped=True)

    # Setup list of srs forests for each density relevant to parent forest
    birches = []
    for i in range(points):
      birches.append(srs_forest(self.mode,self.sdl,self.relativistic,\
                                self.lambda0,self.I0,self.ndim,\
                                electrons=self.electrons,nion=self.nion,\
                                Te=self.Te,ne=n[i],Ti=self.Ti,ni=self.ni,\
                                Z=self.Z,mi=self.mi))
      birches[i].set_frequencies(self.omega1,self.omega2)
      birches[i].get_k0()
      k1 = -birches[i].emw_dispersion(self.omega1,target='k')
      birches[i].set_wavenumbers(k1,birches[i].k0-k1)
      birches[i].get_gain_coeff()

    # Initialise wave action arrays
    I0 = np.ones_like(x)*self.I0/self.omega0
    I1 = np.ones_like(x)*I1_seed/self.omega1
    I0bc = I0[0]; I1bc = I1[-1]

    # Setup interpolation array for wave action gain
    gr = np.array([i.gain_coeff for i in birches])
    grf = PchipInterpolator(x,gr)
    
    if pump_depletion:
      # ODE evolution functions
      def Fsrs(xi,Iin):
        I0i, I1i = Iin
        gri = grf(xi)
        f1 = -gri*I0i*I1i
        f2 = -gri*I0i*I1i
        return np.vstack((f1,f2))
      def bc(ya,yb):
        return np.array([np.abs(ya[0]-I0bc),np.abs(yb[1]-I1bc)])

      # Solve bvp and convert to intensity for return
      y = np.vstack((I0,I1))
      res = solve_bvp(Fsrs,bc,x,y,tol=1e-10,max_nodes=1e5)
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
      if self.nc0 is None:
        self.get_nc0()
      fig, axs = plt.subplots(2,2,sharex='col',figsize=(12,12/1.618034))
      axs[0,0].plot(x*1e6,n/self.nc0)
      axs[0,0].set_ylabel('n_e/n_c')
      axs[0,1].plot(x*1e6,gr)
      axs[0,1].set_ylabel('Wave Gain [m/Ws^2]')
      axs[1,0].semilogy(x*1e6,I0)
      axs[1,0].set_ylabel('I0 [W/m^2]')
      axs[1,0].set_xlabel('x [um]')
      axs[1,1].semilogy(x*1e6,I1)
      axs[1,1].set_ylabel('I1 [W/m^2]')
      axs[1,1].set_xlabel('x [um]')
      fig.suptitle(f'Mode: {self.mode}; SDL: {self.sdl}; Relativistic: {self.relativistic}; '\
          +f'\nne ref: {self.ne:0.2e} m^-3; Te: {self.Te:0.2e} K; '\
          +f'\nI00: {self.I0:0.2e} W/m^2; lambda0: {self.lambda0:0.2e} m')
      plt.tight_layout()
      plt.show()

    return x,n,I0,I1,gr

  # Extension of BVP solver to include wave mixing from noise sources
  def wave_mixing_solve(self,I1_noise:float,xrange:tuple, \
      nrange:tuple,ntype:str,points=101,plots=False,pump_depletion=True,\
      I1_seed:Optional[float]=0.0,om1_seed:Optional[float]=None):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('Non-SDL wave-mixing solve not implemented.')

    # Establish density profile
    x = np.linspace(xrange[0],xrange[1],points)
    if ntype == 'linear':
      m = (nrange[1]-nrange[0])/(xrange[1]-xrange[0])
      n = np.minimum(nrange[0] + np.maximum(x - xrange[0], 0) * m, nrange[1])
    elif ntype == 'exp':
      dr = abs(xrange[0]-xrange[1])
      Ln = dr/np.log(nrange[1]/nrange[0])
      r = abs(x-xrange[1])
      n = nrange[1]*np.exp(-r/Ln)
    else:
      raise Exception("ntype must be one of \'linear\' or \'exp\'")

    # Resonance solve for each density point
    birches = []
    for i in range(points):
      birches.append(srs_forest(self.mode,self.sdl,self.relativistic,\
                                self.lambda0,self.I0,self.ndim,\
                                electrons=self.electrons,nion=self.nion,\
                                Te=self.Te,ne=n[i],Ti=self.Ti,ni=self.ni,\
                                Z=self.Z,mi=self.mi))
      birches[i].resonance_solve()
      birches[i].get_gain_coeff()

    # Get interpolated functions
    om0 = birches[0].omega0
    om1res = np.array([i.omega1 for i in birches])
    om1resf = PchipInterpolator(x,om1res)
    nef = PchipInterpolator(x,n)
    grres = np.array([i.gain_coeff for i in birches])
    grresf = PchipInterpolator(x,grres)

    # Non-resonant gain function
    def gain(xi,om1i):
      birch = copy.deepcopy(self)
      ne = float(nef(xi))
      birch.set_electrons(electrons=True,Te=birch.Te,ne=ne)
      birch.set_frequencies(om1i,om0-om1i)
      birch.get_k0()
      k1 = birch.emw_dispersion(om1i,target='k')
      birch.set_wavenumbers(k1,birch.k0+k1)
      birch.get_gain_coeff()
      return birch.gain_coeff

    # Check om1_seed input
    if om1_seed is None:
      om1_seed = birches[-1].omega1
      om1 = copy.deepcopy(om1res)
      om1f = PchipInterpolator(x,om1)
      I1_seed = 0.0
      gr = np.zeros_like(x)
      grf = PchipInterpolator(x,gr)
    elif om1_seed <= 0.0:
      raise Exception('Seed Raman frequency must be positive')
    else:
      gr = np.array([gain(x[i],om1_seed) for i in range(len(x))])
      grf = PchipInterpolator(x,gr)
      om1 = np.ones_like(x)*om1_seed
      om1f = PchipInterpolator(x,om1)

    # Initialise intensity arrays
    I0 = np.ones_like(x)*self.I0
    I1 = np.ones_like(x)*(I1_seed)
    I0bc = I0[0]; I1bc = I1[-1]

    if pump_depletion:
      # ODE evolution functions
      def Fsrs(xi,Ii):
        I0i, I1i = Ii
        # Establish forest and set quantitis
        #nonlocal om1mf
        om1m = om1f(xi)
        om1res = om1resf(xi)
        gr0 = grresf(xi)
        gri = grf(xi)
        #noisecont = grresf(xi[1:])*I1_noise*I0i[1:]/om0*np.diff(xi)
        #dI1 = np.zeros_like(I1i)
        #dI1[:-1] = I1i[:-1] - I1i[1:]
        #dI1[-1] = I1_seed
        #dI1[:-1] = np.maximum(0.0,dI1[:-1]-noisecont)
        #for i in range(len(xi)):
        #  if I1i[i] > 1e-10:
        #    om1m[i] = (np.sum(noisecont[i:]*om1res[1+i:]) \
        #        +np.sum(dI1[i:]*om1m[i:]))/I1i[i]
        #  else:
        #    om1m[i] = om1res[i]
        #om1m = (np.sum(noisecont*om1res[1:])+np.sum(dI1*om1m))/I1i
        #om1mf = PchipInterpolator(xi,om1m)
        #gri = np.array([gain(xi[i],om1m[i]) for i in range(len(xi))])
        f1 = -I0i*(gri/om1m*I1i+gr0/om1res*I1_noise)
        f2 = -I0i/om0*(gri*I1i+grresf(xi)*I1_noise)
        return np.vstack((f1,f2))
      def bc(ya,yb):
        return np.array([ya[0]-I0bc,yb[1]-I1bc])

      # Iteratively solve BVP and update frequencies
      conv = I1_noise
      while (conv > I1_noise/1000):
        I0old = I0[-1]
        I1old = I1[0]
        y = np.vstack((I0,I1))
        #res = solve_bvp(Fsrs,bc,x,y,tol=1e-10,max_nodes=1e5)
        res = solve_bvp(Fsrs,bc,x,y)#,tol=1e-10,max_nodes=1e5)
        I0 = res.sol(x)[0]
        I1 = res.sol(x)[1]#+I1_noise
        noisecont = grresf(x[1:])*I1_noise*I0[1:]/om0*np.diff(x)
        dI1 = np.zeros_like(I1)
        dI1[:-1] = I1[:-1] - I1[1:]
        dI1[-1] = I1_seed
        dI1[:-1] = np.maximum(0.0,dI1[:-1]-noisecont)
        for i in range(len(x)):
          if I1[i] > 100:
            om1[i] = (np.sum(noisecont[i:]*om1res[1+i:]) \
                +np.sum(dI1[i:]*om1[i:]))/I1[i]
          else:
            om1[i] = om1res[i]
        #plt.plot(x,om1res,label='resonant')
        #plt.plot(x,om1,label='mixed')
        #plt.legend()
        #plt.show()
        #om1m = (np.sum(noisecont*om1res[1:])+np.sum(dI1*om1m))/I1i
        om1mf = PchipInterpolator(x,om1)
        gr = np.array([gain(x[i],om1[i]) for i in range(len(x))])
        grf = PchipInterpolator(x,gr)
        conv = np.abs(I0[-1]-I0old)+np.abs(I1[0]-I1old)
        print(f'Convergence: {conv:0.2e}')
      '''
      y = np.vstack((I0,I1))
      res = solve_bvp(Fsrs,bc,x,y)#,tol=1e-10,max_nodes=1e5)
      I0 = res.sol(x)[0]
      I1 = res.sol(x)[1]
      '''
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
      I1 = res.sol(x)[0]

    if plots:
      if self.nc0 is None:
        self.get_nc0()
      fig, axs = plt.subplots(2,2,sharex='col',figsize=(12,12/1.618034))
      axs[0,0].plot(x*1e6,n/self.nc0)
      axs[0,0].set_ylabel('n_e/n_c')
      axs[0,1].plot(x*1e6,gr)
      axs[0,1].set_ylabel('Wave Gain [m/Ws^2]')
      axs[1,0].semilogy(x*1e6,I0)
      axs[1,0].set_ylabel('I0 [W/m^2]')
      axs[1,0].set_xlabel('x [um]')
      axs[1,1].semilogy(x*1e6,I1)
      axs[1,1].set_ylabel('I1 [W/m^2]')
      axs[1,1].set_xlabel('x [um]')
      fig.suptitle(f'Mode: {self.mode}; SDL: {self.sdl}; Relativistic: {self.relativistic}; '\
          +f'\nne ref: {self.ne:0.2e} m^-3; Te: {self.Te:0.2e} K; '\
          +f'\nI00: {self.I0:0.2e} W/m^2; lambda0: {self.lambda0:0.2e} m')
      plt.tight_layout()
      plt.show()

    return x,n,I0,I1,gr
