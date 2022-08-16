#!/bin/python3

from .core import *
from .laser import *
from .wave import *
from scipy.optimize import bisect,minimize,minimize_scalar
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
    self.kappa1 = None

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

  def alt_resonance_solve(self):

    if self.omega0 is None:
      self.get_omega0()
    if self.k0 is None:
      self.get_k0()

    # Set frequency, calculate resulting gain
    def obj_fun(om1):
      self.omega1 = om1
      self.omega2 = self.omega0 - om1
      self.k1 = self.emw_dispersion(om1,target='k')
      self.k2 = self.k0 + self.k1
      self.get_gain_coeff()
      return -self.gain_coeff

    # Optimise objective function
    k2 = self.k0 + self.omega0/sc.c*np.sqrt(1-2*self.ompe/self.omega0)
    om2 = self.bohm_gross(k2,target='omega')
    om1 = self.omega0 - om2
    res = minimize(obj_fun,om1,tol=1e-14,method="Nelder-Mead")

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
    if self.k1 is None:
      self.get_k1()
    self.vg1 = self.emw_group_velocity(self.omega1,self.k1)

  # EPW group velocity
  def get_vg2(self):
    if self.k2 is None:
      self.get_k2()
    if self.mode == 'fluid':
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
  def get_k2(self):
    if self.omega2 is None:
      self.resonance_solve()
    else:
      if relativistic:
        raise Exception('get_k2 method not implemented for relativistic dispersion')
      if mode == 'kinetic':
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
    self.kappa1 = self.damping1/self.vg1

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

    # Resonance solve on parent forest if not already
    if self.omega2 is None or self.omega1 is None \
        or self.k2 is None or self.k1 is None:
      self.resonance_solve(undamped=True)

    # Establish density profile
    x,n = den_profile(xrange,nrange,ntype,points)
    
    # Get gain for Raman seed at each point in space
    gr = np.array([self.__gain__(n[i],self.omega1) for i in range(points)])
    grf = PchipInterpolator(x,gr)

    # Initialise wave action arrays
    I0 = np.ones_like(x)*self.I0/self.omega0
    I1 = np.ones_like(x)*I1_seed/self.omega1
    I0bc = I0[0]; I1bc = I1[-1]

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
      self.__srs_plots__(x,n,gr,I0,I1)

    return x,n,I0,I1,gr

  # 1D BVP solve with parent forest setting resonance conditions
  def ray_trace_solve(self,xrange:tuple,nrange:tuple,ntype:str, \
      I1_noise:Optional[float]=0.0,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None,points:Optional[int]=101, \
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False):

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
      grres, om1res, ompe, k0, kappa0 = self.__resonance_range__(n,absorption)
    else:
      grres, om1res, ompe, k0 = self.__resonance_range__(n,absorption)

    # Check seed inputs
    if I1_seed > 1e-10 and om1_seed is None:
      print('Seed Raman frequency not specified, defaulting to mid x-range resonant value')
      om1_seed = om1res[points//2]
      seed = True
    elif I1_seed > 1e-10:
      seed = True
    else:
      om1_seed = om1res[-1]
      seed = False
    
    # Initialise cell arrays and seed powers
    I0 = np.zeros_like(x)
    I1 = np.zeros_like(x)
    I1n = I1_noise/om1res
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
    while conv > 1e-2 and niter < 100:
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
        # Dominant signal
        exch = gr[i]*P*(I1old[i]/om1[i])*dx
        if exch > 1e-153:
          forest = self.__raman_mode__(n[i-1],om1[i])
          rrays.append(rray(i-1,exch,forest))
          if pump_depletion:
            P = np.maximum(0.0,P-exch)

        # Noise signal
        exch = grres[i]*P*I1n[i]*dx
        if exch > 1e-153:
          forest = self.__raman_mode__(n[i-1],om1res[i])
          rrays.append(rray(i-1,exch,forest))
          if pump_depletion:
            P = np.maximum(0.0,P-exch)

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

    # Optionally plot
    if plots:
      self.__srs_plots__(x,n,gr,I0,I1,centred=True,xc=xc)

    return x,xc,n,I0,I1,gr

  # Ray trace solver with more flexibility in coordinates, temp and den profiles
  def ray_trace_solve2(self,x:np.ndarray,n:np.ndarray,Te:np.ndarray, \
      I1_noise:Optional[float]=0.0,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None, P0:Optional[float]=None, \
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False,geometry:Optional[str]='planar'):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('bvp_solve only works in strong damping limit; self.sdl must be True.')

    # Cell counts and centers
    points = len(x)
    cells = len(n)
    xc = np.array([(x[i]+x[i+1])/2 for i in range(cells)])
    
    # Resonance range
    if absorption:
      grres, om1res, ompe, k0, kappa0 = self.__resonance_range__(n,absorption,Te)
    else:
      grres, om1res, ompe, k0 = self.__resonance_range__(n,absorption,Te)

    # Check seed inputs
    if I1_seed > 1e-10 and om1_seed is None:
      print('Seed Raman frequency not specified, defaulting to mid x-range resonant value')
      om1_seed = om1res[points//2]
      seed = True
    elif I1_seed > 1e-10:
      seed = True
    else:
      om1_seed = om1res[-1]
      seed = False
    
    # Initialise cell arrays and seed powers
    I0 = np.zeros_like(x)
    I1 = np.zeros_like(x)
    I1n = I1_noise/om1res
    om0 = self.omega0
    om1 = np.ones_like(xc)
    gr = np.zeros_like(xc)
    if P0 is None:
      P0 = self.I0#/om0
    P1 = I1_seed#om1_seed

    # Cell volumes 
    dr = np.diff(x)
    if geometry == 'planar':
      drV = np.ones_like(dr)
    elif geometry == 'cylindrical':
      V = np.array([np.pi*(x[i]**2-x[i-1]**2) for i in range(1,points)])
      drV = dr/V
    elif geometry == 'spherical':
      V = np.array([4/3*np.pi*(x[i]**3-x[i-1]**3) for i in range(1,points)])
      drV = dr/V

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
    conv = 1; niter = 0; ra_frac = 0.3
    while conv > 1e-2 and niter < 100:
      # Initialisation
      I0old = copy.deepcopy(I0)
      I1old = copy.deepcopy(I1)
      I0[:] = 0.0
      I1[:] = 0.0

      # Add Raman Seed ray to list
      rrays = []
      if seed:
        forest = self.__raman_mode__(n[cells-1],om1_seed,Te[cells-1])
        rrays.append(rray(cells-1,P1,forest))
      
      # Launch laser ray
      l = lray(0,P0,1)
      while (l.cid < cells and l.cid >= 0):

        if n[l.cid] > self.nc0:
          l.pwr *= 1-ra_frac
          l.dire = -l.dire
        else:
          # Cell update
          lfac = drV[l.cid]/self.omega0
          Wcell = l.pwr*lfac
          I0[l.cid] += Wcell

          # IB
          if absorption:
            l.pwr *= 1-np.minimum(kappa0[l.cid]*dr[l.cid],1)
            Wcell = l.pwr*lfac

          # SRS
          # Dominant signal
          exch = np.minimum(gr[l.cid]*Wcell*(I1old[l.cid]/om1[l.cid])*dr[l.cid],Wcell)
          if exch > 1e-153:
            Wcell -= exch
            pact = exch*drV[l.cid]
            forest = self.__raman_mode__(n[l.cid-1],om1[l.cid],Te[l.cid-1])
            rrays.append(rray(l.cid-1,pact*om1[l.cid],-l.dire,forest))
            if pump_depletion:
              l.pwr = l.pwr-pact*self.omega0

          # Noise signal
          exch = np.minimum(gr[l.cid]*Wcell*I1n[l.cid]*dr[l.cid],Wcell)
          if exch > 1e-153:
            pact = exch*drV[l.cid]
            forest = self.__raman_mode__(n[l.cid-1],om1res[l.cid],Te[l.cid-1])
            rrays.append(rray(l.cid-1,pact*om1[l.cid],-l.dire,forest))
            if pump_depletion:
              l.pwr = l.pwr-pact*self.omega0

        if l.pwr < 1e-153:
          break

        l.cid += l.dire

      # Update exit value
      I0[-1] += l.pwr*drV[-1]/self.omega0

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
              r.forest.get_kappa1()
              r.pwr *= 1-np.minimum(r.forest.kappa1*dr[r.cid],1)
              Wcell = r.pwr*rfac

            # Lower power threshold
            if r.pwr < 1e-153:
              break

          # Propagate
          r.cid -= 1
          
        # Update exit value
        I1[-1] += r.pwr*drV[-1]

      # Update cell arrays for next iteration
      om1 = np.where(I1[:-1] < 1e-153, om1res, om1/I1[:-1])
      gr = np.array([self.__gain__(n[i],om1[i],ompe[i],k0[i],Te[i]) for i in range(cells)])

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

    # Optionally plot
    if plots:
      self.__srs_plots__(x,n,gr,I0,I1,centred=True,xc=xc)

    return x,xc,n,I0,I1,gr

  def ray_trace_solve3(self,xrange:tuple,nrange:tuple,ntype:str, \
      I1_noise:Optional[float]=0.0,I1_seed:Optional[float]=0.0, \
      om1_seed:Optional[float]=None,points:Optional[int]=101, \
      plots:Optional[bool]=False,pump_depletion:Optional[bool]=True, \
      absorption:Optional[bool]=False):

    # Establish density profile
    x = np.linspace(xrange[0],xrange[1],points)
    xc,n = den_profile(xrange,nrange,ntype,points,centred=True)
    Te = np.ones_like(n)*self.Te
    P0 = self.I0#/self.omega0

    return self.ray_trace_solve2(x,n,Te,I1_noise,I1_seed,om1_seed,P0,plots,\
        pump_depletion,absorption,'planar')

  # Extension of BVP solver to include wave mixing from noise sources
  def wave_mixing_solve2(self,I1_noise:float,xrange:tuple, \
      nrange:tuple,ntype:str,points=101,plots=False,pump_depletion=True,\
      I1_seed:Optional[float]=0.0,om1_seed:Optional[float]=None):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('Non-SDL wave-mixing solve not implemented.')

    # Establish density profile
    x,n = den_profile(xrange,nrange,ntype,points)

    # Resonance solve for each density point
    grres, om1res, ompe, k0 = self.__resonance_range__(n)
    om1resf = PchipInterpolator(x,om1res)
    grresf = PchipInterpolator(x,grres)

    # Check om1_seed input
    if om1_seed is None:
      om1_seed = birches[-1].omega1
      om1 = copy.deepcopy(om1res)
      om1f = PchipInterpolator(x,om1)
      I1_seed = 0.0
      seed = False
      omega1s = om1res
    elif om1_seed <= 0.0:
      raise Exception('Seed Raman frequency must be positive')
    else:
      om1 = np.ones_like(x)*om1_seed
      om1f = PchipInterpolator(x,om1)
      seed = True
      omega1s = np.r_[om1res,np.array([om1_seed])]

    # Initialise intensity arrays
    om0 = self.omega0
    I0 = np.ones_like(x)*self.I0/om0
    I1s = np.ones_like(x)*(I1_seed)/om1_seed
    I1_noise /= points # Scale total noise by number of modes tracked
    I1n = np.ones((points,points))*I1_noise
    I1nbc = np.zeros((points,1))
    for i in range(points):
      I1n[i,:] /= om1res[i]
      I1nbc[i,:] = I1n[i,-1]
    I0bc = I0[0]; I1sbc = I1s[-1]
    if seed:
      I = np.vstack((I0,I1n,I1s))
      Ibc = np.vstack((I0bc,I1nbc,I1sbc))
      n1 = points + 1
    else:
      I = np.vstack((I0,I1n))
      Ibc = np.vstack((I0bc,I1nbc))
      n1 = points

    # Calculate gain matrix and functions
    grf = []
    gr = np.zeros((n1,points))
    for i in range(n1):
      print(i)
      for j in range(points):
        gr[i,j] = self.__gain__(n[j],omega1s[i],ompe[j],k0[j])
      grf.append(PchipInterpolator(x,gr[i,:]))

    for i in range(n1):
      plt.plot(x,gr[i])
    plt.show()
    if pump_depletion:
      # ODE evolution functions
      def Fsrs(xi,Ii):
        # Establish forest and set quantitis
        I0i = Ii[0,:]
        I1i = Ii[1:,:]
        f1p = np.zeros((n1,len(xi)))
        for i in range(n1):
          f1p[i,:] = grf[i](xi)*I1i[i,:]
        f1ps = np.sum(f1p,axis=0)
        f1 = np.array([-I0i*f1ps])
        f2 = np.array([-I0i*f1p[i,:] for i in range(n1)])
        return np.vstack((f1,f2))
      def bc(ya,yb):
        I0b = np.array([ya[0]-Ibc[0]])
        I1b = np.array([yb[i] - Ibc[i] for i in range(1,n1+1)])
        return np.r_[I0b,I1b][:,0]

      res = solve_bvp(Fsrs,bc,x,I,verbose=2)#,tol=1e-10)
      I0 = res.sol(x)[0]*self.omega0
      I1 = np.zeros(points)
      om1 = np.zeros(points)
      gr = np.zeros(points)
      for i in range(points):
        I1[i] = np.sum(res.sol(x)[1:,i]*omega1s)
      for i in range(points):
        om1[i] = np.sum(res.sol(x)[1:,i]*omega1s**2)/I1[i]
        gr[i] = self.__gain__(n[i],om1[i],ompe[i],k0[i])

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
      self.__srs_plots__(x,n,gr,I0,I1)

    return x,n,I0,I1,gr

  # Extension of BVP solver to include wave mixing from noise sources
  def wave_mixing_solve(self,I1_noise:float,xrange:tuple, \
      nrange:tuple,ntype:str,points=101,plots=False,pump_depletion=True,\
      I1_seed:Optional[float]=0.0,om1_seed:Optional[float]=None):

    # Check SDL flag true
    if not self.sdl:
      raise Exception('Non-SDL wave-mixing solve not implemented.')

    # Establish density profile
    x,n = den_profile(xrange,nrange,ntype,points)

    # Resonance solve for each density point
    grres, om1res, ompe, k0 = self.__resonance_range__(n)
    grresf = PchipInterpolator(x,grres)
    om1resf = PchipInterpolator(x,om1res)

    # Check om1_seed input
    if om1_seed is None:
      om1_seed = om1res[-1]
      om1 = copy.deepcopy(om1res)
      om1f = PchipInterpolator(x,om1)
      I1_seed = 0.0
      gr = np.zeros_like(x)
      grf = PchipInterpolator(x,gr)
    elif om1_seed <= 0.0:
      raise Exception('Seed Raman frequency must be positive')
    else:
      gr = np.array([self.__gain__(n[i],om1_seed,ompe[i],k0[i]) for i in range(len(x))])
      grf = PchipInterpolator(x,gr)
      om1 = np.ones_like(x)*om1_seed
      om1f = PchipInterpolator(x,om1)

    # Initialise intensity arrays
    om0 = self.omega0
    I0 = np.ones_like(x)*self.I0
    I1 = np.ones_like(x)*(I1_seed)
    I0bc = I0[0]; I1bc = I1[-1]

    if pump_depletion:
      # ODE evolution functions
      def Fsrs(xi,Ii):
        I0i, I1i = Ii
        # Establish forest and set quantitis
        om1m = om1f(xi)
        om1res = om1resf(xi)
        gr0 = grresf(xi)
        gri = grf(xi)
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
        om1mf = PchipInterpolator(x,om1)
        gr = np.array([self.__gain__(n[i],om1[i],ompe[i],k0[i]) for i in range(len(x))])
        grf = PchipInterpolator(x,gr)
        conv = np.abs(I0[-1]-I0old)+np.abs(I1[0]-I1old)
        print(f'Convergence: {conv:0.2e}')
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
      self.__srs_plots__(x,n,gr,I0,I1)

    return x,n,I0,I1,gr

  # SRS gain function for any density and Raman frequency
  def __gain__(self,ne:float,om1:float,ompe:Optional[float]=None,\
      k0:Optional[float]=None,Te:Optional[float]=None):
    birch = self.__raman_mode__(ne,om1,Te)
    if k0 is None:
      birch.get_k0()
    else:
      birch.k0 = k0
    if ompe is not None:
      birch.ompe = ompe
    k1 = -birch.emw_dispersion(om1,target='k')
    birch.set_wavenumbers(k1,birch.k0-k1)
    birch.get_gain_coeff()
    return birch.gain_coeff

  # Resonance solve across a density range
  def __resonance_range__(self,n:np.ndarray,absorption:Optional[bool]=False,\
      Te:Optional[np.ndarray]=None):
    birches = []
    for i in range(len(n)):
      birches.append(copy.deepcopy(self))
      if Te is None:
        birches[i].set_electrons(electrons=True,Te=self.Te,ne=n[i])
      else:
        birches[i].set_electrons(electrons=True,Te=Te[i],ne=n[i])
      birches[i].resonance_solve()
      birches[i].get_gain_coeff()
      if absorption:
        birches[i].get_kappa0()
    om1res = np.array([i.omega1 for i in birches])
    grres = np.array([i.gain_coeff for i in birches])
    ompe = np.array([i.ompe for i in birches])
    k0 = np.array([i.k0 for i in birches])
    if absorption:
      kappa0 = np.array([i.kappa0 for i in birches])
      return grres, om1res, ompe, k0, kappa0
    else:
      return grres, om1res, ompe, k0

  # Creates copy of parent forest with new raman frequency and reference density
  def __raman_mode__(self,ne,om1,Te:Optional[float]=None):
    forest = copy.deepcopy(self)
    if Te is None:
      forest.set_electrons(electrons=True,ne=ne,Te=self.Te)
    else:
      forest.set_electrons(electrons=True,ne=ne,Te=Te)
    forest.set_frequencies(om1,self.omega0-om1)
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
    axs[0,1].set_ylabel('Wave Gain [m/Ws^2]')
    axs[1,0].semilogy(x*1e6,I0)
    axs[1,0].set_ylabel('I0 [W/m^2]')
    axs[1,0].set_xlabel('x [um]')
    axs[1,1].semilogy(x*1e6,I1)
    axs[1,1].set_ylabel('I1 [W/m^2]')
    axs[1,1].set_xlabel('x [um]')
    fig.suptitle(f'Mode: {self.mode}; SDL: {self.sdl}; Relativistic: {self.relativistic}; '\
        +f'\nTe: {self.Te:0.2e} K; I00: {self.I0:0.2e} W/m^2; lambda0: {self.lambda0:0.2e} m')
    plt.tight_layout()
    plt.show()
