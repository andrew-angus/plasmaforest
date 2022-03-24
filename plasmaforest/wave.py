#!/bin/python3

from .core import *
import astropy.constants as ac
from scipy.optimize import newton, minimize
from scipy.integrate import solve_ivp

# Core class, mainly a wrapper of select plasmapy functionality
# Currently restricted to single-ion species
# Specifying ion parameters optional
# Take inputs in SI units
@typechecked
class wave_forest(forest):
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.zeta0 = None # Known natural zeta solution for kinetic dispersion
    self.K0 = None # Known natural K solution for kinetic dispersion
  
  # Solve the EMW dispersion relation in a plasma
  def emw_dispersion(self,arg:floats,target:str) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if target == 'omega':
      return np.sqrt(sqr(self.ompe) + sqr(sc.c*arg))
    elif target == 'k':
      return np.sqrt((sqr(arg) - sqr(self.ompe))\
          /sqr(sc.c))
    else:
      raise Exception("target must be one of \'omega\' or \'k\'.")

  # Return residual of emw dispersion relation in dimensionless units for accuracy
  def emw_dispersion_res(self,omega:floats,k:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    kvac = omega/sc.c
    k0 = k/kvac
    ompe0 = self.ompe/omega
    return -1.0+sqr(k0)+sqr(ompe0)

  # Fluid EPW dispersion relation
  def bohm_gross(self,arg:floats,target:str) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    if target == 'omega':
      return np.sqrt(sqr(self.ompe) + prefac*sqr(self.vthe*arg))
    elif target == 'k':
      return np.sqrt((sqr(arg) - sqr(self.ompe))\
          /(prefac*sqr(self.vthe)))
    else:
      raise Exception("target must be one of \'omega\' or \'k\'.")

  # Residual of fluid EPW dispersion relation, dimensionless for accuracy
  def bohm_gross_res(self,omega:floats,k:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = np.sqrt(0.5*gamma)
    #return -sqr(omega/self.ompe)+sqr(prefac*self.vthe*k/self.ompe)+1.0
    return 1-sqr(prefac*self.vthe*k/omega)-sqr(self.ompe/omega)

  # Calculate zeta for both plasma dispersion function and its derivative
  def __zeta__(self,omega:flomplex,k:flomplex,species:str) -> flomplex:
    if species == 'e':
      if self.vthe is None:
        self.get_vth(species='e')
      a = self.vthe
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      a = self.vthi
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")
    return omega/(k*a)

  # Plasma susceptibility calculated with the plasma dispersion function
  def susceptibility(self,omega:flomplex,k:flomplex,species:str) -> flomplex: 
    if species == 'e':
      if self.vthe is None:
        self.get_vth(species='e')
      if self.ompe is None:
        self.get_omp(species='e')
      a = self.vthe
      omp = self.ompe
    elif species == 'i':
      if self.vthi is None:
        self.get_vth(species='i')
      if self.ompi is None:
        self.get_omp(species='i')
      omp = self.ompi
      a = self.vthi
    else:
      raise Exception("species must be one of \'e\' or \'i\'.")
     
    zeta = self.__zeta__(omega,k,species)
    return -sqr(omp/(k*a))*dZfun(zeta)

  # Linear kinetic permittivity equation
  def kinetic_permittivity(self,omega:flomplex,k:flomplex,full:Optional[bool]=True) -> flomplex:
    dis = 1 + self.susceptibility(omega=omega,k=k,species='e')
    if full:
      dis += np.sum(self.susceptibility(omega=omega,k=k,species='i'))
    return dis

  # Phase velocity of a wave
  def phase_velocity(self,omega:floats,k:floats) -> floats:
    return omega/k

  # EMW group velocity
  def emw_group_velocity(self,omega:floats,k:floats) -> floats:
    return sqr(sc.c)*k/omega

  # EPW bohm-gross group velocity
  def bohm_gross_group_velocity(self,omega:floats,k:floats) -> floats:
    if self.vthe is None:
      self.get_vth(species='e')
    gamma = (2+self.ndim)/self.ndim
    prefac = 0.5*gamma
    return prefac*sqr(self.vthe)*k/omega

  # General EMW critical density
  def emw_nc(self,omega:floats) -> floats:
    return sc.epsilon_0*sc.m_e*sqr(omega/sc.e)

  # General EMW collisional damping rate
  # Use Kruer Ch 5
  def emw_damping(self,omega:floats) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if self.vthe is None:
      self.get_vth(species='e')
    if self.dbyl is None:
      self.get_dbyl()
    if self.coulomb_log_ei is None:
      self.get_coulomb_log(species='ei')
    vth1Drms = self.vthe/np.sqrt(2)
    impact = np.log(np.exp(self.coulomb_log_ei)/self.dbye*(vth1Drms/omega))
    return sqr(self.ompe/omega)/(3*pwr(2*np.pi,3/2))*self.Z \
        /(self.ne+np.sum(self.ni))*pwr(self.ompe/vth1Drms,3)*self.ompe*impact/2
  
  # Refractive index function
  def emw_ri(self,nc:floats) -> floats:
    self.electron_check()
    return np.sqrt(1-self.ne/nc)

  # EMW E field in plasma
  def emw_E(self,I:floats,ri:floats) -> floats:
    return np.sqrt(2*I/(sc.epsilon_0*sc.c*ri))

  # EMW B field in plasma
  def emw_B(self,I:floats,vp:floats) -> floats:
    return np.sqrt(2*I*sc.mu_0/vp)

  # EMW electron quiver velocity
  def emw_vos(self,E:floats,omega:floats) -> floats:
    return sc.e*E/(sc.m_e*omega)

  # EPW collisional damping
  # Calculated according to Rand - Collision Damping of Electron Plasma Waves (1965)
  def epw_coll_damping(self,omega:floats) -> floats:
    # Check for attributes
    if self.coulomb_log_ei is None:
      self.get_coulomb_log(species='ei')
    if self.vthe is None:
      self.get_vth(species='e')
    if self.ompe is None:
      self.get_omp(species='e')

    # Calculate in cgs units
    ecgs = ac.e.gauss.value
    hbarcgs = ac.hbar.cgs.value
    necgs = (self.ne/u.m**3).cgs.value
    nicgs = (self.ni/u.m**3).cgs.value
    mecgs = (sc.m_e*u.kg).cgs.value
    vthecgs = (self.vthe/np.sqrt(2)*u.m/u.s).cgs.value

    # Calculate switch quantities
    omrat = omega/self.ompe
    G = 0.27*omrat-0.091
    regime = sqr(ecgs)/(hbarcgs*vthecgs)
    if regime < 1:
      logfac = np.log(2*mecgs*sqr(vthecgs)/(hbarcgs*self.ompe))-0.442-G
    else:
      logfac = np.log(2*mecgs*pwr(vthecgs,3)/(sqr(ecgs)*self.ompe))-1.077-G
    
    A = 16*np.pi/3*np.sqrt(2*np.pi)*pwr(ecgs,6)*sqr(self.Z)*necgs*nicgs\
        /pwr(mecgs*vthecgs*omega,3)*logfac
    
    return 0.5*A*self.ompe

  def epw_landau_damping(self,omega:floats,k:floats,\
      mode:str,relativistic:Optional[bool]=False) -> floats:
    if self.ompe is None:
      self.get_omp(species='e')
    if mode == 'fluid':
      if self.vthe is None:
        self.get_vth(species='e')
      if self.dbye is None:
        self.get_dbyl()

      # Relativistic calc of Landau damping according to:
      # Bers - Relativistic Landau damping of electron plasma waves 
      # in stimulated Raman scattering (2009)
      if relativistic:
        vth = self.vthe/np.sqrt(2)
        mu = sqr(sc.c/vth)
        N = sc.c*k/omega
        kdb = k*self.dbye
        z0 = mu*np.abs(N)/np.sqrt(sqr(N)-1)
        gamma = np.sqrt(np.pi/8)*omega*pwr(mu,3/2)*np.exp(mu*(1-N/np.sqrt(sqr(N)-1)))\
            /(np.abs(N)*(sqr(N)-1))*(1+2/z0+2/sqr(z0))/(1+6*sqr(kdb)-5/(2*mu))

      # First order approximation of non-relativistic landau damping
      # Calculated according to Swanson - Plasma Waves (2012)
      else:
        """
        vthe1d = self.vthe/np.sqrt(2)
        dk = k*vthe1d/omega
        omega2 = omega*(1+3/2*sqr(dk))
        print(self.ompe,omega,omega2)
        gamma = np.sqrt(np.pi/8)*omega2/pwr(dk,3)\
            *np.exp(-0.5*sqr(omega2/(k*vthe1d))) # lpse
        print(f'{gamma:0.3e}')
        """
        #gamma = np.sqrt(np.pi)*sqr(self.ompe*omega)/pwr(k*self.vthe,3)\
            #*np.exp(-sqr(omega/(k*self.vthe)))
        K = k*self.dbye
        gamma = omega*np.exp(-0.5*(3+1/sqr(K)))*np.sqrt(np.pi/8)*(1-4.5*sqr(K))/np.abs(pwr(K,3))
    elif mode == 'kinetic':
      eps = self.kinetic_permittivity(omega,k,full=False)
      depsdom = 2*omega/sqr(self.ompe) # Fluid approximation
      gamma = np.imag(eps)/depsdom

    return gamma

  # Solve kinetic dispersion to find natural omega/k from respective part
  def epw_kinetic_dispersion(self,arg:floats,target:str):
    if self.dbye is None:
      self.get_dbyl()
    if self.vthe is None:
      self.get_vth(species='e')

    if target == 'omega':
      # Integrate along solution branch to desired k
      K = arg*self.dbye
      zeta = self.zeta_int(K)
      omega = zeta*self.vthe*arg
      return omega

    elif target == 'k':
      # Initial guess at k from fluid dispersion  
      kguess = self.bohm_gross(arg,target='k')
      Kguess = kguess*self.dbye
      # Iterate ode solver till correct real omega found
      # Objective function
      def omegar_diff(K):
        zeta = self.zeta_int(K)
        omegar = np.real(zeta*self.vthe*K/self.dbye)
        return omegar - arg
      # Solve for K
      res = newton(omegar_diff,Kguess,tol=np.finfo(np.float64).eps)
      k = res/self.dbye
      return k

    else:
      raise Exception("target must be one of \'omega\' or \'k\'.")

  # Get known starting point on natural EPW mode branch
  def __natural_epw_zeta__(self):
    # Refine tabulated value with root-finding
    zeta0 = 2.3 - 0.1j # Initial guess
    self.zeta0 = self.__zero_dZim__(zeta0)

    # Get K0 value corresponding to this zeta
    dZfun0 = dZfun(self.zeta0)
    self.K0 = np.real(np.sqrt(dZfun0/2))

  # Look for zero imag dZ from close initial guess
  def __zero_dZim__(self,zeta0:complex) -> complex:
    # Objective function     def imdZ(zeta):
    def imdZ(zeta):
      zeta = zeta[0] + 1j*zeta[1]
      return np.imag(dZfun(zeta))

    # Root find and return zeta
    zeta00 = np.array([np.real(zeta0),np.imag(zeta0)])
    res = newton(imdZ,zeta00,tol=np.finfo(np.float64).eps)
    zeta = res[0] + 1j*res[1]
    return zeta

  # Integrate zeta ODE till desired K value
  def zeta_int(self,Kf:float,refine:Optional[bool]=True) -> complex:
    # Get initial conditions if not already obtained
    if self.zeta0 is None or self.K0 is None:
      self.__natural_epw_zeta__()

    # ODE driver
    def zeta_ode(K,zeta):
      return np.array([4*K/ddZfun(zeta)])

    # Integrator
    zetain = np.array([np.real(self.zeta0),np.imag(self.zeta0)])
    #res = odeint(zeta_ode,zetain,Ksolve)
    res = solve_ivp(zeta_ode,(self.K0,Kf),np.array([self.zeta0]))
    zeta = res.y[-1,-1]
    
    # Optionally refine result
    if refine:
      zeta = self.__zero_dZim__(zeta)
      K = np.real(np.sqrt(dZfun(zeta)/2))
      res = solve_ivp(zeta_ode,(K,Kf),np.array([zeta]))
      zeta = res.y[-1,-1]

    return zeta

  # Get undamped mode by searching for zero of real permittivity for real argument
  def undamped_dispersion(self,k:float):
    # Objective functions
    def realeps(omega):
      return np.abs(np.real(self.kinetic_permittivity(omega[0],k,full=False)))

    # Get zero of real permittivity by minimisation
    omega0 = self.bohm_gross(k,target='omega')
    res = minimize(realeps,np.array([omega0]),tol=np.finfo(np.float64).eps)
    omega = res.x[0]

    return omega


# Plasma dispersion function
@typechecked
def Zfun(zeta:flomplex) -> flomplex:
  Z = pp.dispersion.plasma_dispersion_func(zeta)
  return Z

# Derivative of plasma dispersion function
@typechecked
def dZfun(zeta:flomplex) -> flomplex:
  dZ = -2*(1+zeta*Zfun(zeta))
  return dZ

# 2nd derivative of plasma dispersion function
@typechecked
def ddZfun(zeta:flomplex) -> flomplex:
  ddZ = -2*(Zfun(zeta)+zeta*dZfun(zeta))
  return ddZ
