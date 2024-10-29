#!/usr/bin/python3
#--------------------------------------------------------------------------------------------------
# This module calls individual solver routines based on the type of finite element analysis
# requested by the user in the input file. Currently, this module can handle single-phase elasticity
# or elastodynamics, mutliphase poroelasticity or poroelastodynamics for biphasic mixtures, or
# multiphase thermoporoelasticity or thermoporoelastodyanmics for biphasic mixtures and,
# if the user supplies a first order ODE in 'solver_user.py', general-purpose first order ODEs.
# This module also converts input parameters into an object that can be called by subroutines.
# Details on that process is provided in the comments below.
#
# Author:       Zachariah Irwin
# Affiliation:  University of Colorado Boulder
# Last Edits:   October 16, 2024
#--------------------------------------------------------------------------------------------------
import sys, os, shutil, glob, argparse

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import simInput
except ImportError:
  sys.exit("MODULE WARNING. simInput.py not found! Check configuration.")

try:
  import solver_driver
  import solver_user
  import solver_u
  import solver_upf
  import solver_ut
  import solver_uufpf
  import solver_upftstf
  import solver_uufpftstf
except ImportError:
  sys.exit("MODULE WARNING. solvers not found! Check configuration.")

#-------------------------------------------
# Utility class to store problem parameters.
#-------------------------------------------
class Parameters:

  def __init__(self, inputData):

    #--------------------------------------------------------------------------------------------------
    # Integration parameters.
    #
    # Notes:
    # ------
    # - Physics             describes the physical formulation of the problem
    # - integrationScheme   is the integration scheme chosen for the time discretization
    # - Lumping             controls whether or not mass matrices will be lumped for explicit 
    #                       integration, and if so, which ones get lumped
    # - beta                is the Newmark-beta parameter 'beta'
    # - gamma               is the Newmark-beta parameter 'gamma'
    # - alpha               is the trapezoidal parameter 'alpha' (also known as 'theta')
    # - TStart              is the time at which the simulation starts (typically 0)         [s]
    # - TStop               is the time at which the simulation ends                         [s]
    # - dt0                 is the initial time step                                         [s]
    # - tolr                is the Newton-Raphson relative tolerance
    # - tola                is the Newton-Raphson absolute tolerance
    # - kmax                is the Newton-Raphson maximum number of iterations
    # - isAdaptiveStepping  controls whether or not to use adaptive time stepping (refer to
    #                       Laadhari et al. (2017) for the implicit application or Cash and Karp
    #                       (1990) for the explicit application)
    # - adaptiveKMax        is the maximum number of Newton-Raphson iterations before decreasing time
    #                       step
    # - adaptiveIncrease    is the factor by which to increase the time step
    # - adaptiveDecrease    is the factor by which to decrease the time step
    # - adaptiveDTMax       is the maximum number the time step will be increased to  [s]
    # - norm_ord            is the type of norm used in to evaluate error the in adaptive Runge-Kutta
    #                       time stepping schemes
    # - SF                  is the scale factor by which to adjust time step in the adaptive
    #                       Runge-Kutta time stepping schemes
    #--------------------------------------------------------------------------------------------------
    self.Physics           = inputData.m_Physics
    self.integrationScheme = inputData.m_IntegrationScheme
    #------------------------------------------------------------------------
    # Set 'mass' matrix lumping scheme parameters.
    # 
    # Note: temperature mass matrix lumping is incomplete as of latest edits.
    #------------------------------------------------------------------------
    self.Lumping          = inputData.m_Lumping
    self.SolidTempLumping = False
    self.FluidTempLumping = False
    if self.Lumping == 'Solid':
      self.Lumping         = True
      self.SolidLumping    = True
      self.FluidLumping    = False
      self.PressureLumping = False
    elif self.Lumping == 'Solid-Temp':
      self.Lumping          = True
      self.SolidLumping     = True
      self.SolidTempLumping = True
      self.FluidLumping     = False
      self.PressureLumping  = False
    elif self.Lumping == 'Solid-Pressure':
      self.Lumping         = True
      self.SolidLumping    = True
      self.FluidLumping    = False
      self.PressureLumping = True
    elif self.Lumping == 'Solid-Fluid':
      self.Lumping         = True
      self.SolidLumping    = True
      self.FluidLumping    = True
      self.PressureLumping = False
    elif self.Lumping == 'Fluid':
      self.Lumping         = True
      self.SolidLumping    = False
      self.FluidLumping    = True
      self.PressureLumping = False
    elif self.Lumping == 'Pressure':
      self.Lumping         = True
      self.SolidLumping    = False
      self.FluidLumping    = False
      self.PressureLumping = True
    elif self.Lumping == 'Fluid-Pressure' or self.Lumping == 'Pressure-Fluid':
      self.Lumping         = True
      self.SolidLumping    = False
      self.FluidLumping    = True
      self.PressureLumping = True
    elif self.Lumping == 'All':
      self.Lumping          = True
      self.SolidLumping     = True
      self.FluidLumping     = True
      self.PressureLumping  = True
      self.SolidLumping     = True
    else:
      self.Lumping         = False
      self.SolidLumping    = False
      self.FluidLumping    = False
      self.PressureLumping = False
    #-------------------------------------
    # Set implicit integration parameters.
    #-------------------------------------
    self.beta  = inputData.m_beta
    self.gamma = inputData.m_gamma
    self.alpha = inputData.m_alpha
    #----------------------------------
    # Set simulation timing parameters.
    #----------------------------------
    self.TStart  = inputData.m_TStart
    self.TStop   = inputData.m_TStop
    self.dt0     = inputData.m_DT
    self.TOutput = inputData.m_TOutput
    self.NSteps  = int(np.ceil((self.TStop - self.TStart)/self.dt0))
    self.n_save  = int(self.NSteps/self.TOutput)

    self.t  = self.TStart
    self.dt = self.dt0
    #----------------------
    # Set error parameters.
    #----------------------
    self.tolr      = inputData.m_tolr
    self.tola      = inputData.m_tola
    self.kmax      = inputData.m_kmax
    self.norm_ord  = inputData.m_norm_ord
    self.SF        = inputData.m_SF
    #-----------------------------------
    # Set adaptive time step parameters.
    #-----------------------------------
    self.isAdaptiveStepping = inputData.m_isAdaptiveStepping
    self.adaptiveKMax       = inputData.m_adaptiveKMax
    self.adaptiveIncrease   = inputData.m_adaptiveIncrease
    self.adaptiveDecrease   = inputData.m_adaptiveDecrease
    self.adaptiveDTMax      = inputData.m_adaptiveDTMax
    self.adaptiveDTMin      = inputData.m_adaptiveDTMin
    self.adaptiveStart      = inputData.m_adaptiveStart
    self.adaptiveStop       = inputData.m_adaptiveStop
    self.adaptiveSave       = inputData.m_adaptiveSave
    self.dt_save            = self.dt0*self.n_save
    if self.dt_save        == 0.0:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nToo many data points requested for initial time step.")
    if self.adaptiveDTMax  >= self.dt_save:
      self.adaptiveDTMax    = self.dt_save 
    self.t_save             = self.TStart + self.dt_save
    #--------------------------------
    # Set Butcher tableau parameters.
    #--------------------------------
    #-----------------------------
    # Refer to Cash & Karp (1990).
    #-----------------------------
    if 'VRKF' or 'RKFNC' or 'RKCKP' in self.integrationScheme:
      self.numRKStages = 6

      self.Ci    = np.zeros((6), dtype=np.float64)
      self.Ci[0] = (0)
      self.Ci[1] = (1/5)
      self.Ci[2] = (3/10)
      self.Ci[3] = (3/5)
      self.Ci[4] = (1)
      self.Ci[5] = (7/8)

      self.Aij      = np.zeros((6,5), dtype=np.float64)
      self.Aij[1,0] = (1/5)
      self.Aij[2,0] = (3/40)
      self.Aij[2,1] = (9/40)
      self.Aij[3,0] = (3/10)
      self.Aij[3,1] = (-9/10)
      self.Aij[3,2] = (6/5)
      self.Aij[4,0] = (-11/54)
      self.Aij[4,1] = (5/2)
      self.Aij[4,2] = (-70/27)
      self.Aij[4,3] = (35/27)
      self.Aij[5,0] = (1631/55296)
      self.Aij[5,1] = (175/512)
      self.Aij[5,2] = (575/13824)
      self.Aij[5,3] = (44275/110592)
      self.Aij[5,4] = (253/4096)

      self.Bij      = np.zeros((5,6), dtype=np.float64)
      self.Bij[0,0] = (1)
      self.Bij[1,0] = (-3/2)
      self.Bij[1,1] = (5/2)
      self.Bij[2,0] = (19/54)
      self.Bij[2,2] = (-10/27)
      self.Bij[2,3] = (55/54)
      self.Bij[3,0] = (2825/27648)
      self.Bij[3,2] = (18575/48384)
      self.Bij[3,3] = (13525/55296)
      self.Bij[3,4] = (277/14336)
      self.Bij[3,5] = (1/4)
      self.Bij[4,0] = (37/378)
      self.Bij[4,2] = (250/621)
      self.Bij[4,3] = (125/594)
      self.Bij[4,5] = (512/1771)
    #------------------------------------
    # Refer to Bogacki & Shampine (1989).
    #------------------------------------
    elif 'RKBS' in self.integrationScheme:
      self.numRKSkeltages = 4
      
      self.Ci    = np.zeros((4), dtype=np.float64)
      self.Ci[0] = (0)
      self.Ci[1] = (1/2)
      self.Ci[2] = (3/4)
      self.Ci[3] = (1)

      self.Aij      = np.zeros((4,3), dtype=np.float64)
      self.Aij[1,0] = (1/2)
      self.Aij[2,0] = (0)
      self.Aij[2,1] = (3/4)
      self.Aij[3,0] = (2/9)
      self.Aij[3,1] = (1/3)
      self.Aij[3,2] = (4/9)

      self.Bij      = np.zeros((3,4), dtype=np.float64)
      self.Bij[2,0] = (2/9)
      self.Bij[2,1] = (1/3)
      self.Bij[2,2] = (4/9)
      self.Bij[1,0] = (7/24)
      self.Bij[1,1] = (1/4)
      self.Bij[1,2] = (1/3)
      self.Bij[1,3] = (1/8)
    #--------------------------------------------------------------------
    # Only need the time integration parameters for user-defined Physics.
    # See solver_user.py for details on customizable implementation.
    #--------------------------------------------------------------------
    if self.Physics == 'user':
      return
    #-----------------------
    # Set gravity parameter.
    #-----------------------
    self.Gravity  = inputData.m_Gravity
    #--------------------------------------------------------------------------------------------------
    # Material parameters for pore fluid (irrelevant for solid mechanics).
    #
    # Notes:
    # ------
    # - KF              is the bulk modulus of the pore fluid               [Pa]
    # - fluidShearVisc  is the shear viscosity of the pore fluid            [Pa-s]
    # - fluidBulkVisc   is the bulk viscosity of the pore fluid             [Pa-s]
    # - rhofR_0         is the initial real mass density of the pore fluid  [m^3/kg]
    # - nf_0            is the initial volume fraction of the pore fluid
    # - cvf             is the specific heat capacity of the pore fluid     [J/kg-K]
    # - Af              is the volumetric CTE of the pore fluid             [1/K]
    # - kf              is the thermal conductivity of the pore fluid       [W/m-K]
    # - Tf_0            is the initial temperature of the pore fluid        [K]
    #--------------------------------------------------------------------------------------------------
    self.KF             = inputData.m_KF
    self.fluidShearVisc = inputData.m_fluidShearVisc
    self.fluidBulkVisc  = inputData.m_fluidBulkVisc
    self.DarcyBrinkman  = inputData.m_DarcyBrinkman
    self.rhofR_0        = inputData.m_rhofR_0
    self.nf_0           = inputData.m_nf_0
    self.cvf            = inputData.m_cvf
    self.Af             = inputData.m_Af
    self.kf             = inputData.m_kf
    self.p_f0           = inputData.m_p_f0
    self.Tf_0           = inputData.m_Tf_0
    if 'pf' in self.Physics:
      self.cf           = (1.4*self.p_f0/(self.nf_0*self.rhofR_0))**(0.5)
    else:
      self.cf           = 0
    #--------------------------------------------------------------------------------------------------
    # Material parameters for solid.
    #
    # Notes:
    # ------
    # - KSkel      is the bulk modulus of the solid skeleton              [Pa]
    # - emod       is the Young's modulus of the solid skeleton           [Pa]
    # - mu         is the shear modulus of the solid skeleton             [Pa]
    # - lambd      is Lame's first parameter                              [Pa]
    # - nu         is the Poisson ratio
    # - MSkel      is the P-wave modulus of the solid skeleton            [Pa]
    # - rhosR_0    is the initial real mass density of the solid          [m^3/kg]
    # - ns_0       is the initial volume fraction of the solid
    # - intrPerm   is the intrinsic permeability of the solid skeleton    [m^2]
    # - cvs        is the specific heat capacity of the solid             [J/kg-K]
    # - As         is the volumetric CTE of the solid                     [1/K]
    # - ks         is the thermal conductivity of the solid               [W/m-K]
    # - Ts_0       is the initial temperature of the solid                [K]
    #--------------------------------------------------------------------------------------------------
    self.KS         = inputData.m_KS
    self.KSkel      = inputData.m_KSkel
    self.emod       = inputData.m_emod
    self.mu         = inputData.m_mu
    self.lambd      = inputData.m_lambda
    self.nu         = inputData.m_nu
    try:
      self.MSkel  = self.KSkel + (4/3)*self.mu
    except TypeError:
      self.MSkel  = None
    self.rhosR_0    = inputData.m_rhosR_0
    self.ns_0       = 1 - self.nf_0
    self.intrPerm   = inputData.m_intrPerm
    self.cvs        = inputData.m_cvs
    self.As         = inputData.m_As
    self.ks         = inputData.m_ks
    self.Ts_0       = inputData.m_Ts_0
    #--------------------------------------------------------------------------------------------------
    # Geoemetry parameters.
    #
    # Notes:
    # ------
    # - H0         is the height of the column                                        [m]
    # - R          is the radius/width of the column                                  [m]
    # - Area       is the cross-sectional area of the column                          [m^2]
    #--------------------------------------------------------------------------------------------------
    self.H0 = inputData.m_H0
    self.R  = inputData.m_R

    if inputData.m_Geo == 'Cylinder' or inputData.m_Geo == 'cylinder':
      self.Area = np.pi*self.R**2
    elif inputData.m_Geo == 'Rectangle' or inputData.m_Geo == 'rectangle':
      self.Area = self.R**2
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nGeometry not recognized.")
    #--------------------------------------------------------------------------------------------------
    # Constitutive modeling.
    # 
    # Notes:
    # ------
    # - isDynamics   controls whether or not inertia terms will be ignored (static) or included 
    #                (dynamic)
    # - fluidModel   controls the constitutive equation for rho_fR0
    # - solidModel   controls the constitutive equation for P_11
    # - nu_0         is the amplitude of the viscous damping parameter (for viscoelastic contributions)
    #                [s] (to be used in conjunction with the neo-Hookean material model only)
    # - C0           is the quadratic bulk viscosity coefficient (shock, refer to Wilkins (1980))
    # - C1           is the linear bulk viscosity coefficient (refer to Wilkins (1980))
    # - alpha_stab   is the stability parameter for pore fluid pressure (refer to Regueiro (2014))
    # - SUPG         is a flag enabling SUPG stabilization for pore fluid temperature (refer to
    #                Irwin (2024) [thesis] or Koch (2016))
    # - c1           is the bulk constant for lung tissue in air (refer to Clayton and Freed (2019))
    # - c2           is the squeeze constant for lung tissue in air (refer to Clayton and Freed (2019))
    # - B1           is viscoelastic dynamic multiplier (refer to Clayton and Freed (2019))
    # - mu_prime     is the derivative of shear modulus at reference pressure (refer to C&F (2019))
    # - Z0           is the normalized injury threshold (refer to Clayton and Freed (2019))
    # - alpha_D      is the rate-independent damage kinetics (refer to Clayton and Freed (2019))
    # - omega_D      is the gradient/rate enhanced damage kinetics (refer to Clayton and Freed (2019))
    # - poros_func0  is the F(nf_0) relation at the initial condition
    # - khat_mult    is the multiplier for the choice of hyrdraulic conductivity      [m^2/Pa-s]
    # - khat         is the initial condition for hydraulic conductivity              [m^2/Pa-s]
    # - kappa        is the strain multiplier (refer to Lai et al. (1981), Markert (2005)
    # - k_exchange   is the convective heat transfer coefficient                      [W/m^3-K]
    #--------------------------------------------------------------------------------------------------
    self.isDynamics   = inputData.m_isDynamics
    self.fluidModel   = inputData.m_fluidModel
    if self.integrationScheme == 'Central-difference' and 'pf' in self.Physics:
      if self.fluidModel != 'Exponential':
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nOnly the 'Exponential' fluid model has been implemented for central-difference time integration.")
    self.solidModel   = inputData.m_solidModel
    self.nu_0         = inputData.m_nu_0
    self.C0           = inputData.m_C0
    self.C1           = inputData.m_C1
    self.alpha_stab   = inputData.m_alpha_stab
    self.SUPG         = inputData.m_SUPG
    self.h0           = inputData.m_h0
    self.h1           = inputData.m_h1
    if self.SUPG and self.kf == 0:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSUPG stabilization for pore fluid energy balance requires non-zero thermal conductivity of the pore fluid.")
    #-----------------------------------------------------------------------
    # Re-define the single-phase parameters from the solid-fluid parameters.
    # If fluidModel is not 'Bulk', then use the input file parameters (e.g.,
    # for single-phase geomechanics simulations).
    #-----------------------------------------------------------------------
    if 'pf' not in self.Physics and self.fluidModel == 'Bulk':
      self.Bb0T    = 1/(self.ns_0/self.KS      + self.nf_0/self.p_f0)
      self.KSkel   = 1/(self.ns_0/self.KS      + self.nf_0/(1.4*self.p_f0))
      self.cvs     =    self.ns_0*self.cvs     + self.nf_0*self.cvf
      self.As      =    self.ns_0*self.As      + self.nf_0*self.Af
      self.ks      =    self.ns_0*self.ks      + self.nf_0*self.kf
      self.rhosR_0 =    self.ns_0*self.rhosR_0 + self.nf_0*self.rhofR_0
      self.lambd   = self.KSkel - (2/3)*self.mu
      self.MSkel   = self.KSkel + (4/3)*self.mu
      self.p_f0    = 0
    #--------------------------------------------------------
    # Set Clayton & Freed lung constitutive model parameters.
    #--------------------------------------------------------
    if 'Clayton-Freed' in self.solidModel:
      self.c1       = inputData.m_c1
      self.c2       = inputData.m_c2
      self.B1       = inputData.m_B1
      self.mu_prime = inputData.m_mu_prime
      self.Z0       = inputData.m_Z0
      self.alpha_D  = inputData.m_alpha_D
      self.omega_D  = inputData.m_omega_D
      #----------------------------------------
      # Assign dynamic viscoelastic parameters.
      #----------------------------------------
      if self.B1 > 0:
        if 'pf' not in self.Physics:
          self.B2 = self.B1*self.KS/(6*self.mu)
        else:
          self.B2 = self.B1*self.KSkel/(6*self.mu)
      else:
        self.B2 = 0
    #---------------------------------------
    # Set hydraulic conductivity parameters.
    #---------------------------------------
    if 'pf' in self.Physics:
      self.khatType     = inputData.m_khatType

      if self.khatType == 'Kozeny-Carman':
        self.poros_func0  = self.nf_0**3/(1 - self.nf_0**2)
        self.khat_mult    = self.intrPerm/(self.fluidShearVisc*self.poros_func0)
        self.khat         = self.khat_mult*self.poros_func0
      elif self.khatType == 'Strain-Exponential' or self.khatType == 'Hyperbolic':
        self.kappa        = inputData.m_kappa
        self.khat_mult    = self.intrPerm/self.fluidShearVisc
        self.khat         = self.khat_mult
      elif self.khatType == 'Constant':
        self.khat         = self.intrPerm/self.fluidShearVisc
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nFunctional form for hydraulic conductivity not recognized.")
    #-------------------------------------------
    # Set heat convection coefficient parameter.
    #-------------------------------------------
    self.k_exchange = inputData.m_k_exchange
    #--------------------------------------------------------------------------------------------------
    # Mesh parameters.
    #
    # Notes:
    # ------
    # - Element_Type          dictates the element-wise degrees of freedom
    #
    #                         options:
    #
    #                         TYPE               PHYSICS         NO. LOCAL DOF   CONTINUITY        
    #                         ----               -------         -------------   ----------
    #                         - Q3H              (u)              4              C1
    #                         - Q3               (u)              4              C0
    #                         - Q2               (u)              3              C0
    #                         - Q1               (u)              2              C0
    #                         - Q3H-P1           (u-pf)           6              C1
    #                         - Q3-P1            (u-pf)           6              C0
    #                         - Q2-P1            (u-pf)           5              C0
    #                         - Q1-P1            (u-pf)           4              C0
    #                         - Q3H-T1           (u-ts)           6              C1
    #                         - Q2-T1            (u-ts)           5              C0
    #                         - Q1-T1            (u-ts)           4              C0
    #                         - Q3H-Q3H-P1       (u-uf-pf)       10              C1
    #                         - Q3H-Q2-P1        (u-uf-pf)        9              C1
    #                         - Q3H-Q1-P1        (u-uf-pf)        8              C1
    #                         - Q3-Q3-P1         (u-uf-pf)       10              C0
    #                         - Q2-Q2-P1         (u-uf-pf)        8              C0
    #                         - Q2-Q1-P1         (u-uf-pf)        7              C0
    #                         - Q1-Q1-P1         (u-uf-pf)        6              C0
    #                         - Q3H-P1-T1-T1     (u-pf-ts-tf)    10              C1
    #                         - Q3H-Q3H-P1-T1-T1 (u-uf-pf-ts-tf) 14              C1
    #                         - Q3H-Q1-P1-T1-T1  (u-uf-pf-ts-tf) 12              C1
    #
    # - ndofSe                is the number of local element DOF for solid displacement
    # - ndofFe                is the number of local element DOF for fluid displacement
    # - ndofPe                is the number of local element DOF for fluid pressure
    # - ndofTse               is the number of local element DOF for solid temperature
    # - ndofTfe               is the number of local element DOF for fluid temperature
    # - ne                    is the number of elements in the 1D mesh
    # - H0e                   is the size of each element
    # - jac                   is the element jacobian
    # - GaussOrder            is the quadrature order (i.e., number of Gauss points)
    #--------------------------------------------------------------------------------------------------
    self.Element_Type = inputData.m_Element_Type
    if self.Element_Type.split('-')[0] == 'Q3' or self.Element_Type.split('-')[0] == 'Q3H':
      self.ndofSe = int(4)
    elif self.Element_Type.split('-')[0] == 'Q2':
      self.ndofSe = int(3)
    elif self.Element_Type.split('-')[0] == 'Q1':
      self.ndofSe = int(2)
    try:
      if self.Element_Type.split('-')[1] == 'Q3' or self.Element_Type.split('-')[1] == 'Q3H':
        self.ndofFe = int(4)
      elif self.Element_Type.split('-')[1] == 'Q2':
        self.ndofFe = int(3)
      elif self.Element_Type.split('-')[1] == 'Q1':
        self.ndofFe = int(2)
      else:
        self.ndofFe = 0
    except IndexError:
      self.ndofFe = 0
    try:
      if self.ndofFe > self.ndofSe:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid displacement interpolation order must be <= solid displacement interpolation order.")
    except AttributeError:
      pass
    self.ndofPe      = int(2)
    if 'T2' in self.Element_Type: # Stability unknown--do not use in conjunction with SUPG
      self.ndofTse   = int(3)
      self.ndofTfe   = int(3)
    else:
      self.ndofTse   = int(2)
      self.ndofTfe   = int(2)
    self.ne          = inputData.m_ne
    self.H0e         = self.H0/self.ne
    self.jac         = self.H0e/2.
    self.Gauss_Order = inputData.m_Gauss_Order
    #--------------------------------------------------------------------------------------------------
    # Initial & boundary condition parameters.
    #
    # Notes:
    # ------
    # - rho_0                         is the total mass density                                [kg/m^3]
    # - p_f0                          is the initial pressure of the pore fluid                [Pa]
    # - Ts_0                          is the initial temperature of the solid                  [K]
    # - Tf_0                          is the initial temperature of the pore fluid             [K]
    #
    # - solidDisplacementApply        controls whether or not a displacement boundary
    #                                 condition is applied to the solid
    # - solidDisplacementLocation     is the location(s) at which displacement is applied
    # - solidDisplacementApplication  is the functional form for solid displacement Dirichlet BC
    # - solidDisplacementValue        is the magnitude of the solid displacement Dirichlet BC  [m]
    #
    # - solidVelocityApply            controls whether or not a velocity boundary
    #                                 condition is applied to the solid
    # - solidVelocityLocation         is the location(s) at which velocity is applied
    # - solidVelocityApplication      is the functional form for solid velocity Dirichlet BC
    # - solidVelocityValue            is the magnitude of the solid velocity Dirichlet BC      [m/s]
    #
    # - tractionApply                 controls whether or not a traction boundary condition is
    #                                 applied
    # - tractionLocation              is the location(s) at which traction is applied
    # - tractionApplication           is the functional form for traction
    # - tractionOmega                 is the frequency for the applied sinusoidal load
    #                                 (if applicable)                                          [rad/s]
    # - tractionValue                 is the magnitude of the traction load                    [Pa]
    # - tractionStart                 is when the traction load starts being applied           [s]
    # - tractionStop                  is when the traction load stops  being applied           [s]
    #
    # - pressureApply                 controls whether or not the end(s) of the column is
    #                                 exposed to a Dirichlet BC on pore fluid pressure
    # - pressureLocation              is the location(s) at which the pore fluid pressure
    #                                 Dirichlet BC is applied
    # - pressureApplication           is the functional for pore fluid pressure Dirichlet BC
    # - pressureValue                 is the magnitude of the pore fluid pressure Dirichlet BC [Pa]
    #
    # - fluxApply                     controls whether or not the end(s) of the column is
    #                                 exposed to a Neumann BC on pore fluid pressure
    # - fluxLocation                  is the location(s) at which the pore fluid flux Neumann
    #                                 BC is applied
    # - fluxApplication               is the functional form for pore fluid flux Neumann BC
    # - fluxValue                     is the magnitude of the pore fluid flux Neumann BC       [m/s]
    # - fluidDisplacementApply        controls whether or not a displacement boundary
    #                                 condition is applied to the pore fluid
    # - fluidDisplacementLocation     is the location(s) at which pore fluid displacement is
    #                                 applied
    # - fluidDisplacementApplication  is the functional form for pore fluid displacement
    #                                 Dirichlet BC
    # - fluidDisplacementValue        is the magnitude of the pore fluid displacement          [m]
    #                                 Dirichlet BC
    # - fluidVelocityApply            controls whether or not a velocity boundary
    #                                 condition is applied to the pore fluid
    # - fluidVelocityLocation         is the location(s) at which pore fluid velocity is
    #                                 applied
    # - fluidVelocityApplication      is the functional form for pore fluid velocity
    #                                 Dirichlet BC
    # - fluidVelocityValue            is the magnitude of the pore fluid velocity Dirichlet    [m/s]
    #                                 BC
    # - solidTempApply                controls whether or not a temperature boundary
    #                                 condition is applied to the solid
    # - solidTemperatureLocation      is the location(s) at which temperature is applied
    # - solidTemperatureApplication   is the functional form for solid temperature Dirichlet
    #                                 BC
    # - solidTemperatureValue         is the magnitude of the solid temperature Dirichlet BC   [K]
    # - fluidTempApply                controls whether or not a temperature boundary
    #                                 condition is applied to the pore fluid
    # - fluidTemperatureLocation      is the location(s) at which pore fluid temperature is
    #                                 applied
    # - fluidTemperatureApplication   is the functional form for pore fluid temperature
    #                                 Dirichlet BC
    # - fluidTemperatureValue         is the magnitude of the pore fluid temperature Dirichlet 
    #                                 BC                                                       [K] 
    # - solidHeatFluxApply            controls whether or not a heat flux boundary
    #                                 condition is applied to the solid
    # - solidHeatFluxLocation         is the location(s) at which heat flux is applied
    # - solidHeatFluxApplication      is the functional form for solid heat flux Dirichlet
    #                                 BC
    # - solidHeatFluxValue            is the magnitude of the solid heat flux Dirichlet BC   [W/m^2-K]
    # - fluidHeatFluxApply            controls whether or not a heat flux boundary
    #                                 condition is applied to the pore fluid
    # - fluidHeatFluxLocation         is the location(s) at which pore fluid heat flux is
    #                                 applied
    # - fluidHeatFluxApplication      is the functional form for pore fluid heat flux
    #                                 Dirichlet BC
    # - fluidHeatFluxValue            is the magnitude of the pore fluid heat flux Dirichlet [W/m^2-K]
    #                                 BC     
    #--------------------------------------------------------------------------------------------------
    #----------------------------------
    # Set initial condition parameters.
    #----------------------------------
    self.rho_0 = inputData.m_rho_0
    #---------------------------------------------------------
    # Check to see if initial temperatures are at equilibrium.
    # If not, the physics breaks down in the theory.
    #---------------------------------------------------------
    if self.Ts_0 != self.Tf_0 and 'tf' in self.Physics:
      print("-----------------\nINPUT FILE ERROR:\n-----------------")
      print("Initial solid phase temperature must match initial pore fluid phase temperature.")
      print("If they do not match, the theory upon which this code was written breaks down.")
      sys.exit()
    #----------------------
    # Set the gas constant.
    #----------------------
    if self.Physics == 'u-t' and self.solidModel == 'Ideal-Gas':
      self.RGas = self.p_f0/(self.rhosR_0*self.Ts_0)
    else:
      try:
        self.RGas = self.p_f0/(self.rhofR_0*self.Tf_0)
      except TypeError:
        self.RGas = 0
    #--------------------------------------
    # Set solid displacement BC parameters.
    #--------------------------------------
    self.solidDisplacementApply       = inputData.m_solidDisplacementApply
    self.solidDisplacementLocation    = inputData.m_solidDisplacementLocation
    self.solidDisplacementApplication = inputData.m_solidDisplacementApplication
    try:
      self.solidDisplacementValue     = inputData.m_solidDisplacementValue
    except AttributeError:
      self.solidDisplacementValueTop  = inputData.m_solidDisplacementValueTop
      self.solidDisplacementValueBot  = inputData.m_solidDisplacementValueBot
    try:
      self.solidDisplacementT0        = inputData.m_solidDisplacementT0
      self.solidDisplacementT1        = inputData.m_solidDisplacementT1
    except AttributeError:
      self.solidDisplacementT0Bot     = inputData.m_solidDisplacementT0Bot
      self.solidDisplacementT0Top     = inputData.m_solidDisplacementT0Top
      self.solidDisplacementT1Bot     = inputData.m_solidDisplacementT1Bot
      self.solidDisplacementT1Top     = inputData.m_solidDisplacementT1Top
    #----------------------------------
    # Set solid velocity BC parameters.
    #----------------------------------
    self.solidVelocityApply       = inputData.m_solidVelocityApply
    self.solidVelocityLocation    = inputData.m_solidVelocityLocation
    self.solidVelocityApplication = inputData.m_solidVelocityApplication
    try:
      self.solidVelocityValue     = inputData.m_solidVelocityValue
    except AttributeError:
      self.solidVelocityValueTop  = inputData.m_solidVelocityValueTop
      self.solidVelocityValueBot  = inputData.m_solidVelocityValueBot
    try:
      self.solidVelocityT0        = inputData.m_solidVelocityT0
      self.solidVelocityT1        = inputData.m_solidVelocityT1
    except AttributeError:
      self.solidVelocityT0Bot     = inputData.m_solidVelocityT0Bot
      self.solidVelocityT0Top     = inputData.m_solidVelocityT0Top
      self.solidVelocityT1Bot     = inputData.m_solidVelocityT1Bot
      self.solidVelocityT1Top     = inputData.m_solidVelocityT1Top
    #---------------------------------
    # Set traction load BC parameters.
    #---------------------------------
    self.tractionApply       = inputData.m_tractionApply
    self.tractionLocation    = inputData.m_tractionLocation
    self.tractionApplication = inputData.m_tractionApplication
    try:
      self.tractionOmega     = inputData.m_tractionOmega
    except AttributeError:
      self.tractionOmegaTop  = inputData.m_tractionOmegaTop
      self.tractionOmegaBot  = inputData.m_tractionOmegaBot
    try:
      self.tractionValue     = inputData.m_tractionValue
    except AttributeError:
      self.tractionValueTop  = inputData.m_tractionValueTop
      self.tractionValueBot  = inputData.m_tractionValueBot
    try:
      self.tractionT0        = inputData.m_tractionT0
      self.tractionT1        = inputData.m_tractionT1
    except AttributeError:
      self.tractionT0Bot     = inputData.m_tractionT0Bot
      self.tractionT0Top     = inputData.m_tractionT0Top
      self.tractionT1Bot     = inputData.m_tractionT1Bot
      self.tractionT1Top     = inputData.m_tractionT1Top 
    #-------------------------------------------------
    # Set pore fluid pressure Dirichlet BC parameters.
    #-------------------------------------------------
    self.pressureApply       = inputData.m_pressureApply
    self.pressureLocation    = inputData.m_pressureLocation
    self.pressureApplication = inputData.m_pressureApplication
    try:
      self.pressureValue     = inputData.m_pressureValue
    except AttributeError:
      self.pressureValueTop  = inputData.m_pressureValueTop
      self.pressureValueBot  = inputData.m_pressureValueBot
    try:
      self.pressureT0        = inputData.m_pressureT0
      self.pressureT1        = inputData.m_pressureT1
    except AttributeError:
      self.pressureT0Bot     = inputData.m_pressureT0Bot
      self.pressureT0Top     = inputData.m_pressureT0Top
      self.pressureT1Bot     = inputData.m_pressureT1Bot
      self.pressureT1Top     = inputData.m_pressureT1Top
    #-------------------------------------------
    # Set pore fluid flux Neumann BC parameters.
    #-------------------------------------------
    self.fluxApply       = inputData.m_fluxApply
    self.fluxLocation    = inputData.m_fluxLocation
    self.fluxApplication = inputData.m_fluxApplication
    try:
      self.fluxValue     = inputData.m_fluxValue
    except AttributeError:
      self.fluxValueTop  = inputData.m_fluxValueTop
      self.fluxValueBot  = inputData.m_fluxValueBot
    try:
      self.fluxT0        = inputData.m_fluxT0
      self.fluxT1        = inputData.m_fluxT1
    except AttributeError:
      self.fluxT0Bot     = inputData.m_fluxT0Bot
      self.fluxT0Top     = inputData.m_fluxT0Top
      self.fluxT1Bot     = inputData.m_fluxT1Bot
      self.fluxT1Top     = inputData.m_fluxT1Top
    #-------------------------------------------
    # Set pore fluid displacement BC parameters.
    #-------------------------------------------
    self.fluidDisplacementApply       = inputData.m_fluidDisplacementApply
    self.fluidDisplacementLocation    = inputData.m_fluidDisplacementLocation
    self.fluidDisplacementApplication = inputData.m_fluidDisplacementApplication
    try:
      self.fluidDisplacementValue     = inputData.m_fluidDisplacementValue
    except AttributeError:
      self.fluidDisplacementValueTop  = inputData.m_fluidDisplacementValueTop
      self.fluidDisplacementValueBot  = inputData.m_fluidDisplacementValueBot
    try:
      self.fluidDisplacementT0        = inputData.m_fluidDisplacementT0
      self.fluidDisplacementT1        = inputData.m_fluidDisplacementT1
    except AttributeError:
      self.fluidDisplacementT0Bot     = inputData.m_fluidDisplacementT0Bot
      self.fluidDisplacementT0Top     = inputData.m_fluidDisplacementT0Top
      self.fluidDisplacementT1Bot     = inputData.m_fluidDisplacementT1Bot
      self.fluidDisplacementT1Top     = inputData.m_fluidDisplacementT1Top
    #---------------------------------------
    # Set pore fluid velocity BC parameters.
    #---------------------------------------
    self.fluidVelocityApply       = inputData.m_fluidVelocityApply
    self.fluidVelocityLocation    = inputData.m_fluidVelocityLocation
    self.fluidVelocityApplication = inputData.m_fluidVelocityApplication
    try:
      self.fluidVelocityValue     = inputData.m_fluidVelocityValue
    except AttributeError:
      self.fluidVelocityValueTop  = inputData.m_fluidVelocityValueTop
      self.fluidVelocityValueBot  = inputData.m_fluidVelocityValueBot
    try:
      self.fluidVelocityT0        = inputData.m_fluidVelocityT0
      self.fluidVelocityT1        = inputData.m_fluidVelocityT1
    except AttributeError:
      self.fluidVelocityT0Bot     = inputData.m_fluidVelocityT0Bot
      self.fluidVelocityT0Top     = inputData.m_fluidVelocityT0Top
      self.fluidVelocityT1Bot     = inputData.m_fluidVelocityT1Bot
      self.fluidVelocityT1Top     = inputData.m_fluidVelocityT1Top
    #-------------------------------------
    # Set solid temperature BC parameters.
    #-------------------------------------
    self.solidTempApply       = inputData.m_solidTempApply
    self.solidTempLocation    = inputData.m_solidTempLocation
    self.solidTempApplication = inputData.m_solidTempApplication
    try:
      self.solidTempValue     = inputData.m_solidTempValue
    except AttributeError:
      self.solidTempValueTop  = inputData.m_solidTempValueTop
      self.solidTempValueBot  = inputData.m_solidTempValueBot
    try:
      self.solidTempT0        = inputData.m_solidTempT0
      self.solidTempT1        = inputData.m_solidTempT1
    except AttributeError:
      self.solidTempT0Bot     = inputData.m_solidTempT0Bot
      self.solidTempT0Top     = inputData.m_solidTempT0Top
      self.solidTempT1Bot     = inputData.m_solidTempT1Bot
      self.solidTempT1Top     = inputData.m_solidTempT1Top
    #------------------------------------------
    # Set pore fluid temperature BC parameters.
    #------------------------------------------
    self.fluidTempApply       = inputData.m_fluidTempApply
    self.fluidTempLocation    = inputData.m_fluidTempLocation
    self.fluidTempApplication = inputData.m_fluidTempApplication
    try:
      self.fluidTempValue     = inputData.m_fluidTempValue
    except AttributeError:
      self.fluidTempValueTop  = inputData.m_fluidTempValueTop
      self.fluidTempValueBot  = inputData.m_fluidTempValueBot
    try:
      self.fluidTempT0        = inputData.m_fluidTempT0
      self.fluidTempT1        = inputData.m_fluidTempT1
    except AttributeError:
      self.fluidTempT0Bot     = inputData.m_fluidTempT0Bot
      self.fluidTempT0Top     = inputData.m_fluidTempT0Top
      self.fluidTempT1Bot     = inputData.m_fluidTempT1Bot
      self.fluidTempT1Top     = inputData.m_fluidTempT1Top
    #-----------------------------------
    # Set solid heat flux BC parameters.
    #-----------------------------------
    self.solidHeatFluxApply       = inputData.m_solidHeatFluxApply
    self.solidHeatFluxLocation    = inputData.m_solidHeatFluxLocation
    self.solidHeatFluxApplication = inputData.m_solidHeatFluxApplication
    try:
      self.solidHeatFluxValue     = inputData.m_solidHeatFluxValue
    except AttributeError:
      self.solidHeatFluxValueTop  = inputData.m_solidHeatFluxValueTop
      self.solidHeatFluxValueBot  = inputData.m_solidHeatFluxValueBot
    try:
      self.solidHeatFluxT0        = inputData.m_solidHeatFluxT0
      self.solidHeatFluxT1        = inputData.m_solidHeatFluxT1
    except AttributeError:
      self.solidHeatFluxT0Bot     = inputData.m_solidHeatFluxT0Bot
      self.solidHeatFluxT0Top     = inputData.m_solidHeatFluxT0Top
      self.solidHeatFluxT1Bot     = inputData.m_solidHeatFluxT1Bot
      self.solidHeatFluxT1Top     = inputData.m_solidHeatFluxT1Top
    #----------------------------------------
    # Set pore fluid heat flux BC parameters.
    #----------------------------------------
    self.fluidHeatFluxApply       = inputData.m_fluidHeatFluxApply
    self.fluidHeatFluxLocation    = inputData.m_fluidHeatFluxLocation
    self.fluidHeatFluxApplication = inputData.m_fluidHeatFluxApplication
    try:
      self.fluidHeatFluxValue     = inputData.m_fluidHeatFluxValue
    except AttributeError:
      self.fluidHeatFluxValueTop  = inputData.m_fluidHeatFluxValueTop
      self.fluidHeatFluxValueBot  = inputData.m_fluidHeatFluxValueBot
    try:
      self.fluidHeatFluxT0        = inputData.m_fluidHeatFluxT0
      self.fluidHeatFluxT1        = inputData.m_fluidHeatFluxT1
    except AttributeError:
      self.fluidHeatFluxT0Bot     = inputData.m_fluidHeatFluxT0Bot
      self.fluidHeatFluxT0Top     = inputData.m_fluidHeatFluxT0Top
      self.fluidHeatFluxT1Bot     = inputData.m_fluidHeatFluxT1Bot
      self.fluidHeatFluxT1Top     = inputData.m_fluidHeatFluxT1Top
    #-------------
    # MMS details.
    #-------------
    try:
      self.MMS                      = inputData.m_MMS
      self.MMS_SolidSolutionType    = inputData.m_MMS_SolidSolutionType
      self.MMS_PressureSolutionType = inputData.m_MMS_PressureSolutionType
      self.MMS_FluidSolutionType    = inputData.m_MMS_FluidSolutionType
      #-----------------------------------
      # Check restrictions for dev branch.
      #-----------------------------------
      if self.MMS:
        err_msg0   = "-----------------\nINPUT FILE ERROR:\n-----------------\n"
        err_msg1   = "Currently, MMS must be used under the following restrictions:\n"
        err_msg2   = "- (u) or (u-pf) formulation with inertia terms or (u-uf-pf) formulation\n"
        err_msg3   = "- neo-Hookean or neo-Hookean-Eipper (multiphase only) solid material\n"
        err_msg4   = "- Linear-Bulk pore fluid density model\n"
        err_msg4a  = "- Nearly-inviscid pore fluid\n"
        err_msg5   = "- Kozeny-Carman (neo-Hookean) or Hyperbolic (neo-Hookean-Eipper) hydraulic conductivity model\n"
        err_msg6   = "- S2T2/S2T3/MS2T3/S3T2/S3T3/MS3T3/S4T3/MS4T3 solution space for solid      displacement\n"
        err_msg6a  = "- S2T3/S3T3/MS3T3/S4T3/MS4T3                 solution space for pore fluid displacement\n"
        err_msg7   = "- S1T1/S1T2                                  solution space for pore fluid pressure\n"
        err_msg8   = "- Neumann BCs at the top, Dirichlet BCs at the bottom\n"
        err_msg8a  = "    Specifically: \n"
        err_msg8a += "    - an applied traction                 with: APPLICATION-TYPE : MMS\n"
        err_msg8b  = "    - an applied displacement(s)          with: LOCATION : Bottom ; APPLICATION-TYPE : Constant ; MAGNITUDE : 0 ; T0 (s) : 0 ; T1 (s) : END TIME\n"
        err_msg8c  = "    and, if applicable: \n"
        err_msg8c += "    - an applied pore fluid pressure      with: LOCATION : BOTTOM ; APPLICATION-TYPE : MMS\n"
        err_msg8d  = "    - an applied pore fluid pressure flux with: APPLICATION-TYPE : MMS\n\n"
        err_msg9   = "- Resctrictions of the following parameters:\n" 
        err_msg9a  = "  - lambda                 = 1 Pa\n"
        err_msg9b  = "  - mu                     = 1 Pa\n"
        err_msg9c  = "  - rhosR_0                = 2 kg/m^3\n"
        err_msg9d  = "  - rhofR_0                = 1 kg/m^3 or None\n"
        err_msg9e  = "  - nf_0                   = 0.5      or None\n"
        # err_msg9f  = "  - Intrinsic permeability = 1 m^2    or None\n"
        err_msg9g  = "  - Shear viscosity        = 1 Pa-s   or None\n"
        err_msg9h  = "  - pf_0                   = 0 Pa\n"
        err_msg10  = "\nRecommendations for parameter values are as follows:\n"
        err_msg10a = " - KF              ~ >= O(100) Pa (neo-Hookean); KF ~ O(1) (neo-Hookean-Eipper)\n"
        err_msg10b = " - Gravity         ~ O(10)     m/s^2\n"
        err_msg10c = " - Height and Area ~ O(1)      m\n"
        err_msg10d = " - 5th order Gauss quadrature\n"
        err_msg10e = " - Newmark-beta integration"
        
        err_msg = err_msg0 + err_msg1 + err_msg2 + err_msg3 + err_msg4 + err_msg4a + err_msg5 + err_msg6 + err_msg6a + err_msg7 + err_msg8 + err_msg8a + err_msg8b + err_msg8c + err_msg8d\
                  + err_msg9 + err_msg9a + err_msg9b + err_msg9c + err_msg9d + err_msg9e + err_msg9g + err_msg9h\
                  + err_msg10 + err_msg10a + err_msg10b + err_msg10c + err_msg10d + err_msg10e

        if 'neo-Hookean' not in self.solidModel:
          if self.solidModel != 'Saint-Venant-Kirchhoff':
            sys.exit(err_msg)
        if self.fluidModel != 'None':
          if self.fluidModel !='Linear-Bulk':
            sys.exit(err_msg)
        if not self.isDynamics:
          sys.exit(err_msg)
        if self.MMS_SolidSolutionType != 'S2T2':
          if self.MMS_SolidSolutionType != 'S2T3':
            if self.MMS_SolidSolutionType != 'MS2T3':
              if self.MMS_SolidSolutionType != 'S3T2':
                if self.MMS_SolidSolutionType != 'S3T3':
                  if self.MMS_SolidSolutionType != 'MS3T3':
                    if self.MMS_SolidSolutionType != 'S4T3':
                      if self.MMS_SolidSolutionType != 'MS4T3':
                        sys.exit(err_msg)
        if self.MMS_FluidSolutionType is not None:
          if self.MMS_FluidSolutionType != 'S2T3':
            if self.MMS_FluidSolutionType != 'S3T3':
              if self.MMS_FluidSolutionType != 'MS3T3':
                if self.MMS_FluidSolutionType != 'S4T3':
                  if self.MMS_FluidSolutionType != 'MS4T3':
                    sys.exit(err_msg)
        if self.MMS_PressureSolutionType is not None:
          if self.MMS_PressureSolutionType != 'S1T1':
            if self.MMS_PressureSolutionType != 'S1T2':
              if self.MMS_PressureSolutionType != 'S1T3':
                if self.MMS_PressureSolutionType != '2S1T3':
                  sys.exit(err_msg)
        if self.solidDisplacementApply:
          if self.solidDisplacementLocation != 'Bottom':
            sys.exit(err_msg)
        if self.solidVelocityApply:
          sys.exit(err_msg)
        if self.tractionApply:
          if self.tractionLocation != 'Top' or self.tractionApplication != 'MMS':
            sys.exit(err_msg)
        if self.fluidDisplacementApply:
          if self.fluidDisplacementLocation != 'Bottom':
            sys.exit(err_msg)
        if self.fluidVelocityApply:
          sys.exit(err_msg)
        if self.pressureApply:
          if self.pressureLocation != 'Bottom' or self.pressureApplication != 'MMS':
            sys.exit(err_msg)
        if self.fluxApply:
          if self.fluxLocation != 'Top' or self.fluxApplication != 'MMS':
            sys.exit(err_msg)
        if self.solidTempApply or self.solidHeatFluxApply or self.fluidTempApply or self.fluidHeatFluxApply:
          sys.exit(err_msg)
        if self.lambd != 1:
          sys.exit(err_msg)
        if self.mu != 1:
          sys.exit(err_msg)
        if self.rhosR_0 != 2:
          sys.exit(err_msg)
        if self.Physics == 'u-pf':
          if not self.fluxApply:
            sys.exit(err_msg)
          else:
            if self.fluxLocation != 'Top':
              sys.exit(err_msg)
          if self.rhofR_0 != 1:
            sys.exit(err_msg)
          if self.fluidShearVisc != 1:
            sys.exit(err_msg)
          if self.DarcyBrinkman:
            sys.exit(err_msg)
          if self.khatType != 'Constant':
            if self.khatType != 'Kozeny-Carman':
              if self.khatType != 'Hyperbolic':
                sys.exit(err_msg)
          if self.p_f0 > 0:
            sys.exit(err_msg)
    except AttributeError:
      self.MMS = False

    return
#-------------------
# Begin main script.
#-------------------
if __name__ == '__main__':

  np.seterr(all='raise') # Raise runtime errors for division by zero, or log(0)
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})

  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to run 1-D finite element simulations.\
                                                Specifically, it calls an input file reader to parse user inputs,\
                                                and then converts appropriate inputs to a data object that can\
                                                then be referenced by lower level programs. Based on user input,\
                                                an appropriate physics module is called to solve the governing\
                                                equations.')
  parser.add_argument('inputFile', metavar='i', type=str,
                      help='the file path to the input file')
  parser.add_argument('-t', '--test', action='store_true',
                      help='enable running of test cases')
  parser.add_argument('-r', '--remove', action='store_true',
                      help='remove old data from save directory')
  parser.add_argument('-d', '--debug', action='store_true',
                       help='run in debug mode')
  parser.add_argument('-m', '--isothermal', action='store_true',
                      help='flag for enabling the isothermal assumption')
  parser.add_argument('-s', '--staggered', action='store_true',
                      help='flag for switching order of solution staggering in RK (u-pf-ts-tf) formulation, i.e., solve tfDot before p_fDot')
  args = parser.parse_args()
  #-----------------
  # Read input file.
  #-----------------
  inputData = simInput.SimInputs(args.inputFile)
  inputData.readInputFile()

  if args.test:
    inputData.m_simPath = args.inputFile.split('.dat')[0].split('input')[0]
  #----------------------------
  # Initiate solver parameters.
  #----------------------------
  params = Parameters(inputData)

  params.dataDir = inputData.m_simPath

  if not os.path.exists(params.dataDir):
    os.makedirs(params.dataDir)
  #-------------------------------------
  # Copy input file to data destination.
  #-------------------------------------
  if not args.test:
    inputFileName = args.inputFile.split('/')[-1]
    shutil.copy(args.inputFile, params.dataDir + inputFileName)
  #----------------------------------------------------
  # Delte old data from directory if present and asked.
  #----------------------------------------------------
  oldDataList = glob.glob(params.dataDir + "*.npy")
  if len(oldDataList) > 0 and args.remove:
    for oldData in oldDataList:
      if '_std' not in oldData:
        os.remove(oldData)
    print("Successfully removed prior data.")

  if args.debug:
    params.debug = True
    params.printTraceback = True
  else:
    params.debug = False
    params.printTraceback = False
  if args.isothermal:
    params.isothermalAssumption = True
  else:
    params.isothermalAssumption = False
  if args.staggered:
    params.staggered = True
  else:
    params.staggered = False
  #------------
  # Run solver.
  #------------
  if params.Physics == 'driver':
    solver_driver.main(params)
  elif params.Physics == 'user':
    solver_user.main(params)
  elif params.Physics == 'u':
    solver_u.main(params)
  elif params.Physics == 'u-pf':
    solver_upf.main(params) 
  elif params.Physics == 'u-t':
    solver_ut.main(params)
  elif params.Physics == 'u-uf-pf':
    solver_uufpf.main(params)
  elif params.Physics == 'u-pf-ts-tf':
    solver_upftstf.main(params)
  elif params.Physics == 'u-uf-pf-ts-tf':
    solver_uufpftstf.main(params)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPhysics of problem not recognized.")

