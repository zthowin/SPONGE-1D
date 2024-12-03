#----------------------------------------------------------------------------------------
# Module housing element object variables.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    November 27, 2024
#----------------------------------------------------------------------------------------
import sys

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import Lib
except ImportError:
  sys.exit("MODULE WARNING. 'Lib.py' not found, check configuration.")

__methods__     = []
register_method = Lib.register_method(__methods__)

@register_method
def compute_variables(self, Parameters, VariationalEq=None):
  # Top level function to compute element variables for a given
  # variational equation at current time-step.
  if VariationalEq == 'G':
    self.compute_G_variables(Parameters)
  elif VariationalEq == 'H':
    self.compute_H_variables(Parameters)
  elif VariationalEq == 'I':
    self.compute_I_variables(Parameters)
  elif VariationalEq == 'J':
    self.compute_J_variables(Parameters)
  elif VariationalEq == 'K':
    self.compute_K_variables(Parameters)
  elif VariationalEq == 'All':
    self.compute_all_variables(Parameters)
  else:
    sys.exit("------\nERROR:\n------\nVariational equation not recognized, check source code.")
  return

@register_method
def compute_all_variables(self, Parameters):
  # Top level function to compute element variables
  # used in multiple variational equations.
  self.get_a_s()
  if 'pf' in Parameters.Physics:
    self.get_p_f()
  self.get_dudX()
  self.get_dvdX()
  self.get_F11()
  self.get_J()
  self.get_JDot()
  if 'pf' in Parameters.Physics:
    self.get_ns(Parameters)
    self.get_nf()
    self.get_khat(Parameters)
    self.get_dkhat(Parameters)
  self.get_rhosR(Parameters)
  if Parameters.LagrangeApply:
    self.get_dnfdX()
  if 't' in Parameters.Physics:
    self.get_ts()
    self.get_tsDot()
    self.get_dtsdX()
    self.get_qs(Parameters)
  if 'tf' in Parameters.Physics:
    self.get_tf()
    self.get_tfDot()
    self.get_dtfdX()
    self.get_qf(Parameters)
    self.get_dnfdX()
  if 'pf' in Parameters.Physics:
    self.get_rhofR(Parameters)
  self.get_rho(Parameters)
  self.get_rho_0(Parameters)
  self.get_rhos_0(Parameters)
  if 'pf' in Parameters.Physics:
    self.get_rhof_0()
  self.get_c(Parameters)
  self.get_Q(Parameters)
  self.get_P11(Parameters)
  if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
    self.get_dt(Parameters)
  if 'pf' in Parameters.Physics:
    self.get_p_fDot()
    self.get_dp_fdX()
    if Parameters.alpha_stab > 0:
      self.get_dp_fDotdX()
  if 'uf' in Parameters.Physics:
    self.get_v_s()
    self.get_v_f()
    self.get_a_f()
    if Parameters.DarcyBrinkman:
      self.get_dvfdX()
      self.get_FES(Parameters)
      self.get_DIV_FES(Parameters)
  if 'pf' in Parameters.Physics:
    self.get_vDarcy(Parameters)

  return

@register_method
def compute_G_variables(self, Parameters):
  # Top level function to compute element variables used
  # in the balance of momentum of the solid skeleton.
  self.get_a_s()
  self.get_dudX()
  self.get_dvdX()
  self.get_F11()
  self.get_J()
  if 'pf' in Parameters.Physics:
    self.get_p_f()
    if 'tf' in Parameters.Physics:
      self.get_tf()
    # if Parameters.integrationScheme == 'Predictor-corrector':
      # self.get_p_fDot()
    # try: # For predictor-corrector
    #   if Parameters.StarStar:
    #     self.p_fLast = np.einsum('ik, k', self.Np, self.p_f_globalLast)
    # except AttributeError:
    #   pass
  self.get_rho(Parameters)
  self.get_rho_0(Parameters)
  if 't' in Parameters.Physics:
    self.get_ts()
  if 'uf' in Parameters.Physics:
    self.get_a_f()
    self.get_dvfdX()
    self.get_Qf(Parameters)
    self.get_rhof_0()
    if 't' not in Parameters.Physics:
      self.get_rhos_0(Parameters)
    if Parameters.DarcyBrinkman:
      self.get_dvfdX()
      self.get_FES(Parameters)
  if 'tf' in Parameters.Physics:
    if 'uf' not in Parameters.Physics:
      self.get_rhof_0()
  self.get_c(Parameters)
  self.get_Q(Parameters)
  self.get_P11(Parameters)
  if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
    self.get_dt(Parameters)
  
  return

@register_method
def compute_H_variables(self, Parameters):
  # Top level function to compute element variables used
  # in the balance of mass of the mixture.
  self.get_p_f()
  self.get_dudX()
  self.get_dvdX()
  self.get_F11()
  self.get_J()
  self.get_JDot()
  self.get_p_fDot()
  if 'Central-difference' in Parameters.integrationScheme:
    self.get_p_fDDot()
  self.get_dp_fdX()
  self.get_ns(Parameters)
  self.get_nf()
  if 'tf' in Parameters.Physics:
    self.get_ts()
    self.get_tf()
    self.get_tfDot()
    self.get_dtfdX()
    self.get_dnfdX()
  self.get_rhofR(Parameters)
  self.get_khat(Parameters)
  if Parameters.alpha_stab > 0 and 'RK' not in Parameters.integrationScheme:
    self.get_dp_fDotdX()
    if Parameters.integrationScheme == 'Central-difference':
      self.get_dp_fDDotdX()
  if 'uf' in Parameters.Physics:
    self.get_a_f()
    self.get_dvfdX()
    self.get_d2vfdX2()
    self.get_Qf(Parameters)
    self.get_drhofRdX(Parameters)
    self.get_d2udX2()
    self.get_DIV_Qf(Parameters)
    if Parameters.DarcyBrinkman:
      self.get_dvfdX()
      self.get_DIV_FES(Parameters)
  else:
    self.get_a_s()
  # if Parameters.integrationScheme == 'Predictor-corrector':
  #   self.get_rho(Parameters)
  #   self.get_rho_0(Parameters)
  self.get_vDarcy(Parameters)
  return

@register_method
def compute_I_variables(self, Parameters):
  # Top level function to compute element variables used
  # in the balance of momentum of the pore fluid.
  self.get_v_s()
  self.get_v_f()
  self.get_a_f()
  self.get_p_f()
  self.get_dp_fdX()
  self.get_dudX()
  self.get_J()
  self.get_ns(Parameters)
  self.get_nf()
  if Parameters.DarcyBrinkman:
    self.get_F11()
    self.get_dvfdX()
    self.get_FES(Parameters)
    self.get_DIV_FES(Parameters)
  self.get_d2udX2()
  self.get_dvfdX()
  self.get_d2vfdX2()
  if 'tf' in Parameters.Physics:
    if not Parameters.DarcyBrinkman:
      self.get_F11()
      self.get_dnfdX()
    self.get_ts()
    self.get_tf()
    self.get_dtfdX()
  self.get_rhofR(Parameters)
  self.get_rhof_0()
  self.get_khat(Parameters)
  self.get_drhofRdX(Parameters)
  self.get_Qf(Parameters)
  self.get_DIV_Qf(Parameters)
  return

@register_method
def compute_J_variables(self, Parameters):
  # Top level function to compute element variables used
  # in the balance of energy of the solid.
  self.get_dudX()
  self.get_dvdX()
  self.get_F11()
  self.get_J()
  self.get_dtsdX()
  self.get_JDot()
  self.get_ts()
  self.get_rhosR(Parameters)
  if Parameters.Physics == 'u-t':
    self.get_rho(Parameters)
    self.get_rho_0(Parameters)
    self.rhos_0 = self.rho_0
    self.get_rho(Parameters)
    self.get_c(Parameters)
    self.get_Q(Parameters)
    self.get_qs(Parameters)
  else:
    self.get_p_f()
    self.get_tf()
    self.get_ns(Parameters)
    self.get_nf()
    self.get_rhos_0(Parameters)
    self.get_khat(Parameters)
    self.get_dnfdX()
    if 'uf' in Parameters.Physics:
      self.get_v_f()
      self.get_v_s()
    else:
      self.get_dp_fdX()
      self.get_rhofR(Parameters)
      self.get_a_s()
      self.get_vDarcy(Parameters)
    self.get_c(Parameters)
    self.get_Q(Parameters)
    self.get_qs(Parameters)
  return

@register_method
def compute_K_variables(self, Parameters):
  # Top level function to compute element variables used
  # in the balance of energy of the pore fluid.
  self.get_p_f()
  self.get_ts()
  self.get_tf()
  self.get_dudX()
  self.get_F11()
  self.get_J()
  self.get_dtfdX()
  self.get_ns(Parameters)
  self.get_nf()
  self.get_qf(Parameters)
  self.get_rhofR(Parameters)
  self.get_rhof_0()
  self.get_dnfdX()
  if 'uf' in Parameters.Physics:
    self.get_v_f()
    self.get_v_s()
    self.get_dvfdX()
    self.get_Qf(Parameters)
  else:
    self.get_p_fDot()
    self.get_JDot()
    self.get_a_s()
    self.get_dp_fdX()
    self.get_khat(Parameters)
    self.get_vDarcy(Parameters)
  return

@register_method
def get_p_f(self):
  # Interpolate pore fluid pressure.
  self.p_f = np.einsum('ik, i', self.Np, self.p_f_global, dtype=np.float64)
  return

@register_method
def get_p_fDot(self):
  # Interpolate first time derivative on pore fluid pressure.
  self.p_fDot = np.einsum('ik, i', self.Np, self.p_fDot_global, dtype=np.float64)
  return

@register_method
def get_p_fDDot(self):
  # Interpolate second time derivative on pore fluid pressure.
  self.p_fDDot = np.einsum('ik, i', self.Np, self.p_fDDot_global, dtype=np.float64)
  return

@register_method
def get_dp_fdX(self):
  # Interpolate pore fluid pressure gradient.
  self.dp_fdX = np.einsum('ik, i', self.Bp, self.p_f_global, dtype=np.float64)
  return

@register_method
def get_dp_fDotdX(self):
  # Interpolate first time derivative on pore fluid pressure gradient.
  self.dp_fDotdX = np.einsum('ik, i', self.Bp, self.p_fDot_global, dtype=np.float64)
  return

@register_method
def get_dp_fDDotdX(self):
  # Interpolate second time derivative on pore fluid pressure gradient.
  self.dp_fDDotdX = np.einsum('ik, i', self.Bp, self.p_fDDot_global, dtype=np.float64)
  return

@register_method
def get_v_s(self):
  # Interpolate solid skeleton velocity.
  self.v_s = np.einsum('ik, i', self.Nu, self.v_s_global, dtype=np.float64)
  return

@register_method
def get_a_s(self):
  # Interpolate solid skeleton acceleration.
  self.a_s = np.einsum('ik, i', self.Nu, self.a_s_global, dtype=np.float64)
  return

@register_method
def get_dudX(self):
  # Compute solid skeleton displacement gradient.
  self.dudX = np.einsum('ik, i', self.Bu, self.u_s_global, dtype=np.float64)
  return


@register_method
def get_d2udX2(self):
  # Compute second order gradient of solid displacement.
  self.d2udX2 = np.einsum('ik, i', self.B2u, self.u_s_global, dtype=np.float64)
  return

@register_method
def get_dvdX(self):
  # Compute solid skeleton velocity gradient.
  self.dvdX = np.einsum('ik, i', self.Bu, self.v_s_global, dtype=np.float64)
  return

@register_method
def get_dadX(self):
  # Compute solid skeleton acceleration gradient.
  self.dadX = np.einsum('ik, i', self.Bu, self.a_s_global, dtype=np.float64)
  return

@register_method
def get_v_f(self):
  # Interpolate pore fluid velocity.
  self.v_f = np.einsum('ik, i', self.Nuf, self.v_f_global, dtype=np.float64)
  return

@register_method
def get_a_f(self):
  # Interpolate pore fluid acceleration.
  self.a_f = np.einsum('ik, i', self.Nuf, self.a_f_global, dtype=np.float64)
  return

@register_method
def get_dvfdX(self):
  # Compute fluid velocity gradient.
  self.dvfdX = np.einsum('ik, i', self.Buf, self.v_f_global, dtype=np.float64)
  return

@register_method
def get_d2vfdX2(self):
  # Compute second derivative of pore fluid velocity.
  self.d2vfdX2 = np.einsum('ik, i', self.B2uf, self.v_f_global, dtype=np.float64)
  return

@register_method
def get_ts(self):
  # Interpolate solid temperature.
  self.ts = np.einsum('ik, i', self.Nts, self.ts_global, dtype=np.float64)
  return

@register_method
def get_tsDot(self):
  # Interpolate first time derivative of solid temperature.
  self.tsDot = np.einsum('ik, i', self.Nts, self.tsDot_global, dtype=np.float64)
  return

@register_method
def get_dtsdX(self):
  # Interpolate solid temperature gradient.
  self.dtsdX = np.einsum('ik, i', self.Bts, self.ts_global, dtype=np.float64)
  return

@register_method
def get_tf(self):
  # Interpolate pore fluid temperature.
  self.tf = np.einsum('ik, i', self.Ntf, self.tf_global, dtype=np.float64)
  return

@register_method
def get_tfDot(self):
  # Interpolate first time derivative of pore fluid temperature.
  self.tfDot = np.einsum('ik, i', self.Ntf, self.tfDot_global, dtype=np.float64)
  return

@register_method
def get_dtfdX(self):
  # Interpolate pore fluid temperature gradient.
  self.dtfdX = np.einsum('ik, i', self.Btf, self.tf_global, dtype=np.float64)
  return

@register_method
def get_dnfdX(self):
  # Compute gradient of porosity.
  self.get_d2udX2()
  self.dnfdX = self.ns*self.d2udX2/self.F11
  return

@register_method
def get_FES(self, Parameters):
  # Compute viscous pore fluid Cauchy stress tensor.
  self.FES = self.nf*self.dvfdX*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/self.F11
  return

@register_method
def get_DIV_FES(self, Parameters):
  # Compute divergence on viscous pore fluid stress tensor term.
  self.get_d2vfdX2()
  self.get_dnfdX()
  self.DIV_FES = (self.dnfdX*self.dvfdX + self.nf*self.d2vfdX2)*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/(self.F11)
  return

@register_method
def get_DIV_Qf(self, Parameters):
  # Compute the gradient of the pore fluid shock viscosity.
  if np.any(self.dvfdX) < 0:
    if Parameters.fluidModel == 'Exponential':
      T1 = self.d2udX2/(self.nf*self.J) + self.dp_fdX/Parameters.KF
      T2 = 2*Parameters.C0*self.dvfdX/self.J - Parameters.C1*Parameters.cf

      self.DIV_Qf = np.zeros(self.Gauss_Order)

      self.DIV_Qf[self.Qfidxs] = (self.Qf*T1 + self.nf*self.rhofR*self.d2vfdX2*Parameters.H0e*T2)[self.Qfidxs]
    else:
      warnings.simplefilter('once', RuntimeWarning("WARNING. Pore fluid shock viscosity not implemented for constitutive model of choice, automatically disabling."))
      self.DIV_Qf = 0
      self.Qf     = 0
  else:
    self.DIV_Qf = 0
  return

@register_method
def get_vDarcy(self, Parameters):
  # Compute the Darcy velocity.
  try:
    self.vDarcy = -self.khat*(self.dp_fdX/self.F11)
    if 'uf' in Parameters.Physics:
      self.vDarcy -= self.khat*(self.rhofR*(self.a_f - Parameters.Gravity) + self.DIV_Qf)
      if Parameters.DarcyBrinkman:
        self.vDarcy += self.khat*(self.DIV_FES/(self.nf*self.F11))
    else: 
      # if Parameters.integrationScheme == 'Predictor-corrector':
        #   self.vDarcy = -self.khat*(self.dp_fdX/self.F11 + self.rhofR*(self.a_s - Parameters.Gravity - Parameters.dt*self.dp_fDotdX/self.rho_0))
        # else:
      self.vDarcy -= self.khat*self.rhofR*(self.a_s - Parameters.Gravity)
    if 'tf' in Parameters.Physics:
      self.vDarcy -= self.khat*self.p_f*self.dnfdX*(1 - self.ts/self.tf)/(self.nf*self.F11)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =", self.p_f)
    print("Pressure instability in Darcy velocity; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_F11(self):
  # Compute [1,1] entry of deformation gradient.
  self.F11 = 1. + self.dudX
  return

@register_method
def get_J(self):
  # Compute Jacobian of deformation.
  self.J = 1. + self.dudX
  return

@register_method
def get_JDot(self):
  # Compute time derivative of Jacobian of deformation.
  self.JDot = np.einsum('ik, i', self.Bu, self.v_s_global, dtype=np.float64)
  return

@register_method
def get_Q(self, Parameters):
  # Compute the shock viscosity applied to the single-phase/solid skeleton.
  try:
    if np.any(self.dvdX < 0) and (Parameters.C0 > 0.0 and Parameters.C1 > 0.0):
      #---------------------------------------------------
      # Standard/lumping technique (apply across element).
      #---------------------------------------------------
      if self.Gauss_Order == 2:
        self.Q = self.ns*self.rhosR*Parameters.H0e*self.dvdX*\
                 (Parameters.C0*Parameters.H0e*self.dvdX - Parameters.C1*self.c)
      #-----------------------------------------------
      # Non-standard technique (apply at Gauss point).
      #-----------------------------------------------
      else:
        self.Q             = np.zeros(self.Gauss_Order)
        self.Qidxs         = np.where(self.dvdX < 0)
        self.Q[self.Qidxs] = (self.ns*self.rhosR*Parameters.H0e*self.dvdX*\
                              (Parameters.C0*Parameters.H0e*self.dvdX - Parameters.C1*self.c))[self.Qidxs]
    else:
      self.Q = 0
    
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in bulk viscosity response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return


@register_method
def get_Qf(self, Parameters):
  # Compute the shock viscosity applied to the pore fluid.
  try:
    if np.any(self.dvfdX < 0):
      self.Qf              = np.zeros(self.Gauss_Order)
      self.Qfidxs          = np.where(self.dvfdX < 0)
      self.Qf[self.Qfidxs] = (self.nf*self.rhofR*Parameters.H0e*self.dvfdX*\
                             (Parameters.C0*Parameters.H0e*self.dvfdX/self.J\
                              - Parameters.C1*Parameters.cf))[self.Qfidxs]
    else:
      self.Qf = 0
    
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in bulk viscosity response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.Qf = 0
  return

@register_method
def get_P11(self, Parameters):
  # Compute [1,1] entry of First Piola-Kirchhoff stress tensor.
  #---------------------
  # Classic neo-Hookean.
  #---------------------
  if Parameters.solidModel == 'neo-Hookean':
    try:
      self.P11 = self.F11 * (Parameters.mu + (Parameters.lambd*np.log(self.F11) - Parameters.mu)/(self.F11**2) \
                             + Parameters.nu_0*self.dvdX*self.F11**(-3)*(Parameters.lambd + 2*(Parameters.mu - np.log(self.F11))))
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Deformation < 0 in stress response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
      raise FloatingPointError
  #------------------------------------------------------------
  # Ehlers-Eipper incompressible model, Ehlers & Eipper (1998).
  #------------------------------------------------------------
  elif Parameters.solidModel == 'neo-Hookean-Eipper':
    try:
      self.P11 = self.F11*Parameters.mu*(1 - self.F11**(-2))\
                 + Parameters.lambd*((1 - Parameters.ns_0)**2)*(1/(1 - Parameters.ns_0) - 1/(self.J - Parameters.ns_0))
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Deformation -> 0 in stress response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  #--------------------------------------------------------------------
  # Saint-Venant Kirchhoff (small strain elasticity for finite strain).
  #--------------------------------------------------------------------
  elif Parameters.solidModel == 'Saint-Venant-Kirchhoff':
    self.P11 = self.F11 * ((self.F11**2 - 1)*(Parameters.lambd/2 + Parameters.mu))
  #------------------------
  # Clayton & Freed (2019).
  #------------------------
  elif Parameters.solidModel == 'Clayton-Freed':
    try:
      if 'pf' not in Parameters.Physics:
        self.Gpf = (1 - (Parameters.KS*Parameters.mu_prime/Parameters.mu)*np.log(self.J))
        self.Gpf[self.Gpf < 0] = 0.0

        self.tau0 = Parameters.KSkel*np.log(self.J)*np.exp(Parameters.c1*np.log(self.J)**2)\
                   - (3*Parameters.mu_prime*Parameters.KS/Parameters.c2)\
                     *(np.exp((2*Parameters.c2/9)*np.log(self.J)**2) - 1)\
                     + Parameters.KS*Parameters.B1*np.log(self.J)
        self.tau1 = 2*Parameters.mu*np.exp((2*Parameters.c2/9)*np.log(self.J)**2)\
                     *self.Gpf*np.log(self.J)
      else:
        self.Gpf = (1 - (Parameters.KSkel*Parameters.mu_prime/Parameters.mu)*np.log(self.J))
        self.Gpf[self.Gpf < 0] = 0.0
        
        self.tau0 = Parameters.KSkel*np.log(self.J)*np.exp(Parameters.c1*np.log(self.J)**2)\
                   - (3*Parameters.mu_prime*Parameters.KSkel/Parameters.c2)\
                     *(np.exp((2*Parameters.c2/9)*np.log(self.J)**2) - 1)\
                     + Parameters.KSkel*Parameters.B1*np.log(self.J)
        self.tau1 = 2*Parameters.mu*np.exp((2*Parameters.c2/9)*np.log(self.J)**2)\
                     *self.Gpf*np.log(self.J)
      
      self.tau3  = -self.tau1
      self.tau1 += 2*Parameters.mu*Parameters.B2*np.log(self.J)
      self.tau3 -= 2*Parameters.mu*Parameters.B2*np.log(self.J)
      self.P11   = (self.tau0 + (1/3)*(self.tau1 - self.tau3))/self.J
    except (FloatingPointError, ZeroDivisionError):
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Deformation < 0 in stress response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  #------------------------------------
  # Clayton & Freed (2019), linearized.
  #------------------------------------
  elif Parameters.solidModel == 'Clayton-Freed-Linear':
    try:
      if pf not in Parameters.Physics:
        self.Gpf = (1 - (Parameters.KS*Parameters.mu_prime/Parameters.mu)*np.log(self.J))
        self.Gpf[self.Gpf < 0] = 0.0

        self.tau0 = Parameters.KSkel*np.log(self.J)*np.exp(Parameters.c1*np.log(self.J)**2)\
                    - (2/3)*Parameters.KS*Parameters.mu_prime*np.log(self.J)**2
        self.tau1 = 2*Parameters.mu*np.exp((2*Parameters.c2/9)*np.log(self.J)**2)\
                     *self.Gpf*np.log(self.J)
      else:
        self.Gpf = (1 - (Parameters.KSkel*Parameters.mu_prime/Parameters.mu)*np.log(self.J))
        self.Gpf[self.Gpf < 0] = 0.0

        self.tau0 = Parameters.KSkel*np.log(self.J)*np.exp(Parameters.c1*np.log(self.J)**2)\
                    - (2/3)*Parameters.KSkel*Parameters.mu_prime*np.log(self.J)**2
        self.tau1 = 2*Parameters.mu*np.exp((2*Parameters.c2/9)*np.log(self.J)**2)\
                     *self.Gpf*np.log(self.J)
      
      self.tau3 = -self.tau1
      self.P11  = (self.tau0 + (1/3)*(self.tau1 - self.tau3))/self.J
    except (FloatingPointError, ZeroDivisionError):
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Deformation < 0 in stress response; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  #------------------------
  # Single-phase gas model.
  #------------------------
  elif Parameters.solidModel == 'Ideal-Gas':
    self.P11  = -Parameters.RGas*(self.rho_0/self.J)*self.ts 
    self.P11 -= self.Q 
    return
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid constitutive model not recognized.")
  #-------------------------
  # Linear thermoelasticity.
  #-------------------------
  if Parameters.Physics == 'u-t':
    self.P11 -= (Parameters.As*Parameters.Bb0T*(self.ts - Parameters.Ts_0))/self.F11
  elif 'tf' in Parameters.Physics:
    self.P11 -= (Parameters.As*Parameters.KSkel*(self.ts - Parameters.Ts_0))/self.F11
  
  self.P11 -= self.Q
  return

@register_method
def get_sig11(self, Parameters):
  # Compute Cauchy stresses.
  self.SES = self.P11
  if 'pf' not in Parameters.Physics:
    self.sig11 = self.SES
  else:
    self.sig11 = self.SES - self.p_f + Parameters.p_f0
    if Parameters.DarcyBrinkman:
      self.sig11 += self.FES
  return

@register_method
def get_ps_E(self, Parameters):
  # Compute solid skeleton pressure.
  try:
    #---------------------
    # Classic neo-Hookean.
    #---------------------
    if Parameters.solidModel == 'neo-Hookean': 
      self.sig22 = Parameters.lambd*np.log(self.J)/self.J
    #------------------------------------------------------------
    # Ehlers-Eipper incompressible model, Ehlers & Eipper (1998).
    #------------------------------------------------------------
    elif Parameters.solidModel == 'neo-Hookean-Eipper':
      self.sig22 = Parameters.lambd*((1 - Parameters.ns_0)**2)*self.J*(1/(1 - Parameters.ns_0) - 1/(self.J - Parameters.ns_0))
    #--------------------------------------------------------------------
    # Saint-Venant Kirchhoff (small strain elasticity for finite strain).
    #--------------------------------------------------------------------
    elif Parameters.solidModel == 'Saint-Venant-Kirchhoff':
      self.sig22 = 0
    #------------------------
    # Clayton & Freed (2019).
    #------------------------
    elif 'Clayton-Freed' in Parameters.solidModel:
      self.sig22 = (self.tau0 - (1/3)*self.tau1)/self.J
    #------------------------
    # Single-phase gas model.
    #------------------------
    elif Parameters.solidModel == 'Ideal-Gas':
      self.ps_E = Parameters.RGas*(self.rho_0/self.J)*self.ts
      return
    
    self.sig22 -= self.Q

    self.ps_E = (1/3)*(self.P11 + 2*self.sig22)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in solid pressure; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
  return

@register_method
def get_tau(self, Parameters):
  # Compute solid skeleton shear stress.
  try:
    if Parameters.solidModel == 'Ideal-Gas':
      self.tau = 0
    else:
      self.tau = self.P11 - self.sig22
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in shear; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_ns(self, Parameters):
  # Compute volume fraction of solid skeleton.
  self.ns = Parameters.ns_0/self.J
  return

@register_method
def get_nf(self):
  # Compute volume fraction of pore fluid.
  self.nf = 1 - self.ns
  return

@register_method
def get_rhofR(self, Parameters):
  # Compute real mass density of pore fluid.
  try:
    if Parameters.fluidModel == 'Exponential':
      self.rhofR = Parameters.rhofR_0*np.exp((self.p_f - Parameters.p_f0)/Parameters.KF)
    elif Parameters.fluidModel == 'Linear-Bulk':
      if np.any(np.abs(self.p_f - Parameters.p_f0)) > 0:
        self.rhofR = Parameters.rhofR_0*(self.p_f - Parameters.p_f0)/Parameters.KF
      else:
        self.rhofR = np.ones(Parameters.Gauss_Order)*Parameters.rhofR_0
    elif Parameters.fluidModel == 'Linear-IC':
      self.rhofR = Parameters.rhofR_0*self.p_f/Parameters.p_f0
    elif Parameters.fluidModel == 'Isentropic':
      self.rhofR = Parameters.rhofR_0*(self.p_f/Parameters.p_f0)**(1/1.4)
    elif Parameters.fluidModel == 'Ideal-Gas':
      if 'ts-tf' in Parameters.Physics:
        self.rhofR = self.p_f/(Parameters.RGas*self.tf)
      else:
        try:
          self.rhofR = self.p_f/(Parameters.RGas*Parameters.Tf_0)
        except TypeError:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nInitial pore fluid temperature not set for ideal gas pore fluid consitutive isothermal model.")
    elif Parameters.fluidModel == 'Exponential-Thermal':
      self.rhofR = Parameters.rhofR_0*np.exp((self.p_f - Parameters.p_f0)/Parameters.KF - Parameters.Af*self.tf)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive form for pore fluid real mass density not recognized.")
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =", self.p_f)
    print("Pressure instability in real mass density; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_drhofRdX(self, Parameters):
  # Compute the gradient of the real mass density of the pore fluid.
  try: 
    if Parameters.fluidModel == 'Exponential':
      self.drhofRdX = (self.rhofR/Parameters.KF)*self.dp_fdX
    elif Parameters.fluidModel == 'Ideal-Gas':
      self.drhofRdX = 1/(Parameters.RGas*self.tf)*self.dp_fdX - (self.rhofR/self.tf)*self.dtfdX
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive form for pore fluid real mass density not applicable for computing its gradient.")
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =", self.p_f)
    print("Pressure instability in real mass density gradient; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_rhosR(self, Parameters):
  # Compute real mass density of the solid phase.
  self.rhosR = Parameters.rhosR_0*np.ones(self.Gauss_Order)
  return

@register_method
def get_rho(self, Parameters):
  # Compute total mass density of mixture in Eulerian frame.
  if 'pf' not in Parameters.Physics:
    self.ns     = np.ones(self.Gauss_Order)
    self.get_rhosR(Parameters)
    self.rhos_0 = self.rhosR
    self.rho    = self.rhosR/self.J
  else:
    self.get_ns(Parameters)
    self.get_nf()
    self.get_rhosR(Parameters)
    self.get_rhofR(Parameters)
    self.rho = self.ns*self.rhosR + self.nf*self.rhofR
  return

@register_method
def get_rho_0(self, Parameters):
  # Compute total mass density in Lagrangian frame.
  self.rho_0 = self.rho*self.J
  return

@register_method
def get_rhos_0(self, Parameters):
  # Compute partial mass density of the solid skeleton in Lagrangian frame.
  self.rhos_0 = self.J*self.ns*self.rhosR
  return

@register_method
def get_rhof_0(self):
  # Compute partial mass density of the pore fluid in Lagrangian frame.
  self.rhof_0 = self.J*self.nf*self.rhofR
  return

@register_method
def get_khat(self, Parameters):
  # Compute hyrdraulic conductivity.
  #-----------------------------------------------
  # Classic Kozeny-Carman model, see Cousy (2004).
  #-----------------------------------------------
  if Parameters.khatType == 'Kozeny-Carman':
    self.khat = Parameters.khat_mult*((self.nf**3)/(1 - self.nf**2))
  #-------------------
  # Lai et al. (1981).
  #-------------------
  elif Parameters.khatType == 'Strain-Exponential':
    self.khat = Parameters.khat_mult*np.exp(Parameters.kappa*self.dudX)
  #----------------
  # Markert (2005).
  #----------------
  elif Parameters.khatType == 'Hyperbolic':
    self.khatF = (self.J - Parameters.ns_0)/(1 - Parameters.ns_0)
    self.khat  = Parameters.khat_mult*(np.sign(self.khatF)*(np.abs(self.khatF)**Parameters.kappa))
  elif Parameters.khatType == 'Constant':
    self.khat = Parameters.khat
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive form for hydraulic conductivity not recognized.")
  return

@register_method
def get_dkhat(self, Parameters):
  # Compute Gateuax derivative of functional form for
  # the hydraulic conductivity.
  #-----------------------------------------------
  # Classic Kozeny-Carman model, see Cousy (2004).
  #-----------------------------------------------
  if Parameters.khatType == 'Kozeny-Carman':
    self.dkhat = self.khat*self.ns*(3/self.nf + 2*self.nf/(1 - self.nf**2))/self.J
  #-------------------
  # Lai et al. (1981).
  #-------------------
  elif Parameters.khatType == 'Strain-Exponential':
    self.dkhat = Parameters.kappa*self.khat
  #----------------
  # Markert (2005).
  #----------------
  elif Parameters.khatType == 'Hyperbolic':
    self.dkhat = (Parameters.kappa*Parameters.khat_mult/(1 - Parameters.ns_0))*\
                 np.sign(self.khatF)*(np.abs(self.khatF)**(Parameters.kappa - 1))
  elif Parameters.khatType == 'Constant':
    self.dkhat = np.zeros(self.Gauss_Order)
  return

@register_method
def get_Total_Pressure(self, Parameters):
  # Compute total pressure; p_f0 is used to scale relative to atmosphere.
  return ((self.ps_E - self.p_f) + Parameters.p_f0)

@register_method
def get_Fluid_Pressure(self, Parameters):
  # Compute pore fluid pressure.
  return (-self.p_f + Parameters.p_f0)

@register_method
def get_etas(self, Parameters):
  # Compute solid entropy.
  try:
    self.etas = Parameters.cvs*np.log(self.ts/Parameters.Ts_0)
    if 'pf' not in Parameters.Physics and 'Clayton-Freed' in Parameters.solidModel:
      self.etas += Parameters.As*Parameters.Bb0T*np.log(self.J)
    else:
      self.etas += Parameters.As*Parameters.KSkel*np.log(self.J)
    if 'tf' in Parameters.Physics:
      self.etas -= self.J*Parameters.As*(self.ts/self.tf)*(self.p_f/Parameters.rhosR_0)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in solid entropy; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_etaf(self, Parameters):
  # Compute pore fluid entropy.
  if Parameters.fluidModel == 'Ideal-Gas':
    try:
      self.etaf = Parameters.cvf*np.log(self.tf/Parameters.Tf_0) - Parameters.RGas*np.log(self.rhofR/Parameters.rhofR_0)
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure =", self.p_f)
      print("Pressure instability in pore fluid entropy; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.etaf = Parameters.cvf*np.log(self.tf/Parameters.Tf_0) + Parameters.KF*Parameters.Af/self.rhofR
  return

@register_method
def get_qs(self, Parameters):
  # Compute solid heat flux.
  self.qs = -Parameters.ks*self.dtsdX/self.F11
  if 'pf' in Parameters.Physics:
    self.qs *= self.ns
  return

@register_method
def get_qf(self, Parameters):
  # Compute pore fluid heat flux.
  self.qf = -self.nf*Parameters.kf*self.dtfdX/self.F11
  return

@register_method
def get_c(self, Parameters):
  # Compute longitudinal wave speed.
  try:
    #---------------------------------------------------
    # Compute wave speed traveling through the skeleton.
    #---------------------------------------------------
    if Parameters.solidModel == 'Ideal-Gas':
      #-----------------------------------------------------
      # Wave speed through air as a function of temperature.
      #-----------------------------------------------------
      self.c1 = np.sqrt((Parameters.RGas*1.4)*self.ts)
    else:
      try:
        self.c1 = np.sqrt(Parameters.MSkel/(self.rhos_0/self.J))
      except AttributeError:
        self.get_rhos_0(Parameters)
        self.c1 = np.sqrt(Parameters.MSkel/(self.rhos_0/self.J))
    
    self.c  = self.c1
    #--------------------------------------------------
    # Compute wave speed traveling through the mixture.
    #--------------------------------------------------
    if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
      # If single-phase material, wave speed through skeleton is same as mixture
      if Parameters.Physics == 'u':
        self.c2 = self.c1
      else:
        # Numerator is calculation of mixture bulk modulus (ad-hoc?)
        self.c2 = np.sqrt(((self.ns*Parameters.KS + self.nf*Parameters.KF) + (4/3)*Parameters.mu)/self.rho)
      #----------------------------------------------------
      # Use the larger wave speed in time step calculation.
      #----------------------------------------------------
      if np.any(np.max(self.c1) > np.max(self.c2)):
        self.cdt = self.c1
      else:
        self.cdt = self.c2
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in calculation of element wave speed; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_dt(self, Parameters):
  # The CFL condition that restricts the maximum time step over the element.
  # Used in central-difference integration scheme.
  try:
    self.dt = Parameters.SF*(Parameters.H0e/self.J)/self.cdt
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Deformation < 0 in calculation of element time-step; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return
