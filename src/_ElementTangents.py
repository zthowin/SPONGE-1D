#----------------------------------------------------------------------------------------
# Module housing element object tangents.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 25, 2024
#----------------------------------------------------------------------------------------
import sys, warnings

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

#-------------------------
# Begin top-level methods.
#-------------------------
@register_method
def compute_tangents(self, Parameters, VariationalEq=None):
  # Top level function to compute element tangents for a given
  # variational equation at current time-step. 
  if VariationalEq == 'G':
    self.compute_G_Tangents(Parameters)
  elif VariationalEq == 'H':
    self.compute_H_Tangents(Parameters)
  elif VariationalEq == 'I':
    self.compute_I_Tangents(Parameters)
  elif VariationalEq == 'J':
    self.compute_J_Tangents(Parameters)
  elif VariationalEq == 'K':
    self.compute_K_Tangents(Parameters)
  else:
    sys.exit("------\nERROR:\n------\nVariational equation not recognized, check source code.")
  return

@register_method
def compute_G_Tangents(self, Parameters):
  # Top level function to compute element tangents used
  # in the balance of momentum of the solid skeleton.
  self.G_Mtx = np.zeros((Parameters.ndofSe, self.numDOF))
  if Parameters.Physics == 'u':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal' or Parameters.integrationScheme == 'Quasi-static':
      self.get_G_uu_Implicit(Parameters)
    else:
      self.get_G_uu_Explicit(Parameters)
    self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
  elif Parameters.Physics == 'u-pf':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_G_uu_Implicit(Parameters)
      self.get_G_up(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
      self.G_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.G_up_Mtx
    else:
      self.get_G_uu_Explicit(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
  elif Parameters.Physics == 'u-uf-pf':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_G_uu_Implicit(Parameters)
      self.get_G_up(Parameters)
      self.get_G_uuf(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
      self.G_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)] += self.G_uuf_Mtx
      self.G_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] += self.G_up_Mtx
    else:
      self.get_G_uu_Explicit(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
  elif 't' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_G_uu_Implicit(Parameters)
      self.get_G_uts(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
      if 'tf' in Parameters.Physics:
        self.get_G_up(Parameters)
        self.get_G_utf(Parameters) 
        self.G_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.G_up_Mtx
        self.G_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse)] += self.G_uts_Mtx
        self.G_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse):] += self.G_utf_Mtx
      else:
        self.G_Mtx[:,Parameters.ndofSe:] += self.G_uts_Mtx
    elif 'RK' in Parameters.integrationScheme:
      self.get_G_uu_Explicit(Parameters)
      self.G_Mtx[:,0:Parameters.ndofSe] += self.G_uu_Mtx
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime integration scheme for thermo(poro)elastodynamics must be Newmark-beta/Trapezoidal or a variation of Runge-Kutta.")
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPhysics for problem not recognized.")
  return

@register_method
def compute_H_Tangents(self, Parameters):
  # Top level function to compute element tangents used
  # in the balance of mass of the mixture.
  self.H_Mtx = np.zeros((Parameters.ndofPe, self.numDOF))
  if Parameters.Physics == 'u-pf':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_H_pp_Implicit(Parameters)
      self.get_H_pu(Parameters)
      self.H_Mtx[:,0:Parameters.ndofSe] += self.H_pu_Mtx
      self.H_Mtx[:,Parameters.ndofSe:Parameters.ndofSe + Parameters.ndofPe] += self.H_pp_Mtx
    # elif Parameters.integrationScheme == 'Predictor-corrector':
    #   self.get_H_pp_Implicit(Parameters)
    #   self.H_Mtx[:,Parameters.GaussD:Parameters.GaussD + Parameters.GaussP] += self.H_pp_Mtx
    else:
      self.get_H_pp_Explicit(Parameters)
      if Parameters.integrationScheme == 'Central-difference':
        self.get_H_pu(Parameters)
        self.H_int += np.einsum('ik,k', self.H_pu_Mtx, self.a_s_global)
      self.H_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.H_pp_Mtx
  elif Parameters.Physics == 'u-uf-pf':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_H_pp_Implicit(Parameters)
      self.get_H_pu(Parameters)
      self.get_H_puf(Parameters)
      self.H_Mtx[:,0:Parameters.ndofSe] += self.H_pu_Mtx
      self.H_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)] += self.H_puf_Mtx
      self.H_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] += self.H_pp_Mtx
    else:
      self.get_H_pp_Explicit(Parameters)
      self.H_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] += self.H_pp_Mtx
  elif 'tf' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.get_H_pp_Implicit(Parameters)
      self.get_H_pu(Parameters)
      self.get_H_pts(Parameters)
      self.get_H_ptf(Parameters)
      self.H_Mtx[:,0:Parameters.ndofSe] += self.H_pu_Mtx
      self.H_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.H_pp_Mtx
      self.H_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse)] += self.H_pts_Mtx
      self.H_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse):] += self.H_ptf_Mtx
    elif 'RK' in Parameters.integrationScheme:
      self.get_H_pp_Explicit(Parameters)
      self.H_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] += self.H_pp_Mtx
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime integration scheme for thermoporoelastodynamics must be Newmark-beta/Trapezoidal or a variation of Runge-Kutta.")
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPhysics for problem not recognized.")
  return

@register_method
def compute_I_Tangents(self, Parameters):
  # Top level function to compute element tangents used
  # in the balance of momentum of the pore fluid.
  self.I_Mtx = np.zeros((Parameters.ndofFe, self.numDOF))
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    self.get_I_ufu(Parameters)
    self.get_I_ufp(Parameters)
    self.get_I_ufuf_Implicit(Parameters)
    self.I_Mtx[:,0:Parameters.ndofSe] += self.I_ufu_Mtx
    self.I_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)] += self.I_ufuf_Mtx
    self.I_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] += self.I_ufp_Mtx
  else:
    self.get_I_ufuf_Explicit(Parameters)
    self.I_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)] += self.I_ufuf_Mtx
  return

@register_method
def compute_J_Tangents(self, Parameters):
  # Top level function to compute element tangents used
  # in the balance of energy of the solid.
  self.J_Mtx = np.zeros((Parameters.ndofTse, self.numDOF))
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    self.get_J_tsts_Implicit(Parameters)
    self.get_J_tsu(Parameters)
    self.J_Mtx[:,0:Parameters.ndofSe] += self.J_tsu_Mtx
    if 'pf' in Parameters.Physics:
      self.get_J_tsp(Parameters)
      self.get_J_tstf(Parameters)
      self.J_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.J_tsp_Mtx
      self.J_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse)] += self.J_tsts_Mtx
      self.J_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse):] += self.J_tstf_Mtx
    else:
      self.J_Mtx[:,Parameters.ndofSe:] += self.J_tsts_Mtx

  elif 'RK' in Parameters.integrationScheme:
    self.get_J_tsts_Explicit(Parameters)
    if 'pf' in Parameters.Physics:
      if 'uf' in Parameters.Physics:
        self.J_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse)] += self.J_tsts_Mtx
      else:
        self.J_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse)] += self.J_tsts_Mtx
    else:
      self.J_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe+ Parameters.ndofTse)] += self.J_tsts_Mtx

  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime integration scheme for thermo(poro)elastodynamics must be Newmark-beta/Trapezoidal or a variation of Runge-Kutta.")

  return

@register_method
def compute_K_Tangents(self, Parameters):
  # Top level function to compute element tangents used
  # in the balance of energy of the pore fluid.
  self.K_Mtx = np.zeros((Parameters.ndofTfe, self.numDOF))

  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    self.get_K_tftf_Implicit(Parameters)
    self.get_K_tfu(Parameters)
    self.K_Mtx[:,0:Parameters.ndofSe] += self.K_tfu_Mtx
    self.get_K_tfp(Parameters)
    self.get_K_tfts(Parameters)
    self.K_Mtx[:,Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] += self.K_tfp_Mtx
    self.K_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse)] += self.K_tfts_Mtx
    self.K_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse):] += self.K_tftf_Mtx

  elif 'RK' in Parameters.integrationScheme:
    self.get_K_tftf_Explicit(Parameters)
    if 'uf' in Parameters.Physics:
      self.K_Mtx[:,(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTfe):] += self.K_tftf_Mtx
    else:
      self.K_Mtx[:,(Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTfe):] += self.K_tftf_Mtx

  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime integration scheme for thermoporoelastodynamics must be Newmark-beta/Trapezoidal or a variation of Runge-Kutta.")

  return
#-----------------------
# End top-level methods.
#-----------------------
#-------------------------
# Begin mid-level methods.
#-------------------------
@register_method
def get_G_uu_Explicit(self, Parameters):
  # Sum the components of the mass matrix for the variational
  # equation of the balance of momentum of the solid assuming an
  # explicit numerical integration scheme for the time 
  # discretization.
  self.G_uu_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofSe))
  #--------------------------------------------------------
  # Only when a_s != a_f do we use the solid mass density.
  # Otherwise, the mixture density is the more appropriate.
  #--------------------------------------------------------
  if 'uf' in Parameters.Physics:
    self.lumpedRho = self.rhos_0
  else:
    self.lumpedRho = self.rho_0
    
  try:
    if Parameters.SolidLumping:
      if Parameters.ndofSe == 3 and self.Gauss_Order == 3:
        self.G_uu_Mtx[0,0] =   Parameters.Area*self.Jacobian*self.lumpedRho[0]
        self.G_uu_Mtx[1,1] = 4*Parameters.Area*self.Jacobian*self.lumpedRho[1]
        self.G_uu_Mtx[2,2] =   Parameters.Area*self.Jacobian*self.lumpedRho[2]
        self.G_uu_Mtx     /= 3
      elif Parameters.ndofSe == 2:
        if self.Gauss_Order == 2:
          self.G_uu_Mtx[0,0] = Parameters.Area*self.Jacobian*self.lumpedRho[0]
          self.G_uu_Mtx[1,1] = Parameters.Area*self.Jacobian*self.lumpedRho[1]
        elif self.Gauss_Order == 1:
          warnings.simplefilter('once', RuntimeWarning("WARNING. Using reduced integration can lead to inaccurate results for solid mass lumping."))
          self.G_uu_Mtx = Parameters.Area*self.Jacobian*self.lumpedRho*np.identity(2)
        else:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nInterpolation and quadrature order are not consistent for solid mass lumping.\nQuadratic interpolation requires 3-point quadrature; linear interpolation requires 2-point quadrature.")
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nInterpolation and quadrature order are not consistent for solid mass lumping.\nQuadratic interpolation requires 3-point quadrature; linear interpolation requires 2-point quadrature.")
    else:
      self.get_M_uu_G1(Parameters)
      self.G_uu_Mtx += self.M_uu_G1
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_uu.")
    raise FloatingPointError
  return

@register_method
def get_G_uu_Implicit(self, Parameters):
  # Sum the components of the solid displacement-solid displacement
  # stiffness matrices for the variational equation of the balance of
  # momentum of the mixture/solid assuming an implicit numerical
  # integration scheme for the time discretization.
  self.G_uu_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofSe))
  try:
    if Parameters.isDynamics:
      self.get_M_uu_G1(Parameters)
      self.G_uu_Mtx += self.M_uu_G1
    self.get_K_uu_G2(Parameters)
    self.G_uu_Mtx += self.K_uu_G2
    if Parameters.DarcyBrinkman:
      self.get_K_uu_G5(Parameters)
      self.G_uu_Mtx += self.K_uu_G5
    if 'tf' in Parameters.Physics:
      self.get_K_uu_G3(Parameters)
      self.G_uu_Mtx += self.K_uu_G3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_uu")
    raise FloatingPointError
  return

@register_method
def get_G_up(self, Parameters):
  # Sum the components of the solid displacement-pore fluid pressure
  # stiffness matrices for the variational equation of the balance of
  # momentum of the mixture assuming an implicit numerical
  # integration scheme for the time discretization.
  self.G_up_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofPe))
  try:
    if Parameters.Physics == 'u-uf-pf':
      self.get_K_up_G1(Parameters)
      self.G_up_Mtx += self.K_up_G1
    self.get_K_up_G3(Parameters)
    self.G_up_Mtx += self.K_up_G3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_up")
    raise FloatingPointError
  return

@register_method
def get_G_uuf(self, Parameters):
  # Sum the components of the solid displacement-pore fluid displacement
  # stiffness matrices for the variational equation of the balance of
  # momentum of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.G_uuf_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofFe))
  self.get_K_uuf_G1(Parameters)
  try:
    self.G_uuf_Mtx += self.K_uuf_G1
    if Parameters.DarcyBrinkman:
      self.get_K_uuf_G5(Parameters)
      self.G_uuf_Mtx += self.K_uuf_G5
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_uuf")
    raise FloatingPointError
  return

@register_method
def get_G_uts(self, Parameters):
  # Sum the components of the solid displacement-solid temperature
  # stiffness matrices for the variational equation of the balance of
  # momentum of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.G_uts_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofTse))
  try:
    self.get_K_uts_G2(Parameters)
    self.G_uts_Mtx += self.K_uts_G2
    if 'pf' in Parameters.Physics:
      self.get_K_uts_G3(Parameters)
      self.G_uts_Mtx += self.K_uts_G3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_uts")
    raise FloatingPointError
  return

@register_method
def get_G_utf(self, Parameters):
  # Sum the components of the solid displacement-pore fluid temperature
  # stiffness matrices for the variational equation of the balance of
  # momentum of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.G_utf_Mtx = np.zeros((Parameters.ndofSe, Parameters.ndofTfe))
  try:
    self.get_K_utf_G3(Parameters)
    self.G_utf_Mtx += self.K_utf_G3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of G_utf")
    raise FloatingPointError
  return

@register_method
def get_H_pp_Explicit(self, Parameters):
  # Sum the components of the mass matrix for the variational
  # equation of the balance of mass of the mixture assuming an
  # explicit numerical integration scheme for the time 
  # discretization.
  self.H_pp_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofPe))
  try:
    if Parameters.PressureLumping:
      if Parameters.alpha_stab > 0:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPressure stabilization not compatible with mass-lumping.")
      if Parameters.fluidModel == 'Ideal-Gas':
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nIdeal gas model not compatible with mass-lumping.")
      if Parameters.Gauss_Order == 2:
        if 'Central-difference' in Parameters.integrationScheme:
          self.H_pp_Mtx[0,0]  = (Parameters.Area*self.Jacobian*self.J[0]*self.nf[0]*Parameters.dt/(2*Parameters.KF))
          self.H_pp_Mtx[1,1]  = (Parameters.Area*self.Jacobian*self.J[1]*self.nf[1]*Parameters.dt/(2*Parameters.KF))
        else:
          self.H_pp_Mtx[0,0]  = (Parameters.Area*self.Jacobian*self.J[0]*self.nf[0]/Parameters.KF)
          self.H_pp_Mtx[1,1]  = (Parameters.Area*self.Jacobian*self.J[1]*self.nf[1]/Parameters.KF)
      elif Parameters.Gauss_Order == 1:
        if 'Central-difference' in Parameters.integrationScheme:
          self.H_pp_Mtx = (Parameters.Area*self.Jacobian*self.J*self.nf*Parameters.dt*np.identity(2)/(2*Parameters.KF))
        else:
          self.H_pp_Mtx = (Parameters.Area*self.Jacobian*self.J*self.nf*np.identity(2)/Parameters.KF)
    else:
      self.get_M_pp_H1(Parameters)
      self.H_pp_Mtx += self.M_pp_H1
      if Parameters.alpha_stab > 0:
        self.get_M_pp_HStab(Parameters)
        self.H_pp_Mtx += self.M_pp_HStab
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_pp")
    raise FloatingPointError
  return

@register_method
def get_H_pp_Implicit(self, Parameters):
  # Sum the components of the pore fluid pressure-pore fluid pressure
  # stiffness matrices for the variational equation of the balance of
  # mass of the mixture assuming an implicit numerical
  # integration scheme for the time discretization.
  self.H_pp_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofPe))
  self.get_M_pp_H1(Parameters)
  self.get_K_pp_H2(Parameters)
  self.get_K_pp_H3(Parameters)
  self.get_K_pp_H4(Parameters)
  try:
    self.H_pp_Mtx += self.M_pp_H1 + self.K_pp_H2 + self.K_pp_H3 + self.K_pp_H4
    if Parameters.alpha_stab > 0:
      self.get_M_pp_HStab(Parameters)
      self.H_pp_Mtx += self.M_pp_HStab
    if 'tf' in Parameters.Physics:
      self.get_K_pp_H6(Parameters)
      self.get_K_pp_H7(Parameters)
      self.H_pp_Mtx += self.K_pp_H6 + self.K_pp_H7
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_pp.")
    raise FloatingPointError
  return

@register_method
def get_H_pu(self, Parameters):
  # Sum the components of the pore fluid pressure-solid displacement
  # stiffness matrices for the variational equation of the balance of
  # mass of the mixture assuming an implicit numerical
  # integration scheme for the time discretization.
  self.H_pu_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofSe))
  #------------------------------------------------------------------------------
  # Compute dkhat again for central-difference scheme since it relies on a_{n+1}.
  #------------------------------------------------------------------------------
  self.get_dkhat(Parameters)
  self.get_K_pu_H1(Parameters)
  self.get_K_pu_H2(Parameters)
  if Parameters.integrationScheme == 'Central-difference':
    self.K_pu_H3 = np.zeros((Parameters.ndofPe, Parameters.ndofSe))
  else:
    self.get_K_pu_H3(Parameters)
  self.get_K_pu_H4(Parameters)
  try:
    self.H_pu_Mtx += self.K_pu_H1 + self.K_pu_H2 + self.K_pu_H3 + self.K_pu_H4
    if Parameters.alpha_stab > 0:
      self.get_K_pu_HStab(Parameters)
      self.H_pu_Mtx += self.K_pu_HStab
    if Parameters.DarcyBrinkman:
      self.get_K_pu_H5(Parameters)
      self.H_pu_Mtx += self.K_pu_H5
    if 'tf' in Parameters.Physics:
      self.get_K_pu_H6(Parameters)
      self.get_K_pu_H7(Parameters)
      self.H_pu_Mtx += self.K_pu_H6 + self.K_pu_H7
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_pu.")
    raise FloatingPointError
  return

@register_method
def get_H_puf(self, Parameters):
  # Sum the components of the pore fluid pressure-pore displacement
  # stiffness matrices for the variational equation of the balance of
  # mass of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.H_puf_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofFe))
  self.get_K_puf_H2(Parameters)
  self.get_K_puf_H4(Parameters)
  try:
    self.H_puf_Mtx += self.K_puf_H2 + self.K_puf_H4
    if Parameters.DarcyBrinkman:
      self.get_K_puf_H5(Parameters)
      self.H_puf_Mtx += self.K_puf_H5
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_puf.")
    raise FloatingPointError
  return

@register_method
def get_H_pts(self, Parameters):
  # Sum the components of the pore fluid pressure-solid temperature
  # stiffness matrices for the variational equation of the balance of
  # mass of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.H_pts_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofTse))
  self.get_K_pts_H2(Parameters)
  self.get_K_pts_H6(Parameters)
  self.get_K_pts_H7(Parameters)
  try:
    self.H_pts_Mtx += self.K_pts_H2 + self.K_pts_H6 + self.K_pts_H7
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_pts.")
    raise FloatingPointError
  return

@register_method
def get_H_ptf(self, Parameters):
  # Sum the components of the pore fluid pressure-pore fluid temperature
  # stiffness matrices for the variational equation of the balance of
  # mass of the mixture assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.H_ptf_Mtx = np.zeros((Parameters.ndofPe, Parameters.ndofTfe))
  self.get_K_ptf_H2(Parameters)
  self.get_K_ptf_H6(Parameters)
  self.get_K_ptf_H7(Parameters)
  try:
    self.H_ptf_Mtx += self.K_ptf_H2 + self.K_ptf_H6 + self.K_ptf_H7
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of H_ptf.")
    raise FloatingPointError
  return

@register_method
def get_I_ufuf_Explicit(self, Parameters):
  # Sum the components of the mass matrix for the variational
  # equation of the balance of momentum of the fluid assuming an
  # explicit numerical integration scheme for the time 
  # discretization.
  self.I_ufuf_Mtx = np.zeros((Parameters.ndofFe, Parameters.ndofFe))
  if Parameters.FluidLumping:
    if Parameters.ndofFe == 3 and self.Gauss_Order == 3:
      self.I_ufuf_Mtx[0,0] =   Parameters.Area*self.Jacobian*self.rhof_0[0]
      self.I_ufuf_Mtx[1,1] = 4*Parameters.Area*self.Jacobian*self.rhof_0[1]
      self.I_ufuf_Mtx[2,2] =   Parameters.Area*self.Jacobian*self.rhof_0[2]
      self.I_ufuf_Mtx     /= 3
    elif Parameters.ndofFe == 2:
      if self.Gauss_Order == 2:
        self.I_ufuf_Mtx[0,0] = Parameters.Area*self.Jacobian*self.rhof_0[0]
        self.I_ufuf_Mtx[1,1] = Parameters.Area*self.Jacobian*self.rhof_0[1]
      elif self.Gauss_Order == 1:
        self.I_ufuf_Mtx = Parameters.Area*self.Jacobian*self.rhof_0*np.identity(2)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nInterpolation and quadrature order are not consistent for fluid mass lumping.\nQuadratic interpolation requires 3-point quadrature; linear interpolation requires 2-point quadrature.")
  else:
    self.get_M_ufuf_I1(Parameters)
    try:
      self.I_ufuf_Mtx += self.M_ufuf_I1
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of I_ufuf.")
      raise FloatingPointError
  return

@register_method
def get_I_ufuf_Implicit(self, Parameters):
  # Sum the components of the pore fluid displacement-pore fluid displacement
  # stiffness matrices for the variational equation of the balance of
  # momentum of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.I_ufuf_Mtx = np.zeros((Parameters.ndofFe, Parameters.ndofFe))
  self.get_M_ufuf_I1(Parameters)
  self.get_K_ufuf_I3(Parameters)
  try:
    self.I_ufuf_Mtx += self.M_ufuf_I1 + self.K_ufuf_I3
    if Parameters.DarcyBrinkman:
      self.get_K_ufuf_I5(Parameters)
      self.I_ufuf_Mtx += self.K_ufuf_I5
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of I_ufuf.")
    raise FloatingPointError
  return

@register_method
def get_I_ufu(self, Parameters):
  # Sum the components of the pore fluid displacement-solid displacement
  # stiffness matrices for the variational equation of the balance of
  # momentum of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.I_ufu_Mtx = np.zeros((Parameters.ndofFe, Parameters.ndofSe))
  self.get_K_ufu_I1(Parameters)
  self.get_K_ufu_I2(Parameters)
  self.get_K_ufu_I3(Parameters)
  self.get_K_ufu_I4(Parameters)
  try:
    self.I_ufu_Mtx += self.K_ufu_I1 + self.K_ufu_I2 + self.K_ufu_I3 + self.K_ufu_I4
    if Parameters.DarcyBrinkman:
      self.get_K_ufu_I5(Parameters)
      self.I_ufu_Mtx += self.K_ufu_I5
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of I_ufu.")
    raise FloatingPointError
  return

@register_method
def get_I_ufp(self, Parameters):
  # Sum the components of the pore fluid displacement-pore fluid pressure
  # stiffness matrices for the variational equation of the balance of
  # momentum of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.I_ufp_Mtx = np.zeros((Parameters.ndofFe, Parameters.ndofPe))
  self.get_K_ufp_I1(Parameters)
  self.get_K_ufp_I2(Parameters)
  self.get_K_ufp_I4(Parameters)
  try:
    self.I_ufp_Mtx += self.K_ufp_I1 + self.K_ufp_I2 + self.K_ufp_I4
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of I_ufp.")
    raise FloatingPointError
  return

@register_method
def get_J_tsts_Explicit(self, Parameters):
  # Sum the components of the mass matrix for the variational
  # equation of the balance of energy of the solid assuming an
  # explicit numerical integration scheme for the time 
  # discretization.
  self.J_tsts_Mtx = np.zeros((Parameters.ndofTse, Parameters.ndofTse))
  if Parameters.SolidTempLumping:
    if self.Gauss_Order == 2:
      self.J_tsts_Mtx[0,0] = (Parameters.Area*self.Jacobian*self.rhos_0[0]*Parameters.cvs)
      self.J_tsts_Mtx[1,1] = (Parameters.Area*self.Jacobian*self.rhos_0[1]*Parameters.cvs)
    elif self.Gauss_Order == 1:
      self.J_tsts_Mtx = (Parameters.Area*self.Jacobian*self.rhos_0*Parameters.cvs*np.identity(2))
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nInterpolation and quadrature order are not consistent for solid temperature mass lumping.\nQuadratic interpolation requires 3-point quadrature; linear interpolation requires 2-point quadrature.")
  else:
    self.get_M_tsts_J1(Parameters)
    try:
      self.J_tsts_Mtx += self.M_tsts_J1
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsts.")
      raise FloatingPointError
  return

@register_method
def get_J_tsts_Implicit(self, Parameters):
  # Sum the components of the mass and stiffness matrix(es) for the
  # variational equation of the balance of energy of the solid
  # assuming an implicit numerical integration scheme for the 
  # time discretization.
  self.J_tsts_Mtx = np.zeros((Parameters.ndofTse, Parameters.ndofTse))
  self.get_M_tsts_J1(Parameters)
  self.get_K_tsts_J2(Parameters)
  self.get_K_tsts_J3(Parameters)
  try:
    self.J_tsts_Mtx += self.M_tsts_J1 + self.K_tsts_J2 + self.K_tsts_J3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsts.")
    raise FloatingPointError
  if 'pf' in Parameters.Physics:
    self.get_K_tsts_J4(Parameters)
    self.get_K_tsts_J5(Parameters)
    self.get_K_tsts_J6(Parameters)
    try:
      self.J_tsts_Mtx += self.K_tsts_J4 + self.K_tsts_J5 + self.K_tsts_J6
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsts (thermoporoelastodynamics).")
      raise FloatingPointError
  return

@register_method
def get_J_tsu(self, Parameters):
  # Sum the components of the solid temperature-solid displacement
  # stiffness matrices for the variational equation of the balance of
  # energy of the solid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.J_tsu_Mtx = np.zeros((Parameters.ndofTse, Parameters.ndofSe))
  self.get_K_tsu_J1(Parameters)
  self.get_K_tsu_J2(Parameters)
  self.get_K_tsu_J3(Parameters)
  try:
    self.J_tsu_Mtx += self.K_tsu_J1 + self.K_tsu_J2 + self.K_tsu_J3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsu.")
    raise FloatingPointError
  if 'pf' in Parameters.Physics:
    self.get_K_tsu_J4(Parameters)
    self.get_K_tsu_J5(Parameters)
    self.get_K_tsu_J6(Parameters)
    try: 
      self.J_tsu_Mtx += self.K_tsu_J4 + self.K_tsu_J5 + self.K_tsu_J6
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsu (thermoporoelastodynamics).")
      raise FloatingPointError
  return

@register_method
def get_J_tsp(self, Parameters):
  # Sum the components of the solid temperature-pore fluid pressure
  # stiffness matrices for the variational equation of the balance of
  # energy of the solid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.J_tsp_Mtx = np.zeros((Parameters.ndofTse, Parameters.ndofPe))
  self.get_K_tsp_J2(Parameters)
  self.get_K_tsp_J5(Parameters)
  self.get_K_tsp_J6(Parameters)
  try:
    self.J_tsp_Mtx += self.K_tsp_J2 + self.K_tsp_J5 + self.K_tsp_J6
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tsp.")
    raise FloatingPointError
  return

@register_method
def get_J_tstf(self, Parameters):
  # Sum the components of the solid temperature-pore fluid temperature
  # stiffness matrices for the variational equation of the balance of
  # energy of the solid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.J_tstf_Mtx = np.zeros((Parameters.ndofTse, Parameters.ndofTfe))
  self.get_K_tstf_J2(Parameters)
  self.get_K_tstf_J4(Parameters)
  self.get_K_tstf_J5(Parameters)
  self.get_K_tstf_J6(Parameters)
  try:
    self.J_tstf_Mtx += self.K_tstf_J2 + self.K_tstf_J4 + self.K_tstf_J5 + self.K_tstf_J6
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of J_tstf.")
    raise FloatingPointError
  return

@register_method
def get_K_tftf_Explicit(self, Parameters):
  # Sum the components of the mass matrix for the variational
  # equation of the balance of energy of the pore fluid assuming an
  # explicit numerical integration scheme for the time 
  # discretization.
  self.K_tftf_Mtx = np.zeros((Parameters.ndofTfe, Parameters.ndofTfe))
  if Parameters.FluidTempLumping:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid temperature lumping is not possible.\nThese physics require resolving the porosity gradient, which uses 4-pt quadrature.\n4-pt quadrature lumping techniques haven not been implemented.")
  else:
    self.get_M_tftf_K1(Parameters)
    try:
      self.K_tftf_Mtx += self.M_tftf_K1
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of K_tftf.")
      raise FloatingPointError
  return

@register_method
def get_K_tftf_Implicit(self, Parameters):
  # Sum the components of the mass and stiffness matrix(es) for the
  # variational equation of the balance of energy of the pore fluid
  # assuming an implicit numerical integration scheme for the 
  # time discretization.
  self.K_tftf_Mtx = np.zeros((Parameters.ndofTfe, Parameters.ndofTfe))
  self.get_M_tftf_K1(Parameters)
  self.get_K_tftf_K2(Parameters)
  self.get_K_tftf_K4(Parameters)
  self.get_K_tftf_K6(Parameters)
  self.get_K_tftf_K7(Parameters)
  self.get_K_tftf_K8(Parameters)
  try:
    self.K_tftf_Mtx += self.M_tftf_K1 + self.K_tftf_K2 + self.K_tftf_K4 + self.K_tftf_K6 + \
                       self.K_tftf_K7 + self.K_tftf_K8
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of K_tftf.")
    raise FloatingPointError
  return

@register_method
def get_K_tfu(self, Parameters):
  # Sum the components of the pore fluid temperature-solid displacement
  # stiffness matrices for the variational equation of the balance of
  # energy of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.K_tfu_Mtx = np.zeros((Parameters.ndofTfe, Parameters.ndofSe))
  self.get_K_tfu_K1(Parameters)
  self.get_K_tfu_K2(Parameters)
  self.get_K_tfu_K3(Parameters)
  self.get_K_tfu_K4(Parameters)
  self.get_K_tfu_K5(Parameters)
  self.get_K_tfu_K6(Parameters)
  self.get_K_tfu_K7(Parameters)
  self.get_K_tfu_K8(Parameters)
  try:
    self.K_tfu_Mtx += self.K_tfu_K1 + self.K_tfu_K2 + self.K_tfu_K3 + self.K_tfu_K4 +\
                      self.K_tfu_K5 + self.K_tfu_K6 + self.K_tfu_K7 + self.K_tfu_K8
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of K_tfu.")
    raise FloatingPointError
  return

@register_method
def get_K_tfp(self, Parameters):
  # Sum the components of the pore fluid temperature-pore fluid pressure
  # stiffness matrices for the variational equation of the balance of
  # energy of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.K_tfp_Mtx = np.zeros((Parameters.ndofTfe, Parameters.ndofPe))
  self.get_K_tfp_K1(Parameters)
  self.get_K_tfp_K2(Parameters)
  self.get_K_tfp_K3(Parameters)
  self.get_K_tfp_K4(Parameters)
  self.get_K_tfp_K5(Parameters)
  self.get_K_tfp_K6(Parameters)
  try:
    self.K_tfp_Mtx += self.K_tfp_K1 + self.K_tfp_K2 + self.K_tfp_K3 + self.K_tfp_K4 +\
                      self.K_tfp_K5 + self.K_tfp_K6
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of K_tfp.")
    raise FloatingPointError
  return

@register_method
def get_K_tfts(self, Parameters):
  # Sum the components of the pore fluid temperature-solid temperature
  # stiffness matrices for the variational equation of the balance of
  # energy of the pore fluid assuming an implicit Newmark-beta numerical
  # integration scheme for the time discretization.
  self.K_tfts_Mtx = np.zeros((Parameters.ndofTfe, Parameters.ndofTse))
  self.get_K_tfts_K2(Parameters)
  self.get_K_tfts_K4(Parameters)
  self.get_K_tfts_K6(Parameters)
  self.get_K_tfts_K8(Parameters)
  try:
    self.K_tfts_Mtx += self.K_tfts_K2 + self.K_tfts_K4 + self.K_tfts_K6 + self.K_tfts_K8
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow in calculation of K_tfts.")
    raise FloatingPointError
  return
#-----------------------
# End mid-level methods.
#-----------------------
#-------------------------
# Begin low-level methods.
#-------------------------
#---------------------
# Begin mass matrices.
#---------------------
@register_method
def get_M_uu_G1(self, Parameters):
  # Compute M_uu_G1.
  if 'uf' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta':
      Nu_Factor    = self.rhos_0
      try:
        Bu_Factor  = self.a_f*self.rhofR*Parameters.beta*(Parameters.dt**2)
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\n")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in M_uu_G1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
      self.M_uu_G1 = np.einsum('ik, jk, k', self.Nu, self.Nu*Nu_Factor + self.Bu*Bu_Factor, self.weights)
    else:
      self.M_uu_G1 = np.einsum('ik, jk, k', self.Nu, self.Nu, self.rhos_0*self.weights)
  else:
    self.M_uu_G1 = np.einsum('ik, jk, k', self.Nu, self.Nu, self.rho_0*self.weights)

  self.M_uu_G1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_M_pp_H1(self, Parameters):
  # Compute M_pp_H1.
  self.M_pp_H1 = np.einsum('ik, jk, k', self.Np, self.Np, self.J*self.nf*self.weights/self.HDiv)

  if Parameters.integrationScheme == 'Newmark-beta':
    if Parameters.isDynamics:
      self.M_pp_H1 *= Parameters.gamma*Parameters.dt
      if Parameters.fluidModel == 'Ideal-Gas':
        self.M_pp_H1 -= np.einsum('ik, jk, k', self.Np, self.Np, self.p_fDot*self.weights*\
                                  self.J*self.nf*Parameters.beta*(Parameters.dt**2)/\
                                  (self.p_f**2))
      elif Parameters.fluidModel == 'Isentropic':
        self.M_pp_H1 -= np.einsum('ik, jk, k', self.Np, self.Np, self.J*self.nf*self.p_fDot*\
                                  self.weights*Parameters.beta*(Parameters.dt**2)/\
                                  (self.p_f*self.HDiv))
    else:
      if Parameters.fluidModel == 'Ideal-Gas':
        self.M_pp_H1 -= np.einsum('ik, jk, k', self.Np, self.Np, self.p_fDot*self.weights*\
                                  self.J*self.nf*Parameters.gamma*Parameters.dt/(self.p_f**2))
      elif Parameters.fluidModel == 'Isentropic':
        self.M_pp_H1 -= np.einsum('ik, jk, k', self.Np, self.Np, self.J*self.nf*self.p_fDot*\
                                  self.weights*Parameters.gamma*Parameters.dt/\
                                  (self.p_f*self.HDiv))
  elif Parameters.integrationScheme == 'Central-difference':
    self.M_pp_H1 *= Parameters.dt/2

  self.M_pp_H1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_M_pp_HStab(self, Parameters):
  # Compute M_pp_HStab.
  self.M_pp_HStab = np.einsum('ik, jk, k', self.Bp, self.Bp, Parameters.alpha_stab*self.weights/(self.F11))
  
  if Parameters.integrationScheme == 'Newmark-beta':
    self.M_pp_HStab *= Parameters.gamma*Parameters.dt
  elif Parameters.integrationScheme == 'Central-difference':
    self.M_pp_HStab *= Parameters.dt/2

  self.M_pp_HStab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_M_ufuf_I1(self, Parameters):
  # Compute M_ufuf_I1.
  self.M_ufuf_I1  = np.einsum('ik, jk, k', self.Nuf, self.Nuf, self.rhof_0*self.weights)
  self.M_ufuf_I1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_M_tsts_J1(self, Parameters):
  # Compute M_tsts_J1.
  self.M_tsts_J1  = np.einsum('ik, jk, k', self.Nts, self.Nts, Parameters.cvs*self.rhos_0*self.weights)
  if Parameters.integrationScheme == 'Newmark-beta':
    self.M_tsts_J1 *= Parameters.gamma*Parameters.dt
  self.M_tsts_J1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_M_tftf_K1(self, Parameters):
  # Compute M_tftf_K1.
  if Parameters.fluidModel == 'Ideal-Gas':
    if 'uf' in Parameters.Physics:
      self.M_tftf_K1  = np.einsum('ik, jk, k', self.Ntf, self.Ntf, Parameters.cvf*self.rhof_0*self.weights)
    else:
      self.M_tftf_K1  = np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.cvf + Parameters.RGas)*self.rhof_0*self.weights)
    if Parameters.SUPG:
      if Parameters.computeKStabTangent:
        self.M_tftf_K1 -= np.einsum('ik, jk, k', self.Btf, self.Ntf, self.tauStab*self.vstar*self.weights)

    if 'RK' not in Parameters.integrationScheme:
      if Parameters.isDynamics:
        self.M_tftf_K1 *= Parameters.gamma*Parameters.dt
        self.M_tftf_K1 -= np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.cvf + Parameters.RGas)*\
                                    self.rhof_0*(self.tfDot/self.tf)*self.weights*Parameters.beta*\
                                    (Parameters.dt**2))
      else:
        self.M_tftf_K1 -= np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.cvf + Parameters.RGas)*\
                                    self.rhof_0*(self.tfDot/self.tf)*self.weights*Parameters.gamma*\
                                    Parameters.dt)
  
  elif Parameters.fluidModel == 'Exponential-Thermal':
    if 'uf' in Parameters.Physics:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nExponential-Thermal model not yet implemented for (u-uf-pf-ts-tf) formulation.")
    else:
      self.M_tftf_K1 = np.einsum('ik, jk, k', self.Ntf, self.Ntf, (self.rhof_0*Parameters.cvf +\
                                 self.J*self.nf*self.tf*Parameters.KF*(Parameters.Af**2))*\
                                 self.weights)
      if 'RK' not in Parameters.integrationScheme:
        if Parameters.isDynamics:
          self.M_tftf_K1 *= Parameters.gamma*Parameters.dt
          self.M_tftf_K1 += np.einsum('ik, jk, k', self.Ntf, self.Ntf, Parameters.KF*\
                                      (Parameters.Af**2)*self.J*self.nf*self.tfDot*self.weights*\
                                      Parameters.beta*(Parameters.dt**2)) 
        else:
          self.M_tftf_K1 += np.einsum('ik, jk, k', self.Ntf, self.Ntf, Parameters.KF*\
                                      (Parameters.Af**2)*self.J*self.nf*self.tfDot*self.weights*\
                                      Parameters.gamma*Parameters.dt) 
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  self.M_tftf_K1 *= Parameters.Area*self.Jacobian
  return

#-------------------
# End mass matrices.
#-------------------
#--------------------------
# Begin stiffness matrices.
#--------------------------
@register_method
def get_K_uu_G2(self, Parameters):
  # Compute K_uu_G2.
  if Parameters.isDynamics:
    #---------------------
    # Classic neo-Hookean.
    #---------------------
    if Parameters.solidModel == 'neo-Hookean':
      #------------------------------------------------
      # Viscous damping contribution, Li et al. (2004).
      #------------------------------------------------
      if Parameters.nu_0 > 1e-12:
        term_1 = (Parameters.lambd + 2*Parameters.mu)*\
                 (Parameters.nu_0*Parameters.gamma*Parameters.dt - \
                  2*Parameters.nu_0*self.dvDX*Parameters.beta*(Parameters.dt**2)/self.F11)
        term_2 = np.log(self.F11)*(2*Parameters.nu_0*Parameters.gamma*Parameters.dt - \
                 (4*Parameters.nu_0*self.dvdX - Parameters.lambd)*\
                 Parameters.beta*(Parameters.dt**2))
        term_3 = (Parameters.lambd + Parameters.mu + Parameters.mu*(self.F11**2) - \
                  2*Parameters.nu_0*self.dvdX/self.F11)*Parameters.beta*(Parameters.dt**2)

        self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, ((term_1 + term_2 + term_3)/\
                                 (self.F11**2))*self.weights)
      else:
        self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                                 (Parameters.beta*(Parameters.dt**2))*\
                                 (Parameters.mu + (Parameters.lambd + Parameters.mu - \
                                  Parameters.lambd*np.log(self.J))/(self.F11**2)))
    #------------------------------------------------------------
    # Ehlers-Eipper incompressible model, Ehlers & Eipper (1998).
    #------------------------------------------------------------
    elif Parameters.solidModel == 'neo-Hookean-Eipper':
      self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                               (Parameters.mu*(1 + self.F11**(-2)) + \
                                Parameters.lambd*((1 - Parameters.ns_0)**2)/\
                                (self.J - Parameters.ns_0)**2)*\
                                Parameters.beta*Parameters.dt**2)
    #--------------------------------------------------------------------
    # Saint-Venant Kirchhoff (small strain elasticity for finite strain).
    #--------------------------------------------------------------------
    elif Parameters.solidModel == 'Saint-Venant-Kirchhoff':
      self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                               Parameters.beta*(Parameters.dt**2)*\
                               (2*self.F11)*(0.5*Parameters.lambd + Parameters.mu))
    #------------------------
    # Clayton & Freed (2019).
    #------------------------
    elif Parameters.solidModel == 'Clayton-Freed' or Parameters.solidModel == 'Clayton-Freed-Linear':
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nImplicit tangent not implemented for Clayton & Freed model, use explicit scheme.")
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid constitutive model not recognized.")
    #------------------------------
    # Shock viscosity contribution.
    #------------------------------
    if np.any(self.dvdX < 0) and (Parameters.C0 > 0.0 and Parameters.C1 > 0.0):
      #---------------------------------------------------
      # Standard/lumping technique (apply across element).
      #---------------------------------------------------
      if self.Gauss_Order == 2:
        bulk_term = self.rhos_0*Parameters.H0e*(Parameters.C0*Parameters.H0e/(self.F11**2)*self.dvdX*\
                    ((1. - 2/self.F11)*self.dvdX*(Parameters.beta*Parameters.dt**2) + 2*Parameters.gamma*Parameters.dt)\
                    - Parameters.C1*self.c/self.F11*((3/2)*self.dvdX*Parameters.beta*Parameters.dt**2/self.F11 \
                    - Parameters.gamma*Parameters.dt))
      #-----------------------------------------------
      # Non-standard technique (apply at Gauss point).
      #-----------------------------------------------
      else:
        bulk_term             = np.zeros(self.Gauss_Order)
        bulk_term[self.Qidxs] = (self.rhos_0*Parameters.H0e*(Parameters.C0*Parameters.H0e/(self.F11**2)*self.dvdX*\
                                ((1. - 2/self.F11)*self.dvdX*(Parameters.beta*Parameters.dt**2)
                                + 2*Parameters.gamma*Parameters.dt)\
                           - Parameters.C1*self.c/self.F11*((3/2)*self.dvdX*Parameters.beta*Parameters.dt**2/self.F11 \
                           - Parameters.gamma*Parameters.dt)))[self.Qidxs]
      
      self.K_uu_G2 -= np.einsum('ik,jk,k', self.Bu, self.Bu, self.weights*bulk_term)
  #----------------------
  # Ignore inertia terms.
  #----------------------
  else:
    #---------------------
    # Classic neo-Hookean.
    #---------------------
    if Parameters.solidModel == 'neo-Hookean':
      #------------------------------------------------
      # Viscous damping contribution, Li et al. (2004).
      #------------------------------------------------
      if Parameters.nu_0 > 1e-12:
        term_1 = (Parameters.lambd + 2*Parameters.mu)*\
                 (Parameters.nu_0 - 2*Parameters.nu_0*self.dvdX*\
                  Parameters.gamma*Parameters.dt/self.F11)
        term_2 = np.log(self.F11)*(2*Parameters.nu_0 - \
                 (4*Parameters.nu_0*self.dvdX - Parameters.lambd)*Parameters.gamma*Parameters.dt)
        term_3 = (Parameters.lambd + Parameters.mu + Parameters.mu*(self.F11**2) - \
                  2*Parameters.nu_0*self.dvdX/self.F11)*Parameters.gamma*Parameters.dt
        self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, ((term_1 + term_2 + term_3)/\
                                 (self.F11**2))*self.weights)
      else:
        if Parameters.integrationScheme == 'Trapezoidal':
          self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                                   Parameters.gamma*Parameters.dt*\
                                   (Parameters.mu + (Parameters.lambd + Parameters.mu - \
                                   Parameters.lambd*np.log(self.J))/(self.F11**2)))
        elif Parameters.integrationScheme == 'Quasi-static':
          self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                                   (Parameters.mu + (Parameters.lambd + Parameters.mu - \
                                   Parameters.lambd*np.log(self.J))/(self.F11**2)))
        else:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nNon-dynamic integration scheme not recognized.")
    #------------------------------------------------------------
    # Ehlers-Eipper incompressible model, Ehlers & Eipper (1998).
    #------------------------------------------------------------
    elif Parameters.solidModel == 'neo-Hookean-Eipper':
      self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                               (Parameters.mu*(1 + self.F11**(-2))\
                                + Parameters.lambd*((1 - Parameters.ns_0)**2)/\
                                (self.J - Parameters.ns_0)**2)*\
                                Parameters.gamma*Parameters.dt)
    #--------------------------------------------------------------------
    # Saint-Venant Kirchhoff (small strain elasticity for finite strain).
    #--------------------------------------------------------------------
    elif Parameters.solidModel == 'Saint-Venant-Kirchhoff':
      self.K_uu_G2 = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*\
                               Parameters.gamma*Parameters.dt*\
                               (2*self.F11)*(0.5*Parameters.lambd + Parameters.mu))
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid constitutive model not recognized.")
  #----------------------------
  # Thermoelastic contribution.
  #---------------------------- 
  if 't' in Parameters.Physics:
    temp_term = np.einsum('ik, jk, k', self.Bu, self.Bu,
                          (self.ts - Parameters.Ts_0)*self.weights/\
                          (self.J**2)*Parameters.As) 
    #------------------------------------------------
    # Single-phase, use bulk isothermal bulk modulus.
    #------------------------------------------------
    if Parameters.Physics == 'u-t': 
      temp_term *= Parameters.Bb0T
    #--------------------------------------------------
    # Multiphase, use skeleton isothermal bulk modulus.
    #--------------------------------------------------
    else:
      temp_term *= Parameters.KSkel
    if Parameters.isDynamics:
      temp_term *= Parameters.beta*Parameters.dt**2
    else:
      temp_term *= Parameters.gamma*Parameters.dt
    
    self.K_uu_G2 += temp_term

  self.K_uu_G2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uu_G3(self, Parameters):
  # Compute K_uu_G3.
  self.K_uu_G3 = np.einsum('ik, jk, k', self.Bu, self.Bu,
                           (self.ns*self.p_f/self.J)*(self.ts/self.tf - 1)*self.weights)
  if Parameters.isDynamics:
    self.K_uu_G3 *= Parameters.beta*Parameters.dt**2
  else:
    self.K_uu_G3 *= Parameters.gamma*Parameters.dt

  self.K_uu_G3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uu_G5(self, Parameters):
  # Compute K_uu_G5.
  self.K_uu_G5  = np.einsum('ik, jk, k', self.Bu, self.Bu, self.weights*self.dvfdX*(self.ns - self.nf)*\
                            (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                            Parameters.beta*(Parameters.dt**2)/(self.F11**2))
  self.K_uu_G5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_up_G1(self, Parameters):
  # Compute K_up_G1.
  try:
    if Parameters.fluidModel == 'Ideal-Gas':
      # Implicit has yet to be implemented in full for (u-uf-pf-ts-tf) 
      # This is merely a placeholder
      if 'tf' in Parameters.Physics:
        self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.J*self.nf*self.a_f*\
                                 self.weights/(Parameters.RGas*self.tf))
      #------------------------------------------
      # Isothermal ideal gas model for (u-uf-pf).
      #------------------------------------------
      else:
        self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.J*self.nf*self.a_f*\
                                 self.weights/(Parameters.RGas*Parameters.Tf_0))
    elif Parameters.fluidModel == 'Exponential':
      self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.rhof_0*self.a_f*\
                               self.weights/Parameters.KF)
    elif Parameters.fluidModel == 'Isentropic':
      self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.a_f*self.rhof_0*\
                               self.weights/(1.4*self.p_f)) 
    elif Parameters.fluidModel == 'Linear-IC':
      self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.a_f*self.J*self.nf*\
                               Parameters.rhofR_0*self.weights/Parameters.p_f0) 
    elif Parameters.fluidModel == 'Linear-Bulk':
      self.K_up_G1 = np.einsum('ik, jk, k', self.Nu, self.Np, self.a_f*self.J*self.nf*\
                               Parameters.rhofR_0*self.weights/Parameters.KF) 
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_up_G1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_up_G1 *= Parameters.beta*(Parameters.dt**2)
  self.K_up_G1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_up_G3(self, Parameters):
  # Compute K_up_G3.
  if 'tf' in Parameters.Physics:
    self.K_up_G3 = -np.einsum('ik, jk, k', self.Bu, self.Np,
                             ((self.ts/self.tf)*self.ns + self.nf)*self.weights)
  else:
    self.K_up_G3 = -np.einsum('ik, jk, k', self.Bu, self.Np, self.weights)

  if Parameters.isDynamics:
    self.K_up_G3 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_up_G3 *= Parameters.gamma*Parameters.dt

  self.K_up_G3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uuf_G1(self, Parameters):
  # Compute K_uuf_G1.
  self.K_uuf_G1  = np.einsum('ik, jk, k', self.Nu, self.Nuf, self.rhof_0*self.weights)
  self.K_uuf_G1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uuf_G5(self, Parameters):
  # Compute K_uuf_G5.
  self.K_uuf_G5  = np.einsum('ik, jk, k', self.Bu, self.Buf, self.weights*self.nf*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                             Parameters.gamma*Parameters.dt/self.F11)
  self.K_uuf_G5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uts_G2(self, Parameters):
  # Compute K_uts_G2.
  self.K_uts_G2 = -np.einsum('ik, jk, k', self.Bu, self.Nts, Parameters.As*Parameters.KSkel*\
                              self.J*self.weights)

  if Parameters.isDynamics:
    self.K_uts_G2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_uts_G2 *= Parameters.gamma*Parameters.dt

  self.K_uts_G2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_uts_G3(self, Parameters):
  # Compute K_uts_G3.
  self.K_uts_G3 = -np.einsum('ik, jk, k', self.Bu, self.Nts, (self.ns*self.p_f/self.tf)*self.weights)

  if Parameters.isDynamics:
    self.K_uts_G3 *= Parameters.beta*Parameters.dt**2
  else:
    self.K_uts_G3 *= Parameters.gamma*Parameters.dt

  self.K_uts_G3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_utf_G3(self, Parameters):
  # Compute K_utf_G3.
  self.K_utf_G3 = np.einsum('ik, jk, k', self.Bu, self.Ntf,
                             (self.ns*self.p_f*self.ts/(self.tf**2))*self.weights)

  if Parameters.isDynamics:
    self.K_utf_G3 *= Parameters.beta*Parameters.dt**2
  else:
    self.K_utf_G3 *= Parameters.gamma*Parameters.dt

  self.K_utf_G3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H1(self, Parameters):
  # Compute K_pu_H1.
  if Parameters.integrationScheme == 'Newmark-beta':
    try:
      self.K_pu_H1 = np.einsum('ik, jk, k', self.Np, self.Bu, self.weights*((self.p_fDot/self.HDiv)*\
                               Parameters.beta*(Parameters.dt**2) + Parameters.gamma*Parameters.dt))
      if 'Linear' in Parameters.fluidModel:
        self.K_pu_H1 += np.einsum('ik, jk, k', self.Np, self.Bu, self.weights*(self.p_fDot/self.HDiv)*\
                                  self.nf*Parameters.beta*(Parameters.dt**2))
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure = ", self.p_f)
      print("Pressure instability in K_pu_H1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  elif Parameters.integrationScheme == 'Trapezoidal':
    try:
      self.K_pu_H1 = np.einsum('ik, jk, k', self.Np, self.Bu, self.weights*((self.p_fDot/self.HDiv)*\
                               Parameters.gamma*Parameters.dt + 1))
      if 'Linear' in Parameters.fluidModel:
        self.K_pu_H1 += np.einsum('ik, jk, k', self.Np, self.Bu, self.weights*(self.p_fDot/self.HDiv)*\
                                  self.nf*Parameters.gamma*Parameters.dt)
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure = ", self.p_f)
      print("Pressure instability in K_pu_H1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  elif Parameters.integrationScheme == 'Central-difference':
    self.K_pu_H1 = np.einsum('ik, jk, k', self.Np, self.Bu, self.weights*Parameters.dt/2)

  self.K_pu_H1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H2(self, Parameters):
  # Compute K_pu_H2.
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    Bu_Factor = self.dkhat*self.vDarcy/self.khat + self.khat*self.dp_fdX/(self.F11**2)
    if 'Linear' in Parameters.fluidModel:
      Bu_Factor += self.J*self.vDarcy
    if Parameters.isDynamics:
      Bu_Factor *= Parameters.beta*(Parameters.dt**2)
    else:
      Bu_Factor *= Parameters.gamma*Parameters.dt

    if Parameters.Physics == 'u-pf' or Parameters.Physics == 'u-pf-ts-tf':
      if Parameters.isDynamics:
        try:
          Nu_Factor = self.khat*self.rhofR
        except FloatingPointError:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Pore fluid pressure = ", self.p_f)
          print("Pressure instability in K_pu_H2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
          raise FloatingPointError
        self.K_pu_H2 = np.einsum('ik, jk, k', self.Np, self.Bu*Bu_Factor - self.Nu*Nu_Factor, self.dp_fdX*self.weights/self.HDiv)

      else:
        self.K_pu_H2 = np.einsum('ik, jk, k', self.Np, self.Bu*Bu_Factor,\
                                 self.dp_fdX*self.weights/self.HDiv)

    elif 'uf' in Parameters.Physics:
      if Parameters.DarcyBrinkman:
        Bu_Factor += self.khat*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                     (self.dvfdX*self.dnfdX*(self.ns/(self.nf**2) + 5/self.nf) + 2*self.d2vfdX2)/\
                     (self.F11**3)
        B2u_Factor = -self.khat*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                     (self.ns/self.nf)*self.dvfdX/(self.F11**4)

        self.K_pu_H2 = np.einsum('ik, jk, k', self.Np, self.Bu*Bu_Factor + self.B2u*B2u_Factor,\
                                 Parameters.beta*(Parameters.dt**2)*\
                                 self.dp_fdX*self.weights/self.HDiv)
        
      else:
        self.K_pu_H2 = np.einsum('ik, jk, k', self.Np, self.Bu*Bu_Factor,\
                                 self.dp_fdX*self.weights/self.HDiv)

    if 'tf' in Parameters.Physics:
      Bu_Factor  = self.khat*self.dp_fdX*self.ns*(1 - self.ts/self.tf)*\
                   (self.dnfdX/self.nf + (3/self.J)*self.d2udX2)/(self.nf*(self.J**2))
      B2u_Factor = self.khat*self.dp_fdX*self.ns*(1 - self.ts/self.tf)/(self.nf*(self.J**2))

      if Parameters.isDynamics:
        Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
        B2u_Factor *= Parameters.beta*(Parameters.dt**2)
      else:
        Bu_Factor  *= Parameters.gamma*Parameters.dt
        B2u_Factor *= Parameters.gamma*Parameters.dt

      self.K_pu_H2 -= np.einsum('ik, jk, k', self.Np, self.Bu, Bu_Factor*self.weights)
      self.K_pu_H2 -= np.einsum('ik, jk, k', self.Np, self.B2u, B2u_Factor*self.weights)

  elif Parameters.integrationScheme == 'Central-difference':
    self.K_pu_H2 = -np.einsum('ik, jk, k', self.Np, self.Nu, self.dp_fdX*self.khat*\
                              self.rhofR*self.weights/self.HDiv)
  
  self.K_pu_H2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H3(self, Parameters):
  # Compute K_pu_H3.
  try:
    self.K_pu_H3 = np.einsum('ik, jk, k', self.Bp, self.Bu, (self.dp_fdX/self.F11)*\
                             (self.dkhat - self.khat/self.F11)*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_pu_H3; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  if Parameters.integrationScheme == 'Newmark-beta':
    self.K_pu_H3 *= Parameters.beta*(Parameters.dt**2)
  elif Parameters.integrationScheme == 'Trapezoidal':
    self.K_pu_H3 *= Parameters.gamma*Parameters.dt
  
  self.K_pu_H3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H4(self, Parameters):
  # Compute K_pu_H4.
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    if Parameters.Physics == 'u-pf' or Parameters.Physics == 'u-pf-ts-tf':
      Bu_Factor = self.dkhat*(self.a_s - Parameters.Gravity)
      if Parameters.isDynamics:
        Bu_Factor *= Parameters.beta*(Parameters.dt**2)
      else:
        Bu_Factor *= Parameters.gamma*Parameters.dt
      try:
        if Parameters.isDynamics:
          self.K_pu_H4 = np.einsum('ik, jk, k', self.Bp, self.Bu*Bu_Factor + self.Nu*self.khat,\
                                   self.rhofR*self.weights)
        else:
          self.K_pu_H4 = np.einsum('ik, jk, k', self.Bp, self.Bu*Bu_Factor, self.rhofR*self.weights)
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in K_pu_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
    elif Parameters.Physics == 'u-uf-pf':
      Bu_Factor = self.dkhat*(self.a_f - Parameters.Gravity)*\
                  Parameters.beta*(Parameters.dt**2)
      try:
        self.K_pu_H4 = np.einsum('ik, jk, k', self.Bp, self.Bu*Bu_Factor, self.rhofR*self.weights)
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in K_pu_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPhysics for problem not recognized.")
  
  elif Parameters.integrationScheme == 'Central-difference':
    try:
      self.K_pu_H4 = np.einsum('ik, jk, k', self.Bp, self.Nu, self.khat*self.rhofR*self.weights)
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure = ", self.p_f)
      print("Pressure instability in K_pu_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError

  self.K_pu_H4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H5(self, Parameters):
  # Compute K_pu_H5.
  Bu_Factor  = self.dkhat*self.DIV_FES/(self.nf*self.khat*self.F11)\
               - (self.dnfdX*self.dvfdX*(self.ns/(self.nf**2) + 5/self.nf) + 2*self.d2vfdX2)*\
                 (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/(self.F11**3)
  B2u_Factor = (self.ns/self.nf)*self.dvfdX*\
               (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/\
               (self.F11**4)

  self.K_pu_H5  = np.einsum('ik, jk, k', -self.Bp, self.Bu*Bu_Factor + self.B2u*B2u_Factor,\
                            self.weights*self.khat*Parameters.beta*(Parameters.dt**2))
  self.K_pu_H5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H6(self, Parameters):
  # Compute K_pu_H6.
  Bu_Factor  = self.dkhat*self.dnfdX - (self.khat*self.ns/self.J)*(self.dnfdX/self.nf +\
               3*self.d2udX2/self.J)
  B2u_Factor = self.ns*self.khat/self.J

  Bu_Factor  *= (self.p_f/(self.J*self.nf))*(1 - self.ts/self.tf)
  B2u_Factor *= (self.p_f/(self.J*self.nf))*(1 - self.ts/self.tf)

  self.K_pu_H6 = np.einsum('ik, jk, k', self.Bp, Bu_Factor*self.Bu + B2u_Factor*self.B2u,
                           self.weights)
  
  if Parameters.isDynamics:
    self.K_pu_H6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_pu_H6 *= Parameters.gamma*Parameters.dt

  self.K_pu_H6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_H7(self, Parameters):
  # Compute K_pu_H7.
  Bu_Factor  = self.tfDot + self.dkhat*self.vDarcy*self.dtfdX/self.khat +\
               self.khat*self.dtfdX/(self.J**2)*(self.dp_fdX - (self.p_f*self.ns/self.nf)*\
               (1 - self.ts/self.tf)*(self.dnfdX/self.nf + 3*self.d2udX2/self.J))
  B2u_Factor = -self.p_f*self.khat*self.ns*self.dtfdX*(1 - self.ts/self.tf)/\
               ((self.J**2)*self.nf)
  
  if Parameters.fluidModel == 'Ideal-Gas':
    Bu_Factor  /= self.tf
    B2u_Factor /= self.tf
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Bu_Factor  *= Parameters.Af
    B2u_Factor *= Parameters.Af
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  
  if Parameters.isDynamics:
    Nu_Factor = -self.khat*self.rhofR*self.dtfdX
    if Parameters.fluidModel == 'Ideal-Gas':
      Nu_Factor /= self.tf
    elif Parameters.fluidModel == 'Exponential-Thermal':
      Nu_Factor *= Parameters.Af
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

    Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor *= Parameters.beta*(Parameters.dt**2)
    self.K_pu_H7 = -np.einsum('ik, jk, k', self.Np, Nu_Factor*self.Nu + Bu_Factor*self.Bu +\
                              B2u_Factor*self.B2u, self.weights)
  else:
    Bu_Factor  *= Parameters.gamma*Parameters.dt
    B2u_Factor *= Parameters.gamma*Parameters.dt
    self.K_pu_H7 = -np.einsum('ik, jk, k', self.Np, Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                              self.weights)

  self.K_pu_H7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pu_HStab(self, Parameters):
  # Compute K_pu_HStab.
  try:
    self.K_pu_HStab = np.einsum('ik, jk, k', self.Bp, self.Bu, -Parameters.alpha_stab*\
                                self.dp_fDotdX*self.weights/(self.F11**2))
    if Parameters.integrationScheme == 'Newmark-beta':
      self.K_pu_HStab *= Parameters.beta*Parameters.dt**2
    elif Parameters.integrationScheme == 'Trapezoidal':
      self.K_pu_HStab *= Parameters.gamma*Parameters.dt
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_pu_HStab; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError

  self.K_pu_HStab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pp_H2(self, Parameters):
  # Compute K_pp_H2.
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    Bp_Factor = self.vDarcy - self.khat*self.dp_fdX/self.F11
    try:
      if Parameters.Physics == 'u-pf':
        if Parameters.fluidModel == 'Exponential':
          Np_Factor = self.khat*self.dp_fdX*self.rhofR*\
                      (self.a_s - Parameters.Gravity)/Parameters.KF
        elif Parameters.fluidModel == 'Linear-Bulk':
          Np_Factor = self.khat*self.dp_fdX*Parameters.rhofR_0*\
                      (self.a_s - Parameters.Gravity)/Parameters.KF
        elif Parameters.fluidModel == 'Linear-IC':
          Np_Factor = self.khat*self.dp_fdX*Parameters.rhofR_0*\
                      (self.a_s - Parameters.Gravity)/Parameters.p_f0
        elif Parameters.fluidModel == 'Ideal-Gas':
          Np_Factor = self.khat*self.dp_fdX*\
                      (self.a_s - Parameters.Gravity)/(Parameters.RGas*Parameters.Tf_0) 
        elif Parameters.fluidModel == 'Isentropic':
          Np_Factor = self.khat*self.dp_fdX*self.rhofR*\
                      (self.a_s - Parameters.Gravity)/(1.4*self.p_f)
        else:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive relation not recognized.")

      elif 'uf' in Parameters.Physics:
        if Parameters.fluidModel == 'Exponential':
          Np_Factor = self.khat*self.dp_fdX*self.rhofR*(self.a_f - Parameters.Gravity)/Parameters.KF
        elif Parameters.fluidModel == 'Linear-Bulk':
          Np_Factor = self.khat*self.dp_fdX*Parameters.rhofR_0*(self.a_f - Parameters.Gravity)/Parameters.KF
        elif Parameters.fluidModel == 'Linear-IC':
          Np_Factor = self.khat*self.dp_fdX*Parameters.rhofR_0*(self.a_f - Parameters.Gravity)/Parameters.p_f0
        elif Parameters.fluidModel == 'Ideal-Gas':
          if 'tf' in Parameters.Physics:
            # This is a placeholder and is not yet fully implemented
            Np_Factor = self.khat*self.dp_fdX*(self.a_f - Parameters.Gravity)/\
                        (Parameters.RGas*self.tf) 
          else:
            Np_Factor = self.khat*self.dp_fdX*(self.a_f - Parameters.Gravity)/\
                        (Parameters.RGas*Parameters.Tf_0) 
        elif Parameters.fluidModel == 'Isentropic':
          Np_Factor = self.khat*self.dp_fdX*self.rhofR*\
                      (self.a_f - Parameters.Gravity)/(1.4*self.p_f)
        else:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive relation not recognized.")
      elif 'tf' in Parameters.Physics:
        if Parameters.fluidModel == 'Ideal-Gas':
          Np_Factor = self.khat*self.dp_fdX*(self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) 
        elif Parameters.fluidModel == 'Exponential-Thermal':
          Np_Factor = self.khat*self.dp_fdX*self.rhofR*(self.a_s - Parameters.Gravity)/Parameters.KF
        else:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure = ", self.p_f)
      print("Pressure instability in K_pp_H2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
    self.K_pp_H2 = np.einsum('ik, jk, k', self.Np, self.Bp*Bp_Factor - self.Np*Np_Factor, self.weights/self.HDiv)

    if Parameters.fluidModel == 'Ideal-Gas':
      self.K_pp_H2 -= np.einsum('ik, jk, k', self.Np, self.Np, self.dp_fdX*self.vDarcy*self.weights/(self.p_f**2))
    
    if 'tf' in Parameters.Physics:
      if Parameters.fluidModel == 'Ideal-Gas':
        self.K_pp_H2 -= np.einsum('ik, jk, k', self.Np, self.Np, (self.khat/self.p_f)*self.dp_fdX*\
                                  self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)*self.weights)
      elif Parameters.fluidModel == 'Exponential-Thermal':
        self.K_pp_H2 -= np.einsum('ik, jk, k', self.Np, self.Np, (self.khat/Parameters.KF)*self.dp_fdX*\
                                  self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)*self.weights)
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    if Parameters.isDynamics:
      self.K_pp_H2 *= Parameters.beta*(Parameters.dt**2)
    else:
      self.K_pp_H2 *= Parameters.gamma*Parameters.dt
    
    
  # elif Parameters.integrationScheme == 'Predictor-corrector':
    # self.K_pp_H2 = np.einsum('ik, jk, k', self.Np, self.Bp, self.vDarcy*self.weights*Parameters.alpha*Parameters.dt/Parameters.KF)
   
  self.K_pp_H2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pp_H3(self, Parameters):
  # Compute K_pp_H3.
  self.K_pp_H3 = np.einsum('ik, jk, k', self.Bp, self.Bp, self.khat*self.weights/self.F11)

  if Parameters.isDynamics:
    if Parameters.integrationScheme == 'Newmark-beta':
      self.K_pp_H3 *= Parameters.beta*(Parameters.dt**2)
    # elif Parameters.integrationScheme == 'Predictor-corrector':
      # self.K_pp_H3 *= Parameters.alpha*Parameters.dt
  else:
    self.K_pp_H3 *= Parameters.gamma*Parameters.dt

  self.K_pp_H3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pp_H4(self, Parameters):
  # Compute K_pp_H4.
  if Parameters.Physics == 'u-pf':
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      if Parameters.fluidModel == 'Exponential':
        try:
          self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.rhofR*self.khat*\
                                   (self.a_s - Parameters.Gravity)*self.weights/Parameters.KF)
        except FloatingPointError:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Pore fluid pressure = ", self.p_f)
          print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
          raise FloatingPointError
      elif Parameters.fluidModel == 'Ideal-Gas':
        try:
          self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*\
                                   (self.a_s - Parameters.Gravity)*self.weights/\
                                   (Parameters.RGas*Parameters.Tf_0))
        except FloatingPointError:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Pore fluid pressure = ", self.p_f)
          print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
          raise FloatingPointError
      elif Parameters.fluidModel == 'Isentropic':
        try:
          self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.rhofR*self.khat*\
                                   (self.a_s - Parameters.Gravity)*self.weights/(1.4*self.p_f))
        except FloatingPointError:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Pore fluid pressure = ", self.p_f)
          print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
          raise FloatingPointError
      else:
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*(self.a_s - Parameters.Gravity)*self.weights)

    # elif Parameters.integrationScheme == 'Predictor-corrector':
      # self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Bp, -self.khat*self.rhofR*self.weights*Parameters.Biot*Parameters.alpha*Parameters.dt/self.rho_0)
    
  elif Parameters.Physics == 'u-uf-pf':
    if Parameters.fluidModel == 'Exponential':
      try:
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.rhofR*self.khat*\
                                 (self.a_f - Parameters.Gravity)*self.weights/Parameters.KF)
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
    elif Parameters.fluidModel == 'Ideal-Gas':
      try:
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*\
                                 (self.a_f - Parameters.Gravity)*self.weights/\
                                 (Parameters.RGas*Parameters.Tf_0))
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
    elif Parameters.fluidModel == 'Isentropic':
      try:
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.rhofR*self.khat*\
                                 (self.a_f - Parameters.Gravity)*self.weights/(1.4*self.p_f))
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Pore fluid pressure = ", self.p_f)
        print("Pressure instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
        raise FloatingPointError
    else:
      self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*\
                               (self.a_f - Parameters.Gravity)*self.weights)

  elif 'tf' in Parameters.Physics:
    try:    
      if Parameters.fluidModel == 'Ideal-Gas':
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*\
                                 (self.a_s - Parameters.Gravity)*self.weights/\
                                 (Parameters.RGas*self.tf))
      elif Parameters.fluidModel == 'Exponential-Thermal':
        self.K_pp_H4 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*self.rhofR*\
                                 (self.a_s - Parameters.Gravity)*self.weights/Parameters.KF)
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure =    ", self.p_f)
      print("Pore fluid temperature = ", self.tf)
      print("Pressure or temperature instability in K_pp_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError

  if Parameters.fluidModel == 'Linear-Bulk':
    self.K_pp_H4 *= Parameters.rhofR_0/Parameters.KF
  elif Parameters.fluidModel == 'Linear-IC':
    self.K_pp_H4 *= Parameters.rhofR_0/Parameters.p_f0
  
  if Parameters.isDynamics:
    self.K_pp_H4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_pp_H4 *= Parameters.gamma*Parameters.dt

  self.K_pp_H4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pp_H6(self, Parameters):
  # Compute K_pp_H6.
  self.K_pp_H6 = np.einsum('ik, jk, k', self.Bp, self.Np, self.khat*self.dnfdX*\
                           self.weights*(1 - self.ts/self.tf)/(self.J*self.nf))
  
  if Parameters.isDynamics:
    self.K_pp_H6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_pp_H6 *= Parameters.gamma*Parameters.dt

  self.K_pp_H6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pp_H7(self, Parameters):
  # Compute K_pp_H7.
  Bp_Factor = 1/self.J 
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = (self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.rhofR*(self.a_s - Parameters.Gravity)/Parameters.KF + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_pp_H7 = np.einsum('ik, jk, k', self.Np, Np_Factor*self.Np + Bp_Factor*self.Bp,
                             self.khat*self.dtfdX*self.weights/self.tf)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_pp_H7 = np.einsum('ik, jk, k', self.Np, Np_Factor*self.Np + Bp_Factor*self.Bp,
                             self.khat*self.dtfdX*self.weights*Parameters.Af)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  
  if Parameters.isDynamics:
    self.K_pp_H7 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_pp_H7 *= Parameters.gamma*Parameters.dt

  self.K_pp_H7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_puf_H2(self, Parameters):
  # Compute K_puf_H2.
  try:
    if Parameters.DarcyBrinkman:
      Buf_Factor  = (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                    self.dnfdX/(self.nf*(self.F11**2))
      B2uf_Factor = (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/(self.F11**2)
      self.K_puf_H2 = -np.einsum('ik, jk, k', self.Np, self.rhofR*self.Nuf + \
                                 Buf_Factor*self.Buf + B2uf_Factor*self.B2uf,\
                                 self.dp_fdX*self.khat*self.weights/self.HDiv)
    else:
      self.K_puf_H2  = -np.einsum('ik, jk, k', self.Np, self.Nuf,\
                                  self.dp_fdX*self.khat*self.rhofR*self.weights/self.HDiv)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_puf_H2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_puf_H2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_puf_H4(self, Parameters):
  # Compute K_puf_H4.
  try:
    self.K_puf_H4  = np.einsum('ik, jk, k', self.Bp, self.Nuf, self.khat*self.rhofR*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_puf_H4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_puf_H4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_puf_H5(self, Parameters):
  # Compute K_puf_H5.
  self.K_puf_H5  = -np.einsum('ik, jk, k', self.Bp, self.Buf*self.dnfdX/self.nf + self.B2uf,\
                              self.khat*(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*\
                              self.weights*Parameters.gamma*Parameters.dt/(self.F11**2))
  self.K_puf_H5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pts_H2(self, Parameters):
  # Compute K_pts_H2.
  self.K_pts_H2 = np.einsum('ik, jk, k', self.Np, self.Nts, self.khat*self.dp_fdX*self.weights/\
                            (self.J*self.nf*self.tf))

  if Parameters.isDynamics:
    self.K_pts_H2 *= Parameters.beta*(Parameters.dt**2)
  else: 
    self.K_pts_H2 *= Parameters.gamma*Parameters.dt

  self.K_pts_H2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pts_H6(self, Parameters):
  # Compute K_pts_H6.
  self.K_pts_H6 = -np.einsum('ik, jk, k', self.Bp, self.Nts, self.khat*self.p_f*self.dnfdX*\
                             self.weights/(self.J*self.nf*self.tf))

  if Parameters.isDynamics:
    self.K_pts_H6 *= Parameters.beta*(Parameters.dt**2)
  else: 
    self.K_pts_H6 *= Parameters.gamma*Parameters.dt

  self.K_pts_H6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_pts_H7(self, Parameters):
  # Compute K_pts_H7.
  self.K_pts_H7  = -np.einsum('ik, jk, k', self.Np, self.Nts, self.khat*self.p_f*self.dnfdX*\
                              self.dtfdX*self.weights/(self.J*self.nf*(self.tf**2)))

  if Parameters.isDynamics:
    self.K_pts_H7 *= Parameters.beta*(Parameters.dt**2)
  else: 
    self.K_pts_H7 *= Parameters.gamma*Parameters.dt

  self.K_pts_H7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ptf_H2(self, Parameters):
  # Compute K_ptf_H2.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_ptf_H2 = np.einsum('ik, jk, k', self.Np, self.Ntf, (self.khat/(self.tf**2))*\
                              self.dp_fdX*((self.a_s - Parameters.Gravity)/Parameters.RGas - \
                              self.dnfdX*self.ts/(self.J*self.nf))*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_ptf_H2 = np.einsum('ik, jk, k', self.Np, self.Ntf, self.dp_fdX*self.khat*\
                              (self.rhofR*Parameters.Af*(self.a_s - Parameters.Gravity) -\
                              self.p_f*self.ts*self.dnfdX/(self.J*self.nf*(self.tf**2)))*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_ptf_H2 *= Parameters.beta*(Parameters.dt**2)
  else: 
    self.K_ptf_H2 *= Parameters.gamma*Parameters.dt

  self.K_ptf_H2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ptf_H6(self, Parameters):
  # Compute K_ptf_H6.
  self.K_ptf_H6 = np.einsum('ik, jk, k', self.Bp, self.Ntf, self.khat*self.p_f*self.dnfdX*\
                            self.ts*self.weights/(self.J*self.nf*(self.tf**2)))

  if Parameters.isDynamics:
    self.K_ptf_H6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_ptf_H6 *= Parameters.gamma*Parameters.dt

  self.K_ptf_H6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ptf_H7(self, Parameters):
  # Compute K_ptf_H7.
  if Parameters.fluidModel == 'Ideal-Gas':
    Ntf_Factor_1 = self.khat*self.rhofR*self.dtfdX*((self.a_s - Parameters.Gravity) -\
                   self.dnfdX*self.ts*Parameters.RGas/(self.J*self.nf)) -\
                   self.vDarcy*self.dtfdX
    Ntf_Factor_2 = self.J*self.nf*self.tf
    Btf_Factor   = self.vDarcy/self.tf
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Ntf_Factor_1 = self.khat*self.dtfdX*(self.rhofR*Parameters.Af*(self.a_s - Parameters.Gravity) - \
                   self.dnfdX*self.ts*self.p_f/(self.J*self.nf*(self.tf**2)))
    Ntf_Factor_2 = self.J*self.nf
    Btf_Factor   = self.vDarcy*Parameters.Af
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    Ntf_Factor_1 *= Parameters.beta*(Parameters.dt**2)
    Ntf_Factor_2 *= Parameters.gamma*Parameters.dt
    Btf_Factor   *= Parameters.beta*(Parameters.dt**2)
  else: 
    Ntf_Factor_1 *= Parameters.gamma*Parameters.dt
    Btf_Factor   *= Parameters.gamma*Parameters.dt

  Ntf_Factor  = Ntf_Factor_1 + Ntf_Factor_2
  if Parameters.fluidModel == 'Ideal-Gas':  
    Ntf_Factor /= self.tf**2
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Ntf_Factor *= Parameters.Af
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  self.K_ptf_H7  = -np.einsum('ik, jk, k', self.Np, Ntf_Factor*self.Ntf + Btf_Factor*self.Btf,\
                              self.weights)

  self.K_ptf_H7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufu_I1(self, Parameters):
  # Compute K_ufu_I1.
  try:
    self.K_ufu_I1  = np.einsum('ik, jk, k', self.Nuf, self.Bu,\
                               self.a_f*self.rhofR*self.weights*Parameters.beta*(Parameters.dt**2))
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_ufu_I1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_ufu_I1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufu_I2(self, Parameters):
  # Compute K_ufu_I2.
  try:
    if Parameters.LagrangeApply:
      T1 = -np.einsum('ik, jk, k', self.Buf, self.Bu, (self.ns/self.J)*self.p_f*self.weights)
      T2 = -np.einsum('ik, jk, k', self.Nuf, self.B2u, self.p_f*(self.ns/self.J)*self.weights)
      T3 =  np.einsum('ik, jk, k', self.Nuf, self.Bu, 2*self.p_f*self.ns*self.d2udX2*self.weights/\
                      (self.J**2))
      self.K_ufu_I2 = T1 + T2 + T3
    else:
      self.K_ufu_I2  = np.einsum('ik, jk, k', self.Nuf, self.Bu, self.ns*self.dp_fdX*self.weights/self.J)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_ufu_I2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_ufu_I2 *= Parameters.beta*(Parameters.dt**2)
  self.K_ufu_I2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufu_I3(self, Parameters):
  # Compute K_ufu_I3.
  Nu_Factor = self.J*Parameters.gamma*Parameters.dt
  Bu_Factor = (1 + 2*self.ns/self.nf - self.J*self.dkhat/self.khat)*\
              (self.v_f - self.v_s)*Parameters.beta*(Parameters.dt**2)
  self.K_ufu_I3  = np.einsum('ik, jk, k', self.Nuf, self.Bu*Bu_Factor - self.Nu*Nu_Factor,\
                             (self.nf**2)*self.weights/self.khat)
  self.K_ufu_I3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufu_I4(self, Parameters):
  # Compute K_ufu_I4.
  try:
    self.K_ufu_I4  = np.einsum('ik, jk, k', self.Nuf, self.Bu, -self.rhofR*self.weights*Parameters.Gravity*Parameters.beta*(Parameters.dt**2))
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_ufu_I4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_ufu_I4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufu_I5(self, Parameters):
  # Compute K_ufu_I5.
  Bu_Factor      = -4*self.dvfdX*self.dnfdX + (self.ns - self.nf)*self.d2vfdX2
  B2u_Factor     = self.ns*self.dvfdX/self.J
  self.K_ufu_I5  = np.einsum('ik, jk, k', -self.Nuf, self.Bu*Bu_Factor + self.B2u*B2u_Factor,\
                             (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)\
                             *(Parameters.beta*(Parameters.dt**2))*self.weights/(self.F11**2))

  self.K_ufu_I5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufuf_I3(self, Parameters):
  # Compute K_ufuf_I3.
  self.K_ufuf_I3  = np.einsum('ik, jk, k', self.Nuf, self.Nuf, self.J*(self.nf**2)*self.weights*Parameters.gamma*Parameters.dt/self.khat)
  self.K_ufuf_I3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufuf_I5(self, Parameters):
  # Compute K_ufuf_I5.
  self.K_ufuf_I5  = np.einsum('ik, jk, k', -self.Nuf, (self.dnfdX*self.Buf + self.nf*self.B2uf),\
                              self.weights*Parameters.gamma*Parameters.dt\
                              *(Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)/self.F11)
  self.K_ufuf_I5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufp_I1(self, Parameters):
  # Compute K_ufp_I1.
  try:
    if Parameters.fluidModel == 'Ideal-Gas':
      self.K_ufp_I1 = np.einsum('ik, jk, k', self.Nuf, self.Np, self.a_f*self.J*\
                                                                self.nf*self.weights/\
                                                                (Parameters.RGas*Parameters.Tf_0))
    elif 'Linear' not in Parameters.fluidModel:
      self.K_ufp_I1 = np.einsum('ik, jk, k', self.Nuf, self.Np, self.a_f*self.rhof_0*self.weights)
      if Parameters.fluidModel == 'Exponential':
        self.K_ufp_I1 /= Parameters.KF
      elif Parameters.fluidModel == 'Isentropic':
        self.K_ufp_I1 /= 1.4*self.p_f
    else:
      self.K_ufp_I1 = np.einsum('ik, jk, k', self.Nuf, self.Np,\
                                self.a_f*self.J*self.nf*Parameters.rhofR_0*self.weights)
      if 'IC' in Parameters.fluidModel:
        self.K_ufp_I1 /= Parameters.p_f0
      elif 'Bulk' in Parameters.fluidModel:
        self.K_ufp_I1 /= Parameters.KF
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_ufp_I1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_ufp_I1 *= Parameters.beta*(Parameters.dt**2)
  self.K_ufp_I1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufp_I2(self, Parameters):
  # Compute K_ufp_I2.
  if Parameters.LagrangeApply:
    T1 = -np.einsum('ik, jk, k', self.Buf, self.Np, self.nf*self.weights)
    T2 = -np.einsum('ik, jk, k', self.Nuf, self.Np, self.dnfdX*self.weights)
    self.K_ufp_I2 = T1 + T2
  else:
    self.K_ufp_I2  = np.einsum('ik, jk, k', self.Nuf, self.Bp, self.nf*self.weights)
  self.K_ufp_I2 *= Parameters.beta*(Parameters.dt**2)
  self.K_ufp_I2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_ufp_I4(self, Parameters):
  # Compute K_ufp_I4.
  try:
    if Parameters.fluidModel == 'Ideal-Gas': # Isothermal model
      self.K_ufp_I4 = np.einsum('ik, jk, k', self.Nuf, self.Np, -self.J*self.nf*\
                                self.weights*Parameters.Gravity/(Parameters.RGas*Parameters.Tf_0))
    elif 'Linear' not in Parameters.fluidModel:
      self.K_ufp_I4 = np.einsum('ik, jk, k', self.Nuf, self.Np, 
                                -self.rhof_0*self.weights*Parameters.Gravity)
      if Parameters.fluidModel == 'Exponential':
        self.K_ufp_I4 /= Parameters.KF
      elif Parameters.fluidModel == 'Isentropic':
        self.K_ufp_I4 /= 1.4*self.p_f
    else:
      self.K_ufp_I4 = np.einsum('ik, jk, k', self.Nuf, self.Np, 
                                -self.J*self.nf*Parameters.rhofR_0*self.weights*Parameters.Gravity)
      if 'IC' in Parameters.fluidModel:
        self.K_ufp_I4 /= Parameters.p_f0
      elif 'Bulk' in Parameters.fluidModel:
        self.K_ufp_I4 /= Parameters.KF
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in K_ufp_I4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.K_ufp_I4 *= Parameters.beta*(Parameters.dt**2)
  self.K_ufp_I4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J1(self, Parameters):
  # Compute_K_tsu_J1.
  self.K_tsu_J1 = -np.einsum('ik, jk, k', self.Nts, self.Bu, self.rhos_0*self.tsDot*\
                             self.weights/self.J)

  if Parameters.isDynamics:
    self.K_tsu_J1 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsu_J1 *= Parameters.gamma*Parameters.dt

  self.K_tsu_J1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J2(self, Parameters):
  # Compute K_tsu_J2.
  if Parameters.isDynamics:
    #----------------------------------------------
    # Compute shock viscosity heating contribution.
    #----------------------------------------------
    if np.any(self.dvdX < 0) and (Parameters.C0 > 0.0 and Parameters.C1 > 0.0):
      #---------------------------------------------------
      # Standard/lumping technique (apply across element).
      #---------------------------------------------------
      if self.Gauss_Order == 2:
        bulk_term = self.rhos_0*Parameters.H0e*\
                    (Parameters.C0*Parameters.H0e*(self.dvdX/self.J)*((1 - 2/self.J)*\
                    self.dvdX*Parameters.beta*(Parameters.dt**2) + 2*Parameters.gamma*\
                    Parameters.dt) - Parameters.C1*self.c*(1.5*self.dvdX*Parameters.beta*\
                    (Parameters.dt**2)/self.J - Parameters.gamma*Parameters.dt))
      #-----------------------------------------------
      # Non-standard technique (apply at Gauss point).
      #-----------------------------------------------
      else:
        bulk_term             = np.zeros(self.Gauss_Order)
        bulk_term[self.Qidxs] = (self.rhos_0*Parameters.H0e*\
                                 (Parameters.C0*Parameters.H0e*(self.dvdX/self.J)*((1 - 2/self.J)*\
                                  self.dvdX*Parameters.beta*(Parameters.dt**2) +\
                                  2*Parameters.gamma*Parameters.dt) -\
                                  Parameters.C1*self.c*(1.5*self.dvdX*Parameters.beta*\
                                  (Parameters.dt**2)/self.J -\
                                  Parameters.gamma*Parameters.dt)))[self.Qidxs]
      #--------------
      # Single-phase.
      #--------------
      if Parameters.Physics == 'u-t':
        self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu, (bulk_term + \
                                  (Parameters.Bb0T*Parameters.As*self.ts/self.J)*\
                                  (Parameters.gamma*Parameters.dt - (self.JDot/self.J)*\
                                  (Parameters.beta*(Parameters.dt**2))) + self.Q*\
                                  Parameters.gamma*Parameters.dt)*self.weights)
      #------------
      # Multiphase.
      #------------
      else:
        self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu, (bulk_term + \
                                  (Parameters.KSkel*Parameters.As*self.ts/self.J)*\
                                  (Parameters.gamma*Parameters.dt - (self.JDot/self.J)*\
                                  (Parameters.beta*(Parameters.dt**2))) + self.Q*\
                                  Parameters.gamma*Parameters.dt)*self.weights)
    #------------------
    # No shock heating.
    #------------------
    else:
      #--------------
      # Single-phase.
      #--------------
      if Parameters.Physics == 'u-t':
        self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu,\
                                  Parameters.Bb0T*Parameters.As*self.ts/self.J*\
                                  (-Parameters.beta*(Parameters.dt**2)*self.JDot/self.J +\
                                   Parameters.gamma*Parameters.dt)*self.weights)
      #------------
      # Multiphase.
      #------------
      else:
        self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu,\
                                  Parameters.KSkel*Parameters.As*self.ts/self.J*\
                                  (-Parameters.beta*(Parameters.dt**2)*self.JDot/self.J +\
                                   Parameters.gamma*Parameters.dt)*self.weights)
    #---------------------------------------------------------------
    # Multiphase volume fraction & pore fluid pressure contribution.
    #---------------------------------------------------------------
    if 'pf' in Parameters.Physics:
      self.K_tsu_J2 += np.einsum('ik, jk, k', self.Nts, self.Bu, self.ns*self.p_f*(self.ts/self.tf)*\
                                 (-Parameters.beta*(Parameters.dt**2)*(self.JDot/self.J) + \
                                 Parameters.gamma*Parameters.dt)*self.weights)
  #----------------------
  # Ignore inertia terms.
  #----------------------
  else:
    #--------------
    # Single-phase.
    #--------------
    if Parameters.Physics == 'u-t':
      self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu, Parameters.Bb0T*Parameters.As*\
                                (self.ts/self.J)*self.weights*\
                                (-self.JDot*Parameters.gamma*Parameters.dt + 1))
    #------------
    # Multiphase.
    #------------
    else:
      self.K_tsu_J2 = np.einsum('ik, jk, k', self.Nts, self.Bu, Parameters.KSkel*Parameters.As*\
                                (self.ts/self.J)*self.weights*\
                                (-self.JDot*Parameters.gamma*Parameters.dt + 1))
    #---------------------------------------------------------------
    # Multiphase volume fraction & pore fluid pressure contribution.
    #---------------------------------------------------------------
    if 'pf' in Parameters.Physics:
      self.K_tsu_J2 += np.einsum('ik, jk, k', self.Nts, self.Bu, self.ns*self.p_f*(self.ts/self.tf)*\
                                 (-Parameters.gamma*Parameters.dt*(self.JDot/self.J) + 1)*self.weights)

  self.K_tsu_J2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J3(self, Parameters):
  # Compute K_tsu_J3.
  if 'pf' in Parameters.Physics:
    self.K_tsu_J3  = -np.einsum('ik, jk, k', self.Bts, self.Bu, 2*Parameters.ks*self.dtsdX*\
                                self.weights*((self.ns/self.J)**2))
  else:
    self.K_tsu_J3 = -np.einsum('ik, jk, k', self.Bts, self.Bu, Parameters.ks*self.dtsdX*\
                               self.weights/(self.J**2))
  
  if Parameters.isDynamics:
    self.K_tsu_J3 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsu_J3 *= Parameters.gamma*Parameters.dt

  self.K_tsu_J3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J4(self, Parameters):
  # Compute K_tsu_J4.
  self.K_tsu_J4 = np.einsum('ik, jk, k', self.Nts, self.Bu, Parameters.k_exchange*\
                            (self.ts - self.tf)*self.weights)

  if Parameters.isDynamics:
    self.K_tsu_J4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsu_J4 *= Parameters.gamma*Parameters.dt

  self.K_tsu_J4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J5(self, Parameters):
  # Compute K_tsu_J5.
  Bu_Factor  = self.vDarcy*((self.vDarcy/self.khat)*(1 + self.J*self.dkhat/self.khat) +\
               (2/self.J)*(self.dp_fdX - (self.ns*self.p_f/self.nf)*(self.dnfdX/self.nf +\
               (3/self.J)*self.d2udX2)))
  B2u_Factor = -2*self.ns*self.p_f*self.vDarcy*(1 - self.ts/self.tf)/(self.J*self.nf)

  if Parameters.isDynamics:
    Nu_Factor   = -2*self.J*self.vDarcy*self.rhofR
    Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor *= Parameters.beta*(Parameters.dt**2)

    self.K_tsu_J5 = np.einsum('ik, jk, k', self.Nts, self.B2u*B2u_Factor + self.Bu*Bu_Factor +\
                              self.Nu*Nu_Factor, self.weights)
  else:
    Bu_Factor  *= Parameters.gamma*Parameters.dt
    B2u_Factor *= Parameters.gamma*Parameters.dt

    self.K_tsu_J5 = np.einsum('ik, jk, k', self.Nts, self.B2u*B2u_Factor + self.Bu*Bu_Factor,\
                              self.weights)
  self.K_tsu_J5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsu_J6(self, Parameters):
  # Compute K_tsu_J6.
  Bu_Factor  = (self.ts/self.tf)*(self.p_f/self.nf)*(self.dnfdX*((self.dkhat/self.khat)*self.vDarcy + \
               (self.khat/(self.J**2))*(self.dp_fdX - (self.ns*self.p_f/self.nf)*\
               (1 - self.ts/self.tf)*(self.dnfdX/self.nf + (3/self.J)*self.d2udX2))) -\
               (self.ns/self.J)*self.vDarcy*(self.dnfdX/self.nf + (2/self.J)*self.d2udX2))
  B2u_Factor = (self.ts/self.tf)*(self.p_f/self.nf)*(self.ns/self.J)*(self.vDarcy -\
               (self.khat/self.J)*(self.p_f/self.nf)*(1 - self.ts/self.tf))
  
  if Parameters.isDynamics:
    if Parameters.fluidModel == 'Ideal-Gas':
      Nu_Factor = -(self.ts/self.nf)*((self.rhofR**2)/Parameters.RGas)*self.dnfdX*self.khat
    elif Parameters.fluidModel == 'Exponential-Thermal':
      Nu_Factor = -(self.ts/self.tf)*(self.p_f/self.nf)*self.dnfdX*self.khat*self.rhofR

    Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor *= Parameters.beta*(Parameters.dt**2)
    self.K_tsu_J6 = np.einsum('ik, jk, k', self.Nts, Nu_Factor*self.Nu + Bu_Factor*self.Bu +\
                              B2u_Factor*self.B2u, self.weights)
  else:
    Bu_Factor  *= Parameters.gamma*Parameters.dt
    B2u_Factor *= Parameters.gamma*Parameters.dt
    self.K_tsu_J6 = np.einsum('ik, jk, k', self.Nts, Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                              self.weights)

  self.K_tsu_J6 *= Parameters.Area*self.Jacobian
  return 
  
@register_method
def get_K_tsp_J2(self, Parameters):
  # Compute K_tsp_J2.
  self.K_tsp_J2 = np.einsum('ik, jk, k', self.Nts, self.Np, self.ns*self.JDot*(self.ts/self.tf)*\
                            self.weights)
  
  if Parameters.isDynamics:
    self.K_tsp_J2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsp_J2 *= Parameters.gamma*Parameters.dt

  self.K_tsp_J2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsp_J5(self, Parameters):
  # Compute K_tsp_J5.
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = self.J*((self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) +\
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.J*(self.rhofR*(self.a_s - Parameters.Gravity)/Parameters.KF +\
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  
  self.K_tsp_J5 = np.einsum('ik, jk, k', self.Nts, self.Np*Np_Factor + self.Bp,\
                            -2*self.vDarcy*self.weights)
  
  if Parameters.isDynamics:
    self.K_tsp_J5 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsp_J5 *= Parameters.gamma*Parameters.dt

  self.K_tsp_J5 *= Parameters.Area*self.Jacobian
  return
  
@register_method
def get_K_tsp_J6(self, Parameters):
  # Compute K_tsp_J6.
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = self.vDarcy - self.khat*self.p_f*((self.a_s - Parameters.Gravity)/(Parameters.RGas*\
                self.tf) + self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.vDarcy - self.khat*self.p_f*(self.rhofR*(self.a_s - Parameters.Gravity)/\
                Parameters.KF + self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  Bp_Factor = -self.khat*self.p_f/self.J
  
  self.K_tsp_J6  = np.einsum('ik, jk, k', self.Nts, Np_Factor*self.Np + Bp_Factor*self.Bp,
                             (self.ts/self.tf)*self.dnfdX*self.weights/self.nf)

  if Parameters.isDynamics:
    self.K_tsp_J6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsp_J6 *= Parameters.gamma*Parameters.dt

  self.K_tsp_J6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsts_J2(self, Parameters):
  # Compute K_tsts_J2.
  if 'pf' in Parameters.Physics:
    self.K_tsts_J2 = np.einsum('ik, jk, k', self.Nts, self.Nts, self.JDot*self.weights*\
                               (Parameters.KSkel*Parameters.As/self.J + self.ns*self.p_f/self.tf))
  else:
    self.K_tsts_J2 = np.einsum('ik, jk, k', self.Nts, self.Nts, self.JDot*self.weights*\
                               Parameters.Bb0T*Parameters.As/self.J)

  if Parameters.isDynamics:
    self.K_tsts_J2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsts_J2 *= Parameters.gamma*Parameters.dt

  self.K_tsts_J2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsts_J3(self, Parameters):
  # Compute K_tsts_J3.
  if 'pf' in Parameters.Physics:
    self.K_tsts_J3 = np.einsum('ik, jk, k', self.Bts, self.Bts, Parameters.ks*self.ns*\
                               self.weights/self.J)
  else:
    self.K_tsts_J3 = np.einsum('ik, jk, k', self.Bts, self.Bts, Parameters.ks*self.weights/self.J)

  if Parameters.isDynamics:
    self.K_tsts_J3 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsts_J3 *= Parameters.gamma*Parameters.dt

  self.K_tsts_J3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsts_J4(self, Parameters):
  # Compute K_tsts_J4.
  self.K_tsts_J4 = np.einsum('ik, jk, k', self.Nts, self.Nts, Parameters.k_exchange*self.J*\
                             self.weights)

  if Parameters.isDynamics:
    self.K_tsts_J4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsts_J4 *= Parameters.gamma*Parameters.dt

  self.K_tsts_J4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsts_J5(self, Parameters):
  # Compute K_tsts_J5.
  self.K_tsts_J5 = np.einsum('ik, jk, k', self.Nts, self.Nts, 2*self.p_f*self.vDarcy*\
                             self.dnfdX*self.weights/(self.nf*self.tf))

  if Parameters.isDynamics:
    self.K_tsts_J5 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsts_J5 *= Parameters.gamma*Parameters.dt

  self.K_tsts_J5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tsts_J6(self, Parameters):
  # Compute K_tsts_J6.
  self.K_tsts_J6  = np.einsum('ik, jk, k', self.Nts, self.Nts, self.p_f*self.dnfdX/(self.nf*self.tf)*\
                              (self.vDarcy + ((self.khat*self.p_f)/(self.J*self.nf))*self.dnfdX*\
                              (self.ts/self.tf))*self.weights)

  if Parameters.isDynamics:
    self.K_tsts_J6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tsts_J6 *= Parameters.gamma*Parameters.dt

  self.K_tsts_J6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tstf_J2(self, Parameters):
  # Compute K_tstf_J2.
  self.K_tstf_J2 = -np.einsum('ik, jk, k', self.Nts, self.Ntf, self.ns*self.p_f*self.JDot*\
                              self.ts*self.weights/(self.tf**2))
  
  if Parameters.isDynamics:
    self.K_tstf_J2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tstf_J2 *= Parameters.gamma*Parameters.dt

  self.K_tstf_J2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tstf_J4(self, Parameters):
  # Compute K_tstf_J4.
  self.K_tstf_J4  = -np.einsum('ik, jk, k', self.Nts, self.Ntf, Parameters.k_exchange*self.J*\
                               self.weights)

  if Parameters.isDynamics:
    self.K_tstf_J4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tstf_J4 *= Parameters.gamma*Parameters.dt

  self.K_tstf_J4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tstf_J5(self, Parameters):
  # Compute K_tstf_J5.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tstf_J5 = np.einsum('ik, jk, k', self.Nts, self.Ntf, 2*self.J*self.vDarcy*\
                               (self.rhofR/self.tf)*((self.a_s - Parameters.Gravity) - \
                               (Parameters.RGas*self.ts)/(self.J*self.nf)*self.dnfdX)*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tstf_J5 = np.einsum('ik, jk, k', self.Nts, self.Ntf, 2*self.J*self.vDarcy*\
                               (self.rhofR*Parameters.Af*((self.a_s - Parameters.Gravity) -\
                               (self.p_f*self.ts)/(self.J*self.nf*(self.tf**2))*self.dnfdX))*\
                               self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tstf_J5 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tstf_J5 *= Parameters.gamma*Parameters.dt

  self.K_tstf_J5 *= Parameters.Area*self.Jacobian
  return 

@register_method
def get_K_tstf_J6(self, Parameters):
  # Compute K_tstf_J6.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tstf_J6 = np.einsum('ik, jk, k', self.Nts, self.Ntf, (self.ts/(self.tf**2))*self.dnfdX*\
                               (self.p_f/self.nf)*(self.khat*self.rhofR*\
                               ((self.a_s - Parameters.Gravity) - (Parameters.RGas*self.ts)/\
                               (self.J*self.nf)*self.dnfdX) - self.vDarcy)*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tstf_J6 = np.einsum('ik, jk, k', self.Nts, self.Ntf, (self.ts/self.tf)*self.dnfdX*\
                               (self.p_f/self.nf)*(self.khat*self.rhofR*Parameters.Af*\
                               ((self.a_s - Parameters.Gravity) - self.khat*(self.p_f*self.ts)/\
                               (self.J*self.nf*(self.tf**2))*self.dnfdX) - self.vDarcy)*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tstf_J6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tstf_J6 *= Parameters.gamma*Parameters.dt

  self.K_tstf_J6 *= Parameters.Area*self.Jacobian
  return 

@register_method
def get_K_tfu_K1(self, Parameters):
  # Compute K_tfu_K1.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tfu_K1 = np.einsum('ik, jk, k', self.Ntf, self.Bu, (Parameters.cvf + Parameters.RGas)*\
                              self.rhofR*self.tfDot*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tfu_K1 = np.einsum('ik, jk, k', self.Ntf, self.Bu, (self.rhofR*Parameters.cvf + \
                              Parameters.KF*(Parameters.Af**2)*self.tf)*self.tfDot*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tfu_K1 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfu_K1 *= Parameters.gamma*Parameters.dt

  self.K_tfu_K1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K2(self, Parameters):
  # Compute K_tfu_K2.
  Bu_Factor   = self.dkhat*self.vDarcy/self.khat + (self.khat/(self.J**2))*(self.dp_fdX - \
                (self.ns*self.p_f/self.nf)*(1 - self.ts/self.tf)*(self.dnfdX/self.nf + \
                (3/self.J)*self.d2udX2))
  B2u_Factor  = -(self.p_f/self.nf)*(self.ns*self.khat/(self.J**2))*(1 - self.ts/self.tf)
  
  if Parameters.isDynamics:
    Nu_Factor   = -self.khat*self.rhofR
    Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor *= Parameters.beta*(Parameters.dt**2)
    if Parameters.fluidModel == 'Ideal-Gas':
      self.K_tfu_K2 = np.einsum('ik, jk, k', self.Ntf, Nu_Factor*self.Nu + Bu_Factor*self.Bu + \
                                B2u_Factor*self.B2u, (Parameters.cvf + Parameters.RGas)*self.rhofR*\
                                self.dtfdX*self.weights)
    elif Parameters.fluidModel == 'Exponential-Thermal':
      self.K_tfu_K2 = np.einsum('ik, jk, k', self.Ntf, Nu_Factor*self.Nu + Bu_Factor*self.Bu + \
                                B2u_Factor*self.B2u, (self.rhofR*Parameters.cvf + \
                                Parameters.KF*(Parameters.Af**2)*self.tf)*\
                                self.dtfdX*self.weights)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  else:
    Bu_Factor  *= Parameters.gamma*Parameters.dt
    B2u_Factor *= Parameters.gamma*Parameters.dt
    if Parameters.fluidModel == 'Ideal-Gas':
      self.K_tfu_K2 = np.einsum('ik, jk, k', self.Ntf, Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                                (Parameters.cvf + Parameters.RGas)*self.rhofR*self.dtfdX*self.weights)
    elif Parameters.fluidModel == 'Exponential-Thermal':
      self.K_tfu_K2 = np.einsum('ik, jk, k', self.Ntf,Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                                (self.rhofR*Parameters.cvf + Parameters.KF*(Parameters.Af**2)*self.tf)*\
                                self.dtfdX*self.weights)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  self.K_tfu_K2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K3(self, Parameters):
  # Compute K_tfu_K3.
  if Parameters.isDynamics:
    self.K_tfu_K3 = -np.einsum('ik, jk, k', self.Ntf, self.Bu, self.ns*self.p_f*\
                               ((self.JDot/self.J)*Parameters.beta*(Parameters.dt**2) - \
                               Parameters.gamma*Parameters.dt)*self.weights)
  else:
    self.K_tfu_K3 = -np.einsum('ik, jk, k', self.Ntf, self.Bu, self.ns*self.p_f*\
                               ((self.JDot/self.J)*Parameters.gamma*Parameters.dt - 1)*\
                               self.weights)

  self.K_tfu_K3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K4(self, Parameters):
  # Compute K_tfu_K4.
  Bu_Factor  = (self.p_f/self.nf)*(self.dnfdX*(self.ns*self.vDarcy/(self.J*self.nf) -\
               (self.dkhat/self.khat)*self.vDarcy - (self.khat/(self.J**2))*\
               (self.dp_fdX - (self.ns*self.p_f/self.nf)*(1 - self.ts/self.tf)*\
               (self.dnfdX/self.nf + (3/self.J)*self.d2udX2))) +\
               2*self.ns*self.d2udX2*self.vDarcy/(self.J**2))
  B2u_Factor = (self.ns/self.J)*(self.p_f/self.nf)*((self.khat/self.J)*self.p_f*self.dnfdX*\
               (1 - self.ts/self.tf) - self.vDarcy) 
  if Parameters.isDynamics:
    Nu_Factor   = (self.p_f/self.nf)*self.dnfdX*self.khat*self.rhofR
    Bu_Factor  *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor *= Parameters.beta*(Parameters.dt**2)
    self.K_tfu_K4 = np.einsum('ik, jk, k', self.Ntf, Nu_Factor*self.Nu + Bu_Factor*self.Bu +\
                              B2u_Factor*self.B2u, self.weights)
  else:
    Bu_Factor  *= Parameters.gamma*Parameters.dt
    B2u_Factor *= Parameters.gamma*Parameters.dt
    self.K_tfu_K4 = np.einsum('ik, jk, k', self.Ntf, Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                              self.weights)

  self.K_tfu_K4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K5(self, Parameters):
  # Compute K_tfu_K5.
  self.K_tfu_K5 = -np.einsum('ik, jk, k', self.Ntf, self.Bu, self.p_fDot*self.weights)

  if Parameters.isDynamics:
    self.K_tfu_K5 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfu_K5 *= Parameters.gamma*Parameters.dt

  self.K_tfu_K5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K6(self, Parameters):
  # Compute K_tfu_K6.
  Bu_Factor  = -self.dkhat*self.vDarcy/self.khat - (self.khat/(self.J**2))*(self.dp_fdX - \
               (self.ns*self.p_f/self.nf)*(1 - self.ts/self.tf)*(self.dnfdX/self.nf + \
               (3/self.J)*self.d2udX2))
  B2u_Factor = (self.p_f/self.nf)*(self.ns*self.khat/(self.J**2))*(1 - self.ts/self.tf)

  if Parameters.isDynamics:
    Nu_Factor     = self.khat*self.rhofR
    Bu_Factor    *= Parameters.beta*(Parameters.dt**2)
    B2u_Factor   *= Parameters.beta*(Parameters.dt**2)
    self.K_tfu_K6 = np.einsum('ik, jk, k', self.Ntf, Nu_Factor*self.Nu + Bu_Factor*self.Bu + \
                              B2u_Factor*self.B2u, self.dp_fdX*self.weights)
  else:
    Bu_Factor    *= Parameters.gamma*Parameters.dt
    B2u_Factor   *= Parameters.gamma*Parameters.dt
    self.K_tfu_K6 = np.einsum('ik, jk, k', self.Ntf, Bu_Factor*self.Bu + B2u_Factor*self.B2u,\
                              self.dp_fdX*self.weights)

  self.K_tfu_K6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K7(self, Parameters):
  # Compute K_tfu_K7.
  self.K_tfu_K7 = np.einsum('ik, jk, k', self.Btf, self.Bu, ((self.ns - self.nf)/(self.J**2))*\
                            self.dtfdX*self.weights)

  if Parameters.isDynamics:
    self.K_tfu_K7 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfu_K7 *= Parameters.gamma*Parameters.dt

  self.K_tfu_K7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfu_K8(self, Parameters):
  # Compute K_tfu_K8.
  self.K_tfu_K8 = -np.einsum('ik, jk, k', self.Ntf, self.Bu, Parameters.k_exchange*\
                             (self.ts -self.tf)*self.weights)

  if Parameters.isDynamics:
    self.K_tfu_K8 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfu_K8 *= Parameters.gamma*Parameters.dt

  self.K_tfu_K8 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K1(self, Parameters):
  # Compute K_tfp_K1.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tfp_K1 = np.einsum('ik, jk, k', self.Ntf, self.Np, (Parameters.cvf + Parameters.RGas)*\
                              (self.J*self.nf/(Parameters.RGas*self.tf))*self.tfDot*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tfp_K1 = np.einsum('ik, jk, k', self.Ntf, self.Np, Parameters.cvf*self.rhof_0*\
                              self.tfDot*self.weights/Parameters.KF) 
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tfp_K1 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfp_K1 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K2(self, Parameters):
  # Compute K_tfp_K2.
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = self.vDarcy/(Parameters.RGas*self.tf) - self.khat*self.rhofR*\
               ((self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) + self.dnfdX*\
               (1 - self.ts/self.tf)/(self.J*self.nf))
    Bp_Factor = -self.khat*self.rhofR/self.J

    self.K_tfp_K2 = np.einsum('ik, jk, k', self.Ntf, Np_Factor*self.Np + Bp_Factor*self.Bp,
                              (Parameters.cvf + Parameters.RGas)*self.dtfdX*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.rhofR*self.vDarcy*Parameters.cvf/Parameters.KF - \
                (self.rhofR*Parameters.cvf + Parameters.KF*(Parameters.Af**2)*self.tf)*\
                self.khat*(self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf) + self.rhofR*\
                (self.a_s - Parameters.Gravity)/Parameters.KF)
    Bp_Factor = -(self.rhofR*Parameters.cvf + Parameters.KF*(Parameters.Af**2)*self.tf)*\
                 self.khat/self.J

    self.K_tfp_K2 = np.einsum('ik, jk, k', self.Ntf, Np_Factor*self.Np + Bp_Factor*self.Bp,\
                              self.dtfdX*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tfp_K2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfp_K2 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K3(self, Parameters):
  # Compute K_tfp_K3.
  self.K_tfp_K3 = -np.einsum('ik, jk, k', self.Ntf, self.Np, self.ns*self.JDot*self.weights)
  
  if Parameters.isDynamics:
    self.K_tfp_K3 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfp_K3 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K4(self, Parameters):
  # Compute K_tfp_K4.
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = self.khat*self.p_f*((self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)) - self.vDarcy
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.khat*self.p_f*(self.rhofR*(self.a_s - Parameters.Gravity)/Parameters.KF + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf)) - self.vDarcy
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  Bp_Factor = self.khat*self.p_f/self.J

  self.K_tfp_K4  = np.einsum('ik, jk, k', self.Ntf, Np_Factor*self.Np + Bp_Factor*self.Bp,\
                             (self.dnfdX/self.nf)*self.weights)

  if Parameters.isDynamics:
    self.K_tfp_K4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfp_K4 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K5(self, Parameters):
  # Compute K_tfp_K5.
  self.K_tfp_K5 = -np.einsum('ik, jk, k', self.Ntf, self.Np, self.J*self.nf*self.weights)

  if Parameters.isDynamics:
    self.K_tfp_K5 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfp_K6(self, Parameters):
  # Compute K_tfp_K6.
  if Parameters.fluidModel == 'Ideal-Gas':
    Np_Factor = self.dp_fdX*self.khat*((self.a_s - Parameters.Gravity)/(Parameters.RGas*self.tf) + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  elif Parameters.fluidModel == 'Exponential-Thermal':
    Np_Factor = self.dp_fdX*self.khat*(self.rhofR*(self.a_s - Parameters.Gravity)/Parameters.KF + \
                self.dnfdX*(1 - self.ts/self.tf)/(self.J*self.nf))
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  Bp_Factor = self.dp_fdX*self.khat/self.J - self.vDarcy
  
  self.K_tfp_K6 = np.einsum('ik, jk, k', self.Ntf, Np_Factor*self.Np + Bp_Factor*self.Bp, \
                            self.weights)

  if Parameters.isDynamics:
    self.K_tfp_K6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfp_K6 *= Parameters.gamma*Parameters.dt

  self.K_tfp_K6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfts_K2(self, Parameters):
  # Compute K_tfts_K2.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tfts_K2 = np.einsum('ik, jk, k', self.Ntf, self.Nts, (Parameters.cvf + Parameters.RGas)*\
                               (self.khat/self.J)*((self.p_f)**2/self.nf)*self.weights/\
                               (Parameters.RGas*(self.tf**2)))
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tfts_K2 = np.einsum('ik, jk, k', self.Ntf, self.Nts, (self.rhofR*Parameters.cvf + \
                               Parameters.KF*(Parameters.Af**2)*self.tf)*self.p_f*self.khat*\
                               self.dnfdX*self.dtfdX*self.weights/(self.J*self.nf*self.tf))
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
  
  if Parameters.isDynamics:
    self.K_tfts_K2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfts_K2 *= Parameters.gamma*Parameters.dt

  self.K_tfts_K2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfts_K4(self, Parameters):
  # Compute K_tfts_K4.
  self.K_tfts_K4 = -np.einsum('ik, jk, k', self.Ntf, self.Nts, (self.khat/(self.J*self.tf))*\
                              ((self.p_f*self.dnfdX/self.nf)**2)*self.weights)

  if Parameters.isDynamics:
    self.K_tfts_K4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfts_K4 *= Parameters.gamma*Parameters.dt

  self.K_tfts_K4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfts_K6(self, Parameters):
  # Compute K_tfts_K6.
  self.K_tfts_K6 = -np.einsum('ik, jk, k', self.Ntf, self.Nts, self.dp_fdX*self.dnfdX*\
                              (self.khat*self.p_f/(self.J*self.nf*self.tf))*self.weights)

  if Parameters.isDynamics:
    self.K_tfts_K6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfts_K6 *= Parameters.gamma*Parameters.dt

  self.K_tfts_K6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tfts_K8(self, Parameters):
  # Compute K_tfts_K8.
  self.K_tfts_K8 = -np.einsum('ik, jk, k', self.Ntf, self.Nts, Parameters.k_exchange*self.J*self.weights)

  if Parameters.isDynamics:
    self.K_tfts_K8 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tfts_K8 *= Parameters.gamma*Parameters.dt

  self.K_tfts_K8 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tftf_K2(self, Parameters):
  # Compute K_tftf_K2.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tftf_K2 = np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.cvf + Parameters.RGas)*\
                               (1/self.tf)*self.dtfdX*(self.khat*(self.rhofR**2)*(self.a_s - \
                               Parameters.Gravity - (Parameters.RGas*self.ts)/(self.J*self.nf)*\
                               self.dnfdX) - self.vDarcy*self.p_f/(Parameters.RGas*self.tf))*\
                               self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tftf_K2 = np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.Af*(Parameters.KF*\
                               Parameters.Af - self.rhofR*Parameters.cvf)*self.vDarcy + \
                               (self.rhofR*Parameters.cvf + Parameters.KF*(Parameters.Af**2)*\
                               self.tf)*self.khat*(self.rhofR*Parameters.Af*(self.a_s - \
                               Parameters.Gravity) - self.p_f*self.ts*self.dnfdX/(self.J*\
                               self.nf*(self.tf**2))))*self.dtfdX*self.weights) 
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tftf_K2 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tftf_K2 *= Parameters.gamma*Parameters.dt

  self.K_tftf_K2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tftf_K4(self, Parameters):
  # Compute K_tftf_K4.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tftf_K4 = -np.einsum('ik, jk, k', self.Ntf, self.Ntf, (Parameters.RGas*self.khat/self.nf)*\
                                (self.rhofR**2)*self.dnfdX*((self.a_s - Parameters.Gravity) -\
                                self.dnfdX*Parameters.RGas*self.ts/(self.J*self.nf))*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tftf_K4 = -np.einsum('ik, jk, k', self.Ntf, self.Ntf, self.khat*self.dnfdX/self.nf*\
                                (self.rhofR*Parameters.Af*(self.a_s - Parameters.Gravity) - \
                                self.p_f*self.ts*self.dnfdX/(self.J*self.nf*(self.tf**2)))*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tftf_K4 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tftf_K4 *= Parameters.gamma*Parameters.dt

  self.K_tftf_K4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tftf_K6(self, Parameters):
  # Compute K_tftf_K6.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_tftf_K6 = -np.einsum('ik, jk, k', self.Ntf, self.Ntf, self.dp_fdX*self.khat*\
                                (self.rhofR/self.tf)*(self.a_s - Parameters.Gravity - \
                                self.dnfdX*Parameters.RGas*self.ts/(self.J*self.nf))*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_tftf_K6 = -np.einsum('ik, jk, k', self.Ntf, self.Ntf, self.dp_fdX*self.khat*\
                                (self.rhofR*Parameters.Af*(self.a_s - Parameters.Gravity) - \
                                self.p_f*self.ts*self.dnfdX/(self.J*self.nf*(self.tf**2)))*self.weights)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")

  if Parameters.isDynamics:
    self.K_tftf_K6 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tftf_K6 *= Parameters.gamma*Parameters.dt

  self.K_tftf_K6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tftf_K7(self, Parameters):
  # Compute K_tftf_K7.
  self.K_tftf_K7 = np.einsum('ik, jk, k', self.Btf, self.Btf, (self.nf/self.J)*Parameters.kf*\
                             self.weights)

  if Parameters.isDynamics:
    self.K_tftf_K7 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tftf_K7 *= Parameters.gamma*Parameters.dt

  self.K_tftf_K7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K_tftf_K8(self, Parameters):
  # Compute K_tftf_K8.
  self.K_tftf_K8 = np.einsum('ik, jk, k', self.Ntf, self.Ntf, Parameters.k_exchange*self.J*self.weights)

  if Parameters.isDynamics:
    self.K_tftf_K8 *= Parameters.beta*(Parameters.dt**2)
  else:
    self.K_tftf_K8 *= Parameters.gamma*Parameters.dt

  self.K_tftf_K8 *= Parameters.Area*self.Jacobian
  return 
#------------------------
# End stiffness matrices.
#------------------------
#-----------------------
# End low-level methods.
#-----------------------
