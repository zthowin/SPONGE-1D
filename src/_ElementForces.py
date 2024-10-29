#----------------------------------------------------------------------------------------
# Module housing element object internal force vectors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 25, 2024
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
def compute_internal_forces(self, Parameters, VariationalEq=None):
  # Helper function to compute the correct internal forces
  # depending on the physics.
  if VariationalEq == 'G':
    self.get_G_Forces(Parameters)
  elif VariationalEq == 'H':
    self.get_H_Forces(Parameters)
  elif VariationalEq == 'I':
    self.get_I_Forces(Parameters)
  elif VariationalEq == 'J':
    self.get_J_Forces(Parameters)
  elif VariationalEq == 'K':
    self.get_K_Forces(Parameters)
  else:
    sys.exit("------\nERROR:\n------\nVariational equation not recognized, check source code.")
  return

@register_method
def get_G_Forces(self, Parameters):
  # Sum the internal force vectors for the variational equation of
  # the balance of momentum of the solid.
  if Parameters.integrationScheme == 'Newmark-beta':# or Parameters.integrationScheme == 'Predictor-corrector' :
    # try:
    #   if not Parameters.StarStar:
    #     self.get_G1(Parameters)
    #   else:
    #     self.G_1 = np.zeros(Parameters.GaussD)
    # except AttributeError:
    self.get_G1(Parameters)
  elif 'RK' in Parameters.integrationScheme:
    if 'uf' in Parameters.Physics:
      self.get_G1(Parameters)
    else:
      self.G_1 = np.zeros(Parameters.ndofSe)
  else:
    self.G_1 = np.zeros(Parameters.ndofSe)

  if Parameters.Physics == 'u' or Parameters.Physics == 'u-t':
    self.G_3 = np.zeros(Parameters.ndofSe)
  else:
    self.get_G3(Parameters)
    
  self.get_G2(Parameters)
  self.get_G4(Parameters)

  try:
    self.G_int = self.G_1 + self.G_2 + self.G_3 + self.G_4
    if 'uf' in Parameters.Physics:
      if Parameters.DarcyBrinkman:
        self.get_G5(Parameters)
        self.G_int += self.G_5
#      self.get_G6(Parameters) # Pore fluid shock viscosity
#     self.G_int += self.G_6
    if Parameters.MMS:
      self.get_GMMS(Parameters)
      self.G_int += self.G_MMS
  except FloatingPointError:
    if 'pf' in params.Physics:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Pore fluid pressure =", self.p_f)
      print("Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    else:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_H_Forces(self, Parameters):
  # Sum the internal force vectors for the variational equation of
  # the balance of mass of the mixture.
  if Parameters.fluidModel == 'Ideal-Gas':
    self.HDiv = self.p_f
  elif 'Exponential' in Parameters.fluidModel:
    self.HDiv = Parameters.KF
  elif 'Isentropic' in Parameters.fluidModel:
    self.HDiv = 1.4*self.p_f 
  elif 'Linear-Bulk' in Parameters.fluidModel:
    self.HDiv = (self.J/Parameters.KF)**(-1)
  elif 'Linear-IC' in Parameters.fluidModel:
    self.HDiv = (self.J/Parameters.p_f0)**(-1)
  else:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive relation not recognized.")
    raise RuntimeError

  self.get_H1(Parameters)
  self.get_H2(Parameters)
  self.get_H3(Parameters)
  self.get_H4(Parameters)

  try:
    self.H_int = self.H_1 + self.H_2 + self.H_3 + self.H_4
    if Parameters.alpha_stab > 0:
      self.get_HStab(Parameters)
      self.H_int += self.H_Stab
    if Parameters.DarcyBrinkman:
      self.get_H5(Parameters)
      self.H_int += self.H_5
    if 'tf' in Parameters.Physics:
      self.get_H6(Parameters)
      self.get_H7(Parameters)
      self.H_int += self.H_6 + self.H_7
    if Parameters.MMS:
      self.get_HMMS(Parameters)
      self.H_int += self.H_MMS
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =", self.p_f)
    print("Pressure instability in H; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_I_Forces(self, Parameters):
  # Sum the internal force vectors for the variational equation of
  # the balance of momentum of the fluid.
  if Parameters.integrationScheme == 'Newmark-beta':
    self.get_I1(Parameters)
  else:
    self.I_1 = np.zeros(Parameters.ndofFe)
  self.get_I2(Parameters)
  self.get_I3(Parameters)
  self.get_I4(Parameters)
#  self.get_I7(Parameters) # Pore fluid shock viscosity

  try:
    self.I_int = self.I_1 + self.I_2 + self.I_3 + self.I_4
    # + self.I_7
    if Parameters.DarcyBrinkman:
      self.get_I5(Parameters)
      self.I_int += self.I_5
    if Parameters.Physics == 'u-uf-pf-ts-tf':
      self.get_I6(Parameters)
      self.I_int += self.I_6
    if Parameters.MMS:
      self.get_IMMS(Parameters)
      self.I_int += self.I_MMS
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =", self.p_f)
    print("Pressure instability in I; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  return

@register_method
def get_J_Forces(self, Parameters):
  # Sum the internal force vectors for the variational equation of
  # the balance of energy of the solid.
  self.get_J2(Parameters)
  self.get_J3(Parameters)
  try:
    self.J_int = self.J_2 + self.J_3
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Solid temperature =", self.ts)
    print("Solid temperature instability in J; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  if 'RK' not in Parameters.integrationScheme:
    self.get_J1(Parameters)
    try:
      self.J_int += self.J_1
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Solid temperature rate =", self.tsDot)
      print("Solid temperature instability in J; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  if 'tf' in Parameters.Physics:
    self.get_J4(Parameters)
    self.get_J5(Parameters)
    self.get_J6(Parameters)
    self.get_J7(Parameters)
    try:
      self.J_int += self.J_3 + self.J_4 + self.J_5 + self.J_6 + self.J_7
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Solid temperature =", self.ts)
      print("Solid temperature instability in J; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  return

@register_method
def get_K_Forces(self, Parameters):
  # Sum the internal force vectors for the variational equation of
  # the balance of energy of the pore fluid.
  self.get_K2(Parameters)
  self.get_K3(Parameters)
  self.get_K4(Parameters)
  self.get_K7(Parameters)
  self.get_K8(Parameters)

  try:
    self.K_int = self.K_2 + self.K_3 + self.K_4 + self.K_7 + self.K_8
    if 'uf' in Parameters.Physics and Parameters.fluidModel == 'Exponential-Thermal':
      self.get_K5(Parameters)
      self.K_int += self.K_5
    if 'uf' not in Parameters.Physics:
      self.get_K5(Parameters)
      self.get_K6(Parameters)
      self.K_int += self.K_5 + self.K_6
    if Parameters.SUPG:
      self.get_KStab(Parameters)
      self.K_int += self.K_Stab
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Fluid temperature =", self.tf)
    print("Fluid temperature instability in K; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  if 'RK' not in Parameters.integrationScheme:
    self.get_K1(Parameters)
    try:
      self.K_int += self.K_1
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Fluid temperature rate =", self.tfDot)
      print("Fluid temperature instability in K; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
      raise FloatingPointError
  return

@register_method
def get_G1(self, Parameters):
  # Compute G_1^INT (acceleration contribution to G).
  if 'uf' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta':
      self.G_1 = np.einsum('ik, k', self.Nu, (self.rhos_0*self.a_s + self.rhof_0*self.a_f)*self.weights)
    else:
      self.G_1 = np.einsum('ik, k', self.Nu, (self.rhof_0*self.a_f)*self.weights)
  else:
    self.G_1 = np.einsum('ik, k', self.Nu, self.rho_0*self.a_s*self.weights)
  self.G_1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_G2(self, Parameters):
  # Compute G_2^INT (effective stress contribution to G).
  # try: # Predictor-corrector
  #   if Parameters.StarStar:
  #     self.G_2 = np.zeros(Parameters.ndofSe)
  #   else:
  #     self.G_2  = np.einsum('ik, k', self.Bu, self.P11*self.weights)
  # except AttributeError:
  self.G_2  = np.einsum('ik, k', self.Bu, self.P11*self.weights)
  self.G_2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_G3(self, Parameters):
  # Compute G_3^INT (pore fluid pressure contribution to G).
  try:
    # try: # Predictor-corrector
    #   if Parameters.StarStar:
    #     self.G_3 = np.einsum('ik,k', self.Bu, (self.p_f - self.p_fLast)*self.weights*Parameters.Biot)
    #   else:
    #     self.G_3 = np.einsum('ik, k', self.Bu, (-self.p_f)*self.weights*Parameters.Biot)
    # except AttributeError:
    if 'tf' in Parameters.Physics:
      self.G_3 = -np.einsum('ik, k', self.Bu, self.p_f*(self.ns*self.ts/self.tf + self.nf)*self.weights)
    else:
      self.G_3 = -np.einsum('ik, k', self.Bu, self.p_f*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in G_3; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.G_3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_G4(self, Parameters):
  # Compute G_4^INT (gravitational/body force contribution to G).
  self.G_4  = -np.einsum('ik, k', self.Nu, self.weights*Parameters.Gravity)
  self.G_4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_G5(self, Parameters):
  # Compute G_5^INT (pore fluid viscous tensor contribution to G).
  self.G_5  = np.einsum('ik, k', self.Bu, self.FES*self.weights)
  self.G_5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_G6(self, Parameters):
  # Compute G_6^INT (pore fluid shock viscosity contribution to G).
  self.G_6  = -np.einsum('ik, k', self.Bu, self.Qf*self.weights)
  self.G_6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_GMMS(self, Parameters):
  # Compute f_mms,u (the manufactured r.h.s. contribution to G).
  if Parameters.Physics == 'u':
    if Parameters.MMS_SolidSolutionType == 'S2T2':
      self.f_mmsu = -2*Parameters.Gravity + 4*self.X**2\
                    - 2*(Parameters.tk**2)*(1 + (2 - np.log(1 + 2*self.X*Parameters.tk**2))/(1 + 2*self.X*Parameters.tk**2)**2)
    
    elif Parameters.MMS_SolidSolutionType == 'S2T3':
      self.f_mmsu = -2*Parameters.Gravity + 12*Parameters.tk*self.X**2\
                    - 2*(Parameters.tk**3)*(1 + (2 - np.log(1 + 2*self.X*Parameters.tk**3))/(1 + 2*self.X*Parameters.tk**3)**2)

    elif Parameters.MMS_SolidSolutionType == 'MS2T3':
      self.f_mmsu = -2*Parameters.Gravity - 12*Parameters.tk*self.X**2\
                    + 2*(Parameters.tk**3)*(1 + (2 - np.log(1 - 2*self.X*Parameters.tk**3))/(1 - 2*self.X*Parameters.tk**3)**2)

    elif Parameters.MMS_SolidSolutionType == 'S3T2':
      self.f_mmsu = -2*Parameters.Gravity + 4*self.X**3\
                    - 6*self.X*(Parameters.tk**2)*(1 + (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**2)))/(1 + 3*(self.X**2)*(Parameters.tk**2))**2)

    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      self.f_mmsu = -2*Parameters.Gravity + 12*Parameters.tk*self.X**3\
                    - 6*self.X*(Parameters.tk**3)*(1 + (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**3)))/(1 + 3*(self.X**2)*(Parameters.tk**3))**2)

    elif Parameters.MMS_SolidSolutionType == 'MS3T3':
      self.f_mmsu = -2*Parameters.Gravity - 12*Parameters.tk*self.X**3\
                    + 6*self.X*(Parameters.tk**3)*(1 + (2 - np.log(1 - 3*(self.X**2)*(Parameters.tk**3)))/(1 - 3*(self.X**2)*(Parameters.tk**3))**2)
 
    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      self.f_mmsu = -2*Parameters.Gravity + 12*Parameters.tk*self.X**4\
                    - 12*(self.X**2)*(Parameters.tk**3)*(1 + (2 - np.log(1 + 4*(self.X**3)*(Parameters.tk**3)))/(1 + 4*(self.X**3)*(Parameters.tk**3))**2)

    elif Parameters.MMS_SolidSolutionType == 'MS4T3':
      self.f_mmsu = -2*Parameters.Gravity - 12*Parameters.tk*self.X**4\
                    + 12*(self.X**2)*(Parameters.tk**3)*(1 + (2 - np.log(1 - 4*(self.X**3)*(Parameters.tk**3)))/(1 - 4*(self.X**3)*(Parameters.tk**3))**2)

  elif Parameters.Physics == 'u-pf':
    if Parameters.MMS_SolidSolutionType == 'S2T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          a = (Parameters.tk**2)*(self.X - Parameters.H0)*(0.25 + self.X*(Parameters.tk**3))

          self.f_mmsu = -(Parameters.tk**2) - 12*Parameters.tk*(self.X**2)*(a - 0.5*Parameters.KF)/Parameters.KF\
                        + Parameters.Gravity*(2*a/Parameters.KF - 1)\
                        - 2*(Parameters.tk**3)*(1 + (2 - np.log(1 + 2*self.X*Parameters.tk**3))/(1 + 2*self.X*Parameters.tk**3)**2)

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**2 * Parameters.tk**3
          a = 0.25 + h
          b = 0.50 + h
          c = 0.50 + 2*h/self.X
          d = Parameters.H0 - self.X
          e = 1 + 0.0625/(a**2) + 0.25/(b**2)
      
          self.f_mmsu = -Parameters.tk**2 - 2*(Parameters.tk**3)*e\
                        + 12*Parameters.tk*(self.X**2)*((Parameters.tk**2)*d*a\
                                                        + 0.5*Parameters.KF)/Parameters.KF\
                        - Parameters.Gravity*(1 + (Parameters.tk**2)*d*c/Parameters.KF)

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          a = (Parameters.tk**2)*((self.X - Parameters.H0)*(3*(Parameters.tk**3)*(self.X**3) + 0.5))/Parameters.KF

          self.f_mmsu = -(Parameters.tk**2) + (Parameters.Gravity - 6*Parameters.tk*(self.X**3))*(a - 1)\
                        -6*self.X*(Parameters.tk**3)*(1 + (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**3)))/((1 + 3*(self.X**2)*(Parameters.tk**3))**2))

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**2 * Parameters.tk**3
          a = 0.5 + 3*h
          b = 1/6 + h
          c = 1/3 + h
          d = Parameters.H0 - self.X
          
          self.f_mmsu = -Parameters.Gravity*(1 + (Parameters.tk**2)*d*a/Parameters.KF)\
                        + Parameters.tk*(6*(self.X**3) + 18*((Parameters.tk*self.X)**5)*d/Parameters.KF\
                                         + (Parameters.tk**2)*self.X*\
                                           (3*Parameters.H0*(self.X**2)/Parameters.KF - 3*(self.X**3)/Parameters.KF\
                                            - 6 - (1/6)/(b**2) - (2/3)/(c**2)) - Parameters.tk)
                                                                    
        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          a = (Parameters.tk**2)*((self.X - Parameters.H0)*(0.5 + 4*(self.X**3)*(Parameters.tk**3)))/Parameters.KF
          
          self.f_mmsu = -(Parameters.tk**2) + (Parameters.Gravity - 6*(self.X**4)*Parameters.tk)*(a-1)\
                        - 12*(self.X**2)*(Parameters.tk**3)*(1 + (2 - np.log(1 + 4*(self.X**3)*(Parameters.tk**3)))/(1 + 4*(self.X**3)*(Parameters.tk**3))**2)

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**3 * Parameters.tk**3
          a = 0.5   + 4*h
          b = 0.125 + h
          c = 0.25  + h
          d = Parameters.H0 - self.X
          
          self.f_mmsu = -Parameters.Gravity*(1 + (Parameters.tk**2)*d*a/Parameters.KF)\
                        + Parameters.tk*(6*(self.X**4) + 24*(Parameters.tk**5)*(self.X**7)*d/Parameters.KF\
                                         + (Parameters.tk**2)*(self.X**2)*\
                                           (3*Parameters.H0*(self.X**2)/Parameters.KF - 3*(self.X**3)/Parameters.KF\
                                            - 12 - 0.1875/(b**2) - 0.75/(c**2)) - Parameters.tk)
                                                                    
        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError
  
  elif Parameters.Physics == 'u-uf-pf':
    if Parameters.MMS_SolidSolutionType == 'S2T3':
      if Parameters.MMS_FluidSolutionType == 'S2T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            self.f_mmsu = -Parameters.tk**2 + 6*Parameters.tk*(self.X**2)\
                          + (Parameters.H0 - self.X)*(0.5 + (self.X**2)*Parameters.tk**3)\
                             *(3*(self.X**2)*(Parameters.tk**3)\
                               - Parameters.Gravity*Parameters.tk**2)/Parameters.KF\
                          - Parameters.Gravity\
                          - 2*(Parameters.tk**3)*(1 + (2 - np.log(1 + 2*self.X*(Parameters.tk**3))/(1 + 2*self.X*(Parameters.tk**3))**2))

          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            h = self.X**2 * Parameters.tk**3
            a = 0.25 + h
            b = 0.50 + h
            c = 0.5  + 2*h/self.X
            d = Parameters.H0 - self.X
            e = 1 + 0.0625/(a**2) + 0.25/(b**2)
        
            self.f_mmsu = -Parameters.tk**2 + 6*Parameters.tk*(self.X**2)\
                          + 6*h*d*a/Parameters.KF - 2*(Parameters.tk**3)*e\
                          - Parameters.Gravity*(1 + (Parameters.tk**2)*d*c/Parameters.KF)

          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError
    
    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      if Parameters.MMS_FluidSolutionType == 'S3T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            self.f_mmsu = -Parameters.tk**2 + 6*Parameters.tk*(self.X**3)\
                          + 9*(Parameters.tk**3)*(self.X**3)*(Parameters.H0 - self.X)*\
                            (1/6 + (Parameters.tk**3)*(self.X**2))/Parameters.KF\
                          - Parameters.Gravity*(1 + (Parameters.tk**2)*(Parameters.H0 - self.X)*\
                                                    (0.5 + 3*(Parameters.tk**3)*(self.X**2))/Parameters.KF)\
                          - 6*self.X*(Parameters.tk**3)*(1 + (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**3))/(1 + 3*(self.X**2)*(Parameters.tk**3))**2))
          
          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            h = self.X**2 * Parameters.tk**3
            a = 1/6 + h
            b = 1/3 + h
            c = 0.5 + 3*h
            d = Parameters.H0 - self.X
            e = 1 + (5/18)/(a**2) + (1/9)/(b**2)
            
            self.f_mmsu = -Parameters.tk**2 + 6*Parameters.tk*(self.X**3)\
                          + 9*((self.X*Parameters.tk)**3)*d*a/Parameters.KF\
                          - 6*(Parameters.tk**3)*self.X*e\
                          - Parameters.Gravity*(1 + (Parameters.tk**2)*d*c/Parameters.KF)

          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError
        
        elif Parameters.MMS_PressureSolutionType == 'S1T3':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            self.f_mmsu = -Parameters.tk**3 + 6*Parameters.tk*(self.X**3)\
                          + 9*(Parameters.tk**4)*(self.X**3)*(Parameters.H0 - self.X)*(1/6 + (Parameters.tk**3)*(self.X**2))/Parameters.KF\
                          + Parameters.Gravity*((Parameters.tk**3)*((0.5 + 3*(Parameters.tk**3)*(self.X**2))*(self.X - Parameters.H0))/Parameters.KF - 1)\
                          - 6*self.X*(Parameters.tk**3)*(1 +  (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**3))/(1 + 3*(self.X**2)*(Parameters.tk**3))**2))

          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

        elif Parameters.MMS_PressureSolutionType == '2S1T3':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            self.f_mmsu = -Parameters.tk**3 + 6*Parameters.tk*(self.X**3)\
                                    + 9*(Parameters.tk**4)*(self.X**3)*(2*Parameters.H0 - self.X)*(1/6 + (Parameters.tk**3)*(self.X**2))/Parameters.KF\
                                    + Parameters.Gravity*(0.5*(Parameters.tk**3)*((self.X - 2*Parameters.H0)*(6*(self.X**2)*(Parameters.tk**3) + 1))/Parameters.KF - 1)\
                                    - 6*self.X*(Parameters.tk**3)*(1 +  (2 - np.log(1 + 3*(self.X**2)*(Parameters.tk**3))/(1 + 3*(self.X**2)*(Parameters.tk**3))**2))
          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      if Parameters.MMS_FluidSolutionType == 'S4T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            self.f_mmsu = -Parameters.tk**2 + 6*Parameters.tk*(self.X**4)\
                          -Parameters.Gravity*(1 + (Parameters.tk**2)*(0.5 + 4*(self.X**3)*(Parameters.tk**3))*\
                          (Parameters.H0 - self.X)/Parameters.KF)\
                          - 12*(self.X**2)*(Parameters.tk**3)*(1 + (2 - np.log(1 + 4*(self.X**3)*(Parameters.tk**3))/(1 + 4*(self.X**3)*(Parameters.tk**3))**2))

          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            self.f_mmsu = -Parameters.tk**2 + 6*(self.X**4)*Parameters.tk\
                          + 12*(Parameters.tk**3)*(Parameters.H0 - self.X)*\
                            (self.X**4)*(0.125 + (self.X**3)*(Parameters.tk**3))/Parameters.KF\
                          - 12*(Parameters.tk**3)*(self.X**2)*\
                            (1 + .015625/((0.125 + (self.X**3)*(Parameters.tk**3))**2)\
                             + 0.0625/((0.25 + (self.X**3)*(Parameters.tk**3)**2)))\
                          - Parameters.Gravity*(1 + (Parameters.tk**2)*\
                            (0.5 + 4*(Parameters.tk**3)*(self.X**3))*\
                            (Parameters.H0 - self.X)/Parameters.KF)

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError
  try:
    self.G_MMS  = -np.einsum('ik, k', self.Nu, self.f_mmsu*self.weights)
  except AttributeError:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nInvalid MMS solution combination.")
    raise RuntimeError
  self.G_MMS *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H1(self, Parameters):
  # Compute H_1^INT (time derivative on pressure & deformation
  # contribution to H).
  try:
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':# or Parameters.integrationScheme == 'Predictor-corrector':
      self.H_1 = np.einsum('ik, k', self.Np, (self.J*self.nf*self.p_fDot/self.HDiv + self.JDot)*self.weights)
    elif Parameters.integrationScheme == 'Central-difference':
      self.get_dadX()
      self.H_1 = np.einsum('ik, k', self.Np, ((self.J*self.nf/Parameters.KF)*(self.p_fDot + (Parameters.dt/2)*self.p_fDDot) +\
                                              (self.dvdX + (Parameters.dt/2)*self.dadX))*self.weights)
    else:
      self.H_1 = np.einsum('ik, k', self.Np, self.JDot*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure =               ", self.p_f)
    print("Pore fluid pressure per unit time = ", self.p_fDot)
    print("Pressure instability in H_1; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H2(self, Parameters):
  # Compute H_2^INT (Darcy's law contribution to H).
  try:
    if Parameters.integrationScheme == 'Central-difference':
      self.H_2 = -np.einsum('ik, k', self.Np, self.dp_fdX*(self.khat*(self.dp_fdX/self.F11 - self.rhofR*Parameters.Gravity))*self.weights/Parameters.KF)
    else:
      self.H_2 = np.einsum('ik, k', self.Np, self.dp_fdX*self.vDarcy*self.weights/self.HDiv)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H3(self, Parameters):
  # Compute H_3^INT (Darcy's law pore fluid pressure gradient contribution to H).
  try:
    self.H_3  = np.einsum('ik, k', self.Bp, self.khat*self.dp_fdX*self.weights/self.F11)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_3; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H4(self, Parameters):
  # Compute H_4^INT (Darcy's law inertial contribution to H).
  try:
    if Parameters.integrationScheme == 'Central-difference':
      self.H_4 = -np.einsum('ik, k', self.Bp, self.khat*self.rhofR*Parameters.Gravity*self.weights)
    else:
      if 'uf' in Parameters.Physics:
        self.H_4 = np.einsum('ik, k', self.Bp, self.khat*self.rhofR*(self.a_f - Parameters.Gravity)*self.weights)
      else:
        # if Parameters.integrationScheme == 'Predictor-corrector':
        #   self.H_4 = np.einsum('ik, k', self.Bp, self.khat*self.rhofR*(self.a_s - Parameters.Gravity - self.dp_fDotdX*Parameters.dt/self.rho_0)*self.weights)
        # else:
        self.H_4 = np.einsum('ik, k', self.Bp, self.khat*self.rhofR*(self.a_s - Parameters.Gravity)*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_4; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H5(self, Parameters):
  # Compute H_5^INT (Darcy-Brinkman contribution to H).
  try:
    self.H_5  = -np.einsum('ik, k', self.Bp, self.khat*(self.DIV_FES/(self.nf*self.F11))*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_5; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H6(self, Parameters):
  # Compute H_6^INT (Darcy's law thermal contribution to H).
  try:
    self.H_6 = np.einsum('ik, k', self.Bp, (self.khat/self.nf)*self.p_f*self.dnfdX*(1 - self.ts/self.tf)*self.weights/self.F11)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_6; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.H_6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_H7(self, Parameters):
  # Compute H_7^INT (pore fluid temperature contribution to H).
  try:
    if Parameters.fluidModel == 'Ideal-Gas':
      self.H_7 = -np.einsum('ik, k', self.Np, (1/self.tf)*(self.J*self.nf*self.tfDot + self.dtfdX*self.vDarcy)*self.weights)
    else:
      self.H_7 = -np.einsum('ik, k', self.Np, Parameters.Af*(self.J*self.nf*self.tfDot + self.dtfdX*self.vDarcy)*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Rate of fluid temperature = ", self.tfDot)
    print("Fluid temperature instability in H_7; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError

  self.H_7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_HStab(self, Parameters):
  # Compute H_Stab (stabilization contribution to H).
  try:
    if Parameters.integrationScheme == 'Central-difference':
      self.H_Stab = np.einsum('ik, k', self.Bp, Parameters.alpha_stab*(self.dp_fDotdX + self.dp_fDDotdX*Parameters.dt/2)*self.weights/self.F11)
    elif Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      self.H_Stab = np.einsum('ik, k', self.Bp, Parameters.alpha_stab*self.dp_fDotdX*self.weights/self.F11)
    else:
      self.H_Stab = 0
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in H_Stab; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
  
  self.H_Stab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_HMMS(self, Parameters):
# Compute f_mms,pf (the manufactured source term for H_INT).
  if Parameters.Physics == 'u-pf':
    if Parameters.MMS_SolidSolutionType == 'S2T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          f  = self.X*Parameters.tk**3
          ff = self.X*Parameters.tk
          a  = 0.250  + f
          b  = 0.500  + f
          bb = 1 - 0.25/b
          c  = 0.1875 + 0.5*f
          j  = 0.375 + f
          d  = Parameters.H0 - self.X
          e  = 1/12 + (1/6)*f

          g = (Parameters.Gravity + 12*Parameters.H0*Parameters.tk*self.X - 18*Parameters.tk*(self.X**2))/Parameters.KF\
              + 0.5*(Parameters.tk**3)/(b**2)
          h = (1/12)*Parameters.KF + Parameters.Gravity*d*e - Parameters.tk*(self.X**2)*d*b
          i = 6*d*(self.X*ff - Parameters.Gravity/6)/Parameters.KF - 0.5/b

          self.f_mmsp = Parameters.tk*(6*ff + 4*d*a/Parameters.KF - 6*Parameters.intrPerm*Parameters.tk*(b**2)*g*(bb**3)/c\
                                       + 6*Parameters.intrPerm*(Parameters.tk**3)*(b**2)*(bb**3)*i/(Parameters.KF*c)\
                                       + 72*Parameters.intrPerm*(Parameters.tk**4)*(a**4)*h/(Parameters.KF*(j**2)*(b**3))\
                                       + 54*Parameters.intrPerm*(Parameters.tk**4)*(a**2)*h/(Parameters.KF*j*(b**3)))

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X * Parameters.tk**3
          a = 0.25 + h
          b = 0.50 + h
          d = Parameters.H0 - self.X
          c = 6*d*(-Parameters.Gravity/6 + self.X**2 * Parameters.tk)/Parameters.KF - 0.5/b
          e = Parameters.Gravity/Parameters.KF + 12*Parameters.H0*Parameters.tk*self.X/Parameters.KF\
              - 18*Parameters.tk*(self.X**2)/Parameters.KF + 0.5*(Parameters.tk**3)/(b**2)
          
          self.f_mmsp = Parameters.tk*(6*Parameters.tk*self.X + 8*d*a*b/Parameters.KF\
                                       - Parameters.tk*Parameters.intrPerm*((4*a)**Parameters.kappa)*e\
                                       - 2*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                                         Parameters.intrPerm*((2*a)**(Parameters.kappa - 1))*c\
                                       + (Parameters.tk**3)*Parameters.intrPerm*(2*b)*((4*a)**Parameters.kappa)*\
                                         c/Parameters.KF)

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          a = 1/6  + (self.X**2)*(Parameters.tk**3)
          b = 1/3  + (self.X**2)*(Parameters.tk**3)
          c = 0.25 + (self.X**2)*(Parameters.tk**3)
          d = Parameters.H0 - self.X
          f = self.X*(Parameters.tk**3)
          h = 1/12 + (1/3)*self.X*f
          e = (1/18)*Parameters.KF + f*((self.X**2)*f - Parameters.H0*self.X*f - (1/3)*d)\
              + Parameters.Gravity*(Parameters.H0*((1/18) + (1/6)*self.X*f) - (1/18)*self.X - (1/6)*(self.X**2)*f)
          g = Parameters.Gravity/Parameters.KF + Parameters.tk*self.X*((18*Parameters.H0*self.X - 24*(self.X**2))/Parameters.KF + (2*(Parameters.tk**2)/3)/(b**2))

          self.f_mmsp = Parameters.tk*(9*Parameters.tk*(self.X**2) + 6*d*a/Parameters.KF\
                                       + 6*Parameters.intrPerm*(Parameters.tk**3)*(b**2)*((1 - 1/(6*b))**3)*((6*d/Parameters.KF)*(Parameters.tk*(self.X**3) - (1/6)*Parameters.Gravity)\
                                                                                         - 1/(3*b))/(Parameters.KF*h)\
                                       + 216*Parameters.intrPerm*f*Parameters.tk*(a**4)*e/(Parameters.KF*(c**2)*(b**3))\
                                     + 108*Parameters.intrPerm*f*Parameters.tk*(a**2)*e/(Parameters.KF*(c*(b**3)))\
                                     - 6*Parameters.intrPerm*Parameters.tk*(b**2)*((1 - 1/(6*b))**3)*g/h)
        
        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**2 * Parameters.tk**3
          d = Parameters.H0 - self.X
          a = 1/6 + h
          b = 1/3 + h
          c = 6*d*((self.X**3)*Parameters.tk - Parameters.Gravity/6)/Parameters.KF - (1/3)/b
          e = Parameters.Gravity/Parameters.KF\
              + self.X*Parameters.tk*(18*Parameters.H0*self.X/Parameters.KF - 24*(self.X**2)/Parameters.KF\
                                      + (2/3)*(Parameters.tk**2)/(b**2))
          f = 0.5 + 3*h
          g = 1.0 + 6*h
          i = 1.0 + 3*h

          self.f_mmsp = Parameters.tk*(9*Parameters.tk*(self.X**2) + 18*d*a*b/Parameters.KF\
                                       - 6*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                                         Parameters.intrPerm*self.X*(f**(Parameters.kappa - 1))*c\
                                       + (Parameters.tk**3)*Parameters.intrPerm*i*(g**Parameters.kappa)*\
                                         c/Parameters.KF\
                                       - Parameters.tk*Parameters.intrPerm*(g**Parameters.kappa)*e)

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          h  = (Parameters.tk**3)*(self.X**3)
          a  = 0.125000 +      h
          b  = 0.187500 +      h
          c  = 0.046875 + 0.25*h
          cc = 0.25 + h
          aa = 1 - 0.125/cc
          d  = Parameters.H0 - self.X
          f  = Parameters.tk*(self.X**4)
          
          e  = (1/24)*Parameters.KF - f*d*cc + Parameters.Gravity*d*(1/24 + (1/6)*h)
          g  = Parameters.Gravity/Parameters.KF + (self.X**2)*Parameters.tk*((24*Parameters.H0*self.X - 30*(self.X**2))/Parameters.KF\
                                                                             + 0.75*(Parameters.tk**2)/(cc**2))

          self.f_mmsp = Parameters.tk*(12*Parameters.tk*(self.X**3) + 8*d*a/Parameters.KF\
                        + 6*Parameters.intrPerm*(Parameters.tk**3)*(cc**2)*(aa**3)*((6*d*(f - g/6))/Parameters.KF - 0.25/cc)/(Parameters.KF*c)\
                        + 432*Parameters.intrPerm*(self.X**2)*(Parameters.tk**4)*(a**4)*e/(Parameters.KF*(b**2)*(cc**3))\
                        + 162*Parameters.intrPerm*(self.X**2)*(Parameters.tk**4)*(a**2)*e/(Parameters.KF*b*(cc**3))\
                        - 6*Parameters.intrPerm*Parameters.tk*(cc**2)*(aa**3)*g/c)

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**3 * Parameters.tk**3
          d = Parameters.H0 - self.X
          a = 0.125 + h
          b = 0.250 + h
          c = 6*d*((self.X**4)*Parameters.tk - Parameters.Gravity/6)/Parameters.KF - 0.25/b
          e = Parameters.Gravity/Parameters.KF\
              + (self.X**2)*Parameters.tk*(24*Parameters.H0*self.X/Parameters.KF - 30*(self.X**2)/Parameters.KF\
                                           + 0.75*(Parameters.tk**2)/(b**2))
          f = 0.5 + 4*h
          g = 1.0 + 8*h
          i = 1.0 + 4*h

          self.f_mmsp = Parameters.tk*(12*Parameters.tk*(self.X**3) + 32*d*a*b/Parameters.KF\
                                       - 12*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                                         Parameters.intrPerm*(self.X**2)*(f**(Parameters.kappa - 1))*c\
                                       + (Parameters.tk**3)*Parameters.intrPerm*i*(g**Parameters.kappa)*\
                                         c/Parameters.KF\
                                       - Parameters.tk*Parameters.intrPerm*(g**Parameters.kappa)*e)

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError
       
  elif Parameters.Physics == 'u-uf-pf':
    if Parameters.MMS_SolidSolutionType == 'S2T3':
      if Parameters.MMS_FluidSolutionType == 'S2T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            f  = self.X*Parameters.tk**3
            ff = self.X*Parameters.tk
            a  = 0.250  + f
            b  = 0.500  + f
            bb = 1 - 0.25/b
            c  = 0.1875 + 0.5*f
            j  = 0.375 + f
            d  = Parameters.H0 - self.X
            e  = 1/6 + (1/3)*f

            g = (Parameters.Gravity + 6*Parameters.H0*Parameters.tk*self.X - 9*Parameters.tk*(self.X**2))/Parameters.KF\
                + 0.5*(Parameters.tk**3)/(b**2)
            h = (1/6)*Parameters.KF + Parameters.Gravity*d*e - Parameters.tk*(self.X**2)*d*b
            i = 3*d*(self.X*ff - Parameters.Gravity/3)/Parameters.KF - 0.5/b

            self.f_mmsp = Parameters.tk*(6*ff + 4*d*a/Parameters.KF - 6*Parameters.intrPerm*Parameters.tk*(b**2)*g*(bb**3)/c\
                                         + 6*Parameters.intrPerm*(Parameters.tk**3)*(b**2)*(bb**3)*i/(Parameters.KF*c)\
                                         + 36*Parameters.intrPerm*(Parameters.tk**4)*(a**4)*h/(Parameters.KF*(j**2)*(b**3))\
                                         + 27*Parameters.intrPerm*(Parameters.tk**4)*(a**2)*h/(Parameters.KF*j*(b**3)))

          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            h = self.X * Parameters.tk**3
            a = 0.25 + h
            b = 0.50 + h
            d = Parameters.H0 - self.X
            c = 3*d*(-Parameters.Gravity/3 + self.X**2 * Parameters.tk)/Parameters.KF - 0.5/b
            e = Parameters.Gravity/Parameters.KF + 6*Parameters.H0*Parameters.tk*self.X/Parameters.KF\
                - 9*Parameters.tk*(self.X**2)/Parameters.KF + 0.5*(Parameters.tk**3)/(b**2)
            
            self.f_mmsp = Parameters.tk*(6*Parameters.tk*self.X + 8*d*a*b/Parameters.KF\
                                         - Parameters.tk*Parameters.intrPerm*((4*a)**Parameters.kappa)*e\
                                         - 2*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                                           Parameters.intrPerm*((2*a)**(Parameters.kappa - 1))*c\
                                         + (Parameters.tk**3)*Parameters.intrPerm*(2*b)*((4*a)**Parameters.kappa)*\
                                           c/Parameters.KF)
                                                 
          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      if Parameters.MMS_FluidSolutionType == 'S3T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            i = self.X**2 * Parameters.tk**3
            a = 1/6  + i
            b = 1/3  + i
            c = 0.25 + i
            d = Parameters.H0 - self.X
            h = 1/12 + (1/3)*i
            e = (1/9)*Parameters.KF - Parameters.tk*(self.X**3)*d*(1/3 + i)\
                + Parameters.Gravity*d*(1/9 + (1/3)*i)
            g = Parameters.Gravity/Parameters.KF + Parameters.tk*self.X*((9*Parameters.H0*self.X - 12*(self.X**2))/Parameters.KF + (2*(Parameters.tk**2)/3)/(b**2))

            self.f_mmsp = Parameters.tk*(9*Parameters.tk*(self.X**2) + 18*d*a*b/Parameters.KF\
                                         + 108*Parameters.intrPerm*self.X*(Parameters.tk**4)*(a**4)*e/\
                                           (Parameters.KF*(c**2)*(b**3))\
                                         + 54*Parameters.intrPerm*self.X*(Parameters.tk**4)*(a**2)*e/\
                                           (Parameters.KF*c*(b**3))\
                                         - 162*Parameters.intrPerm*self.X*(Parameters.tk**3)*(a**3)*e/\
                                           ((Parameters.KF**2)*(1/12 + (7/12)*i + i**2))\
                                         - 6*Parameters.intrPerm*Parameters.tk*(b**2)*((1 - 1/(6*b))**3)*g/h)

          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            i = self.X**2 * Parameters.tk**3
            a = 1/6  + i
            b = 1/3  + i
            c = 1.0  + 6*i
            h = 1.0  + 3*i
            d = Parameters.H0 - self.X
            e = 3*d*(Parameters.tk*(self.X**3) - Parameters.Gravity/3)/Parameters.KF - (1/3)/b
            g = Parameters.Gravity/Parameters.KF + Parameters.tk*self.X*(9*Parameters.H0*self.X/Parameters.KF\
                                                                         - 12*(self.X**2)/Parameters.KF\
                                                                         + (2/3)*(Parameters.tk**2)/(b**2))
          
            self.f_mmsp = Parameters.tk*(9*Parameters.tk*(self.X**2) + 18*d*a*b/Parameters.KF\
                                         - 6*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                                           Parameters.intrPerm*self.X*((c/2)**(Parameters.kappa - 1))*e\
                                         + (Parameters.tk**3)*Parameters.intrPerm*h*(c**Parameters.kappa)*\
                                           e/Parameters.KF\
                                         - Parameters.tk*Parameters.intrPerm*(c**Parameters.kappa)*g)

          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

        elif Parameters.MMS_PressureSolutionType == 'S1T3':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            a = 1/6  + (self.X**2)*(Parameters.tk**3)
            b = 1/3  + (self.X**2)*(Parameters.tk**3)
            c = 0.25 + (self.X**2)*(Parameters.tk**3)
            d = Parameters.H0 - self.X
            f = self.X*(Parameters.tk**3)
            h = 1/9 + (1/3)*(self.X**2)*(Parameters.tk**3)
            j = 1/12 + (1/3)*(self.X**2)*(Parameters.tk**3)
            e = 3*d*(Parameters.tk*self.X**3 - (1/3)*Parameters.Gravity)/Parameters.KF - 1/(3*b)
            i = (1/9)*Parameters.KF - Parameters.tk*(self.X**3)*((self.X**2)*(Parameters.tk**3) + (1/3))*d
            g = Parameters.Gravity/Parameters.KF + Parameters.tk*self.X*((9*Parameters.H0*self.X - 12*(self.X**2))/Parameters.KF + (2*(Parameters.tk**2)/3)/(b**2))

            self.f_mmsp = (Parameters.tk**2)*(9*(self.X**2) + 9*d*a/Parameters.KF\
                                         + 6*Parameters.intrPerm*(Parameters.tk**4)*(b**2)*((1 - (1/(6*b)))**3)*e/(Parameters.KF*j)\
                                         + 108*(Parameters.tk**4)*Parameters.intrPerm*self.X*(a**4)*(i + Parameters.Gravity*h*d)/(Parameters.KF*(c**2)*(b**3))\
                                         + 54*(Parameters.tk**4)*Parameters.intrPerm*self.X*(a**2)*(i + Parameters.Gravity*h*d)/(Parameters.KF*c*(b**2))\
                                         + 6*Parameters.tk*Parameters.intrPerm*(b**2)*((1 - (1/(6*b)))**3)*g/j)
          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

        elif Parameters.MMS_PressureSolutionType == '2S1T3':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            a = 1/6  + (self.X**2)*(Parameters.tk**3)
            b = 1/3  + (self.X**2)*(Parameters.tk**3)
            c = 0.25 + (self.X**2)*(Parameters.tk**3)
            d = 2*Parameters.H0 - self.X
            f = self.X*(Parameters.tk**3)
            h = 1/9 + (1/3)*(self.X**2)*(Parameters.tk**3)
            j = 1/12 + (1/3)*(self.X**2)*(Parameters.tk**3)
            e = 3*d*(Parameters.tk*self.X**3 - (1/3)*Parameters.Gravity)/Parameters.KF - 1/(3*b)
            i = (1/9)*Parameters.KF - Parameters.tk*(self.X**3)*((self.X**2)*(Parameters.tk**3) + (1/3))*d
            g = Parameters.Gravity/Parameters.KF + Parameters.tk*self.X*((18*Parameters.H0*self.X - 12*(self.X**2))/Parameters.KF + (2*(Parameters.tk**2)/3)/(b**2))

            self.f_mmsp = (Parameters.tk**2)*(9*(self.X**2) + 9*d*a/Parameters.KF\
                                         + 6*Parameters.intrPerm*(Parameters.tk**4)*(b**2)*((1 - (1/(6*b)))**3)*e/(Parameters.KF*j)\
                                         + 108*(Parameters.tk**4)*Parameters.intrPerm*self.X*(a**4)*(i + Parameters.Gravity*h*d)/(Parameters.KF*(c**2)*(b**3))\
                                         + 54*(Parameters.tk**4)*Parameters.intrPerm*self.X*(a**2)*(i + Parameters.Gravity*h*d)/(Parameters.KF*c*(b**2))\
                                         + 6*Parameters.tk*Parameters.intrPerm*(b**2)*((1 - (1/(6*b)))**3)*g/j)
          else:
            print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
            raise RuntimeError

    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      if Parameters.MMS_FluidSolutionType == 'S4T3':
        if Parameters.MMS_PressureSolutionType == 'S1T2':
          if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
            h = (Parameters.tk**3)*(self.X**3)
            a = 0.125  + h
            b = 0.250  + h
            c = 0.1875 + h
            d = Parameters.H0 - self.X
            e = 1/12   + (1/3)*h
            g = Parameters.Gravity/Parameters.KF + Parameters.tk*(self.X**2)*\
                (12*Parameters.H0*self.X/Parameters.KF - 15*(self.X**2)/Parameters.KF\
                 + 0.75*(Parameters.tk**2)/(b**2))
            
            self.f_mmsp = Parameters.tk*(12*Parameters.tk*(self.X**3) + 32*a*b*d/Parameters.KF\
                          + 216*(Parameters.tk**4)*Parameters.intrPerm*(self.X**2)*(a**4)*\
                            (Parameters.KF/12 - Parameters.tk*(self.X**4)*d*b + Parameters.Gravity*\
                             e*d)/(Parameters.KF*(c**2)*(b**3))\
                          + 81*(Parameters.tk**4)*Parameters.intrPerm*(self.X**2)*(a**2)*\
                            (Parameters.KF/12 - Parameters.tk*(self.X**4)*d*b + Parameters.Gravity*\
                             e*d)/(Parameters.KF*c*(b**3))\
                          - 288*(Parameters.tk**3)*Parameters.intrPerm*(a**3)*\
                            (Parameters.KF/12 - Parameters.tk*(self.X**4)*d*b + Parameters.Gravity*\
                             e*d)/((Parameters.KF**2)*(0.046875 + 0.4735*h + h**2))\
                          - 6*Parameters.tk*Parameters.intrPerm*(b**2)*((1 - 0.125/b)**3)*g/\
                            (0.046875 + 0.25*h))

          elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
            h = (Parameters.tk**3)*(self.X**3)
            a = 0.125 + h
            b = 0.250 + h
            c = 1.000 + 8*h
            d = Parameters.H0 - self.X
            e = 1/12 + (1/3)*h
            f = (0.25 + h)*(self.X**2)
            g = Parameters.Gravity/Parameters.KF + (self.X**2)*Parameters.tk*\
                (12*self.X*Parameters.H0/Parameters.KF - 15*(self.X**2)/Parameters.KF +\
                0.75*(Parameters.tk**2)/(b**2))
            i = 3*d*(-(1/3)*Parameters.Gravity + Parameters.tk*(self.X**4))
            j = 0.25/b

            self.f_mmsp = Parameters.tk*(12*Parameters.tk*(self.X**3) + 32*d*a*b/Parameters.KF\
                          -12*(2**Parameters.kappa)*Parameters.kappa*(Parameters.tk**4)*\
                           Parameters.intrPerm*(self.X**2)*((c/2)**(Parameters.kappa - 1))*\
                           (i/Parameters.KF - j)\
                          + (Parameters.tk**3)*Parameters.intrPerm*(1 + 4*h)*(c**Parameters.kappa)*\
                            (i/Parameters.KF - j)/Parameters.KF\
                          - Parameters.tk*Parameters.intrPerm*(c**Parameters.kappa)*g) 

        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

  try:
    self.H_MMS  = -np.einsum('ik, k', self.Np, self.f_mmsp*self.weights)
  except AttributeError:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nInvalid MMS solution combination.")
    raise RuntimeError
  self.H_MMS *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I1(self, Parameters):
  # Compute I_1^INT (pore fluid acceleration contribution to I).
  self.I_1  = np.einsum('ik, k', self.Nuf, self.rhof_0*self.a_f*self.weights)
  self.I_1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I2(self, Parameters):
  # Compute I_2^INT (pore fluid pressure gradient contribution to I).
  try:
    if Parameters.LagrangeApply:
      T1       = np.einsum('ik, k', self.Buf, self.nf*self.p_f*self.weights)
      T2       = np.einsum('ik, k', self.Nuf, self.dnfdX*self.p_f*self.weights)
      self.I_2 = -(T1 + T2)
    else:
      self.I_2  = np.einsum('ik, k', self.Nuf, self.nf*self.dp_fdX*self.weights)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Pore fluid pressure = ", self.p_f)
    print("Pressure instability in I_2; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.tk, Parameters.dt))
    raise FloatingPointError
  self.I_2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I3(self, Parameters):
  # Compute I_3^INT (velocity contribution to I).
  self.I_3  = np.einsum('ik, k', self.Nuf, self.J*(self.nf**2)*(self.v_f - self.v_s)*self.weights/self.khat)
  self.I_3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I4(self, Parameters):
  # Compute I_4^INT (gravitational contribution to I).
  self.I_4  = -np.einsum('ik, k', self.Nuf, self.rhof_0*Parameters.Gravity*self.weights)
  self.I_4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I5(self, Parameters):
  # Compute I_5^INT (Darcy-Brinkman contribution to I).
  self.I_5  = -np.einsum('ik, k', self.Nuf, self.DIV_FES*self.weights)
  self.I_5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_I6(self, Parameters):
  # Compute I_6 (thermal contribution to I).
  self.I_6  = np.einsum('ik, k', self.Nuf, self.p_f*self.dnfdX*(1 - self.ts/self.tf)*self.weights)
  self.I_6 *= Parameters.Area*self.Jacobian
  return

#@register_method
#def get_I7(self, Parameters):
#  # Compute I_7 (shock viscosity gradient contribution to I).
#  self.I_7  = np.einsum('ik, k', self.Nuf, self.DIV_Qf*self.weights)
#  self.I_7 *= Parameters.Area*self.Jacobian
#  self.I_7 = 0
#  return

@register_method
def get_IMMS(self, Parameters):
  # Compute f_mms_uf.
  if Parameters.MMS_SolidSolutionType == 'S2T3':
    if Parameters.MMS_FluidSolutionType == 'S2T3':
      if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
        a = (1/4)                   + self.X**2 * Parameters.tk**3
        b = (1/2)                   + self.X**2 * Parameters.tk**3
        c = ((1/4)*(self.X**2)*(3/8 + self.X*Parameters.tk**3))/Parameters.intrPerm
        d = Parameters.H0 - self.X

        self.f_mmsuf = Parameters.tk**2 * (-2*Parameters.Gravity*d*(a**2)/(Parameters.KF*b)\
                                           + 6*Parameters.tk*d*(self.X**2)*(a**2)/(Parameters.KF*b)\
                                           - c/b - (1 - 1/(4*b))**2)/(1 - 1/(4*b))

      elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
        h = self.X * Parameters.tk**3
        a = 0.25 + h
        b = 0.50 + h
        d = Parameters.H0 - self.X
        
        self.f_mmsuf = Parameters.tk**2 * (1 - 0.25/b)*(6*Parameters.tk*(self.X**2)*d*b/Parameters.KF\
                                                        - 3*(self.X**2)*a*((4*a)**(Parameters.kappa - 1))/\
                                                          Parameters.intrPerm\
                                                        - 2*Parameters.Gravity*d*b/Parameters.KF - 1)

      else:
        print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
        raise RuntimeError

  elif Parameters.MMS_SolidSolutionType == 'S3T3':
    if Parameters.MMS_FluidSolutionType == 'S3T3':
      if Parameters.MMS_PressureSolutionType == 'S1T2':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          h = self.X**2 * Parameters.tk**3
          a = 1/6 + h
          b = 1/3 + h
          c = (0.0625 + 0.25*h) * self.X**3
          d = Parameters.H0 - self.X

          self.f_mmsuf = Parameters.tk**2 * (9*Parameters.tk*d*(self.X**3)*(a**2)/(Parameters.KF*b)\
                                             -3*Parameters.Gravity*d*(a**2)/(Parameters.KF*b)\
                                             -c/(Parameters.intrPerm*b) - (1 - 1/(6*b))**2)/(1 - 1/(6*b))

        elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
          h = self.X**2 * Parameters.tk**3
          b = 1/3 + h
          c = (0.75 + 4.5*h) * self.X**3
          e = 1 + 6*h
          d = Parameters.H0 - self.X
          
          self.f_mmsuf = Parameters.tk**2 * (1 - (1/6)/b)*(9*Parameters.tk*d*(self.X**3)*b/Parameters.KF\
                                             - (e**(Parameters.kappa - 1))*c/Parameters.intrPerm\
                                             - 3*Parameters.Gravity*d*b/Parameters.KF - 1)
        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

      elif Parameters.MMS_PressureSolutionType == 'S1T3' and Parameters.khatType == 'Kozeny-Carman':
        if Parameters.solidModel == 'neo-Hookean':
          a = (1/6)             +       self.X**2 * Parameters.tk**3
          b = (1/3)             +       self.X**2 * Parameters.tk**3
          c = ((1/16)*self.X**3 + (1/4)*self.X**5 * Parameters.tk**3)/Parameters.intrPerm
          d = Parameters.H0 - self.X
          
          self.f_mmsuf = Parameters.tk**2 * (-3*Parameters.Gravity*d*a/Parameters.KF\
                                             + 9*(Parameters.tk**2)*d*(self.X**3)*a/Parameters.KF\
                                             - c/a - Parameters.tk*(1 - 1/(6*b)))
        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError
      
      elif Parameters.MMS_PressureSolutionType == '2S1T3':
        if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
          a = (1/6)             +       self.X**2 * Parameters.tk**3
          b = (1/3)             +       self.X**2 * Parameters.tk**3
          c = ((1/16)*self.X**3 + (1/4)*self.X**5 * Parameters.tk**3)/Parameters.intrPerm
          d = 2*Parameters.H0 - self.X
          
          self.f_mmsuf = Parameters.tk**2 * (-3*Parameters.Gravity*d*a/Parameters.KF\
                                             + 9*(Parameters.tk**2)*d*(self.X**3)*a/Parameters.KF\
                                             - c/a - Parameters.tk*(1 - 1/(6*b)))
        else:
          print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
          raise RuntimeError

  elif Parameters.MMS_SolidSolutionType == 'S4T3':
    if Parameters.MMS_FluidSolutionType == 'S4T3':
      if Parameters.solidModel == 'neo-Hookean' and Parameters.khatType == 'Kozeny-Carman':
        a = (1/8)             +       self.X**3 * Parameters.tk**3
        b = (1/4)             +       self.X**3 * Parameters.tk**3
        c = ((3/64)*self.X**4 + (1/4)*self.X**7 * Parameters.tk**3)/Parameters.intrPerm
        d = Parameters.H0 - self.X

        self.f_mmsuf = Parameters.tk**2 * (-4*Parameters.Gravity*d*(a**2)/(Parameters.KF*b)\
                                           + 12*Parameters.tk*d*(self.X**4)*(a**2)/(Parameters.KF*b)\
                                           - c/b - (1 - 0.125/b)**2)/(1 - 0.125/b)

      elif Parameters.solidModel == 'neo-Hookean-Eipper' and Parameters.khatType == 'Hyperbolic':
        h = self.X**3 * Parameters.tk**3
        a = 0.125 + h
        b = 0.25 + h
        d = Parameters.H0 - self.X
        e = -(0.046875 + 6*h)*(self.X**4) 

        self.f_mmsuf = Parameters.tk**2 * (12*Parameters.tk*d*(self.X**4)*(a**2)/(Parameters.KF*b)\
                                           + e/(Parameters.intrPerm*b) - 4*Parameters.Gravity*d*\
                                           (a**2)/(Parameters.KF*b) - (1 - 0.1255/b)**2)/\
                       (1 - 0.125/b)
      else:
        print("-----------------\nINPUT FILE ERROR:\n-----------------\nConstitutive model-solution combination unavailable for MMS.")
        raise RuntimeError

  try:
    self.I_MMS  = -np.einsum('ik, k', self.Nuf, self.f_mmsuf*self.weights)
  except AttributeError:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nInvalid MMS solution combination for uf.")
    raise RuntimeError
  self.I_MMS *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J1(self, Parameters):
  # Compute J_1^INT (temperature rate contribution to J).
  self.J_1  = np.einsum('ik, k', self.Nts, Parameters.cvs*self.rhos_0*self.tsDot*self.weights)
  self.J_1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J2(self, Parameters):
  # Compute J_2^INT (thermoelastic contribution to J).
  if Parameters.Physics == 'u-t' and 'Ideal-Gas' not in Parameters.solidModel:
    self.J_2 = np.einsum('ik, k', self.Nts, self.JDot*(Parameters.Bb0T*Parameters.As*self.ts/self.J + self.Q)*self.weights)
  elif Parameters.Physics == 'u-t' and 'Ideal-Gas' in Parameters.solidModel:
    self.J_2 = np.einsum('ik, k', self.Nts, self.JDot*(self.ps_E + self.Q)*self.weights)

  else:
    self.J_2 = np.einsum('ik, k', self.Nts, self.JDot*(Parameters.KSkel*Parameters.As*self.ts/self.J + self.Q)*self.weights)
  if 'tf' in Parameters.Physics:
    self.J_2 += np.einsum('ik, k', self.Nts, self.JDot*self.ns*self.p_f*self.ts*self.weights/self.tf)

  self.J_2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J3(self, Parameters):
  # Compute J_3^INT (heat conduction contribution to J).
  self.J_3  = -np.einsum('ik, k', self.Bts, self.qs*self.weights)
  self.J_3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J4(self, Parameters):
  # Compute J_4^INT (thermal convective coupling contribution to J).
  self.J_4  = np.einsum('ik, k', self.Nts, self.J*Parameters.k_exchange*(self.ts - self.tf)*self.weights)
  self.J_4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J5(self, Parameters):
  # Compute J_5^INT (interphase power contribution to J).
  if 'uf' in Parameters.Physics:
    self.J_5 = np.einsum('ik, k', self.Nts, self.J*(self.nf**2)*((self.v_f - self.v_s)**2)*self.weights/self.khat)
  else:
    self.J_5 = np.einsum('ik, k', self.Nts, self.J*(self.vDarcy**2)*self.weights/self.khat)

  self.J_5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J6(self, Parameters):
  # Compute J_6^INT (interphase power contribution to J).
  if 'uf' in Parameters.Physics:
    self.J_6 = np.einsum('ik, k', self.Nts, (self.ts/self.tf)*self.p_f*self.dnfdX*\
                                            (self.v_f - self.v_s)*self.weights)
  else:
    self.J_6  = np.einsum('ik, k', self.Nts, (self.ts/self.tf)*(self.p_f/self.nf)*\
                                             self.dnfdX*self.vDarcy*self.weights)
  self.J_6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_J7(self, Parameters):
  # Artificial heat viscosity (conduction).
  if np.any(self.Q != 0) and Parameters.h0 > 0 and Parameters.h1 > 0: 
    self.H    = np.zeros(Parameters.Gauss_Order)
    if Parameters.solidModel == 'neo-Hookean-Eipper':
      bulk_term = Parameters.H0e*(Parameters.h0*np.abs(self.dvdX) + \
                  Parameters.h1*self.c)*(self.d2udX2*(Parameters.mu*(self.J**2 - 1) + \
                  self.J*Parameters.lambd*(1 - Parameters.ns_0)*\
                  (1 - (1 - Parameters.ns_0)/(self.J - Parameters.ns_0)) + \
                  Parameters.KSkel*Parameters.As*Parameters.Ts_0) + \
                  self.J*self.rhos_0*Parameters.cvs*self.dtsdX)

      self.H[self.Qidxs] = bulk_term[self.Qidxs]

    self.J_7  = -np.einsum('ik, k', self.Nts, self.H*self.weights)
    self.J_7 *= Parameters.Area*self.Jacobian
  else:
    self.J_7 = 0
  return

@register_method
def get_K1(self, Parameters):
  # Compute K_1^INT (temperature rate contribution to K).
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_1 = np.einsum('ik, k', self.Ntf, (Parameters.cvf + Parameters.RGas)*self.rhof_0*\
                         self.tfDot*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_1 = np.einsum('ik, k', self.Ntf, (self.rhof_0*Parameters.cvf + \
                         self.J*self.nf*Parameters.KF*(Parameters.Af**2)*self.tf)*\
                         self.tfDot*self.weights)
  else:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    raise RuntimeError

  self.K_1 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K2(self, Parameters):
  # Compute K_2^INT (thermal gradient contribution to K).
  if Parameters.fluidModel == 'Ideal-Gas':
    if 'uf' in Parameters.Physics:
      self.K_2 = np.einsum('ik, k', self.Ntf, self.rhof_0*Parameters.cvf*self.dtfdX*\
                                              (self.v_f - self.v_s)*self.weights/self.F11)
    else:
      self.K_2 = np.einsum('ik, k', self.Ntf, self.rhofR*(Parameters.cvf + Parameters.RGas)*\
                                              self.dtfdX*self.vDarcy*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    if 'uf' in Parameters.Physics:
      self.K_2 = np.einsum('ik, k', self.Ntf, (self.rhof_0*Parameters.cvf + self.J*self.nf*Parameters.Af*\
                           (self.tf*Parameters.Af*Parameters.KF - self.p_f))*self.dtfdX*\
                           (self.v_f - self.v_s)*self.weights/self.F11)
    else:
      self.K_2 = np.einsum('ik, k', self.Ntf, (self.rhofR*Parameters.cvf + self.tf*\
                                               Parameters.KF*(Parameters.Af**2))*self.dtfdX*\
                                               self.vDarcy*self.weights)
  else:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    raise RuntimeError

  self.K_2 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K3(self, Parameters):
  # Compute K_3^INT (stress power contribution to K).
  if 'uf' in Parameters.Physics:
    self.K_3 = np.einsum('ik, k', self.Ntf, self.nf*self.p_f*self.dvfdX*self.weights)
  else:
    self.K_3 = -np.einsum('ik, k', self.Ntf, self.ns*self.p_f*self.JDot*self.weights)

  self.K_3 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K4(self, Parameters):
  # Compute K_4^INT (stress power contribution to K).
  if 'uf' in Parameters.Physics:
    if Parameters.DarcyBrinkman:
      self.K_4 = -np.einsum('ik, k', self.Ntf, (self.nf/self.J)*(self.dvfdX**2)*\
                            (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)*self.weights)
    else:
      self.K_4 = np.zeros(Parameters.ndofTfe)
      return
  else:
    self.K_4 = -np.einsum('ik, k', self.Ntf, (self.p_f/self.nf)*self.dnfdX*self.vDarcy*self.weights)

  self.K_4 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K5(self, Parameters):
  # Compute K_5^INT (stress power contribution to K (u-pf-ts-tf);
  #                  thermal convection contribution to K (u-uf-pf-ts-tf)).
  if 'uf' in Parameters.Physics and Parameters.fluidModel == 'Exponential-Thermal':
    self.K_5 = np.einsum('ik, k', self.Ntf, self.J*self.nf*(self.p_f/Parameters.KF - self.tf*Parameters.Af)*\
                         (self.p_fDot + self.dp_fdX*(self.v_f - self.v_s)/self.F11)*self.weights)
  else:
    if Parameters.fluidModel == 'Ideal-Gas':
      self.K_5 = -np.einsum('ik, k', self.Ntf, self.J*self.nf*self.p_fDot*self.weights)
    elif Parameters.fluidModel == 'Exponential-Thermal':
      self.K_5 = -np.einsum('ik, k', self.Ntf, self.J*self.nf*self.tf*self.p_fDot*\
                                               Parameters.Af*self.weights)
    else:
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
      raise RuntimeError

  self.K_5 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K6(self, Parameters):
  # Compute K_6^INT (stress power contribution to K (u-pf-ts-tf)).
  if Parameters.fluidModel == 'Ideal-Gas':
    self.K_6 = -np.einsum('ik, k', self.Ntf, self.dp_fdX*self.vDarcy*self.weights)
  elif Parameters.fluidModel == 'Exponential-Thermal':
    self.K_6 = -np.einsum('ik, k', self.Ntf, Parameters.Af*self.tf*self.dp_fdX*\
                                             self.vDarcy*self.weights)
  else:
    print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid constitutive model not consistent with locally inhomogeneous temperature formulation.")
    raise RuntimeError

  self.K_6 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K7(self, Parameters):
  # Compute K_7^INT (thermal conduction contribution to K).
  self.K_7  = -np.einsum('ik, k', self.Btf, self.qf*self.weights)
  self.K_7 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K8(self, Parameters):
  # Compute K_8^INT (thermal convective coupling contribution to K).
  self.K_8  = -np.einsum('ik, k', self.Ntf, self.J*Parameters.k_exchange*(self.ts - self.tf)*self.weights)
  self.K_8 *= Parameters.Area*self.Jacobian
  return

@register_method
def get_KStab(self, Parameters):
  # Compute KStab (SUPG stabilization contribution to K).
  #-----------------------------------
  # Assume stability is zero to start.
  #-----------------------------------
  self.K_Stab = 0
  #--------------
  # Define vstar.
  #--------------
  if 'uf' in Parameters.Physics:
    self.vstar = self.rhofR*self.nf*Parameters.cvf*(self.v_f - self.v_s)
  else:
    self.vstar = self.rhofR*(Parameters.cvf + Parameters.RGas)*self.vDarcy 
  if np.any(np.abs(self.vstar)) > 0:
    Parameters.computeKStabTangent = True
    #-------------------------
    # Compute diffusive force.
    #-------------------------
    self.dstar = self.nf*Parameters.kf
    #-------------------------------
    # Compute Peclet number for 1-D.
    #-------------------------------
    self.Pe_TF = self.J*Parameters.H0e*self.vstar/(2*self.dstar)
    #----------------------------------------------------------
    # Compute 'beta', the intermediary stabilization parameter.
    #----------------------------------------------------------
    if np.all(self.Pe_TF > -3) and np.all(self.Pe_TF < 3):
      self.BetaPe = self.Pe_TF/3
    else:
      self.BetaPe = np.sign(self.Pe_TF)
    #------------------------------------------
    # Compute the stabilization parameter \tau.
    #------------------------------------------
    self.tauStab = self.BetaPe*self.J*Parameters.H0e/(2*(self.vstar**2)) 
#    self.tauStab = ((2*(self.vstar**2)/(self.J*Parameters.H0e)) +\
#                    9*((4*self.dstar)/((self.J*Parameters.H0e)**2))**2)**(-0.5)
    #----------------------------------------------
    # Compute components of stabilization residual.
    #----------------------------------------------
    self.get_K2_Stab(Parameters)
    self.get_K3_Stab(Parameters)
    self.get_K4_Stab(Parameters)
    if 'uf' in Parameters.Physics:
      self.K5_Stab = np.zeros(Parameters.ndofTfe)
      self.K6_Stab = np.zeros(Parameters.ndofTfe)
    else:
      self.get_K5_Stab(Parameters)
      self.get_K6_Stab(Parameters)
    
    self.K_Stab += self.K2_Stab + self.K3_Stab + self.K4_Stab + self.K5_Stab + self.K6_Stab
    return
  else:
    Parameters.computeKStabTangent = False
    return
  
@register_method
def get_K2_Stab(self, Parameters):
  self.K2_Stab = np.einsum('ik, k', self.Btf, self.tauStab*(self.vstar**2)*self.dtfdX*self.weights/self.F11)
  self.K2_Stab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K3_Stab(self, Parameters):
  if 'uf' in Parameters.Physics:
    self.K3_Stab = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*self.nf*self.p_f*\
                                                 self.dvfdX*self.weights/self.F11)
  else:
    self.K3_Stab  = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*self.ns*self.p_f*\
                                                  self.JDot*self.weights/self.J)

  self.K3_Stab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K4_Stab(self, Parameters):
  if 'uf' in Parameters.Physics:
    if Parameters.DarcyBrinkman:
      self.K4_Stab = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*self.weights*\
                                                   self.nf*(self.dvfdX/self.F11)**2)*\
                                                   (Parameters.fluidBulkVisc + 2*Parameters.fluidShearVisc)
    else:
      self.K4_Stab = np.zeros(Parameters.ndofTfe)
      return
  else:
    self.K4_Stab = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*self.nf*self.p_fDot*self.weights)

  self.K4_Stab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K5_Stab(self, Parameters):
  self.K5_Stab  = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*Parameters.k_exchange*\
                                                (self.ts - self.tf)*self.weights)
  self.K5_Stab *= Parameters.Area*self.Jacobian
  return

@register_method
def get_K6_Stab(self, Parameters):
  self.K6_Stab  = np.einsum('ik, k', -self.Btf, self.tauStab*self.vstar*\
                                               ((self.p_f/self.nf)*self.dnfdX + self.dp_fdX)*\
                                               self.vDarcy*self.weights/self.F11)
  self.K6_Stab *= Parameters.Area*self.Jacobian
  return

