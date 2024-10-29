#----------------------------------------------------------------------------------------
# Module housing top-level element class.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    September 27, 2024
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

try:
  import _ElementVariables
except ImportError:
  sys.exit("MODULE WARNING. '_ElementVariables.py' not found, check configuration.")

try:
  import _ElementForces
except ImportError:
  sys.exit("MODULE WARNING. '_ElementForces.py' not found, check configuration.")

try:
  import _ElementTangents
except ImportError:
  sys.exit("MODULE WARNING. '_ElementTangents.py' not found, check configuration.")

@Lib.add_methods_from(_ElementVariables, _ElementForces, _ElementTangents)

class Element:
  
  def __init__(self, a_GaussOrder=None, a_ID=None):
    # Set Gauss quadrature order.
    self.set_Gauss_Order(a_GaussOrder)
    # Set element ID.
    self.set_Element_ID(a_ID)
    return

  def set_Gauss_Order(self, a_Order):
    # Initialize the gauss quadrature order.
    self.Gauss_Order = a_Order
    return

  def set_Element_ID(self, a_ID):
    # Initialize the element number.
    self.ID = a_ID
    return

  def set_Gauss_Points(self, Parameters):
    # Initialize the Gauss quadrature points.
    if self.Gauss_Order == 1:
      self.points = np.array([0.0], dtype=np.float64)
    elif self.Gauss_Order == 2:
      if Parameters.Lumping:
        self.points = np.array([-1, 1])
      else:
        self.points = np.array([-0.57735, 0.57735], dtype=np.float64)
    elif self.Gauss_Order == 3:
      self.points = np.array([-0.774597, 0., 0.774597], dtype=np.float64) 
    elif self.Gauss_Order == 4:
      self.points = np.array([-0.861136, -0.339981, 0.33981, 0.861136], dtype=np.float64)
    elif self.Gauss_Order == 5:
      self.points = np.array([-0.90618, -0.538469, 0.0, 0.538469, 0.90618], dtype=np.float64)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHigher order Gauss quadrature has not been implemented.\nMaximum is 5-pt.")
    return

  def set_Gauss_Weights(self):
    # Initialize the Gauss quadrature weights.
    if self.Gauss_Order == 1:
      self.weights = np.array([2.0], dtype=np.float64)
    elif self.Gauss_Order == 2:
      self.weights = np.array([1.0, 1.0], dtype=np.float64)
    elif self.Gauss_Order == 3:
      self.weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=np.float64)
    elif self.Gauss_Order == 4:
      self.weights = np.array([0.347855, 0.652145, 0.652145, 0.347855], dtype=np.float64)
    elif self.Gauss_Order == 5:
      self.weights = np.array([0.236987, 0.478629, 0.568889, 0.478629, 0.236987], dtype=np.float64)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHigher order Gauss quadrature has not been implemented.\nMaximum is 5-pt.")
    return

  def set_Coordinates(self, Parameters):
    # Initialize element coordinates.
    # Assumes a fixed mesh.
    if Parameters.Element_Type.split('-')[0] == 'Q3H':
      self.coordinates = np.linspace(Parameters.H0e*self.ID, Parameters.H0e*(1 + self.ID), Parameters.ndofSe - 2)
      self.X           = np.einsum('ik, i', self.Nu_Intp, self.coordinates, dtype=np.float64)
    else:
      self.coordinates = np.linspace(Parameters.H0e*self.ID, Parameters.H0e*(1 + self.ID), Parameters.ndofSe)
      self.X           = np.einsum('ik, i', self.Nu, self.coordinates, dtype=np.float64)
    return

  def set_Jacobian(self, Parameters):
    # Compute the element Jacobian.
    # self.Jacobian = (self.coordinates[1] - self.coordinates[0])/2 # Placeholder for potential AMR
    self.Jacobian = Parameters.jac
    return

  def evaluate_Shape_Functions(self, Parameters):
    # Initialize the shape functions used for interpolation.
    #--------------------------------------
    # Solid skeleton interpolation options.
    #--------------------------------------
    #----------------------------------------------------------------------------
    # Hermite cubic polynomial; NOT a Lagrange cubic polynomial.
    #
    # Element DOFs are numbered as follows:
    #
    # du/dX (2)          du/dX (3)
    #   x------------------x
    #   u   (0)            u   (1)
    #
    # This is in contrast to the global DOFs which are ordered:
    #
    # du/dX (1)          du/dX (3)
    #   x------------------x
    #   u   (0)            u   (2)
    #----------------------------------------------------------------------------
    if Parameters.ndofSe == 4 and Parameters.Element_Type.split('-')[0] == 'Q3H':
      self.Nu_Intp = np.array([(1 - self.points)/2,\
                               (1 + self.points)/2], dtype=np.float64)

      self.Nu  = np.array([(1.0 - self.points)**2 * (2.0 + self.points)/4.0,\
                           (1.0 + self.points)**2 * (2.0 - self.points)/4.0,\
                           (1.0 - self.points)**2 * (1.0 + self.points)/4.0,\
                           (1.0 + self.points)**2 * (self.points - 1.0)/4.0], dtype=np.float64)

      self.Bu  = np.array([0.75*(self.points**2 - 1.0),\
                          -0.75*(self.points**2 - 1.0),\
                           0.25*(3*self.points**2 - 2.0*self.points - 1.0),\
                           0.25*(3*self.points**2 + 2.0*self.points - 1.0)], dtype=np.float64)

      self.Bu /= self.Jacobian

      self.B2u = np.array([(3.0*self.points/2.0),\
                          -(3.0*self.points/2.0),\
                           ((3.0*self.points - 1)/2.0),\
                           ((3.0*self.points + 1)/2.0)], dtype=np.float64)

      self.B2u /= self.Jacobian**2

      self.Nu[2:4,:]  *= self.Jacobian
      self.Bu[2:4,:]  *= self.Jacobian
      self.B2u[2:4,:] *= self.Jacobian
    #----------------------------------------------------------------------------
    # Lagrange cubic polynomial.
    #
    # Element DOFs are consistent with global DOFs and are numbered as follows:
    #
    #   x------x-----x-----x
    #  (0)    (1)   (2)   (3)
    #
    #----------------------------------------------------------------------------
    elif Parameters.ndofSe == 4 and Parameters.Element_Type.split('-')[0] == 'Q3':
      third = 1/3
      self.Nu = np.array([-9  * (self.points + third)*(self.points - third)*(self.points - 1) ,\
                           27 * (self.points + 1    )*(self.points - third)*(self.points - 1) ,\
                          -27 * (self.points + 1    )*(self.points + third)*(self.points - 1) ,\
                           9  * (self.points + third)*(self.points - third)*(self.points + 1)],\
                          dtype=np.float64)
      self.Nu /= 16
      
      self.Bu = np.array([     (-27*self.points**2 + 18*self.points + 1),\
                           9 * (  9*self.points**2 -  2*self.points - 3),\
                          -9 * (  9*self.points**2 +  2*self.points - 3),\
                               ( 27*self.points**2 + 18*self.points - 1)],\
                          dtype=np.float64)
      self.Bu /= self.Jacobian*16

      self.B2u = np.array([-9*(3*self.points - 1),\
                            9*(9*self.points - 1),\
                           -9*(9*self.points + 1),\
                            9*(3*self.points + 1)],\
                           dtype=np.float64)
      self.B2u /= (self.Jacobian**2)*8
    #----------------------------------------------------------------------------
    # Lagrange quadratic polynomial.
    #
    # Element DOFs are consistent with global DOFs and are numbered as follows:
    #
    #   x---------x--------x
    #   u (0)     u (1)    u (2)
    #
    #----------------------------------------------------------------------------
    elif Parameters.ndofSe == 3:
      self.Nu  = np.array([self.points*(self.points - 1.0)/2.0,\
                           1.0 - self.points**2,\
                           self.points*(self.points + 1.0)/2.0], dtype=np.float64)

      self.Bu  = np.array([self.points - 0.5,\
                           -2.0*self.points,\
                           self.points + 0.5,], dtype=np.float64)
      self.Bu /= self.Jacobian
    #----------------------------------------------------------------------------
    # Lagrange linear polynomial.
    #
    # Element DOFs are consistent with global DOFs and are numbered as follows:
    #
    #   x------------------x
    #   u (0)              u (1)
    #----------------------------------------------------------------------------
    elif Parameters.ndofSe == 2:
      self.Nu = np.array([(1 - self.points)/2,\
                          (1 + self.points)/2], dtype=np.float64)

      self.Bu       = (1/self.Jacobian)*np.ones((2, self.points.shape[0]), dtype=np.float64)
      self.Bu[0,:] *= -1/2
      self.Bu[1,:] *= 1/2

    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nUnable to initialize solid skeleton shape functions.")
    #---------------------------------------------------------------------------------------
    # Pore fluid pressure is always assumed to use Lagrange linear polynomial interpolation.
    #
    # Element DOFs are numbered as follows:
    #
    #   x------------------x
    #   pf (0)             pf (1)
    #---------------------------------------------------------------------------------------
    if 'pf' in Parameters.Physics:
      self.Np = np.array([(1 - self.points)/2,\
                          (1 + self.points)/2], dtype=np.float64)

      self.Bp       = (1/self.Jacobian)*np.ones((2, self.points.shape[0]), dtype=np.float64)
      self.Bp[0,:] *= -1/2
      self.Bp[1,:] *= 1/2
    #-----------------------------------------------
    # Pore fluid displacement interpolation options.
    #
    # See notes on solid skeleton above.
    #-----------------------------------------------
    if 'uf' in Parameters.Physics:
      if Parameters.ndofFe == Parameters.ndofSe:
        self.Nuf  = np.copy(self.Nu)
        self.Buf  = np.copy(self.Bu)
        try:
          self.B2uf = np.copy(self.B2u)
        except AttributeError:
          pass

      elif Parameters.ndofFe == 4 and Parameters.Element_Type.split('-')[1] == 'Q3H':
        self.Nuf  = np.array([(1.0 - self.points)**2 * (2.0 + self.points)/4.0,\
                              (1.0 + self.points)**2 * (2.0 - self.points)/4.0,\
                              (1.0 - self.points)**2 * (1.0 + self.points)/4.0,\
                              (1.0 + self.points)**2 * (self.points - 1.0)/4.0], dtype=np.float64)

        self.Buf  = np.array([0.75*(self.points**2 - 1.0),\
                             -0.75*(self.points**2 - 1.0),\
                              0.25*(3*self.points**2 - 2.0*self.points - 1.0),\
                              0.25*(3*self.points**2 + 2.0*self.points - 1.0)], dtype=np.float64)
        self.Bu  /= self.Jacobian

        self.B2uf = np.array([(3.0*self.points/2.0),\
                             -(3.0*self.points/2.0),\
                              ((3.0*self.points - 1)/2.0),\
                              ((3.0*self.points + 1)/2.0)], dtype=np.float64)
        self.B2uf /= self.Jacobian**2

        self.Nuf[2:4,:]  *= self.Jacobian
        self.Buf[2:4,:]  *= self.Jacobian
        self.B2uf[2:4,:] *= self.Jacobian

      elif Parameters.ndofFe == 4 and Parameters.Element_Type.split('-')[1] == 'Q3':
        third = 1/3
        self.Nuf = np.array([-9  * (self.points + third)*(self.points - third)*(self.points - 1) ,\
                              27 * (self.points + 1    )*(self.points - third)*(self.points - 1) ,\
                             -27 * (self.points + 1    )*(self.points + third)*(self.points - 1) ,\
                              9  * (self.points + third)*(self.points - third)*(self.points + 1)],\
                             dtype=np.float64)
        self.Nuf /= 16
        
        self.Buf = np.array([     (-27*self.points**2 + 18*self.points + 1),\
                              9 * (  9*self.points**2 -  2*self.points - 3),\
                             -9 * (  9*self.points**2 +  2*self.points - 3),\
                                  ( 27*self.points**2 + 18*self.points - 1)],\
                             dtype=np.float64)
        self.Buf /= self.Jacobian*16

        self.B2uf = np.array([-9*(3*self.points - 1),\
                               9*(9*self.points - 1),\
                              -9*(9*self.points + 1),\
                               9*(3*self.points + 1)],\
                              dtype=np.float64)
        self.B2uf /= (self.Jacobian**2)*8

      elif Parameters.ndofFe == 3:
        self.Nuf  = np.array([self.points*(self.points - 1)/2,\
                              self.points*(self.points + 1)/2,\
                              1 - self.points**2], dtype=np.float64)

        self.Buf  = (1.0/self.Jacobian)*np.array([self.points - 0.5,\
                                                 self.points + 0.5,\
                                                 -2.0*self.points], dtype=np.float64)

      elif Parameters.ndofFe == 2:
        self.Nuf = np.copy(self.Np)
        self.Buf = np.copy(self.Bp)

      if Parameters.LagrangeApply:
        third       = 1/3
        self.N_intp = np.array([-9  * (self.points + third)*(self.points - third)*(self.points - 1) ,\
                                 27 * (self.points + 1    )*(self.points - third)*(self.points - 1) ,\
                                -27 * (self.points + 1    )*(self.points + third)*(self.points - 1) ,\
                                 9  * (self.points + third)*(self.points - third)*(self.points + 1)],\
                               dtype=np.float64)
        self.N_intp /= 16
    #---------------------------------------------------------------------------------------
    # Phase temperatures are always assumed to use Lagrange linear polynomial interpolation.
    #
    # Element DOFs are numbered as follows:
    #
    #   Ts (0)             Ts (1)
    #   x------------------x
    #   Tf (0)             Tf (1)
    #---------------------------------------------------------------------------------------
    if 'ts-tf' in Parameters.Physics:
#      if 'T2' in Parameters.Element_Type:
#        self.Nts  = np.array([self.points*(self.points - 1.0)/2.0,\
#                             1.0 - self.points**2,\
#                             self.points*(self.points + 1.0)/2.0], dtype=np.float64)
#
#        self.Bts  = np.array([self.points - 0.5,\
#                             -2.0*self.points,\
#                             self.points + 0.5,], dtype=np.float64)
#        self.Bts /= self.Jacobian
#        self.Ntf = np.copy(self.Nts)
#        self.Btf = np.copy(self.Bts)
#      else:
      self.Nts = np.copy(self.Np)
      self.Bts = np.copy(self.Bp)
      self.Ntf = np.copy(self.Np)
      self.Btf = np.copy(self.Bp)

    if Parameters.Physics == 'u-t':
      self.Nts = np.array([(1 - self.points)/2,\
                          (1 + self.points)/2], dtype=np.float64)

      self.Bts       = (1/self.Jacobian)*np.ones((2, self.points.shape[0]), dtype=np.float64)
      self.Bts[0,:] *= -1/2
      self.Bts[1,:] *= 1/2

    return

  def get_Global_DOF(self, a_LM):
    # Set the global degrees of freedom of this element.
    self.DOF    = a_LM[:,self.ID]
    self.numDOF = self.DOF.shape[0]
    return

  def set_Global_Solutions(self, a_D, a_V, a_A, Parameters, *args):
    # Set the local solution variables at the current time step.
    self.set_u_s_global(a_D[self.DOF[0:Parameters.ndofSe]])
    self.set_v_s_global(a_V[self.DOF[0:Parameters.ndofSe]])
    self.set_a_s_global(a_A[self.DOF[0:Parameters.ndofSe]])
    if Parameters.Physics.startswith('u-pf'):
      self.DOFP = self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)] - Parameters.ndofS
      self.set_p_f_global(a_D[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)]])
      self.set_p_fDot_global(a_V[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)]])
      if 'Central-difference' in Parameters.integrationScheme:
        self.set_p_fDDot_global(a_A[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofPe)]])
      # if len(args) > 0: # For predictor-corrector
        # self.set_#p_f_globalLast(args[0][self.DOF[Parameters.GaussD:Parameters.GaussD + Parameters.GaussP]])
    elif Parameters.Physics == 'u-t':
      self.DOFTs = self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofTse)] - Parameters.ndofS
      self.set_ts_global(a_D[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofTse)]])
      self.set_tsDot_global(a_V[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofTse)]])
    elif 'uf' in Parameters.Physics:
      self.DOFF = self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)] - Parameters.ndofS
      self.DOFP = self.DOF[(Parameters.ndofSe + Parameters.ndofFe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)] - Parameters.ndofS - Parameters.ndofF
      self.set_u_f_global(a_D[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)]])
      self.set_v_f_global(a_V[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)]])
      self.set_a_f_global(a_A[self.DOF[Parameters.ndofSe:(Parameters.ndofSe + Parameters.ndofFe)]])
      self.set_p_f_global(a_D[self.DOF[Parameters.ndofSe + Parameters.ndofFe:(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)]])
      self.set_p_fDot_global(a_V[self.DOF[Parameters.ndofSe + Parameters.ndofFe:(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe)]])
    if 'ts-tf' in Parameters.Physics:
      self.DOFTs = self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse)] - Parameters.ndofS - Parameters.ndofF - Parameters.ndofP
      self.DOFTf = self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe)] - Parameters.ndofS - Parameters.ndofF - Parameters.ndofP - Parameters.ndofTs
      self.set_ts_global(a_D[self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse)]])
      self.set_tsDot_global(a_V[self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse)]])
      self.set_tf_global(a_D[self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe)]])
      self.set_tfDot_global(a_V[self.DOF[(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse):(Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe)]])
    # Catch instability issues for elastodynamics; does not occur for coupled physics 
    if np.any(np.abs(self.u_s_global)) > 1e8 or np.any(np.abs(self.v_s_global)) > 1e8 or np.any(np.abs(self.a_s_global)) > 1e8:
      print("u_s_global:", self.u_s_global)
      print("v_s_global:", self.v_s_global)
      print("a_s_global:", self.a_s_global)
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------\nEncountered over/underflow assigning local degrees of freedom.")
      raise RuntimeError    
    
    return

  def apply_Local_BC(self, a_g, Parameters):
    # Apply boundary conditions at the element scale.
    #--------------------------------------------------
    # Check to see if element contains a Dirichlet DOF.
    #--------------------------------------------------
    if np.any(self.DOF < 0):
      #------------------------------------------
      # Get the local DOF(s) for Dirichlet BC(s).
      #------------------------------------------
      self.idxs = np.where((self.DOF < 0))[0]
      for self.idx in self.idxs:
        if self.idx < Parameters.ndofSe:
          self.apply_Solid_BC(a_g, Parameters)
        if Parameters.Physics == 'u-t':
          if self.idx >= Parameters.ndofSe:
            self.apply_SolidTemp_BC(a_g, Parameters)
        if 'uf' in Parameters.Physics:
          if self.idx >= Parameters.ndofSe and self.idx < Parameters.ndofSe + Parameters.ndofFe:
            self.apply_Fluid_BC(a_g, Parameters)
          elif self.idx >= Parameters.ndofSe + Parameters.ndofFe and self.idx < Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe:
            self.apply_Pressure_BC(a_g, Parameters)
        elif 'pf' in Parameters.Physics:
          if self.idx >= Parameters.ndofSe and self.idx < Parameters.ndofSe + Parameters.ndofPe:
            self.apply_Pressure_BC(a_g, Parameters)
        if 'ts-tf' in Parameters.Physics:
          if self.idx >= Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe and self.idx < self.numDOF - 2:
            self.apply_SolidTemp_BC(a_g, Parameters)
          elif self.idx >= Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse:
            self.apply_FluidTemp_BC(a_g, Parameters)

    return

  def set_u_s_global(self, a_D):
    # Initialize the uninterpolated solid skeleton displacement.
    self.u_s_global = a_D
    return

  def set_u_f_global(self, a_D):
    # Initialize the uninterpolated pore fluid displacement.
    self.u_f_global = a_D
    return

  def set_p_f_global(self, a_D):
    # Initialize the uninterpolated pore fluid pressure.
    self.p_f_global = a_D
    return

  def set_p_f_globalLast(self, a_D):
    # Initialize the uninterpolated pore fluid pressure.
    self.p_f_globalLast = a_D
    return

  def set_ts_global(self, a_D):
    # Initialize the uninterpolated solid temperature.
    self.ts_global = a_D
    return

  def set_tf_global(self, a_D):
    # Initialize the uninterpolated pore fluid temperature.
    self.tf_global = a_D
    return

  def set_v_s_global(self, a_V):
    # Initialize the uninterpolated solid skeleton velocity.
    self.v_s_global = a_V
    return

  def set_v_f_global(self, a_V):
    # Initialize the uninterpolated pore fluid velocity.
    self.v_f_global = a_V
    return

  def set_p_fDot_global(self, a_V):
    # Initialize the uninterpolated first time derivative of
    # pore fluid pressure.
    self.p_fDot_global = a_V
    return

  def set_tsDot_global(self, a_V):
    # Initialize the uninterpolated first time derivative of solid temperature.
    self.tsDot_global = a_V
    return

  def set_tfDot_global(self, a_V):
    # Initialize the uninterpolated first time derivative of pore fluid temperature.
    self.tfDot_global = a_V
    return

  def set_a_s_global(self, a_A):
    # Initialize the uninterpolated solid skeleton acceleration.
    self.a_s_global = a_A
    return

  def set_p_fDDot_global(self, a_A):
    # Initialize the uninterpolated first time derivative of
    # pore fluid pressure.
    self.p_fDDot_global = a_A
    return

  def set_a_f_global(self, a_A):
    # Initialize the uninterpolated pore fluid acceleration.
    self.a_f_global = a_A
    return

  def apply_Solid_BC(self, a_g, Parameters):
    # Apply the Dirichlet BCs to the solid skeleton.
    # Newmark-beta integration with constant accleration is used for unknown Dirichlet BC values.
    if self.ID == 0:
      self.u_s_global[self.idx] = a_g[0,0,self.idx,0]
      self.v_s_global[self.idx] = a_g[0,1,self.idx,0]
      self.a_s_global[self.idx] = a_g[0,2,self.idx,0]
    elif self.ID == Parameters.ne - 1:
      self.u_s_global[self.idx] = a_g[0,0,self.idx,1]
      self.v_s_global[self.idx] = a_g[0,1,self.idx,1]
      self.a_s_global[self.idx] = a_g[0,2,self.idx,1]
    return

  def apply_SolidTemp_BC(self, a_g, Parameters):
    # Apply the Dirichlet BCs to the solid temperature.
    # Newmark-beta integration with constant accleration is used for unknown Dirichlet BC values.
    if Parameters.Physics == 'u-t':
      sub = Parameters.ndofSe
    else:
      sub = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe

    if self.ID == 0:
      self.ts_global   [self.idx - sub] = a_g[0,0,self.idx,0]
      self.tsDot_global[self.idx - sub] = a_g[0,1,self.idx,0]
    elif self.ID == Parameters.ne - 1:
      self.ts_global   [self.idx - sub] = a_g[0,0,self.idx,1]
      self.tsDot_global[self.idx - sub] = a_g[0,1,self.idx,1]
    return

  def apply_Pressure_BC(self, a_g, Parameters):
    # Apply the Dirichlet BCs to the pore fluid pressure.
    # Newmark-beta integration with constant accleration is used for unknown Dirichlet BC values.
    sub = Parameters.ndofSe + Parameters.ndofFe

    if self.ID == 0:
      self.p_f_global    [self.idx - sub] = a_g[0,0,self.idx,0]
      self.p_fDot_global [self.idx - sub] = a_g[0,1,self.idx,0]
      if Parameters.integrationScheme == 'Central-difference':
        self.p_fDDot_global[self.idx - sub] = a_g[0,2,self.idx,0]
    elif self.ID == Parameters.ne - 1:
      self.p_f_global    [self.idx - sub] = a_g[0,0,self.idx,1]
      self.p_fDot_global [self.idx - sub] = a_g[0,1,self.idx,1]
      if Parameters.integrationScheme == 'Central-difference':
        self.p_fDDot_global[self.idx - sub] = a_g[0,2,self.idx,1]
    return

  def apply_Fluid_BC(self, a_g, Parameters):
    # Apply the Dirichlet BCs to the pore fluid.
    # Newmark-beta integration with constant accleration is used for unknown Dirichlet BC values.
    sub = Parameters.ndofSe
    if self.ID == 0:
      self.u_f_global[self.idx - sub]  = a_g[0,0,self.idx,0]
      self.v_f_global[self.idx - sub]  = a_g[0,1,self.idx,0]
      self.a_f_global[self.idx - sub]  = a_g[0,2,self.idx,0]
    elif self.ID == Parameters.ne - 1:
      self.u_f_global[self.idx - sub]  = a_g[0,0,self.idx,1]
      self.v_f_global[self.idx - sub]  = a_g[0,1,self.idx,1]
      self.a_f_global[self.idx - sub]  = a_g[0,2,self.idx,1]
    return

  def apply_FluidTemp_BC(self, a_g, Parameters):
    # Apply the Dirichlet BCs to the pore fluid temperature.
    # Newmark-beta integration with constant accleration is used for unknown Dirichlet BC values.
    sub = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse

    if self.ID == 0:
      self.tf_global   [self.idx - sub] = a_g[0,0,self.idx,0]
      self.tfDot_global[self.idx - sub] = a_g[0,1,self.idx,0]
    elif self.ID == Parameters.ne - 1:
      self.tf_global   [self.idx - sub] = a_g[0,0,self.idx,1]
      self.tfDot_global[self.idx - sub] = a_g[0,1,self.idx,1]
    return
