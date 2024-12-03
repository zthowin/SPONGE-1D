#--------------------------------------------------------------------------------------
# Module housing individual assemblies and solvers for each variational equation.
#
# Author:       Richard Regueiro, Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   December 3, 2024
#--------------------------------------------------------------------------------------
import sys

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import classElement
except ImportError:
  sys.exit("MODULE WARNING. classElement.py not found, check configuration.")

try:
  import moduleFE
except ImportError:
  sys.exit("MODULE WARNING. moduleFE.py not found, check configuration.")

#--------------------------------------------------------------------------------------------------
# Function to compute stresses, strains, and internal state variables.
# ----------
# Arguments:
# ----------
# LM:          (int,   size: # element DOFs x # elements) location matrix
# g:           (float, size: # element DOFs x # 2)        Dirichlet BCs
# D:           (float, size: # DOFs)                      global DOF for displacement(s)
# V:           (float, size: # DOFs)                      global DOF for velocity(s)
# A:           (float, size: # DOFs)                      global DOF for acceleration(s)
# Parameters:  (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# stress_strain:   (float, size: # elements, 5,  # Gauss points)    stresses and strains
# isv:             (float, size: # elements, 13, # Gauss points)    internal state variables
#--------------------------------------------------------------------------------------------------
def get_SSISV(LM, g, D, V, A, Parameters):
  #---------------------------
  # Initialize storage arrays.
  #--------------------------- 
  stress_strain = np.zeros((Parameters.ne, 5,  Parameters.Gauss_Order), dtype=np.float64)
  isv           = np.zeros((Parameters.ne, 19, Parameters.Gauss_Order), dtype=np.float64)
  if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
    dt_el = np.zeros((Parameters.ne), dtype=np.float64)
  #--------------------
  # Loop over elements.
  #--------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    if Parameters.MMS:
      element.set_Coordinates(Parameters) # defines 'X', i.e., the nodal coordinates
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='G')
    if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
      dt_el[element_ID] = np.min(element.dt)
    #-------------------------------------------------------------
    # Compute remaining variables for stress-strain & ISV storage.
    #-------------------------------------------------------------
    element.get_sig11(Parameters)
    element.get_ps_E(Parameters)
    element.get_tau(Parameters)
    element.get_ns(Parameters)

    stress_strain[element_ID,0,:] = element.P11
    stress_strain[element_ID,1,:] = element.sig11
    stress_strain[element_ID,2,:] = element.J
    stress_strain[element_ID,3,:] = element.tau
    stress_strain[element_ID,4,:] = element.dvdX

    isv[element_ID,2,:]  = element.ps_E
    isv[element_ID,18,:] = element.Q

    if 't' in Parameters.Physics:
      element.get_ts()
      element.get_tsDot()
      element.get_etas(Parameters)
      element.get_dtsdX()
      element.get_qs(Parameters)
      isv[element_ID,7,:]  = element.ts
      isv[element_ID,9,:]  = element.etas
      isv[element_ID,11,:] = element.qs
      isv[element_ID,13,:] = element.tsDot 
    if 'pf' in Parameters.Physics:
      element.get_p_fDot()
      element.get_a_s()
      element.get_dp_fdX()
      element.get_ns(Parameters)
      element.get_nf()
      element.get_rhofR(Parameters)
      element.get_khat(Parameters)
      if 'uf' in Parameters.Physics:
        element.get_Qf(Parameters)
        element.get_DIV_Qf(Parameters)
      if 'tf' in Parameters.Physics:
        element.get_dnfdX()
        element.get_tf()
        element.get_tfDot()
        element.get_etaf(Parameters)
        element.get_dtfdX()
        element.get_qf(Parameters)
        isv[element_ID,8,:]  = element.tf
        isv[element_ID,10,:] = element.etaf
        isv[element_ID,12,:] = element.qf
        isv[element_ID,14,:] = element.tfDot
        isv[element_ID,17,:] = element.dnfdX
      if Parameters.DarcyBrinkman:
        element.get_DIV_FES(Parameters)
      element.get_vDarcy(Parameters)
      isv[element_ID,0,:] = element.get_Total_Pressure(Parameters)
      isv[element_ID,1,:] = element.get_Fluid_Pressure(Parameters)
      isv[element_ID,3,:] = element.ns
      isv[element_ID,4,:] = element.rhofR
      isv[element_ID,5,:] = element.khat
      isv[element_ID,6,:] = element.vDarcy
      isv[element_ID,15,:] = element.p_fDot
      isv[element_ID,16,:] = element.dp_fdX
      
      # if Parameters.integrationScheme == 'Predictor-corrector':
      #   element.get_rho(Parameters)
      #   element.get_rho_0(Parameters)
      #   element.get_dp_fDotdX()
  
  return stress_strain, isv
#--------------------------------------------------------------------------------------------------
# Function to assemble a system of variational equations.
# ----------
# Arguments:
# ----------
# LM:          (int,   size: # element DOFs x # elements) location matrix
# g:           (float, size: # element DOFs x # 2)        Dirichlet BCs
# F:           (float, size: varies)                      external force vectors
# D:           (float, size: # DOFs)                      global DOF for displacement(s)
# V:           (float, size: # DOFs)                      global DOF for velocity(s)
# A:           (float, size: # DOFs)                      global DOF for acceleration(s)
# Parameters:  (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR:          (float, size: # DOFs x # DOFs)             global stiffness matrix
# R:           (float, size: # DOFs)                      global residual
#--------------------------------------------------------------------------------------------------
def assemble_System(LM, g, F, D, V, A, Parameters):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R    = np.zeros((Parameters.ndof),                  dtype=np.float64)
  dR   = np.zeros((Parameters.ndof, Parameters.ndof), dtype=np.float64)
  #------------------------------------
  # Extract the individual Neumann BCs.
  #------------------------------------
  GEXT = F[0]
  if 'pf' in Parameters.Physics:
    HEXT = F[1]
  else:
    try:
      JEXT = F[1]
    except IndexError:
      pass
  try:
    JEXT = F[2]
    KEXT = F[3]
  except IndexError:
    pass
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    if Parameters.MMS:
      element.set_Coordinates(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='All')
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='G')
    if 'pf' in Parameters.Physics: 
      element.compute_internal_forces(Parameters, VariationalEq='H')
    if 'uf' in Parameters.Physics:
      element.compute_internal_forces(Parameters, VariationalEq='I')
    if 't' in Parameters.Physics:
      element.compute_internal_forces(Parameters, VariationalEq='J')
      if 'tf' in Parameters.Physics:
        element.compute_internal_forces(Parameters, VariationalEq='K')
    #-----------------------------------------------
    # Compute element mass (and stiffness) matrices.
    #-----------------------------------------------
    element.compute_tangents(Parameters, VariationalEq='G')
    if 'pf' in Parameters.Physics:
      element.compute_tangents(Parameters, VariationalEq='H')
    if 'uf' in Parameters.Physics:
      element.compute_tangents(Parameters, VariationalEq='I')
    if 't' in Parameters.Physics:
      element.compute_tangents(Parameters, VariationalEq='J')
      if 'tf' in Parameters.Physics:
        element.compute_tangents(Parameters, VariationalEq='K')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    if Parameters.Physics == 'u-pf':
      R_int = np.hstack((element.G_int, element.H_int))
      K_int = np.vstack((element.G_Mtx, element.H_Mtx))
    if Parameters.Physics == 'u-t':
      R_int = np.hstack((element.G_int, element.J_int))
      K_int = np.vstack((element.G_Mtx, element.J_Mtx))
    elif Parameters.Physics == 'u-uf-pf':
      R_int = np.hstack((element.G_int, element.I_int, element.H_int))
      K_int = np.vstack((element.G_Mtx, element.I_Mtx, element.H_Mtx))
    elif Parameters.Physics == 'u-pf-ts-tf':
      R_int = np.hstack((element.G_int, element.H_int, element.J_int, element.K_int))
      K_int = np.vstack((element.G_Mtx, element.H_Mtx, element.J_Mtx, element.K_Mtx))

    for i in range(element.numDOF):
      I = element.DOF[i]

      if I > -1:
        try:
          R[I] += R_int[i]
        except IndexError:
          pass

        for j in range(element.numDOF):
          J = element.DOF[j]
          if J > -1:
            try:
              dR[I,J] += K_int[i,j]
            except IndexError:
              pass
  #------------------------
  # Update global residual.
  #------------------------
  R -= GEXT
  if 'pf' in Parameters.Physics:
    R -= HEXT
  if 't' in Parameters.Physics:
    R -= JEXT
    if 'tf' in Parameters.Physics:
      R -= KEXT
  return dR, R
#--------------------------------------------------------------------------------------------------
# Function to solve linear momentum balance of the solid skeleton.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# GEXT:         (float, size: # skeleton DOFs)             external traction vector
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_s:         (float, size: # solid DOFs x # solid DOFs) global stiffness matrix
# R_s:          (float, size: # solid DOFs)                global residual
#--------------------------------------------------------------------------------------------------
def assemble_G(LM, g, GEXT, D, V, A, Parameters):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R_s  = np.zeros((Parameters.ndofS),                   dtype=np.float64)
  dR_s = np.zeros((Parameters.ndofS, Parameters.ndofS), dtype=np.float64)

  if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
    dt_el = np.zeros(int(Parameters.ne))
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    if Parameters.MMS:
      element.set_Coordinates(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='G')
    if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
      dt_el[element_ID] = np.min(element.dt)
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='G')
    #-----------------------------------------------
    # Compute element mass (and stiffness) matrices.
    #-----------------------------------------------
    element.compute_tangents(Parameters, VariationalEq='G')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    for i in range(Parameters.ndofSe):
      I = element.DOF[i]

      if I != -1:
        R_s[I] += element.G_int[i]

        for j in range(Parameters.ndofSe):
          J = element.DOF[j]

          if J != -1:
            dR_s[I,J] += element.G_Mtx[i,j]
  #------------------------
  # Update global residual.
  #------------------------
  # Old code for predictor-corrector algorithm
  # try:
  #   if Parameters.StarStar:
  #     if not Parameters.drainedApply:
  #       GEXT[Parameters.tractionDOF] = Parameters.Biot*(D[-1] - args[1][-1])*Parameters.Area
  #     else:
  #       GEXT[Parameters.tractionDOF] = 0
  # except AttributeError:
  #   pass
  R_s -= GEXT
  if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
    Parameters.dtnew = np.min(dt_el)

  return dR_s, R_s
#--------------------------------------------------------------------------------------------------
# Function to solve linear momentum balance of the solid skeleton. Used in predictor-corrector step.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # elements) Dirichlet BCs
# GEXT:         (float, size: # skeleton DOFs)             external traction vector
# D:            (float, size: # skeleton DOFs)             global IC for displacement
# V:            (float, size: # skeleton DOFs)             global IC for velocity
# A:            (float, size: # skeleton DOFs)             global IC for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_s:            (float, size: # skeleton DOFs)      LHS of variational equation
# R_s:             (float, size: # skeleton DOFs)      RHS of variational equation
#--------------------------------------------------------------------------------------------------
# def assemble_G_StarStar(LM, g, GEXT, D, V, A, Parameters, D_Last):
#   #-------------------------------------------------
#   # Initialize global residual and stiffness matrix.
#   #-------------------------------------------------
#   R_s  = np.zeros((Parameters.ndofS),                   dtype=np.float64)
#   dR_s = np.zeros((Parameters.ndofS, Parameters.ndofS), dtype=np.float64)
#   #------------------------
#   # Begin assembly process.
#   #------------------------
#   for element_ID in range(Parameters.ne):
#     #--------------------
#     # Initialize element.
#     #--------------------
#     element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
#     element.set_Gauss_Points(Parameters)
#     element.set_Gauss_Weights()
#     element.set_Jacobian(Parameters)
#     element.evaluate_Shape_Functions(Parameters)
#     element.get_Global_DOF(LM)
#     element.set_Global_Solutions(D, V, A, Parameters, D_Last)
#     element.apply_Local_BC(g, Parameters)
#     #---------------------------
#     # Compute element variables.
#     #---------------------------
#     element.compute_variables(Parameters, VariationalEq='G')
#     if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
#       dt_el[element_ID] = np.min(element.dt)
#     #----------------------------------------
#     # Compute element internal force vectors.
#     #----------------------------------------
#     element.compute_internal_forces(Parameters, VariationalEq='G')
#     #-----------------------------------------------
#     # Compute element mass (and stiffness) matrices.
#     #-----------------------------------------------
#     element.compute_tangents(Parameters, VariationalEq='G')
#     #---------------------------------------------------------------
#     # Assemble element contributions to global residual and tangent.
#     #---------------------------------------------------------------
#     for i in range(Parameters.ndofSe):
#       I = element.DOF[i]

#       if I != -1:
#         R_s[I] += element.G_int[i]

#         for j in range(Parameters.ndofSe):
#           J = element.DOF[j]

#           if J != -1:
#             dR_s[I,J] += element.G_Mtx[i,j]
#   #------------------------
#   # Update global residual.
#   #------------------------
#   if not Parameters.drainedApply:
#     GEXT[Parameters.tractionDOF] = (D[-1] - D_Last[-1])*Parameters.Area
#   else:
#     GEXT[Parameters.tractionDOF] = 0
    
#   R_s += GEXT
#   if Parameters.isAdaptiveStepping and Parameters.integrationScheme == 'Central-difference':
#     Parameters.dtnew = np.min(dt_el)

#   return dR_s, R_s
#--------------------------------------------------------------------------------------------------
# Function to assemble the variational equation of the mass balance of the mixture.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# HEXT:         (float, size: # pressure DOFs)             external flux vector
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_p:            (float, size: # pressure DOFs x # pressure DOFs) global stiffness matrix
# R_p:             (float, size: # pressure DOFs)                   global residual
#--------------------------------------------------------------------------------------------------
def assemble_H(LM, g, HEXT, D, V, A, Parameters, *args):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R_p  = np.zeros((Parameters.ndofP),                   dtype=np.float64)
  dR_p = np.zeros((Parameters.ndofP, Parameters.ndofP), dtype=np.float64)
  #------------------------------------------------------------------
  # Set increment 'inc' for the pressure DOF.
  #
  # This allows element.H_Mtx to be size 
  # ndofPe x (ndofSe + ndofFe + ndofPe) or ndofPe x (ndofSe + ndofPe)
  # or ndofPe x (ndofSe + nodfPe + ndofTse + ndofTfe) in, e.g., the
  # implicit scheme (see _ElementTangents), such that here we insert,
  # e.g., M_{pf,pf} components into the correct position of 
  # element.H_Mtx.
  #------------------------------------------------------------------
  if 'uf' not in Parameters.Physics:
    inc = Parameters.ndofSe
  else:
    inc = Parameters.ndofSe + Parameters.ndofFe
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    if Parameters.MMS:
      element.set_Coordinates(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='H')
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='H')
    #-----------------------------------------------
    # Compute element mass (and stiffness) matrices.
    #-----------------------------------------------
    try:
      #----------------------------------------------------------------
      # These lines are used in the central-difference time integration
      # scheme, where the updated solid acceleration is passed in as 
      # an optional argument.
      # 
      # The update occurs here since the internal force vectors rely
      # on solid skeleton acceleration a_n, but the tangent
      # k_pu_Hx (x=1,2,4) must be multiplied by a_{n+1}. 
      #----------------------------------------------------------------
      element.set_Global_Solutions(D, V, args[0], Parameters)
      element.apply_Local_BC(g, Parameters)
    except IndexError:
      pass
    element.compute_tangents(Parameters, VariationalEq='H')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    for i in range(Parameters.ndofPe):
      I = element.DOFP[i]

      if I >= 0:
        R_p[I] += element.H_int[i]

        for j in range(Parameters.ndofPe):
          J = element.DOFP[j]

          if J >= 0:
            dR_p[I,J] += element.H_Mtx[i,j+inc]
  #------------------------
  # Update global residual.
  #------------------------
  R_p -= HEXT

  return dR_p, R_p
#--------------------------------------------------------------------------------------------------
# Function to solve linear momentum balance of the pore fluid.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_f:            (float, size: # fluid DOFs x # fluid DOFs) global stiffness matrix
# R_f:             (float, size: # fluid DOFs)                global residual
#--------------------------------------------------------------------------------------------------
def assemble_I(LM, g, D, V, A, Parameters):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R_f  = np.zeros((Parameters.ndofF),                   dtype=np.float64)
  dR_f = np.zeros((Parameters.ndofF, Parameters.ndofF), dtype=np.float64)
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    if Parameters.MMS:
      element.set_Coordinates(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='I')
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='I')
    #-----------------------------
    # Compute element mass matrix.
    #-----------------------------
    element.compute_tangents(Parameters, VariationalEq='I')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    for i in range(Parameters.ndofFe):
      I = element.DOFF[i]

      if I >= 0:
        R_f[I] += element.I_int[i]

        for j in range(Parameters.ndofFe):
          J = element.DOFF[j]

          if J >= 0:
            dR_f[I,J] += element.I_Mtx[i,j+Parameters.ndofSe]

  return dR_f, R_f
#--------------------------------------------------------------------------------------------------
# Function to solve balance of energy of the solid.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_Ts:            (float, size: # solid temp DOFs x # solid temp DOFs) global stiffness matrix
# R_Ts:             (float, size: # solid temp DOFs)                     global residual
#--------------------------------------------------------------------------------------------------
def assemble_J(LM, g, F_HS, D, V, A, Parameters):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R_ts  = np.zeros((Parameters.ndofTs),                    dtype=np.float64)
  dR_ts = np.zeros((Parameters.ndofTs, Parameters.ndofTs), dtype=np.float64)
  #------------------------------------------------------------------
  # Set increment 'inc' for the solid temperature DOF.
  #
  # This allows element.J_Mtx to be size 
  # ndofTse x (ndofSe + ndofTse) or
  # ndofTse x (ndofSe + ndofPe + ndofTse + ndofTfe) or
  # ndofTse x (ndofSe + ndofFe + ndofFe + ndofTse + ndofTfe) in,
  # e.g., the implicit scheme, such that here we insert, e.g.,
  # M_{ts,ts} components into the correct position of element.J_Mtx.
  #------------------------------------------------------------------
  if 'pf' not in Parameters.Physics:
    inc = Parameters.ndofSe
  else:
    if 'uf' not in Parameters.Physics:
      inc = Parameters.ndofSe + Parameters.ndofPe
    else:
      inc = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='J')
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='J')
    #-----------------------------
    # Compute element mass matrix.
    #-----------------------------
    element.compute_tangents(Parameters, VariationalEq='J')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    for i in range(Parameters.ndofTse):
      I = element.DOFTs[i]

      if I >= 0:
        R_ts[I] += element.J_int[i]

        for j in range(Parameters.ndofTse):
          J = element.DOFTs[j]

          if J >= 0:
            dR_ts[I,J] += element.J_Mtx[i,j+inc]

  R_ts -= F_HS
  
  return dR_ts, R_ts
#--------------------------------------------------------------------------------------------------
# Function to solve balance of energy of the pore fluid.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR_tf:            (float, size: # fluid temp DOFs x # fluid temp DOFs)  global stiffness matrix
# R_tf:             (float, size: # fluid temp DOFs)                      global residual
#--------------------------------------------------------------------------------------------------
def assemble_K(LM, g, F_HF, D, V, A, Parameters):
  #-------------------------------------------------
  # Initialize global residual and stiffness matrix.
  #-------------------------------------------------
  R_tf  = np.zeros((Parameters.ndofTf),                    dtype=np.float64)
  dR_tf = np.zeros((Parameters.ndofTf, Parameters.ndofTf), dtype=np.float64)
  #------------------------------------------------------------------
  # Set increment 'inc' for the pore fluid temperature DOF.
  # 
  # This allows element.K_Mtx to be size 
  # ndofTfe x (ndofSe + ndofPe + ndofTse + ndofTfe) or
  # ndofTfe x (ndofSe + nodfFe + ndofPe + ndofTse + ndofTfe) in, e.g.,
  # the implicit scheme, such that here we insert, e.g., M_{tf,tf}
  # components into the correct position of element.H_Ktx.
  #------------------------------------------------------------------
  if 'uf' not in Parameters.Physics:
    inc = Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse
  else:
    inc = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse
  #------------------------
  # Begin assembly process.
  #------------------------
  for element_ID in range(Parameters.ne):
    #--------------------
    # Initialize element.
    #--------------------
    element = classElement.Element(a_GaussOrder=Parameters.Gauss_Order, a_ID=element_ID)
    element.set_Gauss_Points(Parameters)
    element.set_Gauss_Weights()
    element.set_Jacobian(Parameters)
    element.evaluate_Shape_Functions(Parameters)
    element.get_Global_DOF(LM)
    element.set_Global_Solutions(D, V, A, Parameters)
    element.apply_Local_BC(g, Parameters)
    #---------------------------
    # Compute element variables.
    #---------------------------
    element.compute_variables(Parameters, VariationalEq='K')
    #----------------------------------------
    # Compute element internal force vectors.
    #----------------------------------------
    element.compute_internal_forces(Parameters, VariationalEq='K')
    #-----------------------------
    # Compute element mass matrix.
    #-----------------------------
    element.compute_tangents(Parameters, VariationalEq='K')
    #---------------------------------------------------------------
    # Assemble element contributions to global residual and tangent.
    #---------------------------------------------------------------
    for i in range(Parameters.ndofTfe):
      I = element.DOFTf[i]

      if I >= 0:
        R_tf[I] += element.K_int[i]

        for j in range(Parameters.ndofTfe):
          J = element.DOFTf[j]

          if J >= 0:
            dR_tf[I,J] += element.K_Mtx[i,j+inc]

  R_tf -= F_HF

  return dR_tf, R_tf
#--------------------------------------------------------------------------------------------------
# Function to update the global stiffness matrix and global residual for the Lagrange multiplier
# no-flux constraint.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# g:            (float, size: # element DOFs x # 2)        Dirichlet BCs
# D:            (float, size: # DOFs)                      global DOF for displacement
# V:            (float, size: # DOFs)                      global DOF for velocity
# A:            (float, size: # DOFs)                      global DOF for acceleration
# dR:           (float, size: # DOFs x # DOFs)             global stiffness matrix
# R:            (float, size: # DOFs)                      global residual
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# dR:           (float, size: # DOFs x # DOFs)             augmented global stiffness matrix
# R:            (float, size: # DOFs)                      augmented global residual
#--------------------------------------------------------------------------------------------------
def assemble_Lagrange(LM, g, D, V, A, dR, R, Parameters):
  #-------------------------------
  # Only apply at the top for now.
  #-------------------------------
  STop = Parameters.ndofS - 2
  FTop = Parameters.ndofS + Parameters.ndofF - 1
  #----------------------------
  # Initialize topmost element.
  #----------------------------
  e = classElement.Element(a_GaussOrder=4, a_ID=Parameters.ne - 1)
  e.set_Gauss_Points(Parameters)
  e.set_Gauss_Weights()
  e.set_Jacobian(Parameters)
  e.evaluate_Shape_Functions(Parameters)
  e.get_Global_DOF(LM)
  e.set_Global_Solutions(D, V, A, Parameters)
  e.apply_Local_BC(g, Parameters)
  #------------------------------------------------------------------------
  # Grab the nodally interpolated values for Jacobian and volume fractions.
  #------------------------------------------------------------------------
  e.get_dudX()
  e.get_J()
  e.J = np.linalg.solve(e.N_intp.T, e.J) # resets J
  e.get_ns(Parameters)
  e.get_nf()
  #---------------------------------------------------------
  # Assume linear shape functions for Lagrange test function
  # such that NL[\xi = 1] = 1.
  #---------------------------------------------------------
  L_EXT = e.nf[-1]*(V[FTop] - V[STop])*Parameters.Area
#  L_EXT = (V[FTop] - V[STop])*Parameters.Area
  #-----------------------------
  # Insert into global residual.
  #-----------------------------
  R[-1] += L_EXT
  #---------------------------------------------
  # Add contribution to pore fluid displacement.
  #---------------------------------------------
  R[FTop] += D[-1]*Parameters.Area
  #-------------------------------------------------------
  # Compute K_uf,lambda, assumes linear test functions for
  # the pore fluid displacement.
  #-------------------------------------------------------
  K_ufl = Parameters.beta*(Parameters.dt**2)*Parameters.Area
  #-------------------
  # Compute K_lambda,u
  #--------------------
  K_lu_B  = (e.ns[-1]/e.J[-1])*(V[FTop] - V[STop])*Parameters.beta*(Parameters.dt**2)
  K_lu_N  = -e.nf[-1]*Parameters.gamma*Parameters.dt
#  K_lu_N  = -Parameters.gamma*Parameters.dt
  K_lu_B *= Parameters.Area
  K_lu_N *= Parameters.Area
  #-------------------------------------------------------
  # Compute K_lambda,uf, assumes linear test functions for
  # the pore fluid displacement.
  #-------------------------------------------------------
  K_luf  = e.nf[-1]*Parameters.gamma*Parameters.dt
#  K_luf  = Parameters.gamma*Parameters.dt
  K_luf *= Parameters.Area
  #---------------------------
  # Insert into global tangent
  #---------------------------
  dR[FTop, -1]     = K_ufl
  dR[-1, STop + 1] = K_lu_B # Gradient     \BEU goes in at the Hermite gradient DOF
  dR[-1, STop]     = K_lu_N # Non-gradient \NEU goes in at the Hermite DOF
  dR[-1, FTop]     = K_luf
  return dR, R
#-----------------------------------------------------------
# Helper function to solve the system of implicit equations.
#-----------------------------------------------------------
def solve_System(dR, R, Parameters):
  try:
    return np.linalg.solve(dR, -R)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution of system variables.")
    raise FloatingPointError
    return
#---------------------------------------------------------
# Helper function to solve the variational equation of the
# balance of momentum of the solid/mixture.
#---------------------------------------------------------
def solve_G(dR, R, Parameters):
  #-------------------------------------
  # Update solution variable at t_{n+1}.
  #-------------------------------------
  try:
    if Parameters.SolidLumping:
      diag = np.diag(dR)
      np.fill_diagonal(dR, 1/diag)
      S_s  = np.dot(-R, dR)

    else:
      S_s = np.linalg.solve(dR, -R)

  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution for solid skeleton displacement.")
    raise FloatingPointError

  return S_s
#---------------------------------------------------------
# Helper function to solve the variational equation of the
# balance of mass of the mixture.
#---------------------------------------------------------
def solve_H(dR, R, Parameters):
  #-------------------------------------
  # Update solution variable at t_{n+1}.
  #-------------------------------------
  try:
    if Parameters.PressureLumping:
      diag = np.diag(dR)
      np.fill_diagonal(dR, 1/diag)
      S_pf  = np.dot(-R, dR)

    else:
      S_pf = np.linalg.solve(dR, -R)

  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution for pore fluid pressure.")
    raise FloatingPointError

  return S_pf
#---------------------------------------------------------
# Helper function to solve the variational equation of the
# balance of momentum of the fluid.
#---------------------------------------------------------
def solve_I(dR, R, Parameters):
  #-------------------------------------
  # Update solution variable at t_{n+1}.
  #-------------------------------------
  try:
    if Parameters.FluidLumping:
      diag = np.diag(dR)
      np.fill_diagonal(dR, 1/diag)
      S_f  = np.dot(-R, dR)

    else:
      S_f = np.linalg.solve(dR, -R)

  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution for pore fluid displacement.")
    raise FloatingPointError

  return S_f
#---------------------------------------------------------
# Helper function to solve the variational equation of the
# balance of energy of the solid.
#---------------------------------------------------------
def solve_J(dR, R, Parameters):
  #-------------------------------------
  # Update solution variable at t_{n+1}.
  #-------------------------------------
  try:
    if Parameters.SolidTempLumping:
      diag = np.diag(dR)
      np.fill_diagonal(dR, 1/diag)
      S_ts  = np.dot(-R, dR)

    else:
      S_ts = np.linalg.solve(dR, -R)

  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution for solid energy.")
    raise FloatingPointError

  return S_ts
#---------------------------------------------------------
# Helper function to solve the variational equation of the
# balance of energy of the pore fluid.
#---------------------------------------------------------
def solve_K(dR, R, Parameters):
  #-------------------------------------
  # Update solution variable at t_{n+1}.
  #-------------------------------------
  try:
    S_tf = np.linalg.solve(dR, -R)

  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Could not compute solution for solid energy.")
    raise FloatingPointError

  return S_tf

