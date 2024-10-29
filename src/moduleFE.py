#----------------------------------------------------------------------------------------
# Module containing basic FEM helper functions such as initializing the DOFs and
# assigning boundary conditions.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 15, 2024
#----------------------------------------------------------------------------------------
import sys

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

#-------------------------
# Begin top-level methods.
#-------------------------
#------------------------------------------------------------------------------------------------
# Function to construct a location matrix for element-global DOF mappings.
# ----------
# Arguments:
# ----------
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# LM:         (int, size: # element DOFs x # elements)  location matrix
#-------------------------------------------------------------------------------------------------
def initLM(Parameters):
  
  if Parameters.Physics == 'u':
    Parameters.ndofe = Parameters.ndofSe
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3]
      elif Parameters.Element_Type == 'Q3':
        LM[:,e] = [3*e , 3*e + 1, 3*e + 2, 3*e + 3]
      elif Parameters.Element_Type == 'Q2':
        LM[:,e] = [2*e, 2*e + 1, 2*e + 2]
      elif Parameters.Element_Type == 'Q1':
        LM[:,e] = [e, e + 1]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics, or not implemented.")
    #-----------------------------
    # Assign total number of DOFs.
    #-----------------------------
    Parameters.ndof  = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS  = Parameters.ndof
    Parameters.ndofF  = 0
    Parameters.ndofP  = 0
    Parameters.ndofTs = 0
    Parameters.ndofTf = 0

  elif Parameters.Physics == 'u-pf':
    Parameters.ndofe = Parameters.ndofSe + Parameters.ndofPe
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H-P1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,\
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3]
      elif Parameters.Element_Type == 'Q3-P1':
        LM[:,e] = [3*e, 3*e + 1, 3*e + 2, 3*e + 3,\
                   3*Parameters.ne + e + 1, 3*Parameters.ne + e + 2]
      elif Parameters.Element_Type == 'Q2-P1':
        LM[:,e] = [2*e, 2*e + 1, 2*e + 2,\
                   2*Parameters.ne + e + 1, 2*Parameters.ne + e + 2]
      elif Parameters.Element_Type == 'Q1-P1':
        LM[:,e] = [e, e + 1,\
                   Parameters.ne + e + 1, Parameters.ne + e + 2]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics, or not implemented.")
    #-----------------------------
    # Assign total number of DOFs.
    #-----------------------------
    Parameters.ndof  = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS = LM[Parameters.ndofSe - 1, -1] + 1
    #--------------------------------------
    # Assign total number of pressure DOFs.
    #--------------------------------------
    Parameters.ndofP  = Parameters.ndof - Parameters.ndofS
    Parameters.ndofF  = 0
    Parameters.ndofTs = 0
    Parameters.ndofTf = 0

  elif Parameters.Physics == 'u-t':
    Parameters.ndofe = Parameters.ndofSe + Parameters.ndofTse
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H-T1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,\
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3]
      elif Parameters.Element_Type == 'Q3-T1':
        LM[:,e] = [3*e, 3*e + 1, 3*e + 2, 3*e + 3,\
                   3*Parameters.ne + e + 1, 3*Parameters.ne + e + 2]
      elif Parameters.Element_Type == 'Q2-T1':
        LM[:,e] = [2*e, 2*e + 1, 2*e + 2,\
                   2*Parameters.ne + e + 1, 2*Parameters.ne + e + 2]
      elif Parameters.Element_Type == 'Q1-T1':
        LM[:,e] = [e, e + 1,\
                   Parameters.ne + e + 1, Parameters.ne + e + 2]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics, or not implemented.")
    #-----------------------------
    # Assign total number of DOFs.
    #-----------------------------
    Parameters.ndof  = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS = LM[Parameters.ndofSe - 1, -1] + 1
    #--------------------------------------
    # Assign total number of pressure DOFs.
    #--------------------------------------
    Parameters.ndofP  = 0
    Parameters.ndofF  = 0
    Parameters.ndofTs = Parameters.ndof - Parameters.ndofS
    Parameters.ndofTf = 0

  elif Parameters.Physics == 'u-uf-pf':
    Parameters.ndofe = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H-Q3H-P1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*(Parameters.ne + e) + 2, 2*(Parameters.ne + e) + 4, 2*(Parameters.ne + e) + 3, 2*(Parameters.ne + e) + 5,
                   4*Parameters.ne + e + 4, 4*Parameters.ne + e + 5]
      elif Parameters.Element_Type == 'Q3-Q3-P1':
        LM[:,e] = [3*e, 3*e + 1, 3*e + 2, 3*e + 3,
                   3*(Parameters.ne + e) + 1, 3*(Parameters.ne + e) + 2, 3*(Parameters.ne + e) + 3, 3*(Parameters.ne + e) + 4,
                   6*Parameters.ne + e + 2, 6*Parameters.ne + e + 3]
      elif Parameters.Element_Type == 'Q3H-Q2-P1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*(Parameters.ne + e) + 2, 2*(Parameters.ne + e) + 3, 2*(Parameters.ne + e) + 4,
                    4*Parameters.ne + e + 3, 4*Parameters.ne + e + 4]
      elif Parameters.Element_Type == 'Q3H-Q1-P1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3,
                   3*Parameters.ne + e + 3, 3*Parameters.ne + e + 4] 
      elif Parameters.Element_Type == 'Q2-Q2-P1':
        LM[:,e] = [2*e, 2*e + 1, 2*e + 2,
                   2*(Parameters.ne + e) + 1, 2*(Parameters.ne + e) + 2, 2*(Parameters.ne + e) + 3,\
                   4*Parameters.ne + e + 2, 4*Parameters.ne + e + 3]
      elif Parameters.Element_Type == 'Q2-Q1-P1':
        LM[:,e] = [2*e, 2*e + 1, 2*e + 2,
                   2*Parameters.ne + e + 1, 2*Parameters.ne + e + 2,
                   3*Parameters.ne + e + 2, 3*Parameters.ne + e + 3]
      elif Parameters.Element_Type == 'Q1-Q1-P1':
        LM[:,e] = [e, e + 1,
                   Parameters.ne + e + 1, Parameters.ne + e + 2,
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics, or not implemented.")
    #----------------------
    # Assign total of DOFs.
    #----------------------
    Parameters.ndof = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS = LM[Parameters.ndofSe - 1, -1] + 1
    #-----------------------------------
    # Assign total number of fluid DOFs.
    #-----------------------------------
    Parameters.ndofF = LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] + 1 - Parameters.ndofS
    #--------------------------------------
    # Assign total number of pressure DOFs.
    #--------------------------------------
    Parameters.ndofP = Parameters.ndof - Parameters.ndofS - Parameters.ndofF

    Parameters.ndofTs = 0
    Parameters.ndofTf = 0

  elif Parameters.Physics == 'u-pf-ts-tf':
    Parameters.ndofe = Parameters.ndofSe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H-P1-T2-T2':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3,
                   3*(Parameters.ne + e + 1) - e, 3*(Parameters.ne + e + 1) + 1 - e, 3*(Parameters.ne + e + 1) + 2 - e,
                   5*(Parameters.ne + e) + 4 - 3*e , 5*(Parameters.ne + e) + 5 - 3*e, 5*(Parameters.ne + e) + 6 - 3*e]
      elif Parameters.Element_Type == 'Q3H-P1-T1-T1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3,
                   3*Parameters.ne + e + 3, 3*Parameters.ne + e + 4,
                   4*Parameters.ne + e + 4, 4*Parameters.ne + e + 5]
      elif Parameters.Element_Type == 'Q3-P1-T1-T1':
        LM[:,e] = [3*e, 3*e + 1, 3*e + 2, 3*e + 3,
                   3*Parameters.ne + e + 1, 3*Parameters.ne + e + 2,
                   4*Parameters.ne + e + 2, 4*Parameters.ne + e + 3,
                   5*Parameters.ne + e + 3, 5*Parameters.ne + e + 4]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics (check for Hermite cubic solid element).")
    #----------------------
    # Assign total of DOFs.
    #----------------------
    Parameters.ndof = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS = LM[Parameters.ndofSe - 1, -1] + 1
    #--------------------------------------
    # Assign total number of pressure DOFs.
    #--------------------------------------
    Parameters.ndofP = LM[Parameters.ndofSe + Parameters.ndofPe - 1, -1] + 1 - Parameters.ndofS
    #-----------------------------------------------
    # Assign total number of solid temperature DOFs.
    #-----------------------------------------------
    Parameters.ndofTs = LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse - 1, -1] + 1 - Parameters.ndofS - Parameters.ndofP
    #-----------------------------------------------
    # Assign total number of fluid temperature DOFs.
    #-----------------------------------------------
    Parameters.ndofTf = LM[-1, -1] + 1 - Parameters.ndofS - Parameters.ndofP - Parameters.ndofTs

    Parameters.ndofF = 0

  elif Parameters.Physics == 'u-uf-pf-ts-tf':
    Parameters.ndofe = Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe
    LM = np.zeros((Parameters.ndofe, Parameters.ne), dtype=np.int32)
    for e in range(0, Parameters.ne):
      if Parameters.Element_Type == 'Q3H-Q3H-P1-T1-T1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*(Parameters.ne + e) + 2, 2*(Parameters.ne + e) + 4, 2*(Parameters.ne + e) + 3, 2*(Parameters.ne + e) + 5,
                   4*Parameters.ne + e + 4, 4*Parameters.ne + e + 5,
                   5*Parameters.ne + e + 5, 5*Parameters.ne + e + 6,
                   6*Parameters.ne + e + 6, 6*Parameters.ne + e + 7]
      elif Parameters.Element_Type == 'Q3H-Q1-P1-T1-T1':
        LM[:,e] = [2*e, 2*e + 2, 2*e + 1, 2*e + 3,
                   2*Parameters.ne + e + 2, 2*Parameters.ne + e + 3,
                   3*Parameters.ne + e + 3, 3*Parameters.ne + e + 4,
                   4*Parameters.ne + e + 4, 4*Parameters.ne + e + 5,
                   5*Parameters.ne + e + 5, 5*Parameters.ne + e + 6]
      else:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nElement type not consistent with Physics (check for Hermite cubic solid element).")
    #----------------------
    # Assign total of DOFs.
    #----------------------
    Parameters.ndof = LM[-1, -1] + 1
    #-----------------------------------
    # Assign total number of solid DOFs.
    #-----------------------------------
    Parameters.ndofS = LM[Parameters.ndofSe - 1, -1] + 1
    #-----------------------------------
    # Assign total number of fluid DOFs.
    #-----------------------------------
    Parameters.ndofF = LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] + 1 - Parameters.ndofS
    #--------------------------------------
    # Assign total number of pressure DOFs.
    #--------------------------------------
    Parameters.ndofP = LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 1, -1] + 1 - Parameters.ndofS - Parameters.ndofF
    #-----------------------------------------------
    # Assign total number of solid temperature DOFs.
    #-----------------------------------------------
    Parameters.ndofTs = LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse - 1, -1] + 1 - Parameters.ndofS - Parameters.ndofF - Parameters.ndofP
    #-----------------------------------------------
    # Assign total number of fluid temperature DOFs.
    #-----------------------------------------------
    Parameters.ndofTf = LM[-1, -1] + 1 - Parameters.ndofS - Parameters.ndofF - Parameters.ndofP - Parameters.ndofTs
  
  Parameters.nNode   = Parameters.ndof
  Parameters.nNodeS  = Parameters.ndofS
  Parameters.nNodeP  = Parameters.ndofP
  Parameters.nNodeF  = Parameters.ndofF
  Parameters.nNodeTs = Parameters.ndofTs
  Parameters.nNodeTf = Parameters.ndofTf

  return LM
#------------------------------------------------------------------------------------------------
# Function to set up Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initDirichletBCs(a_LM, Parameters):
  #------------------------------------------------------------------
  # Modify DOF for solid displacement or velocity boundary condition.
  #------------------------------------------------------------------
  if Parameters.solidDisplacementApply:
    a_LM = initSolidDisplacementBC(a_LM, Parameters)

  if Parameters.solidVelocityApply:
    a_LM = initSolidVelocityBC(a_LM, Parameters)
  #----------------
  # Pore fluid BCs.
  #----------------
  if 'uf' in Parameters.Physics:
    #-----------------------------------------------------------
    # Modify DOF for pore fluid displacement boundary condition.
    #-----------------------------------------------------------
    if Parameters.fluidDisplacementApply:
      a_LM = initFluidDisplacementBC(a_LM, Parameters)
    #-------------------------------------------------------
    # Modify DOF for pore fluid velocity boundary condition.
    #-------------------------------------------------------
    if Parameters.fluidVelocityApply:
      a_LM = initFluidVelocityBC(a_LM, Parameters)
    else:
      Parameters.LagrangeApply = False
  #---------------------------------------------
  # Modify DOF for pressure boundary conditions.
  #---------------------------------------------
  if 'pf' in Parameters.Physics:
    if Parameters.pressureApply:
      a_LM = initFluidPressureBC(a_LM, Parameters)
  #--------------------------------------------------
  # Modify DOFs for temperature Dirichlet conditions.
  #--------------------------------------------------
  if 'ts' in Parameters.Physics or Parameters.Physics == 'u-t':
    if Parameters.solidTempApply:
      a_LM = initSolidTempBC(a_LM, Parameters)

  if 'tf' in Parameters.Physics:
    if Parameters.fluidTempApply:
      a_LM = initFluidTempBC(a_LM, Parameters)
  #-----------------------
  # So that it looks nice.
  #-----------------------
  a_LM[a_LM < -1] = -1

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to initialize Neumann BCs.
# ----------
# Arguments:
# ----------
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# GEXT:       (int, size: # DOFs or # solid DOFs)                  external force vector for G
# HEXT:       (int, size: # DOFs or # pore fluid pressure DOFs)    external force vector for H
# JEXT:       (int, size: # DOFs or # solid temperature DOFs)      external force vector for J
# KEXT:       (int, size: # DOFs or # pore fluid temperature DOFs) external force vector for K
#-------------------------------------------------------------------------------------------------
def initNeumannBCs(Parameters):
  #--------------------------------------------
  # Set vector for traction boundary condition.
  #--------------------------------------------
  if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
    GEXT = np.zeros((Parameters.ndof))
  else:
    GEXT = np.zeros((Parameters.ndofS), dtype=np.float64)
  if Parameters.tractionApply:
    Parameters.tractionDOFBot = 0
    if 'Q3H' in Parameters.Element_Type:
      Parameters.tractionDOFTop = Parameters.ndofS - 2
    else:
      Parameters.tractionDOFTop = Parameters.ndofS - 1
    if Parameters.solidDisplacementApply:
      if Parameters.tractionLocation in Parameters.solidDisplacementLocation or Parameters.solidDisplacementLocation in Parameters.tractionLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid Dirichlet and solid Neumman boundary conditions location conflict.")
    if Parameters.solidVelocityApply:
      if Parameters.tractionLocation in Parameters.solidVelocityLocation or Parameters.solidVelocityLocation in Parameters.tractionLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid Dirichlet and solid Neumman boundary conditions location conflict.")
  #---------------------------------------------------
  # Modify DOF for pore fluid flux boundary condition.
  #---------------------------------------------------
  if 'pf' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      if Parameters.Physics == 'u-pf' or Parameters.Physics == 'u-pf-ts-tf':
        Parameters.fluxDOFTop = Parameters.ndofS + Parameters.ndofP - 1
        Parameters.fluxDOFBot = Parameters.ndofS
      elif 'uf' in Parameters.Physics:
        Parameters.fluxDOFTop = Parameters.ndofS + Parameters.ndofF + Parameters.ndofP - 1
        Parameters.fluxDOFBot = Parameters.ndofS + Parameters.ndofF
      HEXT                    = np.zeros((Parameters.ndof), dtype=np.float64)
    else:
      Parameters.fluxDOFBot = 0
      Parameters.fluxDOFTop = -1
      HEXT                  = np.zeros((Parameters.ndofP), dtype=np.float64)
    if Parameters.fluxApply and Parameters.pressureApply:
      if Parameters.pressureLocation in Parameters.fluxLocation or Parameters.fluxLocation in Parameters.pressureLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid Dirichlet and pore fluid pressure Neumann boundary condition location conflict.")
  #-----------------------------------------------
  # Modify DOFs for heat flux boundary conditions.
  #-----------------------------------------------
  if 't' in Parameters.Physics:
    if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
      Parameters.solidHeatFluxDOFTop = Parameters.ndofS + Parameters.ndofP + Parameters.ndofTs - 1
      Parameters.solidHeatFluxDOFBot = Parameters.ndofS + Parameters.ndofP
      JEXT                           = np.zeros((Parameters.ndof), dtype=np.float64)
    else:
      JEXT                           = np.zeros((Parameters.ndofTs), dtype=np.float64)
      Parameters.solidHeatFluxDOFTop = Parameters.ndofTs - 1
      Parameters.solidHeatFluxDOFBot = 0
    if Parameters.solidHeatFluxApply and Parameters.solidTempApply:
      if Parameters.solidHeatFluxLocation in Parameters.solidTempLocation or Parameters.solidTempLocation in Parameters.solidHeatFluxLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature Dirichlet and solid temperature Neumman boundary conditions location conflict.")
    if 'tf' in Parameters.Physics:
      if Parameters.integrationScheme == 'Newmark-beta' or Parameters.integrationScheme == 'Trapezoidal':
        Parameters.fluidHeatFluxDOFTop = Parameters.ndofS + Parameters.ndofP + Parameters.ndofTs + Parameters.ndofTf - 1
        Parameters.fluidHeatFluxDOFBot = Parameters.ndofS + Parameters.ndofP + Parameters.ndofTs
        KEXT                           = np.zeros((Parameters.ndof), dtype=np.float64)
      else:
        KEXT                           = np.zeros((Parameters.ndofTf), dtype=np.float64)
        Parameters.fluidHeatFluxDOFTop = Parameters.ndofTf - 1
        Parameters.fluidHeatFluxDOFBot = 0
      if Parameters.fluidHeatFluxApply and Parameters.fluidTempApply:
        if Parameters.fluidHeatFluxLocation in Parameters.fluidTempLocation or Parameters.fluidTempLocation in Parameters.fluidHeatFluxLocation:
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid temperature Dirichlet and pore fluid temperature Neumman boundary conditions location conflict.")

  if Parameters.Physics == 'u':
    return GEXT
  elif Parameters.Physics == 'u-t':
    return GEXT, JEXT
  elif Parameters.Physics == 'u-pf' or Parameters.Physics == 'u-uf-pf':
    return GEXT, HEXT
  elif 'tf' in Parameters.Physics:
    return GEXT, HEXT, JEXT, KEXT
#------------------------------------------------------------------------------------------------
# Function to apply any initial conditions.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)  location matrix
# Parameters: (object)                                  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D:          (float, size: # DOFs)  displacements, pressures, temperatures
# V:          (float, size: # DOFs)  velocities, time derivatives on pressures & temperatures
# A:          (float, size: # DOFs)  accelerations
#-------------------------------------------------------------------------------------------------
def applyIC(a_LM, Parameters):
  #------------------------------------------------------------
  # Set the initial conditions (assumed stationary at t = 0 s).
  #------------------------------------------------------------
  D = np.zeros((Parameters.ndof), dtype=np.float64)
  V = np.zeros((Parameters.ndof), dtype=np.float64)
  A = np.zeros((Parameters.ndof), dtype=np.float64)
  #-------------------------------------
  # Set the initial pore fluid pressure.
  #-------------------------------------
  if Parameters.Physics == 'u-pf':
    startDOF = a_LM[Parameters.ndofSe, 0]
    if startDOF < 0:
      startDOF = a_LM[Parameters.ndofSe + 1, 0]
    for dof in range(startDOF, Parameters.ndof):
      D[dof] += Parameters.p_f0
  #-----------------------------------
  # Set the initial solid temperature.
  #-----------------------------------
  elif Parameters.Physics == 'u-t':
    startDOF = a_LM[Parameters.ndofSe, 0]
    if startDOF < 0:
      startDOF = a_LM[Parameters.ndofSe + 1, 0]
    for dof in range(startDOF, Parameters.ndof):
      D[dof] += Parameters.Ts_0

  elif 'uf' in Parameters.Physics:
    #-------------------------------------
    # Set the initial pore fluid pressure.
    #-------------------------------------
    startDOF = a_LM[Parameters.ndofSe + Parameters.ndofFe, 0]
    if startDOF < 0:
      startDOF = a_LM[Parameters.ndofSe + Parameters.ndofFe + 1, 0]
    for dof in range(startDOF, Parameters.ndofS + Parameters.ndofF + Parameters.ndofP):
      D[dof] += Parameters.p_f0
    #------------------------------
    # Set the initial temperatures.
    #------------------------------
    if 't' in Parameters.Physics:
      for dof in range(Parameters.ndofS + Parameters.ndofF + Parameters.ndofP, Parameters.ndof - Parameters.ndofTf):
        D[dof] += Parameters.Ts_0
      for dof in range(Parameters.ndof - Parameters.ndofTf, Parameters.ndof):
        D[dof] += Parameters.Tf_0
  
  elif Parameters.Physics == 'u-pf-ts-tf':
    #-------------------------------------
    # Set the initial pore fluid pressure.
    #-------------------------------------
    startDOF = a_LM[Parameters.ndofSe, 0]
    if startDOF < 0:
      startDOF = a_LM[Parameters.ndofSe+ 1, 0]
    for dof in range(startDOF, Parameters.ndofS + Parameters.ndofP):
      D[dof] += Parameters.p_f0
    #------------------------------
    # Set the initial temperatures.
    #------------------------------
    for dof in range(Parameters.ndofS + Parameters.ndofP, Parameters.ndof - Parameters.ndofTf):
      D[dof] += Parameters.Ts_0
    for dof in range(Parameters.ndof - Parameters.ndofTf, Parameters.ndof):
      D[dof] += Parameters.Tf_0
  
  return D, V, A
#------------------------------------------------------------------------------------------------
# Master function to update any time-dependent BCs.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_GEXT:     (float, size: # DOFs [or solid DOF])        external traction vector
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# *args:      (float, size: variable)                     a list of external force vectors
#                                                         corresponding to the problem
#                                                         formulation (e.g., HEXT, JEXT, KEXT)
#------------------------------------------------------------------------------------------------
def updateBC(a_g, a_GEXT, a_T, Parameters, *args):

  if Parameters.solidDisplacementApply:
    a_g = applySolidDisplacement(a_g, a_T, Parameters)
    
  if Parameters.solidVelocityApply:
    a_g = applySolidVelocity(a_g, a_T, Parameters)

  if Parameters.tractionApply:
    a_GEXT = applyTraction(a_GEXT, a_T, Parameters)

  if Parameters.Physics == 'u':
    return a_g, a_GEXT
  
  elif Parameters.Physics == 'u-t':
    if Parameters.solidTempApply:
      a_g = applySolidTemp(a_g, a_T, Parameters)

    a_JEXT = args[0]
    if Parameters.solidHeatFluxApply:
      a_JEXT = applySolidHeatFlux(a_JEXT, a_T, Parameters)
    
    return a_g, a_GEXT, a_JEXT
  
  else:
    if Parameters.pressureApply:
      a_g = applyFluidPressure(a_g, a_T, Parameters)
    
    a_HEXT = args[0]
    if Parameters.fluxApply:
      a_HEXT = applyFlux(a_HEXT, a_T, Parameters)
    
    if 'uf' in Parameters.Physics:
      if Parameters.fluidDisplacementApply:
        a_g = applyFluidDisplacement(a_g, a_T, Parameters)

      if Parameters.fluidVelocityApply:
        a_g = applyFluidVelocity(a_g, a_T, Parameters)
    
    if 't' in Parameters.Physics:
      if Parameters.solidTempApply:
        a_g = applySolidTemp(a_g, a_T, Parameters)

      if Parameters.fluidTempApply:
        a_g = applyFluidTemp(a_g, a_T, Parameters)

      a_JEXT = args[1]
      a_KEXT = args[2]
      
      if Parameters.solidHeatFluxApply:
        a_JEXT = applySolidHeatFlux(a_JEXT, a_T, Parameters)
      
      if Parameters.fluidHeatFluxApply:
        a_KEXT = applyFluidHeatFlux(a_KEXT, a_T, Parameters)
      
      return a_g, a_GEXT, a_HEXT, a_JEXT, a_KEXT
    
    else:
      return a_g, a_GEXT, a_HEXT

#------------------------------------------------------------------------------------------------
# Function to insert Dirichlet BCs into the solution arrays.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: (2 x 3 x # element DOFs x 2))  Dirichlet BCs
# a_D:        (float, size: # unknown DOFs)                "D" solutions for unknown DOFs
# a_V:        (float, size: # unknown DOFs)                "V" solutions for unknown DOFs
# a_A:        (float, size: # unknown DOFs)                "A" solutions for unknown DOFs
# a_Dsolve:   (float, size: # DOFs)                        unpopulated "D" solutions for all DOFs
# a_Vsolve:   (float, size: # DOFs)                        unpopulated "V" solutions for all DOFs
# a_Asolve:   (float, size: # DOFs)                        unpopulated "A" solutions for all DOFs
# Parameters: (object)                                     problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_Dsolve:   (float, size: # DOFs)   populated "D" solutions for all DOFs
# a_Vsolve:   (float, size: # DOFs)   populated "V" solutions for all DOFs
# a_Asolve:   (float, size: # DOFs)   populated "A" solutions for all DOFs
#-------------------------------------------------------------------------------------------------
def insertBC(a_g, a_D, a_V, a_A, a_Dsolve, a_Vsolve, a_Asolve, Parameters):
  #-------------------------------------------
  # Insert solid displacement Dirichlet BC(s).
  #-------------------------------------------
  if Parameters.solidDisplacementApply:
    if 'Top' in Parameters.solidDisplacementLocation:
      if Parameters.Element_Type.split('-')[0] == 'Q3H':
        a_Dsolve[Parameters.nNodeS - 2] = a_g[(0,0)+Parameters.solidDisplacementDOFTop]
        a_Vsolve[Parameters.nNodeS - 2] = a_g[(0,1)+Parameters.solidDisplacementDOFTop]
        a_Asolve[Parameters.nNodeS - 2] = a_g[(0,2)+Parameters.solidDisplacementDOFTop]
      else:
        a_Dsolve[Parameters.nNodeS - 1] = a_g[(0,0)+Parameters.solidDisplacementDOFTop]
        a_Vsolve[Parameters.nNodeS - 1] = a_g[(0,1)+Parameters.solidDisplacementDOFTop]
        a_Asolve[Parameters.nNodeS - 1] = a_g[(0,2)+Parameters.solidDisplacementDOFTop]
    if 'Bottom' in Parameters.solidDisplacementLocation:
      a_Dsolve[0] = a_g[(0,0)+Parameters.solidDisplacementDOFTop]
      a_Vsolve[0] = a_g[(0,1)+Parameters.solidDisplacementDOFTop]
      a_Asolve[0] = a_g[(0,2)+Parameters.solidDisplacementDOFTop]
  #---------------------------------------
  # Insert solid velocity Dirichlet BC(s).
  #---------------------------------------
  if Parameters.solidVelocityApply:
    if 'Top' in Parameters.solidVelocityLocation:
      if Parameters.Element_Type.split('-')[0] == 'Q3H':
        a_Dsolve[Parameters.nNodeS - 2] = a_g[(0,0)+Parameters.solidVelocityDOFTop]
        a_Vsolve[Parameters.nNodeS - 2] = a_g[(0,1)+Parameters.solidVelocityDOFTop]
        a_Asolve[Parameters.nNodeS - 2] = a_g[(0,2)+Parameters.solidVelocityDOFTop]
      else:
        a_Dsolve[Parameters.nNodeS - 1] = a_g[(0,0)+Parameters.solidVelocityDOFTop]
        a_Vsolve[Parameters.nNodeS - 1] = a_g[(0,1)+Parameters.solidVelocityDOFTop]
        a_Asolve[Parameters.nNodeS - 1] = a_g[(0,2)+Parameters.solidVelocityDOFTop]
    if 'Bottom' in Parameters.solidVelocityLocation:
      a_Dsolve[0] = a_g[(0,0)+Parameters.solidVelocityDOFTop]
      a_Vsolve[0] = a_g[(0,1)+Parameters.solidVelocityDOFTop]
      a_Asolve[0] = a_g[(0,2)+Parameters.solidVelocityDOFTop]
  #------------------------------------------
  # Insert solid temperature Dirichlet BC(s).
  #------------------------------------------
  if Parameters.Physics == 'u-t':
    if Parameters.solidTempApply:
      if 'Top' in Parameters.solidTempLocation:
        a_Dsolve[Parameters.nNode - 1] = a_g[(0,0)+Parameters.solidTempDOFTop]
        a_Vsolve[Parameters.nNode - 1] = a_g[(0,1)+Parameters.solidTempDOFTop]
        a_Asolve[Parameters.nNode - 1] = a_g[(0,2)+Parameters.solidTempDOFTop]
      if 'Bottom' in Parameters.solidTempLocation:
        a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.solidTempDOFBot]
        a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.solidTempDOFBot]
        a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.solidTempDOFBot]
  #--------------------------------------------
  # Insert pore fluid pressure Dirichlet BC(s).
  #--------------------------------------------
  elif Parameters.Physics == 'u-pf':
    if Parameters.pressureApply:
      if 'Top' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNode - 1] = a_g[(0,0)+Parameters.pressureDOFTop]
        a_Vsolve[Parameters.nNode - 1] = a_g[(0,1)+Parameters.pressureDOFTop]
        a_Asolve[Parameters.nNode - 1] = a_g[(0,2)+Parameters.pressureDOFTop]
      if 'Bottom' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.pressureDOFBot]
        a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.pressureDOFBot]
        a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.pressureDOFBot]

  elif Parameters.Physics == 'u-pf-ts-tf':
    #--------------------------------------------
    # Insert pore fluid pressure Dirichlet BC(s).
    #--------------------------------------------
    if Parameters.pressureApply:
      if 'Top' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeP - 1] = a_g[(0,0)+Parameters.pressureDOFTop]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeP - 1] = a_g[(0,1)+Parameters.pressureDOFTop]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeP - 1] = a_g[(0,2)+Parameters.pressureDOFTop]
      if 'Bottom' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.pressureDOFBot]
        a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.pressureDOFBot]
        a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.pressureDOFBot]
    #------------------------------------------
    # Insert solid temperature Dirichlet BC(s).
    #------------------------------------------
    if Parameters.solidTempApply:
      if 'Top' in Parameters.solidTempLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,0)+Parameters.solidTempDOFTop]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,1)+Parameters.solidTempDOFTop]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,2)+Parameters.solidTempDOFTop]
      if 'Bottom' in Parameters.solidTempLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeP] = a_g[(0,0)+Parameters.solidTempDOFBot]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeP] = a_g[(0,1)+Parameters.solidTempDOFBot]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeP] = a_g[(0,2)+Parameters.solidTempDOFBot]
    #-----------------------------------------------
    # Insert pore fluid temperature Dirichlet BC(s).
    #-----------------------------------------------
    if Parameters.fluidTempApply:
      if 'Top' in Parameters.fluidTempLocation:
        a_Dsolve[Parameters.nNode - 1] = a_g[(0,0)+Parameters.fluidTempDOFTop]
        a_Vsolve[Parameters.nNode - 1] = a_g[(0,1)+Parameters.fluidTempDOFTop]
        a_Asolve[Parameters.nNode - 1] = a_g[(0,2)+Parameters.fluidTempDOFTop]
      if 'Bottom' in Parameters.fluidTempLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,0)+Parameters.fluidTempDOFBot]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,1)+Parameters.fluidTempDOFBot]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,2)+Parameters.fluidTempDOFBot]

  elif 'uf' in Parameters.Physics:
    #------------------------------------------------
    # Insert pore fluid displacement Dirichlet BC(s).
    #------------------------------------------------
    if Parameters.fluidDisplacementApply:
      if 'Top' in Parameters.fluidDisplacementLocation:
        if Parameters.Element_Type.split('-')[1] == 'Q3H':
          a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,0)+Parameters.fluidDisplacementDOFTop]
          a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,1)+Parameters.fluidDisplacementDOFTop]
          a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,2)+Parameters.fluidDisplacementDOFTop]
        else:
          a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,0)+Parameters.fluidDisplacementDOFTop]
          a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,1)+Parameters.fluidDisplacementDOFTop]
          a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,2)+Parameters.fluidDisplacementDOFTop]
      if 'Bottom' in Parameters.fluidDisplacementLocation:
        a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.fluidDisplacementDOFBot]
        a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.fluidDisplacementDOFBot]
        a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.fluidDisplacementDOFBot]
    #--------------------------------------------
    # Insert pore fluid velocity Dirichlet BC(s).
    #--------------------------------------------
    if Parameters.fluidVelocityApply and 'Lagrange' not in Parameters.fluidVelocityApplication:
      if 'Top' in Parameters.fluidVelocityLocation:
        if Parameters.fluidVelocityApplication.split(',')[0] != 'No-Slip':
          if Parameters.Element_Type.split('-')[1] == 'Q3H':
            a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,0)+Parameters.fluidVelocityDOFTop]
            a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,1)+Parameters.fluidVelocityDOFTop]
            a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_g[(0,2)+Parameters.fluidVelocityDOFTop]
          else:
            a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,0)+Parameters.fluidVelocityDOFTop]
            a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,1)+Parameters.fluidVelocityDOFTop]
            a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_g[(0,2)+Parameters.fluidVelocityDOFTop]
        else:
          if Parameters.Element_Type.split('-')[1] == 'Q3H':
            a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_D[Parameters.noSlipDOFTop]
            a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_V[Parameters.noSlipDOFTop]
            a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 2] = a_A[Parameters.noSlipDOFTop]
          else:
            a_Dsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_D[Parameters.noSlipDOFTop]
            a_Vsolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_V[Parameters.noSlipDOFTop]
            a_Asolve[Parameters.nNodeS + Parameters.nNodeF - 1] = a_A[Parameters.noSlipDOFTop]
      if 'Bottom' in Parameters.fluidVelocityLocation:
        try:
          if Parameters.fluidVelocityApplication.split(',')[1] != 'No-Slip':
            a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.fluidVelocityDOFBot]
            a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.fluidVelocityDOFBot]
            a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.fluidVelocityDOFBot]
          else:
            a_Dsolve[Parameters.nNodeS] = a_D[Parameters.noSlipDOFBot]
            a_Vsolve[Parameters.nNodeS] = a_V[Parameters.noSlipDOFBot]
            a_Asolve[Parameters.nNodeS] = a_A[Parameters.noSlipDOFBot]
        except IndexError:
          if Parameters.fluidVelocityApplication != 'No-Slip':
            a_Dsolve[Parameters.nNodeS] = a_g[(0,0)+Parameters.fluidVelocityDOFBot]
            a_Vsolve[Parameters.nNodeS] = a_g[(0,1)+Parameters.fluidVelocityDOFBot]
            a_Asolve[Parameters.nNodeS] = a_g[(0,2)+Parameters.fluidVelocityDOFBot]
          else:
            a_Dsolve[Parameters.nNodeS] = a_D[Parameters.noSlipDOFBot]
            a_Vsolve[Parameters.nNodeS] = a_V[Parameters.noSlipDOFBot]
            a_Asolve[Parameters.nNodeS] = a_A[Parameters.noSlipDOFBot]
    #--------------------------------------------
    # Insert pore fluid pressure Dirichlet BC(s).
    #--------------------------------------------
    if Parameters.pressureApply:
      if 'Top' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP - 1] = a_g[(0,0)+Parameters.pressureDOFTop]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP - 1] = a_g[(0,1)+Parameters.pressureDOFTop]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP - 1] = a_g[(0,2)+Parameters.pressureDOFTop]
      if 'Bottom' in Parameters.pressureLocation:
        a_Dsolve[Parameters.nNodeS + Parameters.nNodeF] = a_g[(0,0)+Parameters.pressureDOFBot]
        a_Vsolve[Parameters.nNodeS + Parameters.nNodeF] = a_g[(0,1)+Parameters.pressureDOFBot]
        a_Asolve[Parameters.nNodeS + Parameters.nNodeF] = a_g[(0,2)+Parameters.pressureDOFBot]

    if Parameters.Physics == 'u-uf-pf-ts-tf':
      #------------------------------------------
      # Insert solid temperature Dirichlet BC(s).
      #------------------------------------------
      if Parameters.solidTempApply:
        if 'Top' in Parameters.solidTempLocation:
          a_Dsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,0)+Parameters.solidTempDOFTop]
          a_Vsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,1)+Parameters.solidTempDOFTop]
          a_Asolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs - 1] = a_g[(0,2)+Parameters.solidTempDOFTop]
        if 'Bottom' in Parameters.solidTempLocation:
          a_Dsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP] = a_g[(0,0)+Parameters.solidTempDOFBot]
          a_Vsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP] = a_g[(0,1)+Parameters.solidTempDOFBot]
          a_Asolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP] = a_g[(0,2)+Parameters.solidTempDOFBot]
      #-----------------------------------------------
      # Insert pore fluid temperature Dirichlet BC(s).
      #-----------------------------------------------
      if Parameters.fluidTempApply:
        if 'Top' in Parameters.fluidTempLocation:
          a_Dsolve[Parameters.nNode - 1] = a_g[(0,0)+Parameters.fluidTempDOFTop]
          a_Vsolve[Parameters.nNode - 1] = a_g[(0,1)+Parameters.fluidTempDOFTop]
          a_Asolve[Parameters.nNode - 1] = a_g[(0,2)+Parameters.fluidTempDOFTop]
        if 'Bottom' in Parameters.fluidTempLocation:
          a_Dsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,0)+Parameters.fluidTempDOFBot]
          a_Vsolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,1)+Parameters.fluidTempDOFBot]
          a_Asolve[Parameters.nNodeS + Parameters.nNodeF + Parameters.nNodeP + Parameters.nNodeTs] = a_g[(0,2)+Parameters.fluidTempDOFBot]

  a_Dsolve[np.where(np.isnan(a_Dsolve))] = a_D[:]
  a_Vsolve[np.where(np.isnan(a_Vsolve))] = a_V[:]
  a_Asolve[np.where(np.isnan(a_Asolve))] = a_A[:]

  return a_Dsolve, a_Vsolve, a_Asolve

#-----------------------
# End top-level methods.
#-----------------------
#-------------------------
# Begin mid-level methods.
#-------------------------
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for solid displacement Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initSolidDisplacementBC(a_LM, Parameters):

  if Parameters.ne == 1 and Parameters.ndofSe <= 2:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nNot enough degrees of freedom to solve problem. Use Q2/Q3/Q3H elements or more elements.")
  
  Parameters.ndof  -= 1
  Parameters.ndofS -= 1

  Parameters.solidDisplacementDOFBot = (0,0)
  if 'Q3H' in Parameters.Element_Type.split('-')[0]:
    Parameters.solidDisplacementDOFTop = (1, -1)
  else:
    Parameters.solidDisplacementDOFTop = (Parameters.ndofSe - 1, -1)

  if Parameters.solidDisplacementLocation == 'Top':
    a_LM[Parameters.ndofSe:, :]             -= 1
    a_LM[Parameters.solidDisplacementDOFTop] = -1
    if 'Q3H' in Parameters.Element_Type.split('-')[0]:
      a_LM[Parameters.ndofSe - 1, -1] -= 1

  elif Parameters.solidDisplacementLocation == 'Bottom':
    a_LM -= 1

  elif Parameters.solidDisplacementLocation == 'Top,Bottom':      
    a_LM             -= 1 
    Parameters.ndof  -= 1
    Parameters.ndofS -= 1
    a_LM[Parameters.ndofSe:, :]             -= 1
    a_LM[Parameters.solidDisplacementDOFTop] = -1
    if 'Q3H' in Parameters.Element_Type.split('-')[0]:
      a_LM[Parameters.ndofSe - 1, -1] -= 1
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid Dirichlet BC location must be configured for Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for solid velocity Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initSolidVelocityBC(a_LM, Parameters):
  if Parameters.ne == 1 and Parameters.ndofSe == 2:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nNot enough degrees of freedom to solve problem. Use Q2/Q3/Q3H elements or more elements.")
  
  if Parameters.solidDisplacementLocation in Parameters.solidVelocityLocation or Parameters.solidVelocityLocation in Parameters.solidDisplacementLocation:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nCannot specify solid displacement and velocity Dirichlet BC on same boundary.")
  
  Parameters.ndof  -= 1
  Parameters.ndofS -= 1

  Parameters.solidVelocityDOFBot = (0,0)
  if 'Q3H' in Parameters.Element_Type.split('-')[0]:
    Parameters.solidVelocityDOFTop = (1, -1)
  else:
    Parameters.solidVelocityDOFTop = (Parameters.ndofSe - 1, -1)

  if Parameters.solidVelocityLocation == 'Top':
    a_LM[Parameters.ndofSe:, :]         -= 1
    a_LM[Parameters.solidVelocityDOFTop] = -1
    if 'Q3H' in Parameters.Element_Type.split('-')[0]:
      a_LM[Parameters.ndofSe - 1, -1] -= 1

  elif Parameters.solidVelocityLocation == 'Bottom':
    a_LM -= 1

  elif Parameters.solidVelocityLocation == 'Top,Bottom':      
    a_LM             -= 1 
    Parameters.ndof  -= 1
    Parameters.ndofS -= 1
    a_LM[Parameters.ndofSe:, :]         -= 1
    a_LM[Parameters.solidVelocityDOFTop] = -1
    if 'Q3H' in Parameters.Element_Type.split('-')[0]:
      a_LM[Parameters.ndofSe - 1, -1] -= 1
  
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid velocity Dirichlet BC location must be configured for Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for pore fluid displacement Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initFluidDisplacementBC(a_LM, Parameters):

  if Parameters.fluxApply:
    if Parameters.fluidDisplacementLocation in Parameters.fluxLocation or Parameters.fluxLocation in Parameters.fluidDisplacementLocation:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid Dirichlet and pore fluid pressure Neuman boundary conditions location conflict.")
  
  Parameters.ndof  -= 1
  Parameters.ndofF -= 1

  Parameters.fluidDisplacementDOFBot = (Parameters.ndofSe, 0)
  if 'Q3H' in Parameters.Element_Type.split('-')[1]:
    Parameters.fluidDisplacementDOFTop = (Parameters.ndofSe + Parameters.ndofFe - 3, -1)
  else:
    Parameters.fluidDisplacementDOFTop = (Parameters.ndofSe + Parameters.ndofFe - 1, -1)

  if Parameters.fluidDisplacementLocation == 'Top':
    a_LM[Parameters.ndofSe + Parameters.ndofFe:,:] -= 1
    if 'Q3H' in Parameters.Element_Type.split('-')[1]:
      a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] -= 1 
    a_LM[Parameters.fluidDisplacementDOFTop] = -1  

  elif Parameters.fluidDisplacementLocation == 'Bottom':
    a_LM[Parameters.ndofSe:,:]              -= 1
    a_LM[Parameters.fluidDisplacementDOFBot] = -1

  elif Parameters.fluidDisplacementLocation == 'Top,Bottom':        
    a_LM[Parameters.ndofSe:,:]                         -= 1
    a_LM[Parameters.ndofSe + Parameters.ndofFe:,:]     -= 1
    if 'Q3H' in Parameters.Element_Type.split('-')[1]:
      a_LM[Parameters.ndofSe + Parameters.ndofFe - 1,-1] -= 1
    a_LM[Parameters.fluidDisplacementDOFBot] = -1
    a_LM[Parameters.fluidDisplacementDOFTop] = -1
    
    Parameters.ndof  -= 1
    Parameters.ndofF -= 1
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid displacement Dirichlet BC location must be configured for Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for pore fluid velocity Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initFluidVelocityBC(a_LM, Parameters):

  if Parameters.fluxApply:
    if Parameters.fluidVelocityLocation in Parameters.fluxLocation or Parameters.fluxLocation in Parameters.fluidVelocityLocation:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet and pore fluid pressure Neuman boundary conditions location conflict.")
  if Parameters.fluidDisplacementApply:
    if Parameters.fluidDisplacementLocation in Parameters.fluidVelocityLocation or Parameters.fluidVelocityLocation in Parameters.fluidDisplacementLocation:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nOverlapping pore fluid Dirichlet boundary conditions.")
  
  if 'Lagrange' in Parameters.fluidVelocityApplication:
    Parameters.LagrangeApply = True
    a_LM                     = initLagrange(a_LM, Parameters)
    #-------------------------------
    # Add the extra DOF for \lambda.
    #-------------------------------
    Parameters.nNode        += 1
    Parameters.ndof         += 1
 
  else:
    Parameters.LagrangeApply = False
    Parameters.ndof  -= 1
    Parameters.ndofF -= 1

    Parameters.fluidVelocityDOFBot = (Parameters.ndofSe, 0)
    if 'Q3H' in Parameters.Element_Type.split('-')[1]:
      Parameters.fluidVelocityDOFTop = (Parameters.ndofSe + Parameters.ndofFe - 3, -1)
    else:
      Parameters.fluidVelocityDOFTop = (Parameters.ndofSe + Parameters.ndofFe - 1, -1)
    #-------------------------------------------------------------------
    # Modify DOF for the no-slip, no-permeation BC (strong enforcement).
    #-------------------------------------------------------------------
    if 'No-Slip' in Parameters.fluidVelocityApplication:
      a_LM = initNoSlipCondition(a_LM, Parameters)
    #--------------------------------------------------------------------------------
    # Extra conditional given so that the no-slip BC is not accidentally overwritten.
    #--------------------------------------------------------------------------------
    if Parameters.fluidVelocityLocation == 'Top' and Parameters.fluidVelocityApplication != 'No-Slip':
      a_LM[Parameters.ndofSe + Parameters.ndofFe:,:] -= 1
      a_LM[Parameters.fluidVelocityDOFTop]            = -1
      if 'Q3H' in Parameters.Element_Type.split('-')[1]:
        a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] -= 1
           
    elif Parameters.fluidVelocityLocation == 'Bottom' and Parameters.fluidVelocityApplication != 'No-Slip':
      a_LM[Parameters.ndofSe:,:]          -= 1
      a_LM[Parameters.fluidVelocityDOFBot] = -1

    elif Parameters.fluidVelocityLocation == 'Top,Bottom':
      #-------------------------------------------
      # Check if the no-slip condition is applied.
      # If it is, do not modify those DOFs.
      #-------------------------------------------
      idxs = [idxs for idxs, app in enumerate(Parameters.fluidVelocityApplication.split(',')) if app == 'No-Slip']
      if len(idxs) == 1:
        #------------------------------------------------------------------
        # No-Slip applied at top, other pore fluid velocity Dirichlet BC
        # applied at the bottom.
        #------------------------------------------------------------------
        if idxs[0] == 0:
          a_LM[Parameters.ndofSe:,:]           -= 1
          a_LM[Parameters.fluidVelocityDOFBot]  = -1
        #------------------------------------------------------------------
        # No-Slip applied at bottom, other pore fluid velocity Dirichlet BC
        # applied at the top.
        #------------------------------------------------------------------
        elif idxs[0] == 1:
          a_LM[Parameters.ndofSe + Parameters.ndofFe:,:] -= 1
          a_LM[Parameters.fluidVelocityDOFTop]            = -1
          if 'Q3H' in Parameters.Element_Type.split('-')[1]:
            a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] -= 1
      #-------------------------------------------------------
      # No-Slip not applied, modify both boundaries in the LM.
      #-------------------------------------------------------
      elif len(idxs) == 0:
        a_LM[Parameters.ndofSe:,:]                     -= 1
        a_LM[Parameters.ndofSe + Parameters.ndofFe:,:] -= 1
        if 'Q3H' in Parameters.Element_Type.split('-')[1]:
          a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1] -= 1
        a_LM[Parameters.fluidVelocityDOFBot] = -1
        a_LM[Parameters.fluidVelocityDOFTop] = -1
                  
        Parameters.ndof  -= 1
        Parameters.ndofF -= 1
    else:
      if 'No-Slip' not in Parameters.fluidVelocityApplication:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet BC location must be configured for Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for the no-flux condition imposed on the pore fluid 
# via Lagrange multiplier weak enforcement.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs + 1 x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initLagrange(a_LM, Parameters):
  #-----------------------------------------------------------
  # Determine where user wants to apply the no-slip condition.
  #-----------------------------------------------------------
  idxs = [idxs for idxs, app in enumerate(Parameters.fluidVelocityApplication.split(',')) if app == 'Lagrange']
  if len(idxs) == 1:
    Parameters.LagrangeLocation = Parameters.fluidVelocityLocation.split(',')[idxs[0]]
  elif len(idxs) == 2:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nWeak enforcement of the no-flux condition can only be applied at one boundary.")
  #------------------------------------------------------------------
  # It is assumed that the Lagrange multiplier is identified as the
  # pseudo-traction acting on the pore fluid. Therefore, the only
  # allowable location must match where the traction load is applied.
  #------------------------------------------------------------------
  if Parameters.LagrangeLocation != Parameters.tractionLocation:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nWeak enforcement of the no-flux condition must be applied at the same location as the external load.")
  #---------------------------------------
  # Add Lagrange mutliplier DOF to the LM.
  #---------------------------------------
  a_LM = np.vstack((a_LM,-1*np.ones((1,a_LM.shape[1]), dtype=np.int32)))
  if Parameters.LagrangeLocation == 'Top':
    a_LM[-1, -1]          = Parameters.ndof
    Parameters.LagrangeID = Parameters.ne
  elif Parameters.LagrangeLocation == 'Bottom':
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nWeak enforcement of the no-flux condition can only be applied at the top boundary (for now).")
    a_LM[Parameters.ndofe, 0] = a_LM[Parameters.ndofe - 1, -1] + 1
    Parameters.LagrangeID     = 0

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for the no-slip condition imposed on the pore fluid.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initNoSlipCondition(a_LM, Parameters):
  #-----------------------------------------------------------
  # Determine where user wants to apply the no-slip condition.
  #
  # E.g., user could supply 'No-Slip,Constant' as two pore
  # fluid velocity boundary conditions. The enumeration
  # determines where the application is specified (assuming 
  # user follows the 'Top,Bottom' ordering convention).
  #-----------------------------------------------------------
  idxs = [idxs for idxs, app in enumerate(Parameters.fluidVelocityApplication.split(',')) if app == 'No-Slip']
  if len(idxs) == 1:
    Parameters.noSlipLocation = Parameters.fluidVelocityLocation.split(',')[idxs[0]]
  elif len(idxs) == 2:
    Parameters.noSlipLocation = Parameters.fluidVelocityLocation
  #---------------------------------------------------------------------
  # Exit if the user has overlapping Dirichlet BC for solid and no-slip.
  #---------------------------------------------------------------------
  if Parameters.solidDisplacementApply and (Parameters.solidDisplacementLocation in Parameters.noSlipLocation or Parameters.noSlipLocation in Parameters.solidDisplacementLocation):
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid displacement Dirichlet BC overlaps with pore fluid velocity no-slip condition.\nImpose the same displacement Dirichlet BC on the pore fluid.")
  if Parameters.solidVelocityApply and (Parameters.solidVelocityLocation in Parameters.noSlipLocation  or Parameters.noSlipLocation in Parameters.solidVelocityLocation):
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid velocity Dirichlet BC overlaps with pore fluid velocity no-slip condition.\nImpose the same velocity Dirichlet BC on the pore fluid.")
  #-----------------------------------------
  # No-slip is applied at one location only.
  #-----------------------------------------
  if len(idxs) == 1:
    if Parameters.noSlipLocation == 'Top':
      if Parameters.pressureApply and 'Top' in Parameters.pressureLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet and pore fluid pressure Dirichlet boundary conditions location conflict.")
      if Parameters.Element_Type.split('-')[1] == 'Q3H':
        a_LM[Parameters.ndofSe + 1,                  -1]  = a_LM[1, -1]
        a_LM[Parameters.ndofSe + 3,                  -1] -= 1
        a_LM[Parameters.ndofSe + Parameters.ndofFe:, : ] -= 1
        Parameters.noSlipDOFTop                           = a_LM[1, -1]
      else:
        if Parameters.Element_Type.split('-')[0] == 'Q3H':
          a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1]  = a_LM[1, -1]
          Parameters.noSlipDOFTop                              = a_LM[1, -1]
        else:
          a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1]  = a_LM[Parameters.ndofSe - 1, -1]
          Parameters.noSlipDOFTop                              = a_LM[Parameters.ndofSe - 1, -1]

        a_LM[Parameters.ndofSe + Parameters.ndofFe:Parameters.ndofe,:] -= 1
    elif Parameters.noSlipLocation == 'Bottom':
      if Parameters.pressureApply and 'Bottom' in Parameters.pressureLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet and pore fluid pressure Dirichlet boundary conditions location conflict.")
      a_LM[Parameters.ndofSe:,:]          -= 1
      a_LM[Parameters.fluidVelocityDOFBot] = a_LM[0, 0]
      Parameters.noSlipDOFBot              = a_LM[0, 0]
  #-------------------------------------
  # No-slip is applied at two locations.
  #-------------------------------------
  elif len(idxs) == 2:
    if Parameters.noSlipLocation == 'Top,Bottom':
      if Parameters.pressureApply and 'Top' in Parameters.pressureLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet and pore fluid pressure Dirichlet boundary conditions location conflict.")
      if Parameters.pressureApply and 'Bottom' in Parameters.pressureLocation:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid velocity Dirichlet and pore fluid pressure Dirichlet boundary conditions location conflict.")
      a_LM[Parameters.ndofSe:,:]          -= 1
      a_LM[Parameters.fluidVelocityDOFBot] = a_LM[0, 0]
      Parameters.noSlipDOFBot              = a_LM[0, 0]
      Parameters.noSlipDOFTop              = a_LM[1, -1]
      if Parameters.Element_Type.split('-')[1] == 'Q3H':
        a_LM[Parameters.ndofSe + 1, -1]                   = a_LM[1, -1]
        a_LM[Parameters.ndofSe + 3, -1]                  -= 1
        a_LM[Parameters.ndofSe + Parameters.ndofFe:, : ] -= 1
      else:
        if Parameters.Element_Type.split('-')[0] == 'Q3H':
          a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1]  = a_LM[1, -1]
          a_LM[Parameters.ndofSe + Parameters.ndofFe:,:]      -= 1
        else:
          a_LM[Parameters.ndofSe + Parameters.ndofFe - 1, -1]  = a_LM[Parameters.ndofSe - 1, -1]
          a_LM[Parameters.ndofSe + Parameters.ndofFe:,:]      -= 1
          Parameters.noSlipDOFTop                              = a_LM[Parameters.ndofSe - 1, -1]
      
      Parameters.ndof  -= 1
      Parameters.ndofF -=1
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nNo-slip BC location must be configured for Top, Bottom, or Top,Bottom.")
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nNo-slip BC location must be configured for Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for pore fluid pressure Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initFluidPressureBC(a_LM, Parameters):
  Parameters.pressureDOFTop = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 1, -1)
  Parameters.pressureDOFBot = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 2,  0)
  if Parameters.pressureLocation == 'Top,Bottom':
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 2:Parameters.ndofe, : ]  -= 1
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 1                  , -1]  = -1
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 2,                    0]  = -1
    Parameters.ndof   -= 2
    Parameters.ndofP  -= 2
    if 'tf' in Parameters.Physics:
      a_LM[Parameters.ndofe - 4:Parameters.ndofe,:] -= 1

  elif Parameters.pressureLocation == 'Top':
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe - 1, -1] = -1
    if 'tf' in Parameters.Physics:
      a_LM[Parameters.ndofe - 4:Parameters.ndofe,:] -= 1
    Parameters.ndof  -= 1
    Parameters.ndofP -= 1

  elif Parameters.pressureLocation == 'Bottom':
    a_LM[Parameters.ndofSe + Parameters.ndofFe:Parameters.ndofe,:] -= 1
    a_LM[Parameters.ndofSe + Parameters.ndofFe, 0]                  = -1
    Parameters.ndof  -= 1
    Parameters.ndofP -= 1

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for solid temperature Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initSolidTempBC(a_LM, Parameters):

  if Parameters.Physics == 'u-t':
    Parameters.solidTempDOFTop = (Parameters.ndofSe + Parameters.ndofTse - 1, -1)
    Parameters.solidTempDOFBot = (Parameters.ndofSe,  0)

    Parameters.ndofTs -= 1
    Parameters.ndof   -= 1
    
    if Parameters.solidTempLocation == 'Top':
      a_LM[Parameters.solidTempDOFTop]                = -1
      a_LM[Parameters.ndofSe + Parameters.ndofTse:,] -= 1
    elif Parameters.solidTempLocation == 'Bottom':
      a_LM[Parameters.ndofSe:,]       -= 1
      a_LM[Parameters.solidTempDOFBot] = -1
    elif Parameters.solidTempLocation == 'Top,Bottom':
      a_LM[Parameters.ndofSe:,]                      -= 1
      a_LM[Parameters.ndofSe + Parameters.ndofTse:,] -= 1
      a_LM[Parameters.solidTempDOFBot]                = -1
      a_LM[Parameters.solidTempDOFTop]                = -1
      Parameters.ndofTs                              -= 1
      Parameters.ndof                                -= 1
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature Dirichlet BC location must be configured at Top, Bottom, or Top,Bottom.")
  
  elif 'tf' in Parameters.Physics:
    # Note that 'ndofFe' will be read as 0 if the element type is for (u-pf-ts-tf)
    Parameters.solidTempDOFTop = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse - 1, -1)
    Parameters.solidTempDOFBot = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse - 2,  0)

    Parameters.ndofTs -= 1
    Parameters.ndof   -= 1

    if Parameters.solidTempLocation == 'Top':
      a_LM[Parameters.solidTempDOFTop]                                                        = -1
      a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse:,] -= 1
    elif Parameters.solidTempLocation == 'Bottom':
      a_LM[Parameters.solidTempDOFBot]                                   = -1
      a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe:,] -= 1
    elif Parameters.solidTempLocation == 'Top,Bottom':
      Parameters.ndofTs -= 1
      Parameters.ndof   -= 1
      a_LM[Parameters.solidTempDOFBot]                                                        = -1
      a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe:,]                      -= 1
      a_LM[Parameters.solidTempDOFTop]                                                        = -1
      a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse:,] -= 1
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature Dirichlet BC location must be configured at Top, Bottom, or Top,Bottom.")
  
  return a_LM
#------------------------------------------------------------------------------------------------
# Function to alter the location matrix for pore fluid temperature Dirichlet BCs.
# ----------
# Arguments:
# ----------
# a_LM:       (int, size: # element DOFs x # elements)   location matrix
# Parameters: (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_LM:       (int, size: # element DOFs x # elements)   modified location matrix
#-------------------------------------------------------------------------------------------------
def initFluidTempBC(a_LM, Parameters):

  # Note that 'ndofFe' will be read as 0 if the element type is for (u-pf-ts-tf)
  Parameters.fluidTempDOFTop = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe - 1, -1)
  Parameters.fluidTempDOFBot = (Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse + Parameters.ndofTfe - 2,  0)
  
  Parameters.ndofTf -= 1
  Parameters.ndof   -= 1

  if Parameters.fluidTempLocation == 'Top':
    a_LM[Parameters.fluidTempDOFTop] = -1
  elif Parameters.fluidTempLocation == 'Bottom':
    a_LM[Parameters.fluidTempDOFBot]                                                        = -1
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse:,] -= 1
  elif Parameters.fluidTempLocation == 'Top,Bottom':
    Parameters.ndofTf -= 1
    Parameters.ndof   -= 1
    a_LM[Parameters.fluidTempDOFBot]                                                        = -1
    a_LM[Parameters.ndofSe + Parameters.ndofFe + Parameters.ndofPe + Parameters.ndofTse:,] -= 1
    a_LM[Parameters.fluidTempDOFTop]                                                        = -1
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid temperature Dirichlet BC location must be configured at Top, Bottom, or Top,Bottom.")

  return a_LM
#------------------------------------------------------------------------------------------------
# Function to set the applied solid displacement.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applySolidDisplacement(a_g, a_T, Parameters):

  if Parameters.solidDisplacementLocation == 'Top':
    a_g[(0,0)+Parameters.solidDisplacementDOFTop] = computeSolidDisplacement(a_T, Parameters.solidDisplacementT0, Parameters.solidDisplacementT1, Parameters.solidDisplacementValue, Parameters.solidDisplacementApplication, Parameters)
    a_g[(0,1)+Parameters.solidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFTop)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.solidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFTop)], Parameters, Level=(0,2))

  elif Parameters.solidDisplacementLocation == 'Bottom':
    a_g[(0,0)+Parameters.solidDisplacementDOFBot] = computeSolidDisplacement(a_T, Parameters.solidDisplacementT0, Parameters.solidDisplacementT1, Parameters.solidDisplacementValue, Parameters.solidDisplacementApplication, Parameters)
    a_g[(0,1)+Parameters.solidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFBot)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.solidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFBot)], Parameters, Level=(0,2))
  
  elif Parameters.solidDisplacementLocation == 'Top,Bottom':
    try:
      a_g[(0,0)+Parameters.solidDisplacementDOFTop] = computeSolidDisplacement(a_T, Parameters.solidDisplacementT0Top, Parameters.solidDisplacementT1Top, Parameters.solidDisplacementValueTop, Parameters.solidDisplacementApplication.split(',')[0], Parameters)
      a_g[(0,0)+Parameters.solidDisplacementDOFBot] = computeSolidDisplacement(a_T, Parameters.solidDisplacementT0Bot, Parameters.solidDisplacementT1Bot, Parameters.solidDisplacementValueBot, Parameters.solidDisplacementApplication.split(',')[1], Parameters)
      a_g[(0,1)+Parameters.solidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFTop)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.solidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFTop)], Parameters, Level=(0,2))
      a_g[(0,1)+Parameters.solidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFBot)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.solidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidDisplacementDOFBot)], Parameters, Level=(0,2))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid displacement Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied solid velocity.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applySolidVelocity(a_g, a_T, Parameters):

  if Parameters.solidVelocityLocation == 'Top':
    a_g[(0,1)+Parameters.solidVelocityDOFTop] = computeSolidVelocity(a_T, Parameters.solidVelocityT0, Parameters.solidVelocityT1, Parameters.solidVelocityValue, Parameters.solidVelocityApplication, Parameters)
    a_g[(0,2)+Parameters.solidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFTop)], Parameters, Level=(1,2))
    a_g[(0,0)+Parameters.solidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFTop)], Parameters, Level=(1,0))

  elif Parameters.solidVelocityLocation == 'Bottom':
    a_g[(0,1)+Parameters.solidVelocityDOFBot] = computeSolidVelocity(a_T, Parameters.solidVelocityT0, Parameters.solidVelocityT1, Parameters.solidVelocityValue, Parameters.solidVelocityApplication, Parameters)
    a_g[(0,2)+Parameters.solidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFBot)], Parameters, Level=(1,2))
    a_g[(0,0)+Parameters.solidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFBot)], Parameters, Level=(1,0))
  
  elif Parameters.solidVelocityLocation == 'Top,Bottom':
    try:
      a_g[(0,1)+Parameters.solidVelocityDOFTop] = computeSolidVelocity(a_T, Parameters.solidVelocityT0Top, Parameters.solidVelocityT1Top, Parameters.solidVelocityValueTop, Parameters.solidVelocityApplication.split(',')[0], Parameters)
      a_g[(0,1)+Parameters.solidVelocityDOFBot] = computeSolidVelocity(a_T, Parameters.solidVelocityT0Bot, Parameters.solidVelocityT1Bot, Parameters.solidVelocityValueBot, Parameters.solidVelocityApplication.split(',')[1], Parameters)
      a_g[(0,2)+Parameters.solidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFTop)], Parameters, Level=(1,2))
      a_g[(0,0)+Parameters.solidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFTop)], Parameters, Level=(1,0))
      a_g[(0,2)+Parameters.solidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFBot)], Parameters, Level=(1,2))
      a_g[(0,0)+Parameters.solidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidVelocityDOFBot)], Parameters, Level=(1,0))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid velocity Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied traction(s).
# ----------
# Arguments:
# ----------
# a_GEXT:     (float, size: # ndof (implicit) or # ndofS (explicit))  
#                       external force vector for "G" (balance of momentum variational equation)
# a_T:        (float)   current simulation time 't'
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_GEXT:     (float, size: # ndof (implicit) or # ndofS (explicit))  
#                       modified external force vector for "G"
#-------------------------------------------------------------------------------------------------
def applyTraction(a_GEXT, a_T, Parameters):

  if Parameters.tractionLocation == 'Top':
    a_GEXT[Parameters.tractionDOFTop] = computeTraction(a_T, Parameters.tractionT0, Parameters.tractionT1, Parameters.tractionValue, Parameters.tractionOmega, Parameters.tractionApplication, Parameters)

  elif Parameters.tractionLocation == 'Bottom':
    a_GEXT[Parameters.tractionDOFBot] = computeTraction(a_T, Parameters.tractionT0, Parameters.tractionT1, Parameters.tractionValue, Parameters.tractionOmega, Parameters.tractionApplication, Parameters)

  elif Parameters.tractionLocation == 'Top,Bottom':
    try:
      a_GEXT[Parameters.tractionDOFTop] = computeTraction(a_T, Parameters.tractionT0Top, Parameters.tractionT1Top, Parameters.tractionValueTop, Parameters.tractionOmegaTop, Parameters.tractionApplication.split(',')[0], Parameters)
      a_GEXT[Parameters.tractionDOFBot] = computeTraction(a_T, Parameters.tractionT0Bot, Parameters.tractionT1Bot, Parameters.tractionValueBot, Parameters.tractionOmegaBot, Parameters.tractionApplication.split(',')[1], Parameters)
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid Neumann BC not set appropriately.")
      raise RuntimeError

  elif Parameters.tractionLocation == 'All':
     computeTraction(a_T, Parameters.tractionT0, Parameters.tractionT1,\
                     Parameters.tractionValue, Parameters.tractionOmega,\
                     Parameters.tractionApplication, Parameters) 

  return a_GEXT
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid pressure.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applyFluidPressure(a_g, a_T, Parameters):

  if Parameters.pressureLocation == 'Top':
    a_g[(0,0)+Parameters.pressureDOFTop] = computeFluidPressure(a_T, Parameters.pressureT0, Parameters.pressureT1, Parameters.pressureValue, Parameters.pressureApplication, Parameters)
    a_g[(0,1)+Parameters.pressureDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.pressureDOFTop)], Parameters, Level=(0,1))

  elif Parameters.pressureLocation == 'Bottom':
    a_g[(0,0)+Parameters.pressureDOFBot] = computeFluidPressure(a_T, Parameters.pressureT0, Parameters.pressureT1, Parameters.pressureValue, Parameters.pressureApplication, Parameters)
    a_g[(0,1)+Parameters.pressureDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.pressureDOFBot)], Parameters, Level=(0,1))

  elif Parameters.pressureLocation == 'Top,Bottom':
    try:
      a_g[(0,0)+Parameters.pressureDOFTop] = computeFluidPressure(a_T, Parameters.pressureT0Top, Parameters.pressureT1Top, Parameters.pressureValueTop, Parameters.pressureApplication.split(',')[0], Parameters)
      a_g[(0,0)+Parameters.pressureDOFBot] = computeFluidPressure(a_T, Parameters.pressureT0Bot, Parameters.pressureT1Bot, Parameters.pressureValueBot, Parameters.pressureApplication.split(',')[1], Parameters)
      a_g[(0,1)+Parameters.pressureDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.pressureDOFTop)], Parameters, Level=(0,1))
      a_g[(0,1)+Parameters.pressureDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.pressureDOFBot)], Parameters, Level=(0,1))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid flux(es).
# ----------
# Arguments:
# ----------
# a_HEXT:     (float, size: # ndof (implicit) or # ndofP (explicit))   
#                       external force vector for "H" (balance of mass variational equation)
# a_T:        (float)   current simulation time 't'
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_HEXT:     (float, size: # ndof (implicit) or # ndofP (explicit))  
#                       modified external force vector for "H"
#-------------------------------------------------------------------------------------------------
def applyFlux(a_HEXT, a_T, Parameters):

  if Parameters.fluxLocation == 'Top':
    a_HEXT[Parameters.fluxDOFTop] = computeFluidFlux(a_T, Parameters.fluxT0, Parameters.fluxT1,\
                                                     Parameters.fluxValue, Parameters.fluxApplication,\
                                                     Parameters)

  elif Parameters.fluxLocation == 'Bottom':
    a_HEXT[Parameters.fluxDOFBot] = computeFluidFlux(a_T, Parameters.fluxT0, Parameters.fluxT1,\
                                                     Parameters.fluxValue, Parameters.fluxApplication,\
                                                     Parameters)

  elif Parameters.fluxLocation == 'Top,Bottom':
    try:
      a_HEXT[Parameters.fluxDOFTop] = computeFluidFlux(a_T, Parameters.fluxT0Top, Parameters.fluxT1Top,\
                                                       Parameters.fluxValueTop, Parameters.fluxApplication.split(',')[0],\
                                                       Parameters)
      a_HEXT[Parameters.fluxDOFBot] = computeFluidFlux(a_T, Parameters.fluxT0Bot, Parameters.fluxT1Bot,\
                                                       Parameters.fluxValueBot, Parameters.fluxApplication.split(',')[1],\
                                                       Parameters)
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure Neumann BC not set appropriately.")
      raise RuntimeError

  return a_HEXT
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid displacement.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applyFluidDisplacement(a_g, a_T, Parameters):

  if Parameters.fluidDisplacementLocation == 'Top':
    a_g[(0,0)+Parameters.fluidDisplacementDOFTop] = computeFluidDisplacement(a_T, Parameters.fluidDisplacementT0, Parameters.fluidDisplacementT1, Parameters.fluidDisplacementValue, Parameters.fluidDisplacementApplication, Parameters)
    a_g[(0,1)+Parameters.fluidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFTop)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.fluidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFTop)], Parameters, Level=(0,2))

  elif Parameters.fluidDisplacementLocation == 'Bottom':
    a_g[(0,0)+Parameters.fluidDisplacementDOFBot] = computeFluidDisplacement(a_T, Parameters.fluidDisplacementT0, Parameters.fluidDisplacementT1, Parameters.fluidDisplacementValue, Parameters.fluidDisplacementApplication, Parameters)
    a_g[(0,1)+Parameters.fluidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFBot)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.fluidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFBot)], Parameters, Level=(0,2))
  
  elif Parameters.fluidDisplacementLocation == 'Top,Bottom':
    try:
      a_g[(0,0)+Parameters.fluidDisplacementDOFTop] = computeFluidDisplacement(a_T, Parameters.fluidDisplacementT0Top, Parameters.fluidDisplacementT1Top, Parameters.fluidDisplacementValueTop, Parameters.fluidDisplacementApplication.split(',')[0], Parameters)
      a_g[(0,0)+Parameters.fluidDisplacementDOFBot] = computeFluidDisplacement(a_T, Parameters.fluidDisplacementT0Bot, Parameters.fluidDisplacementT1Bot, Parameters.fluidDisplacementValueBot, Parameters.fluidDisplacementApplication.split(',')[1], Parameters)
      a_g[(0,1)+Parameters.fluidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFTop)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.fluidDisplacementDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFTop)], Parameters, Level=(0,2))
      a_g[(0,1)+Parameters.fluidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFBot)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.fluidDisplacementDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidDisplacementDOFBot)], Parameters, Level=(0,2))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid displacement Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid velocity.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applyFluidVelocity(a_g, a_T, Parameters):

  if Parameters.fluidVelocityLocation == 'Top,Bottom':
    try:
      if not (Parameters.fluidVelocityApplication.split(',')[0] == 'No-Slip' or Parameters.fluidVelocityApplication.split(',')[0] == 'Lagrange'):
        a_g[(0,1)+Parameters.fluidVelocityDOFTop] = computeFluidVelocity(a_T, Parameters.fluidVelocityT0Top, Parameters.fluidVelocityT1Top, Parameters.fluidVelocityValueTop, Parameters.fluidVelocityApplication.split(',')[0], Parameters)
        a_g[(0,2)+Parameters.fluidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFTop)], Parameters, Level=(1,2))
        a_g[(0,0)+Parameters.fluidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFTop)], Parameters, Level=(1,0))
      
      if not (Parameters.fluidVelocityApplication.split(',')[1] == 'No-Slip' or Parameters.fluidVelocityApplication.split(',')[1] == 'Lagrange'):
        a_g[(0,1)+Parameters.fluidVelocityDOFBot] = computeFluidVelocity(a_T, Parameters.fluidVelocityT0Bot, Parameters.fluidVelocityT1Bot, Parameters.fluidVelocityValueBot, Parameters.fluidVelocityApplication.split(',')[1], Parameters)
        a_g[(0,2)+Parameters.fluidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFBot)], Parameters, Level=(1,2))
        a_g[(0,0)+Parameters.fluidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFBot)], Parameters, Level=(1,0))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid velocity Dirichlet BC not set appropriately.")
      raise RuntimeError

  elif Parameters.fluidVelocityLocation == 'Top':
    if not (Parameters.fluidVelocityApplication == 'No-Slip' or Parameters.fluidVelocityApplication == 'Lagrange'):
      a_g[(0,1)+Parameters.fluidVelocityDOFTop] = computeFluidVelocity(a_T, Parameters.fluidVelocityT0, Parameters.fluidVelocityT1, Parameters.fluidVelocityValue, Parameters.fluidVelocityApplication, Parameters)
      a_g[(0,2)+Parameters.fluidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFTop)], Parameters, Level=(1,2))
      a_g[(0,0)+Parameters.fluidVelocityDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFTop)], Parameters, Level=(1,0))

  elif Parameters.fluidVelocityLocation == 'Bottom':
    if not (Parameters.fluidVelocityApplication == 'No-Slip' or Parameters.fluidVelocityApplication == 'Lagrange'):
      a_g[(0,1)+Parameters.fluidVelocityDOFBot] = computeFluidVelocity(a_T, Parameters.fluidVelocityT0, Parameters.fluidVelocityT1, Parameters.fluidVelocityValue, Parameters.fluidVelocityApplication, Parameters)
      a_g[(0,2)+Parameters.fluidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFBot)], Parameters, Level=(1,2))
      a_g[(0,0)+Parameters.fluidVelocityDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidVelocityDOFBot)], Parameters, Level=(1,0))

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied solid temperature.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applySolidTemp(a_g, a_T, Parameters):

  if Parameters.solidTempLocation == 'Top':
    a_g[(0,0)+Parameters.solidTempDOFTop] = computeSolidTemp(a_T, Parameters.solidTempT0, Parameters.solidTempT1, Parameters.solidTempValue, Parameters.solidTempApplication, Parameters)
    a_g[(0,1)+Parameters.solidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFTop)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.solidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFTop)], Parameters, Level=(0,2))

  elif Parameters.solidTempLocation == 'Bottom':
    a_g[(0,0)+Parameters.solidTempDOFBot] = computeSolidTemp(a_T, Parameters.solidTempT0, Parameters.solidTempT1, Parameters.solidTempValue, Parameters.solidTempApplication, Parameters)
    a_g[(0,1)+Parameters.solidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFBot)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.solidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFBot)], Parameters, Level=(0,2))

  elif Parameters.solidTempLocation == 'Top,Bottom':
    try:
      a_g[(0,0)+Parameters.solidTempDOFTop] = computeSolidTemp(a_T, Parameters.solidTempT0Top, Parameters.solidTempT1Top, Parameters.solidTempValueTop, Parameters.solidTempApplication.split(',')[0], Parameters)
      a_g[(0,0)+Parameters.solidTempDOFBot] = computeSolidTemp(a_T, Parameters.solidTempT0Bot, Parameters.solidTempT1Bot, Parameters.solidTempValueBot, Parameters.solidTempApplication.split(',')[1], Parameters)
      a_g[(0,1)+Parameters.solidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFTop)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.solidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFTop)], Parameters, Level=(0,2))
      a_g[(0,1)+Parameters.solidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFBot)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.solidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.solidTempDOFBot)], Parameters, Level=(0,2))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid temperature.
# ----------
# Arguments:
# ----------
# a_g:        (float, size: # element DOFs x # elements)  element-wise Dirichlet BCs
# a_T:        (float)                                     current simulation time 't'
# Parameters: (object)                                    problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_g:        (float, size: # element DOFs x # elements)  modified element-wise Dirichlet BCs
#-------------------------------------------------------------------------------------------------
def applyFluidTemp(a_g, a_T, Parameters):

  if Parameters.fluidTempLocation == 'Top':
    a_g[(0,0)+Parameters.fluidTempDOFTop] = computeSolidTemp(a_T, Parameters.fluidTempT0, Parameters.fluidTempT1,  Parameters.fluidTempValue, Parameters.fluidTempApplication, Parameters)
    a_g[(0,1)+Parameters.fluidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFTop)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.fluidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFTop)], Parameters, Level=(0,2))

  elif Parameters.fluidTempLocation == 'Bottom':
    a_g[(0,0)+Parameters.fluidTempDOFBot] = computeSolidTemp(a_T, Parameters.fluidTempT0, Parameters.fluidTempT1, Parameters.fluidTempValue, Parameters.fluidTempApplication, Parameters)
    a_g[(0,1)+Parameters.fluidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFBot)], Parameters, Level=(0,1))
    a_g[(0,2)+Parameters.fluidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFBot)], Parameters, Level=(0,2))
  
  elif Parameters.fluidTempLocation == 'Top,Bottom':
    try:
      a_g[(0,0)+Parameters.fluidTempDOFTop] = computeSolidTemp(a_T, Parameters.fluidTempT0Top, Parameters.fluidTempT1Top, Parameters.fluidTempValueTop, Parameters.fluidTempApplication.split(',')[0], Parameters)
      a_g[(0,0)+Parameters.fluidTempDOFBot] = computeSolidTemp(a_T, Parameters.fluidTempT0Bot, Parameters.fluidTempT1Bot, Parameters.fluidTempValueBot, Parameters.fluidTempApplication.split(',')[1], Parameters)
      a_g[(0,1)+Parameters.fluidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFTop)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.fluidTempDOFTop] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFTop)], Parameters, Level=(0,2))
      a_g[(0,1)+Parameters.fluidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFBot)], Parameters, Level=(0,1))
      a_g[(0,2)+Parameters.fluidTempDOFBot] = integrateDirichletBC(a_g[(...,*Parameters.fluidTempDOFBot)], Parameters, Level=(0,2))
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid temperature Dirichlet BC not set appropriately.")
      raise RuntimeError

  return a_g
#------------------------------------------------------------------------------------------------
# Function to set the applied  solid heat flux(es).
# ----------
# Arguments:
# ----------
# a_JEXT:     (float, size: # ndof (implicit) or # ndofTs (explicit) )
#                       external force vector for "J" (solid energy balance variational equation)
# a_T:        (float)   current simulation time 't'
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_JEXT:     (float, size: # ndof (implicit) or # ndofTs (explicit))  
#                       modified external force vector for "J"
#-------------------------------------------------------------------------------------------------
def applySolidHeatFlux(a_JEXT, a_T, Parameters):

  if Parameters.solidHeatFluxLocation == 'Top':
    a_JEXT[Parameters.solidHeatFluxDOFTop] = computeSolidHeatFlux(a_T, Parameters.solidHeatFluxT0, Parameters.solidHeatFluxT1, Parameters.solidHeatFluxValue, Parameters.solidHeatFluxApplication, Parameters)

  elif Parameters.solidHeatFluxLocation == 'Bottom':
    a_JEXT[Parameters.solidHeatFluxDOFBot] = computeSolidHeatFlux(a_T, Parameters.solidHeatFluxT0, Parameters.solidHeatFluxT1, Parameters.solidHeatFluxValue, Parameters.solidHeatFluxApplication, Parameters)

  elif Parameters.solidHeatFluxLocation == 'Top,Bottom':
    try:
      a_JEXT[Parameters.solidHeatFluxDOFTop] = computeSolidHeatFlux(a_T, Parameters.solidHeatFluxT0Top, Parameters.solidHeatFluxT1Top, Parameters.solidHeatFluxValueTop, Parameters.solidHeatFluxApplication.split(',')[0], Parameters)
      a_JEXT[Parameters.solidHeatFluxDOFBot] = computeSolidHeatFlux(a_T, Parameters.solidHeatFluxT0Bot, Parameters.solidHeatFluxT1Bot, Parameters.solidHeatFluxValueBot, Parameters.solidHeatFluxApplication.split(',')[1], Parameters)
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature Neumann BC not set appropriately.")
      raise RuntimeError

  return a_JEXT
#------------------------------------------------------------------------------------------------
# Function to set the applied pore fluid heat flux(es).
# ----------
# Arguments:
# ----------
# a_KEXT:     (float, size: # ndof (implicit) or # ndofTf (explicit)
#                       external force vector for "K" (fluid energy balance variational equation)
# a_T:        (float)   current simulation time 't'
# Parameters: (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# a_JEXT:     (float, size: # ndof (implicit) or # ndofTf (explicit))  
#                       modified external force vector for "K"
#-------------------------------------------------------------------------------------------------
def applyFluidHeatFlux(a_KEXT, a_T, Parameters):

  if Parameters.fluidHeatFluxLocation == 'Top':
    a_KEXT[Parameters.fluidHeatFluxDOFTop] = computeFluidHeatFlux(a_T, Parameters.fluidHeatFluxT0, Parameters.fluidHeatFluxT1, Parameters.fluidHeatFluxValue, Parameters.fluidHeatFluxApplication, Parameters)

  elif Parameters.fluidHeatFluxLocation == 'Bottom':
    a_KEXT[Parameters.fluidHeatFluxDOFBot] = computeFluidHeatFlux(a_T, Parameters.fluidHeatFluxT0, Parameters.fluidHeatFluxT1, Parameters.fluidHeatFluxValue, Parameters.fluidHeatFluxApplication, Parameters)

  elif Parameters.fluidHeatFluxLocation == 'Top,Bottom':
    try:
      a_KEXT[Parameters.fluidHeatFluxDOFTop] = computeFluidHeatFlux(a_T, Parameters.fluidHeatFluxT0Top, Parameters.fluidHeatFluxT1Top, Parameters.fluidHeatFluxValueTop, Parameters.fluidHeatFluxApplication.split(',')[0], Parameters)
      a_KEXT[Parameters.fluidHeatFluxDOFBot] = computeFluidHeatFlux(a_T, Parameters.fluidHeatFluxT0Bot, Parameters.fluidHeatFluxT1Bot, Parameters.fluidHeatFluxValueBot, Parameters.fluidHeatFluxApplication.split(',')[1], Parameters)
    except (AttributeError,ValueError):
      print("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid temperature Neumann BC not set appropriately.")
      raise RuntimeError
      
  return a_KEXT

#-----------------------
# End mid-level methods.
#-----------------------
#-------------------------
# Begin low-level methods.
#-------------------------
#----------------------------------------------------------------------------------------------
# Function to integrate arbitrary Dirichlet BCs in time using Newmark-beta method
# with constant acceleration, i.e., \beta = 1/4, \gamma = 1/2.
# ----------
# Arguments:
# ----------
# gDOF:       (float, size: 2 x 3)  the Dirichlet conditions at time (t_{n+1}, t_n) x (D, V, A)
#                                   for the DOF in question
# Parameters  (object)              problem parameters initiated in runMain.py
# Level:      (int,int)             tuple that describes level (in time) applied and level
#                                   (in time) that needs to be calculated
# --------
# Returns:
# --------
# (float):                          the integrated Dirichlet BC
#----------------------------------------------------------------------------------------------
def integrateDirichletBC(gDOF, Parameters, Level=(0,1)):
  #------------------
  # x_{n+1} is known.
  #------------------
  if Level[0] == 0:
    #-----------------------
    # Compute \dot{x}_{n+1}.
    #-----------------------
    if Level[1] == 1:
      return -gDOF[1,1] + (2/Parameters.dt)*(gDOF[0,0] - gDOF[1,0])
    #------------------------
    # Compute \ddot{x}_{n+1}.
    #------------------------
    elif Level[1] == 2:
      return (4/(Parameters.dt**2))*(gDOF[0,0] - gDOF[1,0]) - (4/Parameters.dt)*gDOF[1,1] - gDOF[1,2]
  #------------------------
  # \dot{x}_{n+1} is known.
  #------------------------
  elif Level[0] == 1:
    #------------------------
    # Compute \ddot{x}_{n+1}.
    #------------------------
    if Level[1] == 2:
      return (2/Parameters.dt)*(gDOF[0,1] - gDOF[1,1]) - gDOF[1,2]
    #-----------------
    # Compute x_{n+1}.
    #-----------------
    elif Level[1] == 0:
      return gDOF[1,0] + Parameters.dt*gDOF[1,1] + 0.25*(Parameters.dt**2)*(gDOF[0,2] + gDOF[1,2])
#------------------------------------------------------------------------------------------------
# Function to calculate the applied solid displacement at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated solid displacement BC
#-------------------------------------------------------------------------------------------------
def computeSolidDisplacement(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if Parameters.MMS:
    return 0

  else:
    if a_Application == 'Constant':
      if a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Step':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Impulse':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
      else:
        return 0
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid displacement application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to calculate the applied solid velocity at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated solid velocity BC
#-------------------------------------------------------------------------------------------------
def computeSolidVelocity(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid velocity application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied load at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Omega:        (float)   loading frequency (if applicable)
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated solid (or mixture) traction
#-------------------------------------------------------------------------------------------------
def computeTraction(a_T, a_T0, a_T1, a_Magnitude, a_Omega, a_Application, Parameters):

  if Parameters.MMS:
    # u(X, t) = X^2 t^2 ; du/dX = 2Xt^2; F = 1 + du/dX = 1 + 2Xt^2
    if Parameters.MMS_SolidSolutionType == 'S2T2':
      F = 1.0 + (2.0*Parameters.H0)*a_T**2
    # u(X, t) = X^2 t^3 ; du/dX = 2Xt^3; F = 1 + du/dX = 1 + 2Xt^3
    elif Parameters.MMS_SolidSolutionType == 'S2T3':
      F = 1.0 + (2.0*Parameters.H0)*a_T**3
    # u(X, t) = -X^2 t^3 ; du/dX = -2Xt^3; F = 1 + du/dX = 1 - 2Xt^3
    elif Parameters.MMS_SolidSolutionType == 'MS2T3':
      F = 1.0 - (2.0*Parameters.H0)*a_T**3
    # u(X, t) = X^3 t^2 ; du/dX = 3X^2t^2; F = 1 + du/dX = 1 + 3X^2t^2
    elif Parameters.MMS_SolidSolutionType == 'S3T2':
      F = 1.0 + (3.0*(Parameters.H0**2))*a_T**2
    # u(X, t) = X^3 t^3 ; du/dX = 3X^2t^3; F = 1 + du/dX = 1 + 3X^2t^3
    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      F = 1.0 + (3.0*(Parameters.H0**2))*a_T**3
    # u(X, t) = -X^3 t^3 ; du/dX = -3X^2t^3; F = 1 + du/dX = 1 - 3X^2t^3
    elif Parameters.MMS_SolidSolutionType == 'MS3T3':
      F = 1.0 - (3.0*(Parameters.H0**2))*a_T**3
    # u(X, t) = X^4 t^3 ; du/dX = 4X^3t^3; F = 1 + du/dX = 1 + 4X^3t^3
    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      F = 1.0 + (4.0*(Parameters.H0**3))*a_T**3
    # u(X, t) = -X^4 t^3 ; du/dX = -4X^3t^3; F = 1 + du/dX = 1 - 4X^3t^3
    elif Parameters.MMS_SolidSolutionType == 'MS4T3':
      F = 1.0 - (4.0*(Parameters.H0**3))*a_T**3
      
    if Parameters.solidModel == 'neo-Hookean':
      return F*(Parameters.mu + (Parameters.lambd*np.log(F) - Parameters.mu)/(F**2))*Parameters.Area 
    elif Parameters.solidModel == 'neo-Hookean-Eipper':
      return F*(Parameters.mu + (Parameters.lambd*((1 - Parameters.ns_0)**2)*(F/(1 - Parameters.ns_0) -\
                                                   F/(F - Parameters.ns_0)) - Parameters.mu)/(F**2))*\
             Parameters.Area 
    elif Parameters.solidModel == 'Saint-Venant-Kirchhoff':
      return F*(F**2 - 1)*(Parameters.lambd/2 + Parameters.mu)*Parameters.Area

  else:
    if a_Application == 'Friedlander':
      return -(a_Magnitude*(np.exp((-a_T/a_T0))*(1 - a_T/a_T0)) + Parameters.p_f0)*Parameters.Area
    elif a_Application == 'Sinusoidal':
      return -(0.5*a_Magnitude*(1 - np.cos(a_Omega*a_T)) + Parameters.p_f0)*Parameters.Area
    elif a_Application == 'Impulse':
      if a_T < a_T0:
        return -(a_Magnitude*(a_T/a_T0) + Parameters.p_f0)*Parameters.Area
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return -(a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1)) + Parameters.p_f0)*Parameters.Area
      else:
        return -Parameters.p_f0*Parameters.Area
    elif a_Application == 'Step':
      if a_T < a_T0:
        return -(a_Magnitude*(a_T/a_T0) + Parameters.p_f0)*Parameters.Area
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return (-a_Magnitude - Parameters.p_f0)*Parameters.Area
      else:
        return -Parameters.p_f0*Parameters.Area
    elif a_Application == 'Constant':
      return -(a_Magnitude + Parameters.p_f0)*Parameters.Area
    elif a_Application == 'Gravity-Step':
      if a_T < a_T0:
        Parameters.Gravity = -9.8*(a_Magnitude*((a_T)/a_T0))
      else:
        Parameters.Gravity = -9.8
    elif a_Application == 'Gravity-Impulse':
      if a_T < a_T0:
        Parameters.Gravity = -9.8*(a_Magnitude*(a_T/a_T0))
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        Parameters.Gravity = -9.8*(a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1)))
      else:
        Parameters.Gravity = 0
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTraction application not recognized.")
#-------------------------------------------------------------------------------------------------
# Function to compute the applied pore fluid pressure. 
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid pressure BC
#-------------------------------------------------------------------------------------------------
def computeFluidPressure(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if Parameters.MMS and a_Application == 'MMS':
    if Parameters.MMS_PressureSolutionType == 'S1T1':
      return Parameters.H0*a_T
    elif Parameters.MMS_PressureSolutionType == 'S1T2':
      return Parameters.H0*(a_T**2)
    elif Parameters.MMS_PressureSolutionType == 'S1T3':
      return Parameters.H0*(a_T**3)
    elif Parameters.MMS_PressureSolutionType == '2S1T3':
      return 2*Parameters.H0*(a_T**3)

  else:
    if a_Application == 'Constant':
      if a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Step':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Impulse':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
      else:
        return 0
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute pore fluid flux at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid flux BC
#------------------------------------------------------------------------------------------------
def computeFluidFlux(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if Parameters.MMS and a_Application == 'MMS':
    if Parameters.MMS_PressureSolutionType == 'S1T1':
      Dp_fDX_H = -Parameters.tk
    elif Parameters.MMS_PressureSolutionType == 'S1T2':
      Dp_fDX_H = -(Parameters.tk**2)
    elif Parameters.MMS_PressureSolutionType == 'S1T3' or Parameters.MMS_PressureSolutionType == '2S1T3':
      Dp_fDX_H = -(Parameters.tk**3)
    
    if Parameters.MMS_SolidSolutionType == 'S2T2':
      F_H = 1 + 2*Parameters.H0*Parameters.tk**2
      a_H = 2*(Parameters.H0**2)
    elif Parameters.MMS_SolidSolutionType == 'S2T3':
      F_H = 1 + 2*Parameters.H0*(Parameters.tk**3)
      a_H = 6*(Parameters.H0**2)*Parameters.tk
    elif Parameters.MMS_SolidSolutionType == 'S3T2':
      F_H = 1 + 3*(Parameters.H0**2)*(Parameters.tk**2)
      a_H = 2*(Parameters.H0**3)
    elif Parameters.MMS_SolidSolutionType == 'S3T3':
      F_H = 1 + 3*(Parameters.H0**2)*(Parameters.tk**3)
      a_H = 6*(Parameters.H0**3)*Parameters.tk
    elif Parameters.MMS_SolidSolutionType == 'S4T3':
      F_H = 1 + 4*(Parameters.H0**3)*(Parameters.tk**3)
      a_H = 6*(Parameters.H0**4)*Parameters.tk
    
    if Parameters.Physics == 'u-uf-pf':
      a_H *= 0.5
      if Parameters.MMS_FluidSolutionType.startswith('M'):
        a_H *= -1
    
    nf_H = 1 - Parameters.ns_0/F_H
    if Parameters.MMS_PressureSolutionType == '2S1T3':
      rhofR_H = Parameters.rhofR_0*Parameters.H0*(Parameters.tk**2)/Parameters.KF 
    else:
      rhofR_H = 0

    if 'Top' in Parameters.pressureLocation and Parameters.pressureApply:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nConflict: Dirichlet and Neumann BC both specified at X = H.")
    # The 1-D evaluation of the pore fluid is given as: 
    # Q = -(n^f v_f) = khat*(dp_f/dX*F11^-1 + rhofr*(a +g)) = const*F(nf(X=H))*(dp_f/dX(X=H)*F11(X=H)^-1 - rhofr(X=H)*(a(X=H) + grav))
    if Parameters.khatType == 'Kozeny-Carman':
      return Parameters.khat_mult*(nf_H**3/(1 - nf_H**2))*((Dp_fDX_H/F_H + rhofR_H*(a_H - Parameters.Gravity)))*Parameters.Area
    elif Parameters.khatType == 'Hyperbolic':
      return Parameters.khat_mult*(((F_H - Parameters.ns_0)/(1 - Parameters.ns_0))**Parameters.kappa)*((Dp_fDX_H/F_H + rhofR_H*(a_H - Parameters.Gravity)))*Parameters.Area
    elif Parameters.khatType == 'Constant':
      return Parameters.khat*((Dp_fDX_H/F_H + rhofR_H*(a_H - Parameters.Gravity)))*Parameters.Area
  else:
    if a_Application == 'Constant':
      if a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Step':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Impulse':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
      else:
        return 0
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid flux application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied fluid displacement at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid displacement BC
#-------------------------------------------------------------------------------------------------
def computeFluidDisplacement(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if Parameters.MMS:
    return 0

  else:
    if a_Application == 'Constant':
      if a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Step':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
        return a_Magnitude
      else:
        return 0
    elif a_Application == 'Impulse':
      if a_T < a_T0:
        return a_Magnitude*(a_T/a_T0)
      elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
        return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
      else:
        return 0
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid displacement application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied fluid velocity at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid velocity BC
#-------------------------------------------------------------------------------------------------
def computeFluidVelocity(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nFluid velocity application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied solid temperature at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated solid temperature BC
#-------------------------------------------------------------------------------------------------
def computeSolidTemp(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid temperature application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied pore fluid temperature at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid temperature BC
#-------------------------------------------------------------------------------------------------
def computeFluidTemp(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid temperature application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied solid heat flux at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated solid heat flux BC
#-------------------------------------------------------------------------------------------------
def computeSolidHeatFlux(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nSolid heat flux application not recognized.")
#------------------------------------------------------------------------------------------------
# Function to compute the applied pore fluid heat flux at ambiguous boundary.
# ----------
# Arguments:
# ----------
# a_T:            (float)   current simulation time 't'
# a_T0:           (float)   application start time
# a_T1:           (float)   application stop  time
# a_Magnitude:    (float)   application magnitude
# a_Application:  (string)  application type
# Parameters:     (object)  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# (float):        the updated pore fluid heat flux BC
#-------------------------------------------------------------------------------------------------
def computeFluidHeatFlux(a_T, a_T0, a_T1, a_Magnitude, a_Application, Parameters):

  if a_Application == 'Constant':
    if a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Step':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < a_T1 + 1e-12:
      return a_Magnitude
    else:
      return 0
  elif a_Application == 'Impulse':
    if a_T < a_T0:
      return a_Magnitude*(a_T/a_T0)
    elif a_T >= a_T0 and a_T < (a_T1 + 1e-12):
      return a_Magnitude*((a_T - a_T1)/(a_T0 - a_T1))
    else:
      return 0
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid heat flux application not recognized.")
#-----------------------
# End low-level methods.
#-----------------------
