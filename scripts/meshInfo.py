#------------------------------------------------------------------------------------------------
# Mid-level script to read pre-simulation information for 1-D Lagrangian finite-element simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 29, 2024
#------------------------------------------------------------------------------------------------
import sys, os

try:
  REPO = os.environ['REPO']
except KeyError:
  sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")

sys.path.insert(1, REPO + '/src/')

try:
  import simInput
  import runMain
  import moduleFE
  import classElement
except ImportError:
  sys.exit("MODULE WARNING. /src/ modules not found, check configuration.")

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")
#---------------------------------------------------------------------------------------
# Function to read simulation data and reconstruct mesh data.
#
# ----------
# Arguments:
# ----------
# Parameters:        (object)  the plot script parameters
# sim_Dir:           (string)  full path to the simulation results
# sim_InputFileName: (string)  simulation input file name
#
# --------
# Returns:
# --------
# sim_Params:        (object)                  the simulation Parameters object
# sim_LM:            (array, int, optional)    the simulation location matrix
# sim_CoordsGauss:   (array, float)            an array of sequential Gauss points
# sim_CoordsD:       (array, float)            locations of displacement DOFs
# sim_CoordsP:       (array, float, optional)  locations of pressure DOFs
# sim_CoordsDF:      (array, float, optional)  locations of fluid DOFs
#---------------------------------------------------------------------------------------
def readPreSimData(Parameters, sim_Dir=None, sim_InputFileName=None):
  #---------------------------------------------
  # Initialize the simulation input data object.
  #---------------------------------------------
  inputData = simInput.SimInputs(sim_Dir + sim_InputFileName)
  #-------------------------------------------
  # Read information from LS-DYNA simulations.
  #-------------------------------------------
  if sim_InputFileName.endswith('.k'):
    sim_Params      = inputData
    sim_Params.readDYNAInputFile()
    sim_CoordsGauss = sim_Params.coordsDYNAG
    sim_CoordsD     = sim_Params.coordsDYNA
    del sim_Params.coordsDYNAG
    del sim_Params.coordsDYNA
    return sim_Params, None, sim_CoordsGauss, sim_CoordsD, None
  #------------------------------------------
  # Read information from Python simulations.
  #------------------------------------------
  else:
    inputData.readInputFile()
    #--------------------------------------
    # Initialize the simulation parameters.
    #--------------------------------------
    sim_Params = runMain.Parameters(inputData)
    #--------------------------------------------
    # Reconstruct the simulation location matrix.
    #--------------------------------------------
    sim_LM = moduleFE.initLM(sim_Params)
    #---------------------------
    # Build the simulation mesh:
    #
    # First, the Gauss points.
    #---------------------------
    if Parameters.averageGauss:
      #----------------------------------
      # Gauss point at element centroids.
      #----------------------------------
      sim_CoordsGauss = np.linspace(sim_Params.H0e/2, sim_Params.H0 - sim_Params.H0e/2, sim_Params.ne)
    else:
      sim_CoordsGauss  = np.zeros((sim_Params.ne, sim_Params.Gauss_Order))
      sim_ModelElement = classElement.Element(a_GaussOrder=sim_Params.Gauss_Order,a_ID=0)
      sim_ModelElement.set_Gauss_Points(sim_Params)
      for e in range(0, sim_Params.ne):
        for xi in range(0, sim_Params.Gauss_Order):
          #-------------------------------------------------------
          # Gauss point mapping to global coordinates is given by:
          # X_i := (1/2)\xi_i + 1/2 
          #-------------------------------------------------------
          sim_CoordsGauss[e,xi] = sim_Params.H0e*(e + 0.5*sim_ModelElement.points[xi] + 0.5)
    #---------------------------
    # Build the simulation mesh:
    #
    # Next, the solid DOFs.
    #---------------------------
    sim_CoordsD = np.zeros((sim_Params.ndofSe, sim_Params.ne))
    for e in range(0, sim_Params.ne):
      for dofe in range(0, sim_Params.ndofSe):
        sim_CoordsD[dofe, e] = (dofe/(sim_Params.ndofSe - 1) + e)*sim_Params.H0e
    #-------------------------------------------------
    # Remove displacement gradient DOFs if applicable.
    #-------------------------------------------------
    if sim_Params.Element_Type.split('-')[0] == 'Q3H':
      sim_CoordsD = np.delete(sim_CoordsD, (1,2), axis=0)
    #---------------------------
    # Build the simulation mesh:
    #
    # Next, the fluid DOFs.
    #---------------------------
    if 'uf' in sim_Params.Physics:
      sim_CoordsDF = np.zeros((sim_Params.ndofFe, sim_Params.ne))
      for e in range(0, sim_Params.ne):
        for dofe in range(0, sim_Params.ndofFe):
          sim_CoordsDF[dofe, e] = (dofe/(sim_Params.ndofFe - 1) + e)*sim_Params.H0e
      #-------------------------------------------------
      # Remove displacement gradient DOFs if applicable.
      #-------------------------------------------------
      if sim_Params.Element_Type.split('-')[1] == 'Q3H':
        sim_CoordsDF = np.delete(sim_CoordsDF, (1,2), axis=0)
    #---------------------------
    # Build the simulation mesh:
    #
    # Next, the pressure DOFs.
    #---------------------------
    if 'pf' in sim_Params.Physics:
      sim_CoordsP = np.zeros((sim_Params.ndofPe, sim_Params.ne))
      for e in range(0, sim_Params.ne):
        for dofe in range(0, sim_Params.ndofPe):
          sim_CoordsP[dofe, e] = (dofe + e)*sim_Params.H0e
    #----------------------------
    # Build the simulation mesh:
    #
    # Next, the temperature DOFs.
    #----------------------------
    if 't' in sim_Params.Physics:
      sim_CoordsTs = np.zeros((sim_Params.ndofTse, sim_Params.ne))
      for e in range(0, sim_Params.ne):
        for dofe in range(0, sim_Params.ndofTse):
          sim_CoordsTs[dofe, e] = (dofe/(sim_Params.ndofTse - 1) + e)*sim_Params.H0e
      if 'tf' in sim_Params.Physics:
        sim_CoordsTf = np.zeros((sim_Params.ndofTfe, sim_Params.ne))
        for e in range(0, sim_Params.ne):
          for dofe in range(0, sim_Params.ndofTfe):
            sim_CoordsTf[dofe, e] = (dofe/(sim_Params.ndofTfe - 1) + e)*sim_Params.H0e
    
    if sim_Params.Physics == 'u-uf-pf-ts-tf':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD, sim_CoordsP, sim_CoordsDF, sim_CoordsTs, sim_CoordsTf
    elif sim_Params.Physics == 'u-pf-ts-tf':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD, sim_CoordsP,  sim_CoordsTs, sim_CoordsTf
    elif sim_Params.Physics == 'u-uf-pf':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD, sim_CoordsP, sim_CoordsDF
    elif sim_Params.Physics == 'u-pf':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD, sim_CoordsP
    elif sim_Params.Physics == 'u-t':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD, sim_CoordsTs
    elif sim_Params.Physics == 'u':
      return sim_Params, sim_LM, sim_CoordsGauss, sim_CoordsD
#---------------------------------------------------------------------------------------
# Function to find the nearest displacement DOF from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_LM:        (array, int)   the simulation location matrix
# a_Coords:    (array, float) the simulation displacement DOF locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_DDOF: (int)  the displacement DOF closest to the probe
#---------------------------------------------------------------------------------------
def getDisplacementDOF(Parameters, a_LM, a_Coords, a_Probe):
  #-------------------------------------------------------
  # Check to see if the probe lies on the DOF coordinates.
  #-------------------------------------------------------
  try:
    sim_Probe_DDOFe, sim_Probe_El = np.where(a_Coords == a_Probe)
    assert(len(sim_Probe_DDOFe) > 0)
  #--------------------------------------------------------
  # If not, round the location of probe up and check again.
  #--------------------------------------------------------
  except AssertionError:
    sim_Probe_DDOFe, sim_Probe_El = np.where(a_Coords == np.around(a_Probe, decimals=3))
    try:
      assert(len(sim_Probe_DDOFe) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  sim_Probe_DDOFe = sim_Probe_DDOFe[0]
  sim_Probe_El    = sim_Probe_El[0]
  sim_Probe_DDOF  = a_LM[sim_Probe_DDOFe, sim_Probe_El]
  return sim_Probe_DDOF
#---------------------------------------------------------------------------------------
# Function to find the nearest displacement DOF from a LS-DYNA simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)  the plot script parameters
# a_Probe:     (float)   the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_DDOF[0]: (int)  the displacement DOF closest to the probe
#---------------------------------------------------------------------------------------
def getDisplacementDOFDYNA(a_Coords, a_Probe):
  try:
    sim_Probe_DDOF = np.where(a_Coords == a_Probe)[0]
    assert(len(sim_Probe_DDOF) > 0)
  except AssertionError:
    sim_Probe_DDOF = np.where(a_Coords == np.around(a_Probe,decimals=3))[0]
    try:
      assert(len(sim_Probe_DDOF) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  return sim_Probe_DDOF[0]
#---------------------------------------------------------------------------------------
# Function to find the nearest pressure DOF from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_SimParams: (object)       the simulation Parameters object
# a_LM:        (array, int)   the simulation location matrix
# a_Coords:    (array, float) the simulation pressure DOF locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_PDOF: (int)  the pressure DOF closest to the probe
#---------------------------------------------------------------------------------------
def getPressureDOF(Parameters, a_SimParams, a_LM, a_Coords, a_Probe):
  try:
    sim_Probe_PDOFe, sim_Probe_El = np.where(a_Coords == a_Probe)
    assert(len(sim_Probe_PDOFe) > 0)
  except AssertionError:
    sim_Probe_PDOFe, sim_Probe_El = np.where(a_Coords == np.around(a_Probe))
    try:
      assert(len(sim_Probe_PDOFe) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  sim_Probe_PDOFe = sim_Probe_PDOFe[0] + a_SimParams.ndofSe + a_SimParams.ndofFe
  sim_Probe_El    = sim_Probe_El[0]
  sim_Probe_PDOF  = a_LM[sim_Probe_PDOFe, sim_Probe_El]
  return sim_Probe_PDOF
#---------------------------------------------------------------------------------------
# Function to find the nearest fluid DOF from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_SimParams: (object)       the simulation Parameters object
# a_LM:        (array, int)   the simulation location matrix
# a_Coords:    (array, float) the simulation fluid DOF locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_DFDOF: (int)  the fluid DOF closest to the probe
#---------------------------------------------------------------------------------------
def getFluidDOF(Parameters, a_SimParams, a_LM, a_Coords, a_Probe):
  try:
    sim_Probe_DFDOFe, sim_Probe_El = np.where(a_Coords == a_Probe)
    assert(len(sim_Probe_DFDOFe) > 0)
  except AssertionError:
    sim_Probe_DFDOFe, sim_Probe_El = np.where(a_Coords == np.around(a_Probe))
    try:
      assert(len(sim_Probe_DFDOFe) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  sim_Probe_DFDOFe = sim_Probe_DFDOFe[0] + a_SimParams.ndofSe
  sim_Probe_El     = sim_Probe_El[0]
  sim_Probe_DFDOF  = a_LM[sim_Probe_DFDOFe, sim_Probe_El]
  return sim_Probe_DFDOF
#---------------------------------------------------------------------------------------
# Function to find the nearest solid temperature DOF from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_SimParams: (object)       the simulation Parameters object
# a_LM:        (array, int)   the simulation location matrix
# a_Coords:    (array, float) the simulation pressure DOF locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_TsDOF: (int)  the solid temperature DOF closest to the probe
#---------------------------------------------------------------------------------------
def getTsDOF(Parameters, a_SimParams, a_LM, a_Coords, a_Probe):
  try:
    sim_Probe_TsDOFe, sim_Probe_El = np.where(a_Coords == a_Probe)
    assert(len(sim_Probe_TsDOFe) > 0)
  except AssertionError:
    sim_Probe_TsDOFe, sim_Probe_El = np.where(a_Coords == np.around(a_Probe))
    try:
      assert(len(sim_Probe_TsDOFe) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  if 'pf' in a_SimParams.Physics:
    sim_Probe_TsDOFe = sim_Probe_TsDOFe[0] + a_SimParams.ndofSe + a_SimParams.ndofFe + a_SimParams.ndofPe
  else:
    sim_Probe_TsDOFe = sim_Probe_TsDOFe[0] + a_SimParams.ndofSe
  sim_Probe_El     = sim_Probe_El[0]
  sim_Probe_TsDOF  = a_LM[sim_Probe_TsDOFe, sim_Probe_El]
  return sim_Probe_TsDOF
#---------------------------------------------------------------------------------------
# Function to find the nearest solid temperature DOF from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_SimParams: (object)       the simulation Parameters object
# a_LM:        (array, int)   the simulation location matrix
# a_Coords:    (array, float) the simulation pressure DOF locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_TfDOF: (int)  the fluid temperature DOF closest to the probe
#---------------------------------------------------------------------------------------
def getTfDOF(Parameters, a_SimParams, a_LM, a_Coords, a_Probe):
  try:
    sim_Probe_TfDOFe, sim_Probe_El = np.where(a_Coords == a_Probe)
    assert(len(sim_Probe_TfDOFe) > 0)
  except AssertionError:
    sim_Probe_TfDOFe, sim_Probe_El = np.where(a_Coords == np.around(a_Probe))
    try:
      assert(len(sim_Probe_TfDOFe) > 0)
    except AssertionError:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  sim_Probe_TfDOFe = sim_Probe_TfDOFe[0] + a_SimParams.ndofSe + a_SimParams.ndofFe + a_SimParams.ndofPe + a_SimParams.ndofTse
  sim_Probe_El     = sim_Probe_El[0]
  sim_Probe_TfDOF  = a_LM[sim_Probe_TfDOFe, sim_Probe_El]
  return sim_Probe_TfDOF
#---------------------------------------------------------------------------------------
# Function to find the nearest Gauss point from a Python simulation to a probe.
#
# ----------
# Arguments:
# ----------
# Parameters:  (object)       the plot script parameters
# a_Coords:    (array, float) the simulation Gauss coordinate locations
# a_Probe:     (float)        the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_Gauss_El: (int)            the element closest to the probe
# sim_Probe_Gauss_Xi: (int, optional)  the Gauss point ID closest to the probe
#---------------------------------------------------------------------------------------
def getGaussPoint(Parameters, a_Coords, a_Probe):
  if Parameters.averageGauss:
    try:
      sim_Probe_Gauss_El = np.where(a_Coords == a_Probe)
      assert(len(sim_Probe_Gauss_El) > 0)
    except AssertionError:
      try:
        sim_Probe_Gauss_El = np.where(a_Coords == np.around(a_Probe))
        assert(len(sim_Probe_Gauss_El) > 0)
      except AssertionError:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
    return sim_Probe_Gauss_El
  
  else:
    try:
      sim_Probe_Gauss_El, sim_Probe_Gauss_Xi = np.where(a_Coords == a_Probe)
      assert(len(sim_Probe_Gauss_El) > 0)
    except AssertionError:
      try:
        sim_Probe_Gauss_El, sim_Probe_Gauss_Xi = np.where(a_Coords == np.around(a_Probe))
        assert(len(sim_Probe_Gauss_El) > 0)
      except AssertionError:
        try:
          dx = a_Coords[1][0] - a_Coords[0][0]
          sim_Probe_Gauss_El, sim_Probe_Gauss_Xi = np.where(np.isclose(a_Coords,a_Probe,rtol=2e-2*dx,atol=2e-2*dx))
          assert(len(sim_Probe_Gauss_El) > 0)
        except AssertionError:
          try:
            sim_Probe_Gauss_El, sim_Probe_Gauss_Xi = np.where(np.isclose(a_Coords,a_Probe,rtol=2e-1*dx,atol=2e-1*dx))
            assert(len(sim_Probe_Gauss_El) > 0)
          except AssertionError:
            try:
              sim_Probe_Gauss_El, sim_Probe_Gauss_Xi = np.where(np.isclose(a_Coords,a_Probe,atol=dx/2))
              assert(len(sim_Probe_Gauss_El) > 0)
            except AssertionError:
              sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
            
    return sim_Probe_Gauss_El[0], sim_Probe_Gauss_Xi[0]
#---------------------------------------------------------------------------------------
# Function to find the nearest Gauss point from an LS-DYNA simulation to a probe.
#
# ----------
# Arguments:
# ----------
# a_Coords:    (array, float) the simulation Gauss coordinate locations
# a_Probe:     (float)  the plot probe
#
# --------
# Returns:
# --------
# sim_Probe_El: (int)  the element closest to the probe
#---------------------------------------------------------------------------------------
def getGaussDYNA(a_Coords, a_Probe):
  try:
    sim_Probe_El = np.where(a_Coords == a_Probe)[0]
    assert(len(sim_Probe_El) > 0)
  except AssertionError:
    sim_Probe_El = np.where(a_Coords == np.around(a_Probe,decimals=3))[0]
    try:
      assert(len(sim_Probe_El) > 0)
    except AssertionError:
      dx           = a_Coords[1] - a_Coords[0]
      sim_Probe_El = np.where(np.isclose(a_Coords,a_Probe,atol=dx/2))[0]
      try:
        assert(len(sim_Probe_El) > 0)
      except AssertionError:
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nMesh coordinates cannot capture probing location.")
  if len(sim_Probe_El > 1):
    sim_Probe_El = sim_Probe_El[0]
  return sim_Probe_El

