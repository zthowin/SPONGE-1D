#--------------------------------------------------------------------------------------------------
# Plotting script for de Boer analytical solution.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 23, 2024
#--------------------------------------------------------------------------------------------------
import sys

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import matplotlib.pyplot as plt
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
  from meshInfo import *
except ImportError:
  sys.exit("MODULE WARNING. meshInfo not found, check configuration.")

#--------------------------------------------------------------------------------------------------
# Function to generate plots comparing de Boer's analytical solution (1993) to numerical data.
#------------
# Parameters:
#------------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# a_DispDict       (dictionary)  mappings for displacement scale factors to appropriate labels
# a_StressDict     (dictionary)  mappings for stress scale factors to appropriate labels
# params           (object)      problem parameters initiated below via the class above
#--------------------------------------------------------------------------------------------------
def makeDeBoerPlots(a_TimeDict, a_DispDict, a_StressDict, params):

  simA_u_fname      = params.simA_Dir + 'displacement_std.npy'
  simA_time_fname   = params.simA_Dir + 'time_std.npy'        
  simA_P11_fname    = params.simA_Dir + 'stress_std.npy'
  simA_press_fname  = params.simA_Dir + 'press_std.npy'
  simA_uf_fname     = params.simA_Dir + 'displacement-uf_std.npy'        

  if params.simB_Dir is not None:
    simB_disp_fname  = params.simB_Dir + 'displacement.npy'
    simB_time_fname  = params.simB_Dir + 'time.npy'
    simB_P11_fname   = params.simB_Dir + 'P11.npy'            
    simB_press_fname = params.simB_Dir + 'pf-press.npy'
    if not os.path.isfile(simB_disp_fname):
      sys.exit("\n------\nERROR:\n------\nDisplacement data not found for Simulation B.")
    if params.simB_isPython:
      if not os.path.isfile(simB_P11_fname):
        sys.exit("\n------\nERROR:\n------\nStress data not found for Simulation B.")
    if params.simB_isDYNA:
      if not os.path.isfile(simB_press_fname):
        sys.exit("\n------\nERROR:\n------\nPore fluid pressure data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_disp_fname  = params.simC_Dir + 'displacement.npy'
    simC_time_fname  = params.simC_Dir + 'time.npy'
    simC_P11_fname   = params.simC_Dir + 'P11.npy'            
    simC_press_fname = params.simC_Dir + 'pf-press.npy'
    if not os.path.isfile(simC_disp_fname):
      sys.exit("\n------\nERROR:\n------\nDisplacement data not found for Simulation C.")
    if params.simC_isPython:
      if not os.path.isfile(simC_P11_fname):
        sys.exit("\n------\nERROR:\n------\nStress data not found for Simulation C.")
    if params.simC_isDYNA:
      if not os.path.isfile(simC_press_fname):
        sys.exit("\n------\nERROR:\n------\nPore fluid pressure data not found for Simulation C.")            
  if params.simD_Dir is not None:
    simD_disp_fname  = params.simD_Dir + 'displacement.npy'
    simD_time_fname  = params.simD_Dir + 'time.npy'
    simD_P11_fname   = params.simD_Dir + 'P11.npy'             
    simD_press_fname = params.simD_Dir + 'pf-press.npy'
    if not os.path.isfile(simD_disp_fname):
      sys.exit("\n------\nERROR:\n------\nDisplacement data not found for Simulation D.")
    if params.simD_isPython:
      if not os.path.isfile(simD_P11_fname):
        sys.exit("\n------\nERROR:\n------\nStress data not found for Simulation D.")
    if params.simD_isDYNA:
      if not os.path.isfile(simD_press_fname):
        sys.exit("\n------\nERROR:\n------\nPore fluid pressure data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_disp_fname  = params.simE_Dir + 'displacement.npy'
    simE_time_fname  = params.simE_Dir + 'time.npy'   
    simE_P11_fname   = params.simE_Dir + 'P11.npy'     
    simE_press_fname = params.simE_Dir + 'pf-press.npy'
    if not os.path.isfile(simE_disp_fname):
      sys.exit("\n------\nERROR:\n------\nDisplacement data not found for Simulation E.")
    if params.simE_isPython:
      if not os.path.isfile(simE_P11_fname):
        sys.exit("\n------\nERROR:\n------\nStress data not found for Simulation E.")
    if params.simE_isDYNA:
      if not os.path.isfile(simE_press_fname):
        sys.exit("\n------\nERROR:\n------\nPore fluid pressure data not found for Simulation E.")

  ext = params.filename.split('.')[1]
  fn  = params.filename.split('.')[0]

  disp_u_plot_fname     = params.outputDir + fn + '-disp-u.'   + ext
  disp_uf_plot_fname    = params.outputDir + fn + '-disp-uf.'  + ext
  pf_plot_fname         = params.outputDir + fn + '-press-PF.' + ext
  stress_plot_fname     = params.outputDir + fn + '-stress.'   + ext

  print("\nLoading in data...")

  try:
    simA_Dsolve  = np.load(simA_u_fname).T
    simA_tsolve  = np.load(simA_time_fname).T
    simA_P11     = np.load(simA_P11_fname).T
    simA_Psolve  = np.load(simA_press_fname).T
    simA_DFsolve = np.load(simA_uf_fname).T
  except FileNotFoundError:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nFor de Boer plotting, Simulation A should contain the directory to the analytical solution.")
  #--------------------------------------
  # Assumes 10m column with 100 elements.
  #--------------------------------------
  if 'Q2' in params.simA_Dir or 'testingPackage' in params.simA_Dir:
    params.simA_H0e = 10/100
    
    simA_H1_Dloc  = int((params.simA_probe_1)*(2/params.simA_H0e))
    simA_H2_Dloc  = int((params.simA_probe_2)*(2/params.simA_H0e))
    simA_H3_Dloc  = int((params.simA_probe_3)*(2/params.simA_H0e))
    simA_H4_Dloc  = int((params.simA_probe_4)*(2/params.simA_H0e))
    simA_H1_Dlocf = int((params.simA_probe_1)*(2/params.simA_H0e))
    simA_H2_Dlocf = int((params.simA_probe_2)*(2/params.simA_H0e))
    simA_H3_Dlocf = int((params.simA_probe_3)*(2/params.simA_H0e))
    simA_H4_Dlocf = int((params.simA_probe_4)*(2/params.simA_H0e))
    simA_H1_Ploc  = int((params.simA_probe_1)*(2/params.simA_H0e))
    simA_H2_Ploc  = int((params.simA_probe_2)*(2/params.simA_H0e))
    simA_H3_Ploc  = int((params.simA_probe_3)*(2/params.simA_H0e))
    simA_H4_Ploc  = int((params.simA_probe_4)*(2/params.simA_H0e))
    simA_H1_Sloc  = int((params.simA_probe_1)*(2/params.simA_H0e))
    simA_H2_Sloc  = int((params.simA_probe_2)*(2/params.simA_H0e))
    simA_H3_Sloc  = int((params.simA_probe_3)*(2/params.simA_H0e))
    simA_H4_Sloc  = int((params.simA_probe_4)*(2/params.simA_H0e))
      
  elif 'Q1' in params.simA_Dir:

    simA_H1_Dloc  = int((params.simA_probe_1)*(1/params.simA_H0e))
    simA_H2_Dloc  = int((params.simA_probe_2)*(1/params.simA_H0e))
    simA_H3_Dloc  = int((params.simA_probe_3)*(1/params.simA_H0e))
    simA_H4_Dloc  = int((params.simA_probe_4)*(1/params.simA_H0e))
    simA_H1_Dlocf = int((params.simA_probe_1)*(1/params.simA_H0e))
    simA_H2_Dlocf = int((params.simA_probe_2)*(1/params.simA_H0e))
    simA_H3_Dlocf = int((params.simA_probe_3)*(1/params.simA_H0e))
    simA_H4_Dlocf = int((params.simA_probe_4)*(1/params.simA_H0e))
    simA_H1_Ploc  = int((params.simA_probe_1)*(1/params.simA_H0e))
    simA_H2_Ploc  = int((params.simA_probe_2)*(1/params.simA_H0e))
    simA_H3_Ploc  = int((params.simA_probe_3)*(1/params.simA_H0e))
    simA_H4_Ploc  = int((params.simA_probe_4)*(1/params.simA_H0e))
    simA_H1_Sloc  = int((params.simA_probe_1)*(1/params.simA_H0e))
    simA_H2_Sloc  = int((params.simA_probe_2)*(1/params.simA_H0e))
    simA_H3_Sloc  = int((params.simA_probe_3)*(1/params.simA_H0e))
    simA_H4_Sloc  = int((params.simA_probe_4)*(1/params.simA_H0e))
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_Dsolve      = np.load(simB_disp_fname)
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_LM          = simB_preSimData[1]
    simB_coordsGauss = simB_preSimData[2]
    simB_coordsD     = simB_preSimData[3]
    simB_coordsP     = simB_preSimData[4]
    simB_probeList   = [params.simB_probe_1,params.simB_probe_2,params.simB_probe_3,params.simB_probe_4]
    simB_DDOFList    = []
    simB_PDOFList    = []

    if params.simB_isPython:
      simB_P11       = np.load(simB_P11_fname)
      simB_tsolve    = np.load(simB_time_fname)
      simB_DFDOFList = []
      simB_GaussList = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simB_probeList)):
        simB_DDOFList.append(getDisplacementDOF(params, simB_LM, simB_coordsD, simB_probeList[probeIndex]))
        simB_PDOFList.append(getPressureDOF(params, simB_simParams, simB_LM, simB_coordsP, simB_probeList[probeIndex]))
        if 'uf' in simB_simParams.Physics:
          simB_coordsDF = simB_preSimData[5]
          simB_DFDOFList.append(getFluidDOF(params, simB_simParams, simB_LM, simB_coordsDF, simB_probeList[probeIndex]))
        simB_GaussList.append(getGaussPoint(params, simB_coordsGauss, simB_probeList[probeIndex]))

    else:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_Dsolve.shape[0])
      simB_press  = np.load(simB_press_fname)
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simB_probeList)):
        simB_DDOFList.append(getDisplacementDOFDYNA(simB_coordsD, simB_probeList[probeIndex]))
        simB_PDOFList.append(getGaussDYNA(simB_coordsGauss, simB_probeList[probeIndex]))
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_Dsolve      = np.load(simC_disp_fname)
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_LM          = simC_preSimData[1]
    simC_coordsGauss = simC_preSimData[2]
    simC_coordsD     = simC_preSimData[3]
    simC_coordsP     = simC_preSimData[4]
    simC_probeList   = [params.simC_probe_1,params.simC_probe_2,params.simC_probe_3,params.simC_probe_4]
    simC_DDOFList    = []
    simC_PDOFList    = []

    if params.simC_isPython:
      simC_P11       = np.load(simC_P11_fname)
      simC_tsolve    = np.load(simC_time_fname)
      simC_DFDOFList = []
      simC_GaussList = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simC_probeList)):
        simC_DDOFList.append(getDisplacementDOF(params, simC_LM, simC_coordsD, simC_probeList[probeIndex]))
        simC_PDOFList.append(getPressureDOF(params, simC_simParams, simC_LM, simC_coordsP, simC_probeList[probeIndex]))
        if 'uf' in simC_simParams.Physics:
          simC_coordsDF = simC_preSimData[5]
          simC_DFDOFList.append(getFluidDOF(params, simC_simParams, simC_LM, simC_coordsDF, simC_probeList[probeIndex]))
        simC_GaussList.append(getGaussPoint(params, simC_coordsGauss, simC_probeList[probeIndex]))
    else:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_Dsolve.shape[0])
      simC_press  = np.load(simC_press_fname)
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simC_probeList)):
        simC_DDOFList.append(getDisplacementDOFDYNA(simC_coordsD, simC_probeList[probeIndex]))
        simC_PDOFList.append(getGaussDYNA(simC_coordsGauss, simC_probeList[probeIndex]))
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_Dsolve      = np.load(simD_disp_fname)
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_LM          = simD_preSimData[1]
    simD_coordsGauss = simD_preSimData[2]
    simD_coordsD     = simD_preSimData[3]
    simD_coordsP     = simD_preSimData[4]
    simD_probeList   = [params.simD_probe_1,params.simD_probe_2,params.simD_probe_3,params.simD_probe_4]
    simD_DDOFList    = []
    simD_PDOFList    = []

    if params.simD_isPython:
      simD_P11        = np.load(simD_P11_fname)
      simD_tsolve     = np.load(simD_time_fname)
      simD_DFDOFList  = []
      simD_GaussList  = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simD_probeList)):
        simD_DDOFList.append(getDisplacementDOF(params, simD_LM, simD_coordsD, simD_probeList[probeIndex]))
        simD_PDOFList.append(getPressureDOF(params, simD_simParams, simD_LM, simD_coordsP, simD_probeList[probeIndex]))
        if 'uf' in simD_simParams.Physics:
          simD_coordsDF = simD_preSimData[5]
          simD_DFDOFList.append(getFluidDOF(params, simD_simParams, simD_LM, simD_coordsDF, simD_probeList[probeIndex]))
        simD_GaussList.append(getGaussPoint(params, simD_coordsGauss, simD_probeList[probeIndex]))

    else:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_Dsolve.shape[0])
      simD_press  = np.load(simD_press_fname)
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simD_probeList)):
        simD_DDOFList.append(getDisplacementDOFDYNA(simD_coordsD, simD_probeList[probeIndex]))
        simD_PDOFList.append(getGaussDYNA(simD_coordsGauss, simD_probeList[probeIndex]))
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_Dsolve      = np.load(simE_disp_fname)
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_LM          = simE_preSimData[1]
    simE_coordsGauss = simE_preSimData[2]
    simE_coordsD     = simE_preSimData[3]
    simE_coordsP     = simE_preSimData[4]
    simE_probeList   = [params.simE_probe_1,params.simE_probe_2,params.simE_probe_3,params.simE_probe_4]
    simE_DDOFList    = []
    simE_PDOFList    = []

    if params.simE_isPython:
      simE_P11       = np.load(simE_P11_fname)
      simE_tsolve    = np.load(simE_time_fname)
      simE_DFDOFList = []
      simE_GaussList = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simE_probeList)):
        simE_DDOFList.append(getDisplacementDOF(params, simE_LM, simE_coordsD, simE_probeList[probeIndex]))
        simE_PDOFList.append(getPressureDOF(params, simE_simParams, simE_LM, simE_coordsP, simE_probeList[probeIndex]))
        if 'uf' in simE_simParams.Physics:
          simE_coordsDF = simE_preSimData[5]
          simE_DFDOFList.append(getFluidDOF(params, simE_simParams, simE_LM, simE_coordsDF, simE_probeList[probeIndex]))
        simE_GaussList.append(getGaussPoint(params, simE_coordsGauss, simE_probeList[probeIndex]))

    else:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_Dsolve.shape[0])
      simE_press  = np.load(simE_press_fname)
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simE_probeList)):
        simE_DDOFList.append(getDisplacementDOFDYNA(simE_coordsD, simE_probeList[probeIndex]))
        simE_PDOFList.append(getGaussDYNA(simE_coordsGauss, simE_probeList[probeIndex]))

  print("Finished loading in data.")
  print("\nGenerating plots...")

  # For older data that did not save Dirichlet BCs in the data structure
#  simB_DDOFList = [a - 1 for a in simB_DDOFList]
#  simB_PDOFList = [a - 2 for a in simB_PDOFList]
#  simC_DDOFList = [a - 1 for a in simC_DDOFList]
#  simC_DFDOFList = [a - 2 for a in simC_DFDOFList]
#  simC_PDOFList = [a - 2 for a in simC_PDOFList]
#  simD_DDOFList = [a - 1 for a in simD_DDOFList]
#  simD_DFDOFList = [a - 2 for a in simD_DFDOFList]
#  simD_PDOFList = [a - 2 for a in simD_PDOFList]
  #-------------------------
  # Plot solid displacement.
  #-------------------------
  fig1 = plt.figure(1)
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_H4_Dloc]*params.dispScaling, 'k-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_4) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_H3_Dloc]*params.dispScaling, 'b-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_3) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_H2_Dloc]*params.dispScaling, 'r-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_2) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_H1_Dloc]*params.dispScaling, 'm-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_1) + 'm')

  if params.simB_Dir is not None:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[0]]*params.dispScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[1]]*params.dispScaling, 'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[2]]*params.dispScaling, 'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[3]]*params.dispScaling, 'ko', fillstyle='none')

  if params.simC_Dir is not None:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[0]]*params.dispScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[1]]*params.dispScaling, 'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[2]]*params.dispScaling, 'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[3]]*params.dispScaling, 'kx', fillstyle='none')

  if params.simD_Dir is not None:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[0]]*params.dispScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[1]]*params.dispScaling, 'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[2]]*params.dispScaling, 'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[3]]*params.dispScaling, 'k+', fillstyle='none')

  if params.simE_Dir is not None:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[0]]*params.dispScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[1]]*params.dispScaling, 'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[2]]*params.dispScaling, 'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[3]]*params.dispScaling, 'k^', fillstyle='none')

  plt.xlim([0, 0.4])
  plt.ylim([-4, 2])
  # plt.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
  plt.legend(bbox_to_anchor=(0.113, 0.888), loc='center', edgecolor='k', framealpha=1.0)
  if params.text:
    if params.simE_Dir is not None:
      plt.text(0.307, 0.72, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title + '\n' + r'$\triangle$ ' + params.simE_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simD_Dir is not None:
      plt.text(0.307, 0.76, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simC_Dir is not None:
      plt.text(0.307, 0.787, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0), alpha=1.0)
  # else:
  #   plt.text(0.935, 0.6, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title, \
  #            fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black', pad=3.0))
  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Displacement ' + a_DispDict[params.dispScaling], fontsize=16)
  if params.title == False:
    # fig1.suptitle('Solid skeleton displacement', fontsize=16)
    pass
  else:
    fig1.suptitle(params.titleName + r' Solid skeleton displacement')
  if params.grid:
    plt.grid(True, which=params.gridWhich)
  plt.savefig(disp_u_plot_fname, bbox_inches='tight', dpi=params.DPI)

  # errorB = np.abs(simA_Dsolve[2001,-1] - simB_Dsolve[50,simB_DDOFList[0]])/np.abs(simA_Dsolve[2001,-1])
  # print(errorB)
  # if params.simC_Dir is not None:
  #   errorC = np.abs((simA_Dsolve[2001,simA_H1_Dloc] - simC_Dsolve[50,simC_DDOFList[0]]))/np.abs(simA_Dsolve[2001,simA_H1_Dloc])
  #   print(errorC)
  # if params.simD_Dir is not None:
  #   errorD = np.abs((simA_Dsolve[2001,simA_H1_Dloc] - simD_Dsolve[50,simD_DDOFList[0]]))/np.abs(simA_Dsolve[2001,simA_H1_Dloc])
  #   print(errorD)
  # if params.simE_Dir is not None:
  #   errorE = np.abs((simA_Dsolve[2001,simA_H1_Dloc] - simE_Dsolve[50,simE_DDOFList[0]]))/np.abs(simA_Dsolve[2001,simA_H1_Dloc])
  #   print(errorE)
  #-------------------------
  # Plot fluid displacement.
  #-------------------------
  fig2 = plt.figure(2)
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, -simA_DFsolve[::params.simA_Skip, simA_H4_Dlocf]*params.dispScaling, 'k-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_4) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, -simA_DFsolve[::params.simA_Skip, simA_H3_Dlocf]*params.dispScaling, 'b-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_3) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, -simA_DFsolve[::params.simA_Skip, simA_H2_Dlocf]*params.dispScaling, 'r-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_2) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, -simA_DFsolve[::params.simA_Skip, simA_H1_Dlocf]*params.dispScaling, 'm-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_1) + 'm')
  
  if params.simB_Dir is not None and params.simB_isPython and 'uf' in simB_simParams.Physics:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DFDOFList[0]]*params.dispScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DFDOFList[1]]*params.dispScaling, 'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DFDOFList[2]]*params.dispScaling, 'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DFDOFList[3]]*params.dispScaling, 'ko', fillstyle='none')

  if params.simC_Dir is not None and params.simC_isPython and 'uf' in simC_simParams.Physics:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DFDOFList[0]]*params.dispScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DFDOFList[1]]*params.dispScaling, 'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DFDOFList[2]]*params.dispScaling, 'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DFDOFList[3]]*params.dispScaling, 'kx', fillstyle='none')

  if params.simD_Dir is not None and params.simD_isPython and 'uf' in simD_simParams.Physics:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DFDOFList[0]]*params.dispScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DFDOFList[1]]*params.dispScaling, 'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DFDOFList[2]]*params.dispScaling, 'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DFDOFList[3]]*params.dispScaling, 'k+', fillstyle='none')

  if params.simE_Dir is not None and params.simE_isPython and 'uf' in simE_simParams.Physics:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DFDOFList[0]]*params.dispScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DFDOFList[1]]*params.dispScaling, 'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DFDOFList[2]]*params.dispScaling, 'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DFDOFList[3]]*params.dispScaling, 'k^', fillstyle='none')
  
  plt.xlim([0, 0.4])
  plt.ylim([0, 6])
  # plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
  plt.legend(bbox_to_anchor=(0.113, 0.888), loc='center', edgecolor='k', framealpha=1.0)
  if params.text:
    if params.simE_Dir is not None:
#      plt.text(0.335, 0.754, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title + '\n' + r'$\triangle$ ' + params.simE_Title, \
#               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
      plt.text(0.307, 0.787, r'-- ' + params.simA_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simD_Dir is not None:
      plt.text(0.307, 0.76, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simC_Dir is not None:
      plt.text(0.307, 0.787, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0), alpha=1.0)
  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Displacement ' + a_DispDict[params.dispScaling], fontsize=16)
  if params.title == False:
    # fig2.suptitle('Pore fluid displacement', fontsize=16)
    pass
  else:
    fig2.suptitle(params.titleName + r' Pore fluid displacement')
  if params.grid:
    plt.grid(True, which=params.gridWhich)
  plt.savefig(disp_uf_plot_fname, bbox_inches='tight', dpi=params.DPI)
  #--------------------------
  # Plot pore fluid pressure.
  #--------------------------
  fig3 = plt.figure(3)
  
  plt.plot(simA_tsolve[2::params.simA_Skip]*params.timeScaling, -simA_Psolve[::params.simA_Skip, simA_H4_Ploc]*params.stressScaling, 'k-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_4) + 'm')
  plt.plot(simA_tsolve[2::params.simA_Skip]*params.timeScaling, -simA_Psolve[::params.simA_Skip, simA_H3_Ploc]*params.stressScaling, 'b-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_3) + 'm')
  plt.plot(simA_tsolve[2::params.simA_Skip]*params.timeScaling, -simA_Psolve[::params.simA_Skip, simA_H2_Ploc]*params.stressScaling, 'r-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_2) + 'm')
  plt.plot(simA_tsolve[2::params.simA_Skip]*params.timeScaling, -simA_Psolve[::params.simA_Skip, simA_H1_Ploc]*params.stressScaling, 'm-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_1) + 'm')

  if params.simB_Dir is not None and params.simB_isPython:
#    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, np.zeros(np.shape(simB_tsolve[::params.simB_Skip])[0]),            'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_PDOFList[0]]*params.stressScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_PDOFList[1]]*params.stressScaling, 'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_PDOFList[2]]*params.stressScaling, 'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_PDOFList[3]]*params.stressScaling, 'ko', fillstyle='none')

  elif params.simB_Dir is not None and params.simB_isDYNA:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, 0*simB_press[::params.simB_Skip,simB_PDOFList[0]]*params.stressScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_press[::params.simB_Skip,simB_PDOFList[1]]*params.stressScaling,   'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_press[::params.simB_Skip,simB_PDOFList[2]]*params.stressScaling,   'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_press[::params.simB_Skip,simB_PDOFList[3]]*params.stressScaling,   'ko', fillstyle='none')

  if params.simC_Dir is not None and params.simC_isPython:
#    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, np.zeros(np.shape(simC_tsolve[::params.simC_Skip])[0]),            'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_PDOFList[0]]*params.stressScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_PDOFList[1]]*params.stressScaling, 'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_PDOFList[2]]*params.stressScaling, 'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_PDOFList[3]]*params.stressScaling, 'kx', fillstyle='none')
  
  elif params.simC_Dir is not None and params.simC_isDYNA:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, 0*simC_press[::params.simC_Skip,simC_PDOFList[0]]*params.stressScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_press[::params.simC_Skip,simC_PDOFList[1]]*params.stressScaling,   'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_press[::params.simC_Skip,simC_PDOFList[2]]*params.stressScaling,   'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_press[::params.simC_Skip,simC_PDOFList[3]]*params.stressScaling,   'kx', fillstyle='none')

  if params.simD_Dir is not None and params.simD_isPython:
#    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, np.zeros(np.shape(simD_tsolve[::params.simD_Skip])[0]), 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_PDOFList[0]]*params.stressScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_PDOFList[1]]*params.stressScaling, 'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_PDOFList[2]]*params.stressScaling, 'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_PDOFList[3]]*params.stressScaling, 'k+', fillstyle='none')

  elif params.simD_Dir is not None and params.simD_isDYNA:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, 0*simD_press[::params.simD_Skip,simD_PDOFList[0]]*params.stressScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_press[::params.simD_Skip,simD_PDOFList[1]]*params.stressScaling,   'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_press[::params.simD_Skip,simD_PDOFList[2]]*params.stressScaling,   'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_press[::params.simD_Skip,simD_PDOFList[3]]*params.stressScaling,   'k+', fillstyle='none')

  if params.simE_Dir is not None and params.simE_isPython:
    plt.plot(simE_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simE_Skip,simE_PDOFList[0]]*params.stressScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simD_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_PDOFList[1]]*params.stressScaling, 'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simD_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_PDOFList[2]]*params.stressScaling, 'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simD_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_PDOFList[3]]*params.stressScaling, 'k^', fillstyle='none')

  elif params.simE_Dir is not None and params.simE_isDYNA:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, 0*simE_press[::params.simE_Skip,simE_PDOFList[0]]*params.stressScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_press[::params.simE_Skip,simE_PDOFList[1]]*params.stressScaling,   'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_press[::params.simE_Skip,simE_PDOFList[2]]*params.stressScaling,   'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_press[::params.simE_Skip,simE_PDOFList[3]]*params.stressScaling,   'k^', fillstyle='none')

  plt.xlim([0, 0.4])
  plt.ylim([-20, 60])
  # plt.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
  plt.legend(bbox_to_anchor=(0.113, 0.888), loc='center', edgecolor='k', framealpha=1.0)
  if params.text:
    if params.simE_Dir is not None:
      plt.text(0.307, 0.72, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title + '\n' + r'$\triangle$ ' + params.simE_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simD_Dir is not None:
      plt.text(0.307, 0.76, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simC_Dir is not None:
      plt.text(0.307, 0.787, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0), alpha=1.0)
  # else:
  #   plt.text(0.935, 0.6, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title, \
  #            fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black', pad=3.0))

  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Pressure ' + a_StressDict[params.stressScaling], fontsize=16)
  if params.title == False:
    # fig3.suptitle('Pore fluid pressure', fontsize=16)
    pass
  else:
    fig3.suptitle(params.titleName + r' Pore fluid pressure')
  if params.grid:
    plt.grid(True, which=params.gridWhich)
  plt.savefig(pf_plot_fname, bbox_inches='tight', dpi=params.DPI)

  #-----------------------
  # Plot effective stress.
  #-----------------------
  fig4 = plt.figure(4)

  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_P11[::params.simA_Skip, simA_H4_Sloc]*params.stressScaling, 'k-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_4) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_P11[::params.simA_Skip, simA_H3_Sloc]*params.stressScaling, 'b-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_3) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_P11[::params.simA_Skip, simA_H2_Sloc]*params.stressScaling, 'r-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_2) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_P11[::params.simA_Skip, simA_H1_Sloc]*params.stressScaling, 'm-', label=r'$X = $' + ' ' + str(10 - params.simA_probe_1) + 'm')

  if params.simB_Dir is not None and params.simB_isPython:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_P11[::params.simB_Skip,simB_GaussList[0][0],simB_GaussList[0][1]]*params.stressScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_P11[::params.simB_Skip,simB_GaussList[1][0],simB_GaussList[1][1]]*params.stressScaling, 'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_P11[::params.simB_Skip,simB_GaussList[2][0],simB_GaussList[2][1]]*params.stressScaling, 'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_P11[::params.simB_Skip,simB_GaussList[3][0],simB_GaussList[3][1]]*params.stressScaling, 'ko', fillstyle='none')

  if params.simC_Dir is not None and params.simC_isPython:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_P11[::params.simC_Skip,simC_GaussList[0][0],simC_GaussList[0][1]]*params.stressScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_P11[::params.simC_Skip,simC_GaussList[1][0],simC_GaussList[1][1]]*params.stressScaling, 'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_P11[::params.simC_Skip,simC_GaussList[2][0],simC_GaussList[2][1]]*params.stressScaling, 'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_P11[::params.simC_Skip,simC_GaussList[3][0],simC_GaussList[3][1]]*params.stressScaling, 'kx', fillstyle='none')

  if params.simD_Dir is not None and params.simD_isPython:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_P11[::params.simD_Skip,simD_GaussList[0][0],simD_GaussList[0][1]]*params.stressScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_P11[::params.simD_Skip,simD_GaussList[1][0],simD_GaussList[1][1]]*params.stressScaling, 'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_P11[::params.simD_Skip,simD_GaussList[2][0],simD_GaussList[2][1]]*params.stressScaling, 'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_P11[::params.simD_Skip,simD_GaussList[3][0],simD_GaussList[3][1]]*params.stressScaling, 'k+', fillstyle='none')

  if params.simE_Dir is not None and params.simE_isPython:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_P11[::params.simE_Skip,simE_GaussList[0][0],simE_GaussList[0][1]]*params.stressScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_P11[::params.simE_Skip,simE_GaussList[1][0],simE_GaussList[1][1]]*params.stressScaling, 'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_P11[::params.simE_Skip,simE_GaussList[2][0],simE_GaussList[2][1]]*params.stressScaling, 'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_P11[::params.simE_Skip,simE_GaussList[3][0],simE_GaussList[3][1]]*params.stressScaling, 'k^', fillstyle='none')

  plt.xlim([0, 0.4])
  plt.ylim([-40, 20])
  # plt.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
  plt.legend(bbox_to_anchor=(0.113, 0.888), loc='center', edgecolor='k', framealpha=1.0)
  if params.text:
    if params.simE_Dir is not None:
#      plt.text(0.307, 0.72, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title + '\n' + r'$\triangle$ ' + params.simE_Title, \
#               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
      plt.text(0.307, 0.754, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simD_Dir is not None:
      plt.text(0.307, 0.76, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simC_Dir is not None:
      plt.text(0.307, 0.787, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title, \
               fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0), alpha=1.0)
  # else:
  #   plt.text(0.935, 0.6, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title, \
  #            fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black', pad=3.0))
  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Stress ' + a_StressDict[params.stressScaling], fontsize=16)
  # plt.yticks([-40, -30, -20, -10, 0, 15],['-40', '-30', '-20', '-10', '0', '15'])
  # plt.yticklabels(['-40', '-30', '-20', '-10', '0', '15'])
  if params.title == False:
    # fig4.suptitle('Solid skeleton effective stress', fontsize=16)
    pass
  else:
    fig4.suptitle(params.titleName + r' Solid skeleton stress')
  if params.grid:
    plt.grid(True, which=params.gridWhich)
  plt.savefig(stress_plot_fname, bbox_inches='tight', dpi=params.DPI)

  return

