#--------------------------------------------------------------------------------------------------
# Plotting script for displacement(s) at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 23, 2024
#--------------------------------------------------------------------------------------------------
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
#------------
# Arguments:
#------------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# a_DispDict       (dictionary)  mappings for displacement scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_DispDict, params):
  #---------------------
  # Generate file names.
  #---------------------
  simA_disp_fname = params.simA_Dir + 'displacement.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_disp_fname):
    sys.exit("------\nERROR:\n------\nDisplacement data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_disp_fname = params.simB_Dir + 'displacement.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_disp_fname):
      sys.exit("------\nERROR:\n------\nDisplacement data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_disp_fname = params.simC_Dir + 'displacement.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_disp_fname):
      sys.exit("------\nERROR:\n------\nDisplacement data not found for Simulation C.") 

  if params.simD_Dir is not None:
    simD_disp_fname = params.simD_Dir + 'displacement.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_disp_fname):
      sys.exit("------\nERROR:\n------\nDisplacement data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_disp_fname = params.simE_Dir + 'displacement.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_disp_fname):
      sys.exit("------\nERROR:\n------\nDisplacement data not found for Simulation E.")

  print("\nLoading in data...")
  #---------------------------------------------------------
  # Load in data and get closest nodes for requested probes.
  #---------------------------------------------------------
  #
  # Simulation A
  #
  simA_Dsolve     = np.load(simA_disp_fname)
  simA_preSimData = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams  = simA_preSimData[0]
  simA_coordsD    = simA_preSimData[3]
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
    simA_LM     = simA_preSimData[1]
    simA_DDOF   = getDisplacementDOF(params, simA_LM, simA_coordsD, params.simA_probe_1)
    if 'uf' in simA_simParams.Physics:
      simA_coordsDF = simA_preSimData[5]
      simA_DFDOF    = getFluidDOF(params, simA_simParams, simA_LM, simA_coordsDF, params.simA_probe_1)
  elif params.simA_isDYNA:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_Dsolve.shape[0])
    simA_DDOF   = getDisplacementDOFDYNA(simA_coordsD, params.simA_probe_1)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  if params.simA_probe_1 % simA_simParams.H0e > 1e-12:
    signA = r'\approx'
  else:
    signA = r'='
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_Dsolve     = np.load(simB_disp_fname)
    simB_preSimData = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams  = simB_preSimData[0]
    simB_coordsD    = simB_preSimData[3]
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
      simB_LM     = simB_preSimData[1]
      simB_DDOF   = getDisplacementDOF(params, simB_LM, simB_coordsD, params.simB_probe_1)
      if 'uf' in simB_simParams.Physics:
        simB_coordsDF = simB_preSimData[5]
        simB_DFDOF    = getFluidDOF(params, simB_simParams, simB_LM, simB_coordsDF, params.simB_probe_1)
    elif params.simB_isDYNA:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_Dsolve.shape[0])
      simB_DDOF   = getDisplacementDOFDYNA(simB_coordsD, params.simB_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
    if params.simB_probe_1 % simB_simParams.H0e > 1e-12:
      signB = r'\approx'
    else:
      signB = r'='
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_Dsolve     = np.load(simC_disp_fname)
    simC_preSimData = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams  = simC_preSimData[0]
    simC_coordsD    = simC_preSimData[3]
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
      simC_LM     = simC_preSimData[1]
      simC_DDOF   = getDisplacementDOF(params, simC_LM, simC_coordsD, params.simC_probe_1)
      if 'uf' in simC_simParams.Physics:
        simC_coordsDF = simC_preSimData[5]
        simC_DFDOF    = getFluidDOF(params, simC_simParams, simC_LM, simC_coordsDF, params.simC_probe_1)
    elif params.simC_isDYNA:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_Dsolve.shape[0])
      simC_DDOF   = getDisplacementDOFDYNA(simC_coordsD, params.simC_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
    if params.simC_probe_1 % simC_simParams.H0e > 1e-12:
      signC = r'\approx'
    else:
      signC = r'='
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_Dsolve     = np.load(simD_disp_fname)
    simD_preSimData = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams  = simD_preSimData[0]
    simD_coordsD    = simD_preSimData[3]
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
      simD_LM     = simD_preSimData[1]
      simD_DDOF   = getDisplacementDOF(params, simD_LM, simD_coordsD, params.simD_probe_1)
      if 'uf' in simD_simParams.Physics:
        simD_coordsDF = simD_preSimData[5]
        simD_DFDOF    = getFluidDOF(params, simD_simParams, simD_LM, simD_coordsDF, params.simD_probe_1)
    elif params.simD_isDYNA:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_Dsolve.shape[0])
      simD_DDOF   = getDisplacementDOFDYNA(simD_coordsD, params.simD_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
    if params.simD_probe_1 % simD_simParams.H0e > 1e-12:
      signD = r'\approx'
    else:
      signD = r'='
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_Dsolve     = np.load(simE_disp_fname)
    simE_preSimData = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams  = simE_preSimData[0]
    simE_coordsD    = simE_preSimData[3]
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
      simE_LM     = simE_preSimData[1]
      simE_DDOF   = getDisplacementDOF(params, simE_LM, simE_coordsD, params.simE_probe_1)
      if 'uf' in simE_simParams.Physics:
        simE_coordsDF = simE_preSimData[5]
        simE_DFDOF    = getFluidDOF(params, simE_simParams, simE_LM, simE_coordsDF, params.simE_probe_1)
    elif params.simE_isDYNA:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_Dsolve.shape[0])
      simE_DDOF   = getDisplacementDOFDYNA(simE_coordsD, params.simE_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
    if params.simE_probe_1 % simE_simParams.H0e > 1e-12:
      signE = r'\approx'
    else:
      signE = r'='

  print("Data loaded successfully.")

  print("\nGenerating plots...")

  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  #--------------------------
  # Plot solid displacements.
  #--------------------------
  if params.solidPlot:
    plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_DDOF]*params.dispScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$u(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOF]*params.dispScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$u(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOF]*params.dispScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$u(X' + signC + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOF]*params.dispScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$u(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOF]*params.dispScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$u(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #-------------------------------
  # Plot pore fluid displacements.
  #-------------------------------
  if params.fluidPlot:
    if 'uf' in simA_simParams.Physics:
      plt.plot(simA_tsolve[::params.simA_SkipSecondary]*params.timeScaling, simA_Dsolve[::params.simA_SkipSecondary,simA_DFDOF]*params.dispScaling, params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$u_\rf(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if 'uf' in simB_simParams.Physics:
        plt.plot(simB_tsolve[::params.simB_SkipSecondary]*params.timeScaling, simB_Dsolve[::params.simB_SkipSecondary,simB_DFDOF]*params.dispScaling, params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$u_\rf(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      if 'uf' in simC_simParams.Physics:
        plt.plot(simC_tsolve[::params.simC_SkipSecondary]*params.timeScaling, simC_Dsolve[::params.simC_SkipSecondary,simC_DFDOF]*params.dispScaling, params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$u_\rf(X' + signC + str(params.simC_probe_1) + r'\text{m}'+  r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if 'uf' in simD_simParams.Physics:
        plt.plot(simD_tsolve[::params.simD_SkipSecondary]*params.timeScaling, simD_Dsolve[::params.simD_SkipSecondary,simD_DFDOF]*params.dispScaling, params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$u_\rf(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      if 'uf' in simE_simParams.Physics:
        plt.plot(simE_tsolve[::params.simE_SkipSecondary]*params.timeScaling, simE_Dsolve[::params.simE_SkipSecondary,simE_DFDOF]*params.dispScaling, params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$u_\rf(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title) 

  if not params.no_labels:
    plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    plt.ylabel(r'Displacement ' + a_DispDict[params.dispScaling], fontsize=params.yAxisFontSize)
  
  if params.ylim0 is not None and params.ylim1 is not None:
    plt.ylim([params.ylim0, params.ylim1])
  if params.xlim0 is not None and params.xlim1 is not None:
    plt.xlim([params.xlim0, params.xlim1])

  if params.is_xticks:
    ax1.set_xticks(params.xticks)
  if params.secondaryXTicks:
    ax1.secondary_xaxis('top').set_xticklabels([])
  if params.is_xticklabels:
    ax1.set_xticklabels(params.xticklabels)

  if params.is_yticks:
    ax1.set_yticks(params.yticks)
  if params.secondaryYTicks:
    ax1.secondary_yaxis('right').set_yticklabels([])
  if params.is_yticklabels:
    ax1.set_yticklabels(params.yticklabels)

  if params.grid:
    plt.grid(True, which=params.gridWhich)

  if params.legend:
    plt.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
               handlelength=params.handleLength, fontsize=params.legendFontSize,\
               edgecolor='k', framealpha=1.0) 

  if params.title:
    fig1.suptitle(params.titleName,y=params.titleLoc,fontsize=params.titleFontSize)

  plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
  plt.close()

  return

