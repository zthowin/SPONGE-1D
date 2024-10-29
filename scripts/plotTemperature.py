#--------------------------------------------------------------------------------------------------
# Plotting script for temperature(s) at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 26, 2024
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
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, params):
  #---------------------
  # Generate file names.
  #---------------------
  if params.simA_isPython:
    simA_disp_fname = params.simA_Dir + 'displacement.npy'
  elif params.simA_isDYNA:
    simA_disp_fname = params.simA_Dir + 'ts.npy'
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
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
    simA_LM     = simA_preSimData[1]
    if simA_simParams.Physics == 'u-uf-pf-ts-tf':
      simA_coordsTs = simA_preSimData[6]
    elif simA_simParams.Physics == 'u-pf-ts-tf':
      simA_coordsTs = simA_preSimData[5]
    else:
      simA_coordsTs = simA_preSimData[4]
    simA_TsDOF  = getTsDOF(params, simA_simParams, simA_LM, simA_coordsTs, params.simA_probe_1)
    simA_Tlabel = r'$\theta$'
    if 'pf' in simA_simParams.Physics:
      simA_Tlabel = r'$\theta^\rs$'
      if 'uf' in simA_simParams.Physics:
        simA_coordsTf = simA_preSimData[7]
      else:
        simA_coordsTf = simA_preSimData[6]
      simA_TfDOF = getTfDOF(params, simA_simParams, simA_LM, simA_coordsTf, params.simA_probe_1)
  elif params.simA_isDYNA:
    simA_Tlabel      = r'$\theta$'
    simA_coordsGauss = simA_preSimData[2]
    simA_tsolve      = np.linspace(0, simA_simParams.TStop, simA_Dsolve.shape[0])
    simA_TsDOF       = getGaussDYNA(simA_coordsGauss, params.simA_probe_1)
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
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
      simB_LM     = simB_preSimData[1]
      if simB_simParams.Physics == 'u-uf-pf-ts-tf':
        simB_coordsTs = simB_preSimData[6]
      elif simB_simParams.Physics == 'u-pf-ts-tf':
        simB_coordsTs = simB_preSimData[5]
      else:
        simB_coordsTs = simB_preSimData[4]
      simB_TsDOF  = getTsDOF(params, simB_simParams, simB_LM, simB_coordsTs, params.simB_probe_1)
      simB_Tlabel = r'$\theta$'
      if 'pf' in simB_simParams.Physics:
        simB_Tlabel = r'$\theta^\rs$'
        if 'uf' in simB_simParams.Physics:
          simB_coordsTf = simB_preSimData[7]
        else:
          simB_coordsTf = simB_preSimData[6]
        simB_TfDOF = getTfDOF(params, simB_simParams, simB_LM, simB_coordsTf, params.simB_probe_1)
    elif params.simB_isDYNB:
      simB_Tlabel      = r'$\theta$'
      simB_coordsGauss = simA_preSimData[2]
      simB_tsolve      = np.linspace(0, simB_simParams.TStop, simB_Dsolve.shape[0])
      simB_TsDOF       = getGaussDYNA(simB_coordsGauss, params.simB_probe_1)
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
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
      simC_LM     = simC_preSimData[1]
      if simC_simParams.Physics == 'u-uf-pf-ts-tf':
        simC_coordsTs = simC_preSimData[6]
      elif simC_simParams.Physics == 'u-pf-ts-tf':
        simC_coordsTs = simC_preSimData[5]
      else:
        simC_coordsTs = simC_preSimData[4]
      simC_TsDOF  = getTsDOF(params, simC_simParams, simC_LM, simC_coordsTs, params.simC_probe_1)
      simC_Tlabel = r'$\theta$'
      if 'pf' in simC_simParams.Physics:
        simC_Tlabel = r'$\theta^\rs$'
        if 'uf' in simC_simParams.Physics:
          simC_coordsTf = simC_preSimData[7]
        else:
          simC_coordsTf = simC_preSimData[6]
        simC_TfDOF = getTfDOF(params, simC_simParams, simC_LM, simC_coordsTf, params.simC_probe_1)
    elif params.simC_isDYNC:
      simC_Tlabel      = r'$\theta$'
      simC_coordsGauss = simC_preSimData[2]
      simC_tsolve      = np.linspace(0, simC_simParams.TStop, simC_Dsolve.shape[0])
      simC_TsDOF       = getGaussDYNA(simC_coordsGauss, params.simB_probe_1)
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
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
      simD_LM     = simD_preSimData[1]
      if simD_simParams.Physics == 'u-uf-pf-ts-tf':
        simD_coordsTs = simD_preSimData[6]
      elif simD_simParams.Physics == 'u-pf-ts-tf':
        simD_coordsTs = simD_preSimData[5]
      else:
        simD_coordsTs = simD_preSimData[4]
      simD_TsDOF  = getTsDOF(params, simD_simParams, simD_LM, simD_coordsTs, params.simD_probe_1)
      simD_Tlabel = r'$\theta$'
      if 'pf' in simD_simParams.Physics:
        simD_Tlabel = r'$\theta^\rs$'
        if 'uf' in simD_simParams.Physics:
          simD_coordsTf = simD_preSimData[7]
        else:
          simD_coordsTf = simD_preSimData[6]
        simD_TfDOF = getTfDOF(params, simD_simParams, simD_LM, simD_coordsTf, params.simD_probe_1)
    elif params.simD_isDYND:
      simD_Tlabel      = r'$\theta$'
      simD_coordsGauss = simD_preSimData[2]
      simD_tsolve      = np.linspace(0, simD_simParams.TStop, simD_Dsolve.shape[0])
      simD_TsDOF       = getGaussDYNA(simD_coordsGauss, params.simD_probe_1)
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
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
      simE_LM     = simE_preSimData[1]
      if simE_simParams.Physics == 'u-uf-pf-ts-tf':
        simE_coordsTs = simE_preSimData[6]
      elif simE_simParams.Physics == 'u-pf-ts-tf':
        simE_coordsTs = simE_preSimData[5]
      else:
        simE_coordsTs = simE_preSimData[4]
      simE_TsDOF  = getTsDOF(params, simE_simParams, simE_LM, simE_coordsTs, params.simE_probe_1)
      simE_Tlabel = r'$\theta$'
      if 'pf' in simE_simParams.Physics:
        simE_Tlabel = r'$\theta^\rs$'
        if 'uf' in simE_simParams.Physics:
          simE_coordsTf = simE_preSimData[7]
        else:
          simE_coordsTf = simE_preSimData[6]
        simE_TfDOF = getTfDOF(params, simE_simParams, simE_LM, simE_coordsTf, params.simE_probe_1)
    elif params.simE_isDYNE:
      simE_Tlabel      = r'$\theta$'
      simE_coordsGauss = simE_preSimData[2]
      simE_tsolve      = np.linspace(0, simE_simParams.TStop, simE_Dsolve.shape[0])
      simE_TsDOF       = getGaussDYNA(simE_coordsGauss, params.simE_probe_1)
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
  #------------------------
  # Plot solid temperature.
  #------------------------
  if params.solidPlot:
    plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip,simA_TsDOF], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=simA_Tlabel + '$(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_TsDOF], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=simB_Tlabel + '$(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_TsDOF], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=simC_Tlabel + '$(X' + signC + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_TsDOF], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=simD_Tlabel + '$(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_TsDOF], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=simE_Tlabel + '$(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #-----------------------------
  # Plot pore fluid temperature.
  #-----------------------------
  if params.fluidPlot:
    if 'tf' in simA_simParams.Physics:
      plt.plot(simA_tsolve[::params.simA_SkipSecondary]*params.timeScaling, simA_Dsolve[::params.simA_SkipSecondary,simA_TfDOF], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$\theta^\rf(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if 'tf' in simB_simParams.Physics:
        plt.plot(simB_tsolve[::params.simB_SkipSecondary]*params.timeScaling, simB_Dsolve[::params.simB_SkipSecondary,simB_TfDOF], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$\theta^\rf(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      if 'tf' in simC_simParams.Physics:
        plt.plot(simC_tsolve[::params.simC_SkipSecondary]*params.timeScaling, simC_Dsolve[::params.simC_SkipSecondary,simC_TfDOF], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$\theta^\rf(X' + signC + str(params.simC_probe_1) + r'\text{m}'+  r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if 'tf' in simD_simParams.Physics:
        plt.plot(simD_tsolve[::params.simD_SkipSecondary]*params.timeScaling, simD_Dsolve[::params.simD_SkipSecondary,simD_TfDOF], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$\theta^\rf(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      if 'tf' in simE_simParams.Physics:
        plt.plot(simE_tsolve[::params.simE_SkipSecondary]*params.timeScaling, simE_Dsolve[::params.simE_SkipSecondary,simE_TfDOF], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$\theta^\rf(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title) 

  if not params.no_labels:
    plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    plt.ylabel(r'Temperature (K)', fontsize=params.yAxisFontSize)
  
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

  ax1.ticklabel_format(axis='both', style='plain', useOffset=False)

  plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
  plt.close()

  return

