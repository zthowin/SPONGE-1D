#--------------------------------------------------------------------------------------------------
# Plotting script for shear stress vs. J at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 26, 2024
#--------------------------------------------------------------------------------------------------
import os, sys

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
#-----------
# Arguments:
#-----------
# params      (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(params):
  #---------------------
  # Generate file names.
  #---------------------
  simA_tau_fname  = params.simA_Dir + 'tau.npy'
  simA_J_fname    = params.simA_Dir + 'J.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_tau_fname) or not os.path.isfile(simA_J_fname):
    sys.exit("-------\nERROR:\n-------\nStress and/or Jacobian data file not found for Simulation A.") 

  if params.simB_Dir is not None:
    simB_tau_fname  = params.simB_Dir + 'tau.npy'
    simB_J_fname    = params.simB_Dir + 'J.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_tau_fname) or not os.path.isfile(simB_J_fname):
      sys.exit("-------\nERROR:\n-------\nStress and/or Jacobian data file not found for Simulation B.")  

  if params.simC_Dir is not None:
    simC_tau_fname  = params.simC_Dir + 'tau.npy'
    simC_J_fname    = params.simC_Dir + 'J.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_tau_fname) or not os.path.isfile(simC_J_fname):
      sys.exit("-------\nERROR:\n-------\nStress and/or Jacobian data file not found for Simulation C.") 

  if params.simD_Dir is not None:
    simD_tau_fname  = params.simD_Dir + 'tau.npy'
    simD_J_fname    = params.simD_Dir + 'J.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_tau_fname) or not os.path.isfile(simD_J_fname):
      sys.exit("-------\nERROR:\n-------\nStress and/or Jacobian data file not found for Simulation D.") 

  if params.simE_Dir is not None:
    simE_tau_fname  = params.simE_Dir + 'tau.npy'
    simE_J_fname    = params.simE_Dir + 'J.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_tau_fname) or not os.path.isfile(simE_J_fname):
      sys.exit("-------\nERROR:\n-------\nStress and/or Jacobian data file not found for Simulation E.")

  print("\nLoading in data...")
  #----------------------------------------------------------------
  # Load in data and get closest Gauss points for requested probes.
  #----------------------------------------------------------------
  #
  # Simulation A
  #
  simA_J           = np.load(simA_J_fname)
  simA_tau         = np.load(simA_tau_fname)/params.scale
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  simA_coordsGauss = simA_preSimData[2]
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
    simA_Gauss  = getGaussPoint(params, simA_coordsGauss, params.simA_probe_1)
  elif params.simA_isDYNA:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_J.shape[0])
    simA_Gauss  = getGaussDYNA(simA_coordsGauss, params.simA_probe_1)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_J           = np.load(simB_J_fname)
    simB_tau         = np.load(simB_tau_fname)/params.scale
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
      simB_Gauss  = getGaussPoint(params, simB_coordsGauss, params.simB_probe_1)
    elif params.simB_isDYNA:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_J.shape[0])
      simB_Gauss  = getGaussDYNA(simB_coordsGauss, params.simB_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_J           = np.load(simC_J_fname)
    simC_tau         = np.load(simC_tau_fname)/params.scale
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
      simC_Gauss  = getGaussPoint(params, simC_coordsGauss, params.simC_probe_1)
    elif params.simC_isDYNA:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_J.shape[0])
      simC_Gauss  = getGaussDYNA(simC_coordsGauss, params.simC_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_J           = np.load(simD_J_fname)
    simD_tau         = np.load(simD_tau_fname)/params.scale
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
      simD_Gauss  = getGaussPoint(params, simD_coordsGauss, params.simD_probe_1)
    elif params.simD_isDYNA:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_J.shape[0])
      simD_Gauss  = getGaussDYNA(simD_coordsGauss, params.simD_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_J           = np.load(simE_J_fname)
    simE_tau         = np.load(simE_tau_fname)/params.scale
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
      simE_Gauss  = getGaussPoint(params, simE_coordsGauss, params.simE_probe_1)
    elif params.simE_isDYNA:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_J.shape[0])
      simE_Gauss  = getGaussDYNA(simE_coordsGauss, params.simE_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")

  print("Data loaded successfully.")
  print("\nGenerating plots...")

  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  #-------------------
  # Plot shear stress.
  #-------------------
  if params.simA_isPython:
    ax1.plot(simA_J[::params.simA_Skip,simA_Gauss[0], simA_Gauss[1]], simA_tau[::params.simA_Skip,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

  elif params.simA_isDYNA:
    ax1.plot(simA_J[::params.simA_Skip,simA_Gauss], simA_tau[::params.simA_Skip,simA_Gauss]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

  if params.simB_Dir is not None:
    if params.simB_isPython:
      ax1.plot(simB_J[::params.simB_Skip,simB_Gauss[0], simB_Gauss[1]], simB_tau[::params.simB_Skip,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)

    elif params.simB_isDYNA:
      ax1.plot(simB_J[::params.simB_Skip,simB_Gauss], simB_tau[::params.simB_Skip,simB_Gauss]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
    
  if params.simC_Dir is not None:
    if params.simC_isPython:
      ax1.plot(simC_J[::params.simC_Skip,simC_Gauss[0], simC_Gauss[1]], simC_tau[::params.simC_Skip,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    elif params.simC_isDYNA:
      ax1.plot(simC_J[::params.simC_Skip,simC_Gauss], simC_tau[::params.simC_Skip,simC_Gauss]*params.stressScaling, \
               params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, \
               label=r'$\tau/G(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

  if params.simD_Dir is not None:
    if params.simD_isPython:
      ax1.plot(simD_J[::params.simD_Skip,simD_Gauss[0], simD_Gauss[1]], simD_tau[::params.simD_Skip,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)

    elif params.simD_isDYNA:
      ax1.plot(simD_J[::params.simD_Skip,simD_Gauss], simD_tau[::params.simD_Skip,simD_Gauss]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
    
  if params.simE_Dir is not None:
    if params.simE_isPython:
      ax1.plot(simE_J[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]], simE_tau[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)

    elif params.simE_isDYNA:
      ax1.plot(simE_J[::params.simE_Skip,simE_Gauss], simE_tau[::params.simE_Skip,simE_Gauss]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\tau/G(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)

  if not params.no_labels:
    ax1.set_xlabel(r'$J$', fontsize=params.xAxisFontSize)
    ax1.set_ylabel(r'Normalized shear', fontsize=params.yAxisFontSize)

  if params.ylim0 is not None and params.ylim1 is not None:
    ax1.set_ylim([params.ylim0, params.ylim1])
  if params.xlim0 is not None and params.xlim1 is not None:
    ax1.set_xlim([params.xlim0, params.xlim1])

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

