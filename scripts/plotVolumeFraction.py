#--------------------------------------------------------------------------------------------------
# Plotting script for volume fraction(s) at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 24, 2024
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
#-----------
# Arguments:
#-----------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, params):
  #----------------
  # Perform checks.
  #----------------
  if not params.simA_isPython:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simD_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_ns_fname   = params.simA_Dir + 'ns.npy'
  simA_J_fname    = params.simA_Dir + 'J.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_J_fname) or not os.path.isfile(simA_ns_fname):
    sys.exit("-------\nERROR:\n-------\nJacobian or solid volume fraction data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_ns_fname   = params.simB_Dir + 'ns.npy'
    simB_J_fname    = params.simB_Dir + 'J.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_J_fname) or not os.path.isfile(simB_ns_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian or solid volume fraction data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_ns_fname   = params.simC_Dir + 'ns.npy'
    simC_J_fname    = params.simC_Dir + 'J.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_J_fname) or not os.path.isfile(simC_ns_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian or solid volume fraction data not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_ns_fname   = params.simD_Dir + 'ns.npy'
    simD_J_fname    = params.simD_Dir + 'J.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_J_fname) or not os.path.isfile(simD_ns_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian or solid volume fraction data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_ns_fname   = params.simE_Dir + 'ns.npy'
    simE_J_fname    = params.simE_Dir + 'J.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_J_fname) or not os.path.isfile(simE_ns_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian or solid volume fraction data not found for Simulation E.")

  print("\nLoading in data...")
  #----------------------------------------------------------------
  # Load in data and get closest Gauss points for requested probes.
  #----------------------------------------------------------------
  #
  # Simulation A
  #
  simA_J           = np.load(simA_J_fname)
  simA_nf          = 1 - np.load(simA_ns_fname)
  simA_tsolve      = np.load(simA_time_fname)
  simA_coordsGauss = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)[2]
  simA_Gauss       = getGaussPoint(params, simA_coordsGauss, params.simA_probe_1)
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_J           = np.load(simB_J_fname)
    simB_nf          = 1 - np.load(simB_ns_fname)
    simB_tsolve      = np.load(simB_time_fname)
    simB_coordsGauss = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)[2]
    simB_Gauss       = getGaussPoint(params, simB_coordsGauss, params.simB_probe_1)
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_J           = np.load(simC_J_fname)
    simC_nf          = 1 - np.load(simC_ns_fname)
    simC_tsolve      = np.load(simC_time_fname)
    simC_coordsGauss = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)[2]
    simC_Gauss       = getGaussPoint(params, simC_coordsGauss, params.simC_probe_1)
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_J           = np.load(simD_J_fname)
    simD_nf          = 1 - np.load(simD_ns_fname)
    simD_tsolve      = np.load(simD_time_fname)
    simD_coordsGauss = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)[2]
    simD_Gauss       = getGaussPoint(params, simD_coordsGauss, params.simD_probe_1)
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_J           = np.load(simE_J_fname)
    simE_nf          = 1 - np.load(simE_ns_fname)
    simE_tsolve      = np.load(simE_time_fname)
    simE_coordsGauss = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)[2]
    simE_Gauss       = getGaussPoint(params, simE_coordsGauss, params.simE_probe_1)

  print("Data loaded successfully.")
  
  print("\nGenerating plots...")

  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  #---------------
  # Plot porosity.
  #---------------
  ax1.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_nf[::params.simA_Skip,simA_Gauss[0], simA_Gauss[1]], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$n^\rf(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)
  
  if params.simB_Dir is not None:
    ax1.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_nf[::params.simB_Skip,simB_Gauss[0], simB_Gauss[1]], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, abel=r'$n^\rf(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
    
  if params.simC_Dir is not None:
    ax1.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_nf[::params.simC_Skip,simC_Gauss[0], simC_Gauss[1]], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$n^\rf(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)
    
  if params.simD_Dir is not None:
    ax1.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_nf[::params.simD_Skip,simD_Gauss[0], simD_Gauss[1]], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$n^\rf(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)

  if params.simE_Dir is not None:
    ax1.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_nf[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$n^\rf(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)  
  #---------------
  # Plot Jacobian.
  #---------------
  if params.jacobianPlot:
    ax2 = ax1.twinx()
    ax2.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_J[::params.simA_Skip,simA_Gauss[0], simA_Gauss[1]], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$J(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      ax2.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_J[::params.simB_Skip,simB_Gauss[0], simB_Gauss[1]], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$J(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      ax2.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_J[::params.simC_Skip,simC_Gauss[0], simC_Gauss[1]], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$J(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      ax2.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_J[::params.simD_Skip,simD_Gauss[0], simD_Gauss[1]], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$J(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      ax2.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_J[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$J(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)

  if not params.no_labels:
    ax1.set_xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    ax1.set_ylabel(r'Porosity $n^\rf$', fontsize=params.yAxisFontSize)
    if params.jacobianPlot:
      ax2.set_ylabel(r'$J$', fontsize=params.yAxisFontSize)

  if params.ylim0 is not None and params.ylim1 is not None:
    ax1.set_ylim([params.ylim0, params.ylim1])
    if params.jacobianPlot:
      ax2.set_ylim([0, 1])
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
    ax1.grid(True, which=params.gridWhich)
  if params.legend:
    if params.jacobianPlot:
      fig1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
    else:
      ax1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
  if params.title:
    fig1.suptitle(params.titleName,y=1,fontsize=16)

  if params.grid:
    ax1.grid(True, which=params.gridWhich)

  if params.legend:
    if params.jacobianPlot:
      fig1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
               handlelength=params.handleLength, fontsize=params.legendFontSize,\
               edgecolor='k', framealpha=1.0)
    else:
      ax1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
                 handlelength=params.handleLength, fontsize=params.legendFontSize,\
                 edgecolor='k', framealpha=1.0) 

  if params.title:
    fig1.suptitle(params.titleName,y=params.titleLoc,fontsize=params.titleFontSize)

  plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
  plt.close()
  
  return

