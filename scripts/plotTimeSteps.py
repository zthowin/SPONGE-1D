#--------------------------------------------------------------------------------------------------
# Plotting script for time steps for multiple simulations.
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
  import matplotlib.ticker as mticker
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

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
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime step plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime step plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime step plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simD_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime step plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nTime step plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_dt_fname  = params.simA_Dir + 'dt.npy'
  simA_tdt_fname = params.simA_Dir + 'tdt.npy'
  if not os.path.isfile(simA_dt_fname) or not os.path.isfile(simA_tdt_fname):
    sys.exit("-------\nERROR:\n-------\nAdaptive time step data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_dt_fname  = params.simB_Dir + 'dt.npy'
    simB_tdt_fname = params.simB_Dir + 'tdt.npy'
    if not os.path.isfile(simB_dt_fname) or not os.path.isfile(simB_tdt_fname):
      sys.exit("-------\nERROR:\n-------\nAdaptive time step data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_dt_fname  = params.simC_Dir + 'dt.npy'
    simC_tdt_fname = params.simC_Dir + 'tdt.npy'
    if not os.path.isfile(simC_dt_fname) or not os.path.isfile(simC_tdt_fname):
      sys.exit("-------\nERROR:\n-------\nAdaptive time step data not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_dt_fname  = params.simD_Dir + 'dt.npy'
    simD_tdt_fname = params.simD_Dir + 'tdt.npy'
    if not os.path.isfile(simD_dt_fname) or not os.path.isfile(simD_tdt_fname):
      sys.exit("-------\nERROR:\n-------\nAdaptive time step data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_dt_fname  = params.simE_Dir + 'dt.npy'
    simE_tdt_fname = params.simE_Dir + 'tdt.npy'
    if not os.path.isfile(simE_dt_fname) or not os.path.isfile(simE_tdt_fname):
      sys.exit("-------\nERROR:\n-------\nAdaptive time step data not found for Simulation E.")
  #--------------
  # Load in data.
  #--------------
  print("\nLoading in data...")

  simA_dtsolve  = np.load(simA_dt_fname)
  simA_tdtsolve = np.load(simA_tdt_fname)

  if params.simB_Dir is not None:
    simB_dtsolve  = np.load(simB_dt_fname)
    simB_tdtsolve = np.load(simB_tdt_fname)

  if params.simC_Dir is not None:
    simC_dtsolve  = np.load(simC_dt_fname)
    simC_tdtsolve = np.load(simC_tdt_fname)

  if params.simD_Dir is not None:
    simD_dtsolve  = np.load(simD_dt_fname)
    simD_tdtsolve = np.load(simD_tdt_fname)

  if params.simE_Dir is not None:
    simE_dtsolve  = np.load(simE_dt_fname)
    simE_tdtsolve = np.load(simE_tdt_fname)

  print("Data loaded successfully.")
  print("\nGenerating plots...")
  #--------------------
  # Plot time steps.
  #--------------------
  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  
  plt.semilogy(simA_tdtsolve[::params.simA_Skip]*params.timeScaling, simA_dtsolve[::params.simA_Skip], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=params.simA_Title)

  if params.simB_Dir is not None:
    plt.semilogy(simB_tdtsolve[::params.simB_Skip]*params.timeScaling, simB_dtsolve[::params.simB_Skip], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=params.simB_Title)
    
  if params.simC_Dir is not None:
    plt.semilogy(simC_tdtsolve[::params.simC_Skip]*params.timeScaling, simC_dtsolve[::params.simC_Skip], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=params.simC_Title)

  if params.simD_Dir is not None:
    plt.semilogy(simD_tdtsolve[::params.simD_Skip]*params.timeScaling, simD_dtsolve[::params.simD_Skip], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=params.simD_Title)
    
  if params.simE_Dir is not None:
    plt.semilogy(simE_tdtsolve[::params.simE_Skip]*params.timeScaling, simE_dtsolve[::params.simE_Skip], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=params.simE_Title)  

  if not params.no_labels:
    plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    plt.ylabel(r'$\Delta t$ (s)', fontsize=params.yAxisFontSize)

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
    ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9)))

  if params.legend:
    plt.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
               handlelength=params.handleLength, fontsize=params.legendFontSize,\
               edgecolor='k', framealpha=1.0) 

  if params.title:
    fig1.suptitle(params.titleName,y=params.titleLoc,fontsize=params.titleFontSize)

  plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
  plt.close()
  
  return

