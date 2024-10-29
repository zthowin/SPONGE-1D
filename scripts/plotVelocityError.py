#--------------------------------------------------------------------------------------------------
# Plotting script for differences in solid and fluid velocities at single depth(s) for multiple 
# simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 24, 2024
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
  #----------------
  # Perform checks.
  #----------------
  if not params.simA_isPython:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity difference plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity difference plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity difference plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity difference plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity difference plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_vel_fname  = params.simA_Dir + 'velocity.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_vel_fname):
    sys.exit("------\nERROR:\n------\nVelocity data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_vel_fname  = params.simB_Dir + 'velocity.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_vel_fname):
      sys.exit("------\nERROR:\n------\nVelocity data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_vel_fname  = params.simC_Dir + 'velocity.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_vel_fname):
      sys.exit("------\nERROR:\n------\nVelocity data not found for Simulation C.") 

  if params.simD_Dir is not None:
    simD_vel_fname  = params.simD_Dir + 'velocity.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_vel_fname):
      sys.exit("------\nERROR:\n------\nVelocity data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_vel_fname  = params.simE_Dir + 'velocity.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_vel_fname):
      sys.exit("------\nERROR:\n------\nVelocity data not found for Simulation E.")

  print("\nLoading in data...")
  #---------------------------------------------------------
  # Load in data and get closest nodes for requested probes.
  #---------------------------------------------------------
  #
  # Simulation A
  #
  simA_Vsolve     = np.load(simA_vel_fname)
  simA_preSimData = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams  = simA_preSimData[0]
  simA_coordsD    = simA_preSimData[3]
  simA_tsolve     = np.load(simA_time_fname)
  simA_LM         = simA_preSimData[1]
  simA_DDOF       = getDisplacementDOF(params, simA_LM, simA_coordsD, params.simA_probe_1)
  if 'uf' in simA_simParams.Physics:
    simA_coordsDF = simA_preSimData[5]
    simA_DFDOF    = getFluidDOF(params, simA_simParams, simA_LM, simA_coordsDF, params.simA_probe_1)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity differences only enabled for (u-uf-pf) and (u-uf-pf-ts-tf) formulations.")
  if params.simA_probe_1 % simA_simParams.H0e > 1e-12:
    signA = r'\approx'
  else:
    signA = r'='
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_Vsolve     = np.load(simB_vel_fname)
    simB_preSimData = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams  = simB_preSimData[0]
    simB_coordsD    = simB_preSimData[3]
    simB_tsolve     = np.load(simB_time_fname)
    simB_LM         = simB_preSimData[1]
    simB_DDOF       = getDisplacementDOF(params, simB_LM, simB_coordsD, params.simB_probe_1)
    if 'uf' in simB_simParams.Physics:
      simB_coordsDF = simB_preSimData[5]
      simB_DFDOF    = getFluidDOF(params, simB_simParams, simB_LM, simB_coordsDF, params.simB_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity differences only enabled for (u-uf-pf) and (u-uf-pf-ts-tf) formulations.")
    if params.simB_probe_1 % simB_simParams.H0e > 1e-12:
      signB = r'\approx'
    else:
      signB = r'='
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_Vsolve     = np.load(simC_vel_fname)
    simC_preSimData = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams  = simC_preSimData[0]
    simC_coordsD    = simC_preSimData[3]
    simC_tsolve     = np.load(simC_time_fname)
    simC_LM         = simC_preSimData[1]
    simC_DDOF       = getDisplacementDOF(params, simC_LM, simC_coordsD, params.simC_probe_1)
    if 'uf' in simC_simParams.Physics:
      simC_coordsDF = simC_preSimData[5]
      simC_DFDOF    = getFluidDOF(params, simC_simParams, simC_LM, simC_coordsDF, params.simC_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity differences only enabled for (u-uf-pf) and (u-uf-pf-ts-tf) formulations.")
    if params.simC_probe_1 % simC_simParams.H0e > 1e-12:
      signC = r'\approx'
    else:
      signC = r'='
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_Vsolve     = np.load(simD_vel_fname)
    simD_preSimData = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams  = simD_preSimData[0]
    simD_coordsD    = simD_preSimData[3]
    simD_tsolve     = np.load(simD_time_fname)
    simD_LM         = simD_preSimData[1]
    simD_DDOF       = getDisplacementDOF(params, simD_LM, simD_coordsD, params.simD_probe_1)
    if 'uf' in simD_simParams.Physics:
      simD_coordsDF = simD_preSimData[5]
      simD_DFDOF    = getFluidDOF(params, simD_simParams, simD_LM, simD_coordsDF, params.simD_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity differences only enabled for (u-uf-pf) and (u-uf-pf-ts-tf) formulations.")
    if params.simD_probe_1 % simD_simParams.H0e > 1e-12:
      signD = r'\approx'
    else:
      signD = r'='
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_Vsolve     = np.load(simE_vel_fname)
    simE_preSimData = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams  = simE_preSimData[0]
    simE_coordsD    = simE_preSimData[3]
    simE_tsolve     = np.load(simE_time_fname)
    simE_LM         = simE_preSimData[1]
    simE_DDOF       = getDisplacementDOF(params, simE_LM, simE_coordsD, params.simE_probe_1)
    if 'uf' in simE_simParams.Physics:
      simE_coordsDF = simE_preSimData[5]
      simE_DFDOF    = getFluidDOF(params, simE_simParams, simE_LM, simE_coordsDF, params.simE_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity differences only enabled for (u-uf-pf) and (u-uf-pf-ts-tf) formulations.")
    if params.simE_probe_1 % simE_simParams.H0e > 1e-12:
      signE = r'\approx'
    else:
      signE = r'='

  print("Data loaded successfully.")

  print("\nGenerating plots...")

  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  #------------------
  # Plot differences.
  #------------------
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, np.abs(simA_Vsolve[::params.simA_Skip,simA_DDOF] - simA_Vsolve[::params.simA_Skip,simA_DFDOF]), params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\lvert v(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t) - v_\rf(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)\rvert$, '  + params.simA_Title)

  if params.simB_Dir is not None:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, np.abs(simB_Vsolve[::params.simB_Skip,simB_DDOF] - simB_Vsolve[::params.simB_Skip,simB_DFDOF]), params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\lvert v(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t) - v_\rf(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)\rvert$, '  + params.simB_Title)
      
  if params.simC_Dir is not None:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, np.abs(simC_Vsolve[::params.simC_Skip,simC_DDOF] - simC_Vsolve[::params.simC_Skip,simC_DFDOF]), params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\lvert v(X' + signC + str(params.simC_probe_1) + r'\text{m}' + r',t) - v_\rf(X' + signC + str(params.simC_probe_1) + r'\text{m}' + r',t)\rvert$, '  + params.simC_Title)

  if params.simD_Dir is not None:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, np.abs(simD_Vsolve[::params.simD_Skip,simD_DDOF] - simD_Vsolve[::params.simD_Skip,simD_DFDOF]), \
             params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, \
             label=r'$\lvert v(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t) - v_\rf(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)\rvert$, '  + params.simD_Title)
      
  if params.simE_Dir is not None:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, np.abs(simE_Vsolve[::params.simE_Skip,simE_DDOF] - simE_Vsolve[::params.simE_Skip,simE_DFDOF]), params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\lvert v(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t) - v_\rf(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r'(,t)\rvert$, '  + params.simE_Title)

  if not params.no_labels:
    plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    plt.ylabel(r'Absolute velocity difference (m/s)', fontsize=params.yAxisFontSize)
  
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

