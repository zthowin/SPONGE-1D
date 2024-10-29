#--------------------------------------------------------------------------------------------------
# Plotting script for animation of Jacobian of deformation contour(s).
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 26, 2024
#--------------------------------------------------------------------------------------------------
import sys, os, subprocess, glob, traceback

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
# a_DispDict       (dictionary)  mappings for displacement scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_DispDict, params):
  #---------------------
  # Generate file names.
  #---------------------
  simA_J_fname    = params.simA_Dir + 'J.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_J_fname):
    sys.exit("-------\nERROR:\n-------\nJacobian data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_J_fname    = params.simB_Dir + 'J.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_J_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian data not found for Simulation B.")  

  if params.simC_Dir is not None:
    simC_J_fname    = params.simC_Dir + 'J.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_J_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian data not found for Simulation C.")  

  if params.simD_Dir is not None:
    simD_J_fname    = params.simD_Dir + 'J.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_J_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_J_fname    = params.simE_Dir + 'J.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_J_fname):
      sys.exit("-------\nERROR:\n-------\nJacobian data not found for Simulation E.")
  
  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_J           = np.load(simA_J_fname)
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  simA_coordsGauss = simA_preSimData[2]*params.dispScaling
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
  elif params.simA_isDYNA:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_J.shape[0])
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_J           = np.load(simB_J_fname)
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
    elif params.simB_isDYNA:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_J.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_J           = np.load(simC_J_fname)
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
    elif params.simC_isDYNA:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_J.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_J           = np.load(simD_J_fname)
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
    elif params.simD_isDYNA:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_J.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_J           = np.load(simE_J_fname)
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
    elif params.simE_isDYNA:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_J.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    if params.simA_isPython:
      simA_J = np.mean(simA_J, axis=2)
    if params.simB_Dir is not None and params.simB_isPython:
      simB_J = np.mean(simB_J, axis=2)
    if params.simC_Dir is not None and params.simC_isPython:
      simC_J = np.mean(simC_J, axis=2)
    if params.simD_Dir is not None and params.simD_isPython:
      simD_J = np.mean(simD_J, axis=2)
    if params.simE_Dir is not None and params.simE_isPython:
      simE_J = np.mean(simE_J, axis=2)
  else:
    if params.simA_isPython:
      simA_J           = simA_J.reshape(simA_J.shape[0],int(simA_J.shape[1]*simA_J.shape[2]))
      simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None and params.simB_isPython:
      simB_J           = simB_J.reshape(simB_J.shape[0],int(simB_J.shape[1]*simB_J.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None and params.simC_isPython:
      simC_J           = simC_J.reshape(simC_J.shape[0],int(simC_J.shape[1]*simC_J.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None and params.simD_isPython:
      simD_J           = simD_J.reshape(simD_J.shape[0],int(simD_J.shape[1]*simD_J.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
    if params.simE_Dir is not None and params.simE_isPython:
      simE_J           = simE_J.reshape(simE_J.shape[0],int(simE_J.shape[1]*simE_J.shape[2]))
      simE_coordsGauss = simE_coordsGauss.flatten()
  #------------------------------------------------------------------------------
  # Check that every simulation has at least as many data points as Simulation A.
  #------------------------------------------------------------------------------
  if params.simB_Dir is not None:
    if simB_tsolve.shape[0] != simA_tsolve.shape[0]:
      print("--------\nWARNING:\n--------\nSimulation B number of data points do not match Simulation A number of data points.")
  if params.simC_Dir is not None:
    if simC_tsolve.shape[0] != simA_tsolve.shape[0]:
      print("--------\nWARNING:\n--------\nSimulation C number of data points do not match Simulation A number of data points.")
  if params.simD_Dir is not None:
    if simD_tsolve.shape[0] != simA_tsolve.shape[0]:
      print("--------\nWARNING:\n--------\nSimulation D number of data points do not match Simulation A number of data points.")
  if params.simE_Dir is not None:
    if simE_tsolve.shape[0] != simA_tsolve.shape[0]:
      print("--------\nWARNING:\n--------\nSimulation E number of data points do not match Simulation A number of data points.")
  #-----------------
  # Generate frames.
  #-----------------
  print("\nGenerating plots...")

  fig = plt.figure(1)

  try:
    for timeIndex in range(params.startID, params.stopID, params.simA_Skip):
      print("Generating .png #%i" %int(timeIndex/params.simA_Skip))
      ax1 = plt.subplot(111)

      plt.plot(simA_coordsGauss, simA_J[timeIndex,:], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
      if params.simB_Dir is not None:
        plt.plot(simB_coordsGauss, simB_J[timeIndex,:], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        plt.plot(simC_coordsGauss, simC_J[timeIndex,:], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        plt.plot(simD_coordsGauss, simD_J[timeIndex,:], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        plt.plot(simE_coordsGauss, simE_J[timeIndex,:], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        plt.ylabel(r'$J$', fontsize=params.yAxisFontSize)
      
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
        ax1.set_title(params.titleName,y=params.titleLoc,fontsize=params.titleFontSize)

      plt.savefig(params.outputDir + params.filename + '-%d.png' %int(timeIndex/params.simA_Skip), bbox_inches='tight', dpi=params.DPI)
      plt.close()

  except IndexError:
    print("Encountered index error, check size of input files to ensure matching data times.")
    print(traceback.format_exc())
    pass

  if not params.noAnimation:
    #----------
    # Make .mp4
    #----------
    print("\nGenerating .mp4...")
    mp4_cmd = ["ffmpeg", "-y", "-framerate", str(params.framerate), "-i", params.outputDir + params.filename + "-%d.png", params.outputDir + params.filename + ".mp4"]
    try:
      subprocess.run(mp4_cmd, check=True, capture_output=True)
      print("Finished generating .mp4.")
    except subprocess.CalledProcessError:
      print("\nERROR. .mp4 not generated.")
    #----------
    # Make .gif
    #----------
    print("\nGenerating .gif...")
    palette_cmd = ["ffmpeg", "-y", "-i", params.outputDir + params.filename + "-%d.png", "-vf", "palettegen", params.outputDir + "palette.png"]
    gif_cmd     = ["ffmpeg", "-y", "-framerate", str(params.framerate), "-i", params.outputDir + params.filename + "-%d.png", "-i", params.outputDir + "palette.png", "-lavfi", "paletteuse", "-loop", "0", params.outputDir + params.filename + ".gif"]
    try:
      subprocess.run(palette_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
      print("\nERROR. .gif palette not generated.")
      print("Try adjusting legend position; interior legends perform better.\n\n")
      sys.exit()
    try:
      subprocess.run(gif_cmd, check=True, capture_output=True)
      print("Finished generating .gif.")
    except subprocess.CalledProcessError:
      print("\nERROR. .gif not generated.")
      print("Try fixing y-axis limits; set y-axis limits perform better.\n\n")
      sys.exit()
      
    if params.isDeletePNGs:  
      print("\nDeleting individual frames...")
      for f in glob.glob(params.outputDir + params.filename + "*.png"):
        os.remove(f)
      print("Finished deleting individual frames.")

  return

