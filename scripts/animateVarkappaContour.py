#--------------------------------------------------------------------------------------------------
# Plotting script for animation of hydraulic conductivity contour(s).
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
  #----------------
  # Perform checks.
  #----------------
  if not params.simA_isPython:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHydraulic conductivity plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHydraulic conductivity plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHydraulic conductivity plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHydraulic conductivity plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nHydraulic conductivity plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_khat_fname = params.simA_Dir + 'khat.npy'
  simA_J_fname    = params.simA_Dir + 'J.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_khat_fname):
    sys.exit("------\nERROR:\n------\nHydraulic conductivity data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_khat_fname = params.simB_Dir + 'khat.npy'
    simB_J_fname    = params.simB_Dir + 'J.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_khat_fname):
      sys.exit("------\nERROR:\n------\nHydraulic conductivity data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_khat_fname = params.simC_Dir + 'khat.npy'
    simC_J_fname    = params.simC_Dir + 'J.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_khat_fname):
      sys.exit("------\nERROR:\n------\nHydraulic conductivity data not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_khat_fname = params.simD_Dir + 'khat.npy'
    simD_J_fname    = params.simD_Dir + 'J.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_khat_fname):
      sys.exit("------\nERROR:\n------\nHydraulic conductivity data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_khat_fname = params.simE_Dir + 'khat.npy'
    simE_J_fname    = params.simE_Dir + 'J.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_khat_fname):
      sys.exit("------\nERROR:\n------\nHydraulic conductivity data not found for Simulation E.")

  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  simA_coordsGauss = simA_preSimData[2]*params.dispScaling
  simA_khat        = np.load(simA_khat_fname)/simA_simParams.khat
  simA_J           = np.load(simA_J_fname)
  simA_tsolve      = np.load(simA_time_fname)
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    simB_khat        = np.load(simB_khat_fname)/simB_simParams.khat
    simB_J           = np.load(simB_J_fname)
    simB_tsolve      = np.load(simB_time_fname)
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    simC_khat        = np.load(simC_khat_fname)/simC_simParams.khat
    simC_J           = np.load(simC_J_fname)
    simC_tsolve      = np.load(simC_time_fname)
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    simD_khat        = np.load(simD_khat_fname)/simD_simParams.khat
    simD_J           = np.load(simD_J_fname)
    simD_tsolve      = np.load(simD_time_fname)
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    simE_khat        = np.load(simE_khat_fname)/simE_simParams.khat
    simE_J           = np.load(simE_J_fname)
    simE_tsolve      = np.load(simE_time_fname)

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    simA_khat = np.mean(simA_khat, axis=2)
    simA_J    = np.mean(simA_J, axis=2)
    if params.simB_Dir is not None:
      simB_khat = np.mean(simB_khat, axis=2)
      simB_J    = np.mean(simB_J, axis=2)
    if params.simC_Dir is not None:
      simC_khat = np.mean(simC_khat, axis=2)
      simC_J    = np.mean(simC_J, axis=2)
    if params.simD_Dir is not None:
      simD_khat = np.mean(simD_khat, axis=2)
      simD_J    = np.mean(simD_J, axis=2)
    if params.simE_Dir is not None:
      simE_khat = np.mean(simE_khat, axis=2)
      simE_J    = np.mean(simE_J, axis=2)
  else:
    if params.simA_isPython:
     simA_khat        = simA_khat.reshape(simA_khat.shape[0],int(simA_khat.shape[1]*simA_khat.shape[2]))
      simA_J           = simA_J.reshape(simA_J.shape[0],int(simA_J.shape[1]*simA_J.shape[2]))
      simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None and params.simB_isPython:
      simB_khat        = simB_khat.reshape(simB_khat.shape[0],int(simB_khat.shape[1]*simB_khat.shape[2]))
      simB_J           = simB_J.reshape(simB_J.shape[0],int(simB_J.shape[1]*simB_J.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None and params.simC_isPython:
      simC_khat        = simC_khat.reshape(simC_khat.shape[0],int(simC_khat.shape[1]*simC_khat.shape[2]))
      simC_J           = simC_J.reshape(simC_J.shape[0],int(simC_J.shape[1]*simC_J.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None and params.simD_isPython:
s      simD_khat        = simD_khat.reshape(simD_khat.shape[0],int(simD_khat.shape[1]*simD_khat.shape[2]))
      simD_J           = simD_J.reshape(simD_J.shape[0],int(simD_J.shape[1]*simD_J.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
 s   if params.simE_Dir is not None and params.simE_isPython:
      simE_khat        = simE_khat.reshape(simE_khat.shape[0],int(simE_khat.shape[1]*simE_khat.shape[2]))
      simE_J           = simE_J.reshape(simE_J.shape[0],int(simE_J.shape[1]*simE_J.shape[2]))
  s    simE_coordsGauss = simE_coordsGauss.flatten()
  #------------------------------------------------------------------------------
  # Check that every simulation has at least as many data points as Simulation A.
  #s------------------------------------------------------------------------------
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
      ax2 = ax1.twinx()
      #--------------------------------------
      # Plot relative hydraulic conductivity.
      #--------------------------------------
      ax1.plot(simA_coordsGauss, simA_khat[timeIndex,:], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\hat{k}/\hat{k}_0(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, simB_khat[timeIndex,:], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\hat{k}/\hat{k}_0(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        ax1.plot(simC_coordsGauss, simC_khat[timeIndex,:], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\hat{k}/\hat{k}_0(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        ax1.plot(simD_coordsGauss, simD_khat[timeIndex,:], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\hat{k}/\hat{k}_0(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        ax1.plot(simE_coordsGauss, simE_khat[timeIndex,:], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\hat{k}/\hat{k}_0(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #------------------------------
      # Plot Jacobian of deformation.
      #------------------------------
      if params.jacobianPlot:
        ax2.plot(simA_coordsGauss, simA_J[timeIndex,:], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

        if params.simB_Dir is not None:
          ax2.plot(simB_coordsGauss, simB_J[timeIndex,:], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          ax2.plot(simC_coordsGauss, simC_J[timeIndex,:], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          ax2.plot(simD_coordsGauss, simD_J[timeIndex,:], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          ax2.plot(simE_coordsGauss, simE_J[timeIndex,:], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$J(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #------------------------------------
      # Plot initial solid volume fraction.
      #------------------------------------
      if params.jacobianPlot:
        ax2.plot(np.linspace(0, simA_simParams.H0*params.dispScaling, simA_coordsGauss.shape[0]), np.ones(simA_coordsGauss.shape[0])*simA_simParams.ns_0, 'k--')
      else:
        ax1.plot(np.linspace(0, simA_simParams.H0*params.dispScaling, simA_coordsGauss.shape[0]), np.zeros(simA_coordsGauss.shape[0]), 'r--')

      if not params.no_labels:
        ax1.set_xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        ax1.set_ylabel(r'$\hat{k}/\hat{k}_0$', fontsize=params.yAxisFontSize)
        if params.jacobianPlot:
          ax2.set_ylabel(r'$J$', fontsize=params.yAxisFontSize)

      if params.jacobianPlot: 
        if params.ylim00 is not None and params.ylim10 is not None:
          ax1.set_ylim([params.ylim00, params.ylim10])
          ax2.set_ylim([params.ylim01, params.ylim11])
      else:
        if params.ylim0 is not None and params.xlim1 is not None:
          ax1.set_ylim([params.ylim0, params.ylim1])
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
        if params.jacobianPlot:
          lines, labels   = ax1.get_legend_handles_labels()
          lines2, labels2 = ax2.get_legend_handles_labels()
          plt.legend(lines + lines2, labels + labels2,\
                     bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
                     handlelength=params.handleLength, fontsize=params.legendFontSize,\
                     edgecolor='k', framealpha=1.0) 
        else:
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

