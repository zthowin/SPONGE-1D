#--------------------------------------------------------------------------------------------------
# Plotting script for animation of pressure contour(s).
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
# a_StressDict     (dictionary)  mappings for stress scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_DispDict, a_StressDict, params):
  #---------------------
  # Generate file names.
  #---------------------
  simA_press_fname = params.simA_Dir + 'press.npy'
  simA_ps_E_fname  = params.simA_Dir + 'ps_E.npy'
  simA_pf_fname    = params.simA_Dir + 'pf.npy'
  simA_time_fname  = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_press_fname):
    sys.exit("-------\nERROR:\n-------\nStress data file not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_press_fname = params.simB_Dir + 'press.npy'
    simB_ps_E_fname  = params.simB_Dir + 'ps_E.npy'
    simB_pf_fname    = params.simB_Dir + 'pf.npy'
    simB_time_fname  = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_press_fname):
      sys.exit("-------\nERROR:\n-------\nStress data file not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_press_fname = params.simC_Dir + 'press.npy'
    simC_ps_E_fname  = params.simC_Dir + 'ps_E.npy'
    simC_pf_fname    = params.simC_Dir + 'pf.npy'
    simC_time_fname  = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_press_fname):
      sys.exit("-------\nERROR:\n-------\nStress data file not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_press_fname = params.simD_Dir + 'press.npy'
    simD_ps_E_fname  = params.simD_Dir + 'ps_E.npy'
    simD_pf_fname    = params.simD_Dir + 'pf.npy'
    simD_time_fname  = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_press_fname):
      sys.exit("-------\nERROR:\n-------\nStress data file not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_press_fname = params.simE_Dir + 'press.npy'
    simE_ps_E_fname  = params.simE_Dir + 'ps_E.npy'
    simE_pf_fname    = params.simE_Dir + 'pf.npy'
    simE_time_fname  = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_press_fname):
      sys.exit("-------\nERROR:\n-------\nStress data file not found for Simulation E.")
  
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
  simA_press       = np.load(simA_press_fname)/params.scale
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
    simA_ps_E   = np.load(simA_ps_E_fname)/params.scale
    if 'pf' in simA_simParams.Physics:
      simA_pf   = (np.load(simA_pf_fname) + params.adjust/params.scale)/params.scale
  elif params.simA_isDYNA:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_press.shape[0])
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    simB_press       = np.load(simB_press_fname)/params.scale
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
      simB_ps_E   = np.load(simB_ps_E_fname)/params.scale
      if 'pf' in simB_simParams.Physics:
        simB_pf   = (np.load(simB_pf_fname) + params.adjust/params.scale)/params.scale
    elif params.simB_isDYNA:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_press.shape[0])
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    simC_press       = np.load(simC_press_fname)/params.scale
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
      simC_ps_E   = np.load(simC_ps_E_fname)/params.scale
      if 'pf' in simC_simParams.Physics:
        simC_pf   = (np.load(simC_pf_fname) + params.adjust/params.scale)/params.scale
    elif params.simC_isDYNA:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_press.shape[0])
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    simD_press       = np.load(simD_press_fname)/params.scale
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
      simD_ps_E   = np.load(simD_ps_E_fname)/params.scale
      if 'pf' in simD_simParams.Physics:
        simD_pf   = (np.load(simD_pf_fname) + params.adjust/params.scale)/params.scale
    elif params.simD_isDYNA:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_press.shape[0])
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    simE_press       = np.load(simE_press_fname)/params.scale
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
      simE_ps_E   = np.load(simE_ps_E_fname)/params.scale
      if 'pf' in simE_simParams.Physics:
        simE_pf   = (np.load(simE_pf_fname) + params.adjust/params.scale)/params.scale
    elif params.simE_isDYNA:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_press.shape[0] - 1)

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    if params.simA_isPython:
      simA_press = np.mean(simA_press, axis=2)
      simA_ps_E  = np.mean(simA_ps_E, axis=2)
      if 'pf' in simA_simParams.Physics:
        simA_pf  = np.mean(simA_pf, axis=2)
    if params.simB_Dir is not None and params.simB_isPython:
      simB_press = np.mean(simB_press, axis=2)
      simB_ps_E  = np.mean(simB_ps_E, axis=2)
      if 'pf' in simB_simParams.Physics:
        simB_pf  = np.mean(simB_pf, axis=2)
    if params.simC_Dir is not None and params.simC_isPython:
      simC_press = np.mean(simC_press, axis=2)
      simC_ps_E  = np.mean(simC_ps_E, axis=2)
      if 'pf' in simC_simParams.Physics:
        simC_pf  = np.mean(simC_pf, axis=2)
    if params.simD_Dir is not None and params.simD_isPython:
      simD_press = np.mean(simD_press, axis=2)
      simD_ps_E  = np.mean(simD_ps_E, axis=2)
      if 'pf' in simD_simParams.Physics:
        simD_pf  = np.mean(simD_pf, axis=2)
    if params.simE_Dir is not None and params.simE_isPython:
      simE_press = np.mean(simE_press, axis=2)
      simE_ps_E  = np.mean(simE_ps_E, axis=2)
      if 'pf' in simE_simParams.Physics:
        simE_pf  = np.mean(simE_pf, axis=2)
  else:
    if 'pf' in simA_simParams.Physics and params.simA_isPython:
      simA_pf        = simA_pf.reshape(simA_pf.shape[0],int(simA_pf.shape[1]*simA_pf.shape[2]))
    simA_ps_E        = simA_ps_E.reshape(simA_ps_E.shape[0],int(simA_ps_E.shape[1]*simA_ps_E.shape[2]))
    simA_press       = simA_press.reshape(simA_press.shape[0],int(simA_press.shape[1]*simA_press.shape[2]))
    simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None and params.simB_isPython:
      if 'pf' in simB_simParams.Physics:
        simB_pf        = simB_pf.reshape(simB_pf.shape[0],int(simB_pf.shape[1]*simB_pf.shape[2]))
      simB_ps_E        = simB_ps_E.reshape(simB_ps_E.shape[0],int(simB_ps_E.shape[1]*simB_ps_E.shape[2]))
      simB_press       = simB_press.reshape(simB_press.shape[0],int(simB_press.shape[1]*simB_press.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None and params.simC_isPython:
      if 'pf' in simC_simParams.Physics:
        simC_pf        = simC_pf.reshape(simC_pf.shape[0],int(simC_pf.shape[1]*simC_pf.shape[2]))
      simC_ps_E        = simC_ps_E.reshape(simC_ps_E.shape[0],int(simC_ps_E.shape[1]*simC_ps_E.shape[2]))
      simC_press       = simC_press.reshape(simC_press.shape[0],int(simC_press.shape[1]*simC_press.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None and params.simD_isPython:
      if 'pf' in simD_simParams.Physics:
        simD_pf        = simD_pf.reshape(simD_pf.shape[0],int(simD_pf.shape[1]*simD_pf.shape[2]))
      simD_ps_E        = simD_ps_E.reshape(simD_ps_E.shape[0],int(simD_ps_E.shape[1]*simD_ps_E.shape[2]))
      simD_press       = simD_press.reshape(simD_press.shape[0],int(simD_press.shape[1]*simD_press.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
    if params.simE_Dir is not None and params.simE_isPython:
      if 'pf' in simE_simParams.Physics:
        simE_pf        = simE_pf.reshape(simE_pf.shape[0],int(simE_pf.shape[1]*simE_pf.shape[2]))
      simE_ps_E        = simE_ps_E.reshape(simE_ps_E.shape[0],int(simE_ps_E.shape[1]*simE_ps_E.shape[2]))
      simE_press       = simE_press.reshape(simE_press.shape[0],int(simE_press.shape[1]*simE_press.shape[2]))
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
      #----------------
      # Total pressure.
      #----------------
      if params.totalPlot:
        plt.plot(simA_coordsGauss, simA_press[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$p(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)  

        if params.simB_Dir is not None:
          plt.plot(simB_coordsGauss, simB_press[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$p(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          plt.plot(simC_coordsGauss, simC_press[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$p(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          plt.plot(simD_coordsGauss, simD_press[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$p(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          plt.plot(simE_coordsGauss, simE_press[timeIndex,:]*params.stressScaling, \
                   params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, \
                   label=r'$p(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #----------------
      # Solid pressure.
      #----------------
      if params.solidPlot:
        if 'pf' in simA_simParams.Physics:
          plt.plot(simA_coordsGauss, simA_ps_E[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$p_E^\rs(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

        if params.simB_Dir is not None:
          if 'pf' in simB_simParams.Physics:
            plt.plot(simB_coordsGauss, simB_ps_E[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$p_E^\rs(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          if 'pf' in simC_simParams.Physics:
            plt.plot(simC_coordsGauss, simC_ps_E[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$p_E^\rs(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          if 'pf' in simD_simParams.Physics:
            plt.plot(simD_coordsGauss, simD_ps_E[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$p_E^\rs(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          if 'pf' in simE_simParams.Physics:
            plt.plot(simE_coordsGauss, simE_ps_E[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$p_E^\rs(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #---------------------
      # Pore fluid pressure.
      #---------------------
      if params.fluidPlot:
        if 'pf' in simA_simParams.Physics:
          plt.plot(simA_coordsGauss, simA_pf[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Charlie, color=params.simA_Color_Charlie, fillstyle=params.simA_fillstyle, label=r'$-p_\rf(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

        if params.simB_Dir is not None:
          if 'pf' in simB_simParams.Physics:
            plt.plot(simB_coordsGauss, simB_pf[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Charlie, color=params.simB_Color_Charlie, fillstyle=params.simB_fillstyle, label=r'$-p_\rf(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          if 'pf' in simC_simParams.Physics:
            plt.plot(simC_coordsGauss, simC_pf[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Charlie, color=params.simC_Color_Charlie, fillstyle=params.simC_fillstyle, label=r'$-p_\rf(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          if 'pf' in simD_simParams.Physics:
            plt.plot(simD_coordsGauss, simD_pf[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Charlie, color=params.simD_Color_Charlie, fillstyle=params.simD_fillstyle, label=r'$-p_\rf(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          if 'pf' in simE_simParams.Physics:
            plt.plot(simE_coordsGauss, simE_pf[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Charlie, color=params.simE_Color_Charlie, fillstyle=params.simE_fillstyle, label=r'$-p_\rf(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        if params.scale > 1:
          plt.ylabel(r'Normalized pressure', fontsize=params.yAxisFontSize)
        else:
          plt.ylabel(r'Pressure ' + a_StressDict[params.stressScaling], fontsize=params.yAxisFontSize)

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

