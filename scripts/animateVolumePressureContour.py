#--------------------------------------------------------------------------------------------------
# Plotting script for animation of volume contour(s) vs. pore fluid pressure contour(s).
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
  sys.exit("MODULE WARNING. preSimData not found, check configuration.")

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
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPorosity plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVolume-Temperature plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_rhofr_fname = params.simA_Dir + 'rhofR.npy'
  simA_pf_fname    = params.simA_Dir + 'pf.npy'
  simA_time_fname  = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_rhofr_fname):
    sys.exit("-------\nERROR:\n-------\nDensity data file not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_rhofr_fname = params.simB_Dir + 'rhofR.npy'
    simB_pf_fname    = params.simB_Dir + 'pf.npy'
    simB_time_fname  = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_rhofr_fname):
      sys.exit("-------\nERROR:\n-------\nDensity data file not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_rhofr_fname = params.simC_Dir + 'rhofR.npy'
    simC_pf_fname    = params.simC_Dir + 'pf.npy'
    simC_time_fname  = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_rhofr_fname):
      sys.exit("-------\nERROR:\n-------\nDensity data file not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_rhofr_fname = params.simD_Dir + 'rhofR.npy'
    simD_pf_fname    = params.simD_Dir + 'pf.npy'
    simD_time_fname  = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_rhofr_fname):
      sys.exit("-------\nERROR:\n-------\nDensity data file not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_rhofr_fname = params.simE_Dir + 'rhofR.npy'
    simE_pf_fname    = params.simE_Dir + 'pf.npy'
    simE_time_fname  = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_rhofr_fname):
      sys.exit("-------\nERROR:\n-------\nDensity data file not found for Simulation E.")
  
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
  simA_tsolve      = np.load(simA_time_fname)
  simA_pf          = (-np.load(simA_pf_fname) + simA_simParams.p_f0)/simA_simParams.p_f0
  simA_rhofr       = np.load(simA_rhofr_fname)/simA_simParams.rhofR_0 
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    simB_tsolve      = np.load(simB_time_fname)
    simB_pf          = (-np.load(simB_pf_fname) + simB_simParams.p_f0)/simB_simParams.p_f0
    simB_rhofr       = np.load(simB_rhofr_fname)/simB_simParams.rhofR_0 
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    simC_tsolve      = np.load(simC_time_fname)
    simC_pf          = (-np.load(simC_pf_fname) + simC_simParams.p_f0)/simC_simParams.p_f0
    simC_rhofr       = np.load(simC_rhofr_fname)/simC_simParams.rhofR_0 
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    simD_tsolve      = np.load(simD_time_fname)
    simD_pf          = (-np.load(simD_pf_fname) + simD_simParams.p_f0)/simD_simParams.p_f0
    simD_rhofr       = np.load(simD_rhofr_fname)/simD_simParams.rhofR_0 
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    simE_tsolve      = np.load(simE_time_fname)
    simE_pf          = (-np.load(simE_pf_fname) + simE_simParams.p_f0)/simE_simParams.p_f0
    simE_rhofr       = np.load(simE_rhofr_fname)/simE_simParams.rhofR_0 

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    simA_pf    = np.mean(simA_pf, axis=2)
    simA_rhofr = np.mean(simA_rhofr, axis=2)
    if params.simB_Dir is not None:
      simB_pf    = np.mean(simB_pf, axis=2)
      simB_rhofr = np.mean(simB_rhofr, axis=2)
    if params.simC_Dir is not None:
      simC_pf    = np.mean(simC_pf, axis=2)
      simC_rhofr = np.mean(simC_rhofr, axis=2)
    if params.simD_Dir is not None:
      simD_pf    = np.mean(simD_pf, axis=2)
      simD_rhofr = np.mean(simD_rhofr, axis=2)
    if params.simE_Dir is not None:
      simE_pf    = np.mean(simE_pf, axis=2)
      simE_rhofr = np.mean(simE_rhofr, axis=2)
  else:
    simA_pf          = simA_pf.reshape(simA_pf.shape[0],int(simA_pf.shape[1]*simA_pf.shape[2]))
    simA_rhofr       = simA_rhofr.reshape(simA_rhofr.shape[0],int(simA_rhofr.shape[1]*simA_rhofr.shape[2]))
    simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None:
      simB_pf          = simB_pf.reshape(simB_pf.shape[0],int(simB_pf.shape[1]*simB_pf.shape[2]))
      simB_rhofr       = simB_rhofr.reshape(simB_rhofr.shape[0],int(simB_rhofr.shape[1]*simB_rhofr.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None:
      simC_pf          = simC_pf.reshape(simC_pf.shape[0],int(simC_pf.shape[1]*simC_pf.shape[2]))
      simC_rhofr       = simC_rhofr.reshape(simC_rhofr.shape[0],int(simC_rhofr.shape[1]*simC_rhofr.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None:
      simD_pf          = simD_pf.reshape(simD_pf.shape[0],int(simD_pf.shape[1]*simD_pf.shape[2]))
      simD_rhofr       = simD_rhofr.reshape(simD_rhofr.shape[0],int(simD_rhofr.shape[1]*simD_rhofr.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
    if params.simE_Dir is not None:
      simE_pf          = simE_pf.reshape(simE_pf.shape[0],int(simE_pf.shape[1]*simE_pf.shape[2]))
      simE_rhofr       = simE_rhofr.reshape(simE_rhofr.shape[0],int(simE_rhofr.shape[1]*simE_rhofr.shape[2]))
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
      ax2 = ax1.twinx()
      #-------------------
      # Pore fluid volume.
      #-------------------
      ax1.plot(simA_coordsGauss, 1/simA_rhofr[timeIndex,:], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$v^{\rm{fR}}/v_0^{\rm{fR}}(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, 1/simB_rhofr[timeIndex,:], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$v^{\rm{fR}}/v_0^{\rm{fR}}(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        ax1.plot(simC_coordsGauss, 1/simC_rhofr[timeIndex,:], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$v^{\rm{fR}}/v_0^{\rm{fR}}(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        ax1.plot(simD_coordsGauss, 1/simD_rhofr[timeIndex,:], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$v^{\rm{fR}}/v_0^{\rm{fR}}(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        ax1.plot(simE_coordsGauss, 1/simE_rhofr[timeIndex,:], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$v^{\rm{fR}}/v_0^{\rm{fR}}(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #------------------------
      # Pore fluid temperature.
      #------------------------
      ax2.plot(simA_coordsGauss, simA_pf[timeIndex,:], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$p_\rf/p_{\rf,0}(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

      if params.simB_Dir is not None:
        ax2.plot(simB_coordsGauss, simB_pf[timeIndex,:], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$p_\rf/p_{\rf,0}(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        ax2.plot(simC_coordsGauss, simC_pf[timeIndex,:], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$p_\rf/p_{\rf,0}(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        ax2.plot(simD_coordsGauss, simD_pf[timeIndex,:], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$p_\rf/p_{\rf,0}(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        ax2.plot(simE_coordsGauss, simE_pf[timeIndex,:], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$p_\rf/p_{\rf,0}(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        ax1.set_xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        ax1.set_ylabel(r'Normalized specific volume', fontsize=params.yAxisFontSize)
        ax2.set_ylabel(r'Normalized pressure', fontsize=params.yAxisFontSize)

      if params.ylim00 is not None and params.ylim10 is not None:
        ax1.set_ylim([params.ylim00, params.ylim10])
        ax2.set_ylim([params.ylim01, params.ylim11])
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
        lines, labels   = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2,\
                   bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition,\
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

