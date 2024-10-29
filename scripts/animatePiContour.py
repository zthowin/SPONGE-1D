#--------------------------------------------------------------------------------------------------
# Plotting script for animation of pore fluid pressure contour(s).
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
  #----------------
  # Perform checks.
  #----------------
  if not params.simA_isPython:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simD_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nPore fluid pressure plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_disp_fname = params.simA_Dir + 'displacement.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_disp_fname):
    sys.exit("------\nERROR:\n------\nPressure data not found for Simulation A.") 

  if params.simB_Dir is not None:
    simB_disp_fname = params.simB_Dir + 'displacement.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_disp_fname):
      sys.exit("------\nERROR:\n------\nPressure data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_disp_fname = params.simC_Dir + 'displacement.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_disp_fname):
      sys.exit("------\nERROR:\n------\nPressure data not found for Simulation C.") 

  if params.simD_Dir is not None:
    simD_disp_fname = params.simD_Dir + 'displacement.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_disp_fname):
      sys.exit("------\nERROR:\n------\nPressure data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_disp_fname = params.simE_Dir + 'displacement.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_disp_fname):
      sys.exit("------\nERROR:\n------\nPressure data not found for Simulation E.")

  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_preSimData = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams  = simA_preSimData[0]
  simA_coordsP    = simA_preSimData[4]
  simA_coordsP    = simA_coordsP.flatten()
  _, idx          = np.unique(simA_coordsP, return_index=True)
  simA_coordsP    = simA_coordsP[np.sort(idx)]*params.dispScaling
  simA_Psolve     = np.load(simA_disp_fname)[:,simA_simParams.nNodeS + simA_simParams.nNodeF:simA_simParams.nNodeS + simA_simParams.nNodeF + simA_simParams.nNodeP] - params.adjust
  simA_tsolve     = np.load(simA_time_fname)
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams  = simB_preSimData[0]
    simB_coordsP    = simB_preSimData[4]
    simB_coordsP    = simB_coordsP.flatten()
    _, idx          = np.unique(simB_coordsP, return_index=True)
    simB_coordsP    = simB_coordsP[np.sort(idx)]*params.dispScaling
    simB_Psolve     = np.load(simB_disp_fname)[:,simB_simParams.nNodeS + simB_simParams.nNodeF:simB_simParams.nNodeS + simB_simParams.nNodeF + simB_simParams.nNodeP] - params.adjust
    simB_tsolve     = np.load(simB_time_fname)
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams  = simC_preSimData[0]
    simC_coordsP    = simC_preSimData[4]
    simC_coordsP    = simC_coordsP.flatten()
    _, idx          = np.unique(simC_coordsP, return_index=True)
    simC_coordsP    = simC_coordsP[np.sort(idx)]*params.dispScaling
    simC_Psolve     = np.load(simC_disp_fname)[:,simC_simParams.nNodeS + simC_simParams.nNodeF:simC_simParams.nNodeS + simC_simParams.nNodeF + simC_simParams.nNodeP] - params.adjust
    simC_tsolve     = np.load(simC_time_fname)
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams  = simD_preSimData[0]
    simD_coordsP    = simD_preSimData[4]
    simD_coordsP    = simD_coordsP.flatten()
    _, idx          = np.unique(simD_coordsP, return_index=True)
    simD_coordsP    = simD_coordsP[np.sort(idx)]*params.dispScaling
    simD_Psolve     = np.load(simD_disp_fname)[:,simD_simParams.nNodeS + simD_simParams.nNodeF:simD_simParams.nNodeS + simD_simParams.nNodeF + simD_simParams.nNodeP] - params.adjust
    simD_tsolve     = np.load(simD_time_fname)
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams  = simE_preSimData[0]
    simE_coordsP    = simE_preSimData[4]
    simE_coordsP    = simE_coordsP.flatten()
    _, idx          = np.unique(simE_coordsP, return_index=True)
    simE_coordsP    = simE_coordsP[np.sort(idx)]*params.dispScaling
    simE_Psolve     = np.load(simE_disp_fname)[:,simE_simParams.nNodeS + simE_simParams.nNodeF:simE_simParams.nNodeS + simE_simParams.nNodeF + simE_simParams.nNodeP] - params.adjust
    simE_tsolve     = np.load(simE_time_fname)

  print("Data loaded successfully.")
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

      plt.plot(simA_coordsP, simA_Psolve[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$p_\rf(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
      if params.simB_Dir is not None:
        plt.plot(simB_coordsP, simB_Psolve[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$p_\rf(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        plt.plot(simC_coordsP, simC_Psolve[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$p_\rf(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        plt.plot(simD_coordsP, simD_Psolve[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$p_\rf(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        plt.plot(simE_coordsP, simE_Psolve[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$p_\rf(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      
      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        if params.adjust > 0:
          plt.ylabel(r'Overpressure ' + a_StressDict[params.stressScaling], fontsize=params.yAxisFontSize)
        else:
          plt.ylabel(r'Pore fluid pressure ' + a_StressDict[params.stressScaling], fontsize=params.yAxisFontSize)

      if params.ylim0 is not None and params.ylim1 is not None:
        plt.ylim([params.ylim0, params.ylim1])
      if params.xlim0 is not None and params.xlim1 is not None:
        plt.xlim([params.xlim0, params.xlim1])

      plt.ticklabel_format(useOffset=False,style='plain')

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
      for f in glob.glob(params.outputDir + params.filename + "*.png"):
        os.remove(f)

  return

