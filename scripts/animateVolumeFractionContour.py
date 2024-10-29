#--------------------------------------------------------------------------------------------------
# Plotting script for animations of volume fraction (currently just pore fluid) contour(s).
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
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_DispDict, params):
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
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_ns_fname):
    sys.exit("------\nERROR:\n------\nSolid volume fraction data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_ns_fname   = params.simB_Dir + 'ns.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_ns_fname):
      sys.exit("------\nERROR:\n------\nSolid volume fraction data not found for Simulation B.") 

  if params.simC_Dir is not None:
    simC_ns_fname   = params.simC_Dir + 'ns.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_ns_fname):
      sys.exit("------\nERROR:\n------\nSolid volume fraction data not found for Simulation C.") 

  if params.simD_Dir is not None:
    simD_ns_fname   = params.simD_Dir + 'ns.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_ns_fname):
      sys.exit("------\nERROR:\n------\nSolid volume fraction data not found for Simulation D.") 

  if params.simE_Dir is not None:
    simE_ns_fname   = params.simE_Dir + 'ns.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_ns_fname):
      sys.exit("------\nERROR:\n------\nSolid volume fraction data not found for Simulation E.") 

  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_ns          = np.load(simA_ns_fname)
  simA_tsolve      = np.load(simA_time_fname)
  simA_coordsGauss = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)[2]*params.dispScaling
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_ns          = np.load(simB_ns_fname)
    simB_tsolve      = np.load(simB_time_fname)
    simB_coordsGauss = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)[2]*params.dispScaling
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_ns          = np.load(simC_ns_fname)
    simC_tsolve      = np.load(simC_time_fname)
    simC_coordsGauss = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)[2]*params.dispScaling
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_ns          = np.load(simD_ns_fname)
    simD_tsolve      = np.load(simD_time_fname)
    simD_coordsGauss = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)[2]*params.dispScaling
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_ns          = np.load(simE_ns_fname)
    simE_tsolve      = np.load(simE_time_fname)
    simE_coordsGauss = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)[2]*params.dispScaling

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    simA_ns = np.mean(simA_ns, axis=2)
    if params.simB_Dir is not None:
      simB_ns = np.mean(simB_ns, axis=2)
    if params.simC_Dir is not None:
      simC_ns = np.mean(simC_ns, axis=2)
    if params.simD_Dir is not None:
      simD_ns = np.mean(simD_ns, axis=2)
    if params.simE_Dir is not None:
      simE_ns = np.mean(simE_ns, axis=2)
  else:
    simA_ns          = simA_ns.reshape(simA_ns.shape[0],int(simA_ns.shape[1]*simA_ns.shape[2]))
    simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None:
      simB_ns          = simB_ns.reshape(simB_ns.shape[0],int(simB_ns.shape[1]*simB_ns.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None:
      simC_ns          = simC_ns.reshape(simC_ns.shape[0],int(simC_ns.shape[1]*simC_ns.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None:
      simD_ns          = simD_ns.reshape(simD_ns.shape[0],int(simD_ns.shape[1]*simD_ns.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
    if params.simE_Dir is not None:
      simE_ns          = simE_ns.reshape(simE_ns.shape[0],int(simE_ns.shape[1]*simE_ns.shape[2]))
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
  print("Generating plots...")

  fig = plt.figure(1)
  
  try:
    for timeIndex in range(params.startID, params.stopID, params.simA_Skip):
      print("Generating .png #%i" %int(timeIndex/params.simA_Skip))
      ax1 = plt.subplot(111)

      plt.plot(simA_coordsGauss, (1 - simA_ns[timeIndex,:])/params.scale, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$n^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
      if params.simB_Dir is not None:
        plt.plot(simB_coordsGauss, (1 - simB_ns[timeIndex,:])/params.scale, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$n^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        plt.plot(simC_coordsGauss, (1 - simC_ns[timeIndex,:])/params.scale, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$n^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        plt.plot(simD_coordsGauss, (1 - simD_ns[timeIndex,:])/params.scale, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$n^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        plt.plot(simE_coordsGauss, (1 - simE_ns[timeIndex,:])/params.scale, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$n^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        if params.scale != 1:
          plt.ylabel(r'Normalized porosity $n^\rf/n^\rf_0$', fontsize=params.yAxisFontSize)
        else:
          plt.ylabel(r'Porosity $n^\rf$', fontsize=params.yAxisFontSize)

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
      ax1.ticklabel_format(useOffset=False, style='plain')
      
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

