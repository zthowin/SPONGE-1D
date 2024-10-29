#--------------------------------------------------------------------------------------------------
# Plotting script for animation of mixture temperatures.
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
  sys.exit("MODULE WARNING. meshInfo not found, check consiguration.")

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
  simA_ts_fname    = params.simA_Dir + 'ts.npy'
  simA_tf_fname    = params.simA_Dir + 'tf.npy'
  simA_rhofr_fname = params.simA_Dir + 'rhofR.npy'
  simA_time_fname  = params.simA_Dir + 'time.npy'
  simA_ns_fname    = params.simA_Dir + 'ns.npy'
  if not os.path.isfile(simA_ts_fname):
    sys.exit("-------\nERROR:\n-------\nTemperature data file not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_ts_fname    = params.simB_Dir + 'ts.npy'
    simB_tf_fname    = params.simB_Dir + 'tf.npy'
    simB_rhofr_fname = params.simB_Dir + 'rhofR.npy'
    simB_time_fname  = params.simB_Dir + 'time.npy'
    simB_ns_fname    = params.simB_Dir + 'ns.npy'
    if not os.path.isfile(simB_ts_fname):
      sys.exit("-------\nERROR:\n-------\nTemperature data file not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_ts_fname    = params.simC_Dir + 'ts.npy'
    simC_tf_fname    = params.simC_Dir + 'tf.npy'
    simC_rhofr_fname = params.simC_Dir + 'rhofR.npy'
    simC_ns_fname    = params.simC_Dir + 'ns.npy'
    simC_time_fname  = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_ts_fname):
      sys.exit("-------\nERROR:\n-------\nTemperature data file not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_ts_fname    = params.simD_Dir + 'ts.npy'
    simD_tf_fname    = params.simD_Dir + 'tf.npy'
    simD_rhofr_fname = params.simD_Dir + 'rhofR.npy'
    simD_time_fname  = params.simD_Dir + 'time.npy'
    simD_ns_fname    = params.simD_Dir + 'ns.npy'
    if not os.path.isfile(simD_ts_fname):
      sys.exit("-------\nERROR:\n-------\nTemperature data file not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_ts_fname    = params.simE_Dir + 'ts.npy'
    simE_tf_fname    = params.simE_Dir + 'tf.npy'
    simE_rhofr_fname = params.simE_Dir + 'rhofR.npy'
    simE_ns_fname    = params.simE_Dir + 'ns.npy'
    simE_time_fname  = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_ts_fname):
      sys.exit("-------\nERROR:\n-------\nTemperature data file not found for Simulation E.")
  
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
  simA_ts          = np.load(simA_ts_fname)
  if params.simA_isPython:
    simA_tsolve    = np.load(simA_time_fname)
    if 'tf' in simA_simParams.Physics:
      simA_tf      = np.load(simA_tf_fname)
      simA_rhofr   = np.load(simA_rhofr_fname) 
      simA_ns      = np.load(simA_ns_fname)
  else:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_ts.shape[0] - 1)
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    simB_ts          = np.load(simB_ts_fname)
    if params.simB_isPython:
      simB_tsolve    = np.load(simB_time_fname)
      if 'tf' in simB_simParams.Physics:
        simB_tf      = np.load(simB_tf_fname)
        simB_rhofr   = np.load(simB_rhofr_fname) 
        simB_ns      = np.load(simB_ns_fname)
    else:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_ts.shape[0] - 1)
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    simC_ts          = np.load(simC_ts_fname)
    if params.simC_isPython:
      simC_tsolve    = np.load(simC_time_fname)
      if 'tf' in simC_simParams.Physics:
        simC_tf      = np.load(simC_tf_fname)
        simC_rhofr   = np.load(simC_rhofr_fname) 
        simC_ns      = np.load(simC_ns_fname)
    else:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_ts.shape[0] - 1)
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    simD_ts          = np.load(simD_ts_fname)
    if params.simD_isPython:
      simD_tsolve    = np.load(simD_time_fname)
      if 'tf' in simD_simParams.Physics:
        simD_tf      = np.load(simD_tf_fname)
        simD_rhofr   = np.load(simD_rhofr_fname) 
        simD_ns      = np.load(simD_ns_fname)
    else:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_ts.shape[0] - 1)
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    simE_ts          = np.load(simE_ts_fname)
    if params.simE_isPython:
      simE_tsolve    = np.load(simE_time_fname)
      if 'tf' in simE_simParams.Physics:
        simE_tf      = np.load(simE_tf_fname)
        simE_rhofr   = np.load(simE_rhofr_fname)
        simE_ns      = np.load(simE_ns_fname)
    else:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_ts.shape[0] - 1)

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    if params.simA_isPython:
      simA_ts = np.mean(simA_ts, axis=2)
      if 'tf' in simA_simParams.Physics:
        simA_tf    = np.mean(simA_tf, axis=2)
        simA_rhofr = np.mean(simA_rhofr, axis=2)
        simA_ns    = np.mean(simA_ns, axis=2)
    if params.simB_Dir is not None:
      if params.simB_isPython:
        simB_ts = np.mean(simB_ts, axis=2)
        if 'tf' in simB_simParams.Physics:
          simB_tf    = np.mean(simB_tf, axis=2)
          simB_rhofr = np.mean(simB_rhofr, axis=2)
          simB_ns    = np.mean(simB_ns, axis=2)
    if params.simC_Dir is not None:
      if params.simC_isPython:
        simC_ts = np.mean(simC_ts, axis=2)
        if 'tf' in simC_simParams.Physics:
          simC_tf    = np.mean(simC_tf, axis=2)
          simC_rhofr = np.mean(simC_rhofr, axis=2)
          simC_ns    = np.mean(simC_ns, axis=2)
    if params.simD_Dir is not None:
      if params.simD_isPython:
        simD_ts = np.mean(simD_ts, axis=2)
        if 'tf' in simD_simParams.Physics:
          simD_tf    = np.mean(simD_tf, axis=2)
          simD_rhofr = np.mean(simD_rhofr, axis=2)
          simD_ns    = np.mean(simD_ns, axis=2)
    if params.simE_Dir is not None:
      if params.simE_isPython:
        simE_ts = np.mean(simE_ts, axis=2)
        if 'tf' in simE_simParams.Physics:
          simE_tf    = np.mean(simE_tf, axis=2)
          simE_rhofr = np.mean(simE_rhofr, axis=2)
          simE_ns    = np.mean(simE_ns, axis=2)
  else:
    if params.simA_isPython:
      simA_ts          = simA_ts.reshape(simA_ts.shape[0],int(simA_ts.shape[1]*simA_ts.shape[2]))
      if 'tf' in simA_simParams.Physics:
        simA_tf        = simA_tf.reshape(simA_tf.shape[0],int(simA_tf.shape[1]*simA_tf.shape[2]))
        simA_rhofr     = simA_rhofr.reshape(simA_rhofr.shape[0],int(simA_rhofr.shape[1]*simA_rhofr.shape[2]))
        simA_ns        = simA_ns.reshape(simA_ns.shape[0],int(simA_ns.shape[1]*simA_ns.shape[2]))
      simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None and params.simB_isPython:
      simB_ts          = simB_ts.reshape(simB_ts.shape[0],int(simB_ts.shape[1]*simB_ts.shape[2]))
      if 'tf' in simB_simParams.Physics:
        simB_tf        = simB_tf.reshape(simB_tf.shape[0],int(simB_tf.shape[1]*simB_tf.shape[2]))
        simB_rhofr     = simB_rhofr.reshape(simB_rhofr.shape[0],int(simB_rhofr.shape[1]*simB_rhofr.shape[2]))
        simB_ns        = simB_ns.reshape(simB_ns.shape[0],int(simB_ns.shape[1]*simB_ns.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
    if params.simC_Dir is not None and params.simC_isPython:
      simC_ts          = simC_ts.reshape(simC_ts.shape[0],int(simC_ts.shape[1]*simC_ts.shape[2]))
      if 'tf' in simC_simParams.Physics:
        simC_tf        = simC_tf.reshape(simC_tf.shape[0],int(simC_tf.shape[1]*simC_tf.shape[2]))
        simC_rhofr     = simC_rhofr.reshape(simC_rhofr.shape[0],int(simC_rhofr.shape[1]*simC_rhofr.shape[2]))
        simC_ns        = simC_ns.reshape(simC_ns.shape[0],int(simC_ns.shape[1]*simC_ns.shape[2]))
      simC_coordsGauss = simC_coordsGauss.flatten()
    if params.simD_Dir is not None and params.simD_isPython:
      simD_ts          = simD_ts.reshape(simD_ts.shape[0],int(simD_ts.shape[1]*simD_ts.shape[2]))
      if 'tf' in simD_simParams.Physics:
        simD_tf        = simD_tf.reshape(simD_tf.shape[0],int(simD_tf.shape[1]*simD_tf.shape[2]))
        simD_rhofr     = simD_rhofr.reshape(simD_rhofr.shape[0],int(simD_rhofr.shape[1]*simD_rhofr.shape[2]))
        simD_ns        = simD_ns.reshape(simD_ns.shape[0],int(simD_ns.shape[1]*simD_ns.shape[2]))
      simD_coordsGauss = simD_coordsGauss.flatten()
    if params.simE_Dir is not None and params.simE_isPython:
      simE_ts          = simE_ts.reshape(simE_ts.shape[0],int(simE_ts.shape[1]*simE_ts.shape[2]))
      if 'tf' in simE_simParams.Physics:
        simE_tf        = simE_tf.reshape(simE_tf.shape[0],int(simE_tf.shape[1]*simE_tf.shape[2]))
        simE_rhofr     = simE_rhofr.reshape(simE_rhofr.shape[0],int(simE_rhofr.shape[1]*simE_rhofr.shape[2]))
        simE_ns        = simE_ns.reshape(simE_ns.shape[0],int(simE_ns.shape[1]*simE_ns.shape[2]))
      simE_coordsGauss = simE_coordsGauss.flatten()
  #--------------------------------------------
  # Calculate mixture temperature if necessary.
  #--------------------------------------------
  if 'tf' in simA_simParams.Physics:
    simA_rhos = simA_ns*simA_simParams.rhosR_0
    simA_rhof = (1 - simA_ns)*simA_rhofr
    simA_t    = (simA_rhos*simA_simParams.cvs*simA_ts + simA_rhof*simA_simParams.cvf*simA_tf)/\
                (simA_rhos*simA_simParams.cvs         + simA_rhof*simA_simParams.cvf)
  else:
    simA_t = simA_ts
  if params.simB_Dir is not None:
    if 'tf' in simB_simParams.Physics:
      simB_rhos = simB_ns*simB_simParams.rhosR_0
      simB_rhof = (1 - simB_ns)*simB_rhofr
      simB_t    = (simB_rhos*simB_simParams.cvs*simB_ts + simB_rhof*simB_simParams.cvf*simB_tf)/\
                  (simB_rhos*simB_simParams.cvs         + simB_rhof*simB_simParams.cvf)
    else:
      simB_t = simB_ts
  if params.simC_Dir is not None:
    if 'tf' in simC_simParams.Physics:
      simC_rhos = simC_ns*simC_simParams.rhosR_0
      simC_rhof = (1 - simC_ns)*simC_rhofr
      simC_t    = (simC_rhos*simC_simParams.cvs*simC_ts + simC_rhof*simC_simParams.cvf*simC_tf)/\
                  (simC_rhos*simC_simParams.cvs         + simC_rhof*simC_simParams.cvf)
    else:
      simC_t = simC_ts
  if params.simD_Dir is not None:
    if 'tf' in simD_simParams.Physics:
      simD_rhos = simD_ns*simD_simParams.rhosR_0
      simD_rhof = (1 - simD_ns)*simD_rhofr
      simD_t    = (simD_rhos*simD_simParams.cvs*simD_ts + simD_rhof*simD_simParams.cvf*simD_tf)/\
                  (simD_rhos*simD_simParams.cvs         + simD_rhof*simD_simParams.cvf)
    else:
      simD_t = simD_ts
  if params.simE_Dir is not None:
    if 'tf' in simE_simParams.Physics:
      simE_rhos = simE_ns*simE_simParams.rhosR_0
      simE_rhof = (1 - simE_ns)*simE_rhofr
      simE_t    = (simE_rhos*simE_simParams.cvs*simE_ts + simE_rhof*simE_simParams.cvf*simE_tf)/\
                  (simE_rhos*simE_simParams.cvs         + simE_rhof*simE_simParams.cvf)
    else:
      simE_t = simE_ts
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
      #-------------------
      # Total temperature.
      #-------------------
      plt.plot(simA_coordsGauss, simA_t[timeIndex,:], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\theta(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

      if params.simB_Dir is not None:
        plt.plot(simB_coordsGauss, simB_t[timeIndex,:], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\theta(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        plt.plot(simC_coordsGauss, simC_t[timeIndex,:], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\theta(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        plt.plot(simD_coordsGauss, simD_t[timeIndex,:], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\theta(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        plt.plot(simE_coordsGauss, simE_t[timeIndex,:], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\theta(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        plt.ylabel(r'Temperature (K)', fontsize=params.yAxisFontSize)

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
      print("\nDeleting individual frames...")
      for f in glob.glob(params.outputDir + params.filename + "*.png"):
        os.remove(f)
      print("Finished deleting individual frames.")

  return

