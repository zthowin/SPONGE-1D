#--------------------------------------------------------------------------------------------------
# Plotting script for animation of velocity contour(s).
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
def main(a_TimeDict, a_DispDict, params):
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
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_Vsolve     = np.load(simA_vel_fname)
  simA_preSimData = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams  = simA_preSimData[0]
  simA_coordsD    = simA_preSimData[3]
  simA_coordsD    = simA_coordsD.flatten()
  _, idx          = np.unique(simA_coordsD, return_index=True)
  simA_coordsD    = simA_coordsD[np.sort(idx)]

  simA_Start = 0
  simA_Stop  = simA_simParams.nNodeS
  simA_Skip  = 1

  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
  else:
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_Dsolve.shape[0])

  if simA_simParams.Element_Type.split('-')[0] == 'Q3H':
    simA_Stop  -= 1
    simA_Skip  += 1
  if params.fluidPlot and 'uf' in simA_simParams.Physics:
    simA_coordsDF   = simA_preSimData[5]
    simA_coordsDF   = simA_coordsDF.flatten()
    _, idx          = np.unique(simA_coordsDF, return_index=True)
    simA_coordsDF   = simA_coordsDF[np.sort(idx)]
    simA_FluidStart = simA_simParams.nNodeS
    simA_FluidStop  = simA_FluidStart + simA_simParams.nNodeF
    simA_FluidSkip  = 1
    if simA_simParams.Element_Type.split('-')[1] == 'Q3H':
      simA_FluidStop  -= 1
      simA_FluidSkip  += 1
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_Vsolve     = np.load(simB_vel_fname)
    simB_preSimData = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams  = simB_preSimData[0]
    simB_coordsD    = simB_preSimData[3]
    simB_coordsD    = simB_coordsD.flatten()
    _, idx          = np.unique(simB_coordsD, return_index=True)
    simB_coordsD    = simB_coordsD[np.sort(idx)]

    simB_Start = 0
    simB_Stop  = simB_simParams.nNodeS
    simB_Skip  = 1

    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
    else:
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_Dsolve.shape[0])
      simB_Skip   = 4

    if simB_simParams.Element_Type.split('-')[0] == 'Q3H':
      simB_Stop  -= 1
      simB_Skip  += 1
    if params.fluidPlot and 'uf' in simB_simParams.Physics:
      simB_coordsDF   = simB_preSimData[5]
      simB_coordsDF   = simB_coordsDF.flatten()
      _, idx          = np.unique(simB_coordsDF, return_index=True)
      simB_coordsDF   = simB_coordsDF[np.sort(idx)]
      simB_FluidStart = simB_simParams.nNodeS
      simB_FluidStop  = simB_FluidStart + simB_simParams.nNodeF
      simB_FluidSkip  = 1
      if simB_simParams.Element_Type.split('-')[1] == 'Q3H':
        simB_FluidStop  -= 1
        simB_FluidSkip  += 1
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_Vsolve     = np.load(simC_vel_fname)
    simC_preSimData = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams  = simC_preSimData[0]
    simC_coordsD    = simC_preSimData[3]
    simC_coordsD    = simC_coordsD.flatten()
    _, idx          = np.unique(simC_coordsD, return_index=True)
    simC_coordsD    = simC_coordsD[np.sort(idx)]

    simC_Start = 0
    simC_Stop  = simC_simParams.nNodeS
    simC_Skip  = 1

    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
    else:
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_Dsolve.shape[0])
      simC_Skip   = 4

    if simC_simParams.Element_Type.split('-')[0] == 'Q3H':
      simC_Stop  -= 1
      simC_Skip  += 1
    if params.fluidPlot and 'uf' in simC_simParams.Physics:
      simC_coordsDF   = simC_preSimData[5]
      simC_coordsDF   = simC_coordsDF.flatten()
      _, idx          = np.unique(simC_coordsDF, return_index=True)
      simC_coordsDF   = simC_coordsDF[np.sort(idx)]
      simC_FluidStart = simC_simParams.nNodeS
      simC_FluidStop  = simC_FluidStart + simC_simParams.nNodeF
      simC_FluidSkip  = 1
      if simC_simParams.Element_Type.split('-')[1] == 'Q3H':
        simC_FluidStop  -= 1
        simC_FluidSkip  += 1
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_Vsolve     = np.load(simD_vel_fname)
    simD_preSimData = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams  = simD_preSimData[0]
    simD_coordsD    = simD_preSimData[3]
    simD_coordsD    = simD_coordsD.flatten()
    _, idx          = np.unique(simD_coordsD, return_index=True)
    simD_coordsD    = simD_coordsD[np.sort(idx)]

    simD_Start = 0
    simD_Stop  = simD_simParams.nNodeS
    simD_Skip  = 1

    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
    else:
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_Dsolve.shape[0])
      simD_Skip   = 4

    if simD_simParams.Element_Type.split('-')[0] == 'Q3H':
      simD_Stop  -= 1
      simD_Skip  += 1
    if params.fluidPlot and 'uf' in simD_simParams.Physics:
      simD_coordsDF   = simD_preSimData[5]
      simD_coordsDF   = simD_coordsDF.flatten()
      _, idx          = np.unique(simD_coordsDF, return_index=True)
      simD_coordsDF   = simD_coordsDF[np.sort(idx)]
      simD_FluidStart = simD_simParams.nNodeS
      simD_FluidStop  = simD_FluidStart + simD_simParams.nNodeF
      simD_FluidSkip  = 1
      if simD_simParams.Element_Type.split('-')[1] == 'Q3H':
        simD_FluidStop  -= 1
        simD_FluidSkip  += 1
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_Vsolve     = np.load(simE_vel_fname)
    simE_preSimData = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams  = simE_preSimData[0]
    simE_coordsD    = simE_preSimData[3]
    simE_coordsD    = simE_coordsD.flatten()
    _, idx          = np.unique(simE_coordsD, return_index=True)
    simE_coordsD    = simE_coordsD[np.sort(idx)]

    simE_Start = 0
    simE_Stop  = simE_simParams.nNodeS
    simE_Skip  = 1

    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
    else:
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_Dsolve.shape[0])
      simE_Skip   = 4

    if simE_simParams.Element_Type.split('-')[0] == 'Q3H':
      simE_Stop  -= 1
      simE_Skip  += 1
    if params.fluidPlot and 'uf' in simE_simParams.Physics:
      simE_coordsDF   = simE_preSimData[5]
      simE_coordsDF   = simE_coordsDF.flatten()
      _, idx          = np.unique(simE_coordsDF, return_index=True)
      simE_coordsDF   = simE_coordsDF[np.sort(idx)]
      simE_FluidStart = simE_simParams.nNodeS
      simE_FluidStop  = simE_FluidStart + simE_simParams.nNodeF
      simE_FluidSkip  = 1
      if simE_simParams.Element_Type.split('-')[1] == 'Q3H':
        simE_FluidStop  -= 1
        simE_FluidSkip  += 1

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
      #---------------------
      # Plot solid velocity.
      #---------------------
      if params.solidPlot:
        plt.plot(simA_coordsD*params.dispScaling, simA_Vsolve[timeIndex,simA_Start:simA_Stop:simA_Skip], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$v(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

        if params.simB_Dir is not None:
          plt.plot(simB_coordsD*params.dispScaling, simB_Vsolve[timeIndex,simB_Start:simB_Stop:simB_Skip], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$v(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          plt.plot(simC_coordsD*params.dispScaling, simC_Vsolve[timeIndex,simC_Start:simC_Stop:simC_Skip], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$v(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          plt.plot(simD_coordsD*params.dispScaling, simD_Vsolve[timeIndex,simD_Start:simD_Stop:simD_Skip], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$v(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          plt.plot(simE_coordsD*params.dispScaling, simE_Vsolve[timeIndex,simE_Start:simE_Stop:simE_Skip], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$v(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #--------------------------
      # Plot pore fluid velocity.
      #--------------------------             
      if params.fluidPlot:
        if params.simA_isPython and 'uf' in simA_simParams.Physics:
          plt.plot(simA_coordsDF*params.dispScaling, simA_Vsolve[timeIndex,simA_FluidStart:simA_FluidStop:simA_FluidSkip], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

        if params.simB_Dir is not None and 'uf' in simB_simParams.Physics:
          if params.simB_isPython:
            plt.plot(simB_coordsDF*params.dispScaling, simB_Vsolve[timeIndex,simB_FluidStart:simB_FluidStop:simB_FluidSkip], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None and 'uf' in simC_simParams.Physics:
          if params.simC_isPython:
            plt.plot(simC_coordsDF*params.dispScaling, simC_Vsolve[timeIndex,simC_FluidStart:simC_FluidStop:simC_FluidSkip], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None and 'uf' in simD_simParams.Physics:
          if params.simD_isPython:
            plt.plot(simD_coordsDF*params.dispScaling, simD_Vsolve[timeIndex,simD_FluidStart:simD_FluidStop:simD_FluidSkip], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None and 'uf' in simE_simParams.Physics:
          if params.simE_isPython:
            plt.plot(simE_coordsDF*params.dispScaling, simE_Vsolve[timeIndex,simE_FluidStart:simE_FluidStop:simE_FluidSkip], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      
      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        plt.ylabel(r'Velocity (m/s)', fontsize=params.yAxisFontSize)

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
      for f in glob.glob(params.outputDir + params.filename + "*.png"):
        os.remove(f)

  return

