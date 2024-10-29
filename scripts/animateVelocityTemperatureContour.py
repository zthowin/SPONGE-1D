#--------------------------------------------------------------------------------------------------
# Plotting script for animation of fluid velocity contour(s) vs. fluid temperature contour(s).
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
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_DispDict, params):
  #----------------
  # Perform checks.
  #----------------
  if not params.simA_isPython:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity-Temperature plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity-Temperature plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity-Temperature plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity-Temperature plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nVelocity-Temperature plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_vel_fname  = params.simA_Dir + 'velocity.npy'
  simA_dis_fname  = params.simA_Dir + 'displacement.npy'
  simA_time_fname = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_vel_fname):
    sys.exit("-------\nERROR:\n-------\nVelocity data file not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_vel_fname  = params.simB_Dir + 'velocity.npy'
    simB_dis_fname  = params.simB_Dir + 'displacement.npy'
    simB_time_fname = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_vel_fname):
      sys.exit("-------\nERROR:\n-------\nVelocity data file not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_vel_fname  = params.simC_Dir + 'velocity.npy'
    simC_dis_fname   = params.simC_Dir + 'displacement.npy'
    simC_time_fname = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_vel_fname):
      sys.exit("-------\nERROR:\n-------\nVelocity data file not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_vel_fname  = params.simD_Dir + 'velocity.npy'
    simD_dis_fname  = params.simD_Dir + 'displacement.npy'
    simD_time_fname = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_vel_fname):
      sys.exit("-------\nERROR:\n-------\nVelocity data file not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_vel_fname  = params.simE_Dir + 'velocity.npy'
    simE_dis_fname  = params.simE_Dir + 'displacement.npy'
    simE_time_fname = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_vel_fname):
      sys.exit("-------\nERROR:\n-------\nVelocity data file not found for Simulation E.")
  
  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  simA_coordsTf = simA_preSimData[2]*params.dispScaling
  simA_tsolve      = np.load(simA_time_fname)
  if 'pf' in simA_simParams.Physics:
    if 'uf' in simA_simParams.Physics:
      if 'tf' in simA_simParams.Physics:
        simA_coordsTf   = simA_preSimData[7]
      simA_Vsolve     = np.load(simA_vel_fname)
      simA_coordsDF   = simA_preSimData[5]
      simA_coordsDF   = simA_coordsDF.flatten()
      _, idx          = np.unique(simA_coordsDF, return_index=True)
      simA_coordsDF   = simA_coordsDF[np.sort(idx)]*params.dispScaling
      simA_FluidStart = simA_simParams.nNodeS
      simA_FluidStop  = simA_FluidStart + simA_simParams.nNodeF
      simA_FluidSkip  = 1
      if simA_simParams.Element_Type.split('-')[1] == 'Q3H':
        simA_FluidStop  -= 1
        simA_FluidSkip  += 1
    else:
      simA_coordsTf = simA_preSimData[6]
    simA_coordsTf = simA_coordsTf.flatten()
    _, idx        = np.unique(simA_coordsTf, return_index=True)
    simA_coordsTf = simA_coordsTf[np.sort(idx)]*params.dispScaling
    simA_Tsolve     = np.load(simA_dis_fname)[:,simA_simParams.nNodeS + simA_simParams.nNodeF + simA_simParams.nNodeP:]
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsTf = simB_preSimData[2]*params.dispScaling
    simB_tsolve      = np.load(simB_time_fname)
    if 'pf' in simB_simParams.Physics:
      if 'uf' in simB_simParams.Physics:
        if 'tf' in simB_simParams.Physics:
          simB_coordsTf   = simB_preSimData[7]
        simB_Vsolve     = np.load(simB_vel_fname)
        simB_coordsDF   = simB_preSimData[5]
        simB_coordsDF   = simB_coordsDF.flatten()
        _, idx          = np.unique(simB_coordsDF, return_index=True)
        simB_coordsDF   = simB_coordsDF[np.sort(idx)]*params.dispScaling
        simB_FluidStart = simB_simParams.nNodeS
        simB_FluidStop  = simB_FluidStart + simB_simParams.nNodeF
        simB_FluidSkip  = 1
        if simB_simParams.Element_Type.split('-')[1] == 'Q3H':
          simB_FluidStop  -= 1
          simB_FluidSkip  += 1
      else:
        simB_coordsTf = simB_preSimData[6]
      simB_coordsTf = simB_coordsTf.flatten()
      _, idx        = np.unique(simB_coordsTf, return_index=True)
      simB_coordsTf = simB_coordsTf[np.sort(idx)]*params.dispScaling
      simB_Tsolve   = np.load(simB_dis_fname)[:,simB_simParams.nNodeS + simB_simParams.nNodeF + simB_simParams.nNodeP:]
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsTf = simC_preSimData[2]*params.dispScaling
    simC_tsolve      = np.load(simC_time_fname)
    if 'pf' in simC_simParams.Physics:
      if 'uf' in simC_simParams.Physics:
        simC_coordsTf   = simC_preSimData[7]
        simC_Vsolve     = np.load(simC_vel_fname)
        simC_coordsDF   = simC_preSimData[5]
        simC_coordsDF   = simC_coordsDF.flatten()
        _, idx          = np.unique(simC_coordsDF, return_index=True)
        simC_coordsDF   = simC_coordsDF[np.sort(idx)]*params.dispScaling
        simC_FluidStart = simC_simParams.nNodeS
        simC_FluidStop  = simC_FluidStart + simC_simParams.nNodeF
        simC_FluidSkip  = 1
        if simC_simParams.Element_Type.split('-')[1] == 'Q3H':
          simC_FluidStop  -= 1
          simC_FluidSkip  += 1
      else:
        simC_coordsTf = simC_preSimData[6]
      simC_coordsTf = simC_coordsTf.flatten()
      _, idx        = np.unique(simC_coordsTf, return_index=True)
      simC_coordsTf = simC_coordsTf[np.sort(idx)]*params.dispScaling
      simC_Tsolve   = np.load(simC_dis_fname)[:,simC_simParams.nNodeS + simC_simParams.nNodeF + simC_simParams.nNodeP:]
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsTf = simD_preSimData[2]*params.dispScaling
    simD_tsolve      = np.load(simD_time_fname)
    if 'pf' in simD_simParams.Physics:
      if 'uf' in simD_simParams.Physics:
        simD_coordsTf   = simD_preSimData[7]
        simD_Vsolve     = np.load(simD_vel_fname)
        simD_coordsDF   = simD_preSimData[5]
        simD_coordsDF   = simD_coordsDF.flatten()
        _, idx          = np.unique(simD_coordsDF, return_index=True)
        simD_coordsDF   = simD_coordsDF[np.sort(idx)]*params.dispScaling
        simD_FluidStart = simD_simParams.nNodeS
        simD_FluidStop  = simD_FluidStart + simD_simParams.nNodeF
        simD_FluidSkip  = 1
        if simD_simParams.Element_Type.split('-')[1] == 'Q3H':
          simD_FluidStop  -= 1
          simD_FluidSkip  += 1
      else:
        simD_coordsTf = simD_preSimData[6]
      simD_coordsTf = simD_coordsTf.flatten()
      _, idx        = np.unique(simD_coordsTf, return_index=True)
      simD_coordsTf = simD_coordsTf[np.sort(idx)]*params.dispScaling
      simD_Tsolve   = np.load(simD_dis_fname)[:,simD_simParams.nNodeS + simD_simParams.nNodeF + simD_simParams.nNodeP:]
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsTf = simE_preSimData[2]*params.dispScaling
    simE_tsolve      = np.load(simE_time_fname)
    if 'pf' in simE_simParams.Physics:
      if 'uf' in simE_simParams.Physics:
        simE_coordsTf   = simE_preSimData[7]
        simE_Vsolve     = np.load(simE_vel_fname)
        simE_coordsDF   = simE_preSimData[5]
        simE_coordsDF   = simE_coordsDF.flatten()
        _, idx          = np.unique(simE_coordsDF, return_index=True)
        simE_coordsDF   = simE_coordsDF[np.sort(idx)]*params.dispScaling
        simE_FluidStart = simE_simParams.nNodeS
        simE_FluidStop  = simE_FluidStart + simE_simParams.nNodeF
        simE_FluidSkip  = 1
        if simE_simParams.Element_Type.split('-')[1] == 'Q3H':
          simE_FluidStop  -= 1
          simE_FluidSkip  += 1
      else:
        simE_coordsTf = simE_preSimData[6]
      simE_coordsTf = simE_coordsTf.flatten()
      _, idx        = np.unique(simE_coordsTf, return_index=True)
      simE_coordsTf = simE_coordsTf[np.sort(idx)]*params.dispScaling
      simE_Tsolve   = np.load(simE_dis_fname)[:,simE_simParams.nNodeS + simE_simParams.nNodeF + simE_simParams.nNodeP:]

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
      ax2 = ax1.twinx()
      #---------------------
      # Pore fluid velocity.
      #---------------------
      if 'uf' in simA_simParams.Physics:
          ax1.plot(simA_coordsDF, simA_Vsolve[timeIndex,simA_FluidStart:simA_FluidStop:simA_FluidSkip], params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

      if params.simB_Dir is not None:
        if 'uf' in simB_simParams.Physics:
            ax1.plot(simB_coordsDF, simB_Vsolve[timeIndex,simB_FluidStart:simB_FluidStop:simB_FluidSkip], params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None:
        if 'uf' in simC_simParams.Physics:
            ax1.plot(simC_coordsDF, simC_Vsolve[timeIndex,simC_FluidStart:simC_FluidStop:simC_FluidSkip], params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None:
        if 'uf' in simD_simParams.Physics:
            ax1.plot(simD_coordsDF, simD_Vsolve[timeIndex,simD_FluidStart:simD_FluidStop:simD_FluidSkip], params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None:
        if 'uf' in simE_simParams.Physics:
            ax1.plot(simE_coordsDF, simE_Vsolve[timeIndex,simE_FluidStart:simE_FluidStop:simE_FluidSkip], params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$v_\rf(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #------------------------
      # Pore fluid temperature.
      #------------------------
      if 'tf' in simA_simParams.Physics:
        ax2.plot(simA_coordsTf, simA_Tsolve[timeIndex,simA_simParams.nNodeTs:], params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, abel=r'$\theta^\rf(X, t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)

      if params.simB_Dir is not None and 'tf' in simB_simParams.Physics:
        ax2.plot(simB_coordsTf, simB_Tsolve[timeIndex,simB_simParams.nNodeTs:], params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$\theta^\rf(X, t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if params.simC_Dir is not None and 'tf' in simC_simParams.Physics:
        ax2.plot(simC_coordsTf, simC_Tsolve[timeIndex,simC_simParams.nNodeTs:], params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$\theta^\rf(X, t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

      if params.simD_Dir is not None and 'tf' in simD_simParams.Physics:
        ax2.plot(simD_coordsTf, simD_Tsolve[timeIndex,simD_simParams.nNodeTs:], params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$\theta^\rf(X, t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

      if params.simE_Dir is not None and 'tf' in simE_simParams.Physics:
        ax2.plot(simE_coordsTf, simE_Tsolve[timeIndex,simE_simParams.nNodeTs:], params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$\theta^\rf(X, t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        ax1.set_xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        ax1.set_ylabel(r'Velocity (m/s)', fontsize=params.yAxisFontSize)
        ax2.set_ylabel(r'Temperature (K)', fontsize=params.yAxisFontSize)

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

