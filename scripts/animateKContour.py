#--------------------------------------------------------------------------------------------------
# Plotting script for animation of the terms contributing to (u-pf-ts-tf) pore fluid energy balance.
#
# Currently restricted to two simulations based on number of lines in the figure being too
# overwhelming for more than two simulations.
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
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nK plotting only enabled for Python code.")
  if params.simB_Dir is not None:
    if not params.simB_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nK plotting only enabled for Python code.")
  if params.simC_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nK plotting only enabled for Python code.")
  if params.simD_Dir is not None:
    if not params.simC_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nK plotting only enabled for Python code.")
  if params.simE_Dir is not None:
    if not params.simE_isPython:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nK plotting only enabled for Python code.")
  #---------------------
  # Generate file names.
  #---------------------
  simA_pf_fname     = params.simA_Dir + 'pf.npy'
  simA_tf_fname     = params.simA_Dir + 'tf.npy'
  simA_ts_fname     = params.simA_Dir + 'ts.npy'
  simA_ns_fname     = params.simA_Dir + 'ns.npy'
  simA_J_fname      = params.simA_Dir + 'J.npy'
  simA_JDot_fname   = params.simA_Dir + 'JDot.npy'
  simA_vDarcy_fname = params.simA_Dir + 'vDarcy.npy'
  simA_pfDot_fname  = params.simA_Dir + 'pfDot.npy'
  simA_dp_fdX_fname = params.simA_Dir + 'dpfdX.npy'
  simA_qf_fname     = params.simA_Dir + 'qf.npy'
  simA_dnfdX_fname  = params.simA_Dir + 'dnfdX.npy'
  simA_rhofR_fname  = params.simA_Dir + 'rhofR.npy'
  simA_time_fname   = params.simA_Dir + 'time.npy'

  if params.simB_Dir is not None:
    simB_pf_fname     = params.simB_Dir + 'pf.npy'
    simB_tf_fname     = params.simB_Dir + 'tf.npy'
    simB_ts_fname     = params.simB_Dir + 'ts.npy'
    simB_ns_fname     = params.simB_Dir + 'ns.npy'
    simB_J_fname      = params.simB_Dir + 'J.npy'
    simB_JDot_fname   = params.simB_Dir + 'JDot.npy'
    simB_vDarcy_fname = params.simB_Dir + 'vDarcy.npy'
    simB_pfDot_fname  = params.simB_Dir + 'pfDot.npy'
    simB_dp_fdX_fname = params.simB_Dir + 'dpfdX.npy'
    simB_qf_fname     = params.simB_Dir + 'qf.npy'
    simB_dnfdX_fname  = params.simB_Dir + 'dnfdX.npy'
    simB_rhofR_fname  = params.simB_Dir + 'rhofR.npy'
    simB_time_fname   = params.simB_Dir + 'time.npy'

  print("\nLoading in data...")
  #-------------------------------------------------------
  # Load in data and get coordinates for contour plotting.
  #-------------------------------------------------------
  #
  # Simulation A
  #
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  if simA_simParams.Physics != 'u-pf-ts-tf':
    sys.exit("--------\nERROR:\n--------\nSimulation A is not (u-pf-ts-tf) formulation. All simulations from input file must be (u-pf-ts-tf) formulation.")
  simA_coordsGauss = simA_preSimData[2]*params.dispScaling
  simA_tsolve      = np.load(simA_time_fname)
  simA_pf          = np.load(simA_pf_fname)
  simA_tf          = np.load(simA_tf_fname)
  simA_ts          = np.load(simA_ts_fname)
  simA_ns          = np.load(simA_ns_fname)
  simA_nf          = 1 - simA_ns
  simA_J           = np.load(simA_J_fname)
  simA_JDot        = np.load(simA_JDot_fname)
  simA_vDarcy      = np.load(simA_vDarcy_fname)
  simA_pfDot       = np.load(simA_pfDot_fname)
  simA_dp_fdX      = np.load(simA_dp_fdX_fname)
  simA_qf          = np.load(simA_qf_fname)
  simA_dnfdX       = np.load(simA_dnfdX_fname)
  simA_rhofR       = np.load(simA_rhofR_fname)
  simA_dtfdX       = -simA_qf/(simA_nf*simA_simParams.kf)
  
  simA_K3   = -simA_ns*simA_pf*simA_JDot
  simA_K4   = -simA_pf*simA_dnfdX*simA_vDarcy/simA_nf
  simA_K7   = -simA_qf
  simA_K8   = -simA_J*simA_simParams.k_exchange*(simA_ts - simA_tf)
  if simA_simParams.fluidModel == 'Ideal-Gas':
    simA_K2 = simA_rhofR*(simA_simParams.cvf + simA_simParams.RGas)
    simA_K5 = -simA_J*simA_nf*simA_pfDot
    simA_K6 = -simA_dp_fdX*simA_vDarcy 
  elif simA_simParams.fluidModel == 'Exponential-Thermal':
    simA_K2 = (simA_rhofR*simA_simParams.cvf + simA_tf*simA_simParams.KF*(simA_simParams.Af**2))
    simA_K5 = -simA_J*simA_nf*simA_pfDot*simA_tf*simA_simParams.Af
    simA_K6 = -simA_dp_fdX*simA_vDarcy*simA_tf*simA_simParams.Af
  simA_K2  *= simA_dtfdX*simA_vDarcy 
  simA_K2 *= simA_simParams.H0e*simA_simParams.Area
  simA_K3 *= simA_simParams.H0e*simA_simParams.Area
  simA_K4 *= simA_simParams.H0e*simA_simParams.Area
  simA_K5 *= simA_simParams.H0e*simA_simParams.Area
  simA_K6 *= simA_simParams.H0e*simA_simParams.Area
  simA_K7 *= simA_simParams.H0e*simA_simParams.Area
  simA_K8 *= simA_simParams.H0e*simA_simParams.Area
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    if simB_simParams.Physics != 'u-pf-ts-tf':
      sys.exit("--------\nERROR:\n--------\nSimulation B is not (u-pf-ts-tf) formulation. Bll simulations from input file must be (u-pf-ts-tf) formulation.")
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    simB_tsolve      = np.load(simB_time_fname)
    simB_pf          = np.load(simB_pf_fname)
    simB_tf          = np.load(simB_tf_fname)
    simB_ts          = np.load(simB_ts_fname)
    simB_ns          = np.load(simB_ns_fname)
    simB_nf          = 1 - simB_ns
    simB_J           = np.load(simB_J_fname)
    simB_JDot        = np.load(simB_JDot_fname)
    simB_vDarcy      = np.load(simB_vDarcy_fname)
    simB_pfDot       = np.load(simB_pfDot_fname)
    simB_dp_fdX      = np.load(simB_dp_fdX_fname)
    simB_qf          = np.load(simB_qf_fname)
    simB_dnfdX       = np.load(simB_dnfdX_fname)
    simB_rhofR       = np.load(simB_rhofR_fname)
    simB_dtfdX       = -simB_qf/(simB_nf*simB_simParams.kf)
    
    simB_K3   = -simB_ns*simB_pf*simB_JDot
    simB_K4   = -simB_pf*simB_dnfdX*simB_vDarcy/simB_nf
    simB_K7   = -simB_qf
    simB_K8   = -simB_J*simB_simParams.k_exchange*(simB_ts - simB_tf)
    if simB_simParams.fluidModel == 'Ideal-Gas':
      simB_K2 = simB_rhofR*(simB_simParams.cvf + simB_simParams.RGas)
      simB_K5 = -simB_J*simB_nf*simB_pfDot
      simB_K6 = -simB_dp_fdX*simB_vDarcy 
    elif simB_simParams.fluidModel == 'Exponential-Thermal':
      simB_K2 = (simB_rhofR*simB_simParams.cvf + simB_tf*simB_simParams.KF*(simB_simParams.Bf**2))
      simB_K5 = -simB_J*simB_nf*simB_pfDot*simB_tf*simB_simParams.Bf
      simB_K6 = -simB_dp_fdX*simB_vDarcy*simB_tf*simB_simParams.Bf
    simB_K2  *= simB_dtfdX*simB_vDarcy 

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    simA_K2 = np.mean(simA_K2, axis=2)
    simA_K3 = np.mean(simA_K3, axis=2)
    simA_K4 = np.mean(simA_K4, axis=2)
    simA_K5 = np.mean(simA_K5, axis=2)
    simA_K6 = np.mean(simA_K6, axis=2)
    simA_K7 = np.mean(simA_K7, axis=2)
    simA_K8 = np.mean(simA_K8, axis=2)
    if params.simB_Dir is not None:
      simB_K2 = np.mean(simB_K2, axis=2)
      simB_K3 = np.mean(simB_K3, axis=2)
      simB_K4 = np.mean(simB_K4, axis=2)
      simB_K5 = np.mean(simB_K5, axis=2)
      simB_K6 = np.mean(simB_K6, axis=2)
      simB_K7 = np.mean(simB_K7, axis=2)
      simB_K8 = np.mean(simB_K8, axis=2)
  else:
    simA_K2 = simA_K2.reshape(simA_K2.shape[0],int(simA_K2.shape[1]*simA_K2.shape[2]))
    simA_K3 = simA_K3.reshape(simA_K3.shape[0],int(simA_K3.shape[1]*simA_K3.shape[2]))
    simA_K4 = simA_K4.reshape(simA_K4.shape[0],int(simA_K4.shape[1]*simA_K4.shape[2]))
    simA_K5 = simA_K5.reshape(simA_K5.shape[0],int(simA_K5.shape[1]*simA_K5.shape[2]))
    simA_K6 = simA_K6.reshape(simA_K6.shape[0],int(simA_K6.shape[1]*simA_K6.shape[2]))
    simA_K7 = simA_K7.reshape(simA_K7.shape[0],int(simA_K7.shape[1]*simA_K7.shape[2]))
    simA_K8 = simA_K8.reshape(simA_K8.shape[0],int(simA_K8.shape[1]*simA_K8.shape[2]))
    simA_coordsGauss = simA_coordsGauss.flatten()
    if params.simB_Dir is not None:
      simB_K2 = simB_K2.reshape(simB_K2.shape[0],int(simB_K2.shape[1]*simB_K2.shape[2]))
      simB_K3 = simB_K3.reshape(simB_K3.shape[0],int(simB_K3.shape[1]*simB_K3.shape[2]))
      simB_K4 = simB_K4.reshape(simB_K4.shape[0],int(simB_K4.shape[1]*simB_K4.shape[2]))
      simB_K5 = simB_K5.reshape(simB_K5.shape[0],int(simB_K5.shape[1]*simB_K5.shape[2]))
      simB_K6 = simB_K6.reshape(simB_K6.shape[0],int(simB_K6.shape[1]*simB_K6.shape[2]))
      simB_K7 = simB_K7.reshape(simB_K7.shape[0],int(simB_K7.shape[1]*simB_K7.shape[2]))
      simB_K8 = simB_K8.reshape(simB_K8.shape[0],int(simB_K8.shape[1]*simB_K8.shape[2]))
      simB_coordsGauss = simB_coordsGauss.flatten()
  #------------------------------------------------------------------------------
  # Check that every simulation has at least as many data points as Simulation A.
  #------------------------------------------------------------------------------
  if params.simB_Dir is not None:
    if simB_tsolve.shape[0] != simA_tsolve.shape[0]:
      print("--------\nWARNING:\n--------\nSimulation B number of data points do not match Simulation A number of data points.")
  #-----------------
  # Generate frames.
  #-----------------
  print("\nGenerating plots...")

  fig = plt.figure(1)

  try:
    for timeIndex in range(params.startID, params.stopID, params.simA_Skip):
      print("Generating .png #%i" %int(timeIndex/params.simA_Skip))
      ax1 = plt.subplot(111)
      #----
      # K2.
      #----
      ax1.plot(simA_coordsGauss, simA_K2[timeIndex,:], \
               params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, \
               label=r'$\cK_2(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, simB_K2[timeIndex,:], \
                 params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, \
                 label=r'$\cK_2(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
      #----
      # K3.
      #----
      ax1.plot(simA_coordsGauss, simA_K3[timeIndex,:], \
               params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, \
               label=r'$\cK_3(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, simB_K3[timeIndex,:], \
                 params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, \
                 label=r'$\cK_3(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
      #----
      # K4.
      #----
      ax1.plot(simA_coordsGauss, simA_K4[timeIndex,:], \
               params.simA_Linestyle_Charlie, color=params.simA_Color_Charlie, fillstyle=params.simA_fillstyle, \
               label=r'$\cK_4(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, simB_K4[timeIndex,:], \
                 params.simB_Linestyle_Charlie, color=params.simB_Color_Charlie, fillstyle=params.simB_fillstyle, \
                 label=r'$\cK_4(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
      #----
      # K5.
      #----
      ax1.plot(simA_coordsGauss, simA_K5[timeIndex,:], \
               params.simA_Linestyle_Delta, color=params.simA_Color_Delta, fillstyle=params.simA_fillstyle, \
               label=r'$\cK_5(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      if params.simB_Dir is not None:
        ax1.plot(simB_coordsGauss, simB_K5[timeIndex,:], \
                 params.simB_Linestyle_Delta, color=params.simB_Color_Delta, fillstyle=params.simB_fillstyle, \
                 label=r'$\cK_5(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
#      #----
#      # K6.
#      #----
#      ax1.plot(simA_coordsGauss, simA_K6[timeIndex,:], \
#               params.simA_Linestyle_Echo, color=params.simA_Color_Echo, fillstyle=params.simA_fillstyle, \
#               label=r'$\cK_6(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
#      if params.simB_Dir is not None:
#        ax1.plot(simB_coordsGauss, simB_K2[timeIndex,:], \
#                 params.simB_Linestyle_Echo, color=params.simB_Color_Echo, fillstyle=params.simB_fillstyle, \
#                 label=r'$\cK_6(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
#      #----
#      # K7.
#      #----
#      ax1.plot(simA_coordsGauss, simA_K7[timeIndex,:], \
#               params.simA_Linestyle_Alpha, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, \
#               label=r'$\cK_7(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
#      if params.simB_Dir is not None:
#        ax1.plot(simB_coordsGauss, simB_K7[timeIndex,:], \
#                 params.simB_Linestyle_Alpha, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, \
#                 label=r'$\cK_7(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)
#      #----
#      # K8.
#      #----
      ax1.plot(simA_coordsGauss, simA_K8[timeIndex,:], \
               params.simA_Linestyle_Alpha, color=params.simA_Color_Charlie, fillstyle=params.simA_fillstyle, \
               label=r'$\cK_8(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
#      if params.simB_Dir is not None:
#        ax1.plot(simB_coordsGauss, simB_K8[timeIndex,:], \
#                 params.simB_Linestyle_Alpha, color=params.simB_Color_Charlie, fillstyle=params.simB_fillstyle, \
#                 label=r'$\cK_8(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

      if not params.no_labels:
        ax1.set_xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        ax1.set_ylabel(r'Power (W)', fontsize=params.yAxisFontSize)

      if params.ylim0 is not None and params.ylim1 is not None:
        ax1.set_ylim([params.ylim0, params.ylim1])
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

