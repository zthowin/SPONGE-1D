#--------------------------------------------------------------------------------------------------
# Plotting script for animation of stress contour(s).
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
  sys.exit("MODULE WARNING. preSimData  not found, check configuration.")

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
  simA_stress_fname       = params.simA_Dir + 'sig11.npy'
  simA_stress_solid_fname = params.simA_Dir + 'P11.npy'
  simA_pf_fname           = params.simA_Dir + 'pf.npy'
  simA_ns_fname           = params.simA_Dir + 'ns.npy'
  simA_time_fname         = params.simA_Dir + 'time.npy'
  if not os.path.isfile(simA_stress_solid_fname):
    sys.exit("------\nERROR:\n------\nSolid stress data not found for Simulation A.")

  if params.simB_Dir is not None:
    simB_stress_fname       = params.simB_Dir + 'sig11.npy'
    simB_stress_solid_fname = params.simB_Dir + 'P11.npy'
    simB_pf_fname           = params.simB_Dir + 'pf.npy'
    simB_ns_fname           = params.simB_Dir + 'ns.npy'
    simB_time_fname         = params.simB_Dir + 'time.npy'
    if not os.path.isfile(simB_stress_solid_fname):
      sys.exit("------\nERROR:\n------\nSolid stress data not found for Simulation B.")

  if params.simC_Dir is not None:
    simC_stress_fname       = params.simC_Dir + 'sig11.npy'
    simC_stress_solid_fname = params.simC_Dir + 'P11.npy'
    simC_pf_fname           = params.simC_Dir + 'pf.npy'
    simC_ns_fname           = params.simC_Dir + 'ns.npy'
    simC_time_fname         = params.simC_Dir + 'time.npy'
    if not os.path.isfile(simC_stress_solid_fname):
      sys.exit("------\nERROR:\n------\nSolid stress data not found for Simulation C.")

  if params.simD_Dir is not None:
    simD_stress_fname       = params.simD_Dir + 'sig11.npy'
    simD_stress_solid_fname = params.simD_Dir + 'P11.npy'
    simD_pf_fname           = params.simD_Dir + 'pf.npy'
    simD_ns_fname           = params.simD_Dir + 'ns.npy'
    simD_time_fname         = params.simD_Dir + 'time.npy'
    if not os.path.isfile(simD_stress_solid_fname):
      sys.exit("------\nERROR:\n------\nSolid stress data not found for Simulation D.")

  if params.simE_Dir is not None:
    simE_stress_fname       = params.simE_Dir + 'sig11.npy'
    simE_stress_solid_fname = params.simE_Dir + 'P11.npy'
    simE_pf_fname           = params.simE_Dir + 'pf.npy'
    simE_ns_fname           = params.simE_Dir + 'ns.npy'
    simE_time_fname         = params.simE_Dir + 'time.npy'
    if not os.path.isfile(simE_stress_solid_fname):
      sys.exit("------\nERROR:\n------\nSolid stress data not found for Simulation E.")

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
  if params.simA_isPython:
    simA_tsolve       = np.load(simA_time_fname)
    simA_stress       = np.load(simA_stress_fname)
    if 'pf' in simA_simParams.Physics:
      simA_pf           = np.load(simA_pf_fname)
      simA_nf           = 1 - np.load(simA_ns_fname)
      simA_stress_solid = np.load(simA_stress_solid_fname)
      simA_stress_fluid = simA_stress - simA_stress_solid - (1 - simA_nf)*simA_pf
  elif params.simA_isDYNA:
    simA_stress = np.load(simA_stress_solid_fname)
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_stress.shape[0])
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simC_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]*params.dispScaling
    if params.simB_isPython:
      simB_tsolve       = np.load(simB_time_fname)
      simB_stress       = np.load(simB_stress_fname)
      if 'pf' in simB_simParams.Physics:
        simB_pf           = np.load(simB_pf_fname)
        simB_nf           = 1 - np.load(simB_ns_fname)
        simB_stress_solid = np.load(simB_stress_solid_fname)
        simB_stress_fluid = simB_stress - simB_stress_solid - (1 - simB_nf)*simB_pf
    elif params.simB_isDYNA:
      simB_stress = np.load(simB_stress_solid_fname)
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_stress.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]*params.dispScaling
    if params.simC_isPython:
      simC_tsolve       = np.load(simC_time_fname)
      simC_stress       = np.load(simC_stress_fname)
      if 'pf' in simC_simParams.Physics:
        simC_pf           = np.load(simC_pf_fname)
        simC_nf           = 1 - np.load(simC_ns_fname)
        simC_stress_solid = np.load(simC_stress_solid_fname)
        simC_stress_fluid = simC_stress - simC_stress_solid - (1 - simC_nf)*simC_pf
    elif params.simC_isDYNA:
      simC_stress = np.load(simC_stress_solid_fname)
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_stress.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]*params.dispScaling
    if params.simD_isPython:
      simD_tsolve       = np.load(simD_time_fname)
      simD_stress       = np.load(simD_stress_fname)
      if 'pf' in simD_simParams.Physics:
        simD_pf           = np.load(simD_pf_fname)
        simD_nf           = 1 - np.load(simD_ns_fname)
        simD_stress_solid = np.load(simD_stress_solid_fname)
        simD_stress_fluid = simD_stress - simD_stress_solid - (1 - simD_nf)*simD_pf
    elif params.simD_isDYNA:
      simD_stress = np.load(simD_stress_solid_fname)
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_stress.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]*params.dispScaling
    if params.simE_isPython:
      simE_tsolve       = np.load(simE_time_fname)
      simE_stress       = np.load(simE_stress_fname)
      if 'pf' in simE_simParams.Physics:
        simE_pf           = np.load(simE_pf_fname)
        simE_nf           = 1 - np.load(simE_ns_fname)
        simE_stress_solid = np.load(simE_stress_solid_fname)
        simE_stress_fluid = simE_stress - simE_stress_solid - (1 - simE_nf)*simE_pf
    elif params.simE_isDYNA:
      simE_stress = np.load(simE_stress_solid_fname)
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_stress.shape[0])
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")

  print("Data loaded successfully.")
  #--------------------------------
  # Perform averaging if necessary.
  #--------------------------------
  if params.averageGauss:
    if params.simA_isPython:
      simA_stress       = np.mean(simA_stress, axis=2)
      simA_stress_solid = np.mean(simA_stress_solid, axis=2)
      simA_stress_fluid = np.mean(simA_stress_fluid, axis=2)
    if params.simB_Dir is not None and params.simB_isPython:
      simB_stress       = np.mean(simB_stress, axis=2)
      simB_stress_solid = np.mean(simB_stress_solid, axis=2)
      simB_stress_fluid = np.mean(simB_stress_fluid, axis=2)
    if params.simC_Dir is not None and params.simC_isPython:
      simC_stress       = np.mean(simC_stress, axis=2)
      simC_stress_solid = np.mean(simC_stress_solid, axis=2)
      simC_stress_fluid = np.mean(simC_stress_fluid, axis=2)
    if params.simD_Dir is not None and params.simD_isPython:
      simD_stress       = np.mean(simD_stress, axis=2)
      simD_stress_solid = np.mean(simD_stress_solid, axis=2)
      simD_stress_fluid = np.mean(simD_stress_fluid, axis=2)
    if params.simE_Dir is not None and params.simE_isPython:
      simE_stress       = np.mean(simE_stress, axis=2)
      simE_stress_solid = np.mean(simE_stress_solid, axis=2)
      simE_stress_fluid = np.mean(simE_stress_fluid, axis=2)
  else:
    if params.simA_isPython:
      if 'pf' in simA_simParams.Physics:
        simA_stress_fluid = simA_stress_fluid.reshape(simA_stress_fluid.shape[0],int(simA_stress_fluid.shape[1]*simA_stress_fluid.shape[2]))
        simA_stress_solid   = simA_stress_solid.reshape(simA_stress_solid.shape[0],int(simA_stress_solid.shape[1]*simA_stress_solid.shape[2]))
      simA_stress         = simA_stress.reshape(simA_stress.shape[0],int(simA_stress.shape[1]*simA_stress.shape[2]))
      simA_coordsGauss    = simA_coordsGauss.flatten()
    if params.simB_Dir is not None and params.simB_isPython:
      if 'pf' in simB_simParams.Physics:
        simB_stress_fluid = simB_stress_fluid.reshape(simB_stress_fluid.shape[0],int(simB_stress_fluid.shape[1]*simB_stress_fluid.shape[2]))
        simB_stress_solid   = simB_stress_solid.reshape(simB_stress_solid.shape[0],int(simB_stress_solid.shape[1]*simB_stress_solid.shape[2]))
      simB_stress         = simB_stress.reshape(simB_stress.shape[0],int(simB_stress.shape[1]*simB_stress.shape[2]))
      simB_coordsGauss    = simB_coordsGauss.flatten()
    if params.simC_Dir is not None and params.simC_isPython:
      if 'pf' in simC_simParams.Physics:
        simC_stress_fluid = simC_stress_fluid.reshape(simC_stress_fluid.shape[0],int(simC_stress_fluid.shape[1]*simC_stress_fluid.shape[2]))
        simC_stress_solid   = simC_stress_solid.reshape(simC_stress_solid.shape[0],int(simC_stress_solid.shape[1]*simC_stress_solid.shape[2]))
      simC_stress         = simC_stress.reshape(simC_stress.shape[0],int(simC_stress.shape[1]*simC_stress.shape[2]))
      simC_coordsGauss    = simC_coordsGauss.flatten()
    if params.simD_Dir is not None and params.simD_isPython:
      if 'pf' in simD_simParams.Physics:
        simD_stress_fluid = simD_stress_fluid.reshape(simD_stress_fluid.shape[0],int(simD_stress_fluid.shape[1]*simD_stress_fluid.shape[2]))
        simD_stress_solid   = simD_stress_solid.reshape(simD_stress_solid.shape[0],int(simD_stress_solid.shape[1]*simD_stress_solid.shape[2]))
      simD_stress         = simD_stress.reshape(simD_stress.shape[0],int(simD_stress.shape[1]*simD_stress.shape[2]))
      simD_coordsGauss    = simD_coordsGauss.flatten()
    if params.simE_Dir is not None and params.simE_isPython:
      if 'pf' in simE_simParams.Physics:
        simE_stress_fluid        = simE_stress_fluid.reshape(simE_stress_fluid.shape[0],int(simE_stress_fluid.shape[1]*simE_stress_fluid.shape[2]))
        simE_stress_solid        = simE_stress_solid.reshape(simE_stress_solid.shape[0],int(simE_stress_solid.shape[1]*simE_stress_solid.shape[2]))
      simE_stress       = simE_stress.reshape(simE_stress.shape[0],int(simE_stress.shape[1]*simE_stress.shape[2]))
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
      #--------------
      # Total stress.
      #--------------
      if params.totalPlot:
        plt.plot(simA_coordsGauss, simA_stress[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11}(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
        if params.simB_Dir is not None:
          plt.plot(simB_coordsGauss, simB_stress[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11}(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None:
          plt.plot(simC_coordsGauss, simC_stress[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11}(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None:
          plt.plot(simD_coordsGauss, simD_stress[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11}(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None:
          plt.plot(simE_coordsGauss, simE_stress[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11}(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #--------------
      # Solid stress.
      #--------------
      if params.solidPlot:
        if 'pf' in simA_simParams.Physics:
          plt.plot(simA_coordsGauss, simA_stress_solid[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
        if params.simB_Dir is not None and 'pf' in simB_simParams.Physics:
          plt.plot(simB_coordsGauss, simB_stress_solid[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None and 'pf' in simC_simParams.Physics:
          plt.plot(simC_coordsGauss, simC_stress_solid[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)
      
        if params.simD_Dir is not None and 'pf' in simD_simParams.Physics:
          plt.plot(simD_coordsGauss, simD_stress_solid[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None and 'pf' in simE_simParams.Physics:
          plt.plot(simE_coordsGauss, simE_stress_solid[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)
      #-------------------------
      # Total pore fluid sterss.
      #-------------------------
      if params.fluidPlot:
        if 'pf' in simA_simParams.Physics:
          plt.plot(simA_coordsGauss, simA_stress_fluid[timeIndex,:]*params.stressScaling, params.simA_Linestyle_Charlie, color=params.simA_Color_Charlie, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simA_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simA_Title)
      
        if params.simB_Dir is not None and 'pf' in simB_simParams.Physics:
          plt.plot(simB_coordsGauss, simB_stress_fluid[timeIndex,:]*params.stressScaling, params.simB_Linestyle_Charlie, color=params.simB_Color_Charlie, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simB_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simB_Title)

        if params.simC_Dir is not None and 'pf' in simC_simParams.Physics:
          plt.plot(simC_coordsGauss, simC_stress_fluid[timeIndex,:]*params.stressScaling, params.simC_Linestyle_Charlie, color=params.simC_Color_Charlie, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simC_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simC_Title)

        if params.simD_Dir is not None and 'pf' in simD_simParams.Physics:
          plt.plot(simD_coordsGauss, simD_stress_fluid[timeIndex,:]*params.stressScaling, params.simD_Linestyle_Charlie, color=params.simD_Color_Charlie, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simD_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simD_Title)

        if params.simE_Dir is not None and 'pf' in simE_simParams.Physics:
          plt.plot(simE_coordsGauss, simE_stress_fluid[timeIndex,:]*params.stressScaling, params.simE_Linestyle_Charlie, color=params.simE_Color_Charlie, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi), t\approx$ ' + "{:.2e}".format(simE_tsolve[timeIndex]*params.timeScaling) + a_TimeDict[params.timeScaling].split('(')[1] + ', ' + params.simE_Title)

      if not params.no_labels:
        plt.xlabel(r'Position ' + a_DispDict[params.dispScaling], fontsize=params.xAxisFontSize)
        plt.ylabel(r'Cauchy stress ' + a_StressDict[params.stressScaling], fontsize=params.yAxisFontSize)

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

