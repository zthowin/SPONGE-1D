#--------------------------------------------------------------------------------------------------
# Plotting script for uniaxial stress(es) at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 24, 2024
#--------------------------------------------------------------------------------------------------
import os, sys

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
# a_StressDict     (dictionary)  mappings for stress scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, a_StressDict, params):
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
  #----------------------------------------------------------------
  # Load in data and get closest Gauss points for requested probes.
  #----------------------------------------------------------------
  #
  # Simulation A
  #
  simA_preSimData  = readPreSimData(params, params.simA_Dir, params.simA_InputFileName)
  simA_simParams   = simA_preSimData[0]
  simA_coordsGauss = simA_preSimData[2]
  if params.simA_isPython:
    simA_tsolve = np.load(simA_time_fname)
    simA_stress = np.load(simA_stress_fname)
    if 'pf' in simA_simParams.Physics:
      simA_stress_solid = np.load(simA_stress_solid_fname)
      simA_pf           = np.load(simA_pf_fname)
      simA_nf           = 1 - np.load(simA_ns_fname)
    simA_Gauss = getGaussPoint(params, simA_coordsGauss, params.simA_probe_1)
  elif params.simA_isDYNA:
    simA_stress = np.load(simA_stress_solid_fname)
    simA_tsolve = np.linspace(0, simA_simParams.TStop, simA_stress.shape[0])
    simA_Gauss  = getGaussDYNA(simA_coordsGauss, params.simA_probe_1)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    simB_preSimData  = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)
    simB_simParams   = simB_preSimData[0]
    simB_coordsGauss = simB_preSimData[2]
    if params.simB_isPython:
      simB_tsolve = np.load(simB_time_fname)
      simB_stress = np.load(simB_stress_fname)
      if 'pf' in simB_simParams.Physics:
        simB_stress_solid = np.load(simB_stress_solid_fname)
        simB_pf           = np.load(simB_pf_fname)
        simB_nf           = 1 - np.load(simB_ns_fname)
      simB_Gauss = getGaussPoint(params, simB_coordsGauss, params.simB_probe_1)
    elif params.simB_isDYNA:
      simB_stress = np.load(simB_stress_solid_fname)
      simB_tsolve = np.linspace(0, simB_simParams.TStop, simB_stress.shape[0])
      simB_Gauss  = getGaussDYNA(simB_coordsGauss, params.simB_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    simC_preSimData  = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)
    simC_simParams   = simC_preSimData[0]
    simC_coordsGauss = simC_preSimData[2]
    if params.simC_isPython:
      simC_tsolve = np.load(simC_time_fname)
      simC_stress = np.load(simC_stress_fname)
      if 'pf' in simC_simParams.Physics:
        simC_stress_solid = np.load(simC_stress_solid_fname)
        simC_pf           = np.load(simC_pf_fname)
        simC_nf           = 1 - np.load(simC_ns_fname)
      simC_Gauss = getGaussPoint(params, simC_coordsGauss, params.simC_probe_1)
    elif params.simC_isDYNA:
      simC_stress = np.load(simC_stress_solid_fname)
      simC_tsolve = np.linspace(0, simC_simParams.TStop, simC_stress.shape[0])
      simC_Gauss  = getGaussDYNA(simC_coordsGauss, params.simC_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    simD_preSimData  = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    simD_simParams   = simD_preSimData[0]
    simD_coordsGauss = simD_preSimData[2]
    if params.simD_isPython:
      simD_tsolve = np.load(simD_time_fname)
      simD_stress = np.load(simD_stress_fname)
      if 'pf' in simD_simParams.Physics:
        simD_stress_solid = np.load(simD_stress_solid_fname)
        simD_pf           = np.load(simD_pf_fname)
        simD_nf           = 1 - np.load(simD_ns_fname)
      simD_Gauss = getGaussPoint(params, simD_coordsGauss, params.simD_probe_1)
    elif params.simD_isDYNA:
      simD_stress = np.load(simD_stress_solid_fname)
      simD_tsolve = np.linspace(0, simD_simParams.TStop, simD_stress.shape[0])
      simD_Gauss  = getGaussDYNA(simD_coordsGauss, params.simD_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    simE_preSimData  = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    simE_simParams   = simE_preSimData[0]
    simE_coordsGauss = simE_preSimData[2]
    if params.simE_isPython:
      simE_tsolve = np.load(simE_time_fname)
      simE_stress = np.load(simE_stress_fname)
      if 'pf' in simE_simParams.Physics:
        simE_stress_solid = np.load(simE_stress_solid_fname)
        simE_pf           = np.load(simE_pf_fname)
        simE_nf           = 1 - np.load(simE_ns_fname)
      simE_Gauss = getGaussPoint(params, simE_coordsGauss, params.simE_probe_1)
    elif params.simE_isDYNA:
      simE_stress = np.load(simE_stress_solid_fname)
      simE_tsolve = np.linspace(0, simE_simParams.TStop, simE_stress.shape[0])
      simE_Gauss  = getGaussDYNA(simE_coordsGauss, params.simE_probe_1)
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nAnalytical solutions not accepted.")

  print("Data loaded successfully.")

  print("\nGenerating plots...")

  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  #----------------
  # Mixture stress.
  #----------------
  if params.totalPlot:
    if params.simA_isPython:
      ax1.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_stress[::params.simA_Skip,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)
    elif params.simA_isDYNA:
      ax1.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_stress[::params.simA_Skip,simA_Gauss]*params.stressScaling, params.simA_Linestyle_Alpha, color=params.simA_Color_Alpha, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if params.simB_isPython:
        ax1.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_stress[::params.simB_Skip,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      elif params.simB_isDYNA:
        ax1.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_stress[::params.simB_Skip,simB_Gauss]*params.stressScaling, params.simB_Linestyle_Alpha, color=params.simB_Color_Alpha, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      if params.simC_isPython:
        ax1.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_stress[::params.simC_Skip,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)
      elif params.simC_isDYNA:
        ax1.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_stress[::params.simC_Skip,simC_Gauss]*params.stressScaling, params.simC_Linestyle_Alpha, color=params.simC_Color_Alpha, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if params.simD_isPython:
        ax1.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_stress[::params.simD_Skip,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      elif params.simD_isDYNA:
        ax1.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_stress[::params.simD_Skip,simD_Gauss]*params.stressScaling, params.simD_Linestyle_Alpha, color=params.simD_Color_Alpha, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      if params.simE_isPython:
        ax1.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_stress[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
      elif params.simE_isDYNA:
        ax1.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_stress[::params.simE_Skip,simE_Gauss]*params.stressScaling, params.simE_Linestyle_Alpha, color=params.simE_Color_Alpha, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11}(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #--------------------
  # Solid extra stress.
  #--------------------
  if params.solidPlot:
    if 'pf' in simA_simParams.Physics:
      ax1.plot(simA_tsolve[::params.simA_SkipSecondary]*params.timeScaling, simA_stress_solid[::params.simA_SkipSecondary,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Bravo, color=params.simA_Color_Bravo, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if 'pf' in simB_simParams.Physics:
        ax1.plot(simB_tsolve[::params.simB_SkipSecondary]*params.timeScaling, simB_stress_solid[::params.simB_SkipSecondary,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Bravo, color=params.simB_Color_Bravo, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
      
    if params.simC_Dir is not None:
      if 'pf' in simC_simParams.Physics:
        ax1.plot(simC_tsolve[::params.simC_SkipSecondary]*params.timeScaling, simC_stress_solid[::params.simC_SkipSecondary,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Bravo, color=params.simC_Color_Bravo, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if 'pf' in simD_simParams.Physics:
        ax1.plot(simD_tsolve[::params.simD_SkipSecondary]*params.timeScaling, simD_stress_solid[::params.simD_SkipSecondary,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Bravo, color=params.simD_Color_Bravo, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
      
    if params.simE_Dir is not None:
      if 'pf' in simE_simParams.Physics:
        ax1.plot(simE_tsolve[::params.simE_SkipSecondary]*params.timeScaling, simE_stress_solid[::params.simE_SkipSecondary,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Bravo, color=params.simE_Color_Bravo, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11(E)}^\rs(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #-------------------------
  # Total pore fluid stress.
  #-------------------------
  if params.fluidPlot: 
    if 'pf' in simA_simParams.Physics:
      simA_stress_fluid = simA_stress - simA_stress_solid - (1 - simA_nf)*simA_pf
      ax1.plot(simA_tsolve[::params.simA_SkipTertiary]*params.timeScaling, simA_stress_fluid[::params.simA_SkipTertiary,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Charlie, color=params.simA_Color_Charlie, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if 'pf' in simB_simParams.Physics:
        simB_stress_fluid = simB_stress - simB_stress_solid - (1 - simB_nf)*simB_pf
        ax1.plot(simB_tsolve[::params.simB_SkipTertiary]*params.timeScaling, simB_stress_fluid[::params.simB_SkipTertiary,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Charlie, color=params.simB_Color_Charlie, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)
    
    if params.simC_Dir is not None:
      if 'pf' in simC_simParams.Physics:
        simC_stress_fluid = simC_stress - simC_stress_solid - (1 - simC_nf)*simC_pf
        ax1.plot(simC_tsolve[::params.simC_SkipTertiary]*params.timeScaling, simC_stress_fluid[::params.simC_SkipTertiary,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Charlie, color=params.simC_Color_Charlie, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if 'pf' in simD_simParams.Physics:
        simD_stress_fluid = simD_stress - simD_stress_solid - (1 - simD_nf)*simD_pf
        ax1.plot(simD_tsolve[::params.simD_SkipTertiary]*params.timeScaling, simD_stress_fluid[::params.simD_SkipTertiary,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Charlie, color=params.simD_Color_Charlie, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)
    
    if params.simE_Dir is not None:
      if 'pf' in simE_simParams.Physics:
        simE_stress_fluid = simE_stress - simE_stress_solid - (1 - simE_nf)*simE_pf
        ax1.plot(simE_tsolve[::params.simE_SkipTertiary]*params.timeScaling, simE_stress_fluid[::params.simE_SkipTertiary,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Charlie, color=params.simE_Color_Charlie, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11}^\rf(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #-------------------------
  # Pore fluid extra stress.
  #-------------------------
  if params.viscousPlot:
    if simA_simParams.DarcyBrinkman:
      simA_stress_fluid_extra = simA_stress - simA_pf - simA_stress_solid
      ax1.plot(simA_tsolve[::params.simA_SkipQuaternary]*params.timeScaling, simA_stress_fluid_extra[::params.simA_SkipQuaternary,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Delta, color=params.simA_Color_Delta, fillstyle=params.simA_fillstyle, label=r'$\sigma_{11(E)}^\rf(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if simB_simParams.DarcyBrinkman:
        simB_stress_fluid_extra = simB_stress - simB_pf - simB_stress_solid
        ax1.plot(simB_tsolve[::params.simB_SkipQuaternary]*params.timeScaling, simB_stress_fluid_extra[::params.simB_SkipQuaternary,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Delta, color=params.simB_Color_Delta, fillstyle=params.simB_fillstyle, label=r'$\sigma_{11(E)}^\rf(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)

    if params.simC_Dir is not None:
      if simC_simParams.DarcyBrinkman:
        simC_stress_fluid_extra = simC_stress - simC_pf - simC_stress_solid
        ax1.plot(simC_tsolve[::params.simC_SkipQuaternary]*params.timeScaling, simC_stress_fluid_extra[::params.simC_SkipQuaternary,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Delta, color=params.simC_Color_Delta, fillstyle=params.simC_fillstyle, label=r'$\sigma_{11(E)}^\rf(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if simD_simParams.DarcyBrinkman:
        simD_stress_fluid_extra = simD_stress - simD_pf - simD_stress_solid
        ax1.plot(simD_tsolve[::params.simD_SkipQuaternary]*params.timeScaling, simD_stress_fluid_extra[::params.simD_SkipQuaternary,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Delta, color=params.simD_Color_Delta, fillstyle=params.simD_fillstyle, label=r'$\sigma_{11(E)}^\rf(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)

    if params.simE_Dir is not None:
      if simE_simParams.Physics != 'u' and simE_simParams.DarcyBrinkman:
        simE_stress_fluid_extra = simE_stress - simE_pf - simE_stress_solid
        ax1.plot(simE_tsolve[::params.simE_SkipQuaternary]*params.timeScaling, simE_stress_fluid_extra[::params.simE_SkipQuaternary,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Delta, color=params.simE_Color_Delta, fillstyle=params.simE_fillstyle, label=r'$\sigma_{11(E)}^\rf(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  #-----------------------------
  # Partial pore fluid pressure.
  #-----------------------------
  if params.partialPlot:
    if 'pf' in simA_simParams.Physics:
      simA_stress_fluid_press = simA_nf*simA_pf
      ax1.plot(simA_tsolve[::params.simA_SkipQuinary]*params.timeScaling, simA_stress_fluid_press[::params.simA_SkipQuinary,simA_Gauss[0], simA_Gauss[1]]*params.stressScaling, params.simA_Linestyle_Echo, color=params.simA_Color_Echo, fillstyle=params.simA_fillstyle, label=r'$-n^\rf p_\rf(X(\xi) \approx ' + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

    if params.simB_Dir is not None:
      if 'pf' in simB_simParams.Physics:
        simB_stress_fluid_press = simB_nf*simB_pf
        ax1.plot(simB_tsolve[::params.simB_SkipQuinary]*params.timeScaling, simB_stress_fluid_press[::params.simB_SkipQuinary,simB_Gauss[0], simB_Gauss[1]]*params.stressScaling, params.simB_Linestyle_Echo, color=params.simB_Color_Echo, fillstyle=params.simB_fillstyle, label=r'$-n^\rf p_\rf(X(\xi) \approx ' + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)

    if params.simC_Dir is not None:
      if 'pf' in simC_simParams.Physics:
        simC_stress_fluid_press = simC_nf*simC_pf
        ax1.plot(simC_tsolve[::params.simC_SkipQuinary]*params.timeScaling, simC_stress_fluid_press[::params.simC_SkipQuinary,simC_Gauss[0], simC_Gauss[1]]*params.stressScaling, params.simC_Linestyle_Echo, color=params.simC_Color_Echo, fillstyle=params.simC_fillstyle, label=r'$-n^\rf p_\rf(X(\xi) \approx ' + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    if params.simD_Dir is not None:
      if 'pf' in simD_simParams.Physics:
        simD_stress_fluid_press = simD_nf*simD_pf
        ax1.plot(simD_tsolve[::params.simD_SkipQuinary]*params.timeScaling, simD_stress_fluid_press[::params.simD_SkipQuinary,simD_Gauss[0], simD_Gauss[1]]*params.stressScaling, params.simD_Linestyle_Echo, color=params.simD_Color_Echo, fillstyle=params.simD_fillstyle, label=r'$-n^\rf p_\rf(X(\xi) \approx ' + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)

    if params.simE_Dir is not None:
      if 'pf' in simE_simParams.Physics:
        simE_stress_fluid_press = simE_nf*simE_pf
        ax1.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_stress_fluid_press[::params.simE_Skip,simE_Gauss[0], simE_Gauss[1]]*params.stressScaling, params.simE_Linestyle_Echo, color=params.simE_Color_Echo, fillstyle=params.simE_fillstyle, label=r'$-n^\rf p_\rf(X(\xi) \approx ' + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)
  
  if not params.no_labels:
    ax1.set_xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
    ax1.set_ylabel(r'Cauchy stress ' + a_StressDict[params.stressScaling], fontsize=params.yAxisFontSize)

  if params.log:
    ax1.set_yscale('log')

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
    fig1.suptitle(params.titleName,y=params.titleLoc,fontsize=params.titleFontSize)

  plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
  plt.close()
  
  return

