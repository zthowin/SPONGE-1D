#--------------------------------------------------------------------------------------------------
# Houses Li et al. poroelasticity (small vs. finite strain) plotting.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 22, 2024
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

try:
  REPO = os.environ['REPO']
except KeyError:
  sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")

sys.path.insert(1, REPO + '/src/')

try:
  from moduleFE import *
except ImportError:
  sys.exit("MODULE WARNING. src/ modules not found, check configuration.")

#--------------------------------------------------------------------------------------------------
# Function to generate comparison plots for small strain and finite strain poroelasticity.
#
# Refer to Li et al. (2004)
#------------
# Parameters:
#------------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# a_DispDict       (dictionary)  mappings for displacement scale factors to appropriate labels
# a_StressDict     (dictionary)  mappings for stress scale factors to appropriate labels
# params           (object)      problem parameters initiated below via the class above
#--------------------------------------------------------------------------------------------------
def makeLiPlots(a_TimeDict, a_DispDict, a_stressDict, params):
  #---------------------
  # Generate file names.
  #---------------------
  simA_disp_fname = params.simA_Dir + 'displacement.npy'
  simA_time_fname = params.simA_Dir + 'time.npy' 

  if params.simB_Dir is not None:
    simB_disp_fname = params.simB_Dir + 'displacement.npy'
    simB_time_fname = params.simB_Dir + 'time.npy' 

  if params.simC_Dir is not None:
    simC_disp_fname = params.simC_Dir + 'displacement.npy'
    simC_time_fname = params.simC_Dir + 'time.npy' 
  #--------------
  # Load in data.
  #--------------
  print("\nLoading in data...")

  simA_Dsolve       = np.load(simA_disp_fname)
  simA_tsolve       = np.load(simA_time_fname)

  if params.simB_Dir is not None:
    simB_Dsolve = np.load(simB_disp_fname)
    simB_tsolve = np.load(simB_time_fname)

  if params.simC_Dir is not None:
    simC_Dsolve = np.load(simC_disp_fname)
    simC_tsolve = np.load(simC_time_fname)

  print("Data loaded successfully.")
  print("Generating plots...")
  #---------------------
  # Generate plot names.
  #---------------------
  disp_u_plot_top_fname = params.outputDir + 'disp-top.pdf'
  pf_plot_top_fname     = params.outputDir + 'press-PF-top.pdf'
  #------------------------------------
  # Plot displacement at top of column.
  #------------------------------------
  fig1 = plt.figure(1)
  ax   = fig1.add_subplot(111)
  
  plt.plot(simA_tsolve[:-1]*params.timeScaling, simA_Dsolve[:-1,params.simA_TopNode_u]*params.dispScaling, 'k-', label=r'Large deformation')

  if params.simA_Title == '40 kPa':
    plt.plot(simA_tsolve[:-1]*params.timeScaling, -np.ones(simA_tsolve[:-1].shape[0])*40e3*10*params.dispScaling/(29e6 + 14e6), 'k--', label=r'Analytical, steady-state solution (at small strain)')
  else:
    plt.plot(simA_tsolve[:-1]*params.timeScaling, -np.ones(simA_tsolve[:-1].shape[0])*2e6*10*params.dispScaling/(29e6 + 14e6), 'k--', label=r'Analytical, steady-state solution (at small strain)')

  if params.simB_Dir is not None:
    plt.plot(simB_tsolve[:-1]*params.timeScaling, simB_Dsolve[:-1,params.simB_TopNode_u]*params.dispScaling,'k-')
    plt.plot(simB_tsolve[:-1]*params.timeScaling, -np.ones(simB_tsolve[:-1].shape[0])*4e6*10*params.dispScaling/(29e6 + 14e6), 'k--')
  
  if params.simC_Dir is not None:
    plt.plot(simC_tsolve[:-1]*params.timeScaling, simC_Dsolve[:-1,params.simC_TopNode_u]*params.dispScaling, 'k-')
    plt.plot(simC_tsolve[:-1]*params.timeScaling, -np.ones(simC_tsolve[:-1].shape[0])*8e6*10*params.dispScaling/(29e6 + 14e6), 'k--')

  # plt.text(0.77, 0.2, r'$t^\sigma_0 = 40$kPa',fontsize=10,transform=plt.gcf().transFigure))
  plt.text(0.77, 0.73, r'$t^\sigma_0 = 2$MPa',fontsize=10, transform=plt.gcf().transFigure)
  plt.text(0.77, 0.58, r'$t^\sigma_0 = 4$MPa',fontsize=10, transform=plt.gcf().transFigure)
  plt.text(0.77, 0.315, r'$t^\sigma_0 = 8$MPa',fontsize=10, transform=plt.gcf().transFigure)

  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Displacement ' + a_DispDict[params.dispScaling], fontsize=16)
  plt.xlim([0, 1])
  plt.ylim([-2, 0])
  # plt.ylim([-10,0])
  # plt.grid()
  plt.legend(loc='upper right')
  # plt.legend(bbox_to_anchor=(0.5, 1.1), loc='center')
  plt.savefig(disp_u_plot_top_fname, bbox_inches='tight', dpi=600)
  plt.close()
  #---------------------------
  # Plot pore fluid pressures.
  #---------------------------
  #--------------------------------
  # Plot pressure at top of column.
  #--------------------------------
  fig4 = plt.figure(4)
  ax = fig4.add_subplot(111)

  plt.plot(simA_tsolve[:-1]*params.timeScaling, simA_Dsolve[:-1,params.simA_TopNode_pf]*params.stressScaling, 'k-')

  if params.simB_Dir is not None:
    plt.plot(simB_tsolve[:-1]*params.timeScaling, simB_Dsolve[:-1,params.simB_TopNode_pf]*params.stressScaling, 'k-')

  if params.simC_Dir is not None:
    plt.plot(simC_tsolve[:-1]*params.timeScaling, simC_Dsolve[:-1,params.simC_TopNode_pf]*params.stressScaling, 'k-')

  # plt.text(0.65, 0.55, r'$\sigma_0^n = 8$MPa',fontsize=10)
  # plt.text(0.935, 0.55, r'$\sigma_0^n = 8$MPa',fontsize=10)
  plt.text(0.77, 0.73, r'$t^\sigma_0 = 2$MPa',fontsize=10, transform=plt.gcf().transFigure)
  plt.text(0.77, 0.58, r'$t^\sigma_0 = 4$MPa',fontsize=10, transform=plt.gcf().transFigure)
  plt.text(0.77, 0.315, r'$t^\sigma_0 = 8$MPa',fontsize=10, transform=plt.gcf().transFigure)
  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  plt.ylabel(r'Pressure ' + a_StressDict[params.stressScaling], fontsize=16)
  plt.xlim([0, 1])
  # plt.ylim([0, 8])
  plt.grid()
  # plt.legend(bbox_to_anchor=(0.5, 1.1), loc='center')
  plt.savefig(pf_plot_top_fname, bbox_inches='tight', dpi=600)
  plt.close()

  print("Plots generated successfully.")

