#--------------------------------------------------------------------------------------------------
# Plotting script for volume fraction(s) at single depth(s) for multiple simulations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    May 11, 2022
#--------------------------------------------------------------------------------------------------
try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import matplotlib.pyplot as plt
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

#--------------------------------------------------------------------------------------------------
#-----------
# Arguments:
#-----------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(a_TimeDict, params):

  if params.simA_ElType == 'Q8':
    sys.exit("ERROR. Volume fraction plotting only enabled for Python code, check inputs for Q8 element type.")
  if params.simB_Dir is not None:
    if params.simB_ElType == 'Q8':
      sys.exit("ERROR. Volume fraction plotting only enabled for Python code, check inputs for Q8 element type.")
  if params.simC_Dir is not None:
    if params.simC_ElType == 'Q8':
      sys.exit("ERROR. Volume fraction plotting only enabled for Python code, check inputs for Q8 element type.")
  if params.simD_Dir is not None:
    if params.simD_ElType == 'Q8':
      sys.exit("ERROR. Volume fraction plotting only enabled for Python code, check inputs for Q8 element type.")
  if params.simE_Dir is not None:
    if params.simE_ElType == 'Q8':
      sys.exit("ERROR. Volume fraction plotting only enabled for Python code, check inputs for Q8 element type.")

  #---------------------
  # Generate file names.
  #---------------------
  simA_ns_fname    = params.simA_Dir + 'ns.npy'
  simA_J_fname     = params.simA_Dir + 'J.npy'
  simA_rhofr_fname = params.simA_Dir + 'rhofR.npy'
  simA_time_fname  = params.simA_Dir + 'time.npy' 

  if params.simB_Dir is not None:
    simB_ns_fname    = params.simB_Dir + 'ns.npy'
    simB_J_fname     = params.simB_Dir + 'J.npy'
    simB_rhofr_fname = params.simB_Dir + 'rhofR.npy'
    simB_time_fname  = params.simB_Dir + 'time.npy' 

  if params.simC_Dir is not None:
    simC_ns_fname    = params.simC_Dir + 'ns.npy'
    simC_J_fname     = params.simC_Dir + 'J.npy'
    simC_rhofr_fname = params.simC_Dir + 'rhofR.npy'
    simC_time_fname  = params.simC_Dir + 'time.npy' 

  if params.simD_Dir is not None:
    simD_ns_fname    = params.simD_Dir + 'ns.npy'
    simD_J_fname     = params.simD_Dir + 'J.npy'
    simD_rhofr_fname = params.simD_Dir + 'rhofR.npy'
    simD_time_fname  = params.simD_Dir + 'time.npy'

  if params.simE_Dir is not None:
    simE_ns_fname    = params.simE_Dir + 'ns.npy'
    simE_J_fname     = params.simE_Dir + 'J.npy'
    simE_rhofr_fname = params.simE_Dir + 'rhofR.npy'
    simE_time_fname  = params.simE_Dir + 'time.npy'

  #--------------
  # Load in data.
  #--------------
  print("\nLoading in data...")
  
  simA_ns    = np.load(simA_ns_fname)
  simA_J     = np.load(simA_J_fname)
  simA_rhofr = np.load(simA_rhofr_fname)
  if params.simA_EndTime == 'Load':
    simA_tsolve = np.load(simA_time_fname)

  if params.simB_Dir is not None:
    simB_ns = np.load(simB_ns_fname)
    simB_J  = np.load(simB_J_fname)
    simB_rhofr = np.load(simB_rhofr_fname)
    if params.simB_EndTime == 'Load':
      simB_tsolve = np.load(simB_time_fname)

  if params.simC_Dir is not None:
    simC_ns = np.load(simC_ns_fname)
    simC_J  = np.load(simC_J_fname)
    simC_rhofr = np.load(simC_rhofr_fname)
    if params.simC_EndTime == 'Load':
      simC_tsolve = np.load(simC_time_fname)

  if params.simD_Dir is not None:
    simD_ns = np.load(simD_ns_fname)
    simD_J  = np.load(simD_J_fname)
    simD_rhofr = np.load(simD_rhofr_fname)
    if params.simD_EndTime == 'Load':
      simD_tsolve = np.load(simD_time_fname)

  if params.simE_Dir is not None:
    simE_ns = np.load(simE_ns_fname)
    simE_J  = np.load(simE_J_fname)
    simE_rhofr = np.load(simE_rhofr_fname)
    if params.simE_EndTime == 'Load':
      simE_tsolve = np.load(simE_time_fname)

  print("Data loaded successfully.")
  print("Generating plots...")

  #----------------------------------------
  # Get closest nodes for requested probes.
  #----------------------------------------
  #
  # Simulation A
  #
  if params.simA_ElType == 'Q2P1':
    simA_coordsGauss = np.zeros((params.simA_El, 3))
    for el in range(params.simA_El):
      simA_coordsGauss[el,0] = params.simA_H0e*(el + 0.5*np.sqrt(3/5))
      simA_coordsGauss[el,1] = params.simA_H0e*(el + np.sqrt(3/5))
      simA_coordsGauss[el,2] = params.simA_H0e*(el + 0.5)
  elif params.simA_ElType == 'Q1P1':
    simA_coordsGauss = np.zeros((params.simA_El, 2))
    for el in range(params.simA_El):
      simA_coordsGauss[el,0] = params.simA_H0e*(el + np.sqrt(1/3))
      simA_coordsGauss[el,1] = params.simA_H0e*(el + np.sqrt(2/3))
  
  idx_A_e, idx_A_g = np.where(simA_coordsGauss == simA_coordsGauss.flat[np.abs(simA_coordsGauss - params.simA_probe_1).argmin()])
  
  if len(np.where(simA_coordsGauss == params.simA_probe_1)[0]) < 1:
    signA = r'\approx'
  else:
    signA = r'='
  #
  # Simulation B
  #
  if params.simB_Dir is not None:
    if params.simB_ElType == 'Q2P1':
      simB_coordsGauss = np.zeros((params.simB_El, 3))
      for el in range(params.simB_El):
        simB_coordsGauss[el,0] = params.simB_H0e*(el + 0.5*np.sqrt(3/5))
        simB_coordsGauss[el,1] = params.simB_H0e*(el + np.sqrt(3/5))
        simB_coordsGauss[el,2] = params.simB_H0e*(el + 0.5)
    elif params.simB_ElType == 'Q1P1':
      simB_coordsGauss = np.zeros((params.simB_El, 2))
      for el in range(params.simA_El):
        simB_coordsGauss[el,0] = params.simB_H0e*(el + np.sqrt(1/3))
        simB_coordsGauss[el,1] = params.simB_H0e*(el + np.sqrt(2/3))
    
    idx_B_e, idx_B_g = np.where(simB_coordsGauss == simB_coordsGauss.flat[np.abs(simB_coordsGauss - params.simB_probe_1).argmin()])
    
    if len(np.where(simB_coordsGauss == params.simB_probe_1)[0]) < 1:
      signB = r'\approx'
    else:
      signB = r'='
  #
  # Simulation C
  #
  if params.simC_Dir is not None:
    if params.simC_ElType == 'Q2P1':
      simC_coordsGauss = np.zeros((params.simC_El, 3))
      for el in range(params.simC_El):
        simC_coordsGauss[el,0] = params.simC_H0e*(el + 0.5*np.sqrt(3/5))
        simC_coordsGauss[el,1] = params.simC_H0e*(el + np.sqrt(3/5))
        simC_coordsGauss[el,2] = params.simC_H0e*(el + 0.5)
    elif params.simC_ElType == 'Q1P1':
      simC_coordsGauss = np.zeros((params.simC_El, 2))
      for el in range(params.simA_El):
        simC_coordsGauss[el,0] = params.simC_H0e*(el + np.sqrt(1/3))
        simC_coordsGauss[el,1] = params.simC_H0e*(el + np.sqrt(2/3))
    
    idx_C_e, idx_C_g = np.where(simC_coordsGauss == simC_coordsGauss.flat[np.abs(simC_coordsGauss - params.simC_probe_1).argmin()])
    
    if len(np.where(simC_coordsGauss == params.simC_probe_1)[0]) < 1:
      signC = r'\approx'
    else:
      signC = r'='
  #
  # Simulation D
  #
  if params.simD_Dir is not None:
    if params.simD_ElType == 'Q2P1':
      simD_coordsGauss = np.zeros((params.simD_El, 3))
      for el in range(params.simD_El):
        simD_coordsGauss[el,0] = params.simD_H0e*(el + 0.5*np.sqrt(3/5))
        simD_coordsGauss[el,1] = params.simD_H0e*(el + np.sqrt(3/5))
        simD_coordsGauss[el,2] = params.simD_H0e*(el + 0.5)
    elif params.simD_ElType == 'Q1P1':
      simD_coordsGauss = np.zeros((params.simD_El, 2))
      for el in range(params.simA_El):
        simD_coordsGauss[el,0] = params.simD_H0e*(el + np.sqrt(1/3))
        simD_coordsGauss[el,1] = params.simD_H0e*(el + np.sqrt(2/3))
    
    idx_D_e, idx_D_g = np.where(simD_coordsGauss == simD_coordsGauss.flat[np.abs(simD_coordsGauss - params.simD_probe_1).argmin()])
    
    if len(np.where(simD_coordsGauss == params.simD_probe_1)[0]) < 1:
      signD = r'\approx'
    else:
      signD = r'='
  #
  # Simulation E
  #
  if params.simE_Dir is not None:
    if params.simE_ElType == 'Q2P1':
      simE_coordsGauss = np.zeros((params.simE_El, 3))
      for el in range(params.simE_El):
        simE_coordsGauss[el,0] = params.simE_H0e*(el + 0.5*np.sqrt(3/5))
        simE_coordsGauss[el,1] = params.simE_H0e*(el + np.sqrt(3/5))
        simE_coordsGauss[el,2] = params.simE_H0e*(el + 0.5)
    elif params.simE_ElType == 'Q1P1':
      simE_coordsGauss = np.zeros((params.simE_El, 2))
      for el in range(params.simA_El):
        simE_coordsGauss[el,0] = params.simE_H0e*(el + np.sqrt(1/3))
        simE_coordsGauss[el,1] = params.simE_H0e*(el + np.sqrt(2/3))
    
    idx_E_e, idx_E_g = np.where(simE_coordsGauss == simE_coordsGauss.flat[np.abs(simE_coordsGauss - params.simE_probe_1).argmin()])
    
    if len(np.where(simE_coordsGauss == params.simE_probe_1)[0]) < 1:
      signE = r'\approx'
    else:
      signE = r'='

  #---------------------------------------------------------------------------------------
  # Perform intermediate calculations to back out the densities, and resulting wave speed.
  # Assume constituents are air and incompressible lung tissue or else edit the following lines.
  #---------------------------------------------------------------------------------------
  rhofr_0 = 1.2
  rhosr_0 = 1e3

  KS    = 22e9
  KSkel = 7.5e3
  KSin  = 213e3
  KF    = 140e3
  G     = 3e3

  simA_nf  = 1 - simA_ns
  simA_rho = (simA_nf[0,0,0]*rhofr_0 + simA_ns[0,0,0]*rhosr_0)/simA_J
  if params.singlephase_C:
    simA_C = np.sqrt((KSin + (4/3)*G)/simA_rho)
  elif params.multiphase_naive_C:
    simA_C = np.sqrt((Kskel + (4/3)*G)/simA_rho)
  elif params.multiphase_complex_C:
    simA_C = np.sqrt((simA_ns*KS + simA_nf*KF + (4/3)*G)/simA_rho)
  simA_CFL = params.simA_H0e*simA_J/simA_C

  if params.simB_Dir is not None:
    simB_nf  = 1 - simB_ns
    simB_rho = (simB_nf[0,0,0]*rhofr_0 + simB_ns[0,0,0]*rhosr_0)/simB_J
    if params.singlephase_C:
      simB_C = np.sqrt((KSin + (4/3)*G)/simB_rho)
    elif params.multiphase_naive_C:
      simB_C = np.sqrt((Kskel + (4/3)*G)/simB_rho)
    elif params.multiphase_complex_C:
      simB_C = np.sqrt((simB_ns*KS + simB_nf*KF + (4/3)*G)/simB_rho)
    simB_CFL = params.simB_H0e*simB_J/simB_C

  if params.simC_Dir is not None:
    simC_nf  = 1 - simC_ns
    simC_rho = (simC_nf[0,0,0]*rhofr_0 + simC_ns[0,0,0]*rhosr_0)/simC_J
    if params.singlephase_C:
      simC_C = np.sqrt((KSin + (4/3)*G)/simC_rho)
    elif params.multiphase_naive_C:
      simC_C = np.sqrt((Kskel + (4/3)*G)/simC_rho)
    elif params.multiphase_complex_C:
      simC_C = np.sqrt((simC_ns*KS + simC_nf*KF + (4/3)*G)/simC_rho)
    simC_CFL = params.simC_H0e*simC_J/simC_C

  if params.simD_Dir is not None:
    simD_nf  = 1 - simD_ns
    simD_rho = (simD_nf[0,0,0]*rhofr_0 + simD_ns[0,0,0]*rhosr_0)/simD_J
    if params.singlephase_C:
      simD_C = np.sqrt((KSin + (4/3)*G)/simD_rho)
    elif params.multiphase_naive_C:
      simD_C = np.sqrt((Kskel + (4/3)*G)/simD_rho)
    elif params.multiphase_complex_C:
      simD_C = np.sqrt((simD_ns*KS + simD_nf*KF + (4/3)*G)/simD_rho)
    simD_CFL = params.simD_H0e*simD_J/simD_C

  if params.simE_Dir is not None:
    simE_nf  = 1 - simE_ns
    simE_rho = (simE_nf[0,0,0]*rhofr_0 + simE_ns[0,0,0]*rhosr_0)/simE_J
    if params.singlephase_C:
      simE_C = np.sqrt((KSin + (4/3)*G)/simE_rho)
    elif params.multiphase_naive_C:
      simE_C = np.sqrt((Kskel + (4/3)*G)/simE_rho)
    elif params.multiphase_complex_C:
      simE_C = np.sqrt((simE_ns*KS + simE_nf*KF + (4/3)*G)/simE_rho)
    simE_CFL = params.simE_H0e*simE_J/simE_C

  #------------------------------------
  # Plot density on ax1 and CFL on ax2.
  #------------------------------------
  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)
  ax2  = ax1.twinx()
  
  ax1.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_rho[::params.simA_Skip,idx_A_e, idx_A_g], \
           params.simA_Linestyle_Alpha, color=params.simA_Color, fillstyle=params.simA_fillstyle, \
           label=r'$\rho(X' + signA + str(params.simA_probe_1) + r'\text{m}' + r',t)$, ' + params.simA_Title)

  ax2.semilogy(simA_tsolve[::params.simA_Skip]*params.timeScaling, np.amin(simA_CFL[::params.simA_Skip], axis=(1,2)), \
               params.simA_Linestyle_Bravo, color=params.simA_Color, fillstyle=params.simA_fillstyle, \
               label=r'$\Delta t_{CFL}(\min\limits_e,t)$, ' + params.simA_Title)

  if params.simB_Dir is not None:
    ax1.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_rho[::params.simB_Skip,idx_B_e, idx_B_g], \
           params.simB_Linestyle_Alpha, color=params.simB_Color, fillstyle=params.simB_fillstyle, \
           label=r'$\rho(X' + signB + str(params.simB_probe_1) + r'\text{m}' + r',t)$, ' + params.simB_Title)

    ax2.semilogy(simB_tsolve[::params.simB_Skip]*params.timeScaling, np.amin(simB_CFL[::params.simB_Skip], axis=(1,2)), \
                 params.simB_Linestyle_Bravo, color=params.simB_Color, fillstyle=params.simB_fillstyle, \
                 label=r'$\Delta t_{CFL}(\min\limits_e,t)$, ' + params.simB_Title)
    
  if params.simC_Dir is not None:
    ax1.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_rho[::params.simC_Skip,idx_C_e, idx_C_g], \
           params.simC_Linestyle_Alpha, color=params.simC_Color, fillstyle=params.simC_fillstyle, \
           label=r'$\rho(X' + signC + str(params.simC_probe_1) + r'\text{m}' + r',t)$, ' + params.simC_Title)

    ax2.semilogy(simC_tsolve[::params.simC_Skip]*params.timeScaling, np.amin(simC_CFL[::params.simC_Skip], axis=(1,2)), \
                 params.simC_Linestyle_Bravo, color=params.simC_Color, fillstyle=params.simC_fillstyle, \
                 label=r'$\Delta t_{CFL}(\min\limits_e,t)$, ' + params.simC_Title)

  if params.simD_Dir is not None:
    ax1.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_rho[::params.simD_Skip,idx_D_e, idx_D_g], \
           params.simD_Linestyle_Alpha, color=params.simD_Color, fillstyle=params.simD_fillstyle, \
           label=r'$\rho(X' + signD + str(params.simD_probe_1) + r'\text{m}' + r',t)$, ' + params.simD_Title)

    ax2.semilogy(simD_tsolve[::params.simD_Skip]*params.timeScaling, np.amin(simD_CFL[::params.simD_Skip], axis=(1,2)), \
                 params.simD_Linestyle_Bravo, color=params.simD_Color, fillstyle=params.simD_fillstyle, \
                 label=r'$\Delta t_{CFL}(\min\limits_e,t)$, ' + params.simD_Title)
    
  if params.simE_Dir is not None:
    ax1.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_rho[::params.simE_Skip,idx_E_e, idx_E_g], \
           params.simE_Linestyle_Alpha, color=params.simE_Color, fillstyle=params.simE_fillstyle, \
           label=r'$\rho(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)

    ax2.semilogy(simE_tsolve[::params.simE_Skip]*params.timeScaling, np.amin(simE_CFL[::params.simE_Skip], axis=(1,2)), \
                 params.simE_Linestyle_Bravo, color=params.simE_Color, fillstyle=params.simE_fillstyle, \
                 label=r'$n\Delta t_{CFL}(X' + signE + str(params.simE_probe_1) + r'\text{m}' + r',t)$, ' + params.simE_Title)

  ax1.set_xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=16)
  ax1.set_ylabel(r'$\rho$ (kg/m$^3$)', fontsize=16)
  ax2.set_ylabel(r'$\Delta t_{CFL}$', fontsize=16)

  if params.ylim0 is not None and params.ylim1 is not None:
    ax1.set_ylim([params.ylim0, params.ylim1])
    ax2.set_ylim([1e-8, 1e-3])
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
    ax1.grid(True, which=params.gridWhich)
  if params.legend:
    fig1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
  if params.title:
    fig1.suptitle(params.titleName)

  plt.savefig(params.outputDir + params.filename + '.png', bbox_inches='tight', dpi=params.DPI)
  plt.close()
  
  return
