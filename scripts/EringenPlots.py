#--------------------------------------------------------------------------------------------------
# Plotting script for Eringen analytical solutions.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 22, 2024
#--------------------------------------------------------------------------------------------------
import sys

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
# Function to generate plots comparing Eringen & Suhubi's (1975) analytical solution to 
# numerical data.
#------------
# Parameters:
#------------
# a_TimeDict       (dictionary)  mappings for time scale factors to appropriate labels
# a_DispDict       (dictionary)  mappings for displacement scale factors to appropriate labels
# params           (object)      problem parameters initiated below via the class above
#--------------------------------------------------------------------------------------------------
def makeEringenPlots(a_TimeDict, a_DispDict, params):

  simA_u_fname      = params.simA_Dir + 'displacement.npy'
  simA_du_fname     = params.simA_Dir + 'gradient.npy'
  simA_time_fname   = params.simA_Dir + 'time.npy'             

  if params.simB_Dir is not None:
    simB_disp_fname         = params.simB_Dir + 'displacement.npy'
    simB_time_fname         = params.simB_Dir + 'time.npy'        

  if params.simC_Dir is not None:
    simC_disp_fname         = params.simC_Dir + 'displacement.npy'
    simC_time_fname         = params.simC_Dir + 'time.npy'                  

  if params.simD_Dir is not None:
    simD_disp_fname         = params.simD_Dir + 'displacement.npy'
    simD_time_fname         = params.simD_Dir + 'time.npy'        

  if params.simE_Dir is not None:
    simE_disp_fname         = params.simE_Dir + 'displacement.npy'
    simE_time_fname         = params.simE_Dir + 'time.npy'        

  disp_u_plot_fname = params.outputDir + params.filename.split('.')[0] + '-disp.' + params.filename.split('.')[1]

  print("\nLoading in data...")

  simA_Dsolve = np.load(simA_u_fname).T
  simA_tsolve = np.load(simA_time_fname).T

  params.simA_H0e = 1

  if 'Q2' in params.simA_Dir:

    simA_H1_Dloc = int(params.simA_probe_1*(2/params.simA_H0e))
    simA_H2_Dloc = int(params.simA_probe_2*(2/params.simA_H0e))
    simA_H3_Dloc = int(params.simA_probe_3*(2/params.simA_H0e))
    simA_H4_Dloc = int(params.simA_probe_4*(2/params.simA_H0e))

  elif 'Q1' in params.simA_Dir:

    simA_H1_Dloc = int(params.simA_probe_1*(1/params.simA_H0e))
    simA_H2_Dloc = int(params.simA_probe_2*(1/params.simA_H0e))
    simA_H3_Dloc = int(params.simA_probe_3*(1/params.simA_H0e))
    simA_H4_Dloc = int(params.simA_probe_4*(1/params.simA_H0e))

  if params.simB_Dir is not None:
    simB_Dsolve     = np.load(simB_disp_fname)
    simB_preSimData = readPreSimData(params, params.simB_Dir, params.simB_InputFileName)

    if params.simB_isPython:
      simB_tsolve    = np.load(simB_time_fname)
      simB_probeList = [params.simB_probe_1,params.simB_probe_2,params.simB_probe_3,params.simB_probe_4]
      simB_DDOFList  = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simB_probeList)):
        simB_DDOFList.append(getDisplacementDOF(params, simB_preSimData[1], simB_preSimData[3], simB_probeList[probeIndex]))

    else:
      simB_tsolve = np.linspace(0, simB_preSimData[0].TStop, simB_Dsolve.shape[0])
      simB_probeList  = [params.simB_probe_1,params.simB_probe_2,params.simB_probe_3,params.simB_probe_4]
      simB_DDOFList   = []
      #--------------------------------------------
      # Grab displacement DOFs for all four probes.
      #--------------------------------------------
      for probeIndex in range(len(simB_probeList)):
        simB_DDOFList.append(getDisplacementDOFDYNA(params, simB_probeList[probeIndex]))
      
  if params.simC_Dir is not None:
    simC_Dsolve = np.load(simC_disp_fname)
    simC_preSimData = readPreSimData(params, params.simC_Dir, params.simC_InputFileName)

    if params.simC_isPython:
      simC_tsolve    = np.load(simC_time_fname)
      simC_probeList = [params.simC_probe_1,params.simC_probe_2,params.simC_probe_3,params.simC_probe_4]
      simC_DDOFList  = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simC_probeList)):
        simC_DDOFList.append(getDisplacementDOF(params, simC_preSimData[1], simC_preSimData[3], simC_probeList[probeIndex]))

    else:
      simC_tsolve    = np.linspace(0, simC_preSimData[0].TStop, simC_Dsolve.shape[0])
      simC_probeList = [params.simC_probe_1,params.simC_probe_2,params.simC_probe_3,params.simC_probe_4]
      simC_DDOFList  = []
      #--------------------------------------------
      # Grab displacement DOFs for all four probes.
      #--------------------------------------------
      for probeIndex in range(len(simC_probeList)):
        simC_DDOFList.append(getDisplacementDOFDYNA(params, simC_probeList[probeIndex]))

  if params.simD_Dir is not None:
    simD_Dsolve = np.load(simD_disp_fname)

    simD_preSimData = readPreSimData(params, params.simD_Dir, params.simD_InputFileName)
    if params.simD_isPython:
      simD_tsolve    = np.load(simD_time_fname)
      simD_probeList = [params.simD_probe_1,params.simD_probe_2,params.simD_probe_3,params.simD_probe_4]
      simD_DDOFList  = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simD_probeList)):
        simD_DDOFList.append(getDisplacementDOF(params, simD_preSimData[1], simD_preSimData[3], simD_probeList[probeIndex]))

    else:
      simD_tsolve    = np.linspace(0, simD_preSimData[0].TStop, simD_Dsolve.shape[0])
      simD_probeList = [params.simD_probe_1,params.simD_probe_2,params.simD_probe_3,params.simD_probe_4]
      simD_DDOFList  = []
      #--------------------------------------------
      # Grab displacement DOFs for all four probes.
      #--------------------------------------------
      for probeIndex in range(len(simD_probeList)):
        simD_DDOFList.append(getDisplacementDOFDYNA(params, simD_probeList[probeIndex]))

  if params.simE_Dir is not None:
    simE_Dsolve = np.load(simE_disp_fname)
    simE_preSimData = readPreSimData(params, params.simE_Dir, params.simE_InputFileName)
    if params.simE_isPython:
      simE_tsolve    = np.load(simE_time_fname)
      simE_probeList = [params.simE_probe_1,params.simE_probe_2,params.simE_probe_3,params.simE_probe_4]
      simE_DDOFList  = []
      #-------------------------------
      # Grab DOFs for all four probes.
      #-------------------------------
      for probeIndex in range(len(simE_probeList)):
        simE_DDOFList.append(getDisplacementDOF(params, simE_preSimData[1], simE_preSimData[3], simE_probeList[probeIndex]))

    else:
      simE_tsolve    = np.linspace(0, simE_preSimData[0].TStop, simE_Dsolve.shape[0])
      simE_probeList = [params.simE_probe_1,params.simE_probe_2,params.simE_probe_3,params.simE_probe_4]
      simE_DDOFList  = []
      #--------------------------------------------
      # Grab displacement DOFs for all four probes.
      #--------------------------------------------
      for probeIndex in range(len(simE_probeList)):
        simE_DDOFList.append(getDisplacementDOFDYNA(simE_preSimData[3], simE_probeList[probeIndex]))

  print("Finished loading in data.")
  print("Generating plots...")
  #-------------------------
  # Plot solid displacement.
  #-------------------------
  plt.figure(1)
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip, simA_H4_Dloc]*params.dispScaling, 'k-', label=r'$X = $' + ' ' + str(params.simA_probe_4) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip, simA_H3_Dloc]*params.dispScaling, 'b-', label=r'$X = $' + ' ' + str(params.simA_probe_3) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip, simA_H2_Dloc]*params.dispScaling, 'r-', label=r'$X = $' + ' ' + str(params.simA_probe_2) + 'm')
  plt.plot(simA_tsolve[::params.simA_Skip]*params.timeScaling, simA_Dsolve[::params.simA_Skip, simA_H1_Dloc]*params.dispScaling, 'm-', label=r'$X = $' + ' ' + str(params.simA_probe_1) + 'm')

  if params.simB_Dir is not None:
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[0]]*params.dispScaling, 'mo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[1]]*params.dispScaling, 'ro', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[2]]*params.dispScaling, 'bo', fillstyle='none')
    plt.plot(simB_tsolve[::params.simB_Skip]*params.timeScaling, simB_Dsolve[::params.simB_Skip,simB_DDOFList[3]]*params.dispScaling, 'ko', fillstyle='none')

  if params.simC_Dir is not None:
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[0]]*params.dispScaling, 'mx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[1]]*params.dispScaling, 'rx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[2]]*params.dispScaling, 'bx', fillstyle='none')
    plt.plot(simC_tsolve[::params.simC_Skip]*params.timeScaling, simC_Dsolve[::params.simC_Skip,simC_DDOFList[3]]*params.dispScaling, 'kx', fillstyle='none')

  if params.simD_Dir is not None:
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[0]]*params.dispScaling, 'm+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[1]]*params.dispScaling, 'r+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[2]]*params.dispScaling, 'b+', fillstyle='none')
    plt.plot(simD_tsolve[::params.simD_Skip]*params.timeScaling, simD_Dsolve[::params.simD_Skip,simD_DDOFList[3]]*params.dispScaling, 'k+', fillstyle='none')

  if params.simE_Dir is not None:
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[0]]*params.dispScaling, 'm^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[1]]*params.dispScaling, 'r^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[2]]*params.dispScaling, 'b^', fillstyle='none')
    plt.plot(simE_tsolve[::params.simE_Skip]*params.timeScaling, simE_Dsolve[::params.simE_Skip,simE_DDOFList[3]]*params.dispScaling, 'k^', fillstyle='none')

  plt.xlim([0, 0.4])
  plt.ylim([-15, 5])
  if params.legend:
    plt.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, fontsize=params.legendFontSize, handlelength=1.0, edgecolor='k', framealpha=1.0)

  if params.text:
    if params.simE_Dir is not None:
      plt.text(0.3, 0.723, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title + '\n' + r'$\triangle$ ' + params.simE_Title, \
               fontsize=params.legendFontSize, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simD_Dir is not None:
      plt.text(0.297, 0.753, r'-- ' + params.simA_Title + '\n' + r'$\circ$ ' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title + '\n' + r'$+$ ' + params.simD_Title, \
               fontsize=params.legendFontSize, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    elif params.simC_Dir is not None:
      plt.text(0.297, 0.788, r'-- ' + params.simA_Title + '\n' + r'$\circ$' + params.simB_Title + '\n' + r'$\times$ ' + params.simC_Title, \
               fontsize=params.legendFontSize, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
    else:
      plt.text(0.297, 0.813, r'-- ' + params.simA_Title + '\n' + r'$\circ$' + params.simB_Title, \
                 fontsize=params.legendFontSize, transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', pad=3.0, alpha=1.0))
  plt.xlabel(r'Time ' + a_TimeDict[params.timeScaling], fontsize=params.xAxisFontSize)
  plt.ylabel(r'Displacement ' + a_DispDict[params.dispScaling], fontsize=params.yAxisFontSize)
  if params.title:
    plt.title(params.titleName)
  if params.grid:
    plt.grid()
  plt.savefig(disp_u_plot_fname, bbox_inches='tight', dpi=300)

  # errorB = np.abs((simA_Dsolve[52,simA_H1_Dloc] - simB_Dsolve[51,simB_DDOFList[0]]))/np.abs(simA_Dsolve[52,simA_H1_Dloc])
  # print(errorB)
  # if params.simC_Dir is not None:
  #   errorC = np.abs((simA_Dsolve[52,simA_H1_Dloc] - simC_Dsolve[51,simC_DDOFList[0]]))/np.abs(simA_Dsolve[52,simA_H1_Dloc])
  #   print(errorC)
  # if params.simD_Dir is not None:
  #   errorD = np.abs((simA_Dsolve[52,simA_H1_Dloc] - simD_Dsolve[51,simD_DDOFList[0]]))/np.abs(simA_Dsolve[52,simA_H1_Dloc])
  #   print(errorD)
  # if params.simE_Dir is not None:
  #   errorE = np.linalg.norm((simA_Dsolve[52,simA_H1_Dloc] - simE_Dsolve[51,simE_DDOFList[0]]))/np.abs(simA_Dsolve[52,simA_H1_Dloc])
  #   print(errorE)

  return    

