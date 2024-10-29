#--------------------------------------------------------------------------------------------------
# Plotting script for MMS errors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 26, 2024
#--------------------------------------------------------------------------------------------------
try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
  from meshInfo import *
except ImportError:
  sys.exit("MODULE WARNING. meshInfo not found, check configuration.")

#--------------------------------------------------------------------------------------------------
#------------
# Arguments:
#------------
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(params, a_DispDict, a_StressDict):

  print("\nGenerating plots...")
  #-------------
  # Plot errors.
  #-------------
  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)

  if params.temporal:

    t_plot  = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    dataFN  = 'summit-input-MMS-C-T.dat'
    start   = 1
    if 'testingPackage' in params.simA_Dir:
      t_plot  = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
      dataFN = 'input-MMS.dat'
      start -= 1

    D_err   = []
    V_err   = []
    A_err   = []
    P_err   = []
    PD_err  = []
    DF_err  = []
    VF_err  = []
    AF_err  = []

    for i in range(start,len(t_plot) + start):
      dataDir = params.simA_Dir + 'L' + str(i) + params.mms_suffix
      
      preSimData = readPreSimData(params, sim_Dir=dataDir, sim_InputFileName=dataFN)
      simParams  = preSimData[0]
      if simParams.Element_Type.split('-')[0] == 'Q3H':
        simCoordsD = np.linspace(0, 1, 2)
      else:
        simCoordsD = np.linspace(0, 1, simParams.nNodeS)

      simCoordsP = np.linspace(0, 1, 2)

      try:
        if simParams.Element_Type.split('-')[1] == 'Q3H':
          simCoordsDF = np.linspace(0, 1, 2)
        else:
          simCoordsDF = np.linspace(0, 1, simParams.nNodeF)
      except IndexError:
        pass

      D = np.load(dataDir + 'displacement.npy')
      V = np.load(dataDir + 'velocity.npy')
      A = np.load(dataDir + 'acceleration.npy')
      T = np.load(dataDir + 'time.npy')

      if simParams.MMS_SolidSolutionType == 'S2T2':
        REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**2
        REF_V  = 2*simCoordsD**2 * T[params.mms_time_id]
        REF_A  = 2*simCoordsD**2
      elif simParams.MMS_SolidSolutionType == 'S2T3':
        REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**3
        REF_V  = 3*simCoordsD**2 * T[params.mms_time_id]**2
        REF_A  = 6*simCoordsD**2 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'MS2T3':
        REF_D  = -1*simCoordsD**2 * T[params.mms_time_id]**3
        REF_V  = -3*simCoordsD**2 * T[params.mms_time_id]**2
        REF_A  = -6*simCoordsD**2 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'S3T3':
        REF_D  = 1*simCoordsD**3 * T[params.mms_time_id]**3
        REF_V  = 3*simCoordsD**3 * T[params.mms_time_id]**2
        REF_A  = 6*simCoordsD**3 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'MS3T3':
        REF_D  = -1*simCoordsD**3 * T[params.mms_time_id]**3
        REF_V  = -3*simCoordsD**3 * T[params.mms_time_id]**2
        REF_A  = -6*simCoordsD**3 * T[params.mms_time_id]

      if simParams.Physics != 'u':
        if simParams.MMS_PressureSolutionType == 'S1T1':
          REF_P  = (simParams.H0 - simCoordsP) * T[params.mms_time_id]
          REF_PD = (simParams.H0 - simCoordsP)
        elif simParams.MMS_PressureSolutionType == 'S1T2':
          REF_P  = 1*(simParams.H0 - simCoordsP) * T[params.mms_time_id]**2
          REF_PD = 2*(simParams.H0 - simCoordsP) * T[params.mms_time_id]

        if simParams.Physics == 'u-uf-pf':
          if simParams.MMS_FluidSolutionType == 'S2T3':
            REF_DF  = 0.5*simCoordsDF**2 * T[params.mms_time_id]**3
            REF_VF  = 1.5*simCoordsDF**2 * T[params.mms_time_id]**2
            REF_AF  = 3.0*simCoordsDF**2 * T[params.mms_time_id]
          elif simParams.MMS_FluidSolutionType == 'S3T3':
            REF_DF  = 0.5*simCoordsDF**3 * T[params.mms_time_id]**3
            REF_VF  = 1.5*simCoordsDF**3 * T[params.mms_time_id]**2
            REF_AF  = 3.0*simCoordsDF**3 * T[params.mms_time_id]

      S_start = 0
      if simParams.Element_Type.split('-')[0] == 'Q3H':
        S_stop  = simParams.nNodeS - 1
        S_step  = 2
      else:
        S_stop  = simParams.nNodeS
        S_step  = 1

      D_at_T  = D[params.mms_time_id, S_start:S_stop:S_step]
      V_at_T  = V[params.mms_time_id, S_start:S_stop:S_step]
      A_at_T  = A[params.mms_time_id, S_start:S_stop:S_step]

      if simParams.Physics == 'u-pf':
        P_start = simParams.nNodeS

        P_at_T  = D[params.mms_time_id, P_start:]
        PD_at_T = V[params.mms_time_id, P_start:]
      elif simParams.Physics == 'u-uf-pf':
        P_start = simParams.nNodeS + simParams.nNodeF
        F_start = simParams.nNodeS
        F_stop  = simParams.nNodeS + simParams.nNodeF
        F_step  = 1
        if simParams.Element_Type.split('-')[1] == 'Q3H':
          F_stop -= 1
          F_step += 1

        DF_at_T  = D[params.mms_time_id, F_start:F_stop:F_step]
        VF_at_T  = V[params.mms_time_id, F_start:F_stop:F_step]
        AF_at_T  = A[params.mms_time_id, F_start:F_stop:F_step]
        P_at_T   = D[params.mms_time_id, P_start:]
        PD_at_T  = V[params.mms_time_id, P_start:]

      if params.normOrd == 'inf':
        normOrdStr = '$_{L_\infty}$'
        D_err.append(np.max(np.abs(D_at_T - REF_D)))
        V_err.append(np.max(np.abs(V_at_T - REF_V)))
        A_err.append(np.max(np.abs(A_at_T - REF_A)))
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          P_err.append(np.max(np.abs(P_at_T - REF_P)))
          PD_err.append(np.max(np.abs(PD_at_T - REF_PD)))
        if simParams.Physics == 'u-uf-pf':
          DF_err.append(np.max(np.abs(DF_at_T - REF_DF)))
          VF_err.append(np.max(np.abs(VF_at_T - REF_VF)))
          AF_err.append(np.max(np.abs(AF_at_T - REF_AF)))

      elif params.normOrd == '2':
        normOrdStr = '$_{L_2}$'
        D_err.append(np.sqrt(np.sum((D_at_T - REF_D)**2))/REF_D.shape[0])
        V_err.append(np.sqrt(np.sum((V_at_T - REF_V)**2))/REF_V.shape[0])
        A_err.append(np.sqrt(np.sum((A_at_T - REF_A)**2))/REF_A.shape[0])
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          P_err.append(np.sqrt(np.sum((P_at_T - REF_P)**2))/REF_P.shape[0])
          PD_err.append(np.sqrt(np.sum((PD_at_T - REF_PD)**2))/REF_PD.shape[0])
        if simParams.Physics == 'u-uf-pf':
          DF_err.append(np.sqrt(np.sum((DF_at_T - REF_DF)**2))/REF_DF.shape[0])
          VF_err.append(np.sqrt(np.sum((VF_at_T - REF_VF)**2))/REF_VF.shape[0])
          AF_err.append(np.sqrt(np.sum((AF_at_T - REF_AF)**2))/REF_AF.shape[0])
    plt.loglog(t_plot, D_err, '-x', label=r'$\lvert\lvert u - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    plt.loglog(t_plot, V_err, '-o', label=r'$\lvert\lvert v - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    plt.loglog(t_plot, A_err, '-s', label=r'$\lvert\lvert a - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
      plt.loglog(t_plot, P_err, '-^', label=r'$\lvert\lvert p_\rf - ref.\rvert\rvert$' + normOrdStr,       color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(t_plot, PD_err, '-v', label=r'$\lvert\lvert \dot{p}_\rf - ref.\rvert\rvert$' + normOrdStr, color=params.simA_Color_Alpha, fillstyle='none')
    if simParams.Physics == 'u-uf-pf':
      plt.loglog(t_plot, DF_err, '-<', label=r'$\lvert\lvert u_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(t_plot, VF_err, '->', label=r'$\lvert\lvert v_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(t_plot, AF_err, '-8', label=r'$\lvert\lvert a_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')

    plt.xlabel(r'$\Delta t$ (s)', fontsize=16)
    plt.ylabel(r'Error norm', fontsize=16)

  if params.spatial:
    nel_list = [5, 10, 20, 40, 80, 160, 320, 640]
    el_plot  = [1/5, 1/10, 1/20, 1/40, 1/80, 1/160, 1/320, 1/640]

    D_err  = []
    V_err  = []
    A_err  = []
    P_err  = []
    PD_err = [] 
    DF_err  = []
    VF_err  = []
    AF_err  = []

    for i in range(1,9):

      nel = nel_list[i-1]

      dataDir = params.simA_Dir + 'L' + str(i) + params.mms_suffix
      dataFN  = 'summit-input-MMS-C-S.dat'

      preSimData = readPreSimData(params, sim_Dir=dataDir, sim_InputFileName=dataFN)
      simParams  = preSimData[0]
      if simParams.Element_Type.split('-')[0] == 'Q3H':
        simCoordsD = np.linspace(0, 1, int(simParams.nNodeS/2))
      else:
        simCoordsD = np.linspace(0, 1, simParams.nNodeS)

      simCoordsP = np.linspace(0, 1, simParams.nNodeP)

      try:
        if simParams.Element_Type.split('-')[1] == 'Q3H':
          simCoordsDF = np.linspace(0, 1, int(simParams.nNodeF/2))
        else:
          simCoordsDF = np.linspace(0, 1, simParams.nNodeF)
      except IndexError:
        pass

      D = np.load(dataDir + 'displacement.npy')
      V = np.load(dataDir + 'velocity.npy')
      A = np.load(dataDir + 'acceleration.npy')
      T = np.load(dataDir + 'time.npy')

      if simParams.MMS_SolidSolutionType == 'S2T2':
        REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**2
        REF_V  = 2*simCoordsD**2 * T[params.mms_time_id]
        REF_A  = 2*simCoordsD**2
      elif simParams.MMS_SolidSolutionType == 'S2T3':
        REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**3
        REF_V  = 3*simCoordsD**2 * T[params.mms_time_id]**2
        REF_A  = 6*simCoordsD**2 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'MS2T3':
        REF_D  = -1*simCoordsD**2 * T[params.mms_time_id]**3
        REF_V  = -3*simCoordsD**2 * T[params.mms_time_id]**2
        REF_A  = -6*simCoordsD**2 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'S3T3':
        REF_D  = 1*simCoordsD**3 * T[params.mms_time_id]**3
        REF_V  = 3*simCoordsD**3 * T[params.mms_time_id]**2
        REF_A  = 6*simCoordsD**3 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'MS3T3':
        REF_D  = -1*simCoordsD**3 * T[params.mms_time_id]**3
        REF_V  = -3*simCoordsD**3 * T[params.mms_time_id]**2
        REF_A  = -6*simCoordsD**3 * T[params.mms_time_id]
      elif simParams.MMS_SolidSolutionType == 'S4T3':
        REF_D  = 1*simCoordsD**4 * T[params.mms_time_id]**3
        REF_V  = 3*simCoordsD**4 * T[params.mms_time_id]**2
        REF_A  = 6*simCoordsD**4 * T[params.mms_time_id]

      if simParams.Physics != 'u':
        if simParams.MMS_PressureSolutionType == 'S1T1':
          REF_P  = (simParams.H0 - simCoordsP) * T[params.mms_time_id]
          REF_PD = (simParams.H0 - simCoordsP)
        elif simParams.MMS_PressureSolutionType == 'S1T2':
          REF_P  = 1*(simParams.H0 - simCoordsP) * T[params.mms_time_id]**2
          REF_PD = 2*(simParams.H0 - simCoordsP) * T[params.mms_time_id]

        if simParams.Physics == 'u-uf-pf':
          if simParams.MMS_FluidSolutionType == 'S2T3':
            REF_DF  = 0.5*simCoordsDF**2 * T[params.mms_time_id]**3
            REF_VF  = 1.5*simCoordsDF**2 * T[params.mms_time_id]**2
            REF_AF  = 3.0*simCoordsDF**2 * T[params.mms_time_id]
          elif simParams.MMS_FluidSolutionType == 'S3T3':
            REF_DF  = 0.5*simCoordsDF**3 * T[params.mms_time_id]**3
            REF_VF  = 1.5*simCoordsDF**3 * T[params.mms_time_id]**2
            REF_AF  = 3.0*simCoordsDF**3 * T[params.mms_time_id]
          elif simParams.MMS_FluidSolutionType == 'MS3T3':
            REF_DF  = -0.5*simCoordsDF**3 * T[params.mms_time_id]**3
            REF_VF  = -1.5*simCoordsDF**3 * T[params.mms_time_id]**2
            REF_AF  = -3.0*simCoordsDF**3 * T[params.mms_time_id]
          elif simParams.MMS_FluidSolutionType == 'S4T3':
            REF_DF  = 0.5*simCoordsD**4 * T[params.mms_time_id]**3
            REF_VF  = 1.5*simCoordsD**4 * T[params.mms_time_id]**2
            REF_AF  = 3.0*simCoordsD**4 * T[params.mms_time_id]
          elif simParams.MMS_FluidSolutionType == 'MS4T3':
            REF_DF  = -0.5*simCoordsD**4 * T[params.mms_time_id]**3
            REF_VF  = -1.5*simCoordsD**4 * T[params.mms_time_id]**2
            REF_AF  = -3.0*simCoordsD**4 * T[params.mms_time_id]

      S_start = 0
      S_stop  = simParams.nNodeS
      S_step  = 1
      if simParams.Element_Type.split('-')[0] == 'Q3H':
        S_stop -= 1
        S_step += 1

      D_at_T  = D[params.mms_time_id, S_start:S_stop:S_step]
      V_at_T  = V[params.mms_time_id, S_start:S_stop:S_step]
      A_at_T  = A[params.mms_time_id, S_start:S_stop:S_step]

      if simParams.Physics == 'u-pf':
        P_start = simParams.nNodeS

        P_at_T  = D[params.mms_time_id, P_start:]
        PD_at_T = V[params.mms_time_id, P_start:]
      elif simParams.Physics == 'u-uf-pf':
        P_start = simParams.nNodeS + simParams.nNodeF
        F_start = simParams.nNodeS
        F_stop  = simParams.nNodeS + simParams.nNodeF
        F_step  = 1
        if simParams.Element_Type.split('-')[1] == 'Q3H':
          F_stop -= 1
          F_step += 1

        DF_at_T  = D[params.mms_time_id, F_start:F_stop:F_step]
        VF_at_T  = V[params.mms_time_id, F_start:F_stop:F_step]
        AF_at_T  = A[params.mms_time_id, F_start:F_stop:F_step]
        P_at_T   = D[params.mms_time_id, P_start:]
        PD_at_T  = V[params.mms_time_id, P_start:]

      if params.normOrd == 'inf':
        normOrdStr = '$_{L_\infty}$'
        D_err.append(np.max(np.abs(D_at_T - REF_D)))
        V_err.append(np.max(np.abs(V_at_T - REF_V)))
        A_err.append(np.max(np.abs(A_at_T - REF_A)))
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          P_err.append(np.max(np.abs(P_at_T - REF_P)))
          PD_err.append(np.max(np.abs(PD_at_T - REF_PD)))
        if simParams.Physics == 'u-uf-pf':
          DF_err.append(np.max(np.abs(DF_at_T - REF_DF)))
          VF_err.append(np.max(np.abs(VF_at_T - REF_VF)))
          AF_err.append(np.max(np.abs(AF_at_T - REF_AF)))

      elif params.normOrd == '2':
        normOrdStr = '$_{L_2}$'
        D_err.append(np.sqrt(np.sum((D_at_T - REF_D)**2))/REF_D.shape[0])
        V_err.append(np.sqrt(np.sum((V_at_T - REF_V)**2))/REF_V.shape[0])
        A_err.append(np.sqrt(np.sum((A_at_T - REF_A)**2))/REF_A.shape[0])
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          P_err.append(np.sqrt(np.sum((P_at_T - REF_P)**2))/REF_P.shape[0])
          PD_err.append(np.sqrt(np.sum((PD_at_T - REF_PD)**2))/REF_PD.shape[0])
        if simParams.Physics == 'u-uf-pf':
          DF_err.append(np.sqrt(np.sum((DF_at_T - REF_DF)**2))/REF_DF.shape[0])
          VF_err.append(np.sqrt(np.sum((VF_at_T - REF_VF)**2))/REF_VF.shape[0])
          AF_err.append(np.sqrt(np.sum((AF_at_T - REF_AF)**2))/REF_AF.shape[0])
    
    plt.loglog(el_plot, D_err, '-x', label=r'$\lvert\lvert u - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    plt.loglog(el_plot, V_err, '-o', label=r'$\lvert\lvert v - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    plt.loglog(el_plot, A_err, '-s', label=r'$\lvert\lvert a - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
      plt.loglog(el_plot, P_err, '-^', label=r'$\lvert\lvert p_\rf - ref.\rvert\rvert$' + normOrdStr,       color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(el_plot, PD_err, '-v', label=r'$\lvert\lvert \dot{p}_\rf - ref.\rvert\rvert$' + normOrdStr, color=params.simA_Color_Alpha, fillstyle='none')
    if simParams.Physics == 'u-uf-pf':
      plt.loglog(el_plot, DF_err, '-<', label=r'$\lvert\lvert u_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(el_plot, VF_err, '->', label=r'$\lvert\lvert v_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
      plt.loglog(el_plot, AF_err, '-8', label=r'$\lvert\lvert a_\rf - ref.\rvert\rvert$' + normOrdStr,           color=params.simA_Color_Alpha, fillstyle='none')
    
    # plt.semilogy(nel_list, D_err2, '-x', label=r'$\lvert\lvert u - ref.\rvert\rvert_{L_\infty}$',           color=params.simA_Color, fillstyle='none')
    # plt.semilogy(nel_list, V_err2, '-o', label=r'$\lvert\lvert v - ref.\rvert\rvert_{L_\infty}$',           color=params.simA_Color, fillstyle='none')
    # plt.semilogy(nel_list, A_err2, '-s', label=r'$\lvert\lvert a - ref.\rvert\rvert_{L_\infty}$',           color=params.simA_Color, fillstyle='none')
    # try:
    #   plt.semilogy(nel_list, P_err2, '-^', label=r'$\lvert\lvert p_\rf - ref.\rvert\rvert_{L_\infty}$',       color=params.simA_Color, fillstyle='none')
    #   plt.semilogy(nel_list, D_err2, '-v', label=r'$\lvert\lvert \dot{p}_\rf - ref.\rvert\rvert_{L_\infty}$', color=params.simA_Color, fillstyle='none')
    # except ValueError:
    #   pass

    # plt.xlabel(r'\# elements/meter', fontsize=16)
    plt.xlabel(r'$h_0^e$ (m)', fontsize=16)
    plt.ylabel(r'Error norm', fontsize=16)

    end = -1

    slope_D  = np.log(D_err[end] /D_err[0] )/np.log(el_plot[end]/el_plot[0])
    try:
      slope_DF = np.log(DF_err[end]/DF_err[0])/np.log(el_plot[end]/el_plot[0])
    except:
      try:
        slope_P  = np.log(P_err[end] /P_err[0] )/np.log(el_plot[end]/el_plot[0])
      except:
        pass
    try:
      print("u slope: ", slope_D, "\n uf slope: ", slope_DF, "\n pf slope: ", slope_P)
    except:
      try:
        print("u slope: ", slope_D, "\n pf slope: ", slope_P)
      except:
        print("u slope: ", slope_D)

  if not params.solution:
    if params.ylim0 is not None and params.ylim1 is not None:
      plt.ylim([params.ylim0, params.ylim1])
    if params.xlim0 is not None and params.xlim1 is not None:
      plt.xlim([params.xlim0, params.xlim1])

    if params.is_xticks:
      ax1.tick_params(direction="in", which='both')
      ax1.set_xticks(params.xticks)
    if params.secondaryXTicks:
      ax2 = ax1.secondary_xaxis('top') 
      ax2.set_xticklabels([])
      ax2.tick_params(direction="in")
    if params.is_xticklabels:
      ax1.set_xticklabels(params.xticklabels)

    if params.is_yticks:
      ax1.set_yticks(params.yticks)
    if params.secondaryYTicks:
      ax3 = ax1.secondary_yaxis('right')
      ax3.set_yticklabels([])
      ax3.tick_params(direction="in", which='both') 
    if params.is_yticklabels:
      ax1.set_yticklabels(params.yticklabels)

    if params.grid:
      plt.grid(True, which=params.gridWhich)
      if not params.solution:
        ax1.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
        ax1.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9)))
    if params.legend:
      handles, labels = ax1.get_legend_handles_labels()
      if len(handles) > 5:
        handles.insert(5, plt.Line2D(np.ones(1),np.ones(1), linestyle='None', marker='None'))
        labels.insert(5,'')
        handles         = np.concatenate((handles[0:3],handles[6:9],handles[3:6]),axis=0)
        labels          = np.concatenate((labels[0:3],labels[6:9],labels[3:6]),axis=0)
        plt.legend(handles, labels, ncol=3, bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
      else:
        if len(handles) > 3:
          plt.legend(handles, labels, ncol=2, bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
        else:
          plt.legend(handles, labels, bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)

    if params.title:
      fig1.suptitle(params.titleName,fontsize=18)

    plt.savefig(params.outputDir + params.filename, bbox_inches='tight', dpi=params.DPI)
    plt.close()

  if params.solution:

    preSimData = readPreSimData(params, sim_Dir=params.simA_Dir, sim_InputFileName=params.simA_InputFileName)
    simParams  = preSimData[0]
    if simParams.Element_Type.split('-')[0] == 'Q3H':
      simCoordsD = np.linspace(0, 1, int(simParams.nNodeS/2))
    else:
      simCoordsD = np.linspace(0, 1, simParams.nNodeS)

    simCoordsP = np.linspace(0, 1, simParams.nNodeP)

    try:
      if simParams.Element_Type.split('-')[1] == 'Q3H':
        simCoordsDF = np.linspace(0, 1, int(simParams.nNodeF/2))
      else:
        simCoordsDF = np.linspace(0, 1, simParams.nNodeF)
    except IndexError:
      pass

    D = np.load(params.simA_Dir + 'displacement.npy')
    V = np.load(params.simA_Dir + 'velocity.npy')
    A = np.load(params.simA_Dir + 'acceleration.npy')
    T = np.load(params.simA_Dir + 'time.npy')

    if simParams.MMS_SolidSolutionType == 'S2T2':
      REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**2
      REF_V  = 2*simCoordsD**2 * T[params.mms_time_id]
      REF_A  = 2*simCoordsD**2
    elif simParams.MMS_SolidSolutionType == 'S2T3':
      REF_D  = 1*simCoordsD**2 * T[params.mms_time_id]**3
      REF_V  = 3*simCoordsD**2 * T[params.mms_time_id]**2
      REF_A  = 6*simCoordsD**2 * T[params.mms_time_id]
    elif simParams.MMS_SolidSolutionType == 'MS2T3':
      REF_D  = -1*simCoordsD**2 * T[params.mms_time_id]**3
      REF_V  = -3*simCoordsD**2 * T[params.mms_time_id]**2
      REF_A  = -6*simCoordsD**2 * T[params.mms_time_id]
    elif simParams.MMS_SolidSolutionType == 'S3T3':
      REF_D  = 1*simCoordsD**3 * T[params.mms_time_id]**3
      REF_V  = 3*simCoordsD**3 * T[params.mms_time_id]**2
      REF_A  = 6*simCoordsD**3 * T[params.mms_time_id]
    elif simParams.MMS_SolidSolutionType == 'MS3T3':
      REF_D  = -1*simCoordsD**3 * T[params.mms_time_id]**3
      REF_V  = -3*simCoordsD**3 * T[params.mms_time_id]**2
      REF_A  = -6*simCoordsD**3 * T[params.mms_time_id]
    elif simParams.MMS_SolidSolutionType == 'S4T3':
      REF_D  = 1*simCoordsD**4 * T[params.mms_time_id]**3
      REF_V  = 3*simCoordsD**4 * T[params.mms_time_id]**2
      REF_A  = 6*simCoordsD**4 * T[params.mms_time_id]

    if simParams.Physics != 'u':
      if simParams.MMS_PressureSolutionType == 'S1T1':
        REF_P   = (simParams.H0 - simCoordsP) * T[params.mms_time_id]
        REF_PD  = (simParams.H0 - simCoordsP)
      elif simParams.MMS_PressureSolutionType == 'S1T2':
        REF_P   = 1*(simParams.H0 - simCoordsP) * T[params.mms_time_id]**2
        REF_PD  = 2*(simParams.H0 - simCoordsP) * T[params.mms_time_id]
      elif simParams.MMS_PressureSolutionType == 'S1T3':
        REF_P   = 1*(simParams.H0 - simCoordsP) * T[params.mms_time_id]**3
        REF_PD  = 3*(simParams.H0 - simCoordsP) * T[params.mms_time_id]**2
      elif simParams.MMS_PressureSolutionType == '2S1T3':
        REF_P   = 1*(2*simParams.H0 - simCoordsP) * T[params.mms_time_id]**3
        REF_PD  = 3*(2*simParams.H0 - simCoordsP) * T[params.mms_time_id]**2

      if simParams.Physics == 'u-uf-pf':
        if simParams.MMS_FluidSolutionType == 'S2T3':
          REF_DF  = 0.5*simCoordsDF**2 * T[params.mms_time_id]**3
          REF_VF  = 1.5*simCoordsDF**2 * T[params.mms_time_id]**2
          REF_AF  = 3.0*simCoordsDF**2 * T[params.mms_time_id]
        elif simParams.MMS_FluidSolutionType == 'S3T3':
          REF_DF  = 0.5*simCoordsDF**3 * T[params.mms_time_id]**3
          REF_VF  = 1.5*simCoordsDF**3 * T[params.mms_time_id]**2
          REF_AF  = 3.0*simCoordsDF**3 * T[params.mms_time_id]
        elif simParams.MMS_FluidSolutionType == 'MS3T3':
          REF_DF  = -0.5*simCoordsDF**3 * T[params.mms_time_id]**3
          REF_VF  = -1.5*simCoordsDF**3 * T[params.mms_time_id]**2
          REF_AF  = -3.0*simCoordsDF**3 * T[params.mms_time_id]
        elif simParams.MMS_FluidSolutionType == 'S4T3':
          REF_DF  = 0.5*simCoordsD**4 * T[params.mms_time_id]**3
          REF_VF  = 1.5*simCoordsD**4 * T[params.mms_time_id]**2
          REF_AF  = 3.0*simCoordsD**4 * T[params.mms_time_id]
        elif simParams.MMS_FluidSolutionType == 'MS4T3':
          REF_DF  = -0.5*simCoordsD**4 * T[params.mms_time_id]**3
          REF_VF  = -1.5*simCoordsD**4 * T[params.mms_time_id]**2
          REF_AF  = -3.0*simCoordsD**4 * T[params.mms_time_id]

    S_start = 0
    S_stop  = simParams.nNodeS
    S_step  = 1
    if simParams.Element_Type.split('-')[0] == 'Q3H':
      S_stop -= 1
      S_step += 1

    D_at_T  = D[params.mms_time_id, S_start:S_stop:S_step]
    V_at_T  = V[params.mms_time_id, S_start:S_stop:S_step]
    A_at_T  = A[params.mms_time_id, S_start:S_stop:S_step]

    if simParams.Physics == 'u-pf':
      P_start = simParams.nNodeS

      P_at_T   = D[params.mms_time_id, P_start:]
      PD_at_T  = V[params.mms_time_id, P_start:]
    elif simParams.Physics == 'u-uf-pf':
      P_start = simParams.nNodeS + simParams.nNodeF
      F_start = simParams.nNodeS
      F_stop  = simParams.nNodeS + simParams.nNodeF
      F_step  = 1
      if simParams.Element_Type.split('-')[1] == 'Q3H':
        F_stop  -= 1
        F_step  += 1

      DF_at_T  = D[params.mms_time_id, F_start:F_stop:F_step]
      VF_at_T  = V[params.mms_time_id, F_start:F_stop:F_step]
      AF_at_T  = A[params.mms_time_id, F_start:F_stop:F_step]
      P_at_T   = D[params.mms_time_id, P_start:]
      PD_at_T  = V[params.mms_time_id, P_start:]
    
    for plot_type_id in range(3):
      fig, ax1 = plt.subplots()
      ax2      = ax1.twinx()

      if plot_type_id == 0:
        ax1.plot(simCoordsD, REF_D*params.dispScaling,  'k-', label=r'$u(X,t)$ analytical')
        ax1.plot(simCoordsD, D_at_T*params.dispScaling, 'kx', label=r'$u(X,t)$ numerical')
        ax1.set_ylabel('Displacement ' + a_DispDict[params.dispScaling])
        
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          ax2.plot(simCoordsP, REF_P*params.stressScaling,  'r-', label=r'$p_\rf(X,t)$ analytical')
          ax2.plot(simCoordsP, P_at_T*params.stressScaling, 'rx', label=r'$p_\rf(X,t)$ numerical')
          ax2.set_ylabel('Pressure ' + a_StressDict[params.stressScaling])
        
        if simParams.Physics == 'u-uf-pf':
          ax1.plot(simCoordsDF, REF_DF*params.dispScaling,  'b-', label=r'$u_\rf(X,t)$ analytical')
          ax1.plot(simCoordsDF, DF_at_T*params.dispScaling, 'bx', label=r'$u_\rf(X,t)$ numerical')

      elif plot_type_id == 1:
        ax1.plot(simCoordsD, REF_V*params.dispScaling,  'k-', label=r'$v(X,t)$ analytical')
        ax1.plot(simCoordsD, V_at_T*params.dispScaling, 'kx', label=r'$v(X,t)$ numerical')
        ax1.set_ylabel('Velocity (m/s)')
        
        if simParams.Physics == 'u-pf' or simParams.Physics == 'u-uf-pf':
          ax2.plot(simCoordsP, REF_PD,  'r-', label=r'$\dot{p}_\rf(X,t)$ analytical')
          ax2.plot(simCoordsP, PD_at_T, 'rx', label=r'$\dot{p}_\rf(X,t)$ numerical')
          ax2.set_ylabel('Pressure rate (Pa/s)')
        
        if simParams.Physics == 'u-uf-pf':
          ax1.plot(simCoordsDF, REF_VF*params.dispScaling,  'b-', label=r'$v_\rf(X,t)$ analytical')
          ax1.plot(simCoordsDF, VF_at_T*params.dispScaling, 'bx', label=r'$v_\rf(X,t)$ numerical')
        
      elif plot_type_id == 2: 
        ax1.plot(simCoordsD, REF_A,  'k-', label=r'$a(X,t)$ analytical')
        ax1.plot(simCoordsD, A_at_T, 'kx', label=r'$a(X,t)$ numerical')
        ax1.set_ylabel('Acceleration (m/s$^2$)')

        if simParams.Physics == 'u-uf-pf':
          ax1.plot(simCoordsDF, REF_AF,  'b-', label=r'$a_\rf(X,t)$ analytical')
          ax1.plot(simCoordsDF, AF_at_T, 'bx', label=r'$a_\rf(X,t)$ numerical')

      if params.ylim0 is not None and params.ylim1 is not None:
        plt.ylim([params.ylim0, params.ylim1])
      if params.xlim0 is not None and params.xlim1 is not None:
        plt.xlim([params.xlim0, params.xlim1])

      if params.is_xticks:
        ax1.tick_params(direction="in", which='both')
        ax1.set_xticks(params.xticks)
      if params.secondaryXTicks:
        ax2 = ax1.secondary_xaxis('top') 
        ax2.set_xticklabels([])
        ax2.tick_params(direction="in")
      if params.is_xticklabels:
        ax1.set_xticklabels(params.xticklabels)

      if params.is_yticks:
        ax1.set_yticks(params.yticks)
      if params.secondaryYTicks:
        ax3 = ax1.secondary_yaxis('right')
        ax3.set_yticklabels([])
        ax3.tick_params(direction="in", which='both') 
      if params.is_yticklabels:
        ax1.set_yticklabels(params.yticklabels)

      if params.grid:
        plt.grid(True, which=params.gridWhich)
      if params.legend:
        ax1.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
        if plot_type_id != 2:
          ax2.legend(bbox_to_anchor=(params.legendX, params.legendY - 0.3), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)

      if params.title:
        fig1.suptitle(params.titleName,fontsize=18)

      plt.savefig(params.outputDir + params.filename.split('.')[0] + '_' + str(plot_type_id) + '.' + params.filename.split('.')[1], bbox_inches='tight', dpi=params.DPI)
      plt.close()

  return

