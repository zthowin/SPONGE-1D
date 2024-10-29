#--------------------------------------------------------------------------------------------------
# Plotting script for errors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    June 13, 2022
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
#------------
# Arguments:
#------------
# params           (object)      problem parameters initiated in customPlots.py from an input file
#--------------------------------------------------------------------------------------------------
def main(params):

  eringen = False
  deBoer  = False

  print("Generating plots...")

  #-------------
  # Plot errors.
  #-------------
  fig1 = plt.figure(1)
  ax1  = fig1.add_subplot(111)

  #----------------------------------------
  # Eringen data:
  # Rows are element size from 1m to 0.01m.
  #----------------------------------------
  if eringen:
    # eringen_error_Q1_NBLA   = np.array([0.01042])
    # eringen_error_Q1_NBCA   = np.array([0.01051])
    # eringen_error_Q1_CDL    = np.array([0.050867, 0.007424, 0.003038])
    # eringen_error_Q1_CDC    = np.array([0.032750, 0.00676, 0.002872])
    # eringen_error_Q1_RKBSL  = np.array([0.007393, 0.009855, 0.003824, 0.003449, 0.004754])
    # eringen_error_Q1_RKBSC  = np.array([0.023117, 0.009814, 0.003560, 0.002672, 0.002010])
    # eringen_error_Q1_RKFNCL = np.array([0.007393, 0.009855, 0.003824, 0.003449, 0.004754])
    # eringen_error_Q1_RKFNCC = np.array([0.023117, 0.009814, 0.003560, 0.002672, 0.002010])

    # eringen_error_Q2_NBLA   = np.array([0.002398])
    # eringen_error_Q2_NBCA   = np.array([0.002112])
    # eringen_error_Q2_CDL    = np.array([11.818949, 0.007055, 0.003995]) # when dt_max = 1e-4
    # #eringen_error_Q2_CDL    = np.array([0.001923, 0.007055, 0.003995]) # when dt_max = 1e-5
    # eringen_error_Q2_CDC    = np.array([0.04, 0.007050, 0.050441])
    # eringen_error_Q2_RKBSL  = np.array([0.009945, 0.008613, 0.0079115, 0.0020837, 0.004715])
    # eringen_error_Q2_RKBSC  = np.array([0.011408, 0.0041538, 0.0063280, 0.0020105, 0.001537])
    # eringen_error_Q2_RKFNCL = np.array([0.009945, 0.008613, 0.0079115, 0.0020837, 0.004715])
    # eringen_error_Q2_RKFNCC = np.array([0.011408, 0.0041538, 0.0063280, 0.0020105, 0.001537])
    eringen_error_Q1_NBLA   = np.array([0.01042])
    eringen_error_Q1_NBCA   = np.array([0.01051])
    eringen_error_Q1_CDL    = np.array([0.0080177, 0.004649, 0.006363])
    eringen_error_Q1_CDC    = np.array([0.0103447, 0.004632, 0.006359])
    eringen_error_Q1_RKBSL  = np.array([0.008566, 0.008220, 0.006678, 0.007483, 0.006889])
    eringen_error_Q1_RKBSC  = np.array([0.011395, 0.002777, 0.006233, 0.006974, 0.007427])
    eringen_error_Q1_RKFNCL = np.array([0.008566, 0.008220, 0.006678, 0.007483, 0.006889])
    eringen_error_Q1_RKFNCC = np.array([0.011395, 0.002777, 0.006233, 0.006974, 0.007427])

    eringen_error_Q2_NBLA   = np.array([0.002398])
    eringen_error_Q2_NBCA   = np.array([0.002112])
    eringen_error_Q2_CDL    = np.array([0.008048, 0.004626, 0.006356]) # when dt_max = 1e-5
    eringen_error_Q2_CDC    = np.array([0.009424, 0.004626, 0.006352])
    eringen_error_Q2_RKBSL  = np.array([0.008920, 0.005485, 0.004432, 0.007163, 0.007398])
    eringen_error_Q2_RKBSC  = np.array([0.003258, 0.006664, 0.004834, 0.007827, 0.007384])
    eringen_error_Q2_RKFNCL = np.array([0.008920, 0.005485, 0.004432, 0.007163, 0.007398])
    eringen_error_Q2_RKFNCC = np.array([0.003258, 0.006664, 0.004834, 0.007827, 0.007384])

    eringen_el_meter = np.array([1, 5, 10, 15, 20])

    # plt.semilogy(eringen_el_meter[0], eringen_error_Q1_NBLA, '-o', color='black', fillstyle='none')
    # plt.semilogy(eringen_el_meter[0], eringen_error_Q1_NBCA, '-o', color='blue', fillstyle='none')
    # plt.semilogy(eringen_el_meter[0], eringen_error_Q2_NBLA, '--o', color='black', fillstyle='none')
    # plt.semilogy(eringen_el_meter[0], eringen_error_Q2_NBCA, '--o', color='blue', fillstyle='none')

    # plt.semilogy(eringen_el_meter[0:3], eringen_error_Q1_CDL, '-o', color='black', fillstyle='none', label=r'$(\bu)$, Q1, Central-difference, L. mass')
    # plt.semilogy(eringen_el_meter[0:3], eringen_error_Q1_CDC, '-s', color='red', fillstyle='none', label=r'$(\bu)$, Q1, Central-difference, C. mass')
    # plt.semilogy(eringen_el_meter[0:3], eringen_error_Q2_CDL, '--o', color='black', fillstyle='none', label=r'$(\bu)$, Q2, Central-difference, L. mass')
    # plt.semilogy(eringen_el_meter[0:3], eringen_error_Q2_CDC, '--s', color='red', fillstyle='none', label=r'$(\bu)$, Q2, Central-difference, C. mass')
    plt.semilogy(eringen_el_meter[0:3], eringen_error_Q1_CDL, '-o', color='black', fillstyle='none', label=r'$(\bu)$, Central-difference, L. mass')
    plt.semilogy(eringen_el_meter[0:3], eringen_error_Q1_CDC, '-s', color='red', fillstyle='none', label=r'$(\bu)$, Central-difference, C. mass')
    plt.semilogy(eringen_el_meter[0:3], eringen_error_Q2_CDL, '--o', color='black', fillstyle='none')
    plt.semilogy(eringen_el_meter[0:3], eringen_error_Q2_CDC, '--s', color='red', fillstyle='none')
    
    # plt.semilogy(eringen_el_meter, eringen_error_Q1_RKBSL, '-v', color='black', fillstyle='none')
    # plt.semilogy(eringen_el_meter, eringen_error_Q1_RKBSC, '-v', color='blue', fillstyle='none')
    # plt.semilogy(eringen_el_meter, eringen_error_Q2_RKBSL, '--v', color='black', fillstyle='none')
    # plt.semilogy(eringen_el_meter, eringen_error_Q2_RKBSC, '--v', color='blue', fillstyle='none')

    # plt.semilogy(eringen_el_meter, eringen_error_Q1_RKFNCL, '-v', color='blue', fillstyle='none', label=r'$(\bu)$, Q1, Runge-Kutta methods, L. mass')
    # plt.semilogy(eringen_el_meter, eringen_error_Q1_RKFNCC, '-^', color='green', fillstyle='none', label=r'$(\bu)$, Q1, Runge-Kutta methods, C. mass')
    # plt.semilogy(eringen_el_meter, eringen_error_Q2_RKFNCL, '--v', color='blue', fillstyle='none', label=r'$(\bu)$, Q2, Runge-Kutta methods, L. mass')
    # plt.semilogy(eringen_el_meter, eringen_error_Q2_RKFNCC, '--^', color='green', fillstyle='none', label=r'$(\bu)$, Q2, Runge-Kutta methods, C. mass')
    plt.semilogy(eringen_el_meter, eringen_error_Q1_RKFNCL, '-v', color='blue', fillstyle='none', label=r'$(\bu)$, Runge-Kutta methods, L. mass')
    plt.semilogy(eringen_el_meter, eringen_error_Q1_RKFNCC, '-^', color='green', fillstyle='none', label=r'$(\bu)$, Runge-Kutta methods, C. mass')
    plt.semilogy(eringen_el_meter, eringen_error_Q2_RKFNCL, '--v', color='blue', fillstyle='none')
    plt.semilogy(eringen_el_meter, eringen_error_Q2_RKFNCC, '--^', color='green', fillstyle='none')

    plt.text(0.61, 0.6, r'\rule[.5ex]{0.65em}{1pt} Q1 elements' + '\n' + r'\rule[.5ex]{0.3em}{1pt}\hspace{0.05em}\rule[.5ex]{0.3em}{1pt} Q2 elements', \
             fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black', pad=3.0))

  if deBoer:
    # deBoer_error_Q1P1_A0_Imp_T   = np.array([0.007178, 0.004728, 0.004657])
    # deBoer_error_Q1P1_A0_Imp_N   = np.array([0.007178, 0.004728, 0.004657])
    # deBoer_error_Q1P1_A0_CD_L    = np.array([0.010725])
    # deBoer_error_Q1P1_A0_CD_C    = np.array([0.006332])
    # deBoer_error_Q1P1_A0_RK_L    = np.array([0.006396])

    # deBoer_error_Q1P1_A10_Imp_T  = np.array([0.007465, 0.005009])
    # deBoer_error_Q1P1_A10_Imp_N  = np.array([0.007447, 0.005008])
    # deBoer_error_Q1P1_A10_RK_C   = np.array([0.003317, 0.004694, 0.004901, 0.004934])

    # deBoer_error_Q1P1_A8_Imp_T   = np.array([0.042299, 0.037057, 0.042370, 0.040211])
    # deBoer_error_Q1P1_A8_Imp_N   = np.array([0.042276, 0.037128, 0.042353, 0.040208])
    # deBoer_error_Q1P1_A8_RK_C    = np.array([0.056566, 0.047387, 0.046970, 0.046968, 0.046917])

    deBoer_el_meter = np.array([1, 5, 10, 15, 20])

    deBoer_error_Q1P1_A0    = np.array([0.007178, 0.004729, 0.004657, 0.004644, 0.004639])
    deBoer_error_Q1P1_A10   = np.array([0.007465, 0.005009, 0.004938])
    deBoer_error_Q1P1_A8    = np.array([0.042299, 0.037057, 0.042370, 0.040211, 0.037718])

    deBoer_error_Q2P1_A0    = np.array([0.017046, 0.005070, 0.004742, 0.004682, 0.004660])

    deBoer_error_Q1Q1P1_A0  = np.array([0.001983, 0.000712, 0.000709, 0.000709, 0.000709])
    deBoer_error_Q1Q1P1_A10 = np.array([0.002227, 0.000947, 0.000943, 0.000944, 0.000944])
    deBoer_error_Q1Q1P1_A8  = np.array([0.033835, 0.031612, 0.031542, 0.031529, 0.031524])

    deBoer_error_Q2Q1P1_A0  = np.array([0.01114, 0.001022, 0.000786, 0.000743, 0.000729])
    deBoer_error_Q2Q1P1_A10 = np.array([0.011442, 0.001259, 0.001021, 0.000978, 0.000963])
    deBoer_error_Q2Q1P1_A8  = np.array([0.047415, 0.032089, 0.031659, 0.031581, 0.031554])

    deBoer_error_Q2Q2P1_A0  = np.array([0.011983, 0.001043, 0.000788, 0.000744, 0.000729])

    # if 'RK' in params.filename:
    #   plt.semilogy(deBoer_el_meter[0], deBoer_error_Q1P1_A0_RK_L, '-o', color='black', fillstyle='none', label=r'$(\bu$-$p_\rf)$, L. mass, $\alpha^\text{stab} = 0$')
    #   plt.semilogy(deBoer_el_meter[0:4], deBoer_error_Q1P1_A10_RK_C, '-s', color='red', fillstyle='none', label=r'$(\bu$-$p_\rf)$, C. mass, $\alpha^\text{stab} = 10^{-10}$')
    #   plt.semilogy(deBoer_el_meter, deBoer_error_Q1P1_A8_RK_C, '-v', color='green', fillstyle='none', label=r'$(\bu$-$p_\rf)$, C. mass, $\alpha^\text{stab} = 10^{-8}$')
    # elif 'Imp' in params.filename:
    #   # plt.semilogy(deBoer_el_meter[0:3], deBoer_error_Q1P1_A0_Imp_T, '-o', color='black', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 0$, undamped')
    #   # plt.semilogy(deBoer_el_meter[0:3], deBoer_error_Q1P1_A0_Imp_N, '-s', color='red', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 0$, damped')

    #   # plt.semilogy(deBoer_el_meter[0:2], deBoer_error_Q1P1_A10_Imp_T, '-o', color='black', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 10^{-10}$, undamped')
    #   # plt.semilogy(deBoer_el_meter[0:2], deBoer_error_Q1P1_A10_Imp_N, '-s', color='red', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 10^{-10}$, damped')

    #   plt.semilogy(deBoer_el_meter[0:4], deBoer_error_Q1P1_A8_Imp_T, '-o', color='black', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 10^{-8}$, undamped')
    #   plt.semilogy(deBoer_el_meter[0:4], deBoer_error_Q1P1_A8_Imp_N, '-s', color='red', fillstyle='none', label=r'$(\bu$-$p_\rf)$, $\alpha^\text{stab} = 10^{-8}$, damped')

    plt.semilogy(deBoer_el_meter, deBoer_error_Q1P1_A0, '-o', color='black', fillstyle='none', label='Q1-P1')
    plt.semilogy(deBoer_el_meter[0:3], deBoer_error_Q1P1_A10, '--o', color='black', fillstyle='none')
    plt.semilogy(deBoer_el_meter, deBoer_error_Q1P1_A8, '-.o', color='black', fillstyle='none')

    plt.semilogy(deBoer_el_meter, deBoer_error_Q2P1_A0, '-o', color='red', fillstyle='none', label='Q2-P1')

    plt.semilogy(deBoer_el_meter, deBoer_error_Q1Q1P1_A0, '-o', color='blue', fillstyle='none', label='Q1-Q1-P1')
    plt.semilogy(deBoer_el_meter, deBoer_error_Q1Q1P1_A10, '--o', color='blue', fillstyle='none')
    plt.semilogy(deBoer_el_meter, deBoer_error_Q1Q1P1_A8, '-.o', color='blue', fillstyle='none')

    plt.semilogy(deBoer_el_meter, deBoer_error_Q2Q1P1_A0, '-o', color='green', fillstyle='none', label='Q2-Q1-P1')
    plt.semilogy(deBoer_el_meter, deBoer_error_Q2Q1P1_A10, '--o', color='green', fillstyle='none')
    plt.semilogy(deBoer_el_meter, deBoer_error_Q2Q1P1_A8, '-.o', color='green', fillstyle='none')

    plt.semilogy(deBoer_el_meter, deBoer_error_Q2Q2P1_A0, '-o', color='magenta', fillstyle='none', label='Q2-Q2-P1')

    plt.text(0.73, 0.765, r'\rule[.5ex]{0.65em}{1pt}\hspace{0.725em}$\alpha^\text{stab}\eq 0$' + '\n' + r'\rule[.5ex]{0.3em}{1.15pt}\hspace{0.2em}\rule[.5ex]{0.3em}{1.15pt}\hspace{0.55em}$\alpha^\text{stab}\eq 10^{-10}$' + '\n' + r'\rule[.5ex]{0.3em}{1pt}.\rule[.5ex]{0.3em}{1pt}\hspace{0.125em} $\alpha^\text{stab}\eq 10^{-8}$', \
             fontsize=10, transform=plt.gcf().transFigure, bbox=dict(facecolor='none', edgecolor='black', pad=3.0))

  plt.xlabel(r'\# elements/meter', fontsize=16)
  plt.ylabel(r'rel. displ. error $ERR_u[-]$', fontsize=16)
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
    plt.legend(bbox_to_anchor=(params.legendX, params.legendY), loc=params.legendPosition, handlelength=params.handleLength, edgecolor='k', framealpha=1.0)
  if params.title:
    fig1.suptitle(params.titleName)

  plt.savefig(params.outputDir + params.filename + '.pdf', bbox_inches='tight', dpi=params.DPI)
  plt.close()

  return
