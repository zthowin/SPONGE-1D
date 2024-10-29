import sys, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

#---------------------------
# Function to animate plots.
#---------------------------
def animate_Pressures(i):

  # Plot the lines (and do not erase previous time points)
  J_pf_lA.set_data(simA_J[:i], simA_pf_norm[:i])
  J_p_lA.set_data(simA_J[:i], simA_press_norm[:i])
  J_K_lA.set_data(simA_J[:i], simA_p_prime_norm[:i])
  J_K_lB.set_data(simB_J[:i], simB_p_prime_norm[:i])

  # Plot leading points (only at time-step 'i')
  J_pf_pA.set_data(simA_J[i-1], simA_pf_norm[i-1])
  J_p_pA.set_data(simA_J[i-1], simA_press_norm[i-1])
  J_K_pA.set_data(simA_J[i-1], simA_p_prime_norm[i-1])
  J_K_pB.set_data(simB_J[i-1], simB_p_prime_norm[i-1])

  displ_lA.set_data(simA_tsolve[:i], simA_Dsolve[:i])
  displ_pA.set_data(simA_tsolve[i-1], simA_Dsolve[i-1])

  displ_lB.set_data(simB_tsolve[:i], simB_Dsolve[:i])
  displ_pB.set_data(simB_tsolve[i-1], simB_Dsolve[i-1])

  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')

  return J_pf_lA, J_p_lA, J_K_lA, J_K_lB, displ_lA, displ_lB, J_pf_pA, J_p_pA, J_K_pA, J_K_pB, displ_pA, displ_pB

#---------------------------
# Function to animate plots.
#---------------------------
def animate_Shear(i):

  # Plot the lines (and do not erase previous time points)
  J_tau_lA.set_data(simA_J[:i], simA_tau_norm[:i])
  J_tau_lB.set_data(simB_J[:i], simB_tau_norm[:i])

  # Plot leading points (only at time-step 'i')
  J_tau_pA.set_data(simA_J[i-1], simA_tau_norm[i-1])
  J_tau_pB.set_data(simB_J[i-1], simB_tau_norm[i-1])

  displ_lA2.set_data(simA_tsolve[:i], simA_Dsolve[:i])
  displ_pA2.set_data(simA_tsolve[i-1], simA_Dsolve[i-1])

  displ_lB2.set_data(simB_tsolve[:i], simB_Dsolve[:i])
  displ_pB2.set_data(simB_tsolve[i-1], simB_Dsolve[i-1])

  ax3.legend(loc='upper right')
  ax4.legend(loc='upper right')

  return J_tau_lA, J_tau_lB, displ_lA2, displ_lB2, J_tau_pA, J_tau_pB, displ_pA2, displ_pB2

#---------------------------
# Function to animate plots.
#---------------------------
def animate_Stress(i):

  # Plot the lines (and do not erase previous time points)
  J_ss_lA.set_data(-simA_le11[:i], -simA_sig11[:i])
  J_ss_lB.set_data(-simB_le11[:i], -simB_sig11[:i])

  # Plot leading points (only at time-step 'i')
  J_ss_pA.set_data(-simA_le11[i-1], -simA_sig11[i-1])
  J_ss_pB.set_data(-simB_le11[i-1], -simB_sig11[i-1])

  displ_lA3.set_data(simA_tsolve[:i], simA_Dsolve[:i])
  displ_pA3.set_data(simA_tsolve[i-1], simA_Dsolve[i-1])

  displ_lB3.set_data(simB_tsolve[:i], simB_Dsolve[:i])
  displ_pB3.set_data(simB_tsolve[i-1], simB_Dsolve[i-1])

  ax5.legend(loc='upper left')
  ax6.legend(loc='upper right')

  return J_ss_lA, J_ss_lB, displ_lA3, displ_lB3, J_ss_pA, J_ss_pB, displ_pA3, displ_pB3

#--------------
# Load in data.
#--------------
# simA_Dir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/Cython/output/Friedlander/fine/upf/implicit-newmark/'
simB_Dir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/LS-DYNA/lung-umat/usermat-files/custom-Friedlander-15kPa-100el/'
# simB_Dir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/LS-DYNA/lung-umat/usermat-files/custom-Yen-15kPa-100el-1769949/'
simA_Dir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/Python/output/Friedlander/fine/upfuf/implicit-newmark/fixed/'
# simB_Dir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/LS-DYNA/lung-umat/usermat-files/custom-Yen-100el-1755943/'
dataDir = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Presentations/Spring 2022/figures/Friedlander/poroelastodynamics/upfuf/fine/shear-strain-animation/'
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

simA_disp_fname         = simA_Dir + 'displacement.npy'
simA_time_fname         = simA_Dir + 'time.npy'        
simA_J_fname            = simA_Dir + 'J.npy'           
simA_press_norm_fname   = simA_Dir + 'press_norm.npy'  
simA_p_prime_norm_fname = simA_Dir + 'p_prime_norm.npy'
simA_pf_norm_fname      = simA_Dir + 'pf_norm.npy'
simA_tau_fname          = simA_Dir + 'shear_norm.npy'
simA_sig11_fname        = simA_Dir + 'sig11.npy'

simB_disp_fname         = simB_Dir + 'displacement.npy'
simB_time_fname         = simB_Dir + 'time.npy'        
simB_J_fname            = simB_Dir + 'J.npy'            
# simB_p_prime_norm_fname = simB_Dir + 'p_prime_norm.npy'
simB_p_prime_norm_fname = simB_Dir + 'press.npy'
simB_tau_fname          = simB_Dir + 'shear.npy' 
simB_sig11_fname        = simB_Dir + 'sig11.npy' 

KS = 2.13e5
G  = 3e3

simA_Dsolve       = np.load(simA_disp_fname)[:,2*100 - 1]*1e3
simA_tsolve       = np.load(simA_time_fname)*1e3
simA_J            = np.load(simA_J_fname)[:,-1,1]
nsteps  = simA_J.shape[0]
simA_press_norm   = np.load(simA_press_norm_fname)[:,-1,1]*7.5e3/KS
simA_p_prime_norm = np.load(simA_p_prime_norm_fname)[:,-1,1]*7.5e3/KS
simA_pf_norm      = np.load(simA_pf_norm_fname)[:,-1,1]*7.5e3/KS -101325/(7.5e3*KS)
# simA_tau_norm     = np.load(simA_tau_fname)[:,-1,1]
simA_tau_norm     = np.abs(simA_J[1:nsteps-1]**2 - 1)/simA_J[1:nsteps-1]
simA_sig11        = np.load(simA_sig11_fname)[:,-1,1]
simA_le11         = np.log(simA_J[1:-1])

# simB_Dsolve       = np.load(simB_disp_fname)[:,2*100 - 1]*1e3
simB_Dsolve       = np.load(simB_disp_fname)[:,403]*1e3
# simB_tsolve       = np.load(simB_time_fname)*1e3
simB_tsolve       = np.linspace(0, 30e-3, nsteps+1)*1e3
simB_J            = np.load(simB_J_fname)[:,-1]
simB_p_prime_norm = np.load(simB_p_prime_norm_fname)[:,-1]/KS
simB_tau_norm     = np.load(simB_tau_fname)[:,-1]/G
simB_sig11        = np.load(simB_sig11_fname)[:,-1]
simB_le11         = np.log(simB_J)[1:nsteps-1]

plt.figure(1)
plt.plot(-simA_le11, -simA_sig11[1:nsteps-1])
plt.plot(-simB_le11, -simB_sig11[1:nsteps-1])
plt.show()

# plt.figure(1)
# plt.plot(simA_J[1:-1], -simA_p_prime_norm[1:-1], label='python p prime')
# plt.plot(simA_J[1:-1], -simA_press_norm[1:-1], label='python p')
# plt.plot(simB_J[1:-1], -simB_p_prime_norm[1:-1], label='dyna p')
# plt.legend()

# plt.figure(2)
# plt.plot(simA_tsolve, simA_p_prime_norm, label='python p prime')
# plt.plot(simA_tsolve, simA_press_norm, label='python p')
# plt.plot(simB_tsolve, simB_p_prime_norm, label='dyna p')
# plt.legend()

# plt.figure(3)
# plt.plot(simA_tsolve, simA_Dsolve, label='python D')
# plt.plot(simB_tsolve, simB_Dsolve, label='dyna D')

# plt.show()

#-----------------------
# Set up figure objects.
#----------------------- 
# skip = 1
# fig1 = plt.figure(figsize=(13,5))
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
# ax1.grid()
# ax2.grid()
# ax2.set_xlim([0,30])

# #-----------------
# # Set LaTeX fonts.
# #-----------------
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# ax1.set_xlabel(r'$\rho_0$/$\rho$', fontsize=16)
# ax1.set_ylabel(r'$p/K$', fontsize=16)
# ax1.set_title(r'Normalized pressure vs. $J$ at top Gauss point', fontsize=14)

# ax2.set_xlabel(r'Time (ms)', fontsize=16)
# ax2.set_ylabel(r'Displacement (mm)', fontsize=16)
# ax2.set_title(r'Displacement vs. Time at $X=H$')

# # These are the lines that will be plotted over time
# J_pf_lA, = ax1.plot(simA_J[1:nsteps-1], simA_pf_norm[1:nsteps-1], '--', label=r'$-p_{\text{f}}/K$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1, color='red')
# J_p_lA,  = ax1.plot(simA_J[1:nsteps-1], simA_press_norm[1:nsteps-1], '-r', label=r'$p/K$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
# J_K_lA,  = ax1.plot(simA_J[1:nsteps-1], simA_p_prime_norm[1:nsteps-1], '.', label=r'$p^{\prime}/K$, $u$-$u_\text{f}$-$p_\text{f}$', markersize=1, color='red')
# J_K_lB,  = ax1.plot(simB_J[1:nsteps-1], simB_p_prime_norm[1:nsteps-1], '-k', label=r'$p/K$, LS-DYNA, custom model', lw=1)

# # These are the markers which are only plotted for a given time step (makes seeing the progress of the lines easier)
# J_pf_pA, = ax1.plot(simA_J[1:nsteps-1], simA_pf_norm[1:nsteps-1], 'o', lw=2, color='red')
# J_p_pA,  = ax1.plot(simA_J[1:nsteps-1], simA_press_norm[1:nsteps-1], 'ro', lw=2)
# J_K_pA,  = ax1.plot(simA_J[1:nsteps-1], simA_p_prime_norm[1:nsteps-1], 'o', lw=2, color='red')
# J_K_pB,  = ax1.plot(simB_J[1:nsteps-1], simB_p_prime_norm[1:nsteps-1], 'ko', lw=2)

# displ_lA, = ax2.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], '-r', label=r'$u(X=H,t)$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
# displ_pA, = ax2.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], 'ro', lw=2)

# displ_lB, = ax2.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], '-k', label=r'$u(X=H,t)$, LS-DYNA (Clayton and Freed)', lw=1)
# displ_pB, = ax2.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], 'ko', lw=2)

# ax1.set_xlabel(r'$\rho_0$/$\rho$', fontsize=16)
# ax1.set_ylabel(r'$p/K$', fontsize=16)
# ax1.set_title(r'Normalized pressure vs. $J$ at top Gauss point', fontsize=14)

# ax2.set_xlabel(r'Time (ms)', fontsize=16)
# ax2.set_ylabel(r'Displacement (mm)', fontsize=16)
# ax2.set_title(r'Displacement vs. Time at $X=H$')

#-----------------------
# Set up figure objects.
#----------------------- 
fig2 = plt.figure(figsize=(13,5))
ax3 = plt.subplot(121)
ax4 = plt.subplot(122)
ax3.grid()
ax4.grid()
ax4.set_xlim([0,30])
# These are the lines that will be plotted over time
J_tau_lA, = ax3.plot(simA_J[1:nsteps-1], simA_tau_norm, '-r', label=r'$\tau/\mu$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
J_tau_lB, = ax3.plot(simB_J[1:nsteps-1], simB_tau_norm[1:nsteps-1], '-k', label=r'$\tau/\mu$, LS-DYNA (Clayton and Freed)', lw=1)

# These are the markers which are only plotted for a given time step (makes seeing the progress of the lines easier)
J_tau_pA, = ax3.plot(simA_J[1:nsteps-1], simA_tau_norm, 'ro', lw=2)
J_tau_pB, = ax3.plot(simB_J[1:nsteps-1], simB_tau_norm[1:nsteps-1], 'ko', lw=2)

displ_lA2, = ax4.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], '-r', label=r'$u(X=H,t)$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
displ_pA2, = ax4.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], 'ro', lw=2)

displ_lB2, = ax4.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], '-k', label=r'$u(X=H,t)$, LS-DYNA (Clayton and Freed)', lw=1)
displ_pB2, = ax4.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], 'ko', lw=2)

ax3.set_xlabel(r'$\rho_0$/$\rho$', fontsize=16)
ax3.set_ylabel(r'$\tau/\mu$', fontsize=16)
ax3.set_title(r'Normalized pure shear vs. $J$ at top Gauss point', fontsize=14)

ax4.set_xlabel(r'Time (ms)', fontsize=16)
ax4.set_ylabel(r'Displacement (mm)', fontsize=16)
ax4.set_title(r'Displacement vs. Time at $X=H$')

#-----------------------
# Set up figure objects.
#----------------------- 
fig3 = plt.figure(figsize=(13,5))
ax5 = plt.subplot(121)
ax6 = plt.subplot(122)
ax5.grid()
ax6.grid()
ax6.set_xlim([0,30])
# These are the lines that will be plotted over time
J_ss_lA, = ax5.plot(-simA_le11, -simA_sig11[1:nsteps-1], '-r', label=r'$\sigma_{11}$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
J_ss_lB, = ax5.plot(-simB_le11, -simB_sig11[1:nsteps-1], '-k', label=r'$\sigma_{11}$, LS-DYNA (Clayton and Freed)', lw=1)

# These are the markers which are only plotted for a given time step (makes seeing the progress of the lines easier)
J_ss_pA, = ax5.plot(-simA_le11, -simA_sig11[1:nsteps-1], 'ro', lw=2)
J_ss_pB, = ax5.plot(-simB_le11, -simB_sig11[1:nsteps-1], 'ko', lw=2)

displ_lA3, = ax6.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], '-r', label=r'$u(X=H,t)$, $u$-$u_\text{f}$-$p_\text{f}$', lw=1)
displ_pA3, = ax6.plot(simA_tsolve[1:nsteps-1], simA_Dsolve[1:nsteps-1], 'ro', lw=2)

displ_lB3, = ax6.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], '-k', label=r'$u(X=H,t)$, LS-DYNA (Clayton and Freed)', lw=1)
displ_pB3, = ax6.plot(simB_tsolve[1:nsteps-1], simB_Dsolve[1:nsteps-1], 'ko', lw=2)

ax5.set_xlabel(r'-$le_{11}$', fontsize=16)
ax5.set_ylabel(r'-$\sigma_{11}$', fontsize=16)
ax5.set_title(r'Hencky strain vs. Cauchy stress at top Gauss point', fontsize=14)

ax6.set_xlabel(r'Time (ms)', fontsize=16)
ax6.set_ylabel(r'Displacement (mm)', fontsize=16)
ax6.set_title(r'Displacement vs. Time at $X=H$')

#-------------
# Now animate.
#-------------
# ani1 = animation.FuncAnimation(fig1, animate_Pressures, nsteps - 1, interval=10, blit=True)
# writer1 = animation.FFMpegFileWriter(fps=10, metadata=dict(artist='bww'), bitrate=1800)
# ani1.save(dataDir + 'pressure.mp4', writer=writer1)
# plt.show()

# ani2 = animation.FuncAnimation(fig2, animate_Shear, nsteps - 1, interval=10, blit=True)
# writer2 = animation.FFMpegFileWriter(fps=10, metadata=dict(artist='bww'), bitrate=1800)
# ani2.save(dataDir + 'shear.mp4', writer=writer2)
# plt.show()

ani3 = animation.FuncAnimation(fig3, animate_Stress, nsteps - 1, interval=10, blit=True)
writer3 = animation.FFMpegFileWriter(fps=10, metadata=dict(artist='bww'), bitrate=1800)
ani3.save(dataDir + 'stress.mp4', writer=writer3)
