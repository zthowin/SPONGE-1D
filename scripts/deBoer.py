#--------------------------------------------------------------------------------------------------
# Module that computes the 1D uniaxial-strain analytical solution for a column of biphasic material
# undergoing compression with a drained boundary condition at its surface.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   July 20, 2021
#--------------------------------------------------------------------------------------------------
import os, sys
import numpy as np
import scipy.special as special

#--------------------------------------------------------------------
# Define helper functions, equation numbers in de Boer et al. (1993).
#--------------------------------------------------------------------

# Equation 65
def get_uDisp(a_T, a_Tau, a_Z, a_A, a_B):

  e_term = np.exp((-a_B/(2.0*a_A))*a_Tau)
  b_term = special.iv(0, ( (a_B*np.sqrt(a_Tau**2 - a_A*a_Z**2 + 0j)) / (2.0*a_A) ))
  h_term = np.heaviside(a_Tau - np.sqrt(a_A)*a_Z, 1.0)

  return e_term*b_term*h_term

# Equation 67
def get_PF(a_nf, a_ns, a_rhof, a_Sv, a_Lambda, a_Mu, a_LDot, a_LDDot):

  return (1.0/((a_nf**2)*(a_Lambda + 2.0*a_Mu))) * (a_ns*a_nf*a_rhof*a_LDDot + a_Sv*a_LDot)

# Equation 71
def get_Stress(a_Tau, a_Z, a_A, a_B):

  e_term = np.exp((-a_B/(2.0*a_A))*a_Tau)
  b_term = special.iv(1, ( (b*np.sqrt(a_Tau**2 - a_A*a_Z**2 + 0j)) / (2.0*a_A) ))
  h_term = np.heaviside(a_Tau - np.sqrt(a_A)*a_Z, 1.0)

  return e_term*b_term*h_term * (a_Z / np.sqrt(a_Tau**2 - a_A*a_Z**2 + 0j))

# Equation 69
def get_Q(a_Tau, a_A, a_B):

  e_term = np.exp((-a_B/(2.0*a_A))*a_Tau)
  b_term = special.iv(0, (a_B/(2.0*a_A))*a_Tau)

  return e_term*b_term

# Equation 70
def get_G(a_T, a_Z, a_A, a_B):

  e_term = np.exp((-a_B/(2.0*a_A))*a_T)
  b_term = special.iv(0, ( (b*np.sqrt(a_T**2 - a_A*a_Z**2 + 0j)) / (2.0*a_A) ))
  h_term = np.heaviside(a_T - np.sqrt(a_A)*a_Z, 1.0)

  second_term = e_term*special.iv(0, (a_B/(2.0*a_A))*a_T)

  return 1/np.sqrt(a) * (e_term*b_term*h_term - second_term)

# ???
def get_SineWave(a_T, a_Tau, a_Freq, a_Amp):

  return -0.5*a_Amp*(1 - np.cos(a_Freq*(a_T - a_Tau)))

# Equation 55
def get_A(a_ns, a_nf, a_rhoF, a_rhoS, a_Lambda, a_Mu):

  return (((a_ns)**2)*a_nf*a_rhoF + ((a_nf)**2)*a_ns*a_rhoS)/((a_Lambda + 2.0*a_Mu)*a_nf**2)

# Equation 56
def get_B(a_Sv, a_Lambda, a_Mu, a_nf):

  return a_Sv/((a_Lambda + 2.0*a_Mu)*a_nf**2)

# Equation 28
def get_Sv(a_nf, a_Gamma, a_khat):

  return (a_nf**2)*a_Gamma/a_khat

#------------
# Main script
#------------
# saveDir    = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/Python/output/deBoer/analytical/master/'
saveDir    = '/projects/zair9172/1D-multiphase/output/deBoer/analytical/Q1/15/'
if not os.path.exists(saveDir):
  os.makedirs(saveDir)

# Define material properties
mu     = 5.6e6
lamb   = 8.4e6
nf     = 0.42
ns     = 1 - nf
rhof   = 1000
rhos   = 2700
gamma  = rhof * 9.81  # Specific weight
khat   = 1e-6
kf     = khat * gamma # Permeability (m/s)

a  = get_A(ns, nf, rhof, rhos, lamb, mu)
Sv = get_Sv(nf, gamma, kf)
b  = get_B(Sv, lamb, mu, nf)

# Define loading function properties
amp  = 40e3
freq = 50

# Perform time integral
h    = 10
nZ   = 15
nen  = 1*h*nZ + 1
z    = np.linspace(h, 0, nen)

dt = 1e-4
tf = 0.4

N = int(tf/dt) + 1

time = np.zeros(N + 1)
u    = np.zeros((nen, N + 1))
uf   = np.zeros((nen, N + 1))
Q    = np.zeros((nen, N + 1))
G    = np.zeros((nen, N + 1))
L    = np.zeros((nen, N + 1))
sig  = np.zeros((nen, N + 1))
pf   = np.zeros((nen, N - 1))

for k in range(nen):
  print("Generating solution at depth:", h - z[k], "meters.")
  for i in range(N):

    time[i+1] = i*dt

    tau  = np.linspace(0, time[i+1], N)
    dtau = tau[1] - tau[0]

    u1         = get_SineWave(time[i+1], tau, freq, amp)*get_uDisp(time[i+1], tau, z[k], a, b)
    u1[0]      = 0.
    u[k, i+1]  = (1.0/(np.sqrt(a)*(lamb + 2.0*mu)))*np.sum(u1)*dtau

    # Equation 66
    uf[k, i+1] = (ns/nf)*u[k, i+1]

    q1        = get_SineWave(time[i+1], tau, freq, amp)*get_Q(tau, a, b)
    Q[k, i+1] = -1.0/np.sqrt(a) * (np.sum(q1)*dtau)

    G[k, i+1] = get_G(time[i+1], z[k], a, b)

    s1          = get_SineWave(time[i+1], tau, freq, amp)*get_Stress(tau, z[k], a, b)
    s1[0]       = 0.

    # Equation 71
    s2          = get_SineWave(time[i+1], np.sqrt(a)*z[k], freq, amp)*np.heaviside(time[i+1] - np.sqrt(a)*z[k], 1.0)*np.exp(-b*z[k]/(2.0*np.sqrt(a)))
    sig[k, i+1] = (b/(2.0*np.sqrt(a)))*np.sum(s1)*dtau + s2

    for j in range(i):
      L[k, i+1] += Q[k, i+1 - j]*G[k, j]*dt

  L_dot  = np.diff(L[k,:])/np.diff(time)
  L_ddot = np.diff(L_dot)/np.diff(time[0:N])

  pf[k,:] = get_PF(nf, ns, rhof, Sv, lamb, mu, L_dot[0:N-1], L_ddot)

np.save(saveDir + 'time.npy',         time, allow_pickle=False) 
np.save(saveDir + 'displacement.npy', u,    allow_pickle=False)
np.save(saveDir + 'displacement-uf.npy', uf, allow_pickle=False)
np.save(saveDir + 'press.npy',        pf,   allow_pickle=False) 
np.save(saveDir + 'stress.npy',       sig,  allow_pickle=False) 
