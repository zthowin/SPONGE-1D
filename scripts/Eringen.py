#--------------------------------------------------------------------------------------------------
# Module that computes the 1D uniaxial-strain analytical solution for a column of solid material
# undergoing compression.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   September 21, 2021
#--------------------------------------------------------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#-------------------------
# Define helper functions.
#-------------------------

def get_integralSine(a_T, a_Tau, a_n):
  return np.sin((2*a_n - 1)*np.pi*c*(a_T - a_Tau)/(2*h))

# Equation 67
def get_fourierSine(a_n, a_z):
  return np.sin((2*a_n - 1)*np.pi*a_z/(2*h))

def get_fourierCosine(a_n, a_z):
  return np.cos((2*a_n - 1)*np.pi*a_z/(2*h))

def get_Load(a_Tau, a_Freq, a_Amp):
  return -0.5*a_Amp*(1 - np.cos(a_Freq*a_Tau))

#------------
# Main script
#------------
saveDir    = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/Python/output/neo-Hookean/Eringen/analytical/N25/Q1/1/'
if not os.path.exists(saveDir):
  os.makedirs(saveDir)

# Define material properties
E   = 50e6
nu  = 0.3
rho = 1986
M   = E*((1 - nu)/((1 + nu)*(1 - 2*nu)))
c   = np.sqrt(M/rho)
h   = 20

# Define loading function properties
amp  = 40e3
freq = 50

# Define number of series terms
nseries = 25

# Perform time integral
nZ   = 20
nen  = 1*20 + 1
z    = np.linspace(0, h, nen)
# z[0] = 20.
# z[1] = 15
# z[2] = 10
# z[3] = 5
nZ = nen

dt = 1e-3
tf = 0.4

N = int(tf/dt)

time     = np.zeros(N + 2)
u        = np.zeros((nZ, N + 2))
du       = np.zeros((nZ, N + 2))
integral = np.zeros((N+2))
u_n      = np.zeros((nZ, nseries + 1, N + 2))
du_n     = np.zeros((nZ, nseries + 1, N + 2))

coef  = 4/(np.pi*rho*c)
coef2 = 2/(h*rho*c)

# Loop over depths
for k in range(nZ):
  print("Generating solution at depth:", z[k], "meters.")

  # Loop over sine series
  for n in range(1, nseries+1):
    term1 = (-1)**n/(2*n - 1)
    term2 = get_fourierSine(n, z[k])
    term3 = (-1)**n
    term4 = get_fourierCosine(n, z[k])

    # Loop over time 'tau'
    for i in range(N+1):
      time[i+1] = i*dt

      tau  = np.linspace(0, time[i+1], N)
      dtau = tau[1] - tau[0]

      load          = get_Load(tau, freq, amp)
      sine_term     = get_integralSine(time[i+1], tau, n)
      integral[i+1] = np.sum(load*sine_term)*dtau

    u_n[k, n, :]  = integral*term1*term2
    du_n[k, n, :] = integral*term3*term4

  u[k, :] = coef*np.sum(u_n[k,:,:], axis=0)
  du[k,:] = coef2*np.sum(du_n[k,:,:], axis=0)

np.save(saveDir + 'time.npy',         time, allow_pickle=False) 
np.save(saveDir + 'displacement.npy', -u,   allow_pickle=False)
np.save(saveDir + 'gradient.npy',    -du,   allow_pickle=False)
