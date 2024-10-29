#--------------------------------------------------------------------------------------------------
# Module that computes the 1-D analytical solution for a bar subjected to heat loads.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   November 12, 2023
#--------------------------------------------------------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#-------------------------
# Define helper functions.
#-------------------------

def get_fourierSine(a_n, a_X):
  return np.sin(((2*a_n + 1)*np.pi*a_X)/(2*L))

def get_exponent(a_n, a_T):
  return np.exp((-((2*a_n + 1)**2)*(np.pi**2)*b*a_T)/(4*(L**2)))

#------------
# Main script
#------------
saveDir    = '/home/zach/Documents/School/Boulder/Graduate/Regueiro/Code/Python/output/neo-Hookean/Thermoelastodynamics-Verification/Heat-Conduction/analytical/1s-AluminumThermalMat/N50/Q1/10/'
if not os.path.exists(saveDir):
  os.makedirs(saveDir)

# Define material properties
rho     = 337       # kg/m^3
A       = 23e-6     # 1/K
c       = 941       # J/kg-K
k       = 205       # W/m-K
# A       = 2.39e-3   # 1/K
# c       = 1.215e3   # J/kg-K
# k       = 0.188     # W/m-K
b       = k/(rho*c) # m^2/s
# b       = 8.34e-5
L       = 1         # m
theta_0 = 20        # K

# Define number of series terms
nseries = 50

# Perform time integral
nX  = 100
nen = 1*nX + 1
X   = np.linspace(0, L, nen)
nX  = nen

dt = 1e-3
# mu = 0.5
# dt = mu*(X[1] - X[0])**2/b
tf = 1

N = int(tf/dt)

time     = np.zeros(N+1)
theta_n  = np.zeros((N+1, nX, nseries))
theta    = np.zeros((N+1, nX))

# Loop over time 't':
for i in range(N+1):
  time[i] = i*dt

  print("Generating solution at time t = %.3f s" %(time[i]))
  # Loop over positions
  for k in range(nX):

    # Loop over sine series
    for n in range(0,nseries):
      term1 = get_fourierSine(n, X[k])
      term2 = get_exponent(n, time[i])
      
      theta_n[i,k,n] = term1*term2/((2*n + 1)*np.pi)

    theta[i,k] = theta_0 - 4*theta_0*np.sum(theta_n[i,k,:], axis=0)

plt.plot(X, theta[-1,:])
plt.show()

np.save(saveDir + 'time.npy',         time, allow_pickle=False) 
np.save(saveDir + 'temperature.npy',  theta,allow_pickle=False)