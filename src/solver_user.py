#--------------------------------------------------------------------------------------
# Module housing a functional form of a Runge-Kutta solver.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   February 18, 2022
#--------------------------------------------------------------------------------------
import sys, os, time

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import matplotlib.pyplot as plt
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
  import moduleRK
except ImportError:
  sys.exit("MODULE WARNING. moduleRK.py not found, check configuration.")

#------------------------------------------
# Two degree of freedom spring mass system.
#------------------------------------------
def springmass(a_T, a_Y):
  f  = np.zeros(4, dtype=np.float64)

  k1 = 1
  k2 = 0.01
  m1 = 1
  m2 = 1

  f[0] = a_Y[1]
  f[1] = (-k1/m1)*a_Y[0] + (k2/m1)*(a_Y[2] - a_Y[0])
  f[2] = a_Y[3]
  f[3] = (-k2/m2)*(a_Y[2] - a_Y[0])

  return f

def main(params):

  start_time = time.time()
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((params.numRKStages, 4), dtype=np.float64)
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  y_solve   = np.zeros((params.TOutput + 1, 4), dtype=np.float64)
  t_solve   = np.zeros((params.TOutput + 1),    dtype=np.float64)
  dt_solve  = np.zeros((int(2e4)), dtype=np.float64)
  tdt_solve = np.zeros((int(2e4)), dtype=np.float64)
  #--------------
  # Apply the IC.
  #--------------
  y0 = np.array([0,0,1,0], dtype=np.float64)
  y  = y0[:]

  print("Solving...")

  acceptSolution = True
  save_flag      = False

  n = 0
  m = n

  while params.t < params.TStop + 1e-12:
    #-----------------------------------------------------------------
    # Check that the time step does not drop below some minimum value.
    #-----------------------------------------------------------------
    if params.dt < params.adaptiveDTMin:
      print("ERROR. Time step dropped below user-defined tolerance.")
      break

    print("n:", n, "t: %fs" %params.t)

    computeStages = True

    #------------------------
    # Update simulation time.
    #------------------------
    if acceptSolution:
      params.t += params.dt
      n        += 1

    while computeStages:
      for i in range(params.numRKStages):
        #--------------------------------
        # Compute intermediate increment.
        #--------------------------------
        y_pred = moduleRK.get_Stage_Predictor(i, y, params.dt*k_i, params)
        #-------------------------
        # Compute stage increment.
        #-------------------------
        k_i[i,:] = springmass(params.t + params.Ci[i]*params.dt, y_pred)
      
      #-------------------
      # Compute solutions.
      #-------------------
      z, error = moduleRK.get_Error(k_i, params)

      if error < 1:
        y             += z
        acceptSolution = True
        computeStages  = False
      else:
        acceptSolution = False

    #----------------------------------------
    # Adjust time step if it grows too large.
    #----------------------------------------
    if params.dt > params.adaptiveDTMax:
      params.dt = params.adaptiveDTMax

    if acceptSolution:
      params.dt0 = params.dt
      t_previous = params.t

      #------------------------------
      # Save adaptive time step data.
      #------------------------------
      dt_solve[n+1]  = params.dt
      tdt_solve[n+1] = params.t

      if save_flag:
        print("Storing solution at t = %fs" %params.t)
        #--------------------
        # Save solution data.
        #--------------------
        y_solve[m+1,:] = y
        t_solve[m+1]   = params.t

        m        += 1
        save_flag = False

      if (t_previous + params.dt >= t_save):
        if t_save - t_previous > 1e-10:
          params.dt = t_save - t_previous
        t_save    += dt_save
        save_flag  = True

  end_time = time.time()
  time_elapsed = (end_time - start_time)/60.0
  print("Finished solving in %.2f minutes." %time_elapsed)
  print("Saving data...")

  #----------------------------------------------------
  # Initialize file names for plotting/post-processing.
  #----------------------------------------------------
  y_fname    = params.dataDir + 'y.npy' 
  time_fname = params.dataDir + 'time.npy'        
  
  np.save(y_fname,    y_solve, allow_pickle=False)
  np.save(time_fname, t_solve, allow_pickle=False)

  print("Data saved successfully.")

  plt.figure(1)
  plt.plot(t_solve, y_solve[:,0], label='x_1')
  plt.plot(t_solve, y_solve[:,2], label='x_2')
  plt.xlabel('time t')
  plt.ylabel('solution x')
  plt.legend()
  plt.grid()

  plt.figure(2)
  plt.plot(y_solve[:,0], y_solve[:,2], 'r-')
  plt.xlabel('solution x_1')
  plt.ylabel('solution x_2')
  plt.grid()

  idx       = np.where(tdt_solve == 0)
  tdt_solve = np.delete(tdt_solve, idx)
  dt_solve  = np.delete(dt_solve, idx)
  plt.figure(3)
  plt.plot(tdt_solve, dt_solve, 'b-+')
  plt.xlabel('t')
  plt.ylabel('dt')
  plt.grid()
  plt.show()

  return
