#--------------------------------------------------------------------------------------
# Module to initiate problem parameters and call a time integrator for the
# constitutive lung model. Does not solve finite-element equations, just stress.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   November 2, 2022
#--------------------------------------------------------------------------------------
import sys, time, traceback

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

import classElement

def main(Parameters):

  start_time = time.time()
  #-------------------------------------
  # Initialize time solution parameters.
  #-------------------------------------
  Parameters.t  = Parameters.TStart
  Parameters.dt = Parameters.dt0

  n_steps            = int(np.ceil((Parameters.TStop - Parameters.TStart)/Parameters.dt0))
  Parameters.n_save  = int(n_steps/Parameters.TOutput)
  Parameters.dt_save = Parameters.dt0*Parameters.n_save
  t_previous         = Parameters.t
  Parameters.t_save  = t_previous + Parameters.dt_save
  if Parameters.dt_save == 0.0:
    sys.exit("ERROR. Too many data points requested for initial time step, check inputs.")

  SS_solve  = np.zeros((Parameters.TOutput+1, Parameters.nel, 1, 1), dtype=np.float64)
  #--------------
  # Apply the IC.
  #--------------
  D    = np.zeros((2), dtype=np.float64)
  V    = np.copy(D)
  V[1] = -0.1
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False

  n = 0
  m = 0

  Parameters.dtnew = Parameters.dt

  try:
    while Parameters.t + 1e-10 < Parameters.TStop:
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #---------------------
      # Update displacement.
      #---------------------
      D[1] += Parameters.dt*V[1]
      #-------------------------------
      # Solve stress at element level.
      #-------------------------------
      element = classElement.Element(a_GaussOrder=1, a_ID=0)
      element.set_Gauss_Points()
      element.set_Gauss_Weights()
      element.set_Jacobian(Parameters)
      element.evaluate_Shape_Functions(Parameters)
      element.set_u_s_global(D)
      element.set_v_s_global(V)
      element.u_s_global[0] = 0
      element.v_s_global[0] = 0
      element.get_dudX()
      element.get_dvdX()
      element.get_F11()
      element.get_J()
      element.get_P11(Parameters)

      if n % Parameters.n_save == 0:
        save_flag = True

      if save_flag:
        print("Saving at t = {:.3e}s".format(Parameters.t))

        SS_solve[m+1,0,0,:] = element.P11
        m                  += 1
        save_flag           = False

  except (FloatingPointError, RuntimeError):
    print("ERROR. Encounted FloatingPointError or RuntimeError.")
    print(traceback.format_exc())
    pass

  P11_fname = Parameters.dataDir + 'P11.npy'
  np.save(P11_fname, SS_solve[:,:,0,:],  allow_pickle=False)
  np.save(Parameters.dataDir + 'time.npy', np.linspace(0, Parameters.t, Parameters.TOutput+1), allow_pickle=False)

