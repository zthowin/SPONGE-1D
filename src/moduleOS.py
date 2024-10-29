#--------------------------------------------------------------------------------------
# Module housing helper functions for semi-explicit methods.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   June 8, 2022
#--------------------------------------------------------------------------------------
import sys, traceback

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import classElement
except ImportError:
  sys.exit("MODULE WARNING. classElement.py not found, check configuration.")

try:
  import moduleFE
except ImportError:
  sys.exit("MODULE WARNING. moduleFE.py not found, check configuration.")

try:
  import moduleVE
except ImportError:
  sys.exit("MODULE WARNING. moduleVE.py not found, check configuration.")

#--------------------------------------------------------------------------------------------------
# Helper function to integrate a system of variational equations using second-order
# operator-splitting method.
#
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# g:            (float, size: # element DOFs, # elements)  Dirichlet boundary conditions
# F:            (float, size: varies)                      external traction vectors
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
#                                                 
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#
#--------------------------------------------------------------------------------------------------
def integrate_upf(LM, g, F, D, V, A, Parameters):
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  F_F = F[0]
  F_Q = F[1]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.zeros((Parameters.TOutput+3, Parameters.ndof),                       dtype=np.float64)
  V_solve   = np.zeros((Parameters.TOutput+3, Parameters.ndof),                       dtype=np.float64)
  A_solve   = np.zeros((Parameters.TOutput+3, Parameters.ndof),                       dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                        dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.nel, Parameters.nstress, Parameters.Gauss_Order), dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.nel, Parameters.nisv,    Parameters.Gauss_Order), dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #--------------
  # Apply the IC.
  #--------------
  D_solve[0,:] = D[:]
  V_solve[0,:] = V[:]
  A_solve[0,:] = A[:]

  SS_solve[0,:,2,:]  = 1.
  ISV_solve[0,:,0,:] = Parameters.p_f0
  ISV_solve[0,:,1,:] = Parameters.p_f0
  ISV_solve[0,:,3,:] = Parameters.ns_0
  ISV_solve[0,:,4,:] = Parameters.rhofR_0
  ISV_solve[0,:,5,:] = Parameters.khat

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  
  Vtilde                 = np.zeros((Parameters.ndof))
  Dtilde                 = np.zeros((Parameters.ndof))
  AStar                  = np.zeros((Parameters.ndof))
  D_Last                 = np.zeros((Parameters.ndof))
  Dtilde[:]              = D[:]
  Vtilde[:]              = V[:]
  AStar[:]               = A[:]
  D_Last[:]              = D[:]
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False

  n = 0
  m = n

  try:
    while Parameters.t + 1e-10 < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0.0:
        Parameters.dt = Parameters.TStop - Parameters.t
      #-----------------------------------------------------------------
      # Check that the time step does not drop below some minimum value.
      #-----------------------------------------------------------------
      if Parameters.dt < Parameters.adaptiveDTMin:
        print("ERROR. Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      # print(Parameters.t)
      #-----------------
      # Update traction.
      #-----------------
      F_F[Parameters.tractionDOF] = moduleFE.applyTraction(Parameters.t, Parameters)
      #-----------------------
      # Update the predictors.
      #-----------------------
      try:
        Vtilde[0:Parameters.ndofS]               = V[0:Parameters.ndofS] + (1 - Parameters.gamma)*Parameters.dt*A[0:Parameters.ndofS]
        Dtilde[0:Parameters.ndofS]               = D[0:Parameters.ndofS] + Parameters.dt*V[0:Parameters.ndofS] + (Parameters.dt**2/2)*A[0:Parameters.ndofS]
        Dtilde[Parameters.ndofS:Parameters.ndof] = D[Parameters.ndofS:Parameters.ndof] + (1 - Parameters.alpha)*Parameters.dt*V[Parameters.ndofS:Parameters.ndof]
      except FloatingPointError:
        print("ERROR. Encountered under/overflow in update of predictors.at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt)) 
        raise FloatingPointError
      #-------------------------------------
      # Calculate intermediate acceleration.
      #-------------------------------------
      D[0:Parameters.ndofS]     = Dtilde[0:Parameters.ndofS]
      dR_s, R_s                 = moduleVE.assemble_G(LM, g, F_F, D, V, A, Parameters)
      AStar[0:Parameters.ndofS] = A[0:Parameters.ndofS] + moduleVE.solve_G(dR_s, R_s, Parameters)
      #---------------------------------
      # Integrate intermediate velocity.
      # 
      # This is left in the code as
      # v_n+1 to reduce the number of 
      # variables used when iterating
      # over the pressure time 
      # derivative. It is overwritten
      # with the *actual* v_n+1 after
      # pressure has converged. 
      #---------------------------------
      try:
        V[0:Parameters.ndofS] = Vtilde[0:Parameters.ndofS] + (Parameters.gamma*Parameters.dt)*AStar[0:Parameters.ndofS]
      except FloatingPointError:
        print("ERROR. Encountered under/overflow in update of intermediate velocity at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise FloatingPointError
      #--------------------------
      # Filter out small numbers.
      #--------------------------
      D[np.abs(D) < 1e-20] = 0.
      V[np.abs(V) < 1e-20] = 0.
      #----------------------------------------------
      # Begin Newton-Raphson iterations for pressure.
      #----------------------------------------------
      Rtol    = 1
      normR   = 1
      k       = 0
      while (Rtol > Parameters.tolr) and (normR > Parameters.tola):
        #----------------------------------
        # Increment iterator and solutions.
        #----------------------------------
        k += 1
        #------------------------------------
        # Form pressure tangent and residual.
        #------------------------------------
        dR_p, R_p = moduleVE.assemble_H(LM, g, F_Q, D, V, AStar, Parameters)
        #----------------------------------
        # Compute pressure velocity update.
        #----------------------------------
        del_PDot = moduleVE.solve_H(dR_p, R_p, Parameters)
        #----------------------------
        # Compute tolerance and norm.
        #----------------------------
        if (k == 1):
          R_p0 = R_p

        try:
          Rtol  = np.linalg.norm(R_p, ord=Parameters.norm_ord)/np.linalg.norm(R_p0, ord=Parameters.norm_ord)
          normR = np.linalg.norm(R_p, ord=Parameters.norm_ord)
        except FloatingPointError:
          print("ERROR. Could not calculate tolerances at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise FloatingPointError
        #------------------
        # Check iterations.
        #------------------
        if k == Parameters.kmax:
          print("Relative tolerance = {:.3e}".format(Rtol))
          print("ERROR. Reached max number of iterations for simulations at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt)) 
          raise RuntimeError
        #---------------------
        # Increment solutions.
        #---------------------
        try:
          V[Parameters.ndofS:Parameters.ndof] += del_PDot
          D[Parameters.ndofS:Parameters.ndof]  = Dtilde[Parameters.ndofS:Parameters.ndof] + Parameters.alpha*Parameters.dt*V[Parameters.ndofS:Parameters.ndof]
          #--------------------------
          # Filter out small numbers.
          #--------------------------
          D[np.abs(D) < 1e-20] = 0.
          V[np.abs(V) < 1e-20] = 0.
        except FloatingPointError:
          print("ERROR. Encountered under/overflow in integration at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt)) 
          raise FloatingPointError
      #-------------------------------
      # Calculate acceleration update.
      #-------------------------------
      Parameters.StarStar   = True
      dR_s, R_s             = moduleVE.assemble_G_StarStar(LM, g, F_F, D, V, A, Parameters, D_Last)
      A[0:Parameters.ndofS] = AStar[0:Parameters.ndofS] + moduleVE.solve_G(dR_s, R_s, Parameters)
      Parameters.StarStar   = False
      #--------------------------
      # Integrate solid velocity.
      #--------------------------
      V[0:Parameters.ndofS] = Vtilde[0:Parameters.ndofS] + Parameters.gamma*Parameters.dt*A[0:Parameters.ndofS]
      #--------------------------
      # Filter out small numbers.
      #--------------------------
      D[np.abs(D) < 1e-20] = 0.
      V[np.abs(V) < 1e-20] = 0.
      A[np.abs(A) < 1e-20] = 0.

      t_previous = Parameters.t
      if n % Parameters.n_save == 0:
        save_flag = True

      if save_flag:
        print("Saving at t = {:.3e}s".format(Parameters.t))
        #--------------------------------
        # Compute stress/strain at t_n+1.
        #--------------------------------
        try:
          dR_s, R_s, SS, ISV = moduleVE.assemble_G(LM, g, F_F, D, V, A, Parameters, True)
        except FloatingPointError:
          print("ERROR. FloatingPointError encountered trying to save data at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt)) 
          raise FloatingPointError
        #--------------------
        # Save solution data.
        #--------------------
        D_solve[m+1,:]       = D[:]
        V_solve[m+1,:]       = V[:]
        A_solve[m+1,:]       = A[:]
        t_solve[m+1]         = Parameters.t
        SS_solve[m+1,:,:,:]  = SS[:]
        ISV_solve[m+1,:,:,:] = ISV[:]

        m        += 1
        save_flag = False

      Parameters.dt0 = Parameters.dt
      D_Last[:] = D[:]

  except (FloatingPointError, RuntimeError):
    print("ERROR. Encountered FloatingPointError or RuntimeError.")
    print(traceback.format_exc())
    pass

  return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
