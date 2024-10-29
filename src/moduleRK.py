#--------------------------------------------------------------------------------------
# Module housing helper functions for Runge-Kutta methods.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   October 16, 2024
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
# Helper function to integrate the variational equation for balance of momentum of the solid.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # 3, # elements)  location matrix
# GEXT:         (float, size: # skeleton DOFs)  external traction vector
# D:            (float, size: # skeleton DOFs)  global IC for displacement
# V:            (float, size: # skeleton DOFs)  global IC for velocity
# A:            (float, size: # skeleton DOFs)  global IC for acceleration
# Parameters:   (object)                        problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # skeleton DOFs)             global solution for displacement
# V_solve:   (float, size: # save times, # skeleton DOFs)             global solution for velocity
# A_solve:   (float, size: # save times, # skeleton DOFs)             global solution for acceleration
# t_solve:   (float, size: # save times)                              simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_u_FO(LM, GEXT, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS), dtype=np.float64)
  #---------------------------------------------------------
  # Set start:stop indices for solution variables in stages.
  #---------------------------------------------------------
  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = ndofSu_e
  ndofSv_e = ndofSv_s + Parameters.ndofS
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT                                  = moduleFE.updateBC(g, GEXT, 0.0, Parameters)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,3,:] = Parameters.ns_0

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n
  t_previous = Parameters.t

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          Parameters.tk = Parameters.t + Parameters.dt*(Parameters.Ci[i] - 1)
          g, GEXT       = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters)
          #----------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #----------------------------------
          try:
            D_pred = get_Stage_Predictor(i, D, Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            V_pred = get_Stage_Predictor(i, V, Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #------------------------------
          # Balance of momentum of solid.
          #------------------------------
          dR, R = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A     = moduleVE.solve_G(dR, R, Parameters)
          #------------------------
          # Update stage increment.
          #------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[:]
          k_i[i, ndofSv_s:ndofSv_e] = A[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D += z[ndofSu_s:ndofSu_e]
              V += z[ndofSv_s:ndofSv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              Parameters.tk = Parameters.t - Parameters.dt0 + Parameters.dt
              g, GEXT       = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters)
              g[1,:]        = g[0,:]
              #-----------------------------------------------
              # Recompute acceleration for next time increment
              # using the accepted solutions at t_{n+1}.
              #-----------------------------------------------
              dR, R = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A     = moduleVE.solve_G(dR, R, Parameters)
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encountered over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution = True
            computeStages  = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D += z[ndofSu_s:ndofSu_e]
            V += z[ndofSv_s:ndofSv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters)
            g[1,:]  = g[0,:]
            #-----------------------------------------------
            # Recompute acceleration for next time increment
            # using the accepted solutions at t_{n+1}.
            #-----------------------------------------------
            dR, R = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A     = moduleVE.solve_G(dR, R, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution = True
          computeStages  = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous     = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t
  
        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_u.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum and balance of
# energy (single-phase).
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# F:            (float, size: varies)                      Neumann BCs
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_ut_FO(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS + Parameters.ndofTs), dtype=np.float64)

  D_pred = np.zeros(Parameters.ndof)
  V_pred = np.zeros(Parameters.ndof)

  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = ndofSu_e
  ndofSv_e = ndofSv_s + Parameters.ndofS
  ndofTs_s = ndofSv_e
  ndofTs_e = ndofTs_s + Parameters.ndofTs

  dofS_s  = 0
  dofS_e  = Parameters.ndofS
  dofTs_s = dofS_e
  dofTs_e = dofTs_s + Parameters.ndofTs
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  GEXT = F[0]
  JEXT = F[1]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT, JEXT                            = moduleFE.updateBC(g, GEXT, 0.0, Parameters, JEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_pred[:] = D[:]
  V_pred[:] = V[:]

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,7,:] = Parameters.Ts_0

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          Parameters.tk = Parameters.t + Parameters.dt*(Parameters.Ci[i] - 1)
          g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, JEXT)
          #----------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #
          # Order:
          # u, \theta
          # v
          #----------------------------------
          try:
            D_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, D[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            D_pred[dofTs_s:dofTs_e] = get_Stage_Predictor(i, D[dofTs_s:dofTs_e], Parameters.dt*k_i[:,ndofTs_s:ndofTs_e], Parameters)
            V_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, V[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #------------------------------
          # Balance of momentum of solid.
          #------------------------------
          dR_s, R_s   = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A_s         = moduleVE.solve_G(dR_s, R_s, Parameters)
          #--------------------------------
          # Balance of energy of the solid.
          #--------------------------------
          dR_ts, R_ts = moduleVE.assemble_J(LM, g, JEXT, D_pred, V_pred, A, Parameters)
          Tsdot       = moduleVE.solve_J(dR_ts, R_ts, Parameters)
          #-------------------------
          # Compute stage increment.
          #-------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[dofS_s:dofS_e]
          k_i[i, ndofSv_s:ndofSv_e] = A_s[:]
          k_i[i, ndofTs_s:ndofTs_e] = Tsdot[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
              D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
              V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              Parameters.tk = Parameters.t - Parameters.dt0 + Parameters.dt
              g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, JEXT)
              g[1,:]        = g[0,:]
              #-----------------------------------------------
              # Recompute acceleration for next time increment
              # using the accepted solutions at t_{n+1}.
              #-----------------------------------------------
              dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
              A[dofS_s:dofS_e] = A_s[:]
              #---------------------------------------------------------
              # Recompute solid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
              Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
              V[dofTs_s:dofTs_e] = Tsdot[:]
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encountered over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution               = True
            computeStages                = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
            D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
            V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, JEXT)
            g[1,:]        = g[0,:]
            #-----------------------------------------------
            # Recompute acceleration for next time increment
            # using the accepted solutions at t_{n+1}.
            #-----------------------------------------------
            dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
            A[dofS_s:dofS_e] = A_s[:]
            #---------------------------------------------------------
            # Recompute solid temperature rate for next time increment
            # using the accepted solutions at t_{n+1}.
            #---------------------------------------------------------
            dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
            Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
            V[dofTs_s:dofTs_e] = Tsdot[:]
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution               = True
          computeStages                = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous     = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t

        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_ut.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum and balance of 
# mass of the mixture.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# F:            (float, size: varies)                      Neumann BCs
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_upf_FO(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS + Parameters.ndofP), dtype=np.float64)

  D_pred = np.zeros(Parameters.ndof)
  V_pred = np.zeros(Parameters.ndof)

  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = ndofSu_e
  ndofSv_e = ndofSv_s + Parameters.ndofS
  ndofP_s  = ndofSv_e
  ndofP_e  = ndofP_s  + Parameters.ndofP

  dofS_s  = 0
  dofS_e  = Parameters.ndofS
  dofP_s  = dofS_e
  dofP_e  = dofP_s + Parameters.ndofP
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  GEXT = F[0]
  HEXT = F[1]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT, HEXT                            = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_pred[:] = D[:]
  V_pred[:] = V[:]

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,3,:] = Parameters.ns_0
  ISV_solve[0,:,4,:] = Parameters.rhofR_0
  ISV_solve[0,:,5,:] = Parameters.khat

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          Parameters.tk = Parameters.t + Parameters.dt*(Parameters.Ci[i] - 1)
          g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT)
          #----------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #
          # Order:
          # u, p_\rf
          # v
          #----------------------------------
          try:
            D_pred[dofS_s:dofS_e] = get_Stage_Predictor(i, D[dofS_s:dofS_e], Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            D_pred[dofP_s:dofP_e] = get_Stage_Predictor(i, D[dofP_s:dofP_e], Parameters.dt*k_i[:,ndofP_s:ndofP_e],   Parameters)
            V_pred[dofS_s:dofS_e] = get_Stage_Predictor(i, V[dofS_s:dofS_e], Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #--------------------------------
          # Balance of momentum of mixture.
          #--------------------------------
          dR_s, R_s   = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A_s         = moduleVE.solve_G(dR_s, R_s, Parameters)
          #-------------------------------
          # Update A with \dot{z}_v (f_a).
          #-------------------------------
          A[dofS_s:dofS_e] = A_s[:]
          #----------------------------
          # Balance of mass of mixture.
          #----------------------------
          dR_p, R_p = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
          Pdot      = moduleVE.solve_H(dR_p, R_p, Parameters)
          #------------------------
          # Update stage increment.
          #------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[dofS_s:dofS_e]
          k_i[i, ndofSv_s:ndofSv_e] = A_s[:]
          k_i[i, ndofP_s:ndofP_e]   = Pdot[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D[dofS_s:dofS_e] += z[ndofSu_s:ndofSu_e]
              D[dofP_s:dofP_e] += z[ndofP_s:ndofP_e]
              V[dofS_s:dofS_e] += z[ndofSv_s:ndofSv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              Parameters.tk = Parameters.t - Parameters.dt0 + Parameters.dt
              g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT)
              g[1,:]        = g[0,:]
              #-----------------------------------------------
              # Recompute acceleration for next time increment
              # using the accepted solutions at t_{n+1}.
              #-----------------------------------------------
              dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
              A[dofS_s:dofS_e] = A_s[:]
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encountered over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution = True
            computeStages  = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D[dofS_s:dofS_e] += z[ndofSu_s:ndofSu_e]
            D[dofP_s:dofP_e] += z[ndofP_s:ndofP_e]
            V[dofS_s:dofS_e] += z[ndofSv_s:ndofSv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
            g[1,:]        = g[0,:]
            #-----------------------------------------------
            # Recompute acceleration for next time increment
            # using the accepted solutions at t_{n+1}.
            #-----------------------------------------------
            dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
            A[dofS_s:dofS_e] = A_s[:]
            #------------------------------------------------
            # Recompute pressure rate for next time increment
            # using the accepted solutions at t_{n+1}.
            #------------------------------------------------
            dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
            Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
            V[dofP_s:dofP_e] = Pdot[:]
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution               = True
          computeStages                = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t

        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_upf.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum and balance of 
# mass of the mixture, and balance of momentum of the fluid.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# F:            (float, size: varies)                      Neumann BCs
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_uufpf_FO(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS + 2*Parameters.ndofF + Parameters.ndofP), dtype=np.float64)

  D_pred = np.zeros(Parameters.ndof)
  V_pred = np.zeros(Parameters.ndof)

  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = Parameters.ndofS
  ndofSv_e = ndofSv_s + Parameters.ndofS
  ndofFu_s = ndofSv_e
  ndofFu_e = ndofFu_s + Parameters.ndofF
  ndofFv_s = ndofFu_e
  ndofFv_e = ndofFv_s + Parameters.ndofF
  ndofP_s  = ndofFv_e
  ndofP_e  = ndofP_s  + Parameters.ndofP

  dofS_s  = 0
  dofS_e  = Parameters.ndofS
  dofF_s  = dofS_e
  dofF_e  = dofF_s + Parameters.ndofF
  dofP_s  = dofF_e
  dofP_e  = dofP_s + Parameters.ndofP
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  GEXT = F[0]
  HEXT = F[1]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT, HEXT                            = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_pred[:] = D[:]
  V_pred[:] = V[:]

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,3,:] = Parameters.ns_0
  ISV_solve[0,:,4,:] = Parameters.rhofR_0
  ISV_solve[0,:,5,:] = Parameters.khat

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          Parameters.tk = Parameters.t + Parameters.dt*(Parameters.Ci[i] - 1)
          g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT)
          #----------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #
          # Order:
          # u, u_\rf, p_\rf
          # v, v_\rf
          #----------------------------------
          try:
            D_pred[dofS_s:dofS_e] = get_Stage_Predictor(i, D[dofS_s:dofS_e], Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            D_pred[dofF_s:dofF_e] = get_Stage_Predictor(i, D[dofF_s:dofF_e], Parameters.dt*k_i[:,ndofFu_s:ndofFu_e], Parameters)
            D_pred[dofP_s:dofP_e] = get_Stage_Predictor(i, D[dofP_s:dofP_e], Parameters.dt*k_i[:,ndofP_s:ndofP_e],   Parameters)
            V_pred[dofS_s:dofS_e] = get_Stage_Predictor(i, V[dofS_s:dofS_e], Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
            V_pred[dofF_s:dofF_e] = get_Stage_Predictor(i, V[dofF_s:dofF_e], Parameters.dt*k_i[:,ndofFv_s:ndofFv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #---------------------------------------
          # Balance of momentum of the pore fluid.
          #---------------------------------------
          dR_f, R_f = moduleVE.assemble_I(LM, g, D_pred, V_pred, A, Parameters)
          A_f       = moduleVE.solve_I(dR_f, R_f, Parameters)
          #---------------------------------
          # Update A with \dot{z}_vf (f_af).
          #---------------------------------
          A[dofF_s:dofF_e] = A_f[:]
          #-------------------------------------------
          # Balance of momentum of the solid skeleton.
          #-------------------------------------------
          dR_s, R_s = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A_s       = moduleVE.solve_G(dR_s, R_s, Parameters)
          #-------------------------------
          # Update A with \dot{z}_v (f_a).
          #-------------------------------
          A[dofS_s:dofS_e] = A_s[:]
          #----------------------------
          # Balance of mass of mixture.
          #----------------------------
          dR_p, R_p = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
          Pdot      = moduleVE.solve_H(dR_p, R_p, Parameters)
          #------------------------
          # Update stage increment.
          #------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[dofS_s:dofS_e]
          k_i[i, ndofSv_s:ndofSv_e] = A_s[:]
          k_i[i, ndofFu_s:ndofFu_e] = V_pred[dofF_s:dofF_e]
          k_i[i, ndofFv_s:ndofFv_e] = A_f[:]
          k_i[i, ndofP_s:ndofP_e]   = Pdot[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D[dofS_s:dofS_e] += z[ndofSu_s:ndofSu_e]
              D[dofF_s:dofF_e] += z[ndofFu_s:ndofFu_e]
              D[dofP_s:dofP_e] += z[ndofP_s:ndofP_e]
              V[dofS_s:dofS_e] += z[ndofSv_s:ndofSv_e]
              V[dofF_s:dofF_e] += z[ndofFv_s:ndofFv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              Parameters.tk = Parameters.t - Parameters.dt0 + Parameters.dt
              g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
              g[1,:]        = g[0,:]
              #------------------------------------------------
              # Recompute accelerations for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_f, R_f        = moduleVE.assemble_I(LM, g, D, V, A, Parameters)
              A_f              = moduleVE.solve_I(dR_f, R_f, Parameters)
              A[dofF_s:dofF_e] = A_f[:]
              dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
              A[dofS_s:dofS_e] = A_s[:]
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encountered over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution               = True
            computeStages                = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D[dofS_s:dofS_e] += z[ndofSu_s:ndofSu_e]
            D[dofF_s:dofF_e] += z[ndofFu_s:ndofFu_e]
            D[dofP_s:dofP_e] += z[ndofP_s:ndofP_e]
            V[dofS_s:dofS_e] += z[ndofSv_s:ndofSv_e]
            V[dofF_s:dofF_e] += z[ndofFv_s:ndofFv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
            g[1,:]        = g[0,:]
            #------------------------------------------------
            # Recompute accelerations for next time increment
            # using the accepted solutions at t_{n+1}.
            #------------------------------------------------
            dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
            dR_f, R_f        = moduleVE.assemble_I(LM, g, D, V, A, Parameters)
            A_f              = moduleVE.solve_I(dR_f, R_f, Parameters)
            A[dofS_s:dofS_e] = A_s[:]
            A[dofF_s:dofF_e] = A_f[:]
            #------------------------------------------------
            # Recompute pressure rate for next time increment
            # using the accepted solutions at t_{n+1}.
            #------------------------------------------------
            dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
            Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
            V[dofP_s:dofP_e] = Pdot[:]
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution               = True
          computeStages                = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous     = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t

        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_uufpf.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum and balance of 
# mass of the mixture, and balance of energy of each phase.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# F:            (float, size: varies)                      Neumann BCs
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_upftstf_FO(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS + Parameters.ndofP\
                                          + Parameters.ndofTs + Parameters.ndofTf), dtype=np.float64)

  D_pred = np.zeros(Parameters.ndof)
  V_pred = np.zeros(Parameters.ndof)

  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = Parameters.ndofS
  ndofSv_e = 2*Parameters.ndofS
  ndofP_s  = 2*Parameters.ndofS
  ndofP_e  = 2*Parameters.ndofS + Parameters.ndofP
  ndofTs_s = ndofP_e
  ndofTs_e = ndofTs_s + Parameters.ndofTs
  ndofTf_s = ndofTs_e
  ndofTf_e = ndofTs_e + Parameters.ndofTf

  dofS_s  = 0
  dofS_e  = Parameters.ndofS
  dofP_s  = dofS_e
  dofP_e  = dofP_s + Parameters.ndofP
  dofTs_s = dofP_e
  dofTs_e = dofTs_s + Parameters.ndofTs
  dofTf_s = dofTs_e
  dofTf_e = dofTf_s + Parameters.ndofTf
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  GEXT = F[0]
  HEXT = F[1]
  JEXT = F[2]
  KEXT = F[3]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT, HEXT, JEXT, KEXT                = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT, JEXT, KEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_pred[:] = D[:]
  V_pred[:] = V[:]

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,3,:] = Parameters.ns_0
  ISV_solve[0,:,4,:] = Parameters.rhofR_0
  ISV_solve[0,:,5,:] = Parameters.khat
  ISV_solve[0,:,7,:] = Parameters.Ts_0
  ISV_solve[0,:,8,:] = Parameters.Tf_0

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          Parameters.tk             = Parameters.t + Parameters.dt*Parameters.Ci[i]
          g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT, JEXT, KEXT)
          #----------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #
          # Order:
          # u, p_\rf, \theta^\rs, \theta^\rf
          # v
          #----------------------------------
          try:
            D_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, D[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            D_pred[dofP_s:dofP_e]   = get_Stage_Predictor(i, D[dofP_s:dofP_e],   Parameters.dt*k_i[:,ndofP_s:ndofP_e],   Parameters)
            D_pred[dofTs_s:dofTs_e] = get_Stage_Predictor(i, D[dofTs_s:dofTs_e], Parameters.dt*k_i[:,ndofTs_s:ndofTs_e], Parameters)
            D_pred[dofTf_s:dofTf_e] = get_Stage_Predictor(i, D[dofTf_s:dofTf_e], Parameters.dt*k_i[:,ndofTf_s:ndofTf_e], Parameters)
            V_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, V[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #--------------------------------
          # Balance of momentum of mixture.
          #--------------------------------
          dR_s, R_s   = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A_s         = moduleVE.solve_G(dR_s, R_s, Parameters)
          #-------------------------------
          # Update A with \dot{z}_v (f_a).
          #-------------------------------
          A[dofS_s:dofS_e] = A_s[:]
          #-----------------------------------
          # Balance of energy of the solid.
          # (no V_pred update needed for K, H)
          #-----------------------------------
          if not Parameters.isothermalAssumption:
            dR_ts, R_ts = moduleVE.assemble_J(LM, g, JEXT, D_pred, V_pred, A, Parameters)
            Tsdot       = moduleVE.solve_J(dR_ts, R_ts, Parameters)
          #---------------------------
          # Solve p_fDot before tfDot.
          #---------------------------
          if not Parameters.staggered:
            #----------------------------
            # Balance of mass of mixture.
            #----------------------------
            dR_p, R_p   = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
            Pdot        = moduleVE.solve_H(dR_p, R_p, Parameters)
            #--------------------------------------------
            # Update V_pred with \dot{z}_pf (f_\dot{pf}).
            #--------------------------------------------
            V_pred[dofP_s:dofP_e] = Pdot[:] 
            #--------------------------------
            # Balance of energy of the fluid.
            #--------------------------------
            if not Parameters.isothermalAssumption:
              dR_tf, R_tf = moduleVE.assemble_K(LM, g, KEXT, D_pred, V_pred, A, Parameters)
              Tfdot       = moduleVE.solve_K(dR_tf, R_tf, Parameters)
          #---------------------------
          # Solve tfDot before p_fDot.
          #---------------------------
          else:
            #--------------------------------
            # Balance of energy of the fluid.
            #--------------------------------
            if not Parameters.isothermalAssumption:
              dR_tf, R_tf = moduleVE.assemble_K(LM, g, KEXT, D_pred, V_pred, A, Parameters)
              Tfdot       = moduleVE.solve_K(dR_tf, R_tf, Parameters)
            #--------------------------------------------
            # Update V_pred with \dot{z}_tf (f_\dot{tf}).
            #--------------------------------------------
              V_pred[dofTf_s:dofTf_e] = Tfdot[:] 
            #----------------------------
            # Balance of mass of mixture.
            #----------------------------
            dR_p, R_p   = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
            Pdot        = moduleVE.solve_H(dR_p, R_p, Parameters)
          #------------------------
          # Update stage increment.
          #------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[dofS_s:dofS_e]
          k_i[i, ndofSv_s:ndofSv_e] = A_s[:]
          k_i[i, ndofP_s:ndofP_e]   = Pdot[:]
          if not Parameters.isothermalAssumption:
            k_i[i, ndofTs_s:ndofTs_e] = Tsdot[:]
            k_i[i, ndofTf_s:ndofTf_e] = Tfdot[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
              D[dofP_s:dofP_e]   += z[ndofP_s:ndofP_e]
              D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
              D[dofTf_s:dofTf_e] += z[ndofTf_s:ndofTf_e]
              V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              Parameters.tk             = Parameters.t - Parameters.dt0 + Parameters.dt
              g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT, JEXT, KEXT)
              g[1,:]                    = g[0,:]
              #-----------------------------------------------
              # Recompute acceleration for next time increment
              # using the accepted solutions at t_{n+1}.
              #-----------------------------------------------
              dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
              A[dofS_s:dofS_e] = A_s[:]
              #---------------------------------------------------------
              # Recompute solid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
                Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
                V[dofTs_s:dofTs_e] = Tsdot[:]
              #---------------------------
              # Solve p_fDot before tfDot.
              #---------------------------
              if not Parameters.staggered:
                #------------------------------------------------
                # Recompute pressure rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #------------------------------------------------
                dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
                Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
                V[dofP_s:dofP_e] = Pdot[:]
                #---------------------------------------------------------
                # Recompute fluid temperature rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #---------------------------------------------------------
                if not Parameters.isothermalAssumption:
                  dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                  Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                  V[dofTf_s:dofTf_e] = Tfdot[:]
              #---------------------------
              # Solve tfDot before p_fDot.
              #---------------------------
              else:
                #---------------------------------------------------------
                # Recompute fluid temperature rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #---------------------------------------------------------
                if not Parameters.isothermalAssumption:
                  dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                  Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                  V[dofTf_s:dofTf_e] = Tfdot[:]
                #------------------------------------------------
                # Recompute pressure rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #------------------------------------------------
                dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
                Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
                V[dofP_s:dofP_e] = Pdot[:]
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encounteed over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution               = True
            computeStages                = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
            D[dofP_s:dofP_e]   += z[ndofP_s:ndofP_e]
            D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
            D[dofTf_s:dofTf_e] += z[ndofTf_s:ndofTf_e]
            V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT, JEXT, KEXT)
            g[1,:]                    = g[0,:]
            #-----------------------------------------------
            # Recompute acceleration for next time increment
            # using the accepted solutions at t_{n+1}.
            #-----------------------------------------------
            dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
            A[dofS_s:dofS_e] = A_s[:]
            #---------------------------------------------------------
            # Recompute solid temperature rate for next time increment
            # using the accepted solutions at t_{n+1}.
            #---------------------------------------------------------
            if not Parameters.isothermalAssumption:
              dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
              Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
              V[dofTs_s:dofTs_e] = Tsdot[:]
            #---------------------------
            # Solve p_fDot before tfDot.
            #---------------------------
            if not Parameters.staggered:
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
              #---------------------------------------------------------
              # Recompute fluid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                V[dofTf_s:dofTf_e] = Tfdot[:]
            #---------------------------
            # Solve tfDot before p_fDot.
            #---------------------------
            else:
              #---------------------------------------------------------
              # Recompute fluid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                V[dofTf_s:dofTf_e] = Tfdot[:]
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution               = True
          computeStages                = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous     = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t

        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_upftstf.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum and balance of 
# mass of the mixture, balance of momentum of the fluid, and balance of energy of each phase.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements)  location matrix
# F:            (float, size: varies)                      Neumann BCs
# D:            (float, size: # DOFs)                      global IC for displacement(s)
# V:            (float, size: # DOFs)                      global IC for velocity(s)
# A:            (float, size: # DOFs)                      global IC for acceleration(s)
# Parameters:   (object)                                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration(s)
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_uufpftstf_FO(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
  #-------------------
  # Initialize stages.
  #-------------------
  k_i = np.zeros((Parameters.numRKStages, 2*Parameters.ndofS + 2*Parameters.ndofF + Parameters.ndofP\
                                          + Parameters.ndofTs + Parameters.ndofTf), dtype=np.float64)

  D_pred = np.zeros(Parameters.ndof)
  V_pred = np.zeros(Parameters.ndof)

  ndofSu_s = 0
  ndofSu_e = Parameters.ndofS
  ndofSv_s = Parameters.ndofS
  ndofSv_e = 2*Parameters.ndofS
  ndofFu_s = 2*Parameters.ndofS
  ndofFu_e = 2*Parameters.ndofS + Parameters.ndofF
  ndofFv_s = 2*Parameters.ndofS + Parameters.ndofF
  ndofFv_e = 2*Parameters.ndofS + 2*Parameters.ndofF
  ndofP_s  = 2*Parameters.ndofS + 2*Parameters.ndofF
  ndofP_e  = 2*Parameters.ndofS + 2*Parameters.ndofF + Parameters.ndofP
  ndofTs_s = ndofP_e
  ndofTs_e = ndofTs_s + Parameters.ndofTs
  ndofTf_s = ndofTs_e
  ndofTf_e = ndofTs_e + Parameters.ndofTf

  dofS_s  = 0
  dofS_e  = Parameters.ndofS
  dofF_s  = dofS_e
  dofF_e  = dofF_s + Parameters.ndofF
  dofP_s  = dofF_e
  dofP_e  = dofP_s + Parameters.ndofP
  dofTs_s = dofP_e
  dofTs_e = dofTs_s + Parameters.ndofTs
  dofTf_s = dofTs_e
  dofTf_e = dofTf_s + Parameters.ndofTf
  #--------------------------------
  # Extract external force vectors.
  #--------------------------------
  GEXT = F[0]
  HEXT = F[1]
  JEXT = F[2]
  KEXT = F[3]
  #-------------------------------------
  # Initialize time step storage arrays.
  #-------------------------------------
  D_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  V_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  A_solve   = np.empty((Parameters.TOutput+3, Parameters.nNode),                           dtype=np.float64)
  t_solve   = np.zeros((Parameters.TOutput+3),                                             dtype=np.float64)
  SS_solve  = np.zeros((Parameters.TOutput+3, Parameters.ne, 5,  Parameters.Gauss_Order),  dtype=np.float64)
  ISV_solve = np.zeros((Parameters.TOutput+3, Parameters.ne, 19, Parameters.Gauss_Order),  dtype=np.float64)
  if Parameters.isAdaptiveStepping:
    dt_solve  = np.zeros((int(1e7)), dtype=np.float64)
    tdt_solve = np.zeros((int(1e7)), dtype=np.float64)
  #-----------------------------------------
  # Store NaN in solution arrays for masking
  # purposes when inserting Dirichlet BCs.
  #-----------------------------------------
  D_solve[:] = np.nan
  V_solve[:] = np.nan
  A_solve[:] = np.nan
  #--------------
  # Apply the IC.
  #--------------
  Parameters.tk = 0.
  g, GEXT, HEXT, JEXT, KEXT                = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT, JEXT, KEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_pred[:] = D[:]
  V_pred[:] = V[:]

  SS_solve[0,:,2,:]  = 1. # det(F)
  ISV_solve[0,:,3,:] = Parameters.ns_0
  ISV_solve[0,:,4,:] = Parameters.rhofR_0
  ISV_solve[0,:,5,:] = Parameters.khat
  ISV_solve[0,:,7,:] = Parameters.Ts_0
  ISV_solve[0,:,8,:] = Parameters.Tf_0

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  acceptSolution = True
  save_flag      = False
  t_saved        = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #--------------------------------------------
      # Only tweak time step for non-fixed schemes.
      #--------------------------------------------
      if Parameters.isAdaptiveStepping:
        #-----------------------------------------------------------------
        # Check that the time step does not drop below some minimum value.
        #
        # Also do not override the save flag (which often shrinks the
        # time step past the user-allowed minimum).
        #-----------------------------------------------------------------
        if Parameters.dt < Parameters.adaptiveDTMin and not save_flag:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #----------------------------------------
      # Compute the Runge-Kutta stage "i" data.
      #----------------------------------------
      computeStages = True
      while computeStages:
        for i in range(Parameters.numRKStages):
          Parameters.tk = Parameters.t + Parameters.dt*(Parameters.Ci[i] - 1)
          #------------------------------
          # Update BCs at stage time t_k.
          #------------------------------
          g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.tk, Parameters, HEXT, JEXT, KEXT)
          #----------------------------------------
          # Compute intermediate increment:
          # z(t_n) + dt*(sum_{j}{i-1}a_ij k_j
          #
          # Order:
          # u, u_\rf, p_\rf, \theta^\rs, \theta^\rf
          # v, v_\rf
          #----------------------------------------
          try:
            D_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, D[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSu_s:ndofSu_e], Parameters)
            D_pred[dofF_s:dofF_e]   = get_Stage_Predictor(i, D[dofF_s:dofF_e],   Parameters.dt*k_i[:,ndofFu_s:ndofFu_e], Parameters)
            D_pred[dofP_s:dofP_e]   = get_Stage_Predictor(i, D[dofP_s:dofP_e],   Parameters.dt*k_i[:,ndofP_s:ndofP_e],   Parameters)
            D_pred[dofTs_s:dofTs_e] = get_Stage_Predictor(i, D[dofTs_s:dofTs_e], Parameters.dt*k_i[:,ndofTs_s:ndofTs_e], Parameters)
            D_pred[dofTf_s:dofTf_e] = get_Stage_Predictor(i, D[dofTf_s:dofTf_e], Parameters.dt*k_i[:,ndofTf_s:ndofTf_e], Parameters)
            V_pred[dofS_s:dofS_e]   = get_Stage_Predictor(i, V[dofS_s:dofS_e],   Parameters.dt*k_i[:,ndofSv_s:ndofSv_e], Parameters)
            V_pred[dofF_s:dofF_e]   = get_Stage_Predictor(i, V[dofF_s:dofF_e],   Parameters.dt*k_i[:,ndofFv_s:ndofFv_e], Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered under/overflow in integration of stage predictors.")
            raise FloatingPointError
          #---------------------------------------
          # Balance of momentum of the pore fluid.
          #---------------------------------------
          dR_f, R_f   = moduleVE.assemble_I(LM, g, D_pred, V_pred, A, Parameters)
          A_f         = moduleVE.solve_I(dR_f, R_f, Parameters)
          #---------------------------------
          # Update A with \dot{z}_vf (f_af).
          #---------------------------------
          A[dofF_s:dofF_e] = A_f[:]
          #-------------------------------------------
          # Balance of momentum of the solid skeleton.
          #-------------------------------------------
          dR_s, R_s   = moduleVE.assemble_G(LM, g, GEXT, D_pred, V_pred, A, Parameters)
          A_s         = moduleVE.solve_G(dR_s, R_s, Parameters)
          #-------------------------------
          # Update A with \dot{z}_v (f_a).
          #-------------------------------
          A[dofS_s:dofS_e] = A_s[:]
          #-----------------------------------
          # Balance of energy of the solid.
          # (no V_pred update needed for K, H)
          #-----------------------------------
          if not Parameters.isothermalAssumption:
            dR_ts, R_ts = moduleVE.assemble_J(LM, g, JEXT, D_pred, V_pred, A, Parameters)
            Tsdot       = moduleVE.solve_J(dR_ts, R_ts, Parameters)
          #---------------------------
          # Solve p_fDot before tfDot.
          #---------------------------
          if not Parameters.staggered:
            #----------------------------
            # Balance of mass of mixture.
            #----------------------------
            dR_p, R_p   = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
            Pdot        = moduleVE.solve_H(dR_p, R_p, Parameters)
            #--------------------------------------------
            # Update V_pred with \dot{z}_pf (f_\dot{pf}).
            #--------------------------------------------
            V_pred[dofP_s:dofP_e] = Pdot[:] 
            #--------------------------------
            # Balance of energy of the fluid.
            #--------------------------------
            if not Parameters.isothermalAssumption:
              dR_tf, R_tf = moduleVE.assemble_K(LM, g, KEXT, D_pred, V_pred, A, Parameters)
              Tfdot       = moduleVE.solve_K(dR_tf, R_tf, Parameters)
          #---------------------------
          # Solve tfDot before p_fDot.
          #---------------------------
          else:
            #--------------------------------
            # Balance of energy of the fluid.
            #--------------------------------
            if not Parameters.isothermalAssumption:
              dR_tf, R_tf = moduleVE.assemble_K(LM, g, KEXT, D_pred, V_pred, A, Parameters)
              Tfdot       = moduleVE.solve_K(dR_tf, R_tf, Parameters)
            #--------------------------------------------
            # Update V_pred with \dot{z}_tf (f_\dot{tf}).
            #--------------------------------------------
              V_pred[dofTf_s:dofTf_e] = Tfdot[:] 
            #----------------------------
            # Balance of mass of mixture.
            #----------------------------
            dR_p, R_p   = moduleVE.assemble_H(LM, g, HEXT, D_pred, V_pred, A, Parameters)
            Pdot        = moduleVE.solve_H(dR_p, R_p, Parameters)
          #------------------------
          # Update stage increment.
          #------------------------
          k_i[i, ndofSu_s:ndofSu_e] = V_pred[dofS_s:dofS_e]
          k_i[i, ndofSv_s:ndofSv_e] = A_s[:]
          k_i[i, ndofFu_s:ndofFu_e] = V_pred[dofF_s:dofF_e]
          k_i[i, ndofFv_s:ndofFv_e] = A_f[:]
          k_i[i, ndofP_s:ndofP_e]   = Pdot[:]
          k_i[i, ndofTs_s:ndofTs_e] = Tsdot[:]
          k_i[i, ndofTf_s:ndofTf_e] = Tfdot[:]
          k_i[np.abs(k_i) < 1e-20]  = 0.
          #----------------------------
          # Loop over stages continues.  
          #----------------------------
        #-------------------
        # Compute solutions.
        #-------------------
        z, error = get_Error(k_i, Parameters, save_flag)

        if Parameters.isAdaptiveStepping:
          if error < 1:
            try:
              #---------------------------------
              # Integrate Runge-Kutta variables.
              #---------------------------------
              D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
              D[dofF_s:dofF_e]   += z[ndofFu_s:ndofFu_e]
              D[dofP_s:dofP_e]   += z[ndofP_s:ndofP_e]
              D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
              D[dofTf_s:dofTf_e] += z[ndofTf_s:ndofTf_e]
              V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
              V[dofF_s:dofF_e]   += z[ndofFv_s:ndofFv_e]
              #--------------------------------------------------
              # Update boundary condition at time t_n to t_{n+1}.
              #--------------------------------------------------
              g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT, JEXT, KEXT)
              g[1,:]                    = g[0,:]
              #------------------------------------------------
              # Recompute accelerations for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_f, R_f        = moduleVE.assemble_I(LM, g, D, V, A, Parameters)
              A_f              = moduleVE.solve_I(dR_f, R_f, Parameters)
              A[dofF_s:dofF_e] = A_f[:]
              dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
              A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
              A[dofS_s:dofS_e] = A_s[:]
              #---------------------------------------------------------
              # Recompute solid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
                Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
                V[dofTs_s:dofTs_e] = Tsdot[:]
              #---------------------------
              # Solve p_fDot before tfDot.
              #---------------------------
              if not Parameters.staggered:
                #------------------------------------------------
                # Recompute pressure rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #------------------------------------------------
                dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
                Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
                V[dofP_s:dofP_e] = Pdot[:]
                #---------------------------------------------------------
                # Recompute fluid temperature rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #---------------------------------------------------------
                if not Parameters.isothermalAssumption:
                  dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                  Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                  V[dofTf_s:dofTf_e] = Tfdot[:]
              #---------------------------
              # Solve tfDot before p_fDot.
              #---------------------------
              else:
                #---------------------------------------------------------
                # Recompute fluid temperature rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #---------------------------------------------------------
                if not Parameters.isothermalAssumption:
                  dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                  Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                  V[dofTf_s:dofTf_e] = Tfdot[:]
                #------------------------------------------------
                # Recompute pressure rate for next time increment
                # using the accepted solutions at t_{n+1}.
                #------------------------------------------------
                dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
                Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
                V[dofP_s:dofP_e] = Pdot[:]
            except FloatingPointError:
              print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
              print("Encountered over/underflow error updating solution.")
              raise FloatingPointError
            acceptSolution               = True
            computeStages                = False
          else:
            acceptSolution = False
        else:
          try:
            #---------------------------------
            # Integrate Runge-Kutta variables.
            #---------------------------------
            D[dofS_s:dofS_e]   += z[ndofSu_s:ndofSu_e]
            D[dofF_s:dofF_e]   += z[ndofFu_s:ndofFu_e]
            D[dofP_s:dofP_e]   += z[ndofP_s:ndofP_e]
            D[dofTs_s:dofTs_e] += z[ndofTs_s:ndofTs_e]
            D[dofTf_s:dofTf_e] += z[ndofTf_s:ndofTf_e]
            V[dofS_s:dofS_e]   += z[ndofSv_s:ndofSv_e]
            V[dofF_s:dofF_e]   += z[ndofFv_s:ndofFv_e]
            #--------------------------------------------------
            # Update boundary condition at time t_n to t_{n+1}.
            #--------------------------------------------------
            g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT, JEXT, KEXT)
            g[1,:]                    = g[0,:]
            #------------------------------------------------
            # Recompute accelerations for next time increment
            # using the accepted solutions at t_{n+1}.
            #------------------------------------------------
            dR_f, R_f        = moduleVE.assemble_I(LM, g, D, V, A, Parameters)
            A_f              = moduleVE.solve_I(dR_f, R_f, Parameters)
            A[dofF_s:dofF_e] = A_f[:]
            dR_s, R_s        = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
            A_s              = moduleVE.solve_G(dR_s, R_s, Parameters)
            A[dofS_s:dofS_e] = A_s[:]
            #---------------------------------------------------------
            # Recompute solid temperature rate for next time increment
            # using the accepted solutions at t_{n+1}.
            #---------------------------------------------------------
            if not Parameters.isothermalAssumption:
              dR_ts, R_ts        = moduleVE.assemble_J(LM, g, JEXT, D, V, A, Parameters)
              Tsdot              = moduleVE.solve_J(dR_ts, R_ts, Parameters)
              V[dofTs_s:dofTs_e] = Tsdot[:]
            #---------------------------
            # Solve p_fDot before tfDot.
            #---------------------------
            if not Parameters.staggered:
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
              #---------------------------------------------------------
              # Recompute fluid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                V[dofTf_s:dofTf_e] = Tfdot[:]
            #---------------------------
            # Solve tfDot before p_fDot.
            #---------------------------
            else:
              #---------------------------------------------------------
              # Recompute fluid temperature rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #---------------------------------------------------------
              if not Parameters.isothermalAssumption:
                dR_tf, R_tf        = moduleVE.assemble_K(LM, g, KEXT, D, V, A, Parameters)
                Tfdot              = moduleVE.solve_K(dR_tf, R_tf, Parameters)
                V[dofTf_s:dofTf_e] = Tfdot[:]
              #------------------------------------------------
              # Recompute pressure rate for next time increment
              # using the accepted solutions at t_{n+1}.
              #------------------------------------------------
              dR_p, R_p        = moduleVE.assemble_H(LM, g, HEXT, D, V, A, Parameters)
              Pdot             = moduleVE.solve_H(dR_p, R_p, Parameters)
              V[dofP_s:dofP_e] = Pdot[:]
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("Encountered over/underflow error updating solution.")
            raise FloatingPointError
          acceptSolution               = True
          computeStages                = False
      #------------------------------------------------------
      # Solution error is less than a user-defined tolerance.
      # Prepare to save data if necessary.
      #------------------------------------------------------
      if acceptSolution:
        t_previous     = Parameters.t
        Parameters.dt0 = Parameters.dt
        #-----------------------------------
        # Flag to determine whether to save
        # for fixed time stepping.
        #-----------------------------------
        if not Parameters.isAdaptiveStepping:
          if n % Parameters.n_save == 0:
            save_flag = True
        #----------------------------------
        # Save data at this time increment.
        #----------------------------------
        if save_flag:
          print("Solution stored at t = {:.3e}s".format(Parameters.t))
          #----------------------------------
          # Compute stress/strain at t_{n+1}.
          #----------------------------------
          try:
            SS, ISV = moduleVE.get_SSISV(LM, g, D, V, A, Parameters)
          except FloatingPointError:
            print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
            print("FloatingPointError encountered trying to save data.")
            raise FloatingPointError
          #---------------------
          # Store solution data.
          #---------------------
          D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                             D_solve[m+1,:],\
                                                                             V_solve[m+1,:],\
                                                                             A_solve[m+1,:], Parameters)
          t_solve[m+1]         = Parameters.t
          SS_solve[m+1,:,:,:]  = SS[:]
          ISV_solve[m+1,:,:,:] = ISV[:]

          m        += 1
          save_flag = False
          t_saved   = Parameters.t

        if Parameters.isAdaptiveStepping:
          #------------------------------
          # Save adaptive time step data.
          #------------------------------
          dt_solve[n]  = Parameters.dt0
          tdt_solve[n] = Parameters.t
          #-----------------------------------
          # Flag to determine whether to save.
          #-----------------------------------
          if (t_previous + Parameters.dt >= Parameters.t_save):
            if Parameters.t_save - t_previous > 1e-10:
              Parameters.dt = Parameters.t_save - t_previous
            Parameters.t_save += Parameters.dt_save
            save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nRK INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())
  #--------------------------------------------------
  # At the end of simulation time, pass solution data
  # back to solver_uufpftstf.
  #--------------------------------------------------
  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to compute a Runge-Kutta stage predictor.
# ----------
# Arguments:
# ----------
# stageIndex:  (int)                      the stage number
# a_Z:         (float, size: varies)      the independent field variable to be updated at the 
#                                         first stage (i.e. previous step value)
# k_i:         (float, size: 5, varies)   the stage solution of the independent field variable
# Parameters:  (object)                   problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# pred_Z:      (float, size: varies)      the predictor for the independent field variable
#--------------------------------------------------------------------------------------------------
def get_Stage_Predictor(stageIndex, Z, k_i, Parameters):

  try:
    if stageIndex == 0:
      pred_Z = Z
    elif stageIndex == 1:
      pred_Z = Z + Parameters.Aij[1,0]*k_i[0,:]
    elif stageIndex == 2:
      pred_Z = Z + Parameters.Aij[2,0]*k_i[0,:] + Parameters.Aij[2,1]*k_i[1,:]
    elif stageIndex == 3:
      pred_Z = Z + Parameters.Aij[3,0]*k_i[0,:] + Parameters.Aij[3,1]*k_i[1,:] + Parameters.Aij[3,2]*k_i[2,:]
    elif stageIndex == 4:
      pred_Z = Z + Parameters.Aij[4,0]*k_i[0,:] + Parameters.Aij[4,1]*k_i[1,:] + Parameters.Aij[4,2]*k_i[2,:] + Parameters.Aij[4,3]*k_i[3,:]
    elif stageIndex == 5:
      pred_Z = Z + Parameters.Aij[5,0]*k_i[0,:] + Parameters.Aij[5,1]*k_i[1,:] + Parameters.Aij[5,2]*k_i[2,:] + Parameters.Aij[5,3]*k_i[3,:] + Parameters.Aij[5,4]*k_i[4,:]
    else:
      sys.exit("------\nERROR:\n------\nStage index not recognized, check source code.")
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Encountered under/overflow in calculation of stage predictors.")
    raise FloatingPointError

  return pred_Z
#--------------------------------------------------------------------------------------------------
# Helper function to compute a RK stage solution.
# ----------
# Arguments:
# ----------
# orderIndex:  (int)                     the order number
# k_i:         (float, size: 5, varies)  the stage solution of the independent field variable
# Parameters:  (object)                  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# sol_Z:       (float, size: varies)     the step solution for the independent field variable
#--------------------------------------------------------------------------------------------------
def get_Solution(orderIndex, k_i, Parameters):

  try:
    if orderIndex == 1:
      sol_Z = Parameters.dt*(Parameters.Bij[0,0]*k_i[0,:])
    elif orderIndex == 2:
      if Parameters.integrationScheme == 'RKBS':
        sol_Z = Parameters.dt*(Parameters.Bij[1,0]*k_i[0,:] + Parameters.Bij[1,1]*k_i[1,:] + Parameters.Bij[1,2]*k_i[2,:] + Parameters.Bij[1,3]*k_i[3,:])
      else:
        sol_Z = Parameters.dt*(Parameters.Bij[1,0]*k_i[0,:] + Parameters.Bij[1,1]*k_i[1,:])
    elif orderIndex == 3:
      sol_Z = Parameters.dt*(Parameters.Bij[2,0]*k_i[0,:] + Parameters.Bij[2,2]*k_i[2,:] + Parameters.Bij[2,3]*k_i[3,:])
    elif orderIndex == 4:
      sol_Z = Parameters.dt*(Parameters.Bij[3,0]*k_i[0,:] + Parameters.Bij[3,2]*k_i[2,:] + Parameters.Bij[3,3]*k_i[3,:] + Parameters.Bij[3,4]*k_i[4,:] + Parameters.Bij[3,5]*k_i[5,:])
    elif orderIndex == 5:
      sol_Z = Parameters.dt*(Parameters.Bij[4,0]*k_i[0,:] + Parameters.Bij[4,2]*k_i[2,:] + Parameters.Bij[4,3]*k_i[3,:] + Parameters.Bij[4,5]*k_i[5,:])
    else:
      sys.exit("------\nERROR:\n------\nOrder of accuracy (m) not recognized, check source code.")
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Encountered under/overflow in calculation of solution.")
    raise FloatingPointError

  return sol_Z
#--------------------------------------------------------------------------------------------------
# Helper function to compute a partial RK stage solution (for varying-order schemes).
# 
# Note: the varying-order schemes require a top-level method which was deprecated in an earlier
# version of SPONGE-1D in favor of the more robust (and often cheaper) fixed-order schemes. Refer
# to Cash & Karp (1990) for details on implementation for general problems.
# ----------
# Arguments:
# ----------
# orderIndex:  (int)                     the order number
# k_i:         (float, size: 5, varies)  the stage solution of the independent field variable
# Parameters:  (object)                  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# sol_Z:        (float, size: varies)    the partial solution for the independent field variable
#--------------------------------------------------------------------------------------------------
def get_Solution_Partial(orderIndex, k_i, Parameters):

  if orderIndex == 2:
    sol_Z = 0.1*Parameters.dt*(k_i[0,:] + k_i[1,:]) # 0.1 is a safety factor
  elif orderIndex == 3:
    sol_Z = 0.1*Parameters.dt*(k_i[0,:] + 2*k_i[1,:] + k_i[2,:])
  else:
    sys.exit("------\nERROR:\n------\nOrder of accuracy not recognized for VRKF partial solutions, check source code.")

  return sol_Z
#--------------------------------------------------------------------------------------------------
# Helper function to compute the error of a Runge-Kutta step.
# ----------
# Arguments:
# ----------
# k_i:     (float, size: 5, varies)  the stage solution of the independent field variable
# Parameters:  (object)              problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# z:       (float, size: varies)     the highest order solution for the independent field variable
# error:   (float)                   the local error associated with this solution   
#--------------------------------------------------------------------------------------------------
def get_Error(k_i, Parameters, save_flag):
  
  if Parameters.numRKStages == 6:
    z      = get_Solution(5, k_i, Parameters)
    z_star = get_Solution(4, k_i, Parameters)
  elif Parameters.numRKStages == 4:
    z      = get_Solution(3, k_i, Parameters)
    z_star = get_Solution(2, k_i, Parameters)
  else:
    sys.exit("------\nERROR:\n------\nCannot compute Runge-Kutta solutions for order accuracies != 5 or 3, check source code.")

  if Parameters.norm_ord == np.inf:
    error = np.linalg.norm((z - z_star), ord=Parameters.norm_ord)
  elif Parameters.norm_ord == 2:
    error = np.sqrt(np.sum((z - z_star)**2))/z.shape[0]
  #---------------------------------------------------
  # Compute the truncation error / the absolute error. 
  #---------------------------------------------------
  error = error**(1/(Parameters.numRKStages - 1)) / Parameters.tola**(1/(Parameters.numRKStages - 1))
  
  if Parameters.isAdaptiveStepping:
    try:
      #----------------------------------------------
      # Calculate new time step based on global error.
      #----------------------------------------------
      newdt = Parameters.dt*Parameters.SF/error
      if not save_flag:
        Parameters.dt *= Parameters.SF/error
      #--------------------------------------------------
      # This block ensures that if the adjusted time
      # step to save data at a specific time point is 
      # larger than the suggested time step given by the
      # adaptive time step algorithm (i.e., the increment
      # required for the save_flag is larger than the 
      # increment produced by the error estimation), the 
      # code will defer to the latter for stability.
      #--------------------------------------------------
      else:
        if newdt < Parameters.dt:
          Parameters.dt *= Parameters.SF/error
    except FloatingPointError:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Global error approached zero.")
      raise FloatingPointError
    if Parameters.dt <= Parameters.adaptiveDTMin and not save_flag:
      print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
      print("Time step below user-defined tolerance.")
      raise FloatingPointError
    elif Parameters.dt > Parameters.adaptiveDTMax:
      Parameters.dt = Parameters.adaptiveDTMax
  
  return z, error

