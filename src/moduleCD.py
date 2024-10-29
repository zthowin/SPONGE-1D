#--------------------------------------------------------------------------------------
# Module housing helper functions for central-difference methods.
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
#
# ----------
# Arguemnts:
# ----------
# LM:           (int,   size: # 3 x # elements) location matrix
# GEXT:         (float, size: # skeleton DOFs)  external traction vector
# D:            (float, size: # skeleton DOFs)  global IC for displacement
# V:            (float, size: # skeleton DOFs)  global IC for velocity
# A:            (float, size: # skeleton DOFs)  global IC for acceleration
# Parameters:   (object)                        problem parameters initiated in runMain.py
#                                                 
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times x # skeleton DOFs)             global solution for displacement
# V_solve:   (float, size: # save times x # skeleton DOFs)             global solution for velocity
# A_solve:   (float, size: # save times x # skeleton DOFs)             global solution for acceleration
# t_solve:   (float, size: # save times)                               simulation times
# SS_solve:  (float, size: # save times x # elements x # stresses x 3) stress solutions
# ISV_solve: (float, size: # save times x # elements x # stresses x 3) ISV solutions
#
#--------------------------------------------------------------------------------------------------
def integrate_u(LM, GEXT, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
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
  g, GEXT                                  = moduleFE.updateBC(g, GEXT, 0.0, Parameters)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_Last = np.copy(D)
  V_Last = np.copy(V)
  A_Last = np.copy(A)

  SS_solve[0,:,2,:]  = 1.
  ISV_solve[0,:,3,:] = Parameters.ns_0

  if Parameters.isAdaptiveStepping and Parameters.adaptiveSave:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False
  t_saved   = 0

  n = 0
  m = n

  Parameters.dtnew = Parameters.dt

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0.0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #----------------------------------------------------------------
      # Check that the time step does not drop below some minimum value
      # nor go above some maximum value.
      #----------------------------------------------------------------
      if Parameters.dt < Parameters.adaptiveDTMin and Parameters.isAdaptiveStepping and not save_flag:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      elif Parameters.dt > Parameters.adaptiveDTMax:
        Parameters.dt = Parameters.adaptiveDTMax
      #----------------------------------
      # Update Dirichlet BCs at time t_n.
      #----------------------------------
      g[1,:] = g[0,:]
      #-----------------------------------
      # Update simulation time to t_{n+1}.
      #-----------------------------------
      Parameters.t += Parameters.dt
      n            += 1
      Parameters.tk = Parameters.t
      #----------------------------
      # Update BCs at time t_{n+1}.
      #----------------------------
      g, GEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters)
      #-------------------------------
      # Compute solid skeleton update.
      #-------------------------------
      dR_s, R_s             = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
      A                     = moduleVE.solve_G(dR_s, R_s, Parameters)
      A[np.abs(A) < 1e-20]  = 0. # Otherwise floating point errors
      V                    += update_dotx(A, A_Last, Parameters)
      D                    += update_x(A_Last, V_Last, Parameters)
      #----------------------------------------------
      # Update solution variables for next time step.
      #----------------------------------------------
      A_Last[:]  = A[:]
      V_Last[:]  = V[:]
      D_Last[:]  = D[:]

      t_previous     = Parameters.t
      Parameters.dt0 = Parameters.dt
      
      if Parameters.isAdaptiveStepping:
        Parameters.dt = Parameters.dtnew

      if not Parameters.isAdaptiveStepping:
        if n % Parameters.n_save == 0:
          save_flag = True

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
        #--------------------
        # Save solution data.
        #--------------------
        D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                           D_solve[m+1,:],\
                                                                           V_solve[m+1,:],\
                                                                           A_solve[m+1,:], Parameters)
        t_solve[m+1]         = Parameters.t
        SS_solve[m+1,:,:,:]  = SS[:]
        ISV_solve[m+1,:,:,:] = ISV[:]
        m                   += 1
        save_flag            = False
        t_saved              = Parameters.t
      #------------------------------
      # Save adaptive time step data.
      #------------------------------
      if Parameters.isAdaptiveStepping:
        dt_solve[n]  = Parameters.dt0
        tdt_solve[n] = Parameters.t
        if (t_previous + Parameters.dt >= Parameters.t_save):
          if Parameters.t_save - t_previous > 1e-10:
            Parameters.dt = Parameters.t_save - t_previous
          Parameters.t_save += Parameters.dt_save
          save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nCD INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())

  if Parameters.isAdaptiveStepping and Parameters.adaptiveSave:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the variational equations for balance of momentum of the mixture and
# balance of mass of the mixture.
#
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs x # elements) location matrix
# F:            (float, size: 2 x # DOFs)                  external force vectors
# D:            (float, size: # DOFs)                      global IC for displacement & pressure
# V:            (float, size: # DOFs)                      global IC for velocity & \dot{p_f}
# A:            (float, size: # DOFs)                      global IC for acceleration & \ddot{p_f}
# Parameters:   (object)                                   problem parameters initiated in runMain.py
#                                                 
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times x # DOFs)       global solution for displacement & pressure
# V_solve:   (float, size: # save times x # DOFs)       global solution for velocity & \dot{p_f}
# A_solve:   (float, size: # save times x # DOFs)       global solution for acceleration & \ddot{p_f}
# t_solve:   (float, size: # save times)                simulation times
# SS_solve:  (float, size: # save times x # elements x # stresses x 3) stress solutions
# ISV_solve: (float, size: # save times x # elements x # stresses x 3) ISV solutions
#
#--------------------------------------------------------------------------------------------------
def integrate_upf(LM, F, D, V, A, Parameters):
  #-----------------------------------------------------------------------------------
  # Initialize element-wise Dirichlet BC storage.
  #
  # For a 1-D FE code, only the top and bottom elements may possess Dirichlet BCs.
  #
  # Size: ([t_{n+1}, t_n], [x, \dot{x}, \ddot{x}], # element DOF, [e = 0, e = ne - 1])
  #-----------------------------------------------------------------------------------
  g = np.zeros((2, 3, Parameters.ndofe, 2), dtype=np.float64)
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
  g, GEXT, HEXT                            = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  D_Last = np.copy(D)
  V_Last = np.copy(V)
  A_Last = np.copy(A)

  SS_solve[0,:,2,:]  = 1.
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

  save_flag = False
  t_saved   = 0

  n = 0
  m = n

  try:
    while Parameters.t < Parameters.TStop:
      #-------------------------------------
      # Do not integrate past the stop time.
      #-------------------------------------
      if Parameters.t + Parameters.dt - Parameters.TStop > 0.0:
        Parameters.dt = Parameters.TStop - Parameters.t
        save_flag     = True
      #----------------------------------------------------------------
      # Check that the time step does not drop below some minimum value
      # nor go above some maximum value.
      #----------------------------------------------------------------
      if Parameters.dt < Parameters.adaptiveDTMin and Parameters.isAdaptiveStepping and not save_flag:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      elif Parameters.dt > Parameters.adaptiveDTMax:
        Parameters.dt = Parameters.adaptiveDTMax
      #----------------------------------
      # Update Dirichlet BCs at time t_n.
      #----------------------------------
      g[1,:] = g[0,:]
      #-----------------------------------
      # Update simulation time to t_{n+1}.
      #-----------------------------------
      Parameters.t += Parameters.dt
      n            += 1
      Parameters.tk = Parameters.t
      #----------------------------
      # Update BCs at time t_{n+1}.
      #----------------------------
      g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
      #-------------------------------
      # Compute solid skeleton update.
      #-------------------------------
      dR_s, R_s              = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
      A[0:Parameters.ndofS]  = moduleVE.solve_G(dR_s, R_s, Parameters)
      A[np.abs(A) < 1e-20]   = 0.
      V[0:Parameters.ndofS] += update_dotx(A[0:Parameters.ndofS], A_Last[0:Parameters.ndofS], Parameters)
      D[0:Parameters.ndofS] += update_x(A_Last[0:Parameters.ndofS], V_Last[0:Parameters.ndofS], Parameters)
      #------------------------------------
      # Compute pore fluid pressure update.
      #------------------------------------
      dR_p, R_p                            = moduleVE.assemble_H(LM, g, HEXT, D_Last, V_Last, A_Last, Parameters, A)
      A[Parameters.ndofS:Parameters.ndof]  = moduleVE.solve_H(dR_p, R_p, Parameters)
      A[np.abs(A) < 1e-20]                 = 0.
      V[Parameters.ndofS:Parameters.ndof] += update_dotx(A[Parameters.ndofS:Parameters.ndof], A_Last[Parameters.ndofS:Parameters.ndof], Parameters)
      D[Parameters.ndofS:Parameters.ndof] += update_x(A_Last[Parameters.ndofS:Parameters.ndof], V_Last[Parameters.ndofS:Parameters.ndof], Parameters)
      #----------------------------------------------
      # Update solution variables for next time step.
      #----------------------------------------------
      A_Last[:] = A[:]
      V_Last[:] = V[:]
      D_Last[:] = D[:]

      t_previous     = Parameters.t
      Parameters.dt0 = Parameters.dt
      if Parameters.isAdaptiveStepping:
        Parameters.dt = Parameters.dtnew

      if not Parameters.isAdaptiveStepping:
        if n % Parameters.n_save == 0:
          save_flag = True

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
        #--------------------
        # Save solution data.
        #--------------------
        D_solve[m+1,:], V_solve[m+1,:], A_solve[m+1,:] = moduleFE.insertBC(g, D, V, A,\
                                                                           D_solve[m+1,:],\
                                                                           V_solve[m+1,:],\
                                                                           A_solve[m+1,:], Parameters)
        t_solve[m+1]         = Parameters.t
        SS_solve[m+1,:,:,:]  = SS[:]
        ISV_solve[m+1,:,:,:] = ISV[:]
        m                   += 1
        save_flag            = False
        t_saved              = Parameters.t
      #------------------------------
      # Save adaptive time step data.
      #------------------------------
      if Parameters.isAdaptiveStepping:
        dt_solve[n]  = Parameters.dt0
        tdt_solve[n] = Parameters.t
        if (t_previous + Parameters.dt >= Parameters.t_save):
          if Parameters.t_save - t_previous > 1e-10:
            Parameters.dt = Parameters.t_save - t_previous
          Parameters.t_save += Parameters.dt_save
          save_flag          = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------\nCD INTEGRATOR ERROR:\n--------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())

  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#----------------------------------------------------
# Helper function to integrate first time derivative.
#----------------------------------------------------
def update_dotx(A_np1, A_n, Parameters):
  try:
    return (0.5*Parameters.dt*(A_np1 + A_n))
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Underflow/overflow, could not update /\dot{x}.")
    raise FloatingPointError
#-----------------------------------------------------
# Helper function to integrate second time derivative.
#-----------------------------------------------------
def update_x(A_n, V_n, Parameters):
  try:
    return (Parameters.dt*V_n + 0.5*(Parameters.dt**2)*A_n)
  except FloatingPointError:
    print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
    print("Underflow/overflow, could not update x.")
    raise FloatingPointError

