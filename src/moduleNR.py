#--------------------------------------------------------------------------------------
# Module housing helper functions for implicit methods.
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
# Helper function to integrate a system of variational equations using second-order Newmark-beta
# method.
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
def integrate_Newmark_beta(LM, F, D, V, A, Parameters):
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
  if Parameters.Physics == 'u-t':
    JEXT = F[1]
  elif 'pf' in Parameters.Physics:
    HEXT = F[1]
    try:
      JEXT = F[2]
      KEXT = F[3]
    except IndexError:
      pass
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
  if Parameters.Physics == 'u':
    g, GEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters)
  elif Parameters.Physics == 'u-t':
    g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, JEXT)
  elif 'ts' not in Parameters.Physics:
    g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  else:
    g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT, JEXT, KEXT)

  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)
  
  SS_solve[0,:,2,:]  = 1. # det(F)
  if Parameters.Physics != 'u' and Parameters.Physics != 'u-t':
    ISV_solve[0,:,3,:] = Parameters.ns_0
    ISV_solve[0,:,4,:] = Parameters.rhofR_0
    ISV_solve[0,:,5,:] = Parameters.khat

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt

  del_A = np.zeros(Parameters.ndof) # Initial increment
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False
  t_saved   = 0

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
      if Parameters.dt < Parameters.adaptiveDTMin and Parameters.isAdaptiveStepping:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      #----------------------------------
      # Update Dirichlet BCs at time t_n.
      #----------------------------------
      g[1,:] = g[0,:]
      #--------------------------------------------
      # Update simulation time from t_n to t_{n+1}.
      #--------------------------------------------
      Parameters.t += Parameters.dt
      n            += 1
      Parameters.tk = Parameters.t
      #----------------------------
      # Update BCs at time t_{n+1}.
      #----------------------------
      if Parameters.Physics == 'u':
        g, GEXT       = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters)
      elif Parameters.Physics == 'u-t':
        g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, JEXT)
      elif 'ts' not in Parameters.Physics:
        g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
      else:
        g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT, JEXT, KEXT)
      #-----------------------
      # Update the predictors.
      #-----------------------
      try:
        Vtilde  = V + (1 - Parameters.gamma)*Parameters.dt*A
        Dtilde  = D + Parameters.dt*V + (1 - 2*Parameters.beta)*(Parameters.dt**2/2)*A
      except FloatingPointError:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Encountered under/overflow in update of predictors.")
        raise FloatingPointError
      Rtol    = 1
      normR   = 1
      k       = 0
      #---------------------------------
      # Begin Newton-Raphson iterations.
      #---------------------------------
      while (Rtol > Parameters.tolr) and (normR > Parameters.tola):
        #----------------------------------
        # Increment iterator and solutions.
        #----------------------------------
        k += 1
        try:
          A += del_A
          V  = Vtilde + (Parameters.gamma*Parameters.dt)*A
          D  = Dtilde + (Parameters.beta*(Parameters.dt**2))*A
        except FloatingPointError:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Encountered under/overflow in integration of predictors.")
          raise FloatingPointError
        #----------------------------------
        # Form global tangent and residual.
        #----------------------------------
        if Parameters.Physics == 'u':
          dR, R = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
        elif Parameters.Physics == 'u-t':
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, JEXT], D, V, A, Parameters)
        elif 'ts' not in Parameters.Physics:
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, HEXT], D, V, A, Parameters)
        else:
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, HEXT, JEXT, KEXT], D, V, A, Parameters)
        #------------------------------------------------------------
        # Alter the tangent and residual for the Lagrange multiplier.
        #------------------------------------------------------------
        if Parameters.Physics == 'u-uf-pf' and Parameters.LagrangeApply:
          dR, R = moduleVE.assemble_Lagrange(LM, g, D, V, A, dR, R, Parameters) 
        #------------------------------------
        # Compute global acceleration update.
        #------------------------------------
        del_A = moduleVE.solve_System(dR, R, Parameters)
        #----------------------------
        # Compute tolerance and norm.
        #----------------------------
        if (k == 1):
          R0 = R

        Rtol  = np.linalg.norm(R, ord=Parameters.norm_ord)/np.linalg.norm(R0, ord=Parameters.norm_ord)
        normR = np.linalg.norm(R, ord=Parameters.norm_ord)
        #------------------------------
        # Apply adaptive time stepping.
        #------------------------------
        if Parameters.isAdaptiveStepping and (Parameters.t > Parameters.adaptiveStart and Parameters.t <= Parameters.adaptiveStop):
          if k > Parameters.adaptiveKMax:
            Parameters.dt = Parameters.dt*Parameters.adaptiveDecrease
        #------------------
        # Check iterations.
        #------------------
        if k == Parameters.kmax:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Relative tolerance = {:.3e}".format(Rtol))
          print("Reached max number of iterations for simulations at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt)) 
          raise RuntimeError
      #-----------------------------------------
      # Solution accepted, update previous time.
      #-----------------------------------------
      t_previous = Parameters.t
      #--------------------------------
      # Apply adaptive time stepping.
      # Refer to Laadhari et al. (2017)
      #--------------------------------
      if Parameters.isAdaptiveStepping:
        if (Parameters.t > Parameters.adaptiveStart and Parameters.t <= Parameters.adaptiveStop):
          if k < Parameters.adaptiveKMax:
            Parameters.dt = Parameters.dt*Parameters.adaptiveIncrease
          if Parameters.dt > Parameters.adaptiveDTMax:
            Parameters.dt = Parameters.adaptiveDTMax
        elif Parameters.t > Parameters.adaptiveStop:
          Parameters.dt = Parameters.adaptiveDTMax

        dt_solve[n]  = Parameters.dt0
        tdt_solve[n] = Parameters.t

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

        m        += 1
        save_flag = False
        t_saved   = Parameters.t

      if Parameters.isAdaptiveStepping:
        if (t_previous + Parameters.dt >= Parameters.t_save):
          if Parameters.t_save - t_previous > 1e-10:
            Parameters.dt = Parameters.t_save - t_previous
          Parameters.t_save += Parameters.dt_save
          save_flag          = True

      Parameters.dt0 = Parameters.dt
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------------\nIMPLICIT INTEGRATOR ERROR:\n--------------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())

  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate a system of variational equations using first-order Trapezoidal
# method.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements) location matrix
# F:            (float, size: varies)                     Neumann BCs
# D:            (float, size: # DOFs)                     global IC for displacement(s)
# V:            (float, size: # DOFs)                     global IC for velocity(s)
# Parameters:   (object)                                  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement(s)
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity(s)
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration = 0
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, 4, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, 7, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_Trapezoidal(LM, F, D, V, Parameters):
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
  if Parameters.Physics == 'u-t':
    JEXT = F[1]
  elif 'pf' in Parameters.Physics:
    HEXT = F[1]
    try:
      JEXT = F[2]
      KEXT = F[3]
    except IndexError:
      pass
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
  A             = np.zeros(Parameters.ndof)
  Parameters.tk = 0.
  if Parameters.Physics == 'u':
    g, GEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters)
  elif Parameters.Physics == 'u-t':
    g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, JEXT)
  elif 'ts' not in Parameters.Physics:
    g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  else:
    g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT, JEXT, KEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)

  SS_solve[0,:,2,:]  = 1. # det(F)
  if Parameters.Physics != 'u' and Parameters.Physics != 'u-t':
    ISV_solve[0,:,3,:] = Parameters.ns_0
    ISV_solve[0,:,4,:] = Parameters.rhofR_0
    ISV_solve[0,:,5,:] = Parameters.khat

  if Parameters.isAdaptiveStepping:
    dt_solve[0]      = Parameters.dt0
    Parameters.dtnew = Parameters.dt

  del_V = np.zeros(Parameters.ndof) # Initial increment
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False
  t_saved   = 0

  n         = 0
  m         = n

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
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      #----------------------------------
      # Update Dirichlet BCs at time t_n.
      #----------------------------------
      g[1,:] = g[0,:]
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      Parameters.tk = Parameters.t
      #--------------------------------
      # Update the BCs at time t_{n+1}.
      #--------------------------------
      if Parameters.Physics == 'u':
        g, GEXT       = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters)
      elif Parameters.Physics == 'u-t':
        g, GEXT, JEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, JEXT)
      elif 'ts' not in Parameters.Physics:
        g, GEXT, HEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT)
      else:
        g, GEXT, HEXT, JEXT, KEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters, HEXT, JEXT, KEXT)
      #----------------------
      # Update the predictor.
      #----------------------
      Dtilde = D + (1 - Parameters.gamma)*Parameters.dt*V
      Rtol   = 1
      normR  = 1
      k      = 0
      #---------------------------------
      # Begin Newton-Raphson iterations.
      #---------------------------------
      while (Rtol > Parameters.tolr) and (normR > Parameters.tola):
        #----------------------------------
        # Increment iterator and solutions.
        #----------------------------------
        k += 1
        V += del_V
        D  = Dtilde + (Parameters.gamma*Parameters.dt)*V
        #----------------------------------
        # Form global tangent and residual.
        #----------------------------------
        if Parameters.Physics == 'u':
          dR, R = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
        elif Parameters.Physics == 'u-t':
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, JEXT], D, V, A, Parameters)
        elif 'ts' not in Parameters.Physics:
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, HEXT], D, V, A, Parameters)
        else:
          dR, R = moduleVE.assemble_System(LM, g, [GEXT, HEXT, JEXT, KEXT], D, V, A, Parameters)
        #------------------------------------
        # Compute global velocity update.
        #------------------------------------
        del_V = moduleVE.solve_System(dR, R, Parameters)
        #----------------------------
        # Compute tolerance and norm.
        #----------------------------
        if (k == 1):
          R0 = R

        Rtol  = np.linalg.norm(R, ord=Parameters.norm_ord)/np.linalg.norm(R0, ord=Parameters.norm_ord)
        normR = np.linalg.norm(R, ord=Parameters.norm_ord)
        #------------------------------
        # Apply adaptive time stepping.
        #------------------------------
        if Parameters.isAdaptiveStepping and (Parameters.t > Parameters.adaptiveStart and Parameters.t <= Parameters.adaptiveStop):
          if k > Parameters.adaptiveKMax:
            Parameters.dt = Parameters.dt*Parameters.adaptiveDecrease
        #------------------
        # Check iterations.
        #------------------
        if k == Parameters.kmax:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Relative tolerance = {:.3e}".format(Rtol))
          print("Reached max number of iterations for simulations at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError

      Parameters.dt0 = Parameters.dt
      t_previous = Parameters.t
      #--------------------------------
      # Apply adaptive time stepping.
      # Refer to Laadhari et al. (2017)
      #--------------------------------
      if Parameters.isAdaptiveStepping:
        if (Parameters.t > Parameters.adaptiveStart and Parameters.t <= Parameters.adaptiveStop):
          if k < Parameters.adaptiveKMax:
            Parameters.dt = Parameters.dt*Parameters.adaptiveIncrease
          if Parameters.dt > Parameters.adaptiveDTMax:
            Parameters.dt = Parameters.adaptiveDTMax
        elif Parameters.t > Parameters.adaptiveStop:
          Parameters.dt = Parameters.adaptiveDTMax

        dt_solve[n+1]  = Parameters.dt
        tdt_solve[n+1] = Parameters.t

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

        m        += 1
        save_flag = False
        t_saved   = Parameters.t

      if Parameters.isAdaptiveStepping:
        if (t_previous + Parameters.dt >= Parameters.t_save):
          if Parameters.t_save - t_previous > 1e-10:
            Parameters.dt = Parameters.t_save - t_previous
          Parameters.t_save += Parameters.dt_save
          save_flag      = True
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------------\nIMPLICIT INTEGRATOR ERROR:\n--------------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())

  if Parameters.isAdaptiveStepping:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve, dt_solve, tdt_solve
  else:
    return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve
#--------------------------------------------------------------------------------------------------
# Helper function to integrate the non-linear balance of momentum equation.
# ----------
# Arguments:
# ----------
# LM:           (int,   size: # element DOFs, # elements) location matrix
# GEXT:         (float, size: varies)                     external traction vector
# D:            (float, size: # DOFs)                     global IC for displacement
# Parameters:   (object)                                  problem parameters initiated in runMain.py
# --------
# Returns:
# --------
# D_solve:   (float, size: # save times, # DOFs)       global solution for displacement
# V_solve:   (float, size: # save times, # DOFs)       global solution for velocity     = 0
# A_solve:   (float, size: # save times, # DOFs)       global solution for acceleration = 0
# t_solve:   (float, size: # save times)               simulation times
# SS_solve:  (float, size: # save times, # elements, # stresses, # Gauss points) stress solutions
# ISV_solve: (float, size: # save times, # elements, # stresses, # Gauss points) ISV solutions
#--------------------------------------------------------------------------------------------------
def integrate_QS(LM, GEXT, D, Parameters):
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
  V = np.zeros(Parameters.ndof)
  A = np.zeros(Parameters.ndof)
  g, GEXT, HEXT                            = moduleFE.updateBC(g, GEXT, 0.0, Parameters, HEXT)
  D_solve[0,:], V_solve[0,:], A_solve[0,:] = moduleFE.insertBC(g, D, V, A,
                                                               D_solve[0,:],
                                                               V_solve[0,:],
                                                               A_solve[0,:], Parameters)

  SS_solve[0,:,2,:] = 1. # det(F)
  del_D             = np.zeros(Parameters.ndof) # Initial increment
  #------------------
  # Start simulation.
  #------------------
  print("Solving...")

  save_flag = False
  t_saved   = 0

  n         = 0
  m         = n

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
      if Parameters.dt < Parameters.adaptiveDTMin and Parameters.isAdaptiveStepping:
        print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
        print("Time step dropped below user-defined tolerance at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
        raise RuntimeError
      #----------------------------------
      # Update Dirichlet BCs at time t_n.
      #----------------------------------
      g[1,:] = g[0,:]
      #------------------------
      # Update simulation time.
      #------------------------
      Parameters.t += Parameters.dt
      n            += 1
      #--------------------------------
      # Update the BCs at time t_{n+1}.
      #--------------------------------
      g, GEXT = moduleFE.updateBC(g, GEXT, Parameters.t, Parameters)
      #----------------------
      # Reset N-R parameters.
      #----------------------
      Rtol   = 1
      normR  = 1
      k      = 0
      #---------------------------------
      # Begin Newton-Raphson iterations.
      #---------------------------------
      while (Rtol > Parameters.tolr) and (normR > Parameters.tola):
        #----------------------------------
        # Increment iterator and solutions.
        #----------------------------------
        k += 1
        D += del_D
        #----------------------------------
        # Form global tangent and residual.
        #----------------------------------
        dR, R = moduleVE.assemble_G(LM, g, GEXT, D, V, A, Parameters)
        #------------------------------------
        # Compute global displacement update.
        #------------------------------------
        del_D = moduleVE.solve_G(dR, R, Parameters)
        #----------------------------
        # Compute tolerance and norm.
        #----------------------------
        if (k == 1):
          R0 = R

        Rtol  = np.linalg.norm(R, ord=Parameters.norm_ord)/np.linalg.norm(R0, ord=Parameters.norm_ord)
        normR = np.linalg.norm(R, ord=Parameters.norm_ord)
        #------------------
        # Check iterations.
        #------------------
        if k == Parameters.kmax:
          print("--------------------\nCOMPUTATIONAL ERROR:\n--------------------")
          print("Relative tolerance = {:.3e}".format(Rtol))
          print("Reached max number of iterations for simulations at t = %.2es and dt = %.2es." %(Parameters.t, Parameters.dt))
          raise RuntimeError

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

        m        += 1
        save_flag = False
        t_saved   = Parameters.t
  #------------------------------------------------------
  # Exit if an error occurred in any of the computations.
  #------------------------------------------------------
  except (FloatingPointError, RuntimeError):
    print("--------------------------\nIMPLICIT INTEGRATOR ERROR:\n--------------------------")
    print("Encountered FloatingPointError or RuntimeError.")
    print("Data will be saved up to t = {:.3e}s.".format(t_saved))
    if Parameters.printTraceback:
      print("---------------\nFULL TRACEBACK:\n---------------")
      print(traceback.format_exc())

  return D_solve, V_solve, A_solve, t_solve, SS_solve, ISV_solve

