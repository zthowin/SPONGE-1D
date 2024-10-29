#--------------------------------------------------------------------------------------
# Module to initiate problem parameters and call a time integrator for the
# elastodynamics/elastics equations.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edits:   July 19, 2024
#--------------------------------------------------------------------------------------
import sys, time

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import moduleFE
except ImportError:
  sys.exit("MODULE WARNING. moduleFE.py not found, check configuration.")

try:
  import moduleRK
except ImportError:
  sys.exit("MODULE WARNING. moduleRK.py not found, check configuration.")

try:
  import moduleCD
except ImportError:
  sys.exit("MODULE WARNING. moduleCD.py not found, check configuration.")

try:
  import moduleNR
except ImportError:
  sys.exit("MODULE WARNING. moduleNR.py not found, check configuration.")

def main(params):

  start_time = time.time()
  #----------------------------
  # Initialize location matrix.
  #----------------------------
  LM = moduleFE.initLM(params)
  #----------------------------
  # Set initial Dirichlet BCs.
  #---------------------------
  LM = moduleFE.initDirichletBCs(LM, params)
  #-----------------------------------
  # Initialize external force vectors.
  #-----------------------------------
  GEXT = moduleFE.initNeumannBCs(params)
  #------------------------
  # Set initial conditions.
  #------------------------
  D, V, A = moduleFE.applyIC(LM, params)
  #---------------------------------
  # Integrate variational equations.
  #---------------------------------
  if params.integrationScheme == 'RKFNC' or params.integrationScheme == 'RKBS':
    savedData = moduleRK.integrate_u_FO(LM, GEXT, D, V, A, params)
  elif params.integrationScheme == 'Central-difference':
    savedData = moduleCD.integrate_u(LM, GEXT, D, V, A, params)
  elif params.integrationScheme == 'Newmark-beta':
    savedData = moduleNR.integrate_Newmark_beta(LM, [GEXT, np.zeros(params.ndof)], D, V, A, params)
  elif params.integrationScheme == 'Trapezoidal':
    savedData = moduleNR.integrate_Trapezoidal(LM, [GEXT, np.zeros(params.ndof)], D, V, params)
  elif params.integrationScheme == 'Quasi-static':
    savedData = moduleNR.integrate_QS(LM, GEXT, D, params)
  else:
    sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nOther integration schemes not yet implemented for (u).")

  end_time = time.time()
  time_elapsed = (end_time - start_time)/60.0
  print("---")
  print("Finished solving in %.2f minutes." %time_elapsed)
  print("Saving data...")
  #----------------------------------------------------
  # Initialize file names for plotting/post-processing.
  #----------------------------------------------------
  disp_fname     = params.dataDir + 'displacement.npy' 
  vel_fname      = params.dataDir + 'velocity.npy'    
  acc_fname      = params.dataDir + 'acceleration.npy'
  time_fname     = params.dataDir + 'time.npy'
  P11_fname      = params.dataDir + 'P11.npy'             
  sig11_fname    = params.dataDir + 'sig11.npy'                
  J_fname        = params.dataDir + 'J.npy'
  JD_fname       = params.dataDir + 'JDot.npy'
  p_fname        = params.dataDir + 'press.npy'
  ps_E_fname     = params.dataDir + 'ps_E.npy'
  tau_fname      = params.dataDir + 'tau.npy'
  Q_fname        = params.dataDir + 'Q.npy'

  dt_fname      = params.dataDir + 'dt.npy'
  tdt_fname     = params.dataDir + 'tdt.npy'
  #--------------
  # Extract data.
  #--------------
  D_solve   = savedData[0]
  V_solve   = savedData[1]
  A_solve   = savedData[2]
  t_solve   = savedData[3]
  SS_solve  = savedData[4]
  ISV_solve = savedData[5]
  #-------------------------------------
  # Eliminate erroneous trailing zeroes.
  #-------------------------------------
  try:
    clip = np.where(t_solve < 1e-12)[0][1]
  except IndexError:
    clip = None
  
  np.save(disp_fname,    D_solve[0:clip],         allow_pickle=False)
  np.save(vel_fname,     V_solve[0:clip],         allow_pickle=False)
  np.save(acc_fname,     A_solve[0:clip],         allow_pickle=False)
  np.save(time_fname,    t_solve[0:clip],         allow_pickle=False)

  np.save(P11_fname,     SS_solve[0:clip,:,0,:],  allow_pickle=False)
  np.save(sig11_fname,   SS_solve[0:clip,:,1,:],  allow_pickle=False)
  np.save(J_fname,       SS_solve[0:clip,:,2,:],  allow_pickle=False)
  np.save(tau_fname,     SS_solve[0:clip,:,3,:],  allow_pickle=False)
  np.save(JD_fname,      SS_solve[0:clip,:,4,:],  allow_pickle=False)

  np.save(p_fname,       ISV_solve[0:clip,:,2,:], allow_pickle=False)
  np.save(ps_E_fname,    ISV_solve[0:clip,:,2,:], allow_pickle=False)
  np.save(Q_fname,       ISV_solve[0:clip,:,18,:], allow_pickle=False)

  if params.isAdaptiveStepping and params.adaptiveSave:
    dt_solve  = savedData[6]
    tdt_solve = savedData[7]
    clip_adpt = np.where(tdt_solve == 0.0)[0][2]
    np.save(dt_fname, dt_solve[0:clip_adpt],   allow_pickle=False)
    np.save(tdt_fname, tdt_solve[0:clip_adpt], allow_pickle=False)

  print("Data saved successfully.")
  return
