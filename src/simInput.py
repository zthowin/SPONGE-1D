#---------------------------------------------------------------------------------
# Module providing encapsulated input data handling capabilities to configure the 
# 1D uniaxial strain single-phase or multiphase simulations.
#
# Author:        Zachariah Irwin
# Institution:   University of Colorado Boulder
# Last Edits:    October 18, 2024
#---------------------------------------------------------------------------------
import sys, os

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

#---------------------------------------------------------------------------------
# Problem input data structure, with all input data handled in a protected manner.
#---------------------------------------------------------------------------------
class SimInputs:
    
  #---------------------------------------
  # Initialize the class with member data.
  #---------------------------------------
  def __init__(self, a_FileName):
      
    self.m_InputFile = a_FileName
    temp             = a_FileName.rfind('/')
    
    if temp != -1:
        self.m_RootPath = a_FileName[0:temp]+'/'
    else:
        self.m_RootPath = ''
  #------------------------------------------------------------------
  # Read a formated ASCII input file and populate member data fields.
  #------------------------------------------------------------------
  def readInputFile(self):
      
    inputFileObj = open(self.m_InputFile)
    
    for line in inputFileObj:
        
      if not line.startswith('#'):
          
        lineDict = line.split('=')

        if lineDict[0].strip() == 'Physics':
          self.m_Physics = lineDict[1].strip()

        elif lineDict[0].strip() == 'Dynamics':
          if lineDict[1].strip() == 'True' or lineDict[1].strip() == 'true':
            self.m_isDynamics = True
          else:
            self.m_isDynamics = False

        elif lineDict[0].strip() == 'Project directory':
          tempRoot = lineDict[1].strip()
          if tempRoot != 'default': self.m_RootPath = tempRoot

        elif lineDict[0].strip() == 'Output directory':
          self.m_simPath = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation timing':
          tempList           = lineDict[1].strip().split()
          self.m_TStart      = float(tempList[3])
          self.m_TStop       = float(tempList[8])
          self.m_DT          = float(tempList[13])
          self.m_TOutput     = int(tempList[17])

        elif lineDict[0].strip() == 'Integration setup':
          tempList                 = lineDict[1].strip().split()
          self.m_IntegrationScheme = tempList[2]
          self.m_Lumping           = tempList[6]

        elif lineDict[0].strip() == 'Adaptive time stepping setup':
          tempList = lineDict[1].strip().split()

          if tempList[2] == 'TRUE' or tempList[2] == 'True':
            self.m_isAdaptiveStepping = True
          else:
            self.m_isAdaptiveStepping = False

          self.m_adaptiveKMax     = int(tempList[6])
          self.m_adaptiveIncrease = float(tempList[10])
          self.m_adaptiveDecrease = float(tempList[14])
          self.m_adaptiveDTMax    = float(tempList[19])
          self.m_adaptiveDTMin    = float(tempList[24])
          self.m_adaptiveStart    = float(tempList[29])
          self.m_adaptiveStop     = float(tempList[34])

          if tempList[38] == 'TRUE' or tempList[38] == 'True':
            self.m_adaptiveSave = True
          else:
            self.m_adaptiveSave = False

        elif lineDict[0].strip() == 'Newmark parameters':
          tempList = lineDict[1].strip().split()
          try:
            self.m_beta = float(tempList[2])
          except ValueError:
            num, denom = tempList[2].split('/')
            self.m_beta = float(num)/float(denom)
          try:
            self.m_gamma = float(tempList[6])
          except ValueError:
            num, denom = tempList[6].split('/')
            self.m_gamma = float(num)/float(denom)
          self.m_alpha  = float(tempList[10])

        elif lineDict[0].strip() == 'Element parameters':
          tempList            = lineDict[1].strip().split()
          self.m_Element_Type = tempList[2]
          self.m_ne           = int(tempList[6])
          self.m_Gauss_Order  = int(tempList[13])

        elif lineDict[0].strip() == 'Error parameters':
          tempList = lineDict[1].strip().split()
          try:
            self.m_tolr = float(tempList[3])
          except ValueError:
            self.m_tolr = None  
          try:
            self.m_tola = float(tempList[8])
          except ValueError:
            self.m_tola = None
          try:
            self.m_kmax = int(tempList[13])
          except ValueError:
            self.m_kmax = None
          try:
            self.m_norm_ord = int(tempList[17])
          except ValueError:
            if tempList[17] == 'np.inf' or tempList[17] == 'inf':
              self.m_norm_ord = np.inf
            elif tempList[17] == '-np.inf' or tempList[17] == '-inf':
              self.m_norm_ord = -np.inf
            else:
              self.m_norm_ord = None
          try:
            self.m_SF = float(tempList[21])
          except ValueError:
            self.m_SF = None

        elif lineDict[0].strip() == 'Constitutive modeling':
          tempList = lineDict[1].strip().split()
          self.m_fluidModel = tempList[2]
          self.m_solidModel = tempList[6]
          self.m_nu_0       = float(tempList[11])
          self.m_C0         = float(tempList[15])
          self.m_C1         = float(tempList[19])
          self.m_alpha_stab = float(tempList[23])
          if tempList[27] == 'True' or tempList[27] == 'TRUE':
            self.m_SUPG = True
          else:
            self.m_SUPG = False
          try:
            self.m_h0       = float(tempList[31])
            self.m_h1       = float(tempList[35])
          except IndexError:
            self.m_h0       = 0
            self.m_h1       = 0
        
        elif lineDict[0].strip() == 'Lung parameters':
          tempList        = lineDict[1].strip().split()
          self.m_c1       = float(tempList[2])
          self.m_c2       = float(tempList[6])
          self.m_B1       = float(tempList[10])
          self.m_mu_prime = float(tempList[14])
          self.m_Z0       = float(tempList[18])
          self.m_alpha_D  = float(tempList[22])
          self.m_omega_D  = float(tempList[26])

        elif lineDict[0].strip() == 'Column geometry':
          tempList        = lineDict[1].strip().split()
          self.m_Geo      = tempList[2]
          self.m_R        = float(tempList[7])
          self.m_H0       = float(tempList[12])

        elif lineDict[0].strip() == 'Young\'s modulus (Pa)':
          try:
            self.m_emod = float(lineDict[1].strip())
          except ValueError:
            self.m_emod = None

        elif lineDict[0].strip() == 'Poisson\'s ratio':
          try:
            self.m_nu = float(lineDict[1].strip())
          except ValueError:
            self.m_nu = None

        elif lineDict[0].strip() == 'Shear modulus (G, mu, Pa)':
          try:
            self.m_mu = float(lineDict[1].strip())
          except ValueError:
            self.m_mu = None

        elif lineDict[0].strip() == 'Bulk moduli (Pa)':
          tempList = lineDict[1].strip().split()
          try:
            self.m_KSkel = float(tempList[2])
          except ValueError:
            self.m_KSkel = None
          try:
            self.m_KF = float(tempList[6])
          except ValueError:
            self.m_KF = None        
          try:
            self.m_KS = float(tempList[10])
          except ValueError:
            self.m_KS = None
 
        elif lineDict[0].strip() == 'Lambda (Pa)':
          try:
            self.m_lambda = float(lineDict[1].strip())
          except ValueError:
            self.m_lambda = None

        elif lineDict[0].strip() == 'Body forces (N/kg)':
          tempList       = lineDict[1].strip().split()
          self.m_Gravity = float(tempList[2])

        elif lineDict[0].strip() == 'Volume fraction of constituents':
          tempList    = lineDict[1].strip().split()
          self.m_nf_0 = float(tempList[2])

        elif lineDict[0].strip() == 'Real mass density of constituents (kg/m^3)':
          tempList       = lineDict[1].strip().split()
          self.m_rhosR_0 = float(tempList[2])
          try:
            self.m_rhofR_0 = float(tempList[6])
          except ValueError:
            self.m_rhofR_0 = None

        elif lineDict[0].strip() == 'Specific heat of constituents (J/kg-K)':
          tempList = lineDict[1].strip().split()
          try:
            self.m_cvs = float(tempList[2])
          except ValueError:
            self.m_cvs = None
          try:
            self.m_cvf = float(tempList[6])
          except ValueError:
            self.m_cvf = None

        elif lineDict[0].strip() == 'Volumetric CTE of constituents (1/K)':
          tempList  = lineDict[1].strip().split()
          try:
            self.m_As = float(tempList[2])
          except ValueError:
            self.m_As = None
          try:
            self.m_Af = float(tempList[6])
          except ValueError:
            self.m_Af = None

        elif lineDict[0].strip() == 'Thermal conductivity of constituents (W/m-K)':
          tempList = lineDict[1].strip().split()
          try:
            self.m_ks = float(tempList[2])
          except ValueError:
            self.m_ks = None
          try:
            self.m_kf = float(tempList[6])
          except ValueError:
            self.m_kf = None  

        elif lineDict[0].strip() == 'Convective heat transfer (W/m^3-K)':
          try:
            self.m_k_exchange = float(lineDict[1].strip().split()[0])
          except ValueError:
            self.m_k_exchange = None

        elif lineDict[0].strip() == 'Hydraulic conductivity function':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'None' or tempList[2] == 'none':
            self.m_khatType = None
          else:
            self.m_khatType = tempList[2]
          try:
            self.m_kappa = float(tempList[6])
          except ValueError:
            self.m_kappa = None

        elif lineDict[0].strip() == 'Intrinsic permeability (m^2)':
          try:
            self.m_intrPerm = float(lineDict[1].strip())
          except ValueError:
            self.m_intrPerm = None

        elif lineDict[0].strip() == 'Pore fluid viscosity':
          tempList = lineDict[1].strip().split()
          try:
            self.m_fluidShearVisc = float(tempList[3])
          except ValueError:
            self.m_fluidShearVisc = None
          try:
            self.m_fluidBulkVisc  = float(tempList[8])
          except ValueError:
            self.m_fluidBulkVisc = None
          if tempList[12] == 'True' or tempList[12] == 'TRUE':
            self.m_DarcyBrinkman = True
          else:
            self.m_DarcyBrinkman = False

        elif lineDict[0].strip() == 'Solid displacement BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_solidDisplacementApply     = True
          else:
            self.m_solidDisplacementApply     = False
          self.m_solidDisplacementLocation    = tempList[6]
          self.m_solidDisplacementApplication = tempList[10]
          try:
            self.m_solidDisplacementValue     = float(tempList[15])
          except ValueError:
            self.m_solidDisplacementValueTop  = float(tempList[15].split(',')[0])
            self.m_solidDisplacementValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_solidDisplacementT0        = float(tempList[20])
            self.m_solidDisplacementT1        = float(tempList[25])
          except ValueError:
            self.m_solidDisplacementT0Top     = float(tempList[20].split(',')[0])
            self.m_solidDisplacementT0Bot     = float(tempList[20].split(',')[1])
            self.m_solidDisplacementT1Top     = float(tempList[25].split(',')[0])
            self.m_solidDisplacementT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Solid velocity BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_solidVelocityApply     = True
          else:
            self.m_solidVelocityApply     = False
          self.m_solidVelocityLocation    = tempList[6]
          self.m_solidVelocityApplication = tempList[10]
          try:
            self.m_solidVelocityValue     = float(tempList[15])
          except ValueError:
            self.m_solidVelocityValueTop  = float(tempList[15].split(',')[0])
            self.m_solidVelocityValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_solidVelocityT0        = float(tempList[20])
            self.m_solidVelocityT1        = float(tempList[25])
          except ValueError:
            self.m_solidVelocityT0Top     = float(tempList[20].split(',')[0])
            self.m_solidVelocityT0Bot     = float(tempList[20].split(',')[1])
            self.m_solidVelocityT1Top     = float(tempList[25].split(',')[0])
            self.m_solidVelocityT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Traction BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_tractionApply     = True
          else:
            self.m_tractionApply     = False
          self.m_tractionLocation    = tempList[6]
          self.m_tractionApplication = tempList[10]
          try:
            self.m_tractionOmega     = float(tempList[15])
          except ValueError:
            self.m_tractionOmegaTop  = float(tempList[15].split(',')[0])
            self.m_tractionOmegaBot  = float(tempList[15].split(',')[1])
          try:
            self.m_tractionValue     = float(tempList[20])
          except ValueError:
            self.m_tractionValueTop  = float(tempList[20].split(',')[0])
            self.m_tractionValueBot  = float(tempList[20].split(',')[1])
          try:
            self.m_tractionT0        = float(tempList[25])
            self.m_tractionT1        = float(tempList[30])
          except ValueError:
            self.m_tractionT0Top     = float(tempList[25].split(',')[0])
            self.m_tractionT0Bot     = float(tempList[25].split(',')[1])
            self.m_tractionT1Top     = float(tempList[30].split(',')[0])
            self.m_tractionT1Bot     = float(tempList[30].split(',')[1])

        elif lineDict[0].strip() == 'Pore fluid pressure IC (Pa)':
          self.m_p_f0 = float(lineDict[1].strip())

        elif lineDict[0].strip() == 'Pore fluid pressure BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_pressureApply     = True
          else:
            self.m_pressureApply     = False
          self.m_pressureLocation    = tempList[6]
          self.m_pressureApplication = tempList[10]
          try:
            self.m_pressureValue     = float(tempList[15])
          except ValueError:
            self.m_pressureValueTop  = float(tempList[15].split(',')[0])
            self.m_pressureValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_pressureT0        = float(tempList[20])
            self.m_pressureT1        = float(tempList[25])
          except ValueError:
            self.m_pressureT0Top     = float(tempList[20].split(',')[0])
            self.m_pressureT0Bot     = float(tempList[20].split(',')[1])
            self.m_pressureT1Top     = float(tempList[25].split(',')[0])
            self.m_pressureT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Pore fluid flux BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_fluxApply     = True
          else:
            self.m_fluxApply     = False
          self.m_fluxLocation    = tempList[6]
          self.m_fluxApplication = tempList[10]
          try:
            self.m_fluxValue     = float(tempList[15])
          except ValueError:
            self.m_fluxValueTop  = float(tempList[15].split(',')[0])
            self.m_fluxValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_fluxT0        = float(tempList[20])
            self.m_fluxT1        = float(tempList[25])
          except ValueError:
            self.m_fluxT0Top     = float(tempList[20].split(',')[0])
            self.m_fluxT0Bot     = float(tempList[20].split(',')[1])
            self.m_fluxT1Top     = float(tempList[25].split(',')[0])
            self.m_fluxT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Fluid displacement BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_fluidDisplacementApply     = True
          else:
            self.m_fluidDisplacementApply     = False
          self.m_fluidDisplacementLocation    = tempList[6]
          self.m_fluidDisplacementApplication = tempList[10]
          try:
            self.m_fluidDisplacementValue     = float(tempList[15])
          except ValueError:
            self.m_fluidDisplacementValueTop  = float(tempList[15].split(',')[0])
            self.m_fluidDisplacementValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_fluidDisplacementT0        = float(tempList[20])
            self.m_fluidDisplacementT1        = float(tempList[25])
          except ValueError:
            self.m_fluidDisplacementT0Top     = float(tempList[20].split(',')[0])
            self.m_fluidDisplacementT0Bot     = float(tempList[20].split(',')[1])
            self.m_fluidDisplacementT1Top     = float(tempList[25].split(',')[0])
            self.m_fluidDisplacementT1Bot     = float(tempList[25].split(',')[1])  

        elif lineDict[0].strip() == 'Fluid velocity BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_fluidVelocityApply     = True
          else:
            self.m_fluidVelocityApply     = False
          self.m_fluidVelocityLocation    = tempList[6]
          self.m_fluidVelocityApplication = tempList[10]
          try:
            self.m_fluidVelocityValue     = float(tempList[15])
          except ValueError:
            self.m_fluidVelocityValueTop  = float(tempList[15].split(',')[0])
            self.m_fluidVelocityValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_fluidVelocityT0        = float(tempList[20])
            self.m_fluidVelocityT1        = float(tempList[25])
          except ValueError:
            self.m_fluidVelocityT0Top     = float(tempList[20].split(',')[0])
            self.m_fluidVelocityT0Bot     = float(tempList[20].split(',')[1])
            self.m_fluidVelocityT1Top     = float(tempList[25].split(',')[0])
            self.m_fluidVelocityT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Temperature IC (K)':
          tempList    = lineDict[1].strip().split()
          try:
            self.m_Ts_0 = float(tempList[2])
          except ValueError:
            self.m_Ts_0 = None
          try:
            self.m_Tf_0 = float(tempList[6])
          except ValueError:
            self.m_Tf_0 = None

        elif lineDict[0].strip() == 'Solid temperature BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_solidTempApply       = True
          else:
            self.m_solidTempApply     = False
          self.m_solidTempLocation    = tempList[6]
          self.m_solidTempApplication = tempList[10]
          try:
            self.m_solidTempValue     = float(tempList[15])
          except ValueError:
            self.m_solidTempValueTop  = float(tempList[15].split(',')[0])
            self.m_solidTempValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_solidTempT0        = float(tempList[20])
            self.m_solidTempT1        = float(tempList[25])
          except ValueError:
            self.m_solidTempT0Top     = float(tempList[20].split(',')[0])
            self.m_solidTempT0Bot     = float(tempList[20].split(',')[1])
            self.m_solidTempT1Top     = float(tempList[25].split(',')[0])
            self.m_solidTempT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Fluid temperature BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_fluidTempApply     = True
          else:
            self.m_fluidTempApply     = False
          self.m_fluidTempLocation    = tempList[6]
          self.m_fluidTempApplication = tempList[10]
          try:
            self.m_fluidTempValue     = float(tempList[15])
          except ValueError:
            self.m_fluidTempValueTop  = float(tempList[15].split(',')[0])
            self.m_fluidTempValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_fluidTempT0        = float(tempList[20])
            self.m_fluidTempT1        = float(tempList[25])
          except ValueError:
            self.m_fluidTempT0Top     = float(tempList[20].split(',')[0])
            self.m_fluidTempT0Bot     = float(tempList[20].split(',')[1])
            self.m_fluidTempT1Top     = float(tempList[25].split(',')[0])
            self.m_fluidTempT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Solid heat flux BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_solidHeatFluxApply     = True
          else:
            self.m_solidHeatFluxApply     = False
          self.m_solidHeatFluxLocation    = tempList[6]
          self.m_solidHeatFluxApplication = tempList[10]
          try:
            self.m_solidHeatFluxValue     = float(tempList[15])
          except ValueError:
            self.m_solidHeatFluxValueTop  = float(tempList[15].split(',')[0])
            self.m_solidHeatFluxValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_solidHeatFluxT0        = float(tempList[20])
            self.m_solidHeatFluxT1        = float(tempList[25])
          except ValueError:
            self.m_solidHeatFluxT0Top     = float(tempList[20].split(',')[0])
            self.m_solidHeatFluxT0Bot     = float(tempList[20].split(',')[1])
            self.m_solidHeatFluxT1Top     = float(tempList[25].split(',')[0])
            self.m_solidHeatFluxT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'Fluid heat flux BC':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_fluidHeatFluxApply     = True
          else:
            self.m_fluidHeatFluxApply     = False
          self.m_fluidHeatFluxLocation    = tempList[6]
          self.m_fluidHeatFluxApplication = tempList[10]
          try:
            self.m_fluidHeatFluxValue     = float(tempList[15])
          except ValueError:
            self.m_fluidHeatFluxValueTop  = float(tempList[15].split(',')[0])
            self.m_fluidHeatFluxValueBot  = float(tempList[15].split(',')[1])
          try:
            self.m_fluidHeatFluxT0        = float(tempList[20])
            self.m_fluidHeatFluxT1        = float(tempList[25])
          except ValueError:
            self.m_fluidHeatFluxT0Top     = float(tempList[20].split(',')[0])
            self.m_fluidHeatFluxT0Bot     = float(tempList[20].split(',')[1])
            self.m_fluidHeatFluxT1Top     = float(tempList[25].split(',')[0])
            self.m_fluidHeatFluxT1Bot     = float(tempList[25].split(',')[1])

        elif lineDict[0].strip() == 'MMS':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'TRUE' or tempList[2] == 'True':
            self.m_MMS = True
            self.m_MMS_SolidSolutionType = tempList[6]
            if tempList[10] == 'NONE' or tempList[10] == 'None':
              self.m_MMS_PressureSolutionType = None
            else:
              self.m_MMS_PressureSolutionType = tempList[10]
            if tempList[14] == 'NONE' or tempList[14] == 'None':
              self.m_MMS_FluidSolutionType = None
            else:
              self.m_MMS_FluidSolutionType = tempList[14]
          else:
            self.m_MMS                      = False
            self.m_MMS_SolidSolutionType    = None
            self.m_MMS_PressureSolutionType = None
            self.m_MMS_FluidSolutionType    = None

    inputFileObj.close()

    if self.m_Physics == 'user':
      return

    self.m_H0e    = self.m_H0/self.m_ne
    self.m_jac    = self.m_H0e/2.
    if self.m_Geo == 'Cylinder' or self.m_Geo == 'cylinder':
      self.m_Area = np.pi*self.m_R**2
    elif self.m_Geo == 'Rectangle' or self.m_Geo == 'rectangle':
      self.m_Area = self.m_R**2
    else:
      sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nGeometry not recognized.")

    if self.m_KF is None or self.m_nf_0 == 0:
      self.m_ns_0  = 1.
      self.m_rho_0 = self.m_rhosR_0
    else:
      self.m_ns_0    = 1. - self.m_nf_0
      self.m_rho_0   = self.m_rhosR_0*self.m_ns_0 + self.m_rhofR_0*self.m_nf_0

    try:
      self.m_gammaF = self.m_Gravity*self.m_rhofR_0
    except TypeError:
      self.m_gammaF = None

    self.calculateElasticModuli()

    return
  #------------------------------------------------
  # Subroutine to calculate missing elastic moduli.
  #------------------------------------------------
  def calculateElasticModuli(self):
    #--------------------------------------------------------
    # Handle cases where shear modulus mu must be calculated.
    #--------------------------------------------------------
    if self.m_mu is None and ((self.m_KSkel is not None) and (self.m_emod is not None)):
      self.m_mu = (3*self.m_KSkel*self.m_emod)/(9*self.m_KSkel - self.m_emod)

    elif self.m_mu is None and ((self.m_KSkel is not None) and (self.m_lambda is not None)):
      self.m_mu = (3/2) * (self.m_KSkel - self.m_lambda)

    elif self.m_mu is None and ((self.m_KSkel is not None) and (self.m_nu is not None)):
      self.m_mu = (3*self.m_KSkel*(1 - 2*self.m_nu))/(2*(1 + self.m_nu))

    elif self.m_mu is None and ((self.m_emod is not None) and (self.m_lambda is not None)):
      R         = np.sqrt(self.m_emod**2 + 9*self.m_lambda**2 + 2*self.m_emod*self.m_lambda)
      self.m_mu = (self.m_emod - 3*self.m_lambda + R)/4

    elif self.m_mu is None and ((self.m_nu is not None) and (self.m_emod is not None)):
      self.m_mu = self.m_emod/(2 * (1 + self.m_nu))

    elif self.m_mu is None and ((self.m_lambda is not None) and (self.m_nu is not None)):
      self.m_mu = (self.m_lambda*(1 - self.m_nu))/(2*self.m_nu)

    #-------------------------------------------------------------
    # Handle cases where Lame parameter lambda must be calculated.
    #-------------------------------------------------------------
    if self.m_lambda is None and ((self.m_KSkel is not None) and (self.m_emod is not None)):
      self.m_lambda = (3*self.m_KSkel*(3*self.m_KSkel - self.m_emod))/(9*self.m_KSkel - self.m_emod)

    elif self.m_lambda is None and ((self.m_KSkel is not None) and (self.m_mu is not None)):
      self.m_lambda = self.m_KSkel - (2/3)*self.m_mu

    elif self.m_lambda is None and ((self.m_KSkel is not None) and (self.m_nu is not None)):
      self.m_lambda = (3*self.m_KSkel*self.m_nu)/(1 + self.m_nu)

    elif self.m_lambda is None and ((self.m_emod is not None) and (self.m_mu is not None)):
      self.m_lambda = (self.m_mu*(self.m_emod - 2*self.m_mu))/(3*self.m_mu - self.m_emod)

    elif self.m_lambda is None and ((self.m_emod is not None) and (self.m_nu is not None)):
      self.m_lambda = (self.m_emod*self.m_nu)/((1 + self.m_nu)*(1 - 2*self.m_nu))

    elif self.m_lambda is None and ((self.m_mu is not None) and (self.m_nu is not None)):
      self.m_lambda = (2*self.m_mu*self.m_nu)/(1 - self.m_nu)

    #-----------------------------------------------------------------
    # Handle case where bulk modulus of parenchyma must be calculated.
    #-----------------------------------------------------------------
    if self.m_KSkel is None and ((self.m_emod is not None) and (self.m_lambda is not None)):
      R            = np.sqrt(self.m_emod**2 + 9*self.m_lambda**2 + 2*self.m_emod*self.m_lambda)
      self.m_KSkel = (self.m_emod + 3*self.m_lambda + R)/6

    elif self.m_KSkel is None and ((self.m_emod is not None) and (self.m_mu is not None)):
      self.m_KSkel = (self.m_emod*self.m_mu)/(3*(3*self.m_mu - self.m_emod))

    elif self.m_KSkel is None and ((self.m_emod is not None) and (self.m_nu is not None)):
      self.m_KSkel = self.m_emod/(3*(1 - self.m_nu))

    elif self.m_KSkel is None and ((self.m_lambda is not None) and (self.m_mu is not None)):
      self.m_KSkel = self.m_lambda + (2*self.m_mu/3.)

    elif self.m_KSkel is None and ((self.m_lambda is not None) and (self.m_nu is not None)):
      self.m_KSkel = (self.m_lambda*(1 + self.m_nu))/(3*self.m_nu)

    elif self.m_KSkel is None and ((self.m_mu is not None) and (self.m_nu is not None)):
      self.m_KSkel = (2*self.m_mu*(1 + self.m_nu))/(3*(1 - 2*self.m_nu))

    #--------------------------------------------------------------------
    # Handle case where Young's modulus of parenchyma must be calculated.
    #--------------------------------------------------------------------
    if self.m_emod is None and ((self.m_KSkel is not None) and (self.m_lambda is not None)):
      self.m_emod = (9*self.m_KSkel*(self.m_KSkel - self.m_lambda))/(3*self.m_KSkel - self.m_lambda)

    elif self.m_emod is None and ((self.m_KSkel is not None) and (self.m_mu is not None)):
      self.m_emod = (9*self.m_KSkel*self.m_mu)/(3*self.m_KSkel + self.m_mu)

    elif self.m_emod is None and ((self.m_KSkel is not None) and (self.m_nu is not None)):
      self.m_emod = 3*self.m_KSkel*(1 - 2*self.m_nu)

    elif self.m_emod is None and ((self.m_lambda is not None) and (self.m_mu is not None)):
      self.m_emod = (self.m_mu*(3*self.m_lambda + 2*self.m_mu))/(self.m_lambda + self.m_mu)

    elif self.m_emod is None and ((self.m_lambda is not None) and (self.m_nu is not None)):
      self.m_emod = (self.m_lambda*(1 + self.m_nu)*(1 - 2*self.m_nu))/self.m_nu

    elif self.m_emod is None and ((self.m_mu is not None) and (self.m_nu is not None)):
      self.m_emod = 2*self.m_mu*(1 + self._nu)

    #------------------------------------------------------------------
    # Handle case where Poisson ratio of parenchyma must be calculated.
    #------------------------------------------------------------------
    if self.m_nu is None and ((self.m_KSkel is not None) and (self.m_emod is not None)):
      self.m_nu = (3*self.m_KSkel - self.m_emod)/(6*self.m_KSkel)

    elif self.m_nu is None and ((self.m_KSkel is not None) and (self.m_lambda is not None)):
      self.m_nu = self.m_lambda/(3*self.m_KSkel - self.m_lambda)

    elif self.m_nu is None and ((self.m_KSkel is not None) and (self.m_mu is not None)):
      self.m_nu = (3*self.m_KSkel - 2*self.m_mu)/(2*(3*self.m_KSkel + self.m_mu))

    elif self.m_nu is None and ((self.m_emod is not None) and (self.m_lambda is not None)):
      R         = np.sqrt(self.m_emod**2 + 9*self.m_lambda**2 + 2*self.m_emod*self.m_lambda)
      self.m_nu = (2*self.m_lambda)/(self.m_emod + self.m_lambda + R)

    elif self.m_nu is None and ((self.m_emod is not None) and (self.m_mu is not None)):
      self.m_nu = self.m_emod/(2*self.m_mu) - 1

    elif self.m_nu is None and ((self.m_lambda is not None) and (self.m_mu is not None)):
      self.m_nu = self.m_lambda/(2*(self.m_lambda + self.m_mu))

    return
  #----------------------------------------------------------------------------
  # Read the standard LS-DYNA ASCII input file and populate member data fields.
  #
  # This is not used in SPONGE-1D /src/ code but in the subsequent /scripts/.
  # It assumes 1-D uniaxial strain hex (Q8) meshes.
  #----------------------------------------------------------------------------
  def readDYNAInputFile(self):
    inputFileObj = open(self.m_InputFile)

    for line in inputFileObj:
        
      if not line.startswith('$'):
          
        lineDict = line.split()
        
        if lineDict[0] == "*CONTROL_TIMESTEP":
          next(inputFileObj)
          self.DT = float(next(inputFileObj).split()[0])

        elif lineDict[0] == "*CONTROL_TERMINATION":
          next(inputFileObj)
          self.TStop = float(next(inputFileObj).split()[0])

        elif lineDict[0] == "*NODE":
          #-------------------------------------------
          # Check to see if the next line is a header.
          #-------------------------------------------
          if (next(inputFileObj).split()[1] == 'NID'):
            meshInfo = inputFileObj.readlines()
            for meshLine in meshInfo:
              #----------------------------------------
              # Check to see where the mesh info stops.
              #----------------------------------------
              if meshLine.startswith('$'):
                cut = meshInfo.index(meshLine)
                break
          else:
            meshInfo = inputFileObj.readlines()
            for meshLine in meshInfo:
              #----------------------------------------
              # Check to see where the mesh info stops.
              #----------------------------------------
              if meshLine.startswith('$'):
                cut = meshInfo.index(meshLine)
                break
          #-----------------------------
          # Remove trailing information.
          #-----------------------------
          meshInfo = meshInfo[:cut]
          #---------------------------
          # Get total number of nodes.
          #---------------------------
          numNodes = len(meshInfo)
          #--------------------------------------
          # Initialize array of mesh coordinates.
          #--------------------------------------
          self.coordsDYNA = np.zeros((numNodes, 3))
          #------------------------
          # Build mesh coordinates.
          #------------------------
          dof = 0
          for meshLine in meshInfo:
            data = meshLine.split()
            
            self.coordsDYNA[dof, 0] = float(data[1])
            self.coordsDYNA[dof, 1] = float(data[2])
            self.coordsDYNA[dof, 2] = float(data[3])
            
            dof += 1
          #--------------------------------
          # Build Gauss coordinates.
          #
          # Note: this assumes a "1-D" mesh
          #       and Q8 element type.
          #--------------------------------
          self.coordsDYNAG = np.linspace(self.coordsDYNA[4,2]/2, self.coordsDYNA[-1, 2] - self.coordsDYNA[4,2]/2, int((numNodes - 4)/4))

    self.Physics       = 'u'
    self.DarcyBrinkman = False
    self.ndofS         = numNodes
    self.Element_Type  = 'Q8'
    self.H0e           = self.coordsDYNAG[1] - self.coordsDYNAG[0]
    self.coordsDYNA    = np.delete(self.coordsDYNA, (0,1), axis=1)

    return
  #-----------------------------------------------------------------------
  # A utility function that displays and prints all the configures input
  # variables, and allows the user to verify the simulation configuration.
  #-----------------------------------------------------------------------
  def printAndVerify(self):

    print("---------------------------------------------------")
    print("The project directory for the simulation is       : {:s}".format(self.m_RootPath))
    print("The output directory for the simulation is        : {:s}".format(self.m_simPath))
    print("---------------------------------------------------")
    print("The physical formulation for the simulation is    : {:s}".format(self.m_Physics))
    print("The inertia terms are present                     : {!s}".format(self.m_isDynamics))
    print("---------------------------------------------------")
    print("The numerical integration scheme to be used is    : {!s}".format(self.m_IntegrationScheme))
    print("Mass matrix lumping enabled for                   : {!s}".format(self.m_Lumping))
    if self.m_gamma > 0:
      print("The Newmark integration parameters are            : Beta: {0:}, Gamma: {1:}".\
            format(self.m_beta, self.m_gamma))
    print("The integration start time is                     : {:.3e} s".format(self.m_TStart))
    print("The integration stop time is                      : {:.3e} s".format(self.m_TStop))
    print("The initial integration time step is              : {:.3e} s".format(self.m_DT))
    print("---------------------------------------------------")
    print("The simulation will use adaptive time stepping    : {!s}".format(self.m_isAdaptiveStepping))
    if self.m_isAdaptiveStepping:
      if self.m_gamma > 0:
        print("The number of iterations to decrease time step is : {:d}".format(self.m_adaptiveKMax))
        print("The time step will increase by a factor of        : {:f}".format(self.m_adaptiveIncrease))
        print("The time step will decrease by a factor of        : {:f}".format(self.m_adaptiveDecrease))
        print("The adaptive scheme will start at                 : {:.3e} s".format(self.m_adaptiveTStart))
        print("The adaptive scheme will stop at                  : {:.3e} s".format(self.m_adaptiveTStop))
      else:
        print("The safety factor is                              : {:.2e}".format(self.m_SF))
      print("The maximum time step will be                     : {:.3e} s".format(self.m_adaptiveDTMax))
      print("The minimum time step will be                     : {:.3e} s".format(self.m_adaptiveDTMin))
    print("---------------------------------------------------")
    if self.m_gamma > 0:
      print("The relative tolerance is                         : {:.2e}".format(self.m_tolr))
    print("The absolute tolerance is                         : {:.2e}".format(self.m_tola))
    if self.m_gamma > 0:
      print("The max number of iterations is                   : {:d}".format(self.m_kmax))
    print("The algebraic norm is of order                    : {!s}".format(self.m_norm_ord))
    
    if self.m_Physics == 'user':
      print("---------------------------------------------------")
      return

    print("---------------------------------------------------")
    print("The element type for the mesh is                  : {:s}".format(self.m_Element_Type))
    print("The number of Gauss points for an element is      : {:d}".format(self.m_Gauss_Order))
    print("The number of elements in the mesh is             : {:d}".format(self.m_ne))
    print("The size of each element is                       : {:f} m".format(self.m_H0e))
    print("---------------------------------------------------")
    print("The constitutive model for rhofR is               : {:s}".format(self.m_fluidModel))
    print("The functional form for hydraulic conductivity is : {:s}".format(self.m_khatType))
    if self.m_khatType == 'Strain-Exponential' or self.m_khatType == 'Hyperbolic':
      print("The scaling parameter 'kappa' is                  : {:.2f}".format(self.m_kappa))
    print("The constitutive model for the solid is           : {:s}".format(self.m_solidModel))
    print("The value of the viscous damping parameter is     : {:.2e} s".format(self.m_nu_0))
    print("The value of the shock viscosity constant C0 is   : {:.2f}".format(self.m_C0))
    print("The value of the shock viscosity constant C1 is   : {:.2f}".format(self.m_C1))
    print("The pressure stabilization parameter has a value  : {:.2e}".format(self.m_alpha_stab))
    print("SUPG stabilization is enabled                     : {!s}".format(self.m_SUPG))
    print("---------------------------------------------------")
    print("The following parameters are used in the Clayton-Freed model")
    print()
    print("The non-linear elastic bulk    constant c_1 is    : {:.2f}".format(self.m_c1))
    print("The non-linear elastic squeeze constant c_2 is    : {:.2f}".format(self.m_c2))
    print("The viscoelastic dynamic multiplier \\beta_1 is    : {:.2f}".format(self.m_B1))
    print("The pressure derivative of the shear modulus is   : {:.2f}".format(self.m_mu_prime))
    print("The normalized injury threshold is                : {:.2f}".format(self.m_Z0))
    print("The rate-independent damage kinetic multiplier is : {:.2f}".format(self.m_alpha_D))
    print("The rate enhanced damage kinetics multiplier is   : {:.2f}".format(self.m_omega_D))
    print("---------------------------------------------------")
    print("The column geometry is that of a                  : {:s}".format(self.m_Geo))
    print("The radius/width of the column is                 : {:.2f} m".format(self.m_R))
    print("The height of the column is                       : {:.2f} m".format(self.m_H0))
    print("The effective cross-sectional area is             : {:.4f} m^2".format(self.m_Area))
    print("---------------------------------------------------")
    print("The gravitational force per unit mass is          : {:.2f} N/kg".format(self.m_Gravity))
    print("---------------------------------------------------")
    print("The Young's modulus of the solid skeleton is      : {:.2e} Pa".format(self.m_emod))
    print("Poisson's ratio for the solid skeleton is         : {:.2f}".format(self.m_nu))
    print("The shear modulus of the solid skeleton is        : {:.2e} Pa".format(self.m_mu))
    print("Lame's first parameter for the solid skeleton is  : {:.2e} Pa".format(self.m_lambda))
    print("The bulk modulus of the solid skeleton is         : {:.2e} Pa".format(self.m_KSkel))
    print("The permeability of the solid skeleton is         : {:.2e} m^2".format(self.m_intrPerm))
    print("The volume fraction of the solid skeleton is      : {:f}".format(self.m_ns_0))
    print("The real mass density of the solid skeleton is    : {:.2f} kg/m^3".format(self.m_rhosR_0))
    if self.m_cvs is not None:
      print("The specific heat capacity of the solid is        : {:.2e} J/kg-K".format(self.m_cvs))
    if self.m_As is not None:
      print("The volumetric CTE of the solid skeleton is       : {:.2e} 1/K".format(self.m_As))
    if self.m_ks is not None:
      print("The thermal conductivity of the solid is          : {:.2e} W/m-K".format(self.m_ks))
    if self.m_KF is not None:
      print("---------------------------------------------------")
      print("The bulk modulus of the pore fluid is             : {:.2e} Pa".format(self.m_KF))
      print("The volume fraction of the pore fluid is          : {:f}".format(self.m_nf_0))
      print("The real mass density of the pore fluid is        : {:.2f} kg/m^3".format(self.m_rhofR_0))
      print("The shear viscosity of the pore fluid is          : {:.2e} Pa/s".format(self.m_fluidShearVisc))
      try:
        print("The bulk viscosity of the pore fluid is           : {:.3e} Pa/s".format(self.m_fluidBulkVisc))
      except TypeError:
        pass
      print("The unit weight of the pore fluid is              : {:.3e} N/m^3".format(self.m_gammaF))
      print("Darcy-Brinkman formulation will be enabled        : {!s}".format(self.m_DarcyBrinkman))
      if self.m_cvf is not None:
        print("The specific heat capacity of the pore fluid is   : {:.2e} J/kg-K".format(self.m_cvf))
      if self.m_Af is not None:
        print("The volumetric CTE of the pore fluid is           : {:.2e} 1/K".format(self.m_Af))
      if self.m_kf is not None:
        print("The thermal conductivity of the pore fluid is     : {:.2e} W/m-K".format(self.m_kf))
    print("---------------------------------------------------")
    print("The total mass density is                         : {:.2f} kg/m^3".format(self.m_rho_0))
    if self.m_k_exchange is not None:
      print("The convective heat transfer coefficient is       : {:.2f} W/m^3-K".format(self.m_k_exchange))
    print("---------------------------------------------------")
    print("A solid displacement Dirichlet BC will be applied : {!s}".format(self.m_solidDisplacementApply))
    if self.m_solidDisplacementApply:
      print("The location is                                   : {:s}".format(self.m_solidDisplacementLocation))
      print("The application type is                           : {:s}".format(self.m_solidDisplacementApplication))
      try:
        print("The magnitude is                                  : {:.2f} m".format(self.m_solidDisplacementValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} m".format(self.m_solidDisplacementValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} m".format(self.m_solidDisplacementValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_solidDisplacementT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_solidDisplacementT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_solidDisplacementT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_solidDisplacementT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_solidDisplacementT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_solidDisplacementT1Bot))
    print("---------------------------------------------------")
    print("A solid velocity Dirichlet BC will be applied     : {!s}".format(self.m_solidVelocityApply))
    if self.m_solidVelocityApply:
      print("The location is                                   : {:s}".format(self.m_solidVelocityLocation))
      print("The application type is                           : {:s}".format(self.m_solidVelocityApplication))
      try:
        print("The magnitude is                                  : {:.2f} m/s".format(self.m_solidVelocityValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} m/s".format(self.m_solidVelocityValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} m/s".format(self.m_solidVelocityValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_solidVelocityT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_solidVelocityT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_solidVelocityT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_solidVelocityT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_solidVelocityT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_solidVelocityT1Bot))
    print("---------------------------------------------------")
    print("A traction load BC will be applied                : {!s}".format(self.m_tractionApply))
    if self.m_tractionApply:
      print("The location is                                   : {:s}".format(self.m_tractionLocation))
      print("The application type is                           : {:s}".format(self.m_tractionApplication))
      try:
        print("The magnitude is                                  : {:.2e} Pa".format(self.m_tractionValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2e} Pa".format(self.m_tractionValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2e} Pa".format(self.m_tractionValueBot))
      try:
        print("The loading frequency is                          : {:.2f} rad/s".format(self.m_tractionOmega))
      except ValueError:
        print("The loading frequency at the top    of the mesh is: {:.2f} rad/s".format(self.m_tractionOmegaTop))
        print("The loading frequency at the bottom of the mesh is: {:.2f} rad/s".format(self.m_tractionOmegaBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_tractionT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_tractionT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_tractionT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_tractionT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_tractionT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_tractionT1Bot))
    print("---------------------------------------------------")
    print("The initial pore fluid pressure is                : {:.3e} Pa".format(self.m_p_f0))
    print("---------------------------------------------------")
    print("A pore fluid pressure BC will be applied          : {!s}".format(self.m_pressureApply))
    if self.m_pressureApply:
      print("The location is                                   : {:s}".format(self.m_pressureLocation))
      print("The application type is                           : {:s}".format(self.m_pressureApplication))
      try:
        print("The magnitude is                                  : {:.2e} Pa".format(self.m_pressureValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2e} Pa".format(self.m_pressureValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2e} Pa".format(self.m_pressureValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_pressureT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_pressureT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_pressureT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_pressureT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_pressureT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_pressureT1Bot))
    print("---------------------------------------------------")
    print("A pore fluid flux BC will be applied              : {!s}".format(self.m_fluxApply))
    if self.m_fluxApply:
      print("The location is                                   : {:s}".format(self.m_fluxLocation))
      print("The application type is                           : {:s}".format(self.m_fluxApplication))
      try:
        print("The magnitude is                                  : {:.2f} m/s".format(self.m_fluxValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} m/s".format(self.m_fluxValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} m/s".format(self.m_fluxValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_fluxT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_fluxT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_fluxT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_fluxT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_fluxT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_fluxT1Bot))
    print("---------------------------------------------------")
    print("A pore fluid displacement BC will be applied      : {!s}".format(self.m_fluidDisplacementApply))
    if self.m_fluidDisplacementApply:
      print("The location is                                   : {:s}".format(self.m_fluidDisplacementLocation))
      print("The application type is                           : {:s}".format(self.m_fluidDisplacementApplication))
      try:
        print("The magnitude is                                  : {:.2f} m".format(self.m_fluidDisplacementValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} m".format(self.m_fluidDisplacementValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} m".format(self.m_fluidDisplacementValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_fluidDisplacementT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_fluidDisplacementT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_fluidDisplacementT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidDisplacementT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_fluidDisplacementT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidDisplacementT1Bot))
    print("---------------------------------------------------")
    print("A pore fluid velocity BC will be applied          : {!s}".format(self.m_fluidVelocityApply))
    if self.m_fluidVelocityApply:
      print("The location is                                   : {:s}".format(self.m_fluidVelocityLocation))
      print("The application type is                           : {:s}".format(self.m_fluidVelocityApplication))
      try:
        print("The magnitude is                                  : {:.2f} m/s".format(self.m_fluidVelocityValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} m/s".format(self.m_fluidVelocityValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} m/s".format(self.m_fluidVelocityValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_fluidVelocityT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_fluidVelocityT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_fluidVelocityT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidVelocityT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_fluidVelocityT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidVelocityT1Bot))
    print("---------------------------------------------------")
    try:
      print("The initial solid      temperature is             : {:.2f} K".format(self.m_Ts_0))
      print("The initial pore fluid temperature is             : {:.2f} K".format(self.m_Tf_0))
      print("---------------------------------------------------")
    except TypeError:
      pass
    print("A solid temperature BC will be applied            : {!s}".format(self.m_solidTempApply))
    if self.m_solidTempApply:
      print("The location is                                   : {:s}".format(self.m_solidTempLocation))
      print("The application type is                           : {:s}".format(self.m_solidTempApplication))
      try:
        print("The magnitude is                                  : {:.2f} K".format(self.m_solidTempValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} K".format(self.m_solidTempValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} K".format(self.m_solidTempValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_solidTempT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_solidTempT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_solidTempT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_solidTempT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_solidTempT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_solidTempT1Bot))
    print("---------------------------------------------------")
    print("A pore fluid temperature BC will be applied       : {!s}".format(self.m_fluidTempApply))
    if self.m_fluidTempApply:
      print("The location is                                   : {:s}".format(self.m_fluidTempLocation))
      print("The application type is                           : {:s}".format(self.m_fluidTempApplication))
      try:
        print("The magnitude is                                  : {:.2f} K".format(self.m_fluidTempValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2f} K".format(self.m_fluidTempValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2f} K".format(self.m_fluidTempValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_fluidTempT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_fluidTempT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_fluidTempT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidTempT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_fluidTempT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidTempT1Bot))
    print("---------------------------------------------------")
    print("A solid heat flux BC will be applied              : {!s}".format(self.m_solidHeatFluxApply))
    if self.m_solidHeatFluxApply:
      print("The location is                                   : {:s}".format(self.m_solidHeatFluxLocation))
      print("The application type is                           : {:s}".format(self.m_solidHeatFluxApplication))
      try:
        print("The magnitude is                                  : {:.2e} W/m^2".format(self.m_solidHeatFluxValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2e} W/m^2".format(self.m_solidHeatFluxValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2e} W/m^2".format(self.m_solidHeatFluxValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_solidHeatFluxT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_solidHeatFluxT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_solidHeatFluxT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_solidHeatFluxT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_solidHeatFluxT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_solidHeatFluxT1Bot))
    print("---------------------------------------------------")
    print("A pore fluid heat flux BC will be applied         : {!s}".format(self.m_fluidHeatFluxApply))
    if self.m_fluidHeatFluxApply:
      print("The location is                                   : {:s}".format(self.m_fluidHeatFluxLocation))
      print("The application type is                           : {:s}".format(self.m_fluidHeatFluxApplication))
      try:
        print("The magnitude is                                  : {:.2e} W/m^2".format(self.m_fluidHeatFluxValue)) 
      except ValueError:
        print("The magnitude at the top    of the mesh is        : {:.2e} W/m^2".format(self.m_fluidHeatFluxValueTop))
        print("The magnitude at the bottom of the mesh is        : {:.2e} W/m^2".format(self.m_fluidHeatFluxValueBot))
      try:
        print("It will be applied starting at time               : {:.2e} s".format(self.m_fluidHeatFluxT0))
        print("It will be stopped starting at time               : {:.2e} s".format(self.m_fluidHeatFluxT1))
      except ValueError:
        print("The top    BC will be applied starting at time    : {:.2e} s".format(self.m_fluidHeatFluxT0Top))
        print("The top    BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidHeatFluxT1Top))
        print("The bottom BC will be applied starting at time    : {:.2e} s".format(self.m_fluidHeatFluxT0Bot))
        print("The bottom BC will be stopped starting at time    : {:.2e} s".format(self.m_fluidHeatFluxT1Bot))
    print("---------------------------------------------------")
    print("MMS is enabled                                    : {!s}".format(self.m_MMS))
    if self.m_MMS:
      print("The solid skeleton          solution space is     : {:s}".format(self.m_MMS_SolidSolutionType))
      try:
        print("The pore fluid pressure     solution space is     : {:s}".format(self.m_MMS_PressureSolutionType))
      except TypeError:
        pass
      try:
        print("The pore fluid displacement solution space is     : {:s}".format(self.m_MMS_FluidSolutionType))
      except TypeError:
        pass

    return
      
#--------------------------------------
# Read, print, verify mode for testing.
#--------------------------------------
if __name__ == '__main__':

  if len(sys.argv) < 2:
    sys.exit("Need input filename as an argument.")

  inputFile = sys.argv[1].strip()

  #-------------------
  # Set up input data.
  #-------------------
  inputData = SimInputs(inputFile)
  inputData.readInputFile()

  #-------------------------
  # Print input data fields.
  #-------------------------
  if os.path.splitext(inputFile)[1] == '.dat':
    inputData.printAndVerify()
  elif os.path.splitext(inputFile)[1] == '.k':
    sys.exit("------\nERROR:\n------\nPrint & Verify routine not in place for LS-DYNA input files.")

