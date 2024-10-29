#--------------------------------------------------------------------------------------------------
# Top-level script to initiate parameters for 1-D Lagrangian finite-element simulation plots.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 26, 2024
#--------------------------------------------------------------------------------------------------
import sys, os, shutil, argparse

try:
  REPO = os.environ['REPO']
except KeyError:
  sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")

sys.path.insert(1, REPO + '/src/')

try:
  import simInput
except ImportError:
  sys.exit("MODULE WARNING. /src/ modules not found, check configuration.")

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import matplotlib.pyplot as plt
except ImportError:
  sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
  import plotInput 
except ImportError:
  sys.exit("MODULE WARNING. plotInput.py not found, check configuration.")

try:
  import LiPlots
  import EringenPlots
  import deBoerPlots
  import plotDisplacement
  import plotVelocity
  import plotVelocityError
  import plotAcceleration
  import plotTemperature
  import plotVolumeFraction
  import plotPressure
  import plotPi
  import plotShear
  import plotVonMises
  import plotStress
  import plotStress22
  import plotJ
  import plotTemperatureGauss
  import plotAdiabaticAnalytical
  import plotTimeSteps
  import plotError
  import plotMMSError
  import animatePiContour
  import animatePressureContour
  import animateJacobianContour
  import animateJDotContour
  import animateVolumeFractionContour
  import animateDarcyContour
  import animateDisplacementContour
  import animateVelocityContour
  import animateViscousFluidStressContour
  import animateStressContour
  import animateVarkappaContour
  import animateTemperatureContour
  import animateMixtureTemperatureContour
  import animateConductionContour
  import animatePressureTemperatureContour
  import animateVelocityTemperatureContour
  import animateVolumePressureContour
  import animateVolumeTemperatureContour
  import animatePVTContour
  import animatePJTContour
  import animateKContour
except ImportError:
  sys.exit("MODULE WARNING. Individual plotting scripts not found, check configuration.")

#--------------------------------------------
# Utility class to store plotting parameters.
#--------------------------------------------
class Parameters:

  def __init__(self, inputData, gaussFlag):

    self.averageGauss = gaussFlag

    #-----------------------------
    # General plotting parameters.
    #-----------------------------
    self.isDeletePNGs  = inputData.m_deletePNGs
    self.outputDir     = inputData.m_plotPath
    self.timeScaling   = inputData.m_timeScaling
    self.dispScaling   = inputData.m_displacementScaling
    self.stressScaling = inputData.m_stressScaling
    self.filename      = inputData.m_filename
    self.DPI           = inputData.m_DPI
    self.framerate     = inputData.m_framerate

    self.is_xticks = inputData.m_is_xticks
    if self.is_xticks:
      self.xticks = inputData.m_xticks
    self.is_xticklabels = inputData.m_is_xticklabels
    if self.is_xticklabels:
      self.xticklabels = inputData.m_xticklabels
    self.secondaryXTicks = inputData.m_secondaryXTicks

    self.is_yticks = inputData.m_is_yticks
    if self.is_yticks:
      self.yticks = inputData.m_yticks
    self.is_yticklabels = inputData.m_is_yticklabels
    if self.is_yticklabels:
      self.yticklabels = inputData.m_yticklabels
    self.secondaryYTicks = inputData.m_secondaryYTicks

    self.legend = inputData.m_legend
    if self.legend:
      self.handleLength   = inputData.m_handleLength
      self.legendX        = inputData.m_legendX
      self.legendY        = inputData.m_legendY
      self.legendPosition = inputData.m_legendPosition

    self.grid = inputData.m_grid
    if self.grid:
      self.gridWhich = inputData.m_gridWhich

    self.xlim0 = inputData.m_xlim0
    self.xlim1 = inputData.m_xlim1
    try:
      self.ylim0 = inputData.m_ylim0
      self.ylim1 = inputData.m_ylim1
    except AttributeError:
      self.ylim00 = inputData.m_ylim00
      self.ylim01 = inputData.m_ylim01
      self.ylim02 = inputData.m_ylim02
      self.ylim10 = inputData.m_ylim10
      self.ylim11 = inputData.m_ylim11
      self.ylim12 = inputData.m_ylim12

    self.title = inputData.m_title
    if self.title:
      self.titleLoc  = inputData.m_titleLoc
      self.titleName = inputData.m_titleName

    self.titleFontSize  = inputData.m_titleFontSize
    self.xAxisFontSize  = inputData.m_xAxisFontSize
    self.yAxisFontSize  = inputData.m_yAxisFontSize
    self.legendFontSize = inputData.m_legendFontSize
    #-----------------------------
    # Parameters for Simulation A.
    #-----------------------------
    self.simA_Dir = inputData.m_simA_Path
    if not os.path.exists(self.simA_Dir):
      self.simA_Dir = REPO + '/' + self.simA_Dir
      if not os.path.exists(self.simA_Dir):
        sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nDirectory for Simulation A not found.")
    self.simA_Title          = inputData.m_simA_Title
    self.simA_Skip           = inputData.m_simA_Skip
    self.simA_SkipSecondary  = inputData.m_simA_SkipSecondary
    self.simA_SkipTertiary   = inputData.m_simA_SkipTertiary
    self.simA_SkipQuaternary = inputData.m_simA_SkipQuaternary
    self.simA_SkipQuinary    = inputData.m_simA_SkipQuinary

    self.simA_probe_1 = inputData.m_simA_probe_1
    self.simA_probe_2 = inputData.m_simA_probe_2
    self.simA_probe_3 = inputData.m_simA_probe_3
    self.simA_probe_4 = inputData.m_simA_probe_4

    self.simA_Linestyle_Alpha   = inputData.m_simA_Linestyle_Alpha
    self.simA_Linestyle_Bravo   = inputData.m_simA_Linestyle_Bravo
    self.simA_Linestyle_Charlie = inputData.m_simA_Linestyle_Charlie
    self.simA_Linestyle_Delta   = inputData.m_simA_Linestyle_Delta
    self.simA_Linestyle_Echo    = inputData.m_simA_Linestyle_Echo
    self.simA_Color_Alpha       = inputData.m_simA_Color_Alpha
    self.simA_Color_Bravo       = inputData.m_simA_Color_Bravo
    self.simA_Color_Charlie     = inputData.m_simA_Color_Charlie
    self.simA_Color_Delta       = inputData.m_simA_Color_Delta
    self.simA_Color_Echo        = inputData.m_simA_Color_Echo
    self.simA_fillstyle         = inputData.m_simA_fillstyle
    #-----------------------------------------
    # Get information from Simulation A setup.
    #-----------------------------------------
    for root, dirnames, filenames in os.walk(self.simA_Dir):
      for file in filenames:
        (shortname, extension) = os.path.splitext(file)
        if extension == ".dat" or extension == '.k':
          self.simA_InputFileName = file
          break
    try:
      if extension == ".dat":
        self.simA_isPython = True
        self.simA_isDYNA   = False
      elif extension == ".k":
        self.simA_isPython  = False
        self.simA_isDYNA    = True
    except UnboundLocalError:
      self.simA_isPython = False
      self.simA_isDYNA   = False
      print("--------\nWARNING:\n--------\nInput file not found for Simulation A, analytical solution assumed...")
    #-----------------------------
    # Parameters for Simulation B.
    #-----------------------------
    self.simB_Dir = inputData.m_simB_Path

    if self.simB_Dir is not None:
      if not os.path.exists(self.simB_Dir):
        self.simB_Dir = REPO + '/' + self.simB_Dir
        if not os.path.exists(self.simB_Dir):
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nDirectory for Simulation B not found.")
      self.simB_Title          = inputData.m_simB_Title
      self.simB_Skip           = inputData.m_simB_Skip
      self.simB_SkipSecondary  = inputData.m_simB_SkipSecondary
      self.simB_SkipTertiary   = inputData.m_simB_SkipTertiary
      self.simB_SkipQuaternary = inputData.m_simB_SkipQuaternary
      self.simB_SkipQuinary    = inputData.m_simB_SkipQuinary

      self.simB_probe_1 = inputData.m_simB_probe_1
      self.simB_probe_2 = inputData.m_simB_probe_2
      self.simB_probe_3 = inputData.m_simB_probe_3
      self.simB_probe_4 = inputData.m_simB_probe_4

      self.simB_Linestyle_Alpha   = inputData.m_simB_Linestyle_Alpha
      self.simB_Linestyle_Bravo   = inputData.m_simB_Linestyle_Bravo
      self.simB_Linestyle_Charlie = inputData.m_simB_Linestyle_Charlie
      self.simB_Linestyle_Delta   = inputData.m_simB_Linestyle_Delta
      self.simB_Linestyle_Echo    = inputData.m_simB_Linestyle_Echo
      self.simB_Color_Alpha       = inputData.m_simB_Color_Alpha
      self.simB_Color_Bravo       = inputData.m_simB_Color_Bravo
      self.simB_Color_Charlie     = inputData.m_simB_Color_Charlie
      self.simB_Color_Delta       = inputData.m_simB_Color_Delta
      self.simB_Color_Echo        = inputData.m_simB_Color_Echo
      self.simB_fillstyle         = inputData.m_simB_fillstyle
      #-----------------------------------------
      # Get information from Simulation B setup.
      #-----------------------------------------
      for root, dirnames, filenames in os.walk(self.simB_Dir):
        for file in filenames:
          (shortname, extension) = os.path.splitext(file)
          if extension == ".dat" or extension == '.k':
            self.simB_InputFileName = file
            break

      try:
        if extension == ".dat":
          self.simB_isPython = True
          self.simB_isDYNA   = False
        elif extension == ".k":
          self.simB_isPython  = False
          self.simB_isDYNA    = True
      except UnboundLocalError:
        self.simB_isPython = False
        self.simB_isDYNA   = False
        print("--------\nWARNING:\n--------\nInput file not found for Simulation B, analytical solution assumed...")
    #-----------------------------
    # Parameters for Simulation C.
    #-----------------------------
    self.simC_Dir = inputData.m_simC_Path

    if self.simC_Dir is not None:
      if not os.path.exists(self.simC_Dir):
        self.simC_Dir = REPO + '/' + self.simC_Dir
        if not os.path.exists(self.simC_Dir):
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nDirectory for Simulation C not found.")
      self.simC_Title          = inputData.m_simC_Title
      self.simC_Skip           = inputData.m_simC_Skip
      self.simC_SkipSecondary  = inputData.m_simC_SkipSecondary
      self.simC_SkipTertiary   = inputData.m_simC_SkipTertiary
      self.simC_SkipQuaternary = inputData.m_simC_SkipQuaternary
      self.simC_SkipQuinary    = inputData.m_simC_SkipQuinary

      self.simC_probe_1 = inputData.m_simC_probe_1
      self.simC_probe_2 = inputData.m_simC_probe_2
      self.simC_probe_3 = inputData.m_simC_probe_3
      self.simC_probe_4 = inputData.m_simC_probe_4

      self.simC_Linestyle_Alpha   = inputData.m_simC_Linestyle_Alpha
      self.simC_Linestyle_Bravo   = inputData.m_simC_Linestyle_Bravo
      self.simC_Linestyle_Charlie = inputData.m_simC_Linestyle_Charlie
      self.simC_Linestyle_Delta   = inputData.m_simC_Linestyle_Delta
      self.simC_Linestyle_Echo    = inputData.m_simC_Linestyle_Echo
      self.simC_Color_Alpha       = inputData.m_simC_Color_Alpha
      self.simC_Color_Bravo       = inputData.m_simC_Color_Bravo
      self.simC_Color_Charlie     = inputData.m_simC_Color_Charlie
      self.simC_Color_Delta       = inputData.m_simC_Color_Delta
      self.simC_Color_Echo        = inputData.m_simC_Color_Echo
      self.simC_fillstyle         = inputData.m_simC_fillstyle
      #-----------------------------------------
      # Get information from Simulation C setup.
      #-----------------------------------------
      for root, dirnames, filenames in os.walk(self.simC_Dir):
        for file in filenames:
          (shortname, extension) = os.path.splitext(file)
          if extension == ".dat" or extension == '.k':
            self.simC_InputFileName = file
            break

      try:
        if extension == ".dat":
          self.simC_isPython = True
          self.simC_isDYNA   = False
        elif extension == ".k":
          self.simC_isPython  = False
          self.simC_isDYNA    = True
      except UnboundLocalError:
        self.simC_isPython = False
        self.simC_isDYNA   = False
        print("--------\nWARNING:\n--------\nInput file not found for Simulation C, analytical solution assumed...")

    #-----------------------------
    # Parameters for Simulation D.
    #-----------------------------
    self.simD_Dir = inputData.m_simD_Path

    if self.simD_Dir is not None:
      if not os.path.exists(self.simD_Dir):
        self.simD_Dir = REPO + '/' + self.simD_Dir
        if not os.path.exists(self.simD_Dir):
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nDirectory for Simulation D not found.")
      self.simD_Title          = inputData.m_simD_Title
      self.simD_Skip           = inputData.m_simD_Skip
      self.simD_SkipSecondary  = inputData.m_simD_SkipSecondary
      self.simD_SkipTertiary   = inputData.m_simD_SkipTertiary
      self.simD_SkipQuaternary = inputData.m_simD_SkipQuaternary
      self.simD_SkipQuinary    = inputData.m_simD_SkipQuinary

      self.simD_probe_1 = inputData.m_simD_probe_1
      self.simD_probe_2 = inputData.m_simD_probe_2
      self.simD_probe_3 = inputData.m_simD_probe_3
      self.simD_probe_4 = inputData.m_simD_probe_4

      self.simD_Linestyle_Alpha   = inputData.m_simD_Linestyle_Alpha
      self.simD_Linestyle_Bravo   = inputData.m_simD_Linestyle_Bravo
      self.simD_Linestyle_Charlie = inputData.m_simD_Linestyle_Charlie
      self.simD_Linestyle_Delta   = inputData.m_simD_Linestyle_Delta
      self.simD_Linestyle_Echo    = inputData.m_simD_Linestyle_Echo
      self.simD_Color_Alpha       = inputData.m_simD_Color_Alpha
      self.simD_Color_Bravo       = inputData.m_simD_Color_Bravo
      self.simD_Color_Charlie     = inputData.m_simD_Color_Charlie
      self.simD_Color_Delta       = inputData.m_simD_Color_Delta
      self.simD_Color_Echo        = inputData.m_simD_Color_Echo
      self.simD_fillstyle         = inputData.m_simD_fillstyle
      #-----------------------------------------
      # Get information from Simulation D setup.
      #-----------------------------------------
      for root, dirnames, filenames in os.walk(self.simD_Dir):
        for file in filenames:
          (shortname, extension) = os.path.splitext(file)
          if extension == ".dat" or extension == '.k':
            self.simD_InputFileName = file
            break
      try:
        if extension == ".dat":
          self.simD_isPython = True
          self.simD_isDYNA   = False
        elif extension == ".k":
          self.simD_isPython  = False
          self.simD_isDYNA    = True
      except UnboundLocalError:
        self.simD_isPython = False
        self.simD_isDYNA   = False
        print("--------\nWARNING:\n--------\nInput file not found for Simulation D, analytical solution assumed...")

    #-----------------------------
    # Parameters for Simulation E.
    #-----------------------------
    self.simE_Dir = inputData.m_simE_Path

    if self.simE_Dir is not None:
      if not os.path.exists(self.simE_Dir):
        self.simE_Dir = REPO + '/' + self.simE_Dir
        if not os.path.exists(self.simE_Dir):
          sys.exit("-----------------\nINPUT FILE ERROR:\n-----------------\nDirectory for Simulation E not found.")
      self.simE_Title          = inputData.m_simE_Title
      self.simE_Skip           = inputData.m_simE_Skip
      self.simE_SkipSecondary  = inputData.m_simE_SkipSecondary
      self.simE_SkipTertiary   = inputData.m_simE_SkipTertiary
      self.simE_SkipQuaternary = inputData.m_simE_SkipQuaternary
      self.simE_SkipQuinary    = inputData.m_simE_SkipQuinary

      self.simE_probe_1 = inputData.m_simE_probe_1
      self.simE_probe_2 = inputData.m_simE_probe_2
      self.simE_probe_3 = inputData.m_simE_probe_3
      self.simE_probe_4 = inputData.m_simE_probe_4

      self.simE_Linestyle_Alpha   = inputData.m_simE_Linestyle_Alpha
      self.simE_Linestyle_Bravo   = inputData.m_simE_Linestyle_Bravo
      self.simE_Linestyle_Charlie = inputData.m_simE_Linestyle_Charlie
      self.simE_Linestyle_Delta   = inputData.m_simE_Linestyle_Delta
      self.simE_Linestyle_Echo    = inputData.m_simE_Linestyle_Echo
      self.simE_Color_Alpha       = inputData.m_simE_Color_Alpha
      self.simE_Color_Bravo       = inputData.m_simE_Color_Bravo
      self.simE_Color_Charlie     = inputData.m_simE_Color_Charlie
      self.simE_Color_Delta       = inputData.m_simE_Color_Delta
      self.simE_Color_Echo        = inputData.m_simE_Color_Echo
      self.simE_fillstyle         = inputData.m_simE_fillstyle
      #-----------------------------------------
      # Get information from Simulation E setup.
      #-----------------------------------------
      for root, dirnames, filenames in os.walk(self.simE_Dir):
        for file in filenames:
          (shortname, extension) = os.path.splitext(file)
          if extension == ".dat" or extension == '.k':
            self.simE_InputFileName = file
            break
      try:
        if extension == ".dat":
          self.simE_isPython = True
          self.simE_isDYNA   = False
        elif extension == ".k":
          self.simE_isPython  = False
          self.simE_isDYNA    = True
      except UnboundLocalError:
        self.simE_isPython = False
        self.simE_isDYNA   = False
        print("--------\nWARNING:\n--------\nInput file not found for Simulation E, analytical solution assumed...")

#-------------
# Main script.
#-------------
if __name__ == '__main__':

  np.seterr(all='raise')
  #-----------------
  # Set LaTeX fonts.
  #-----------------
  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsthm,amsfonts,amssymb,amscd,mathtools,xcolor} \input{' + REPO + '/scripts/macros.tex}')
  #------------------------------------
  # Create scaling factor dictionaries.
  #------------------------------------
  timeFactorDict   = {1e6 : r'($\mu$s)', 1e3 : '(ms)', 1 : '(s)', 1/60 : '(min)'}
  dispFactorDict   = {1e6 : r'($\mu$m)', 1e3 : '(mm)' , 1e2 : '(cm)' , 1 : '(m)'}
  stressFactorDict = {1e-6 : '(MPa)', 1e-3 : '(kPa)', 1 : '(Pa)', 1e3 : '(mPa)', 1e6 : r'$(\mu$Pa)'}
  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to generate plots of 1-D Lagrangian FE simulations.\
                                                Specifically, depending on the name of the input file, it will call\
                                                the appropriate plotting script and generate stills at user-defined\
                                                probing locations along the 1-D mesh, or animations for transient\
                                                solutions.')
  parser.add_argument('inputFile', metavar='i', type=str,
                      help='the file path to the input file')
  parser.add_argument('-a', '--average', action='store_true',
                      help='average Gauss point data over the element')
  parser.add_argument('-n', '--no_label', action='store_true',
                      help='disables axis labels')
  parser.add_argument('-T', '--total', action='store_true',
                      help='plot mixture stress or pressure data (if applicable)')
  parser.add_argument('-S', '--solid', action='store_true',
                       help='plot solid skeleton data (if applicable)')
  parser.add_argument('-F', '--fluid', action='store_true',
                      help='plot pore fluid data (if applicable)')
  parser.add_argument('-V', '--viscous', action='store_true',
                      help='plot pore fluid viscous stress (if applicable)')
  parser.add_argument('-P', '--pressure', action='store_true',
                      help='plot pore fluid pressure stress (if applicable)')
  parser.add_argument('-pa', '--partial', action='store_true',
                      help='plot partial pore fluid pressure (if applicable)')
  parser.add_argument('-tr', '--transverse', action='store_true',
                      help='plot stress in the transverse direction (if applicable)')
  parser.add_argument('-j', '--jacobian', action='store_true',
                      help='plot the Jacobian of deformation (if applicable)')
  parser.add_argument('-t', '--text', action='store_true',
                      help='add text boxes for analytical solutions')
  parser.add_argument('-l', '--log', action='store_true',
                      help='sets yscale to logarithmic')
  parser.add_argument('-scale', type=float, default=1,
                      help='modulus to scale pressure results by')
  parser.add_argument('-adjust', type=float, default=0,
                      help='initial pressure to subtract from pore fluid pressure results')
  parser.add_argument('-start', type=int, default=0,
                      help='starting time index for animations')
  parser.add_argument('-stop', type=int, default=10,
                      help='stopping time index for animations')
  parser.add_argument('-na', '--no_animation', action='store_true',
                      help='disable .mp4 and .gif generation for animations')
  parser.add_argument('-mmst', '--mms_temporal', action='store_true',
                      help='plot temporal errors for MMS')
  parser.add_argument('-mmss', '--mms_spatial', action='store_true',
                      help='plot spatial errors for MMS')
  parser.add_argument('-mmsso', '--mms_solution', action='store_true',
                      help='plot solutions for MMS')
  parser.add_argument('-mmsn', '--mms_norm', choices={'2', 'inf'}, default='2',
                      help='specify the type of error norm for MMS temporal or spatial error plots')
  parser.add_argument('-mmsid', '--mms_time_index', type=int, default=-1,
                      help='time index for MMS plots')
  parser.add_argument('-mmssuf', '--mms_suffix', type=str,
                      help='L#-suffix for MMS data directory')
  parser.add_argument('-g', '--gauss', action='store_true',
                      help='flag to plot the Gauss point temperature data')
  parser.add_argument('-gr', type=float, default=4.91e-4,
                      help='value of Gruneisen parameter for analytical solution to temperature')

  args = parser.parse_args()
  #-----------------
  # Read input file.
  #-----------------
  inputData = plotInput.PlotInputs(args.inputFile)
  inputData.readInputFile()

  averageGauss = args.average
  #------------------------------------------------
  # Initialize plotting parameters from input file.
  #------------------------------------------------
  plotParams = Parameters(inputData, averageGauss)
  #-------------------------------------------
  # Add plotting parameters from command line.
  #-------------------------------------------
  plotParams.no_labels    = args.no_label
  plotParams.totalPlot    = args.total
  plotParams.solidPlot    = args.solid
  plotParams.fluidPlot    = args.fluid
  plotParams.viscousPlot  = args.viscous
  plotParams.pressurePlot = args.pressure
  plotParams.partialPlot  = args.partial
  plotParams.jacobianPlot = args.jacobian
  plotParams.text         = args.text
  plotParams.log          = args.log
  plotParams.scale        = args.scale
  plotParams.adjust       = args.adjust
  plotParams.startID      = args.start
  plotParams.stopID       = args.stop
  plotParams.noAnimation  = args.no_animation
  plotParams.temporal     = args.mms_temporal
  plotParams.spatial      = args.mms_spatial
  plotParams.solution     = args.mms_solution
  plotParams.normOrd      = args.mms_norm
  plotParams.mms_time_id  = args.mms_time_index
  plotParams.mms_suffix   = args.mms_suffix
  #-------------------------
  # Create output directory.
  #-------------------------
  if not os.path.exists(plotParams.outputDir):
    os.makedirs(plotParams.outputDir)
  #--------------------------------------
  # Copy input files to output directory.
  #--------------------------------------
  inputFileName = args.inputFile.split('/')[-1]
  shutil.copy(args.inputFile, plotParams.outputDir + inputFileName)

  for root, dirnames, filenames in os.walk(plotParams.simA_Dir):
    for file in filenames:
      (shortname, extension) = os.path.splitext(file)
      if extension == ".dat" or extension == '.k':
        simA_InputFileName = file
        shutil.copy(root + simA_InputFileName, plotParams.outputDir + 'simA_Input')
    break

  if plotParams.simB_Dir is not None:
    for root, dirnames, filenames in os.walk(plotParams.simB_Dir):
      for file in filenames:
        (shortname, extension) = os.path.splitext(file)
        if extension == ".dat" or extension == '.k':
          simB_InputFileName = file
          shutil.copy(root + simB_InputFileName, plotParams.outputDir + 'simB_Input')
      break

  if plotParams.simC_Dir is not None:
    for root, dirnames, filenames in os.walk(plotParams.simC_Dir):
      for file in filenames:
        (shortname, extension) = os.path.splitext(file)
        if extension == ".dat" or extension == '.k':
          simC_InputFileName = file
          shutil.copy(root + simC_InputFileName, plotParams.outputDir + 'simC_Input')
      break

  if plotParams.simD_Dir is not None:
    for root, dirnames, filenames in os.walk(plotParams.simD_Dir):
      for file in filenames:
        (shortname, extension) = os.path.splitext(file)
        if extension == ".dat" or extension == '.k':
          simD_InputFileName = file
          shutil.copy(root + simD_InputFileName, plotParams.outputDir + 'simD_Input')
      break

  if plotParams.simE_Dir is not None:
    for root, dirnames, filenames in os.walk(plotParams.simE_Dir):
      for file in filenames:
        (shortname, extension) = os.path.splitext(file)
        if extension == ".dat" or extension == '.k':
          simE_InputFileName = file
          shutil.copy(root + simE_InputFileName, plotParams.outputDir + 'simE_Input')
      break
  #--------------
  # Create plots.
  #--------------
  if 'Li' in args.inputFile:
    LiPlots.makeLiPlots(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'eringen-input.dat' in args.inputFile:
    EringenPlots.makeEringenPlots(timeFactorDict, dispFactorDict, plotParams)

  elif 'deBoer-input.dat' in args.inputFile:
    deBoerPlots.makeDeBoerPlots(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'displacement-input.dat' in args.inputFile:
    plotDisplacement.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'velocity-input.dat' in args.inputFile:
    plotVelocity.main(timeFactorDict, plotParams)

  elif 'velocityError-input.dat' in args.inputFile:
    plotVelocityError.main(timeFactorDict, plotParams)

  elif 'acceleration-input.dat' in args.inputFile:
    plotAcceleration.main(timeFactorDict, plotParams)

  elif 'volumeFraction-input.dat' in args.inputFile:
    plotVolumeFraction.main(timeFactorDict, plotParams)

  elif 'pressure-input.dat' in args.inputFile:
    plotPressure.main(timeFactorDict, stressFactorDict, plotParams)

  elif 'pi-input.dat' in args.inputFile:
    plotPi.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'stress-input.dat' in args.inputFile:
    if args.transverse:
      plotStress22.main(timeFactorDict, stressFactorDict, plotParams)
    else:
      plotStress.main(timeFactorDict, stressFactorDict, plotParams)

  elif 'shear-input.dat' in args.inputFile:
    plotShear.main(plotParams)

  elif 'vonMises-input.dat' in args.inputFile:
    plotVonMises.main(timeFactorDict, stressFactorDict, plotParams)

  elif 'J-input.dat' in args.inputFile:
    plotJ.main(timeFactorDict, plotParams)

  elif 'time-input.dat' in args.inputFile:
    plotTimeSteps.main(timeFactorDict, plotParams)

  elif 'temperature-input.dat' in args.inputFile:
    if args.gauss:
      plotTemperatureGauss.main(timeFactorDict, plotParams)
    else:
      plotTemperature.main(timeFactorDict, plotParams)

  elif 'adiabaticAnalytical-input.dat' in args.inputFile:
    plotParams.G0 = args.gr
    plotAdiabaticAnalytical.main(timeFactorDict, plotParams)

  elif 'mmsError-input.dat' in args.inputFile:
    plotMMSError.main(plotParams, dispFactorDict, stressFactorDict)

#  elif 'error-input.dat' in args.inputFile: # for de Boer, deprecated
#    plotError.main(plotParams)

  # Deprecated plots to plot wave speed against density
  # elif 'densityCFL-input.dat' in args.inputFile:
  #   if '--single' in sys.argv[2]:
  #     plotParams.singlephase_C        = True   # Wave speed calculated from solid lung tissue K
  #     plotParams.multiphase_naive_C   = False  # Wave speed calculated from skeleton          K
  #     plotParams.multiphase_complex_C = False  # Wave speed calculated from Ks & Kf
  #   elif '--naive' in sys.argv[2]:
  #     plotParams.singlephase_C        = False
  #     plotParams.multiphase_naive_C   = True
  #     plotParams.multiphase_complex_C = False
  #   elif '--complex' in sys.argv[2]:
  #     plotParams.singlephase_C        = False
  #     plotParams.multiphase_naive_C   = False
  #     plotParams.multiphase_complex_C = True
  #   plotDensityCFL.main(timeFactorDict, plotParams)

  # elif 'CFL-input.dat' in args.inputFile:
  #   if '--single' in sys.argv[2]:
  #     plotParams.singlephase_C        = True
  #     plotParams.multiphase_naive_C   = False
  #     plotParams.multiphase_complex_C = False
  #   elif '--naive' in sys.argv[2]:
  #     plotParams.singlephase_C        = False
  #     plotParams.multiphase_naive_C   = True
  #     plotParams.multiphase_complex_C = False
  #   elif '--complex' in sys.argv[2]:
  #     plotParams.singlephase_C        = False
  #     plotParams.multiphase_naive_C   = False
  #     plotParams.multiphase_complex_C = True
  #   plotCFL.main(timeFactorDict, plotParams)

  elif 'piContour-input.dat' in args.inputFile:
    animatePiContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)
 
  elif 'pressureContour-input.dat' in args.inputFile:
    animatePressureContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'jacobianContour-input.dat' in args.inputFile:
    animateJacobianContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'JDotContour-input.dat' in args.inputFile:
    animateJDotContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'volumeFractionContour-input.dat' in args.inputFile:
    animateVolumeFractionContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'darcyContour-input.dat' in args.inputFile:
    animateDarcyContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'displacementContour-input.dat' in args.inputFile:
    animateDisplacementContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'velocityContour-input.dat' in args.inputFile:
    animateVelocityContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'viscousFluidStressContour-input.dat' in args.inputFile:
    animateViscousFluidStressContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'stressContour-input.dat' in args.inputFile:
    animateStressContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'varkappaContour-input.dat' in args.inputFile:
    animateVarkappaContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'temperatureContour-input.dat' in args.inputFile:
    animateTemperatureContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'mixtureTemperatureContour-input.dat' in args.inputFile:
    animateMixtureTemperatureContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'conductionContour-input.dat' in args.inputFile:
    animateConductionContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'pressureTemperatureContour-input.dat' in args.inputFile:
    animatePressureTemperatureContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'velocityTemperatureContour-input.dat' in args.inputFile:
    animateVelocityTemperatureContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'volumePressureContour-input.dat' in args.inputFile:
    animateVolumePressureContour.main(timeFactorDict, dispFactorDict, stressFactorDict, plotParams)

  elif 'volumeTemperatureContour-input.dat' in args.inputFile:
    animateVolumeTemperatureContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'PVTContour-input.dat' in args.inputFile:
    animatePVTContour.main(timeFactorDict, dispFactorDict, plotParams)

  elif 'PJTContour-input.dat' in args.inputFile:
    animatePJTContour.main(timeFactorDict, dispFactorDict, plotParams)
  
  elif 'KContour-input.dat' in args.inputFile:
    animateKContour.main(timeFactorDict, dispFactorDict, plotParams)

  if 'Contour' in args.inputFile:
    print("\nPlots generated successfully.")
  else:
    print("Plots generated successfully.")

