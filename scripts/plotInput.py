#--------------------------------------------------------------------------------------------------
# Module providing encapsulated input data handling capabilities to plot 1-D uniaxial strain 
# Lagrangian finite-element simulations.
#
# Author:        Zachariah Irwin
# Institution:   University of Colorado Boulder
# Last Edits:    June 6, 2024
#--------------------------------------------------------------------------------------------------
import sys, os

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  REPO = os.environ['REPO']
except KeyError:
  sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")

#---------------------------------------------------------------------------------
# Problem input data structure, with all input data handled in a protected manner.
#---------------------------------------------------------------------------------
class PlotInputs:
    
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

    return

  #------------------------------------------------------------------
  # Read a formated ASCII input file and populate member data fields.
  #------------------------------------------------------------------
  def readInputFile(self):
      
    inputFileObj = open(self.m_InputFile)
    
    for line in inputFileObj:
        
      if not line.startswith('#'):
          
        lineDict = line.split('=')

        if lineDict[0].strip() == 'Project directory':
          tempRoot = lineDict[1].strip()
          if tempRoot == 'default': self.m_RootPath = os.getcwd()
          if tempRoot == 'testing': self.m_RootPath = os.environ['REPO']
          if tempRoot == 'None': self.m_RootPath = ''

        elif lineDict[0].strip() == 'Output directory':
          self.m_plotPath = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Name':
          self.m_filename = lineDict[1].strip()

        elif lineDict[0].strip() == 'Animations':
          tempList           = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_deletePNGs = True
          else:
            self.m_deletePNGs = False
          self.m_framerate = int(tempList[6])

        elif lineDict[0].strip() == 'Scaling':
          tempList                   = lineDict[1].strip().split()
          self.m_timeScaling         = float(tempList[2])
          self.m_displacementScaling = float(tempList[6])
          self.m_stressScaling       = float(tempList[10])

        elif lineDict[0].strip() == 'Plot x-ticks':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_is_xticks   = True
            self.m_xticks      = []
            for i in tempList[6].split(',') : self.m_xticks.append (float(i))
            if tempList[10] == 'None' or tempList[10] == 'none':
              self.m_is_xticklabels = False
            else:
              self.m_is_xticklabels = True
              self.m_xticklabels = list(tempList[10].split(','))
          else:
            self.m_is_xticks = False
            self.m_is_xticklabels = False
          if tempList[14] == 'True' or tempList[14] == 'true':
            self.m_secondaryXTicks = True
          else:
            self.m_secondaryXTicks = False

        elif lineDict[0].strip() == 'Plot y-ticks':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_is_yticks   = True
            self.m_yticks      = []
            for i in tempList[6].split(',') : self.m_yticks.append (float(i))
            if tempList[10] == 'None' or tempList[10] == 'none':
              self.m_is_yticklabels = False
            else:
              self.m_is_yticklabels = True
              self.m_yticklabels = list(tempList[10].split(','))
          else:
            self.m_is_yticks = False
            self.m_is_yticklabels = False
          if tempList[14] == 'True' or tempList[14] == 'true':
            self.m_secondaryYTicks = True
          else:
            self.m_secondaryYTicks = False

        elif lineDict[0].strip() == 'Plot legend':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_legend         = True
            self.m_handleLength   = float(tempList[6])
            self.m_legendX        = float(tempList[10])
            self.m_legendY        = float(tempList[14])
            self.m_legendPosition = ' '.join(tempList[18:])
          else:
            self.m_legend = False

        elif lineDict[0].strip() == 'Plot grid':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_grid = True
            self.m_gridWhich = tempList[6]
          else:
            self.m_grid = False

        elif lineDict[0].strip() == 'Plot limits':
          tempList     = lineDict[1].strip().split()
          try:
            self.m_xlim0 = float(tempList[2])
          except ValueError:
            self.m_xlim0 = None
          try:
            self.m_xlim1 = float(tempList[6])
          except ValueError:
            self.m_xlim1 = None
          try:
            self.m_ylim0 = float(tempList[10])
          except ValueError:
            try:
              self.m_ylim00 = float(tempList[10].split(',')[0])
              self.m_ylim01 = float(tempList[10].split(',')[1])
              try:
                self.m_ylim02 = float(tempList[10].split(',')[2])
              except IndexError:
                self.m_ylim02 = None
            except ValueError:
              self.m_ylim0 = None
          try:
            self.m_ylim1 = float(tempList[14])
          except ValueError:
            try:
              self.m_ylim10 = float(tempList[14].split(',')[0])
              self.m_ylim11 = float(tempList[14].split(',')[1])
              try:
                self.m_ylim12 = float(tempList[14].split(',')[2])
              except IndexError:
                self.m_ylim12 = None
            except ValueError:
              self.m_ylim1 = None

        elif lineDict[0].strip() == 'Plot title':
          tempList = lineDict[1].strip().split()
          if tempList[2] == 'True' or tempList[2] == 'true':
            self.m_title = True
            self.m_titleLoc = float(tempList[6])
            self.m_titleName = ' '.join(tempList[10:])
          else:
            self.m_title = False

        elif lineDict[0].strip() == 'Plot DPI':
          self.m_DPI = float(lineDict[1].strip())

        elif lineDict[0].strip() == 'Plot font sizes':
          tempList              = lineDict[1].strip().split()
          self.m_titleFontSize  = float(tempList[2])
          self.m_xAxisFontSize  = float(tempList[6])
          self.m_yAxisFontSize  = float(tempList[10])
          self.m_legendFontSize = float(tempList[14])

        elif lineDict[0].strip() == 'Simulation A directory':
          self.m_simA_Path = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation B directory':
          if lineDict[1].strip()  == 'None':
            self.m_simB_Path = None
          else:
            self.m_simB_Path = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation C directory':
          if lineDict[1].strip()  == 'None':
            self.m_simC_Path = None
          else:
            self.m_simC_Path = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation D directory':
          if lineDict[1].strip()  == 'None':
            self.m_simD_Path = None
          else:
            self.m_simD_Path = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation E directory':
          if lineDict[1].strip()  == 'None':
            self.m_simE_Path = None
          else:
            self.m_simE_Path = self.m_RootPath + lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation A Skip':
          self.m_simA_Skip            = int(lineDict[1].strip().split()[0])
          self.m_simA_SkipSecondary   = int(lineDict[1].strip().split()[2])
          self.m_simA_SkipTertiary    = int(lineDict[1].strip().split()[4])
          self.m_simA_SkipQuaternary  = int(lineDict[1].strip().split()[6])
          self.m_simA_SkipQuinary     = int(lineDict[1].strip().split()[8])

        elif lineDict[0].strip() == 'Simulation B Skip':
          self.m_simB_Skip            = int(lineDict[1].strip().split()[0])
          self.m_simB_SkipSecondary   = int(lineDict[1].strip().split()[2])
          self.m_simB_SkipTertiary    = int(lineDict[1].strip().split()[4])
          self.m_simB_SkipQuaternary  = int(lineDict[1].strip().split()[6])
          self.m_simB_SkipQuinary     = int(lineDict[1].strip().split()[8])

        elif lineDict[0].strip() == 'Simulation C Skip':
          self.m_simC_Skip            = int(lineDict[1].strip().split()[0])
          self.m_simC_SkipSecondary   = int(lineDict[1].strip().split()[2])
          self.m_simC_SkipTertiary    = int(lineDict[1].strip().split()[4])
          self.m_simC_SkipQuaternary  = int(lineDict[1].strip().split()[6])
          self.m_simC_SkipQuinary     = int(lineDict[1].strip().split()[8])

        elif lineDict[0].strip() == 'Simulation D Skip':
          self.m_simD_Skip            = int(lineDict[1].strip().split()[0])
          self.m_simD_SkipSecondary   = int(lineDict[1].strip().split()[2])
          self.m_simD_SkipTertiary    = int(lineDict[1].strip().split()[4])
          self.m_simD_SkipQuaternary  = int(lineDict[1].strip().split()[6])
          self.m_simD_SkipQuinary     = int(lineDict[1].strip().split()[8])

        elif lineDict[0].strip() == 'Simulation E Skip':
          self.m_simE_Skip            = int(lineDict[1].strip().split()[0])
          self.m_simE_SkipSecondary   = int(lineDict[1].strip().split()[2])
          self.m_simE_SkipTertiary    = int(lineDict[1].strip().split()[4])
          self.m_simE_SkipQuaternary  = int(lineDict[1].strip().split()[6])
          self.m_simE_SkipQuinary     = int(lineDict[1].strip().split()[8])

        elif lineDict[0].strip() == 'Simulation A title':
          self.m_simA_Title = lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation B title':
          self.m_simB_Title = lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation C title':
          self.m_simC_Title = lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation D title':
          self.m_simD_Title = lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation E title':
          self.m_simE_Title = lineDict[1].strip()

        elif lineDict[0].strip() == 'Simulation A probes':
          tempList            = lineDict[1].strip().split()
          if tempList[4] == 'None' or tempList[4] == 'none':
            self.m_simA_probe_1 = None
          else:
            self.m_simA_probe_1 = float(tempList[4])
          if tempList[10] == 'None' or tempList[10] == 'none':
            self.m_simA_probe_2 = None
          else:
            self.m_simA_probe_2 = float(tempList[10])
          if tempList[16] == 'None' or tempList[16] == 'none':
            self.m_simA_probe_3 = None
          else:
            self.m_simA_probe_3 = float(tempList[16])
          if tempList[22] == 'None' or tempList[22] == 'none':
            self.m_simA_probe_4 = None
          else:
            self.m_simA_probe_4 = float(tempList[22])

        elif lineDict[0].strip() == 'Simulation B probes':
          tempList            = lineDict[1].strip().split()
          if tempList[4] == 'None' or tempList[4] == 'none':
            self.m_simB_probe_1 = None
          else:
            self.m_simB_probe_1 = float(tempList[4])
          if tempList[10] == 'None' or tempList[10] == 'none':
            self.m_simB_probe_2 = None
          else:
            self.m_simB_probe_2 = float(tempList[10])
          if tempList[16] == 'None' or tempList[16] == 'none':
            self.m_simB_probe_3 = None
          else:
            self.m_simB_probe_3 = float(tempList[16])
          if tempList[22] == 'None' or tempList[22] == 'none':
            self.m_simB_probe_4 = None
          else:
            self.m_simB_probe_4 = float(tempList[22])

        elif lineDict[0].strip() == 'Simulation C probes':
          tempList            = lineDict[1].strip().split()
          if tempList[4] == 'None' or tempList[4] == 'none':
            self.m_simC_probe_1 = None
          else:
            self.m_simC_probe_1 = float(tempList[4])
          if tempList[10] == 'None' or tempList[10] == 'none':
            self.m_simC_probe_2 = None
          else:
            self.m_simC_probe_2 = float(tempList[10])
          if tempList[16] == 'None' or tempList[16] == 'none':
            self.m_simC_probe_3 = None
          else:
            self.m_simC_probe_3 = float(tempList[16])
          if tempList[22] == 'None' or tempList[22] == 'none':
            self.m_simC_probe_4 = None
          else:
            self.m_simC_probe_4 = float(tempList[22])

        elif lineDict[0].strip() == 'Simulation D probes':
          tempList            = lineDict[1].strip().split()
          if tempList[4] == 'None' or tempList[4] == 'none':
            self.m_simD_probe_1 = None
          else:
            self.m_simD_probe_1 = float(tempList[4])
          if tempList[10] == 'None' or tempList[10] == 'none':
            self.m_simD_probe_2 = None
          else:
            self.m_simD_probe_2 = float(tempList[10])
          if tempList[16] == 'None' or tempList[16] == 'none':
            self.m_simD_probe_3 = None
          else:
            self.m_simD_probe_3 = float(tempList[16])
          if tempList[22] == 'None' or tempList[22] == 'none':
            self.m_simD_probe_4 = None
          else:
            self.m_simD_probe_4 = float(tempList[22])

        elif lineDict[0].strip() == 'Simulation E probes':
          tempList            = lineDict[1].strip().split()
          if tempList[4] == 'None' or tempList[4] == 'none':
            self.m_simE_probe_1 = None
          else:
            self.m_simE_probe_1 = float(tempList[4])
          if tempList[10] == 'None' or tempList[10] == 'none':
            self.m_simE_probe_2 = None
          else:
            self.m_simE_probe_2 = float(tempList[10])
          if tempList[16] == 'None' or tempList[16] == 'none':
            self.m_simE_probe_3 = None
          else:
            self.m_simE_probe_3 = float(tempList[16])
          if tempList[22] == 'None' or tempList[22] == 'none':
            self.m_simE_probe_4 = None
          else:
            self.m_simE_probe_4 = float(tempList[22])

        elif lineDict[0].strip() == 'Simulation A lines':
          tempList                      = lineDict[1].strip().split()
          self.m_simA_Linestyle_Alpha   = tempList[2]
          self.m_simA_Linestyle_Bravo   = tempList[6]
          self.m_simA_Linestyle_Charlie = tempList[10]
          self.m_simA_Linestyle_Delta   = tempList[14]
          self.m_simA_Linestyle_Echo    = tempList[18]
          self.m_simA_Color_Alpha       = tempList[22]
          self.m_simA_Color_Bravo       = tempList[26]
          self.m_simA_Color_Charlie     = tempList[30]
          self.m_simA_Color_Delta       = tempList[34]
          self.m_simA_Color_Echo        = tempList[38]
          self.m_simA_fillstyle         = tempList[42]

        elif lineDict[0].strip() == 'Simulation B lines':
          tempList                      = lineDict[1].strip().split()
          self.m_simB_Linestyle_Alpha   = tempList[2]
          self.m_simB_Linestyle_Bravo   = tempList[6]
          self.m_simB_Linestyle_Charlie = tempList[10]
          self.m_simB_Linestyle_Delta   = tempList[14]
          self.m_simB_Linestyle_Echo    = tempList[18]
          self.m_simB_Color_Alpha       = tempList[22]
          self.m_simB_Color_Bravo       = tempList[26]
          self.m_simB_Color_Charlie     = tempList[30]
          self.m_simB_Color_Delta       = tempList[34]
          self.m_simB_Color_Echo        = tempList[38]
          self.m_simB_fillstyle         = tempList[42]

        elif lineDict[0].strip() == 'Simulation C lines':
          tempList                      = lineDict[1].strip().split()
          self.m_simC_Linestyle_Alpha   = tempList[2]
          self.m_simC_Linestyle_Bravo   = tempList[6]
          self.m_simC_Linestyle_Charlie = tempList[10]
          self.m_simC_Linestyle_Delta   = tempList[14]
          self.m_simC_Linestyle_Echo    = tempList[18]
          self.m_simC_Color_Alpha       = tempList[22]
          self.m_simC_Color_Bravo       = tempList[26]
          self.m_simC_Color_Charlie     = tempList[30]
          self.m_simC_Color_Delta       = tempList[34]
          self.m_simC_Color_Echo        = tempList[38]
          self.m_simC_fillstyle         = tempList[42]

        elif lineDict[0].strip() == 'Simulation D lines':
          tempList                      = lineDict[1].strip().split()
          self.m_simD_Linestyle_Alpha   = tempList[2]
          self.m_simD_Linestyle_Bravo   = tempList[6]
          self.m_simD_Linestyle_Charlie = tempList[10]
          self.m_simD_Linestyle_Delta   = tempList[14]
          self.m_simD_Linestyle_Echo    = tempList[18]
          self.m_simD_Color_Alpha       = tempList[22]
          self.m_simD_Color_Bravo       = tempList[26]
          self.m_simD_Color_Charlie     = tempList[30]
          self.m_simD_Color_Delta       = tempList[34]
          self.m_simD_Color_Echo        = tempList[38]
          self.m_simD_fillstyle         = tempList[42]

        elif lineDict[0].strip() == 'Simulation E lines':
          tempList                      = lineDict[1].strip().split()
          self.m_simE_Linestyle_Alpha   = tempList[2]
          self.m_simE_Linestyle_Bravo   = tempList[6]
          self.m_simE_Linestyle_Charlie = tempList[10]
          self.m_simE_Linestyle_Delta   = tempList[14]
          self.m_simE_Linestyle_Echo    = tempList[18]
          self.m_simE_Color_Alpha       = tempList[22]
          self.m_simE_Color_Bravo       = tempList[26]
          self.m_simE_Color_Charlie     = tempList[30]
          self.m_simE_Color_Delta       = tempList[34]
          self.m_simE_Color_Echo        = tempList[38]
          self.m_simE_fillstyle         = tempList[42]

    inputFileObj.close()

    return
