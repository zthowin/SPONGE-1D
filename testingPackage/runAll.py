#!/usr/bin/python3
import os, sys, argparse, yaml, time, subprocess, glob

def run_all():
  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to run multiple test cases.')
  parser.add_argument('case', metavar='c', type=str, default='NULL', choices={'MMS', 'deBoer', 'Shock', 'all'},
                      help='the name of the suite of test cases')
  parser.add_argument('--print', action='store_true',
                      help='flag to execute print statements')
  parser.add_argument('--build', action='store_true',
                      help='flag to build LaTeX report')
  parser.add_argument('--plotOnly', action='store_true',
                      help='flag to plot saved test case data and not re-run test cases (requires data exist)')
  args = parser.parse_args()

  print("-------------------------\nRUNNING " + str.upper(args.case) + " TEST CASES.\n-------------------------\n")
  #-------------------------------------------------------
  # Provide warning to user for lengthy de Boer run times.
  #-------------------------------------------------------
  if args.case == 'deBoer' or args.case == 'all':
    print("-------------------------------------\nWARNING: de Boer test cases are slow.\n-------------------------------------\n")
    proceed = input("Continue? [y/n]\n")
    if proceed == 'n':
      exit(0)
  #-----------------------------
  # Check environment variables.
  #-----------------------------
  try:
    REPO = os.environ['REPO']
  except KeyError:
    sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")
  #------------------------
  # Save original location.
  #------------------------
  OriginalLocation = os.getcwd()
  #----------------------------------------
  # Set location of the run_case.py script.
  #----------------------------------------
  RunScript = REPO + '/testingPackage/runCase.py'
  TestFile  = REPO + '/testingPackage/all.yml' 
  #----------------------
  # Create list of tests.
  #----------------------
  stream = open(TestFile, 'r')
  try:
    yamlDic = yaml.load(stream,Loader=yaml.FullLoader)
  except AttributeError:
    yamlDic = yaml.load(stream)
  stream.close()
  #-------------------------------------
  # Loop over tests, executing each one.
  #-------------------------------------
  try:
    rm_cmd = ['rm', '-f', 'TestResult_summary']
    subprocess.run(rm_cmd, check=True, capture_output=True)
  except subprocess.CalledProcessError:
    sys.exit("\nERROR. Could not remove the test result summary file.")

  if not args.plotOnly:
    for C in yamlDic[args.case]:
      print('\n(o) Executing test: ' + C)
      try:
        if args.print:
          run_cmd = [RunScript, C, '--print']
        else:
          run_cmd = [RunScript, C]
        subprocess.run(run_cmd, check=True, capture_output=False)
      except subprocess.CalledProcessError:
        sys.exit("\nERROR. Could not run the individual runCase script.")
  #--------------------------
  # Make plots for .tex file.
  #--------------------------
  if args.case == 'all':
    plot_input = ['MMS/mmsError-input.dat','deBoer/deBoer-input.dat']
    options    = [['-mmst','-mmsn','2'],['--text'],['-S','-F']]
  elif args.case == 'MMS':
    plot_input = ['MMS/mmsError-input.dat']
    options    = [['-mmst','-mmsn','2']]
  elif args.case == 'deBoer':
    plot_input = ['deBoer/deBoer-input.dat']
    options    = [['--text']]
    
  print()

  for i in range(0,len(plot_input)):
    print("\nPreparing to generate plots for case " + plot_input[i].split('/')[0] + '...')
    try:
      plot_cmd = ['python3', REPO + '/scripts/customPlots.py', REPO + '/testingPackage/Cases/' + plot_input[i], *options[i]]
      subprocess.run(plot_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError:
      sys.exit("\nERROR. Could not generate plots for the test cases.")
  
    if 'MMS' in plot_input[i]:
      #-----------------------------------------------------
      # Edit the plot input file for the other MMS solution.
      #-----------------------------------------------------
      f         = open(REPO + '/testingPackage/Cases/' + plot_input[i], 'r')
      lines     = f.readlines()
      lines[4]  = "Name = uufpf_t_L2.pdf\n"
      lines[19] = "Plot limits = XLIM_0 : 1e-1 ; XLIM_1 : 1e-7 ; YLIM_0 : 1e-15 ; YLIM_1 : 1e-1\n"
      lines[21] = "Plot title = ENABLED : True ; LOCATION : 1.0 ; TITLE : Temporal convergence of Q2-Q2-P1 elements ($t \in [0, 10^{-2}]$ s)\n"
      lines[27] = "Simulation A directory = /testingPackage/Cases/MMS/uufpf/\n"
      f         = open(REPO + '/testingPackage/Cases/' + plot_input[i], 'w')
      f.writelines(lines)
      f.close()
      #-----------------------------
      # Plot the other MMS solution.
      #-----------------------------
      try:
        print()
        subprocess.run(plot_cmd, check=True, capture_output=False)
      except subprocess.CalledProcessError:
        sys.exit("\nERROR. Could not generate plots for the test cases.")
      #--------------------------------------
      # Restore the original plot input file.
      #--------------------------------------
      f = open(REPO + '/testingPackage/Cases/' + plot_input[i], 'r')
      lines     = f.readlines()
      lines[4]  = "Name = u_t_L2.pdf\n"
      lines[19] = "Plot limits = XLIM_0 : 1e-1 ; XLIM_1 : 1e-7 ; YLIM_0 : 1e-15 ; YLIM_1 : 1e-5\n"
      lines[21] = "Plot title = ENABLED : True ; LOCATION : 1.0 ; TITLE : Temporal convergence of Q2 elements ($t \in [0, 10^{-2}]$ s)\n"
      lines[27] = "Simulation A directory = /testingPackage/Cases/MMS/u/\n"
      f         = open(REPO + '/testingPackage/Cases/' + plot_input[i], 'w')
      f.writelines(lines)
      f.close()
  #---------------------------
  # Copy plots to docs folder.
  #---------------------------
  if args.case == 'all':
    cases = ['MMS', 'deBoer']
  else:
    cases = [args.case]

  for case in cases:
    if not os.path.exists(REPO + '/docs/Cases/' + case + '/figures/'):
      os.makedirs(REPO + '/docs/Cases/' + args.case + '/figures/')
    
    files = glob.glob(REPO + '/testingPackage/Cases/' + case + '/figures/*.pdf')
    
    for file in files:
      try:
        cp_cmd = ['cp', file, REPO + '/docs/Cases/' + case + '/figures/']
        subprocess.run(cp_cmd, check=True, capture_output=False)
      except subprocess.CalledProcessError:
        sys.exit("\nERROR. Could not copy the case's figure(s) to the documentation folder.")
    #----------------------------
    # Write results to .tex file.
    #----------------------------
    texfile = open(REPO + '/docs/' + case + '-Tests.tex','w')

    texfile.write('\\subsection{' + case + '}\n')
    texfile.write('\\input{Cases/' + case + '/input.tex}')

    texfile.close()
  #----------------------------------
  # Build the regression test report.
  #----------------------------------
  if args.build:
    print()
    print("Building the regression test report...")
    os.chdir(REPO + '/docs/')
    try:
      bibtex_cmd = ['bibtex', 'regressionTestReport.aux']
      latex_cmd  = ['pdflatex', 'regressionTestReport.tex']
      subprocess.run(latex_cmd, check=True, capture_output=True)
      subprocess.run(latex_cmd, check=True, capture_output=True)
      subprocess.run(bibtex_cmd, check=False, capture_output=True)
      subprocess.run(latex_cmd, check=True, capture_output=True) # fix refs in .pdf
      subprocess.run(latex_cmd, check=True, capture_output=True) # fix refs in .pdf
    except subprocess.CalledProcessError:
      sys.exit("\nERROR. Could not generate regresstion test report.")
    print("Finished building the regression test report.")
  #--------------------------
  # Display results of tests.
  #--------------------------
  os.chdir(OriginalLocation)
  print("\n--------\nRESULTS:\n--------\n")
  try:
    subprocess.run(['cat', 'TestResult_summary'], check=True)
  except subprocess.CalledProcessError:
    sys.exit("\nERROR. Could not display summary of test results.")

#-------------------
# Begin main script.
#-------------------
if __name__=='__main__':

  start_time = time.time()

  run_all()

  end_time = time.time()
  time_elapsed = (end_time - start_time)/60.0
  print()
  print("---------------------------------------------")
  print("Finished running test cases in %.2f minutes." %time_elapsed)
  print("---------------------------------------------")
  print()
