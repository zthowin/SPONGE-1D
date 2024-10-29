#!/usr/bin/python3

import os, sys, argparse, subprocess

def run_case():
  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to run a singular test case.')
  parser.add_argument('case', metavar='c', type=str, default='NULL',
                      help='the name of the test case')
  parser.add_argument('--print', action='store_true',
                      help='flag to execute print statements')
  args = parser.parse_args()
  #-----------------------------
  # Check environment variables.
  #-----------------------------
  try:
    REPO = os.environ['REPO']
  except KeyError:
    sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")
  #-------------------
  # Run the test case.
  #-------------------
  if args.print:
    print()
    print( '     (o) Running case ' + args.case)

  CaseDir = REPO + '/testingPackage/Cases/' + args.case + '/'
  
  if args.print:
    print( '     (o) Test case location = ' + CaseDir)
  
  if not os.path.exists(REPO + '/docs/Cases/' + args.case):
    try:
      doc_cmd = ['mkdir', '-p', REPO + '/docs/Cases/' + args.case]
      subprocess.run(doc_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
      sys.exit("\nERROR. Could not generate case directory.")

  OriginalLocation = os.getcwd()
  os.chdir(CaseDir)
  
  try:
    run_cmd = ['python3', REPO + '/src/runMain.py', CaseDir + 'input-' + args.case.split('/')[0] + '.dat', '-t', '-r']
    with open(CaseDir + 'tmp', 'w') as outFile:
      sim = subprocess.run(run_cmd, check=True, stdout=outFile)
  except subprocess.CalledProcessError:
    sys.exit("\nERROR. Could not run the test case.")
  
  try:
    cp_cmd = ['cp', REPO + '/testingPackage/compare.py', CaseDir]
    subprocess.run(cp_cmd, check=True, capture_output=True)
  except subprocess.CalledProcessError:
    sys.exit("\nERROR. Could not copy the compare script to the case directory.")
  
  try:
    if args.print:
      comp_cmd = ['python3', 'compare.py', args.case, '--print']
    else:
      comp_cmd = ['python3', 'compare.py', args.case]
    subprocess.run(comp_cmd, check=True, capture_output=False)
  except subprocess.CalledProcessError:
    sys.exit("\nERROR. Could not compare the new test case results to the standard results.")
  #---------------------
  # Communicate results.
  #---------------------
  os.chdir(OriginalLocation)

  g = open(CaseDir + 'TestResult_summary','r')
  Result = g.readlines()[0].replace('\n','')
  g.close()

  if args.print:
    print( "     Case " + args.case + ": " + Result)

  with open('TestResult_summary','a') as file: file.write('Case ' + args.case + ': ' + Result + '\n')

if __name__=='__main__':

  run_case()
