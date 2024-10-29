#!/usr/bin/python3
import os, sys, argparse, subprocess

if __name__=='__main__':
  
  REPO = os.environ['REPO']

  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to call the strict comparator.')
  parser.add_argument('case', metavar='c', type=str, default='NULL',
                      help='the name of the test case')
  parser.add_argument('--print', action='store_true',
                      help='flag to execute print statements')
  args = parser.parse_args()

  if args.case == 'Shock':
    try:
      if args.print:
        comp_cmd = [REPO + '/testingPackage/numpy_comparator.py', 'displacement_std.npy', 'displacement.npy', '--print']
      else:
        comp_cmd = [REPO + '/testingPackage/numpy_comparator.py', 'displacement_std.npy', 'displacement.npy']
      subprocess.run(comp_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError:
      sys.exit("\nERROR. Could not run the strict comparator.")
  else:
    try:
      if args.print:
        comp_cmd = [REPO + '/testingPackage/strict_comparator.py', 'displacement_std.npy', 'displacement.npy', '--print']
      else:
        comp_cmd = [REPO + '/testingPackage/strict_comparator.py', 'displacement_std.npy', 'displacement.npy']
      subprocess.run(comp_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError:
      sys.exit("\nERROR. Could not run the strict comparator.")
