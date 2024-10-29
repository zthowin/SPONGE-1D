#!/usr/bin/python3

import os, sys, argparse

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

def numpy_comparator():
  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to compare two global DOFs and check\
                                                whether they are within a specified relative tolerance\
                                                to one another.')
  parser.add_argument('standard', metavar='s', type=str, default='NULL',
                      help='the name of the gold standard result file')
  parser.add_argument('new', metavar='n', type=str, default='NULL',
                      help='the name of the new result file')
  parser.add_argument('--print', action='store_true',
                      help='flag to execute print statements')
  args = parser.parse_args()

  if args.print:
    print( '     (o) Comparing files ' + args.new + ' and ' + args.standard)

  if os.path.isfile(args.new):
    if os.path.isfile(args.standard):
      standard = np.load(args.standard)
      new      = np.load(args.standard)
      if np.isclose(new[:,0:standard.shape[0]], standard, rtol=1e-6):
        Result = True
    else:
      Result = False
  else:
    Result = False

  f = open("TestResult_summary","w")
  if Result == True:
    print("passed",file=f)
  else:
    print("FAILED",file=f)

  f.close()

if __name__=='__main__':

  numpy_comparator()
