#!/usr/bin/python3

import os, sys, argparse, filecmp, subprocess


def strict_comparator():
  #---------------------------
  # Read command line options.
  #---------------------------
  parser = argparse.ArgumentParser(description='This file is used to compare two binary files for an exact match.')
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
      Result = filecmp.cmp(args.new,args.standard)
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

  strict_comparator()
