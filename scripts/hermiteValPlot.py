import numpy as np
import matplotlib.pyplot as plt
import sys, os
try:
  REPO = os.environ['REPO']
except KeyError:
  sys.exit("-------------------\nCOMMAND LINE ERROR:\n-------------------\nSet the REPO environment variable.")

sys.path.insert(1, REPO + '/src/')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsthm,amsfonts,amssymb,amscd,mathtools,xcolor} \input{' + REPO + '/scripts/macros.tex}')

def N1(xi):
  return 0.25*((1 - xi)**2)*(2 + xi)

def N2(xi):
  return 0.25*((1 - xi)**2)*(1 + xi)

def N3(xi):
  return 0.25*((1 + xi)**2)*(2 - xi)

def N4(xi):
  return 0.25*((1 + xi)**2)*(xi - 1)

xiVal = np.linspace(-1, 1, 100)
plt.figure(1)
plt.plot(xiVal, N1(xiVal), 'k-', label=r'$N_{1(H)}^{e,u}$')
plt.plot(xiVal, N2(xiVal), 'r-', label=r'$N_{2(H)}^{e,u}$')
plt.plot(xiVal, N3(xiVal), 'b-', label=r'$N_{3(H)}^{e,u}$')
plt.plot(xiVal, N4(xiVal), 'g-', label=r'$N_{4(H)}^{e,u}$')

plt.legend(edgecolor='k', framealpha=1.0)
plt.xlabel(r'$\xi$[--]',fontsize=16)
plt.ylabel(r'function values [--]',fontsize=16)
plt.xlim([-1,1])
plt.ylim([-0.4,1])
plt.savefig('hermite_sf.pdf', dpi=600, bbox_inches='tight')
