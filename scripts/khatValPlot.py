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

def ns(J,n0s):
  return n0s/J

def nf(ns):
  return 1 - ns

def khatExp(kappa, J):
  return np.exp(kappa*(J-1))

def khatKC(nf,n0s):
  n0f = 1 - n0s
  return ((nf**3)/(1 - nf**2))/((n0f**3)/(1 - n0f**2))

def khatMa(nf,n0s,kappa):
  n0f = 1 - n0s
  return ((nf/(1-nf))*((1-n0f)/n0f))**kappa

def khatEi(nf,n0s,kappa):
  n0f = 1 - n0s
  return (nf/n0f)**kappa

n0s = .25
J = np.linspace(0, 2, 500)
nsval = ns(J, n0s)
nfval = nf(nsval)
kappaval = 3.0
khatExpVal = khatExp(kappaval, J)
khatKCVal =  khatKC(nfval, n0s)
khatMaVal =  khatMa(nfval, n0s, kappaval)
khatEiVal =  khatEi(nfval, n0s, kappaval)

plt.figure(1)
#plt.plot(J, khatMaVal, 'k-', label=r'$\Bigg(\dfrac{n^\rf}{1 - n^\rf}\dfrac{1 - n_{0(\rs)}^\rf}{n_{0(\rs)}^\rf}\Bigg)^\kappa$')
#plt.plot(J, khatEiVal, 'k-.', label=r'$\Bigg(\dfrac{n^\rf}{n_{0(\rs)}^\rf}\Bigg)^\kappa$')
#plt.plot(J, khatExpVal, 'k--', label=r'$\exp[\kappa(J_\rs - 1)]$')
#plt.plot(J, khatKCVal, 'k:', label=r'$\dfrac{\Bigg(\dfrac{(n^\rf)^3}{1 - (n^\rf)^2}\Bigg)}{\Bigg(\dfrac{(n_{0(\rs)}^\rf)^3}{1 - (n^\rf_{0(\rs)})^2}\Bigg)}$')
plt.plot(J, khatMaVal, 'k-', label=r'Markert')
plt.plot(J, khatExpVal, 'k--', label=r'Lai \& Mow')
plt.plot(J, khatEiVal, 'k-.', label=r'Eipper')
plt.plot(J, khatKCVal, 'k:', label=r'Kozeny-Carman')

plt.legend(edgecolor='k', framealpha=1.0)
plt.xlabel(r'$J_\rs$ [--]',fontsize=16)
plt.ylabel(r'function values [--]',fontsize=16)
plt.xlim([0,2])
#plt.ylim([0,2])
plt.ylim([-10,70])
plt.xticks(ticks=[0, 0.25, 1, 2], labels=[0, r'$n_{0(\rs)}^\rs$', 1, 2])
#plt.yticks(ticks=[0, 1, 2], labels=[0, 1, 2])
#plt.savefig('khat_compare.pdf', dpi=600, bbox_inches='tight')
plt.savefig('khat_all.pdf', dpi=600, bbox_inches='tight')
