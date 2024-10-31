# SPONGE-1D

[[_TOC_]]

## Introduction

SPONGE-1D is a serial finite element simulation code that uses the Theory of Porous Media (TPM) to model multiphase (and single-phase) materials in a one-dimensional (1-D) uniaxial strain, unidirectional pore fluid flow environment. Its specialty is in accomodating a range of physical models (formulations) to describe the material response to high strain-rate loadings, e.g., from shock waves. Features include:

1. Adaptive time-stepping explicit Runge-Kutta time integration (in addition to traditional explict central difference and implict Newmark-beta).
2. A wide range of choices for mixed finite element types, including C<sup>1</sup>-continous elements to resolve physics ignored in other numerical models.
3. Shock viscosity to enable stable simulations (in addition to other methods).
4. Various constitutive modeling choices for solid (s) skeleton and pore fluid (f) response (at present, stability is limited to locally homogeneous temperature models).

## Installation

Installation requires Python (3.6+) and the following libraries:
* Scientific Python stack: Numpy (1.18+), Scipy (1.3.1+)

For post-processing with included the scripts, active versions of Matplotlib and LaTeX are required.

## Usage

It is recommended that a common project directory is maintained as follows:
* root-Directory/
  * input-file-Directory/
  * output-file-Directory/

To run SPONGE-1D, execute
`python3 src/runMain.py input-file-Directory/input-file.dat`
in the command line (either on a local workstation, or a supercomputer, provided that Python and the required modules are appropriately loaded).

## Testing package

Users may wish to test the installation of SPONGE-1D prior to use (requires active installation of Matplotlib and LaTeX). To do so, navigate to `testingPackage/` and execute `runAll.py` with the command line argument `all` (or `MMS`, if you are in a hurry). One may also invoke `--build` to build the regression report.
