"""
A restricted open-shell Hartree-Fock script with ODA convergence
using the Psi4NumPy Formalism.

References:
- Equations and algorithm taken from Psi4
- ODA from Cances 2000
- Heavily adapted from ROHF_libJK.py
"""

__authors__ = "M.M. Davis"
__credits__ = ["M.M. Davis"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"

import time
import numpy as np
import helper_HF as scf_helper
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# Triplet O2
mol = psi4.geometry("""
    0 3
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'guess': 'gwh',
                  'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'reference': 'rohf'})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

# Set occupations
nocca = wfn.nalpha()
noccb = wfn.nbeta()
ndocc = min(nocca, noccb)
nocc = max(nocca, noccb)
nsocc = nocc - ndocc

# Set defaults
maxiter = 20
E_conv = 1.0E-8
D_conv = 1.0E-8
guess = 'gwh'

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
nbf = S.shape[0]

print('\nNumber of doubly occupied orbitals: %d' % ndocc)
print('Number of singly occupied orbitals:   %d' % nsocc)
print('Number of basis functions:            %d' % nbf)

V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time()-t))

t = time.time()

# Build H_core
H = T + V

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

if guess == 'gwh':
    F = 0.875 * S * (np.diag(H)[:, None] + np.diag(H))
    F[np.diag_indices_from(F)] = np.diag(H)
elif guess == 'core':
    F = H.copy()
else:
    raise Exception("Unrecognized guess type %s. Please use 'core' or 'gwh'." % guess)


