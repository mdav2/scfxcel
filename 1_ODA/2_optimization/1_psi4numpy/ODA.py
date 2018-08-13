"""
A reference implementation of Optimal Damping Algorithm - a simple form of 
relaxed constraint algorithm (RCA) for SCF acceleration.

References:

Adapted structure from MP2.py in psi4numpy
"""

__authors__    = "M.M. Davis"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni", "Rob Parrish"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat',False)

# memory for numpy in GB
numpy_memory = 2


mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})
# Set defaults
maxiter = 40
E_conv = 1.0E-6
D_conv = 1.0E-3

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('\nNumber of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Run a quick check to make sure everything will fit into memory
I_Size = (nbf**4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute required quantities for SCF
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))
t = time.time()

# Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141
H = T + V
# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

Hp = A.dot(H).dot(A)
e, C2 = np.linalg.eigh(Hp)
C = A.dot(C2)
Cocc = C[:, :ndocc]

D = np.einsum('pi,qi->pq', Cocc, Cocc)
print('\ntotal time taken for setup %.3f seconds' % (time.time() - t))


print('\nStart ODA iterations.\n')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

#1. Initialization (Cances, 2000, pg. 86)
Do = np.copy(D)

J = np.einsum('pqrs,rs->pq', I, D)
K = np.einsum('prqs,rs->pq', I, D)
Fo = H + J * 2 - K

Eo_1e = 2*np.trace(np.matmul(H, Do))
Eo_2e = np.trace(np.matmul(Fo, Do)) - 0.5*Eo_1e
Eo = Eo_1e + Eo_2e

k = 0
Dt = Do
Ft = Fo

#Et_1e = Eo_1e
#Et_2e = Eo_2e
Et = Eo
E_prev = Et
print("ITER | E \n")
print("{}  | {}".format(k,Et+Enuc))

for SCF_ITER in range(1, maxiter + 1):
    #2. Iterations (Cances, 2000 pg. 86)
    #a.
    #Diagonalize Fkt...
    orth_Ft = A.dot(Ft).dot(A)
    e, C2 = np.linalg.eigh(orth_Ft)
    # ... and assemble Dkp by aufbau
    C = A.dot(C2)
    Cocc = C[:, :ndocc]
    Dp = np.einsum('pi,qi->pq', Cocc, Cocc) 

    #b.
    #Dp - Dt 'small' determination
    #if np.trace(Dp - Dt) <= 1E-9:
    #    print("D convergence achieved, exiting")
    #    break

    #c. Assemble Fock Fp = F(Dp) ...
    J = np.einsum('pqrs,rs->pq', I, Dp)
    K = np.einsum('prqs,rs->pq', I, Dp)
    Fp = H + J * 2 - K

    # ... and compute E (Not sure what cances was up to here... removed)
    #Ep_1e = 2*np.trace(np.matmul(H, Dp))
    #Ep_2e = np.trace(np.matmul(Fp, Dp)) - 0.5*Ep_1e
    #Ep = Ep_1e + Ep_2e
    
    #d. Compute diagnostics
    s = np.trace(np.matmul(Ft, (Dp - Dt)))
    c = np.trace(np.matmul((Fp - Ft), (Dp - Dt)))
    
    #e. Conditionally set l (lambda) equal to:
    # l = { 1      if c <= -s/2,
    #       -s/2c  otherwise } ...
    if c <= -1*s/2:
        l = 1
    else:
        l = -1*s/(2*c)

    # ... and update quantities:
    Dt = (1 - l)*Dt + l*Dp
    Ft = (1 - l)*Ft + l*Fp
    #Et = Et + l*s + l**2 * c
    #Et_1e = (1 - l)*Et_1e + l*Ep_1e
    #Et_2e = Et - Et_1e
    Et = np.einsum('pq,pq->', Ft + H, Dt) + Enuc
    k += 1
    print("{}  | {} | {}".format(k, Et, l))
    if abs(Et - E_prev) < 1E-10:
        break
    E_prev = Et
