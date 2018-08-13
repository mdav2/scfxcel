"""
A reference implementation of Optimal Damping Algorithm for UHF references.
ODA is a simple form of relaxed constraint algorithm (RCA) for SCF acceleration.

References:

Adapted structure from UHF_libJK.py in psi4numpy
"""

__authors__    = "M.M. Davis"
__credits__   = ["M.M. Davis"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"

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
    0 3
    O
    O 1 1.2
symmetry c1
""")


psi4.set_options({'basis': 'aug-cc-pvdz',
                  'reference': 'uhf',
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
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

print('\nNumber of doubly occupied orbitals: %d' % nbeta)
print('Number of singly occupied orbitals: %d' % (nalpha - nbeta))
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

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))
t = time.time()

# Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141
H = T + V
# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

def diag_H(H, nocc):
    """Diagonalizes provided Fock matrix for orbital coefficients C and density
    matrix D, using equations from [Szabo:1996] pp. 139 & 145.

    Parameters
    ----------
    H : numpy.array
        Fock matrix to diagonalize. 
    nocc : int
        Number of occupied molecular orbitals.

    Returns
    -------
    C : numpy.array 
        Molecular orbital coefficient matrix
    D : numpy.array
        SCF density matrix
    """
    Hp = A.dot(H).dot(A)        # Eqn. 3.177
    e, C2 = np.linalg.eigh(Hp)  # Solving Eqn. 1.178
    C = A.dot(C2)               # Back transformation, Eqn. 3.174
    Cocc = C[:, :nocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc) # Eqn. 3.145
    return (C, D)

#Form C and D matrices.
Ca, Da = diag_H(H, nalpha)
Cb, Db = diag_H(H, nbeta)
Cocca = psi4.core.Matrix(nbf, nalpha)
npCa = np.asarray(Cocca)
npCa[:] = Ca[:, :nalpha]
Coccb = psi4.core.Matrix(nbf, nbeta)
npCb = np.asarray(Coccb)
npCb[:] = Cb[:, :nbeta]
print('\ntotal time taken for setup %.3f seconds' % (time.time() - t))


print('\nStart ODA iterations.\n')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

#1. Initialization (Cances, 2000, pg. 86)

jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()
jk.C_left_add(Cocca)
jk.C_left_add(Coccb)
jk.compute()
Ja = np.asarray(jk.J()[0])
Jb = np.asarray(jk.J()[1])
Ka = np.asarray(jk.K()[0])
Kb = np.asarray(jk.K()[1])
Fo_a = H + (Ja + Jb) - Ka
Fo_b = H + (Ja + Jb) - Kb

Eo = np.einsum('pq,pq->', Da + Db, H)
Eo += np.einsum('pq,pq->', Da, Fo_a)
Eo += np.einsum('pq,pq->', Db, Fo_b)
Eo *= 0.5
Eo += Enuc

k = 0

Dt_a = Da
Dt_b = Db
Ft_a = Fo_a
Ft_b = Fo_b

Et = Eo
E_prev = Et
print("ITER | E \n")
print("{}  | {}".format(k,Et+Enuc))

for SCF_ITER in range(1, maxiter + 1):
    #2. Iterations (Cances, 2000 pg. 86)
    #a.
    #Diagonalize Fkt...
    # ... and assemble Dkp by aufbau
    Ca, Dp_a = diag_H(Ft_a, nalpha)
    Cb, Dp_b = diag_H(Ft_b, nbeta)
    
    #b.
    #Dp - Dt 'small' determination, what criterion
    #if np.trace(Dp - Dt) <= 1E-9:
    #    print("D convergence achieved, exiting")
    #    break

    #c. Assemble Fock Fp = F(Dp) ...
    npCa[:] = Ca[:, :nalpha]
    npCb[:] = Cb[:, :nbeta]
    jk.compute()

    Ja = np.asarray(jk.J()[0])
    Jb = np.asarray(jk.J()[1])
    Ka = np.asarray(jk.K()[0])
    Kb = np.asarray(jk.K()[1])
    
    Fp_a = H + (Ja + Jb) - Ka
    Fp_b = H + (Ja + Jb) - Kb

    #d. Compute diagnostics
    s = np.trace(np.matmul((Ft_a + Ft_b), (Dp_a + Dp_b) - (Dt_a + Dt_b)))
    c = np.trace(np.matmul(((Fp_a + Fp_b) - (Ft_a + Ft_b)), ((Dp_a + Dp_b) - (Dt_a + Dt_b))))
    #or should be separate diagnostics?
    s_a = np.trace(np.matmul(Ft_a, Dp_a - Dt_a))
    s_b = np.trace(np.matmul(Ft_b, Dp_b - Dt_b))
    c_a = np.trace(np.matmul(Fp_a - Ft_a, Dp_a - Dt_a))
    c_b = np.trace(np.matmul(Fp_b - Ft_b, Dp_b - Dt_b))
    
    #e. Conditionally set l (lambda) equal to:
    # l = { 1      if c <= -s/2,
    #       -s/2c  otherwise } ...
    #for alpha:
    if c_a  <= -1*s_a/2:
        l_a = 1
    else:
        l_a = -1*s_a/(2*c_a)
    #for beta:
    if c_b <= -1*s_b/2:
        l_b = 1
    else:
        l_b = -1*s_b/(2*c_b)
    

    # ... and update quantities:
    Dt_a = (1 - l_a)*Dt_a + l_a*Dp_a
    Dt_b = (1 - l_b)*Dt_b + l_b*Dp_b
    Ft_a = (1 - l_a)*Ft_a + l_a*Fp_a
    Ft_b = (1 - l_b)*Ft_b + l_b*Fp_b
 
    # Compute energy
    Et = np.einsum('pq,pq->', Dt_a + Dt_b, H)
    Et += np.einsum('pq,pq->', Dt_a, Ft_a)
    Et += np.einsum('pq,pq->', Dt_b, Ft_b)
    Et *= 0.5
    Et += Enuc

    #Post iteration cleanups and callbacks
    k += 1
    print("{}  | {} | {} | {}".format(k, Et, l_a, l_b))
    if abs(Et - E_prev) < 1E-10:
        break
    E_prev = Et
