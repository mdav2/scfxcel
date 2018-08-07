import numpy as np
import psi4
import rhf
def rhf_init(mol, basis='sto-3g'):
#pull in basis sets and mintshelper (molecular integrals) from psi4
    basis = psi4.core.BasisSet.build(mol, target=str(basis), key='basis')
    mints = psi4.core.MintsHelper(basis) 
    
#extract useful basic quantities from molecule, mints
    Enuc = mol.nuclear_repulsion_energy()
    natom = mol.natom()
    charge = mol.molecular_charge()
    norb = mints.basisset().nbf()
    nuclear_charges = [mol.Z(A) for A in range(natom)]
    nocc = int((sum(nuclear_charges) - charge)/2)

#Build [zero]D, overlap (S), kinetic (T), one-electron (V), two-electron (g)
#+make orthogonalizer (X) from S, make h from (T + V)
    S =  mints.ao_overlap()
    S.power(-0.5,1e-14)
    X = np.asarray(S)
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    D = np.zeros(shape=T.shape) #core guess
    g = np.asarray(mints.ao_eri())
    h = T + V

#form two-e int. (g) and transpose (gt) in physicists notation
    g = g.transpose(0,2,1,3)
    gt = g.transpose(0,1,3,2)

    niter = 0

    E=0
    #E, dE, D, C, e, F = rhf.scf_iter(nocc,h,X,g,gt,D,E)
    nu = assemble_nu(g, gt, D)
    Fk = assemble_fock(h, nu, D) #initial formation of the fock
    pack_obj={

            'nocc':nocc,
            'g':g,
            'gt':gt,
            'h':h,
            'X':X,
            'D':D,
            'E':E,
            'F':Fk,
            'Enuc':Enuc

            }

    return pack_obj

def assemble_Dkp(C,Dk,nocc):
    r = Dk.shape[0]

    Dkp=np.zeros(Dk.shape) #Dk+1 
    
    for m in range(r):
        for n in range(r):
            Dkp[m][n] = 0
            for i in range(nocc):
                Dkp[m][n] += C[m][i]*C[n][i]
            Dkp[m][n] = 2*Dkp[m][n]
    return Dkp

def compute_energy(h, nu, Dk):
    #Dk is Dk
    r = h.shape[0]
    elec_energy = np.einsum('mn,nm->',(h + 0.5*nu), Dk)

    return elec_energy

def assemble_nu(g,gt,Dk):
    nu=np.einsum('mrns,sr->mn', g - 0.5*gt, Dk)
    return nu

def assemble_fock(h,nu,Dk):
    Fk = h + nu
    return Fk

def diag_fock(f, X):
    Xi = np.linalg.inv(X)
    ft = X.dot(f).dot(X)
    evals, evecs = np.linalg.eigh(ft)
    #print(evals)
    C = X.dot(evecs)
    return C
