import matplotlib.pyplot as plt
import numpy as np
import psi4
import hftools

"""

This is an implementation of ODA (Optimal Damping Algorithm)
as prescribed by Kudin, Scuseria, and Cances, J. Chem. Phys 2002.

Written by M.M. Davis

Last updated 08/01/2018
"""

def oda_init(mol,ref='rhf',basis='sto-3g'):
    print('ODA Initializing ...\n')
    if ref == 'rhf':
        data = hftools.rhf_init(mol,basis)
    elif ref != 'rhf':
        print('Only RHF implemented, apologies! -M')

    data['k']=0

    Do = data['D']
    h = data['h']
    g = data['g']
    gt = data['gt']
    
    nu = hftools.assemble_nu(g, gt, Do)
    Fo = hftools.assemble_fock(h, nu, Do)
    Eo = hftools.compute_energy(h, nu, Do)

    data['Dkt']=Do
    data['Fkt']=Fo
    data['Fs'] = [Fo]
    data['Ds'] = [Do]
    data['Ekp'] = 0
    return data

def oda_step(data, converged, dE_crit):
    #1. Diagonalize Fk~, assemble Dkp (D k plus one)
    #e, data['Fkt'] = np.linalg.eigh(data['Fkt'])
    print('ODA Step {} ... \n'.format(data['k']))
    C = hftools.diag_fock(data['Fkt'], data['X'])
    Dkp = hftools.assemble_Dkp(C, data['Dkt'], data['nocc'])
    data['Ds'].append(Dkp)
    #print(oda_compare_D(Dkp, data['Dkt']))
    h = data['h']
    Dkt = data['Dkt']
    g = data['g']
    gt = data['gt']

    nup = hftools.assemble_nu(g, gt, Dkp)
    Fkp = hftools.assemble_fock(h, nup, Dkp)
    nut = hftools.assemble_nu(g, gt, Dkt)
    Fkt = hftools.assemble_fock(h, nut, Dkt)

    data['Fs'].append(Fkp)
    
    Ekp = hftools.compute_energy(h, nup, Dkp) + data['Enuc']
    dE = Ekp - data['Ekp']
    print('dE: {}'.format(dE))

    if dE < dE_crit and oda_compare_D(Dkp, data['Dkt']): 
        data['converged'] = True
    data['Ekp'] = Ekp
    print(Ekp)
    #print(Ekp)
    data['dk'] = Dkp - data['Dkt']
    #print(data['dk'])
    l_new = oda_new_analytic(Fkp, Fkt, Dkp, Dkt)
    
    #l_analytic = oda_line_search(Dkt, data['dk'], Dkp, g, gt, h, analytic=True)
    #l_numerical = oda_line_search(Dkt, data['dk'], Dkp, g, gt, h, analytic=False)
    #print('ANALYTIC L : {}'.format(l_analytic))
    #print('NUMERICAL L: {}'.format(l_numerical))
    l = l_new
    Dkpt = (1 - l)*data['Dkt'] + l*Dkp
    Fkpt = (1 - l)*data['Fkt'] + l*Fkp

    data['Dkt'] = Dkpt
    data['Fkt'] = Fkpt
    data['k'] += 1
    
def oda_new_analytic(Fkp, Fkt, Dkp, Dkt):
    s = oda_calc_s(Fkt, Dkp, Dkt)
    c = oda_calc_c(Fkp, Fkt, Dkp, Dkt)
    if c <= -1*s/2:
        l = 1
    else:
        l = -1*s/(2*c)
    print(l)
    return l

def oda_calc_s(Fkt, Dkp, Dkt):
    return np.trace((np.matmul(Fkt,(Dkp - Dkt))))

def oda_calc_c(Fkp, Fkt, Dkp, Dkt):
    return np.trace((np.matmul((Fkp - Fkt),(Dkp - Dkt))))

def oda(mol,ref='rhf',basis='sto-3g', iterlim=50, dE_crit=1E-6):
    data = oda_init(mol,ref,basis)
    E = 0
    converged = False
    data['converged'] = converged
    while ((not data['converged']) and (data['k'] < iterlim)):
        oda_step(data, converged, dE_crit)
    return 42 #dummy

def oda_compare_D(Dkp, Dk):
    diff = Dkp - Dk 
    #print(np.trace(diff)/diff.shape[0])
    if abs(np.trace(diff)/diff.shape[0]) < 1E-8:
        return True
    #    return True
    #else:
    #    return False
    return False

def three_point_fit(back_E,center_E,forward_E):
    x=[-1, 0, 1]
    norm_back_E=back_E-center_E
    norm_forward_E=forward_E-center_E
    norm_center_E=0
    y=[norm_back_E,norm_center_E,norm_forward_E]
    return np.polyfit(x,y,2)

def get_dist_to_roots(back_E,center_E,forward_E):

    #print(back_E,center_E,forward_E)
    A,B,C=three_point_fit(back_E,center_E,forward_E)
    #print(A,B,C)
    x0=-1*B/(2*A)
    return x0

def analytic_line_search(Dk, Dk_, g, gt, h):
    #TODO: implement 
    pass

def oda_line_search(Dk, dk, Dkp, g, gt, h, analytic=False):
    #TODO: changed the input to Dk and Dkp rather than dk, need to change\
    #the numerical implementation if desired.

    v = hftools.assemble_nu(g, gt, Dk)
    vp = hftools.assemble_nu(g, gt, Dkp)
    if analytic:
        hd   = hftools.sum_mats(h, Dk)        
        hdp  = hftools.sum_mats(h, Dkp)
        vd   = hftools.sum_mats(v, Dk)
        vdp  = hftools.sum_mats(v, Dkp)
        vpd  = hftools.sum_mats(vp, Dk)
        vpdp = hftools.sum_mats(vp, Dkp)
        
        l = -1*(-1*hd - hdp + vd + 0.5*vdp + 0.5*vpd)/(vd + vdp + vpd + vpdp)
        return l
    lower = 0
    higher = 1
    delim = 0.5
    #p1 = (lower + delim)/2
    p1 = lower
    #p2 = (higher + delim)/2
    p2 = higher
    p3 = delim

    nu1 = hftools.assemble_nu(g, gt, Dk + p1*dk)
    nu2 = hftools.assemble_nu(g, gt, Dk + p2*dk)
    nu3 = hftools.assemble_nu(g, gt, Dk + p3*dk)

    E1 = hftools.compute_energy(h, nu1, Dk + p1*dk)
    E2 = hftools.compute_energy(h, nu2, Dk + p2*dk)
    E3 = hftools.compute_energy(h, nu3, Dk + p3*dk)
    #dist = get_dist_to_roots(E1,E3,E2)
    #l = 0.5 + dist*0.5
    print('E1 {} E2 {}'.format(E1,E2))
    print('ODA LINE SEARCH/MICRO\n')
    #return l
    #xs = np.linspace(0, 1, 101)
    #Es = []
    #for x in xs:
    #    print(x)
    #    nu_temp = hftools.assemble_nu(g, gt, Dk + x*dk)
    #    E_temp = hftools.compute_energy(h, nu_temp, Dk + x*dk)
    #    Es.append(E_temp)    
    #d = 1E-12
    #plt.ylim((1+d)*min(Es),(1-d)*max(Es))
    #plt.plot(xs,Es,'g-')
    #plt.show()
    dE_init = abs(E1 - E2)
    while True:
        if float(E1) < float(E2):
            #print('LOW')
            higher = delim
        elif E1 > E2:
            #print('HIGH')
            lower = delim
        p1 = (lower + delim)/2
        p2 = (higher + delim)/2
        delim = (lower + higher)/2
        nu1 = hftools.assemble_nu(g, gt, Dk + p1*dk)
        nu2 = hftools.assemble_nu(g, gt, Dk + p2*dk)
        E1 = hftools.compute_energy(h, nu1, Dk + p1*dk)
        E2 = hftools.compute_energy(h, nu2, Dk + p2*dk)
        #print('ODA Line search.\nE1: {}\nE2: {}\n'.format(E1,E2))
        #print('Es')
        #print(abs(E2-E1))
        if abs(E2 - E1)/dE_init < 1E-8:
            l = (p1 + p2)/2
            #print('Lambda:\n')
            #print(l)
            return l
    
        

if __name__ == "__main__":
    mol = psi4.geometry("""
    0 1
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    #silence psi4 and make scf_type pk to match this code
    psi4.core.be_quiet()
    psi4.set_options({'scf_type':'pk'})

    E=oda(mol, basis='aug-cc-pvdz', iterlim=1000, dE_crit=1E-10) 
    #psi_energy=psi4.energy('scf/sto-3g',molecule=mol)
    #dE=E-psi_energy
    #psi_match=abs(dE)<1E-6 #agreement between psi4 and this code?
    msg='Keep up the good work!'
    print("E: {}\nMessage: {}".format(E, msg)) 
    #print("PSI4_E: {}\n".format(psi_energy))
    #print("dE: {}\nMatch: {}\n".format(dE,psi_match))
