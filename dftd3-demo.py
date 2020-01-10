#!/usr/bin/env python

import numpy, ctypes
from pyscf import lib
from pyscf import gto, scf, dft

name = 'c2h4_c2h4'

mol = gto.Mole()
mol.atom = '''
C  -0.471925  -0.471925  -1.859111
C   0.471925   0.471925  -1.859111
H  -0.872422  -0.872422  -0.936125
H   0.872422   0.872422  -0.936125
H  -0.870464  -0.870464  -2.783308
H   0.870464   0.870464  -2.783308
C  -0.471925   0.471925   1.859111
C   0.471925  -0.471925   1.859111
H  -0.872422   0.872422   0.936125
H   0.872422  -0.872422   0.936125
H  -0.870464   0.870464   2.783308
H   0.870464  -0.870464   2.783308
'''
mol.basis = 'aug-cc-pvdz'
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

def dftd3_inter(mol):
    _loaderpath = '/home/xchen/develope/libdftd3/lib'
    libdftd3 = numpy.ctypeslib.load_library('libdftd3.so', _loaderpath)

#    e_tot, g_rhf = mf_grad_scan(geom)
    func = 'pbe0'
    version = 4
    tz = 0
    coords = mol.atom_coords()
    itype = numpy.zeros(mol.natm, dtype=numpy.int32)
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        itype[ia] = lib.parameters.NUC[symb]
    edisp = numpy.zeros(1)
    grad = numpy.zeros((mol.natm,3)) 
    libdftd3.wrapper(ctypes.c_int(mol.natm),
             coords.ctypes.data_as(ctypes.c_void_p),
             itype.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_char_p(func.encode('utf-8')),
             ctypes.c_int(version),
             ctypes.c_int(tz),
             edisp.ctypes.data_as(ctypes.c_void_p),
             grad.ctypes.data_as(ctypes.c_void_p))
    return edisp,grad

print(mf_grad_with_dftd3(mol))
