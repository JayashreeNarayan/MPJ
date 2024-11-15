import ctypes
import os

import numpy as np
import numpy.ctypeslib as npct

__all__ = ['c_laplace', 'c_ddot', 'c_daxpy', 'c_daxpy2']

# Import from shared library next to this file as a package
liblaplace = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'liblaplace.so'))

# void laplace_filter(long double *u, long double *u_new, int n)
c_laplace = liblaplace.laplace_filter
c_laplace.restype = None
c_laplace.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_int,
]

# double ddot(double *u, double *v, int n)
c_ddot = liblaplace.ddot
c_ddot.restype = ctypes.c_double
c_ddot.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_int,
]

# void daxpy(double *u, double *v, double *result, double alpha, int n) {
c_daxpy = liblaplace.daxpy
c_daxpy.restype = None
c_daxpy.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int,
]

# void daxpy2(double *v, double *u, double alpha, int n)
c_daxpy2 = liblaplace.daxpy2
c_daxpy2.restype = None
c_daxpy2.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int,
]

# int conj_grad(double *b, double *x0, double *x, double tol, long int n) {
c_conj_grad = liblaplace.conj_grad
c_conj_grad.restype = ctypes.c_int
c_conj_grad.argtypes = [
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int64
]
