''' Some helpful code for doing linear algebra.
author: Daniel Nichols
date: December 2021
'''
# std imports
from math import log2
from itertools import product
from functools import reduce

# tpl imports
import numpy as np


''' Map of pauli matrices keyed on name.
'''
PAULIS = {
    'X': np.array([[0, 1], [ 1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'I': np.array([[1, 0], [ 0, 1]], dtype=np.complex128),
}
HADAMARD_UNITARY = (1/np.sqrt(2.0)) * np.array([[1,1],[1,-1]], dtype=np.float64)


def hilbert_schmidt_inner_product(A, B):
    ''' Compute tr(A^H . B)
    '''
    return (np.dot(A.conj().T, B)).trace()


def hermitian_pauli_expansion(H):
    ''' Decomposes a hermitian matrix with 2^l size into a linear combination of
        Pauli matrices. Used for computing equation (1) in paper.
        An example output is [0.3, 0.7], [['I', 'I'], ['Z', 'X']]
    Args:
        H: hermitian matrix with axes power of two
    Return:
        c, A: c is the list of coefficients. A is the list of Pauli operators.
    '''
    if len(H.shape) != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('input matrix not square')

    if (H.shape[0] & (H.shape[0]-1) != 0) or H.shape[0] == 0:
        raise ValueError('input matrix dim not power of 2')

    if not np.array_equal(H.conj().T, H):
        raise ValueError('input matrix not self-adjoint')

    # c_l and A_l for all l from VQLS paper equation (1)
    c = []
    A = []

    N = H.shape[0]
    log2N = int(log2(N))
    normalizer = 1.0 / float(N)
    for pauli_list in product(PAULIS.items(), repeat=log2N):
        names = list(map(lambda x: x[0], pauli_list))

        Al = reduce(np.kron, map(lambda x: x[1], pauli_list))
        a_ij = normalizer * hilbert_schmidt_inner_product(Al, H)

        if not np.isclose(a_ij, 0):
            c.append( a_ij )
            A.append(names)

    return c, A


def pauli_expansion_to_str(coeffs, paulis):
    ''' Formatted string of pauli expansion. Coeffs and Paulis should be in the 
        format returned by hermitian_pauli_expansion.
    Args:
        coeffs: linear combination coefficients.
        paulis: list of lists of Paulis.
    Return:
        formatted string 
    '''
    return ' + '.join( 
        map(lambda x: '{}*{}'.format(x[0],'⊗'.join(x[1])), zip(coeffs, paulis)) 
    )


def classical_solve(A, b):
    ''' Calls an optimal classical-only algorithm for solving Ax=b.
    Args:
        A: matrix, 2d-array-like
        b: vector: 1d-array like
    Return:
        x: vector such that Ax=b
    '''
    return np.linalg.solve(A, b)
