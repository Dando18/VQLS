'''
author: Daniel Nichols
date: December 2021
'''
from argparse import ArgumentParser

import numpy as np
from scipy.optimize import minimize

from vqls import VQLS
from linear_algebra_utilities import hermitian_pauli_expansion, PAULIS, \
                                     pauli_expansion_to_str, classical_solve, \
                                     HADAMARD_UNITARY


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--shots', type=int, default=10000, 
        help='Number of times to collect measurements for each circuit.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random number seed. Helpful for reproducibility.')
    parser.add_argument('--max-iter', type=int, default=500,
        help='maximum number of iterations for optimizer.')
    return parser.parse_args()


def main():
    args = get_args()

    #A = 0.55*np.kron(PAULIS['I'], np.kron(PAULIS['I'], PAULIS['Z'])) + \
    #    0.45*np.kron(PAULIS['I'], np.kron(PAULIS['I'], PAULIS['I']))

    A = 0.3 *np.kron(PAULIS['Y'], np.kron(PAULIS['I'], PAULIS['I'])) + \
        0.45*np.kron(PAULIS['I'], np.kron(PAULIS['Z'], PAULIS['I'])) + \
        0.25*np.kron(PAULIS['I'], np.kron(PAULIS['X'], PAULIS['Z']))

    # the unitary U s.t. |b> = U|0>
    U = np.kron(np.kron(HADAMARD_UNITARY, HADAMARD_UNITARY), HADAMARD_UNITARY)

    # construct b based on U
    zero_state = np.zeros(U.shape[0])
    zero_state[0] = 1
    b = U.dot(zero_state)

    # create VQLS circuit
    vqls = VQLS(A, U, shots=args.shots)

    np.random.seed(args.seed)
    x0 = [float(np.random.randint(0,3000))/1000 for i in range(0, 9)]
    result = minimize(vqls.C, x0=x0, method='COBYLA', options={'maxiter': args.max_iter})
    print(result)
    alpha_star = np.reshape(result['x'], (-1, 3))

    quantum_x = vqls.compute_v_of_alpha(alpha_star)

    
    # compute answer classically
    classical_x = classical_solve(A, b)
    print('Classical result: {}'.format(classical_x))


    # compute classical cost
    relative = A.dot(quantum_x) / np.linalg.norm(A.dot(quantum_x))
    classical_cost = b.dot(relative) ** 2
    print('Classical cost: {}'.format(classical_cost))

    # norm diff
    normalized_diff = np.linalg.norm(classical_x - quantum_x)
    print('Normalized diff: {}'.format(normalized_diff))


if __name__ == '__main__':
    main()