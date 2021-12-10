'''
author: Daniel Nichols
date: December 2021

Implementation of the VQLS algorithm presented in 
https://arxiv.org/pdf/1909.05820.pdf by Bravo-Prieto et al.
Requires python3, cirq, numpy and scipy to be installed.
To see the command-line options run:    python main.py --help
'''
# std imports
from argparse import ArgumentParser

# tpl imports
import numpy as np
from scipy.optimize import minimize

# local imports
from vqls import VQLS
from linear_algebra_utilities import hermitian_pauli_expansion, PAULIS, \
                                     pauli_expansion_to_str, classical_solve, \
                                     HADAMARD_UNITARY


''' SciPy available optimizers for minimize.
'''
AVAILABLE_OPTIMIZERS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 
                        'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 
                        'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

def get_args():
    ''' Returns parsed command-line arguments. Run `python main.py -h` to see
        available options and descriptions.
    '''
    parser = ArgumentParser()
    parser.add_argument('-s', '--shots', type=int, default=10000, 
        help='Number of times to collect measurements for each circuit.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random number seed. Helpful for reproducibility.')
    parser.add_argument('--sample', action='store_true',
        help='Determine values through many measurements instead of using \
        simulator provided values.')
    parser.add_argument('--optimizer', choices=AVAILABLE_OPTIMIZERS,
        default='Powell', help='Available optimization algorithms for C(alpha)')
    parser.add_argument('--max-iter', type=int, default=500,
        help='Maximum number of iterations for optimizer.')
    parser.add_argument('--ansatz-layers', type=int, default=3,
        help='Number of layers to use in fixed ansatz for V(alpha).')
    return parser.parse_args()


def main():
    args = get_args()

    # the A in Ax=B. Can be any hermitian matrix, but I construct it here as the
    # linear combination of paulis for simplicity.
    A = 0.7 *np.kron(np.kron(PAULIS['I'], np.kron(PAULIS['I'], PAULIS['I'])), PAULIS['I']) + \
        0.45*np.kron(np.kron(PAULIS['I'], np.kron(PAULIS['Z'], PAULIS['I'])), PAULIS['Z']) + \
        0.25*np.kron(np.kron(PAULIS['I'], np.kron(PAULIS['X'], PAULIS['Z'])), PAULIS['I'])

    # the unitary U s.t. |b> = U|0>
    U = np.kron(np.kron(np.kron(HADAMARD_UNITARY, HADAMARD_UNITARY), HADAMARD_UNITARY), HADAMARD_UNITARY)

    # compute b based on U
    zero_state = np.zeros(U.shape[0])
    zero_state[0] = 1
    b = U.dot(zero_state)

    print('Using matrix A with 𝜅(A) = {:.4f}'.format(np.linalg.cond(A, p=2)))

    # create VQLS circuit
    vqls = VQLS(A, U, shots=args.shots, sample=args.sample, 
                ansatz_layers=args.ansatz_layers)

    # run optimizer
    np.random.seed(args.seed)
    total_params = args.ansatz_layers*2*(vqls.num_qubits_-1) + vqls.num_qubits_
    alpha_0 = np.random.random(total_params)
    result = minimize(vqls.C, x0=alpha_0, method=args.optimizer, 
        options={'maxiter': args.max_iter})
    final_cost = result['fun']
    alpha_star = vqls.reshape_alpha(result['x'])

    # print out results
    print('Optimization {}.'.format(
        'successful' if result['success'] else 'failed'))
    print('Final cost: {}'.format(final_cost))
    print('Circuit C(alpha) executions: {}'.format(result['nfev']))
    print('Quantum result: {}'.format(result['x']))

    # compute V(alpha_star) to get actual answer
    quantum_x = vqls.compute_v_of_alpha(alpha_star)

    # compute answer classically
    classical_x = classical_solve(A, b)
    print('Classical result: {}'.format(classical_x))


    # compute errors cost
    ord = 2
    b_hat = A.dot(quantum_x)
    residual = b - b_hat
    backward_error = np.linalg.norm(residual, ord=ord)
    forward_error = np.linalg.norm(classical_x - quantum_x, ord=ord)
    relative_backward_error = backward_error / np.linalg.norm(b, ord=ord)
    relative_forward_error = forward_error/np.linalg.norm(classical_x, ord=ord)
    emf = relative_forward_error / relative_backward_error
    b_hat_norm = b_hat / np.linalg.norm(b_hat, ord=ord)
    norm_bhat_sq = b.dot(b_hat_norm) ** 2

    print('backward error: {}'.format(backward_error))
    print('forward error: {}'.format(forward_error))
    print('relative backward error: {}'.format(relative_backward_error))
    print('relative forward error: {}'.format(relative_forward_error))
    print('EMF: {}'.format(emf))
    print('(b.b̂)^2: {}'.format(norm_bhat_sq))


if __name__ == '__main__':
    main()