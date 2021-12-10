'''
author: Daniel Nichols
date: December 2021
Implementation of VQLS from https://arxiv.org/pdf/1909.05820.pdf
by Bravo-Prieto et al.
'''
# std imports
import itertools
from math import log2

# tpl imports
import cirq
import numpy as np

# local imports
from linear_algebra_utilities import hermitian_pauli_expansion, \
                                     pauli_expansion_to_str


''' Utility for circuits, since Cirq doesn't have a standard naming convention 
    for gates. Map for controlled paulis.
'''
CONTROLLED_PAULIS = {
    'X': cirq.CX,
    'Y': cirq.Y.controlled(),
    'Z': cirq.CZ,
    'I': cirq.I.controlled()
}


class VQLS:

    def __init__(self, A, U, shots=10000, sample=False, ansatz_layers=3):
        ''' Initialize VQLS circuit. Does not execute yet.
        Args:
            A: input matrix. Must have dimensions power of 2.
            U: unitary matrix such that U|0>=|b>.
            shots: number of times to sample measurements. Default 10000.
            sample: If true values are sampled through measurements. Otherwise
                    they are determined exactly through Cirq's state_vector.
        '''
        self.coeffs_, self.V_ = hermitian_pauli_expansion(A)
        print('Decomposed input matrix into A = {}'.format(
                                pauli_expansion_to_str(self.coeffs_, self.V_)))
        
        self.U_ = U
        self.coeffs_ = np.real(self.coeffs_)
        self.shots_ = shots
        self.sample_ = sample

        # not including any ancillas
        self.num_qubits_ = int(log2(A.shape[0]))
        self.ansatz_layers_ = ansatz_layers

        self.qubits_ = []
        self.circuit_ = None
        self.costs_ = []


    def init_qubits_(self, n_qubits):
        ''' Create n_qubits qubits inside this object.
        Args:
            n_qubits: number of qubits to initialize.
        '''
        self.qubits_ = [
            cirq.NamedQubit('q{}'.format(idx))
            for idx in range(n_qubits)
        ]


    def reshape_alpha(self, alpha):
        ''' Shape alpha by layer to match the fixed ansatz scheme from figure 3.
        Args:
            alpha: flattened input vector alpha
        Return:
            alpha: reshaped into (num_layers+1, *)
        '''
        alpha_0 = alpha[:self.num_qubits_]
        params_per_layer = 2 * (self.num_qubits_ - 1)
        alpha_b = np.reshape(alpha[self.num_qubits_:], (-1, params_per_layer))
        return [alpha_0, *alpha_b]


    def __repr__(self):
        ''' Utility to get the current circuit diagram as string.
        Return:
            circuit_string: circuit diagram as string
        '''
        if self.circuit_:
            return str(self.circuit_)
        return 'Empty VQLS Circuit'


    def print(self):
        ''' Print out the current circuit diagram to stdout.
        '''
        print(self)


    def v_ansatz_supplemental_(self, qubits, alpha):
        ''' Compute the ansatz for V(alpha). See figure S1 in paper.
        Args:
            qubits: qubits to perform V(alpha) on.
            alpha: parameteres for Ry gates.
        '''

        self.circuit_.append([
            cirq.ry(alpha[0][idx])(q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CZ(qubits[0], qubits[1]))
        self.circuit_.append(cirq.CZ(qubits[0], qubits[2]))

        self.circuit_.append([
            cirq.ry(alpha[1][idx])(q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CZ(qubits[1], qubits[2]))
        self.circuit_.append(cirq.CZ(qubits[0], qubits[2]))

        self.circuit_.append([
            cirq.ry(alpha[2][idx])(q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    
    def v_ansatz_(self, qubits, alpha):

        # first do Ry on each qubit
        self.circuit_.append([
            cirq.ry(alpha[0][idx])(q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        # now do each layer
        for layer_idx in range(self.ansatz_layers_):
            rots = alpha[layer_idx + 1]

            # CZ on every other pair starting at 0
            self.circuit_.append([
                cirq.CZ(qubits[idx], qubits[idx+1])
                for idx in range(0, len(qubits), 2)
                if idx != len(qubits)-1
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

            # Ry on each qubit
            self.circuit_.append([
                cirq.ry(rots[idx])(q)
                for idx, q in enumerate(qubits)
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            
            # CZ every other pair starting at 1
            self.circuit_.append([
                cirq.CZ(qubits[idx], qubits[idx+1])
                for idx in range(1, len(qubits), 2)
                if idx != len(qubits)-1
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

            # Ry on interior qubits only
            self.circuit_.append([
                cirq.ry(rots[self.num_qubits_ + idx])(q)
                for idx, q in enumerate(qubits[1:-1])
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


    def cv_ansatz_supplemental_(self, qubits, alpha, control, available_qubits):
        ''' Compute the controlled ansatz of V(alpha).
        Args:
            qubits: what qubits to act on.
            alpha: input to V(alpha)
            control: control qubit
            available_qubits: all available qubits in machine
        '''
        self.circuit_.append([
                cirq.ry(alpha[0][idx]).controlled()(control, q)
                for idx, q in enumerate(qubits)
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CCX(control, qubits[1], available_qubits[-1]))
        self.circuit_.append(cirq.CZ(qubits[0], available_qubits[-1]))
        self.circuit_.append(cirq.CCX(control, qubits[1], available_qubits[-1]))

        self.circuit_.append(cirq.CCX(control, qubits[0], available_qubits[-1]))
        self.circuit_.append(cirq.CZ(qubits[2], available_qubits[-1]))
        self.circuit_.append(cirq.CCX(control, qubits[0], available_qubits[-1]))

        self.circuit_.append([
                cirq.ry(alpha[1][idx]).controlled()(control, q)
                for idx, q in enumerate(qubits)
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CCX(control, qubits[2], available_qubits[-1]))
        self.circuit_.append(cirq.CZ(qubits[1], available_qubits[-1]))
        self.circuit_.append(cirq.CCX(control, qubits[2], available_qubits[-1]))

        self.circuit_.append(cirq.CCX(control, qubits[0], available_qubits[-1]))
        self.circuit_.append(cirq.CZ(qubits[2], available_qubits[-1]))
        self.circuit_.append(cirq.CCX(control, qubits[0], available_qubits[-1]))

        self.circuit_.append([
                cirq.ry(alpha[2][idx]).controlled()(control, q)
                for idx, q in enumerate(qubits)
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


    def cv_ansatz_(self, qubits, alpha, control, target):

        # do controlled rotations
        self.circuit_.append([
                cirq.ry(alpha[0][idx]).controlled()(control, q)
                for idx, q in enumerate(qubits)
            ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        # control of each layer
        for layer_idx in range(self.ansatz_layers_):
            rots = alpha[layer_idx + 1]

            # control of every other cz starting on qubit 0
            for idx in range(0, self.num_qubits_, 2):
                if idx == self.num_qubits_-1:
                    continue

                i, j = idx, idx+1
                self.circuit_.append(cirq.CCX(control, qubits[j], target))
                self.circuit_.append(cirq.CZ(qubits[i], target))
                self.circuit_.append(cirq.CCX(control, qubits[j], target))
            
            # controlled rotations
            self.circuit_.append([
                    cirq.ry(rots[idx]).controlled()(control, q)
                    for idx, q in enumerate(qubits)
                ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

            # control of every other cz starting on qubit 1
            for idx in range(1, self.num_qubits_, 2):
                if idx == self.num_qubits_-1:
                    continue

                i, j = idx, idx+1
                self.circuit_.append(cirq.CCX(control, qubits[j], target))
                self.circuit_.append(cirq.CZ(qubits[i], target))
                self.circuit_.append(cirq.CCX(control, qubits[j], target))

            # controlled rotations
            self.circuit_.append([
                    cirq.ry(rots[self.num_qubits_+idx]).controlled()(control, q)
                    for idx, q in enumerate(qubits[1:-1])
                ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)



    def cb_gate(self, control, qubits):
        ''' Apply controlled-U.
        Args:
            control: control qubit
            qubits: qubits that U acts on
        '''
        self.circuit_.append(
            cirq.MatrixGate(self.U_).controlled()(control, *qubits),
            strategy=cirq.InsertStrategy.NEW_THEN_INLINE
        )


    def hadamard_test_(self, A, alpha, qubits, ancilla, compute_im=False):
        ''' Hadamard test. See appendix C and figure 9a in paper.
        Args:
            A: A_l and A_l' in hadamard test circuit
            alpha: input to V(alpha)
            qubits: qubits to comput ansatz, A_l, and A_l' on
            ancilla: control of A_l and A_l'
            compute_im: if True, then do S_dagger on ancilla before A's. This
                        will compute the imaginary part.
        '''
        self.circuit_.append(cirq.H(ancilla))

        if compute_im:
            # S_dagger is same as Z^(-1/2)
            self.circuit_.append(cirq.ZPowGate(exponent=-0.5)(ancilla))

        self.v_ansatz_(qubits, alpha)

        for q_idx, pauli in enumerate(A[0]):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        for q_idx, pauli in enumerate(A[1]):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        self.circuit_.append(
            cirq.H(ancilla),
            strategy=cirq.InsertStrategy.NEW_THEN_INLINE
        )


    def special_hadamard_test(self, V, alpha, qubits, ancilla, target):
        ''' Hadamard test with cb
        '''
        self.circuit_.append(cirq.H(ancilla))

        self.cv_ansatz_(qubits, alpha, ancilla, target)

        for q_idx, pauli in enumerate(V):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        self.cb_gate(ancilla, qubits)

        self.circuit_.append(cirq.H(ancilla))


    def C(self, alpha):
        ''' C(alpha) as defined in the paper. Using C_G global objective.
        Args:
            alpha: complex vector input into cost function.
        '''
        
        alpha = self.reshape_alpha(alpha)

        psi_inner_product = 0.0
        for i, j in itertools.product(range(len(self.V_)), repeat=2):
            self.circuit_ = cirq.Circuit()
            self.simulator_ = cirq.Simulator()
            self.init_qubits_(self.num_qubits_+2)

            self.hadamard_test_(
                [self.V_[i], self.V_[j]],
                alpha,
                self.qubits_[1:-1],
                self.qubits_[0]
            )

            prob_1_on_q0 = 0.0
            if self.sample_:
                self.circuit_.append(
                    cirq.measure(self.qubits_[0], key="q0"),
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

                results = self.simulator_.run(self.circuit_, 
                                            repetitions=self.shots_)
                counts = results.histogram(key='q0')

                if 1 in counts:
                    prob_1_on_q0 = float(counts[1]) / self.shots_
            else:
                result = self.simulator_.simulate(self.circuit_)
                output_state = np.real(result.state_vector())
                prob_1_on_q0 = sum(output_state[len(output_state)//2:] ** 2)
            
            c_l = self.coeffs_[i]
            c_l_prime = np.conj(self.coeffs_[j])

            # beta_ll' = Prob(0) - Prob(1) = 1-2*Prob(0)
            beta_l = 1.0 - 2.0*prob_1_on_q0

            # see equation 14 in paper
            psi_inner_product += c_l * c_l_prime * beta_l

        b_psi_squared = 0.0
        for i, j in itertools.product(range(len(self.V_)), repeat=2):

            gamma_l = 1.0
            for v_i in [self.V_[i], self.V_[j]]:
                self.circuit_ = cirq.Circuit()
                self.simulator_ = cirq.Simulator()
                self.init_qubits_(self.num_qubits_+2)

                self.special_hadamard_test(
                    v_i, alpha, self.qubits_[1:-1], 
                    self.qubits_[0], self.qubits_[-1]
                )

                prob_1_on_q0 = 0.0
                if self.sample_:
                    self.circuit_.append(
                        cirq.measure(self.qubits_[0], key="q0"),
                        strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                    )

                    results = self.simulator_.run(self.circuit_, 
                                                repetitions=self.shots_)
                    counts = results.histogram(key='q0')

                    if 1 in counts:
                        prob_1_on_q0 = float(counts[1]) / self.shots_
                else:
                    result = self.simulator_.simulate(self.circuit_)
                    output_state = np.real(result.state_vector())
                    prob_1_on_q0 = sum(output_state[len(output_state)//2:] ** 2)

                # gamma_ll' = Prob(0) - Prob(1) = 1-2*Prob(0)
                gamma_l *= 1.0 - 2.0*prob_1_on_q0

            c_l = self.coeffs_[i]
            c_l_prime = np.conj(self.coeffs_[j])

            # see equation (16) in paper
            b_psi_squared += c_l * c_l_prime * gamma_l
    
        # see equations (3-7) in paper
        C_alpha = 1.0 - float(b_psi_squared)/psi_inner_product
        self.costs_.append(C_alpha)
        return C_alpha


    def compute_v_of_alpha(self, alpha, sample=False):
        ''' Compute V(alpha) using the ansatz. When called with alpha_star
            this will return |x> = x/||x||
        '''
        self.circuit_ = cirq.Circuit()
        self.simulator_ = cirq.Simulator()
        self.init_qubits_(self.num_qubits_)

        self.v_ansatz_(self.qubits_, alpha)

        sv = None
        if sample:
            self.circuit_.append(
                cirq.measure(*self.qubits_, key='output'), 
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

            results = self.simulator_.run(self.circuit_, repetitions=self.shots_)
            hist = results.histogram(key='output')
            total = sum(hist.values())

            bases = []
            amplitudes = []
            for x in sorted(hist):
                phase = np.sign(hist[x]) * np.sqrt( float(hist[x])/total )
                amplitudes.append(phase)
                bases.append('{}|{:b}>'.format(phase, x))

            print(' + '.join(bases))
            sv = np.array(amplitudes)

        else:
            results = self.simulator_.simulate(self.circuit_)
            sv = results.state_vector()

        return sv

