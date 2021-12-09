'''
Full VQLS implementation.
'''
import itertools

import cirq
import numpy as np

from linear_algebra_utilities import hermitian_pauli_expansion, \
                                     pauli_expansion_to_str


''' Utility for circuits. Map for controlled paulis.
'''
CONTROLLED_PAULIS = {
    'X': cirq.CX,
    'Y': cirq.Y.controlled(),
    'Z': cirq.CZ,
    'I': cirq.I.controlled()
}



class VQLS:

    def __init__(self, A, U, shots=10000):
        self.coeffs_, self.V_ = hermitian_pauli_expansion(A)
        print('Decomposed input matrix into {}'.format(
                                pauli_expansion_to_str(self.coeffs_, self.V_)))
        
        self.U_ = U
        self.coeffs_ = np.real(self.coeffs_)
        self.shots_ = shots

        self.qubits_ = []
        self.circuit_ = None

    def init_qubits_(self, n_qubits):
        self.qubits_ = [
            cirq.NamedQubit('q{}'.format(idx))
            for idx in range(n_qubits)
        ]

    def scratch(self, n_qubits):
        return [
            cirq.NamedQubit('qs{}'.format(idx))
            for idx in range(n_qubits)
        ]

    def __repr__(self):
        if self.circuit_:
            return str(self.circuit_)
        return 'Empty VQLS Circuit'

    def print(self):
        print(self)

    def v_ansatz_(self, qubits, alpha):
        '''
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


    def cv_ansatz_(self, qubits, alpha, ancilla, reg):
        self.circuit_.append([
            cirq.ry(alpha[0][idx]).controlled()(ancilla, q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CCX(ancilla, qubits[1], reg[-1]))
        self.circuit_.append(cirq.CZ(qubits[0], reg[-1]))
        self.circuit_.append(cirq.CCX(ancilla, qubits[1], reg[-1]))

        self.circuit_.append(cirq.CCX(ancilla, qubits[0], reg[-1]))
        self.circuit_.append(cirq.CZ(qubits[2], reg[-1]))
        self.circuit_.append(cirq.CCX(ancilla, qubits[0], reg[-1]))

        self.circuit_.append([
            cirq.ry(alpha[1][idx]).controlled()(ancilla, q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        self.circuit_.append(cirq.CCX(ancilla, qubits[2], reg[-1]))
        self.circuit_.append(cirq.CZ(qubits[1], reg[-1]))
        self.circuit_.append(cirq.CCX(ancilla, qubits[2], reg[-1]))

        self.circuit_.append(cirq.CCX(ancilla, qubits[0], reg[-1]))
        self.circuit_.append(cirq.CZ(qubits[2], reg[-1]))
        self.circuit_.append(cirq.CCX(ancilla, qubits[0], reg[-1]))

        self.circuit_.append([
            cirq.ry(alpha[2][idx]).controlled()(ancilla, q)
            for idx, q in enumerate(qubits)
        ], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


    def cb_gate(self, ancilla, qubits):
        '''
        for q in qubits:
            self.circuit_.append(
                cirq.H.controlled()(ancilla, q), 
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )
        '''
        self.circuit_.append(
            cirq.MatrixGate(self.U_).controlled()(ancilla, *qubits),
            strategy=cirq.InsertStrategy.NEW_THEN_INLINE
        )


    def hadamard_test_(self, V, alpha, qubits, ancilla):
        '''
        '''
        self.circuit_.append(cirq.H(ancilla))

        self.v_ansatz_(qubits, alpha)

        for q_idx, pauli in enumerate(V[0]):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        for q_idx, pauli in enumerate(V[1]):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        self.circuit_.append(
            cirq.H(ancilla),
            strategy=cirq.InsertStrategy.NEW_THEN_INLINE
        )


    def special_hadamard_test(self, V, alpha, qubits, ancilla, reg):
        self.circuit_.append(cirq.H(ancilla))

        self.cv_ansatz_(qubits, alpha, ancilla, reg)

        for q_idx, pauli in enumerate(V):
            self.circuit_.append(
                CONTROLLED_PAULIS[pauli](ancilla, qubits[q_idx]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

        self.cb_gate(ancilla, qubits)

        self.circuit_.append(cirq.H(ancilla))


    def C(self, alpha):
        '''
        C(alpha) as defined in the paper
        '''
        
        alpha = np.reshape(alpha, (-1, 3))

        den = 0.0
        for i, j in itertools.product(range(len(self.V_)), repeat=2):
            self.circuit_ = cirq.Circuit()
            self.simulator_ = cirq.Simulator()
            self.init_qubits_(5)

            expanded_coeffs = self.coeffs_[i] * self.coeffs_[j]

            self.hadamard_test_(
                [self.V_[i], self.V_[j]],
                alpha,
                self.qubits_[1:4],
                self.qubits_[0]
            )
            self.circuit_.append(
                cirq.measure(self.qubits_[0], key="q0"),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

            results = self.simulator_.run(self.circuit_, 
                                          repetitions=self.shots_)
            counts = results.histogram(key='q0')

            circuit_result = 0.0
            if 1 in counts:
                circuit_result = float(counts[1]) / self.shots_ / 10.0
            
            den += expanded_coeffs * (1 - 2*circuit_result)

        num = 0
        for i, j in itertools.product(range(len(self.V_)), repeat=2):
            expanded_coeffs = self.coeffs_[i] * self.coeffs_[j]
            beta = 1.0

            for step in range(2):
                self.circuit_ = cirq.Circuit()
                self.simulator_ = cirq.Simulator()
                self.init_qubits_(5)

                if step == 0:
                    self.special_hadamard_test(
                        self.V_[i], alpha, self.qubits_[1:4], 
                        self.qubits_[0], self.qubits_
                    )
                elif step == 1:
                    self.special_hadamard_test(
                        self.V_[j], alpha, self.qubits_[1:4], 
                        self.qubits_[0], self.qubits_
                    )
                
                self.circuit_.append(
                    cirq.measure(self.qubits_[0], key="q0"),
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

                results = self.simulator_.run(self.circuit_, 
                                              repetitions=self.shots_)
                counts = results.histogram(key='q0')

                circuit_result = 0.0
                if 1 in counts:
                    circuit_result = float(counts[1]) / self.shots_ / 10.0
                beta *= 1 - 2*circuit_result

            num += beta * expanded_coeffs
    
        C_alpha = 1.0 - float(num)/den
        print('cost: {}'.format(C_alpha))
        return C_alpha


    def compute_v_of_alpha(self, alpha):
        ''' Compute V(alpha) as using the ansatz. When called with alpha_star
            this will return |x> = x/||x||
        '''
        self.circuit_ = cirq.Circuit()
        self.simulator_ = cirq.Simulator()
        self.init_qubits_(3)

        self.v_ansatz_(self.qubits_, alpha)

        results = self.simulator_.simulate(self.circuit_)
        sv = results.state_vector()
        print(results)

        self.circuit_ = cirq.Circuit()
        self.simulator_ = cirq.Simulator()
        self.init_qubits_(3)

        self.v_ansatz_(self.qubits_, alpha)
        self.circuit_.append(
            cirq.measure(*self.qubits_, key='output'), 
            strategy=cirq.InsertStrategy.NEW_THEN_INLINE
        )

        results = self.simulator_.run(self.circuit_, repetitions=self.shots_)
        #print(results.histogram(key='output'))

        hist = results.histogram(key='output')
        total = sum(hist.values())

        bases = []
        for x in sorted(hist):
            phase = np.sign(hist[x]) * np.sqrt( float(hist[x])/total )
            bases.append('{}|{:b}>'.format(phase, x))

        print(' + '.join(bases))

        return sv

