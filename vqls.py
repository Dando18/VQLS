'''
Full VQLS implementation.
'''
import itertools
import cirq
import numpy as np
from scipy.optimize import minimize


class VQLS:

    def __init__(self, coeffs, V):
        self.coeffs_ = coeffs
        self.V_ = V

        self.qubits = []
        

    def init_qubits_(self, n_qubits):
        self.qubits_ = [
            cirq.NamedQubit('q{}'.format(idx))
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
        for q in qubits:
            self.circuit_.append(cirq.H.controlled()(ancilla, q), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


    def hadamard_test_(self, V, alpha, qubits, ancilla):
        '''
        '''
        self.circuit_.append(cirq.H(ancilla))

        self.v_ansatz_(qubits, alpha)

        for q_idx, flag in enumerate(V[0]):
            if flag == 1:
                self.circuit_.append(
                    cirq.CZ(ancilla, qubits[q_idx]),
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

        for q_idx, flag in enumerate(V[1]):
            if flag == 1:
                self.circuit_.append(
                    cirq.CZ(ancilla, qubits[q_idx]),
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

        self.circuit_.append(cirq.H(ancilla), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


    def special_hadamard_test(self, V, alpha, qubits, ancilla, reg):
        self.circuit_.append(cirq.H(ancilla))

        self.cv_ansatz_(qubits, alpha, ancilla, reg)

        for q_idx, flag in enumerate(V):
            if flag == 1:
                self.circuit_.append(
                        cirq.CZ(ancilla, qubits[q_idx]),
                        strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                    )

        self.cb_gate(ancilla, qubits)

        self.circuit_.append(cirq.H(ancilla))


    def C(self, alpha, shots=10000):
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

            results = self.simulator_.run(self.circuit_, repetitions=shots)
            counts = results.histogram(key='q0')

            circuit_result = 0.0
            if 1 in counts:
                circuit_result = float(counts[1]) / shots
            
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
                    self.special_hadamard_test(self.V_[i], alpha, self.qubits_[1:4], self.qubits_[0], self.qubits_)
                elif step == 1:
                    self.special_hadamard_test(self.V_[j], alpha, self.qubits_[1:4], self.qubits_[0], self.qubits_)
                
                self.circuit_.append(
                    cirq.measure(self.qubits_[0], key="q0"),
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

                results = self.simulator_.run(self.circuit_, repetitions=shots)
                counts = results.histogram(key='q0')

                circuit_result = 0.0
                if 1 in counts:
                    circuit_result = float(counts[1]) / shots
                beta *= 1 - 2*circuit_result

            num += beta * expanded_coeffs
    
        C_alpha = 1.0 - float(num/den)
        print('cost: {}'.format(C_alpha))
        return C_alpha





def main():
    vqls = VQLS([0.55, 0.45], [[0, 0, 0], [0, 0, 1]])

    x0 = [float(np.random.randint(0,3000))/1000 for i in range(0, 9)]
    result = minimize(vqls.C, x0=x0, method="COBYLA", options={'maxiter': 200})
    print(result)


if __name__ == '__main__':
    main()
