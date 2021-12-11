Variation Quantum Linear Solver
-------------------------------
CMSC 657 Final Project -- Daniel Nichols

This directory contains the code to simulate VQLS. To run the code you need
python3, numpy, scipy, and cirq. I installed these with:

```sh
python3 -m venv cirq-venv
source ./cirq-venv/bin/activate
pip install numpy scipy cirq
```

Then running `main.py` will run VQLS. `python main.py --help` displays the
available options. These are:

```
optional arguments:
  -h, --help            show this help message and exit
  -s SHOTS, --shots SHOTS
                        Number of times to collect measurements for each circuit.
  --seed SEED           Random number seed. Helpful for reproducibility.
  --random-input [RANDOM_INPUT]
                        If provided, then use a random matrix A. Optionally takes value N to generate A as NxN matrix.
  --sample              Determine values through many measurements instead of using simulator provided values.
  --noise               simulate the circuit with noise.
  --optimizer {Nelder-Mead,Powell,CG,BFGS,Newton-CG,L-BFGS-B,TNC,COBYLA,SLSQP,trust-constr,dogleg,trust-ncg,trust-exact,trust-krylov}
                        Available optimization algorithms for C(alpha)
  --max-iter MAX_ITER   Maximum number of iterations for optimizer.
  --ansatz-layers ANSATZ_LAYERS
                        Number of layers to use in fixed ansatz for V(alpha).
```