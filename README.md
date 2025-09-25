# Gate-by-Gate Tensor-Network Sampler (Algorithm 2)

A lightweight implementation of **gate-level sampling** with **quimb + cotengra**.
At step $t$, we build the **prefix circuit** $U_t\cdots U_1\ket{0^n}$, clamp all qubits **outside** the current gate’s support $B$ to the partially generated bitstring $x_A$, leave the $k=|B|$ outputs **open**, contract **once** to get the length-$2^k$ vector $v_t$, then sample $b\in\{0,1\}^k$ from $|v_t|^2$. Repeat over gates to produce exact samples from $|\langle x|U|0^n\rangle|^2$.

This repository **implements Algorithm 2** from:

> **How to simulate quantum measurement without computing marginals** (arXiv:2112.08499).
> We follow the paper’s gate-by-gate sampling scheme, including diagonal-gate skipping.

---

## Features

* **Algorithm 2 (gate-by-gate) sampler** true to the paper.
* **Diagonal-gate skipping** (Z/RZ/PHASE/CZ/other computational-basis diagonal gates).
* **Prefix TN caching** across shots (build $U_t\cdots U_1\ket{0^n}$ once per $t$).
* **Contraction path reuse** with `cotengra.ReusableHyperOptimizer`, with safe fallbacks to `"auto-hq"` / `"greedy"`.
* **OpenQASM 2.0 import** (multi-qreg supported; wires mapped to **global indices**).
* Version-robust contraction (handles older/newer `quimb` where `get='array'` may not exist).

---

## Installation

Tested with Python 3.9+.

```bash
# Core dependencies
pip install numpy quimb cotengra autoray

# Recommended: hypergraph partitioner used by cotengra
pip install kahypar
```

If `pip install kahypar` fails on your platform:

```bash
# Conda alternative
conda install -c conda-forge kahypar
```

You can still run without `kahypar` (cotengra uses other heuristics), but performance may be lower—start with `--opt auto-hq` and switch to `--opt hyper` later.

**Optional backends:** install `cupy` or `torch` to let `autoray` route contractions to GPU.

---

## Files

* `tn_gate_sampler_adv.py` — production script (Algorithm 2 + diagonal skipping + prefix caching + QASM + robust optimizer fallbacks).
* `tn_gate_sampler.py` — minimal, didactic baseline.

---

## Quick Start

### Demo circuit

```bash
python tn_gate_sampler_adv.py --n 5 --shots 16 --seed 1 --verbose
```

### From OpenQASM (multi-qreg OK)

```bash
python tn_gate_sampler_adv.py --qasm path/to/circuit.qasm --shots 1024 --opt auto-hq --verbose
```

Supported QASM subset:
`h, x, y, z, rx(theta), ry(theta), rz(theta), cx/cnot, cz, swap, phase/p(theta)`

Example:

```qasm
OPENQASM 2.0;
qreg q[3];
qreg anc[2];
h q[0];
cx q[0],q[1];
rz(pi/4) q[0];
cz q[1],anc[1];
```

The loader maps all qregs to a single global wire indexing.

---

## Command-line Options

* `--n INT` – number of qubits (ignored with `--qasm`)
* `--qasm PATH` – load a QASM 2.0 file
* `--shots INT` – number of samples
* `--seed INT` – RNG seed
* `--verbose` – per-gate logs (includes diagonal-gate skips)
* `--no-skip-diagonal` – disable diagonal-gate skipping
* `--opt {hyper,auto-hq,greedy}` – contraction strategy

  * Start with `auto-hq` to validate; use `hyper` for faster repeated sampling.

---

## Mapping to the Paper (Algorithm 2)

* **Prefix distributions:** $P_t(x)=|\langle x|U_t\cdots U_1|0^n\rangle|^2$. After step $t$, the sampler’s distribution equals $P_t$; final output follows $P_m$.
* **One-shot vector amplitude on $B$:** open only the $k$ legs on $B$, clamp the rest to $x_A$, contract once to get $v_t\in\mathbb{C}^{2^k}$, sample $b\sim |v_t|^2$, and write back to $x_B$.
* **Work bound:** at most $2^k$ probabilities per step; overall **$\le m2^k$** queries (with gate arity $k$ usually 1–2).
* **Diagonal-gate skipping:** if $U_t$ is diagonal (Z/RZ/CZ…), then $P_t=P_{t-1}$; skip the step (no contraction).

---

## Performance Tips

* Validate with `--opt auto-hq`; then try `--opt hyper` for speed (path reuse).
* Circuits with many diagonal gates benefit more from skipping.
* Prefix caching shines when `--shots` is large.
* For bigger circuits, consider slicing and GPU backends (`autoray` + `cupy`/`torch`).

---

## Troubleshooting

* **`KeyError: 'tree'` (cotengra hyperoptimizer):** use `--opt auto-hq`. The script also resets the optimizer and falls back to `auto-hq`/`greedy`.
* **`Option 'get' ... but got 'array'` (older quimb):** the script auto-detects and calls `contract` without `get`, then unwraps `Tensor.data`.
* **`cannot reshape array of size 1 into shape (2,)`:** typically the open index set collapsed. The script now validates open indices before contraction and guards empty $B$. Use `--verbose` to locate the step $t$ and support $B$.

---

## Cite

If you use this code, please cite:

> *How to simulate quantum measurement without computing marginals.*
> **Algorithm 2** is implemented here in gate-by-gate form (prefix amplitudes, diagonal-gate skipping, and at-most $m2^k$ probability queries).
> arXiv:2112.08499

(Add the full BibTeX from arXiv in your repo.)

---

## License

Choose what fits your project (e.g., MIT or Apache-2.0).

---

## Acknowledgements

* [quimb](https://github.com/jcmgray/quimb) and [cotengra](https://github.com/jcmgray/cotengra) for the tensor-network stack and contraction planning.
* `kahypar` for hypergraph partitioning used by cotengra’s pathfinder.
