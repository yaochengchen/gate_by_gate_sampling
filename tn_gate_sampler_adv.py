#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced gate-level sampler (Algorithm 2) with Tensor Networks (quimb + cotengra).
"""

from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg


# --------- Robust contraction wrapper ---------
def _safe_contract(tn, out_inds, optimizer):
    """
    Contract TN keeping out_inds open.
    - Try get='array' (newer quimb)
    - If 'get' is unsupported (older quimb), call without it
    - Always return a NumPy ndarray
    """
    def _call(opt, use_get):
        if use_get:
            return tn.contract(output_inds=out_inds, optimize=opt, get='array')
        else:
            return tn.contract(output_inds=out_inds, optimize=opt)

    # 1) Try provided optimizer
    for use_get in (True, False):
        try:
            arr = _call(optimizer, use_get)
            break
        except TypeError:
            continue  # 'get' unexpected kw
        except ValueError as e:
            if 'Option' in str(e) and 'get' in str(e):
                continue  # 'get' not allowed in this version
            else:
                # other value errors -> try reset
                try:
                    if hasattr(optimizer, 'reset'): optimizer.reset()
                    arr = _call(optimizer, use_get)
                    break
                except Exception:
                    pass
        except Exception:
            try:
                if hasattr(optimizer, 'reset'): optimizer.reset()
                arr = _call(optimizer, use_get)
                break
            except Exception:
                pass
    else:
        # 2) Fallback to 'auto-hq'
        for use_get in (True, False):
            try:
                arr = _call('auto-hq', use_get)
                break
            except Exception:
                continue
        else:
            # 3) Fallback to 'greedy'
            for use_get in (True, False):
                try:
                    arr = _call('greedy', use_get)
                    break
                except Exception:
                    continue

    # Ensure ndarray
    from quimb.tensor import Tensor as _QTN_Tensor
    if isinstance(arr, _QTN_Tensor):
        arr = arr.transpose(*out_inds).data
    return np.asarray(arr)


@dataclass
class Gate:
    name: str
    qubits: Tuple[int, ...]
    theta: Optional[float] = None
    unitary: Optional[np.ndarray] = None

    @property
    def k(self) -> int:
        return len(self.qubits)


def gate_matrix(g: Gate) -> np.ndarray:
    """Return matrix for gate g (explicit 2q matrices for portability)."""
    if g.unitary is not None:
        U = np.asarray(g.unitary, dtype=complex)
        assert U.ndim == 2 and U.shape[0] == U.shape[1] and (U.shape[0] & (U.shape[0] - 1)) == 0
        return U
    nm = g.name.upper()
    # 1-qubit
    if nm == "H": return qu.hadamard()
    if nm == "X": return qu.pauli('X')
    if nm == "Y": return qu.pauli('Y')
    if nm == "Z": return qu.pauli('Z')
    if nm == "RX":
        assert g.theta is not None; return qu.Rx(g.theta)
    if nm == "RY":
        assert g.theta is not None; return qu.Ry(g.theta)
    if nm == "RZ":
        assert g.theta is not None; return qu.Rz(g.theta)
    if nm == "PHASE":
        assert g.theta is not None; return np.diag([1.0, np.exp(1j * g.theta)]).astype(complex)
    # 2-qubit explicit
    if nm in ("CNOT", "CX"):
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    if nm == "CZ":
        return np.diag([1,1,1,-1]).astype(complex)
    if nm == "SWAP":
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    raise ValueError(f"Unknown gate: {g.name}")


def is_diagonal_unitary(U: np.ndarray, tol: float = 1e-12) -> bool:
    U = np.asarray(U); off = U.copy(); np.fill_diagonal(off, 0.0)
    return np.max(np.abs(off)) <= tol


def is_diagonal_gate(g: Gate) -> bool:
    try:
        U = gate_matrix(g)
    except Exception:
        return False
    return is_diagonal_unitary(U)


_QASM_HEADER_RE = re.compile(r"^\s*OPENQASM\s+2\.0\s*;\s*$", re.IGNORECASE)
_QREG_RE = re.compile(r"^\s*qreg\s+([a-zA-Z_][\w]*)\s*\[\s*(\d+)\s*\]\s*;\s*$", re.IGNORECASE)

def _eval_theta(expr: str) -> float:
    expr = expr.strip().replace("PI", "pi").replace("Pi", "pi")
    allowed = set("0123456789.+-*/() pi")
    if not set(expr) <= allowed:
        raise ValueError(f"Unsupported characters in angle: {expr}")
    expr_py = re.sub(r"\bpi\b", str(math.pi), expr)
    return float(eval(expr_py, {"__builtins__": {}}, {}))


def load_gates_from_qasm(path: str) -> Tuple[int, List[Gate]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("//")]

    if not _QASM_HEADER_RE.match(lines[0]):
        raise ValueError("QASM must start with 'OPENQASM 2.0;'")

    # collect all qregs and assign global offsets
    qregs: Dict[str, int] = {}
    offsets: Dict[str, int] = {}
    total = 0
    for ln in lines[1:]:
        m = _QREG_RE.match(ln)
        if m:
            name = m.group(1); size = int(m.group(2))
            if name in qregs:
                raise ValueError(f"Duplicate qreg: {name}")
            qregs[name] = size
    for name, size in qregs.items():
        offsets[name] = total; total += size
    if total == 0:
        raise ValueError("No qreg found.")

    def _wire_to_global(w: str) -> int:
        m = re.match(r"^([A-Za-z_]\w*)\[(\d+)\]$", w)
        if not m:
            raise ValueError(f"Unsupported wire spec: {w}")
        nm, idx = m.group(1), int(m.group(2))
        if nm not in qregs or idx >= qregs[nm]:
            raise ValueError(f"Wire out of range: {w}")
        return offsets[nm] + idx

    gates: List[Gate] = []
    name_map = {
        "h": "H", "x": "X", "y": "Y", "z": "Z",
        "rx": "RX", "ry": "RY", "rz": "RZ",
        "cx": "CNOT", "cnot": "CNOT", "cz": "CZ", "swap": "SWAP",
        "phase": "PHASE", "p": "PHASE",
    }

    for ln in lines[1:]:
        if _QREG_RE.match(ln): continue
        m = re.match(r"^([A-Za-z_]\w*)\s*(\(([^)]*)\))?\s+(.+);\s*$", ln)
        if not m: continue
        gname = m.group(1).lower(); argstr = m.group(3)
        wires = [w for w in m.group(4).replace(" ", "").split(",") if w]
        qubits = tuple(_wire_to_global(w) for w in wires)
        theta = None
        if argstr is not None: theta = _eval_theta(argstr)
        if gname not in name_map: continue
        gates.append(Gate(name_map[gname], qubits, theta=theta))

    return total, gates


class PrefixNetworkCache:
    def __init__(self, n: int, gates: Sequence[Gate]):
        self.n = n; self.gates = list(gates); self.m = len(gates)
        self.prefix_tns: List[qtn.TensorNetwork] = []
        self._build_all_prefixes()

    def _build_all_prefixes(self):
        n = self.n
        tn_prev = qtn.TensorNetwork([])
        for q in range(n):
            data = np.array([1.0, 0.0], dtype=complex)
            tn_prev.add_tensor(qtn.Tensor(data=data, inds=(f"v0_{q}",), tags={f"IN{q}"}))
        for s, g in enumerate(self.gates, start=1):
            tn = tn_prev.copy()
            in_inds = {q: f"v{s-1}_{q}" for q in range(n)}
            out_inds = {q: f"v{s}_{q}" for q in range(n)}
            U = gate_matrix(g).reshape([2] * (2 * g.k))
            B = tuple(sorted(g.qubits))
            gate_inds = tuple(out_inds[q] for q in B) + tuple(in_inds[q] for q in B)
            tn.add_tensor(qtn.Tensor(data=U, inds=gate_inds, tags={f"G{s}", f"B{B}"}))
            for q in range(n):
                if q in B: continue
                I = np.eye(2, dtype=complex).reshape(2, 2)
                tn.add_tensor(qtn.Tensor(data=I, inds=(out_inds[q], in_inds[q]), tags={f"ID{s}", f"Q{q}"}))
            self.prefix_tns.append(tn)
            tn_prev = tn

    def get(self, t: int) -> qtn.TensorNetwork:
        assert 1 <= t <= self.m
        return self.prefix_tns[t - 1]


def compute_vt_from_prefix(tn_prefix: qtn.TensorNetwork, n: int, t: int, B: Tuple[int, ...], x: np.ndarray,
                           optimizer: Optional[ctg.ReusableHyperOptimizer] = None) -> np.ndarray:
    B = tuple(sorted(B))
    if len(B) == 0:
        return np.array([1.0 + 0.0j])
    out_inds = {q: f"v{t}_{q}" for q in range(n)}
    tn = tn_prefix.copy()
    A = [q for q in range(n) if q not in B]
    proj0 = np.array([1.0, 0.0], dtype=complex); proj1 = np.array([0.0, 1.0], dtype=complex)
    for q in A:
        bit = int(x[q]); proj = proj0 if bit == 0 else proj1
        tn.add_tensor(qtn.Tensor(data=proj, inds=(out_inds[q],), tags={f"CLAMP{q}"}))
    out_B = [out_inds[q] for q in B]
    # validate open inds exist
    missing = [ix for ix in out_B if ix not in tn.ind_map]
    if missing:
        raise ValueError(f"Open indices {missing} not found in prefix TN at step t={t}.")
    arr = _safe_contract(tn, out_B, optimizer if optimizer is not None else "auto-hq")
    v_t = np.asarray(arr).reshape(2 ** len(B)).astype(complex)
    return v_t


class GateLevelSamplerTN:
    def __init__(self, n: int, gates: Sequence[Gate], seed: Optional[int] = None, skip_diagonal: bool = True):
        self.n = n; self.gates = list(gates); self.m = len(gates)
        self.skip_diagonal = skip_diagonal
        self.rng = np.random.default_rng(seed)
        self.prefix_cache = PrefixNetworkCache(n, gates)
        self.optimizer = ctg.ReusableHyperOptimizer(progbar=False, minimize="flops", max_repeats=64)

    def sample_one(self, verbose: bool = False) -> np.ndarray:
        x = np.zeros(self.n, dtype=int)
        for t, g in enumerate(self.gates, start=1):
            if self.skip_diagonal and is_diagonal_gate(g):
                if verbose: print(f"[t={t:02d}] Skip diagonal {g.name} on {g.qubits} | x={x}")
                continue
            B = tuple(g.qubits)
            tn_prefix = self.prefix_cache.get(t)
            v_t = compute_vt_from_prefix(tn_prefix, self.n, t, B, x, optimizer=self.optimizer)
            probs = np.abs(v_t) ** 2; s = probs.sum()
            if s <= 0: raise RuntimeError("Zero norm encountered forming probabilities.")
            probs = probs / s
            idx = self.rng.choice(len(probs), p=probs)
            k = len(B); bits = [(idx >> i) & 1 for i in range(k)][::-1]
            for qb, bit in zip(B, bits): x[qb] = bit
            if verbose:
                pretty = np.round(probs, 6).tolist() if k <= 3 else f"{len(probs)}-entry"
                print(f"[t={t:02d}] {g.name} on {B} | probs={pretty} | b={bits} | x={x}")
        return x

    def sample(self, shots: int, verbose: bool = False) -> np.ndarray:
        return np.vstack([self.sample_one(verbose=verbose) for _ in range(shots)])


def main():
    p = argparse.ArgumentParser(description="Advanced gate-level TN sampler (Algorithm 2).")
    p.add_argument("--n", type=int, default=5, help="Number of qubits (ignored if --qasm).")
    p.add_argument("--shots", type=int, default=8, help="Number of samples to generate.")
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument("--verbose", action="store_true", help="Verbose per-step logging.")
    p.add_argument("--no-skip-diagonal", dest="skipdiag", action="store_false", help="Disable diagonal skipping.")
    p.add_argument("--qasm", type=str, default=None, help="OpenQASM 2.0 file path.")
    p.add_argument("--opt", type=str, default="hyper", choices=["hyper", "auto-hq", "greedy"], help="Contraction optimizer.")
    a = p.parse_args()

    if a.qasm:
        n, gates = load_gates_from_qasm(a.qasm)
        print(f"Loaded QASM: n={n}, m={len(gates)} gates from '{a.qasm}'")
    else:
        n = a.n
        gates = [Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("RZ", (0,), theta=math.pi/3),
                 Gate("RX", (2,), theta=0.7) if n >= 3 else None,
                 Gate("CZ", (1, 2)) if n >= 3 else None]
        gates = [g for g in gates if g is not None]
        print(f"Demo circuit: n={n}, m={len(gates)} gates")

    sampler = GateLevelSamplerTN(n, gates, seed=a.seed, skip_diagonal=a.skipdiag)
    if a.opt != "hyper":
        sampler.optimizer = a.opt

    samples = sampler.sample(a.shots, verbose=a.verbose)

    def bits_to_str(b): return "".join(str(int(x)) for x in b)
    print("\nSamples:")
    for i, s in enumerate(samples): print(f"  #{i+1}: {bits_to_str(s)}")
    uniq, counts = np.unique(samples, axis=0, return_counts=True)
    print("\nHistogram:")
    for u, c in zip(uniq, counts): print(f"  {bits_to_str(u)} : {c}")


if __name__ == "__main__":
    main()
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
"""
