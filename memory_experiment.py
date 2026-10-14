"""Logical error rate per surface-code cycle of the stitched code and of two independent surface codes under SI1000 noise,
with the simulation pipeline of the paper: compilation to CZ gates with StimFlow, SI1000 noise, decoding with PyMatching
through sinter.

    python memory_experiment.py --t 3 --rv 2 --p 1e-3
"""
import argparse
import math
import os

import sinter
import stim
import stimflow

from stitched_surface_code import build_circuit


def cz_compiled(circuit: stim.Circuit) -> stim.Circuit:
    return stimflow.transpile_to_z_basis_interaction_circuit(circuit)


def with_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    return stimflow.NoiseModel.si1000(p).noisy_circuit(circuit)


def stitched_task(t: int, rv: int, p: float) -> sinter.Task:
    circuit = with_noise(cz_compiled(build_circuit(t, rv, 1, memory="ZZ")), p)
    return sinter.Task(circuit=circuit, json_metadata=dict(code="stitched", d=2 * t, rv=rv, p=p, rounds=3 * rv + 4))


def surface_task(d: int, p: float) -> sinter.Task:
    """Stim's rotated surface-code memory without its initial reset layer, so that, as in the stitched circuit, every
    qubit starts in a noiseless |0>."""
    compiled = cz_compiled(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=3 * d))
    circuit, started = stim.Circuit(), False
    for op in compiled.flattened():
        if not started:
            if op.name == "TICK":
                started = True
            elif op.name not in ("R", "RX"):
                circuit.append(op)
            continue
        circuit.append(op)
    return sinter.Task(circuit=with_noise(circuit, p), json_metadata=dict(code="two surface codes", d=d, p=p, rounds=3 * d))


def mu_tot_stitched(p_f0: float, p_f1: float, p_any: float, tau: float) -> float:
    """Total logical error rate per cycle of the four-state Markov chain of the two logical qubits (Appendix B of the
    paper), from the failure probabilities of L0, of L1 and of either after tau cycles, with an ideal initial state.
    The polarizations 1 - 2 p decay as exp(-2 mu tau); mu_tot = (mu_0 + mu_1 + mu_parity) / 2."""
    p_11 = p_f0 + p_f1 - p_any                       # both logical qubits wrong
    p_parity = p_f0 + p_f1 - 2 * p_11                # exactly one wrong
    rates = [-math.log(max(1 - 2 * x, 1e-300)) / (2 * tau) for x in (p_f0, p_f1, p_parity)]
    return sum(rates) / 2


def mu_tot_two_surface(p_f: float, tau: float) -> float:
    """Two independent surface codes, each with failure probability p_f after tau cycles."""
    return -math.log(max(1 - 2 * p_f, 1e-300)) / tau


def per_cycle(stat: sinter.TaskStats) -> float:
    n, tau = stat.shots, stat.json_metadata["rounds"]
    if stat.json_metadata["code"] != "stitched":
        return mu_tot_two_surface(stat.errors / n, tau)
    c = stat.custom_counts
    e0 = c.get("obs_mistake_mask=E_", 0) + c.get("obs_mistake_mask=EE", 0)
    e1 = c.get("obs_mistake_mask=_E", 0) + c.get("obs_mistake_mask=EE", 0)
    return mu_tot_stitched(e0 / n, e1 / n, stat.errors / n, tau)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t", type=int, default=3, help="patch distance; the stitched code has distance d = 2t")
    parser.add_argument("--rv", type=int, default=2, help="GHZ verification rounds per corridor")
    parser.add_argument("--p", type=float, default=1e-3, help="SI1000 physical error rate")
    parser.add_argument("--errors", type=int, default=100, help="stop each point after this many logical errors")
    parser.add_argument("--shots", type=int, default=10**7, help="or after this many shots")
    args = parser.parse_args()
    tasks = [stitched_task(args.t, args.rv, args.p), surface_task(2 * args.t - 1, args.p)]
    stats = sinter.collect(num_workers=os.cpu_count(), tasks=tasks, decoders=["pymatching"],
                           max_errors=args.errors, max_shots=args.shots, count_observable_error_combos=True)
    for s in sorted(stats, key=lambda s: s.json_metadata["code"], reverse=True):
        m = s.json_metadata
        print(f"{m['code']:>18}  d={m['d']:<3} p={m['p']:g}  shots={s.shots:<10} errors={s.errors:<6} "
              f"mu_tot per cycle={per_cycle(s):.2e}")


if __name__ == "__main__":
    main()
