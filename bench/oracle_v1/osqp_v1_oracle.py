"""Reference oracle for libosqp 1.x, reached through the `osqp` Python wheel.

Reads one JSON problem per line on stdin, writes one JSON result per line on stdout.
This exists only because libosqp 1.x has no Julia wrapper (OSQP.jl pins 0.6.2) and its
packed settings struct is not safe to mirror by hand. It is a benchmark and validation
oracle; PureOSQP itself has no Python dependency.

Two request shapes:

Single solve
    in : {"P": [[..]], "q": [..], "A": [[..]], "l": [..], "u": [..], "settings": {..}}
    out: {"status": str, "x": [..], "y": [..], "obj_val": float, "iter": int}

Update sequence -- setup once, then update and re-solve, as a receding-horizon loop does
    in : same, plus "sequence": [{"q": [..], "l": [..], "u": [..]}, ...]
    out: {"setup_time": float, "solves": [{"status", "obj_val", "iter",
          "update_time", "solve_time"}, ...]}

Times are measured inside this process with perf_counter, so they exclude interpreter
startup and the JSON round trip and are comparable to timings taken in Julia.
`None` in a bound vector means the corresponding infinity.
"""
import json
import sys
import time

import numpy as np
import osqp
import scipy.sparse as sp

# libosqp 1.x renamed several settings; accept the 0.6.2 spellings and translate.
RENAMES = {"polish": "polishing", "warm_start": "warm_starting"}


def to_l(v):
    return np.array([-np.inf if t is None else t for t in v], dtype=float)


def to_u(v):
    return np.array([np.inf if t is None else t for t in v], dtype=float)


def finite(v):
    return [None if not np.isfinite(t) else float(t) for t in np.asarray(v, dtype=float)]


def build(problem):
    n = len(problem["q"])
    m = len(problem["l"])
    P = sp.csc_matrix(np.array(problem["P"], dtype=float))
    A = sp.csc_matrix(np.array(problem["A"], dtype=float).reshape(m, n))
    settings = {RENAMES.get(k, k): v for k, v in problem.get("settings", {}).items()}
    settings.setdefault("verbose", False)
    model = osqp.OSQP()
    t0 = time.perf_counter()
    model.setup(P=sp.triu(P, format="csc"), q=np.array(problem["q"], dtype=float), A=A,
                l=to_l(problem["l"]), u=to_u(problem["u"]), **settings)
    return model, time.perf_counter() - t0


def solve_once(problem):
    model, _ = build(problem)
    res = model.solve()
    return {
        "status": str(res.info.status),
        "x": finite(res.x),
        "y": finite(res.y),
        "obj_val": None if not np.isfinite(res.info.obj_val) else float(res.info.obj_val),
        "iter": int(res.info.iter),
        "version": osqp.__version__,
    }


def solve_sequence(problem):
    model, setup_time = build(problem)
    out = []
    for step in problem["sequence"]:
        kwargs = {}
        if "q" in step:
            kwargs["q"] = np.array(step["q"], dtype=float)
        if "l" in step:
            kwargs["l"] = to_l(step["l"])
        if "u" in step:
            kwargs["u"] = to_u(step["u"])
        t0 = time.perf_counter()
        model.update(**kwargs)
        t_update = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = model.solve()
        t_solve = time.perf_counter() - t0
        out.append({
            "status": str(res.info.status),
            "obj_val": None if not np.isfinite(res.info.obj_val) else float(res.info.obj_val),
            "iter": int(res.info.iter),
            "update_time": t_update,
            "solve_time": t_solve,
        })
    return {"setup_time": setup_time, "solves": out, "version": osqp.__version__}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            problem = json.loads(line)
            out = solve_sequence(problem) if "sequence" in problem else solve_once(problem)
        except Exception as exc:  # report, never crash the stream
            out = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
