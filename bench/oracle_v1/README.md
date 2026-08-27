# libosqp 1.x oracle

`OSQP.jl` wraps libosqp **0.6.2**; there is no Julia wrapper for 1.x, and the 1.x
`OSQPSettings` struct is `__attribute__((packed))` with a build-dependent integer and float
width, so mirroring it for `ccall` is not safe to do by hand.

`osqp_v1_oracle.py` therefore drives the `osqp` Python wheel as a **subprocess** speaking
JSON on stdin/stdout. There is no in-process interop and PureOSQP itself has no Python
dependency — this directory is used only by `bench/headtohead_v1.jl`.

Set up:

    python3 -m venv .venv && ./.venv/bin/pip install osqp numpy scipy

Then point `PUREOSQP_PY` at that interpreter, or at any interpreter with `osqp` installed:

    PUREOSQP_PY=./.venv/bin/python3 julia --project=bench bench/headtohead_v1.jl

The comparison is skipped, not failed, when the interpreter or the module is absent.
