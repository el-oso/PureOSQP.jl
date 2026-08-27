# Attribution and license

PureOSQP.jl implements the algorithm of [OSQP](https://osqp.org). It is a **derivative
work**, not an independent invention, and it is released under the same license as
upstream.

## What the derivation relationship actually is

This is **not a clean-room implementation**, and it is not described as one anywhere in
this package. The code was written against the OSQP paper *and against the reference C
implementation*, which was read directly for the details the paper leaves out: the
modified Ruiz equilibration sweep, the `ρ` update rule and its equality/inequality split,
the active-set guess and acceptance rule in polishing, and the exact thresholds in the
infeasibility certificates. OSQP's own C unit tests were ported as well.

Clean room means reimplementing from a specification *without* access to the source,
usually with a reader/writer separation, and it is a technique for reimplementing
something you are **not** licensed to copy. OSQP is Apache-2.0 licensed, so that
precaution is unnecessary here: the license grants the right to create derivative works,
and asks for attribution in return. Claiming clean room would add no protection and would
not be true.

## License

**Apache-2.0**, matching upstream.

That choice is deliberate rather than a default. Apache-2.0 §4 requires a derivative work
to carry the license, retain attribution notices, and state that files were changed;
relicensing the derived parts as MIT would drop those terms rather than satisfy them, and
MIT also has no patent grant, so it would be a strictly weaker instrument for a work
derived from a patent-granting one. Keeping the upstream license is the simple, compliant
choice and costs users nothing — Apache-2.0 is permissive, OSI-approved, and already
common in the Julia ecosystem.

(This is engineering reasoning, not legal advice. If the licensing matters commercially,
have a lawyer look at it.)

## Papers

The algorithm, and the parts of it this package implements:

**Main algorithm.** Derivation, the ADMM iteration, equilibration, adaptive `ρ`, polishing
and the termination criteria:

```bibtex
@article{osqp,
  author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
  title   = {{OSQP}: an operator splitting solver for quadratic programs},
  journal = {Mathematical Programming Computation},
  volume  = {12},
  number  = {4},
  pages   = {637--672},
  year    = {2020},
  doi     = {10.1007/s12532-020-00179-2},
}
```

**Infeasibility detection.** The proofs behind the primal and dual infeasibility
certificates that PureOSQP returns:

```bibtex
@article{osqp-infeasibility,
  author  = {Banjac, G. and Goulart, P. and Stellato, B. and Boyd, S.},
  title   = {Infeasibility detection in the alternating direction method of multipliers
             for convex optimization},
  journal = {Journal of Optimization Theory and Applications},
  volume  = {183},
  number  = {2},
  pages   = {490--519},
  year    = {2019},
  doi     = {10.1007/s10957-019-01575-y},
}
```

**Equilibration.** The scaling scheme, which OSQP applies in modified form to the KKT
matrix:

```bibtex
@techreport{ruiz2001,
  author      = {Ruiz, D.},
  title       = {A scaling algorithm to equilibrate both rows and columns norms in matrices},
  institution = {Rutherford Appleton Laboratory},
  number      = {RAL-TR-2001-034},
  year        = {2001},
}
```

If you use PureOSQP in published work, cite the OSQP papers — the algorithm is theirs.

Two further OSQP papers cover functionality this package does **not** implement, listed so
the record is complete: Schubiger, Banjac and Lygeros (2020) on GPU acceleration and the
PCG linear-system method, and Banjac, Stellato, Moehle, Goulart, Bemporad and Boyd (2017)
on embedded code generation.

## Upstream

- Website: <https://osqp.org>
- Source: <https://github.com/osqp/osqp> (Apache-2.0)
- Copyright the OSQP authors.

## What is deliberately different

These are changes from upstream, not omissions, and Apache-2.0 §4(b) asks that they be
stated:

- the inner KKT system is eliminated to an `n×n` positive definite system and factored with
  a dense Cholesky, rather than factoring the `(n+m)×(n+m)` quasi-definite system with a
  sparse LDLᵀ;
- equilibration is stored as factors and applied lazily, so the caller's `P` and `A` are
  never copied or modified;
- `ρ` adapts on a fixed iteration interval rather than on a fraction of wall-clock setup
  time, so iteration counts do not depend on machine speed;
- where libosqp 0.6.2 and later versions differ, the 0.6.2 behavior is implemented — the
  primal-infeasibility threshold `uᵀmax(δy,0) + lᵀmin(δy,0) < ε‖δy‖` and the
  dual-infeasibility threshold `qᵀδx < c·ε·‖δx‖`, both of which master tightened to `< 0`;
- there is no sparse linear algebra, no code generation, no GPU backend, and no
  duality-gap termination check.
