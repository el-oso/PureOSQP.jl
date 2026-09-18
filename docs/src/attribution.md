# Attribution and license

Four packages, two licenses. What each one derives from differs, and so does the license
it carries.

| package | license | derives from |
|---|---|---|
| `PureQPBase` | MIT | published papers only |
| `PureIPM` | MIT | published papers only |
| `PureDAQP` | MIT | a paper and [DAQP](https://github.com/darnstrom/daqp), which is MIT |
| `PureOSQP` | Apache-2.0 | [OSQP](https://osqp.org), and carries its license |

**Depending on a package does not place yours under its license.** Using `PureQPBase` leaves
your package under whatever license you choose, and `PureOSQP` depending on `PureQPBase` does
not place `PureQPBase` under Apache-2.0. Only a derivative work carries its parent's terms,
and a dependency is not a derivative.

Each package directory carries the license governing it — `PureQPBase/LICENSE`,
`PureIPM/LICENSE`, `PureDAQP/LICENSE`, `PureOSQP/LICENSE`. The `LICENSE` at the repository root
is a map to those four rather than a license itself, since claiming either one there would be
wrong for three of the packages. Everything outside the four package directories — the shared
benchmarks, this documentation, the design notes — is MIT.

## PureOSQP

**This is not a clean-room implementation.** The operator-splitting method was written using
the OSQP paper and the reference C implementation, for details such as the `ρ` update
schedule and the order of the iteration. OSQP's C unit tests were ported into the test suite.

It is therefore a derivative work and is released under **Apache-2.0**, matching upstream.
That is not a preference: a derivative carries its parent's terms.

- Website: <https://osqp.org>
- Source: <https://github.com/osqp/osqp> (Apache-2.0)
- Copyright: the OSQP authors.
- C library: Bartolomeo Stellato, Goran Banjac, and Paul Goulart.

[OSQP.jl](https://github.com/osqp/OSQP.jl) (Twan Koolen, Benoît Legat, and Bartolomeo
Stellato) is the independent implementation used to validate it.

## PureQPBase

**MIT.** The core is not a derivative of OSQP. It was written from the published
descriptions of the methods it implements:

**Equilibration:** [A scaling algorithm to equilibrate both rows and columns norms in
matrices](https://ral.ac.uk/Publications/RAL-TR-2001-034.pdf) (Ruiz, 2001).

**Infeasibility certificates:** [Infeasibility detection in the alternating direction method
of multipliers for convex optimization](https://doi.org/10.1007/s10957-019-01575-y) (Banjac
et al., 2019). The certificate tests hold for any direction, whatever produced it, which is
why operator splitting and the interior-point method use them unchanged.

The linear-system backends, the backend selection, the problem representation and the polishing
and derivative kernels are this package's own.

## PureDAQP

**MIT.** The method is Algorithm 1 of [A dual active-set solver for embedded quadratic
programming using recursive LDLᵀ updates](https://doi.org/10.1109/TAC.2022.3176430) (Arnström,
Bemporad and Axehill, 2022), with that paper's Algorithm 2 as the proximal-point outer loop.

**This is not a clean-room implementation.** The authors' reference C implementation was read
alongside the paper, for details the paper leaves to the implementer: the pivoting rule, the
anti-cycling switch, and the way the `LDLᵀ` buffers are laid out.

- Source: <https://github.com/darnstrom/daqp> (MIT)
- Copyright: Daniel Arnström.

[DAQP.jl](https://github.com/darnstrom/DAQP.jl) is the wrapper around that C solver, and is
what the benchmarks and the oracle tests compare against.

## PureIPM

**MIT.** Not a derivative either. The algorithm follows [On the implementation of a
primal-dual interior point method](https://doi.org/10.1137/0802028) (Mehrotra, 1992).

[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) (Paul Goulart and Yuwen Chen) is
the independent interior-point implementation used to validate it: iteration counts and
objectives are compared against it across the benchmark suite. It is a reference for the
answers, not a source for the code.

## Citing

If you use the operator-splitting method in published work, please cite the OSQP papers:

**Main algorithm:** [OSQP: an operator splitting solver for quadratic
programs](https://doi.org/10.1007/s12532-020-00179-2) (Stellato et al., 2020).

For the interior-point method, cite Mehrotra (1992). For the active-set method, cite Arnström,
Bemporad and Axehill (2022).

## Key differences from libosqp

libosqp 1.0 is the reference for `PureOSQP`: its settings, its defaults, and its termination
and certificate tests. These are the differences from it.

- The inner KKT system is reduced to an `n×n` positive definite system.
- The factored matrix is inverted in place for the dense case, making solves faster.
- Equilibration is stored as factors and applied lazily, so `P` and `A` are never copied.
- A backend is chosen from the declared types of `P` and `A`, so a structured matrix is
  solved through its structure rather than through its sparsity pattern.
- `ρ` has no wall-clock adaptation mode. Deciding when to refactorize by reading a clock
  makes the iteration count a property of the machine; the other three modes are all here.
- Solution derivatives can be computed via implicit differentiation, and the element type is
  `Real` rather than a float, so dual numbers run the solver.
- Errors are thrown, carrying their own messages, rather than returned as codes to look up
  with `osqp_error_message`.
- There is no code generation.
