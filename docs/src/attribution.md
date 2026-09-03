# Attribution and license

PureOSQP.jl is a derivative of [OSQP](https://osqp.org) and is released under the same Apache-2.0 license.

## Derivation

This is not a clean-room implementation. The code was written using the OSQP paper and the reference C implementation for details like equilibration, $\rho$ updates, and termination thresholds. OSQP's C unit tests were also ported.

## License

**Apache-2.0**, matching upstream. This choice ensures compliance with the Apache-2.0 license and provides patent protection.

## Papers

The algorithm is based on these papers:

**Main algorithm:** [OSQP: an operator splitting solver for quadratic programs](https://doi.org/10.1007/s12532-020-00179-2) (Stellato et al., 2020).

**Infeasibility detection:** [Infeasibility detection in the alternating direction method of multipliers for convex optimization](https://doi.org/10.1007/s10957-019-01575-y) (Banjac et al., 2019).

**Equilibration:** [A scaling algorithm to equilibrate both rows and columns norms in matrices](https://ral.ac.uk/Publications/RAL-TR-2001-034.pdf) (Ruiz, 2001).

If you use PureOSQP in published work, please cite the OSQP papers.

## Upstream

- Website: <https://osqp.org>
- Source: <https://github.com/osqp/osqp> (Apache-2.0)
- Copyright: OSQP authors.
- C library: Bartolomeo Stellato, Goran Banjac, and Paul Goulart.

**[OSQP.jl](https://github.com/osqp/OSQP.jl)** (Twan Koolen, Benoît Legat, and Bartolomeo Stellato) is the reference implementation used to validate PureOSQP.

## Key differences from upstream

- The inner KKT system is reduced to an $n \times n$ positive definite system.
- The factored matrix is inverted in place for the dense case, making solves faster.
- Equilibration is stored as factors and applied lazily.
- $\rho$ adapts on a fixed iteration interval.
- The duality-gap termination test is enabled by default, following libosqp 1.x.
- Solution derivatives can be computed via implicit differentiation.
- There is no code generation.
