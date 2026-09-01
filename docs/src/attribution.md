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

### The rest of the OSQP bibliography

The [OSQP citing page](https://osqp.org/citing/) lists five papers. The two above are the
ones this package implements; the remaining three cover functionality it does **not**, and
are reproduced here so the record is complete rather than selective.

```bibtex
@article{osqp-gpu,
  author  = {Schubiger, M. and Banjac, G. and Lygeros, J.},
  title   = {{GPU} acceleration of {ADMM} for large-scale quadratic programming},
  journal = {Journal of Parallel and Distributed Computing},
  volume  = {144},
  pages   = {55--67},
  year    = {2020},
  doi     = {10.1016/j.jpdc.2020.05.021},
}

@inproceedings{osqp-codegen,
  author    = {Banjac, G. and Stellato, B. and Moehle, N. and Goulart, P. and
               Bemporad, A. and Boyd, S.},
  title     = {Embedded code generation using the {OSQP} solver},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2017},
  doi       = {10.1109/CDC.2017.8263928},
}

@inproceedings{miosqp,
  author    = {Stellato, B. and Naik, V. V. and Bemporad, A. and Goulart, P. and Boyd, S.},
  title     = {Embedded mixed-integer quadratic optimization using the {OSQP} solver},
  booktitle = {European Control Conference (ECC)},
  year      = {2018},
  doi       = {10.23919/ECC.2018.8550136},
}
```

Embedded code generation and branch-and-bound for mixed-integer QPs are outside this
package's scope. The GPU paper is not: `linsys = :indirect` is the same matrix-free
preconditioned-CG method, and that paper is also the source for what a GPU is worth on this
algorithm — it targets problems with at least 1e4 nonzeros and reaches its peak near 1e8.

## The name Cholesky

The default dense backend is a Cholesky factorization, so the name appears throughout this
manual. It is worth a paragraph, because the pronunciation most often heard in English is the
one variant with no support from any of the languages involved.

**André-Louis Cholesky** (born 15 October 1875 in Montguyon; died 31 August 1918 of wounds
received in northern France) was a French army officer and geodesist who ended as head of the
Topographical Service of Tunisia. He did not publish the method himself. It appeared
posthumously in 1924, when a fellow officer, Commandant Benoît, wrote it up in the
*Bulletin géodésique* as *"Note sur une méthode de résolution des équations normales…
(Procédé du Commandant Cholesky)"*.

**Say it `/ʃəˈlɛski/` — *shə-LES-kee*.** The first sound is the *sh* of *shoe*.

The reason is that he was French, and French ⟨ch⟩ is /ʃ/. There is a second defensible reading
from the family's origins: his paternal line descended from the **Cholewski** family, which
left Poland during the Great Emigration, and Polish ⟨ch⟩ is /x/ — the fricative in *Bach*, in
Greek χ, in Russian х, in Spanish *j*. That gives *kho-LES-kee*, and it has been argued for in
the field's own literature: a 1990 NA Digest exchange set out three candidates and concluded
that "all three current pronunciations seem acceptable" pending evidence of the name's origin,
noting that a Polish origin would make *Kholesky* correct.

**What has no basis is a hard English /k/ — "koh-LES-kee", the *k* of *kiosk*.** It is neither
the French /ʃ/ nor the Polish /x/. The two are distinct sounds: /x/ is a fricative, air still
flowing; /k/ is a plosive, stopped and released. The /k/ reading most likely comes from the
English habit of pronouncing ⟨ch⟩ as /k/ in words taken from Greek — *chorus*, *chaos*,
*character* — and this name is not Greek.

References: the pronunciation `/ʃəˈlɛski/` is given by
[Wikipedia's article on the decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition);
the Cholewski descent by
[its biography of Cholesky](https://en.wikipedia.org/wiki/Andr%C3%A9-Louis_Cholesky); the
dates, rank and the Benoît publication by the
[MacTutor biography](https://mathshistory.st-andrews.ac.uk/Biographies/Cholesky/); and the
three-way discussion by [NA Digest, Volume 90 Issue 11 (18 March 1990)](https://www.netlib.org/na-digest-html/90/v90n11.html).

## Upstream

- Website: <https://osqp.org>
- Source: <https://github.com/osqp/osqp> (Apache-2.0)
- Copyright the OSQP authors.

## What is deliberately different

These are changes from upstream, not omissions, and Apache-2.0 §4(b) asks that they be
stated:

- the inner KKT system is eliminated to an `n×n` positive definite system, rather than
  factoring the `(n+m)×(n+m)` quasi-definite system as upstream does. That reduction
  squares `A`, so it is not always the right form, and the backend is chosen from the
  matrices: a sparse `A` whose reduced matrix stays sparse is factored by CHOLMOD, and one
  with a dense row — which would densify the reduced matrix however sparse the rest of it
  is — is factored as the full quasi-definite system, sparsely, which is upstream's own
  formulation;
- for the dense reduced form the factored matrix is inverted in place, so each iteration's
  solve is one `symv` rather than two triangular solves;
- equilibration is stored as factors and applied lazily, so the caller's `P` and `A` are
  never copied or modified;
- `ρ` adapts on a fixed iteration interval rather than on a fraction of wall-clock setup
  time, so iteration counts do not depend on machine speed;
- the duality-gap termination test defaults on, following libosqp 1.x, while the two
  infeasibility thresholds follow 0.6.2 — `uᵀmax(δy,0) + lᵀmin(δy,0) < ε‖δy‖` and
  `qᵀδx < c·ε·‖δx‖`, both of which upstream's master tightened to `< 0`;
- solution derivatives are computed by implicit differentiation of the KKT conditions at
  the solution, and the element type is `Real` rather than a float, so dual numbers run the
  solver and AD can differentiate through it;
- there is no code generation. See the [Roadmap](@ref) for why that stays a difference
  rather than a gap.
