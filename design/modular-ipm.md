# PureOSQP.jl: shared matrix support for more than one QP algorithm

Design for steps 1 and 2 of the agreed strategy (internal boundary, then a Mehrotra IPM in the
same package), with the step-3 package layout sketched. Every file:line below was read; claims
that could not be checked by reading are marked **unverified**; claims that rest on the
literature rather than on this code are marked **literature** and are behind a measurement in
§6/§9. Choices the user should confirm are collected in §10.

Conventions: `ws` is today's ADMM `Workspace`; `prob` is the shared problem object introduced
here; `wt` is the weights object. Line numbers refer to the current tree.

---

## 1. Coupling inventory

What the backends, the selection ladder, equilibration, `update!`, polishing and the derivatives
read off the ADMM `Workspace` today.

### 1.1 The `Workspace` itself (`src/types.jl:294-378`)

| group | fields | lines |
|---|---|---|
| problem data + equilibration (algorithm-neutral) | `P A n m q0 l0 u0 q l u D E c` | 299-311 |
| scratch used by products and backends (neutral) | `tmp_n tmp_m work_n work_m` | 326-329 |
| ADMM weights | `rho rho_vec rho_inv_vec constr_type` | 330-333 |
| backend + accelerator | `linsys accel refactor_count` | 334-338 |
| ADMM iterates | `x y z x_prev z_prev xtilde ztilde delta_x delta_y Ax Px Aty rhs_x rhs_z` | 312-325 |
| ADMM run state (residuals, gap terms, primdual integral, counters, timing, status) | 339-376 |
| `settings::Settings{T}` (`sigma` and `scaling` are read by neutral code) | 377 |

`setup_backend` (`types.jl:541-698`) builds all of it in one constructor call (579-597) and is
the only place the ladder is reached (617-619, 635, 649, 654, 666, 678, 692).
`Base.show(::Workspace)` reads `n m rho` (`:399-407`); `factor_fill(ws)` reads `ws.n`
(`linsys.jl:57`); `dimensions` reads `n m` (`api.jl:79`).

### 1.2 Backends: every read of ADMM state

All backends take `ws` and read `n m P A D E c` (neutral) and `rho_vec | rho_inv_vec` +
`settings.sigma` (ADMM). `solve_system!` writes `ws.xtilde`/`ws.ztilde` and uses
`ws.work_n`/`ws.work_m` through `reduced_rhs!` and `mul_A!`.

| backend | `factorize!` reads ρ/σ | `solve_system!` reads ρ | other |
|---|---|---|---|
| `ReducedCholesky` | `linsys.jl:573`, `:593` | via `reduced_rhs!` `:697` | writes `xtilde` `:713`, `ztilde` `:714`; binds locals at entry `:573` |
| `DiagonalReduced` | `:604` | `:719` | |
| `TridiagonalReduced` | `:623` | `:726` | |
| `FullKKT` | `:671` (`sigma`), `:679` (`rho_inv_vec`) | `:748` | reads `ws.D[j]`, `ws.c` per entry `:666-679` (no local binding) |
| `BlockReduced` | `block.jl:76` | `:107` | `colrange(ws.A, i)` `:111` |
| `DiagonalLowRank` | `lowrank.jl:114`, `:98` | `:157` | `refactor_rho!` override `:154` — the only one |
| `KroneckerReduced` | `kronsolve.jl:93-94` | `:111` | `mu` set by rung `:88`, refreshed by `update!` `update.jl:161` |
| `SparseFormedInverse` (ext) | `SparseArraysExt.jl:283,291` | ReducedInverse method | inverts `:293-295` |
| `SparseKKT` (ext) | `:487,492` | `:570` | allocates `sparse(LD)`, `diag`, `inv.(d)`, `perm` every call `:496-504` |
| `SparseCholmod` (ext) | `:982,987` | `:1095` | |
| `SparseLDL` (LDL ext) | `LDLFactorizationsExt.jl:147,150` | `:207` | allocates `fact_L/fact_perm` `:155-156`; reaches SparseArrays ext by `Base.get_extension` `:142` |
| `LDLKKT` (LDL ext) | `:286,289` | `:328` | allocates `:294-295` |
| `BandedReduced` (Banded ext) | `BandedMatricesExt.jl:143` | `:175` | |
| `IndirectCG` (Krylov ext) | `KrylovExt.jl:107` | `:132`; **also** `:139-142` `settings.cg_tol_fraction`, `ws.scaled_prim_res/dual_res`; `:147` warm start from `ws.xtilde`; `:151,154` `cg_max_iter`, `cg_tol_reduction` | `ReducedOperator` holds `ws` `:30-37`, reads `rho_vec` `:49`, `sigma` `:55`, `work_m/work_n` `:45-56`; preconditioner is `Diagonal(ls.prec)` `:150`, identity for any `ProductOperator` operand (`operator.jl:220-228`) |

`refactor_rho!(ls, ws)` default `linsys.jl:100`; `refactor!(ws)`/`refactor_rho!(ws)`
`:764-777`; `refactored!` `:779-798` counts and resets the accelerator (`:796`).
`reduced_diagonal!` (`scaling.jl:323-337`, GPU override `GPUArraysCoreExt.jl:111-118`,
operator overrides `operator.jl:222-228`) already takes `rho, E, D, sigma, c`.

### 1.3 Selection ladder

Every rung and `choose_backend` carries `(P, A, proto, n, m, D, E, c, rho_vec, sigma)`:
`choose_backend` `linsys.jl:355-357, 505-525`; `select_backend` `:407-427`; rungs `:437-503`;
`kronecker_rung` `kronsolve.jl:70-89`; `block_rung` `block.jl:57-72`; `lowrank_rung`
`lowrank.jl:71-87`; extensions `SparseArraysExt.jl:151-186, 588-615, 1146-1181`,
`BandedMatricesExt.jl:89-127`, `GPUArraysCoreExt.jl:44-53`, `linsys.jl:831,843` →
`LDLFactorizationsExt.jl:88-106, 258-277`.

ADMM-specific decisions inside rungs: `kronecker_rung` requires uniform `rho_vec`
(`kronsolve.jl:82-83`) and identity scaling (`:86`); `sparse_kkt_backend` considers the KKT
form only when the reduced form would densify (`:596`); the sparse KKT and reduced rungs
require `P isa SparseMatrixCSC` (`:594, 1151`), so a dense `P` with a sparse `A` lands on
`formed_rung` → `SparseFormedInverse`; `ReducedInverse` inverts on the solve-many assumption
(`linsys.jl:597`, `block.jl:101`, `SparseArraysExt.jl:295`). Gates chosen so the accepted
region wins on both factorization and solve, hence algorithm-independent: `DENSE_FACTOR_FILL`
(`SparseArraysExt.jl:620-631`), `10k ≤ n` (`lowrank.jl:78-85`), `b ≤ n/4`
(`BandedMatricesExt.jl:102-118`), `DENSE_FORM_DENSITY` (`SparseArraysExt.jl:141-149`).
The `select_backend` docstring's claim that a failed `factorize!` is "rebuilt on `FullKKT`"
(`linsys.jl:398-401`) is stale: `setup_backend:696` throws through `refactored!`.

### 1.4 Equilibration and products

`equilibrate!` (`scaling.jl:193-240`) takes arrays. `mul_A!` `:250-258`, `mul_At!`
`:267-272` and its `Tridiagonal` specialization `:278-296`, `mul_P!` `:305-310` read
`ws.A/P/D/E/c` and scratch `ws.tmp_n/tmp_m`. `warm_start!` reads `D E c`
(`types.jl:715,720`).

### 1.5 `update!` (`src/update.jl:35-184`)

Neutral: `:43-53, 56-57, 67-77, 106-153` (validation), `:159-166` (adoption; nothing is
written before every check passed, `:155-158`), `:169-172` (rescale). ADMM: `is_convex(T,
P, ws.settings.sigma)` `:54`; `set_rho_vec!` `:175`; `refactor!` `:178`; uniform-ρ class
guard `:119-135`. Backend invariants through `ws.linsys isa …`: `KroneckerReduced`
`:58, 161`, `DiagonalLowRank` `:78`, `BlockReduced` `:87-105`.

### 1.6 Polishing, residuals, termination

`polish!` (`polish.jl:43-134`) reads `z y l u q D E c P A` (`:51-72`), `settings.delta`
`:47`, `polish_refine_iter` `:98`, `prim_res/dual_res` `:122-124`, calls `residuals_at`
(`:7-26`: reads `settings.scaling`, `E D c`, and uses `ws.work_m`, `ws.Px`, `ws.work_n`,
`ws.Aty` as scratch `:13-24`) and `update_residuals!` `:132`. `update_residuals!`
(`termination.jl:34-85`) writes `Ax Px Aty` `:38,46,49` and reads `settings.scaling` `:36`,
`E D c` `:41,56,79`; `eps_prim/eps_dual` read `Ax Aty Px` back `:119,129`;
`eps_duality_gap` `:142-148`; `is_primal_infeasible` `:207-221` **overwrites `delta_y` with
its projection** `:209`; `is_dual_infeasible` `:246-262`; `check_termination` `:272-300`.
`build_solution` (`admm.jl:282-309`) reads `delta_y/delta_x` `:286,293` and unscales with
`D E c` `:304-305`. `constraint_violation!` (`api.jl:115-129`) reads `settings.scaling`
`:121`, `E` `:126`.

### 1.7 Derivatives (`src/derivative.jl`)

`active_kkt` `:29-115` reads `ws.status` `:33`, `x y z` and `D E c` `:48-50`, `l0 u0` `:56`,
`P A` `:100-105`; the active-set threshold is `τ = sqrt(eps)·max(‖y‖∞, 1)` `:52,64`. The
ChainRules ext (`ChainRulesCoreExt.jl:35-46`) calls `setup(…; polishing = true)` + `solve!`.

### 1.8 Accelerator, verbose, MOI, settings

`pack_fixed_point!`/`unpack_fixed_point!` (`src/accelerate.jl:62-85`) read `rho_inv_vec`
`:65` and `rho_vec` `:83`; the COSMO extension reads no weight field itself
(`COSMOAcceleratorsExt.jl:100-133`) and calls `admm_step!` `:130`. `print_header`
(`admm.jl:63-83`) reads `backend_name(ws.linsys)`. `update_settings!` (`api.jl:31-56`)
accepts a new `sigma` and refactorizes `:48-53` (exercised by the trim entry
`settings_and_rho`, `test/trim/entrypoints.jl:124-132`). MOI `optimize!`
(`MathOptInterfaceExt.jl:169-177`); `ResultCount` returns 1 for every status but
`NON_CONVEX` `:203-206`.

### 1.9 Readers outside `src/` and `ext/`

- Tests: `ws.rho_vec` `test/solve_tests.jl:84-88,182,188,409-411,533-535`,
  `test/linsys_tests.jl:10,37,435,486,491,538`, `test/banded_tests.jl:15,20`;
  `ws.settings.sigma` same lines; `ws.xtilde/ztilde` `linsys_tests.jl:14-15,39,437-438,
  493-494,540-541,595`, `banded_tests.jl:22-23`; `ws.constr_type` `setup_tests.jl:124`,
  `update_tests.jl:40`, `solve_tests.jl:88,411`; `ws.D/E/c` `scaling_tests.jl:12-59`,
  `block_tests.jl:33-34`.
- Direct backend/rung calls: `test/linsys_tests.jl:12,34,270-271,434,490,537,571-573,594,
  601,603,658,712-714`, `test/selection_tests.jl:112-124,223`, `test/block_tests.jl:59`,
  `test/indirect_tests.jl:30-31,94-95`, `test/banded_tests.jl:19`,
  `test/coverage_tests.jl:24`; bench: `rho_update.jl`, `gate_band_beyond.jl`,
  `tridiagonal_rung.jl`, `gate_fill_periteration.jl`, `loop_breakdown.jl:53`,
  `gate_crossover_fill.jl:32`, `strictmode_audit.jl:265-276` (signatures), `:279`
  (`ReducedOperator(ws)`), `:287-288` (`ws.linsys.gram`, `typeof(ws.P)`).
- Docs: `docs/src/matrices.md:276` (`ws.P`, `ws.A`), `docs/src/examples.md:609` (`ws.rho`,
  `ws.settings`), `docs/src/operators.md:112` (`ws.xtilde`, `ws.ztilde`).

---

## 2. Target layout

### 2.1 Steps 1 and 2: one package, three directories

```
src/PureOSQP.jl                 module; includes in the order below; exports
src/core/
  constants.jl                  INFTY, MIN/MAX_SCALING, RHO_* … (types.jl:84-94)
  status.jl                     Status, PolishStatus, has_solution, status_name, Solution
  blockdiagonal.jl kronecker.jl rowcoupled.jl operator.jl   (unchanged)
  problem.jl                    Problem, validate, check_*, is_symmetric, is_convex
  scaling.jl                    equilibrate!, traversals, mul_A!/mul_At!/mul_P!(·, prob, ·),
                                reduced_diagonal!
  elementwise.jl                unchanged, plus the IPM's elementwise kernels
  weights.jl                    SystemWeights, SelectionFor tags
  linsys.jl                     LinearSystem contract, BackendInfo, ReducedCholesky, FullKKT,
                                DiagonalReduced, TridiagonalReduced, reduced_rhs!, ladder
  block.jl lowrank.jl kronsolve.jl   (backends, on prob + wt)
  preconditioner.jl             Preconditioner interface, Identity/Jacobi/ProbedWoodbury (§9.3)
  residuals.jl                  residuals_at!, gap_terms, certificate tests, support kernels
  update.jl                     validate_update!, adopt_update!, check_update(ls, P, A)
  polish.jl                     polish_kernel!
  derivative.jl                 active_kkt(prob, x, y, z) and the two derivatives
src/admm/
  settings.jl workspace.jl rho.jl accelerate.jl termination.jl admm.jl api.jl
src/ipm/
  settings.jl workspace.jl ipm.jl api.jl
src/api.jl                      setup/solve(…; algorithm), capabilities, dimensions,
                                constraint_violation
```

`@verify LinearSystem subtypes = true trim_compat = true` stays at the end of the module.

### 2.2 The ten extensions in steps 1–2

| extension | touches | change |
|---|---|---|
| SparseArrays | traversals, rungs, three backends, `is_convex`, `check_finite` | backend/rung signatures; `IPMSelection` methods (§5) |
| LDLFactorizations | `ldl_backend`, `ldl_kkt_backend`, `ldl_posdef`, two backends | backend signatures |
| BandedMatrices | `choose_backend`, `BandedReduced`, `structural_rows`, `is_symmetric/is_convex` | signatures |
| Krylov | `IndirectCG`, `ReducedOperator`, `indirect_backend` | signatures, tolerance seam, preconditioner slot, `inner_iterations` (§3.6, §9) |
| GPUArraysCore | `choose_backend` refusals, traversals | `choose_backend` signature; an `IPMSelection` refusal (§5) |
| LinearMaps, SciMLOperators | `ProductOperator` constructors, `setup`/`solve` overloads | forward `algorithm` and `preconditioner` keywords (they already forward `kwargs...`) |
| COSMOAccelerators | ADMM hooks, `admm_step!` | none (its reads are in `src/accelerate.jl`) |
| ChainRulesCore | `setup`/`solve!`/derivatives | passes `algorithm` through in step 2 |
| MathOptInterface | `setup`/`solve!`/`Solution` | `algorithm` raw attribute, `NUMERICAL_ERROR` mapping in step 2 |

### 2.3 Step 3: monorepo layout (design only)

```
<repo>/                         umbrella <Umbrella>.jl: solve/setup(…; algorithm), MOI ext,
                                ChainRulesCore ext, docs, oracle/corpus tests, trim entry
                                points, StrictMode audit
lib/<Core>.jl/                  src/core/*; extensions: SparseArrays, LDLFactorizations,
                                BandedMatrices, Krylov, GPUArraysCore, LinearMaps, SciMLOperators
lib/<Core>ADMM.jl/              src/admm/*; depends on <Core>; extension: COSMOAccelerators
lib/<Core>IPM.jl/               src/ipm/*; depends on <Core>
```

An extension goes where the function it extends lives. `PureOSQP.Optimizer` (stub at
`src/PureOSQP.jl:62`) and `solve` → umbrella, so MOI and ChainRules go there.
`LDLFactorizationsExt` reaches `SparseArraysExt` through `Base.get_extension(<Core>, …)`
(`LDLFactorizationsExt.jl:102,142,272,280,296`); both stay Core extensions. Registration
order Core → ADMM, IPM → umbrella through Registrator's `subdir` (**unverified** on this
workstation). Each subpackage has its own `test/` with a `meta_tests.jl` inventory; the trim
test and the audit need every extension and live in the umbrella; a Core-only trim entry set
(`Problem` + `factorize!`/`solve_system!` on `ReducedCholesky` and `FullKKT`) is added.

---

## 3. The backend interface

### 3.1 Weights

```julia
"""
    SystemWeights{T,V}

The diagonal weights and the primal regularization the linear system is built from:

    reduced   P̃ + σI + Ãᵀ diag(w) Ã
    KKT       [P̃ + σI   Ãᵀ  ;  Ã   −diag(w_inv)]

Invariants, maintained by the owner: `w[i] > 0`, `w_inv[i] == inv(w[i])` as the owner
computed it, `sigma > 0`, `length(w) == m`. The vectors are read in place, so a change to
their contents reaches the next `refactor_weights!` without a new object; a change to `sigma`
needs a new object.
"""
struct SystemWeights{T <: Real, V <: AbstractVector{T}}
    w::V
    w_inv::V
    sigma::T
end
```

ADMM: `w = rho_vec`, `w_inv = rho_inv_vec`, `sigma = settings.sigma`; `update_settings!`
assigns `ws.weights = SystemWeights(ws.weights.w, ws.weights.w_inv, new.sigma)` whenever it
refactorizes (`api.jl:48-53`). IPM: `w_inv = s/z + δ_d`, `w = inv.(w_inv)`, `sigma = δ_p`
(§8.2). Folding `δ_d` into `w_inv` keeps one invariant and leaves every backend unchanged.

The object is a field of each workspace, not built per call, so the hot path constructs
nothing. `rho_vec`/`rho_inv_vec` as `Workspace` fields are removed; the tests in §1.9 read
`ws.weights.w`.

`refactor_weights!(ls, prob, wt)` is the backend-level function; the workspace-level
`refactor_rho!(ws)` keeps its name.

### 3.2 Contract

```julia
abstract type LinearSystem end

@contract LinearSystem begin
    factorize!(::Self, ::Problem, ::SystemWeights)::Bool
    solve_system!(::Self, ::Problem, ::SystemWeights, ::Any, ::Any, ::Any, ::Any)::Nothing
    backend_info(::Self)::BackendInfo
end
```

TypeContracts 0.14 checks `hasmethod(f, Tuple{Self, arg_types...})` then infers the return
(`~/.julia/packages/TypeContracts/wFVz0/src/check.jl:167-175`). With `::Problem` and
`::SystemWeights` in the contract, a method may annotate those two slots with exactly those
types or leave them unannotated; anything narrower is reported missing. The vector slots are
`::Any` as today, and every method annotates its return so inference through the abstract
`Problem` gives the declared type.

```julia
"""
    factorize!(ls, prob, wt) -> Bool

Rebuild the factorization of the system `prob` and `wt` define. `false` means this backend
cannot factor it (not positive definite for a reduced backend; a zero pivot for a KKT one).
Reads `prob.P A D E c n m` and `wt`; may use `prob.work_n work_m` as scratch. Every method
binds `P A D E c n m` to locals at entry (as `linsys.jl:573, 603-604` do).
"""
function factorize! end

"""
    refactor_weights!(ls, prob, wt) -> Bool

Refresh after only `wt` changed since the last `factorize!`. Default rebuilds; an override
may keep every part that depends on `P A D E c` alone. (`DiagonalLowRank` keeps its
override, `lowrank.jl:154`.)
"""
refactor_weights!(ls::LinearSystem, prob, wt) = factorize!(ls, prob, wt)

"""
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z) -> Nothing

Solve for `x` and write `z = Ã x` as this backend computes it: a reduced backend forms the
product, a KKT backend recovers it from the eliminated multiplier as `rhs_z + w_inv ⊙ ν`.
None of `rhs_x rhs_z x z` may alias each other or `prob.work_n`, `prob.work_m`, `prob.tmp_n`,
`prob.tmp_m` (`reduced_rhs!` writes `work_n` and `work_m`; the products use `tmp_*`).
"""
function solve_system! end

"""
    solve_multiplier!(ls, prob, wt, rhs_x, rhs_z, x, nu) -> Nothing

Solve the same system for `x` and the multiplier `ν`. Row two of the KKT system reads
`Ã x − w_inv ⊙ ν = rhs_z`, so `ν = w ⊙ (Ã x − rhs_z)`, which is what the default derives
from `solve_system!`; a KKT backend overrides it to hand `ν` over directly, which does not
lose digits when `w_inv` is small.
"""
function solve_multiplier!(ls::LinearSystem, prob, wt, rhs_x, rhs_z, x, nu)
    solve_system!(ls, prob, wt, rhs_x, rhs_z, x, nu)      # nu holds z for a moment
    subtract!(nu, nu, rhs_z)
    scale_by!(nu, wt.w)
    return nothing
end
```

Two solve methods rather than one returning `ν`: ADMM's `z̃` must stay the value `Ãx̃`
computed as today (`linsys.jl:714`) for the oracle's 1e-10 iterate match, and IPM wants `ν`
without the cancellation of `w ⊙ (z − rhs_z)` on a KKT backend near convergence. Overrides:
`FullKKT`, `SparseKKT`, `LDLKKT` (each is the existing `solve_system!` minus its last loop).

`reduced_rhs!(prob, wt, rhs_x, rhs_z) -> prob.work_n` keeps its arithmetic order
(`linsys.jl:695-704`) with `wt.w` for `ws.rho_vec`.

`check_update(ls::LinearSystem, P, A) -> Nothing` (default no-op) hosts the three backend
invariants that `update!` tests today through `ws.linsys isa …` (`update.jl:58,78,87-105`).
`update!` calls it with the effective matrices (the replacement if given, otherwise the current
`prob.P`/`prob.A`) whenever either is replaced. `KroneckerReduced.factorize!` returns `false`
when `P` is not a scalar multiple, removing the separate refresh at `update.jl:161`.

`inner_iterations(ls::LinearSystem) -> Int` (default `0`) reports the inner iterations an
iterative backend has spent over its life; `IndirectCG` counts `stats.niter` per solve.
`Solution.cg_iters` is the difference across one solve, under both algorithms.

### 3.3 Cost reporting

`BackendInfo` is unchanged. The per-backend flop model used to compare backends under the
IPM lives in the bench scripts (`bench/ipm_vs_clarabel.jl`, `bench/ipm_matrixfree.jl`),
computed from `backend_info(ls).factor_nnz` and the dimensions, next to the measurement it is
checked against.

### 3.4 How ADMM calls it

```julia
# admm_step!
scale_subtract!(ws.rhs_x, ws.weights.sigma, ws.x_prev, ws.prob.q)
subtract_scaled!(ws.rhs_z, ws.z_prev, ws.weights.w_inv, ws.y)
# set_tolerance_level!(ws.linsys, max(scaled_prim_res, scaled_dual_res))  immediately before:
solve_system!(ws.linsys, ws.prob, ws.weights, ws.rhs_x, ws.rhs_z, ws.xtilde, ws.ztilde)
update_x!(…); update_zy!(…, ws.weights.w, ws.weights.w_inv, …)

# refactor!(ws)     = refactored!(ws, factorize!(ws.linsys, ws.prob, ws.weights))
# refactor_rho!(ws) = refactored!(ws, refactor_weights!(ws.linsys, ws.prob, ws.weights))
# refactored! keeps the count, the throw and accelerator_reset! (ADMM policy)
# setup_backend and update_settings! call adopt_settings!(ws.linsys, ws.settings) after the workspace is built
```

### 3.5 How IPM calls it

```julia
# outside ipm_step!, once per outer iteration (the backend's allocation lives here):
ws.w_inv .= s ./ z .+ δ_d          # masked per row; equality rows δ_d, free rows 1/δ_d
ws.w     .= inv.(ws.w_inv)
refactor_weights!(ws.linsys, ws.prob, ws.weights) || bump_regularization!(ws)
# ipm_step!: two solves on the same factorization, no allocation
solve_multiplier!(ws.linsys, ws.prob, ws.weights, rhs_x, rhs_z, dx_aff, dy_aff)   # predictor
…                                                                                # corrector
```

### 3.6 The CG seam (Krylov ext)

`IndirectCG{T,V,K,M}` holds a preconditioner `M` (§9.3), a mutable `level::T` to track the
tolerance, and counters `total_iters::Int`, `misses::Int`. The three CG settings (max_iter,
tol_fraction, tol_reduction) are not stored at construction; `indirect_backend(proto, n, m)`
has no such parameters.

```julia
adopt_settings!(ls::LinearSystem, settings) = nothing       # direct backends
adopt_settings!(ls::IndirectCG, settings) = (ls.max_iter = settings.cg_max_iter;
                                             ls.tol_fraction = settings.cg_tol_fraction;
                                             ls.tol_reduction = settings.cg_tol_reduction;
                                             nothing)

set_tolerance_level!(ls::LinearSystem, level) = nothing     # direct backends
set_tolerance_level!(ls::IndirectCG, level) = (ls.level = level; nothing)
last_solve_converged(ls::LinearSystem) = true
last_solve_converged(ls::IndirectCG) = ls.kws.stats.solved   # verify field name: unverified
```

Settings passed at construction do not reach the `:auto` operator path through `indirect_rung`,
and nothing would refresh them on `update_settings!`; `adopt_settings!` is called by `setup_backend`
after the workspace is built and by `update_settings!` whenever settings change.

ADMM calls `set_tolerance_level!` immediately before `solve_system!` with
`max(scaled_prim_res, scaled_dual_res)`; the backend computes `atol` exactly as at
`KrylovExt.jl:140-142`, so `:indirect` iterates do not move. CG warm-starts from the `x` argument,
which holds the previous solution (the same values as `ws.xtilde` today).
`ReducedOperator{T,PB,WT}(prob, wt)` replaces `ReducedOperator(ws)`.

---

## 4. Workspace split

### 4.1 `Problem` (shared)

```julia
mutable struct Problem{T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T}}
    P::MP                 # mutable: update! replaces them (update.jl:160,163)
    A::MA
    n::Int
    m::Int
    q0::V; l0::V; u0::V   # caller's data, clamped to ±INFTY
    q::V;  l::V;  u::V    # equilibrated
    D::V;  E::V;  c::T    # Ruiz factors
    scaling::Int          # sweeps run; 0 ⇒ D, E, c are identity (replaces settings.scaling reads)
    tmp_n::V; tmp_m::V    # scratch of mul_A!/mul_At!/mul_P!
    work_n::V; work_m::V  # scratch of reduced_rhs! and the backends
end
```

`Problem(T, P, q, A, l, u; scaling)` runs `validate`, allocates with the `similar(q0, …)`
discipline (`types.jl:553-560`), and calls `equilibrate!`. `validated_problem(T, n, m, P, q, A, l, u, scaling)`
builds a `Problem` from already-validated data, bypassing the validation step; `setup` validates
once and uses it. It does not run `is_convex`: the shift is the algorithm's (`σ` for ADMM, `δ_p`
for IPM), so each `setup` calls `is_convex(T, P, shift)` where `types.jl:548` does today, and each
documents its shift.

### 4.2 ADMM `Workspace`

```julia
mutable struct Workspace{T, MP, MA, V, VI <: AbstractVector{Int8}, LS <: LinearSystem, AC}
    prob::Problem{T, MP, MA, V}
    linsys::LS
    weights::SystemWeights{T, V}     # w = ρ, w_inv = ρ⁻¹, sigma = σ
    rho::T
    constr_type::VI
    x y z x_prev z_prev xtilde ztilde delta_x delta_y Ax Px Aty rhs_x rhs_z :: V
    accel::AC
    refactor_count::Int
    … every run-state field from types.jl:339-376, unchanged …
    settings::Settings{T}
end
```

Same seven type parameters, so the `MAX_TYPEUNION_LENGTH` argument at `types.jl:514-527` is
untouched; `Problem{T,MP,MA,V}` is determined by them. Field moves: `P A n m q0 l0 u0 q l u D
E c tmp_n tmp_m work_n work_m` → `ws.prob`; `rho_vec rho_inv_vec` → `ws.weights`. No
`getproperty` forwarding; every read is rewritten, including §1.9. `setup(::Type{T}, …;
linsys, kwargs...)` keeps `@constprop :aggressive` and the `Val(linsys)` lift;
`setup_backend` becomes `prob = Problem(…; scaling = settings.scaling)` + `is_convex` + ρ
classification + weights + the same `if LS === …` ladder over `choose_backend(P, A, prob,
wt, ADMMSelection())`.

### 4.3 Who adapts

- **Products** take `prob`; the `Tridiagonal` specialization dispatches on
  `Problem{T, <:AbstractMatrix, <:Tridiagonal}`.
- **Residual kernels** (`core/residuals.jl`): `residuals_at!(prob, x, y, z, Ax, Px, Aty) ->
  (pr, dr, obj)` (today's `polish.jl:7-26`, writing the three vectors it was reading off
  `ws`), `gap_terms(prob, y, Px, x) -> (xtPx, qtx, SCy)` (`termination.jl:67-73`, `tmp_m` as
  scratch as today `:71`), `eps_prim(prob, settings, z, Ax)`, `eps_dual(prob, settings, Aty,
  Px)`. Both workspaces own `Ax Px Aty`. `update_residuals!(ws)` keeps its name, its field
  writes and its arithmetic order, calling the kernels.
- **Certificates**: `is_primal_infeasible(prob, dy, eps) -> Bool` projects `dy` in place, as
  today (`termination.jl:209`); each workspace owns the buffers it hands in (`delta_y` for
  ADMM; `cert_y`, `cert_x` for IPM, §8.7).
- **`update!`**: `validate_update!(prob, ls; P, A, q, l, u)` (`update.jl:43-153` minus the
  ADMM and backend lines of §1.5, which become `check_update` and the ADMM tail) and
  `adopt_update!(prob; …)` (`:159-172`). `update!` calls `check_update(ls, P, A)` with the
  effective matrices (the replacement if given, otherwise the current `prob.P`/`prob.A`) whenever
  either is replaced. ADMM: validate (+ `is_convex` with `σ`, the Kronecker ρ-class guard) →
  adopt → `set_rho_vec!` if bounds moved → `refactor!`. IPM: validate (with `δ_p`) → adopt →
  recompute row classes; `s z` are untouched until the next starting point; no refactorization,
  the next iteration factorizes anyway.
- **Derivatives**: `active_kkt(prob, x, y, z)` on the problem-space point; each workspace's
  method checks `status === SOLVED`, `require_host`, unscales, delegates.
- **Polishing**: `polish_kernel!(prob, x, y, z, prim_res, dual_res, Ax, Px, Aty; delta,
  refine_iter) -> (status, xpol, ypol, zpol)` is `polish.jl:43-125`; ADMM's `polish!(ws)`
  copies on success and runs `update_residuals!`.
- **MOI**: step 2 adds `algorithm` (a `Symbol` in `(:admm, :ipm)`, validated at `MOI.set`)
  and the `NUMERICAL_ERROR` rows (§8.6).
- **`Solution`**: one struct; new field `cg_iters::Int` (both algorithms). IPM fills
  `rho_estimate rho_updates accel_declined primdual_int*` with zeros and documents it.

---

## 5. Algorithm-aware selection

```julia
abstract type SelectionFor end
struct ADMMSelection <: SelectionFor end
struct IPMSelection <: SelectionFor end
```

The tag is the last argument of `choose_backend(P, A, prob, wt, sel)`, `select_backend`, and
every rung (`density_gate_rung(P, A, prob, sel)`, `formed_rung(P, A, prob, sel)`, the rest
`(P, A, prob, wt, sel)`); extension helpers take it too. `P` and `A` stay explicit because the
extensions dispatch on them. Under `ADMMSelection` every rung is the method it is today with
the ten-argument tail collapsed; the ADMM ladder does not change. The IPM owns its deviations
as extra methods:

| rung / backend | `ADMMSelection` | `IPMSelection` |
|---|---|---|
| `kronecker_rung` | as today (uniform `wt.w`, identity scaling) | `nothing` |
| `sparse_kkt_backend` | KKT considered only when the reduced form would densify (`:596`) | pre-gate skipped; KKT factorization tried first, fill gate unchanged |
| `density_gate_rung`, `formed_rung`, `dense_rung` | as today | `FullKKT(proto, n, m)` for `T <: BlasFloat` (materializes `A` once; covers dense-`P`/sparse-`A` and LP pairs); `ReducedCholesky` otherwise (`bunchkaufman!` has no generic method) |
| structured reduced backends (diagonal, tridiagonal, banded, block, lowrank), `SparseCholmod/SparseLDL` | as today | as today; reached only when the pair has that structure; gated by measurement (S10) |
| `IndirectCG` | as today | reachable by `:auto` only after the §9.5 gate; always by `linsys = :indirect` |
| GPU ext `choose_backend` | refusal as today | `choose_backend(P::AbstractGPUMatrix, …, ::IPMSelection)` refuses `:ipm` by name (v1 is CPU-only) |
| named kinds | as today | `:kronecker` throws naming the uniform-weights requirement |

`ReducedInverse` under IPM: the inversion costs `2n³/3` per iteration and buys a `symv` for
2–4 solves. `FullKKT` is preferred instead; a factor-keeping `ReducedCholeskyFactor` is added
only if the S13 bench shows the inversion above ~25% of an IPM iteration on a reached rung.

---

## 6. Migration sequence

Gates, run after every step unless stated: (a) full suite via `julia_run_testitems` with
`max_workers` set; (b) `julia --project=bench bench/strictmode_audit.jl`; (c) the trim test
item; (d) the S0 snapshot artifact through S7. "Identical" means the oracle, c-suite and
corpus items pass unchanged and the snapshot matches. **M** mechanical, **J** judgment.

| # | step | gate | diff. |
|---|---|---|---|
| S0 | **Snapshot artifact.** `bench/snapshot.jl`: for every suite class in `bench/suite_problems.jl` and every structured family in `selection_tests.jl`, record `(backend_name, iter, refactor_count, status, round(obj_val, 10))` at fixed settings with `BLAS.set_num_threads(1)`, on the S0 tree, into `bench/results/snapshot_s0.json`; a bench check compares the current tree to it. Not a committed test item; `meta_tests.jl` unchanged. | artifact generated | M |
| **S-spike** | **Matrix-free measurement, before any refactor** (§9.5). `bench/ipm_matrixfree_spike.jl`: a throwaway dense Mehrotra prototype (bench-only, `Float64`, dense `bunchkaufman!` on the regularized KKT) produces the `(x_k, W_k)` sequence on LinearMap-over-dense QPs; for each `k` the reduced system is solved by `Krylov.cg!` under `δ ∈ {1e-8, 1e-4, 1e-2}` and each preconditioner (none, exact Jacobi, probed Woodbury `k ∈ {5, 20}`), recording iterations to `atol_k`. Answers §9.5 (i)–(iv). Results under `bench/results/`. | numbers committed; §10 decisions taken | J |
| S2a | **`Problem` extraction.** `core/problem.jl`; `Workspace.prob`; every `ws.<moved field>` rewritten; `mul_*` on `prob`; `settings.scaling` reads → `prob.scaling`; backends keep `(ls, ws)` for now but bind `P A D E c n m` to locals at entry; docs lines in §1.9. Gate adds `bench/loop_breakdown.jl` step timings against S0 on neuromancer, ABBA-interleaved with the S0 tree (indicative, §10.7). | identical; audit; trim; timings within noise | M |
| S1+S2b | **Weights and backend signatures, one pass.** `SystemWeights`; `factorize!(ls, prob, wt)`, `refactor_weights!`, `solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z)`; every `rho_vec/rho_inv_vec/settings.sigma` read of §1.2 → `wt`; `Workspace.weights`; `update_settings!` rebuilds `weights` (+ test that changes `sigma` and checks the factorized matrix); `set_rho_vec!`, `pack/unpack_fixed_point!`, `admm_step!` follow; `reduced_rhs!(prob, wt, …)`; `ReducedOperator(prob, wt)`; `check_update` + Kronecker `mu` in `factorize!`; `update!` split; contract; tests/bench/audit signatures `(LS, PB, WT, V, V, V, V)`, `(LS, PB, WT)`. CG seam: `set_tolerance_level!(ls::LinearSystem, level) = nothing` (default), mutable `level::T` field on `IndirectCG`, `admm_step!` sets it to `max(scaled_prim_res, scaled_dual_res)` immediately before `solve_system!`; `adopt_settings!(ls::LinearSystem, settings) = nothing` (default) with `IndirectCG` method filling `max_iter`, `tol_fraction`, `tol_reduction` from settings, called by `setup_backend` after workspace build and by `update_settings!` on every settings change; CG warm-starts from `x` argument. | identical; audit; trim | M |
| S3 | **Selection tag.** `SelectionFor`; collapse rung signatures; `ADMMSelection` threaded from `setup`; no `IPMSelection` methods yet. | identical; `selection_tests` backends unchanged | M |
| S4 | **Shared kernels.** `residuals_at!`, `gap_terms`, `eps_*`, certificate tests on `(prob, buffer, eps)`, `polish_kernel!`, `active_kkt(prob, x, y, z)`; ADMM wrappers keep names and order. | identical; audit (`update_residuals!` row); trim (`derivatives`, `solve_polish`) | M |
| S5 | **CG seam continued.** Counters `total_iters`, `misses`; `last_solve_converged(ls)` reporting CG success; `inner_iterations(ls)` total lifetime CG iterations; `Solution.cg_iters` per-solve count; preconditioner slot with `IdentityPreconditioner`/`JacobiPreconditioner` reproducing today's `Diagonal(ls.prec)` (§9.3). | identical on `:indirect` (`indirect_tests`, `solve_indirect` trim entry) | J |
| S6 | **`solve_multiplier!`** default + `FullKKT`/`SparseKKT`/`LDLKKT` overrides; test at `w_inv = 1e-12`. | suite; audit unchanged | M |
| S7 | **Directory move** to `src/core`, `src/admm`, `src/ipm` (empty). | identical; audit; trim | M |
| S8 | **IPM skeleton on direct backends.** `IPMSettings`, `IPMWorkspace`, `setup(…; algorithm = :ipm)` via `Val`, `seeded`, starting point, `ipm_step!` (predictor–corrector, `τ = 0.99`), proximal regularization (§8.5), residuals through S4 kernels, `SOLVED`/`SOLVED_INACCURATE`/`MAX_ITER_REACHED`, `solve!`, `build_solution`; `IPMSelection` methods of §5 (FullKKT routing, KKT-first sparse, kronecker decline, GPU refusal). Corpus items under `:ipm` with `:auto`/`:kkt`, referee `< 1e-5`, backend name asserted for the dense-`P`/sparse-`A` and LP cases; objective vs `osqp_ref`. | new items pass; ADMM gates identical | J |
| S9 | **Robustness.** Dynamic regularization bump; `NUMERICAL_ERROR` (+ every switch of §8.6); certificate buffers, stall rule, certificate tests on step and normalized iterates; `time_limit`, interrupt. Tests: c-suite ported cases under `:ipm`; a random infeasible `n = 20, m = 40` primal case and a dual one; equality-only and free-row corpus cases. | items pass | J |
| S9b | **HSD decision point** (§9.6, §10): implement the bordered solve on `solve_system!` only if S9's infeasible cases are not detected. | — | J |
| S10 | **Structured backends under IPM, measured.** Each structured family through its recorded backend under `:ipm`: referee tolerance and iteration count recorded per backend into the snapshot; a family whose count exceeds `2×` the `FullKKT` count on the same problem is routed to `FullKKT` under `IPMSelection` (materialized). `Float32` and `Dual` on `:auto`. | items pass; table in docs | J |
| S11 | **Operator IPM** (§9): `ProbedWoodburyPreconditioner`, inner tolerance rule, budgets, `cg_fail_limit`, `bench/ipm_matrixfree.jl` (operator-IPM vs dense-IPM vs ADMM-operator), support-matrix row. Ships behind the §9.5 gate. | bench numbers under `bench/results/`; gate verdict recorded in docs | J |
| S12 | **Polish, derivatives, `update!`, warm start, MOI for IPM.** | `derivative_tests`, `update_tests`, `moi_tests` parametrized where semantics carry | J |
| S13 | **StrictMode + trim for IPM; Clarabel bench; docs.** Audit rows of §8.12; trim entries; `bench/ipm_vs_clarabel.jl`; iteration bounds; API/guarantees/algorithm pages. | audit green; trim green; bench committed | J |

The spike is first because its outcome can change §8.5 (the `δ` defaults and whether
refinement exists at all) and §9 (whether operator IPM ships); everything from S2a on is
independent of it, so the two can also run in parallel on separate worktrees, but the spike's
verdict must be in before S8 starts.

---

## 7. Risks, ranked

1. **Bitwise drift in the ADMM step.** Rewrite reads, never expressions; S0 snapshot + oracle
   after every step.
2. **`@constprop` fragility in `setup`** (`types.jl:514-527`). `Problem` takes only
   `scaling::Int`; `linsys` stays a `Val`; trim entries `solve_kronecker`,
   `solve_lowrank_scaled`, `setup_kronecker` are the detectors.
3. **Operator IPM may not beat ADMM-operator** even with proximal regularization and the
   probed preconditioner (§9.5 (v)). The spike bounds the CG growth early; the S11 bench
   decides shipping. Marked as a user decision.
4. **Reduced backends under IPM** carry `w ≤ 1/δ_d`; with `δ_d = 1e-8` the reduced matrix's
   conditioning is `~1e8‖Ã‖²`, marginal for `Float64` and hopeless for `Float32` — hence
   `T`-dependent `δ`, `FullKKT` routing for `BlasFloat`, and the S10 measurement.
5. **Infeasibility detection is heuristic** (§8.7); the S9 random infeasible cases decide
   whether S9b (HSD) is needed.
6. **Derivatives after an IPM solve**: inactive-row multipliers are `O(μ_final)`, above the
   `sqrt(eps)` threshold at default tolerances (`derivative.jl:52`); polishing cleans the
   active set and is required for IPM derivatives in v1.
7. **`NUMERICAL_ERROR`** is a new `Status` value; every switch in §8.6 must grow with it.
8. **Test churn**: ~120 lines outside `src/ext` read moved fields (§1.9).
9. **Step-3 registration mechanics** untested here.

---

## 8. Mehrotra IPM specification

### 8.1 Problem form and row classes

The IPM runs on the equilibrated `prob` (`P̃ q̃ Ã l̃ ũ`). Each row is classified at setup and
in `update!` into an `Int8` class vector `rclass` and two `Bool` masks `has_l`, `has_u`:

| class | condition (`loose = INFTY(T) * MIN_SCALING(T)`, as `rho_class`) | slacks |
|---|---|---|
| free (`-1`) | `l̃ < −loose && ũ > loose` | none; `y_i ≡ 0`; `w_inv = 1/δ_d` |
| equality (`1`) | `l0 == u0` exactly | none; free multiplier `y_i`; `w_inv = δ_d` |
| inequality (`0`) | otherwise; `has_l = l̃ > −loose`, `has_u = ũ < loose` | `s_l, z_l` if `has_l`; `s_u, z_u` if `has_u` |

    s_l = Ãx − l̃,  s_u = ũ − Ãx,   y_i = z_u,i − z_l,i   (inequality rows; y ∈ N_[l,u](Ax))

Storage: `s_l s_u z_l z_u` as four length-`m` vectors, unused entries `1` for slacks and `0`
for multipliers so every loop is branch-free apart from the mask; `N_s = count(has_l) +
count(has_u)`. Variable bounds arrive as rows of `A` (MOI does this,
`MathOptInterfaceExt.jl:130-139`); there is no separate box-slack path.

### 8.2 Newton system and the weights

Residuals of the *original* (scaled) problem at the current iterate:

    r_d = P̃x + q̃ + Ãᵀy
    r_l = Ãx − l̃ − s_l,   r_u = ũ − Ãx − s_u   (masked)
    r_e = Ãx − l̃                              (equality rows)
    r_cl = s_l ∘ z_l − σμ e (+ Δs_a ∘ Δz_a),  r_cu likewise

Eliminating `Δs`, `Δz` (derived from the KKT conditions of the two-sided form):

    W_i = z_l/s_l + z_u/s_u        (absent bound contributes 0)
    g_i = r_cl/s_l + (z_l/s_l) r_l − r_cu/s_u − (z_u/s_u) r_u
    Δy_i = W_i (ÃΔx)_i + g_i

With the proximal regularization of §8.5 the system is the package's KKT form with
`w_inv = 1/W + δ_d`, `w = 1/w_inv`, `sigma = δ_p`:

    [P̃ + δ_p I   Ãᵀ  ] [Δx]   [ −r_d      ]
    [Ã   −diag(w_inv) ] [Δy] = [ −w_inv ⊙ g ]       rhs_x = −r_d,  rhs_z = −w_inv ⊙ g

Equality rows: `rhs_z,i = −r_e,i`. Free rows: `rhs_z,i = 0` and `Δy_i` is zeroed after the
solve (one masked store), so `y_i ≡ 0` and the row contributes `δ_d ãᵢãᵢᵀ` to the reduced
matrix and nothing else. After `solve_multiplier!` returns `Δx, Δy`:

    Δs_l = ÃΔx + r_l,  Δs_u = −ÃΔx + r_u        (one mul_A! into scratch)
    Δz_l = −(r_cl + z_l ∘ Δs_l) / s_l,  Δz_u = −(r_cu + z_u ∘ Δs_u) / s_u

### 8.3 Starting point and `seeded`

`IPMWorkspace.seeded::Bool` is set by `warm_start!`, by `x0`/`y0`, and by a completed
`solve!`; cleared by `cold_start!`. A solve starts:

1. If `!seeded`: one factorization and one `solve_system!` with `w = w_inv = 1`, `sigma =
   δ_p`, `rhs_x = −q̃`, `rhs_z = t` (`t_i` = midpoint of a two-sided row, the finite bound of a
   one-sided one, `l̃_i` of an equality, `0` of a free row), i.e.
   `(P̃ + δ_p I + ÃᵀÃ) x = −q̃ + Ãᵀt`. If `seeded`: `x` is the workspace's.
2. `s_l = Ãx − l̃`, `s_u = ũ − Ãx` (masked); `θ = max(0, −1.5·min(s))`; `s .+= θ`; `z .= 1`
   on masked entries, or from `y`: `z_u = max(y, 0) + 1`, `z_l = max(−y, 0) + 1`.
3. Mehrotra's balancing: `δ_s = ½(sᵀz)/(eᵀz)`, `δ_z = ½(sᵀz)/(eᵀs)`; `s .+= δ_s`, `z .+= δ_z`.
   Equality multipliers keep `y` or start at `0`.

`warm_starting = false` clears `seeded` before every solve. No claim is made about the
iteration count of a warm-started re-solve; S13 records the measured number.

### 8.4 One outer iteration

Outside `ipm_step!` (allocation allowed, `:warm` audit rows):

1. `μ = (s_lᵀz_l + s_uᵀz_u)/N_s`; weights per §8.2; `refactor_weights!`; on `false`,
   `bump_regularization!` (§8.5) and retry.

Inside `ipm_step!` (`:hot`, no allocation):

2. **Predictor**: `r_cl = s_l∘z_l`, `r_cu = s_u∘z_u`; `g`, `rhs_x`, `rhs_z`;
   `solve_multiplier!` → `Δx_a, Δy_a`; recover `Δs_a, Δz_a`; `α_a` = largest step in `(0, 1]`
   keeping `s, z ≥ 0` (one step length for primal and dual: the dual residual couples `x`
   and `y`); `μ_a`, `σ = (μ_a/μ)³`.
3. **Corrector**: `r_cl = s_l∘z_l + Δs_a∘Δz_a − σμ e` (and `u`); rebuild `g`, `rhs_z`; same
   factorization; `solve_multiplier!`; optional refinement (§8.5); recover `Δs, Δz`.
4. `α = min(1, τ·α_max)`, `τ = step_fraction = 0.99`; update `x, y_E, s, z`; `y = z_u − z_l`
   on inequality rows; zero `Δy` on free rows.

Then residuals and termination (§8.6) every `check_termination` iterations and at exit, with
`time_limit` and `InterruptException` handled as in `admm.jl:161-225`. No Gondzio
correctors, no adaptive `τ`, no separate step lengths in v1.

### 8.5 Regularization model (decision for the user, §10.1)

One model for every backend: **primal–dual proximal regularization** — the Newton system
carries `δ_p I` on the primal block and `δ_d I` on the dual block at every iteration, the
right-hand side is built from the original residuals (the proximal terms are centred on the
current iterate and vanish there), the regularization is never refined away, and termination
is measured on the original residuals. Convergence to the original problem's solution for
any fixed `δ_p, δ_d > 0` is the Friedlander–Orban result (**literature**; more outer
iterations as `δ` grows). What this buys: `δ` can be as large as the linear solver needs —
small for direct backends, large enough to bound `κ` for CG — without a second code path.

- Defaults: `reg_primal = reg_dual = max(T(1e-8), sqrt(eps(T)))` for direct backends
  (`1e-8` for `Float64`, `3.4e-4` for `Float32`); the CG default comes from the spike
  (§9.5), provisionally `1e-4`. Both in scaled space, held in `IPMSettings`, changeable
  through `update_settings!` without refactorization.
- `is_convex(T, P, reg_primal)` at setup; documented as the IPM's shift.
- Dynamic: on `factorize!` returning `false`, multiply both by `10` and retry, up to
  `max_reg_bumps = 5`; then `NUMERICAL_ERROR` with the last point.
- Refinement (direct backends only, `refine_iter` default `1`): one step against the
  *regularized* operator to correct rounding in the factorization —
  `r = [P̃Δx + δ_pΔx + ÃᵀΔy − rhs_x ; ÃΔx − w_inv ⊙ Δy − rhs_z]` (the second block is zero
  by construction after `solve_multiplier!`), correction through the same factorization.
  For `IndirectCG` `refine_iter` is `0`: a restarted CG adds nothing a larger budget would
  not.
- Requirement stated in the `IPMSettings` docstring: a reduced backend needs
  `κ(reduced) · eps(T) < 1`, and `κ ≤ (λ_max(P̃) + ‖Ã‖²/δ_d)/(λ_min(P̃) + δ_p)`.

### 8.6 Termination and status

Residuals from the S4 kernels on `(x, y, z = clamp(Ãx, l̃, ũ))`, tolerances `eps_prim`,
`eps_dual`, `eps_duality_gap` with the same formulas as ADMM, so `eps_abs`, `eps_rel`,
`check_dualgap`, `scaled_termination` mean the same thing under both algorithms — the
*defaults* differ (`1e-8` vs `1e-3`, §8.10) and the API and MOI docs say so. `SOLVED` when all
three pass; at `max_iter`, retry at ten times for `SOLVED_INACCURATE`, else
`MAX_ITER_REACHED`. `check_termination = 1` by default: every iteration pays the residual
kernels, which an IPM's iteration count justifies.

`Status` gains `NUMERICAL_ERROR` (regularization ceiling, CG failure, stall without
certificate, NaN residual — an IPM NaN after `is_convex` passed is a numerical failure, not
`NON_CONVEX`). Switches to update: the `Status` docstring ("Eleven values", `types.jl:4`),
`status_name` (`admm.jl:28-40`), the export list (`src/PureOSQP.jl:27-31`), MOI
`_TERMINATION` (→ `MOI.NUMERICAL_ERROR`) and `ResultCount` (→ `0`,
`MathOptInterfaceExt.jl:203-206`), `docs/src/algorithm.md:383-386`, the API table.
`has_solution` is `false` for it.

### 8.7 Infeasibility detection (decision for the user, §10.4)

v1: certificate tests, not a homogeneous embedding.

- The tests are the package's, formulation-independent, after S4 on `(prob, buffer, eps)`.
  They project the buffer in place (`termination.jl:209`), so `IPMWorkspace` owns `cert_x`,
  `cert_y`; `build_solution` reads them.
- At every termination check, and every iteration once the stall or divergence guard fires,
  the IPM copies `(Δx, Δy)` of the last step and `(x/‖x‖∞, y/‖y‖∞)` into the buffers and runs
  both tests at `eps_prim_inf`/`eps_dual_inf`, with the `*_INACCURATE` retry at ten times.
- Stall rule: `α < 1e-8` on three consecutive iterations, or `μ` not decreasing for ten,
  triggers the tests; if neither fires the run ends `NUMERICAL_ERROR` with the last point.
  Divergence guard: `‖x‖∞` or `‖y‖∞` above `1/sqrt(eps(T))` relative to the data.
- Whether a Mehrotra IPM's direction on an infeasible problem converges to a Farkas
  certificate or merely stalls is not established for this code; S9's random infeasible
  cases (`n = 20, m = 40`, primal and dual) decide. The c-suite six
  (`c_suite_tests.jl:104-123`) are a regression, not the evidence.
- HSD (S9b) stays available: a bordered solve built entirely on `solve_system!` (two backend
  solves plus a rank-one update per system, no backend change). Its cost is exactly a doubled
  product count for operators, which is why it is not the v1 default.

### 8.8 Backends and selection

`IPMSelection` (§5). `IndirectCG` under `:ipm` follows §9. Kronecker declines. GPU arrays are
refused for `:ipm` through the GPU extension's `choose_backend` method.

### 8.9 Equilibration

Identical to ADMM: the workspace holds `prob` with `D E c`; iterates are scaled;
`Solution.x = D ⊙ x̃`, `Solution.y = E ⊙ ỹ / c`; residuals are reported unscaled by the shared
kernels. `scaling = 0` turns it off; operators need `scaling = 0` or `probe = true`
(`operator.jl:49-66`).

### 8.10 v1 support matrix

| feature | IPM v1 |
|---|---|
| `update!(q, l, u, P, A)` | yes; validate → adopt → reclassify; no refactorization at update |
| `warm_start!` / `x0, y0` | seed (§8.3); sets `seeded` |
| `cold_start!` | zeroes `x, y`, clears `seeded` |
| `update_settings!` | yes; `linsys`, `scaling` fixed; `reg_*` free |
| `polishing` | yes, through the kernel; required before derivatives |
| derivatives | yes on `SOLVED` with polishing |
| MOI | `algorithm = :ipm`; `BarrierIterations = iter`; `NUMERICAL_ERROR` mapped |
| `time_limit`, `Ctrl-C` | as ADMM |
| `verbose` | `Core.stdout`, row: `iter obj prim_res dual_res μ α cg_iters` |
| operators (`ProductOperator`, LinearMaps, SciMLOperators) | `scaling = 0` or `probe = true`; `P` declared `posdef`; backend `IndirectCG` only; no polishing, no derivatives, no `:kkt`; reachable by `:auto` only after the §9.5 gate, always by `linsys = :indirect` |
| accelerator, GPU arrays, `profile_primdual` | refused by name / absent |

`IPMSettings{T}`: `max_iter = 100`, `time_limit = Inf`, `eps_abs = eps_rel = 1e-8`,
`eps_prim_inf = eps_dual_inf = 1e-8`, `scaling = 10`, `check_termination = 1`,
`check_dualgap = true`, `scaled_termination = false`, `reg_primal = reg_dual =
max(1e-8, sqrt(eps(T)))`, `reg_indirect = 1e-4` (provisional), `max_reg_bumps = 5`,
`refine_iter = 1`, `step_fraction = 0.99`, `cg_max_iter = 200`, `cg_tol_fraction = 0.1`,
`cg_tol_reduction = 10`, `cg_fail_limit = 3`, `precond_rows = 20`, `precond_probes = 8`,
`polishing = false`, `polish_refine_iter = 3`, `delta = 1e-6`, `warm_starting = true`,
`verbose = false`, `linsys = :auto`. Validated in the constructor as `Settings{T}` is.

### 8.11 Validation plan

1. **Referee** `kkt_residuals` (`test/helpers.jl:36-60`) `< 1e-5` on the structural corpus
   (`corpus_tests.jl:12-42`) under `:ipm` with `:auto` and `:kkt`; on `random_qp` sizes; on
   every structured family of `selection_tests.jl:26-103`, with the backend name asserted.
2. **Oracle**: objective vs `osqp_ref` to `1e-6·max(1,|obj|)`; `x` vs the ADMM solution at
   `eps = 1e-9` to `1e-6`.
3. **Clarabel** (`bench/ipm_vs_clarabel.jl`, Clarabel via `bench/solvers.jl:46-51`): `x` to
   `1e-5` relative on the six `CASES`, iteration counts and wall clock side by side.
4. **Iteration sanity** (test): `iter ≤ 40` on corpus and c-suite; `≤ 25` on the dense random
   QPs. The warm-started re-solve count is recorded in the snapshot, not asserted.
5. **c-suite** under `:ipm`, plus S9's random infeasible cases.
6. **Generic `T`**: `ForwardDiff.Dual` on `:auto` (reduced path, no `bunchkaufman!`),
   `Float32` on `:auto` (reaches `FullKKT`), `BigFloat` on one small case.
7. **Operators**: §9.5.

### 8.12 StrictMode and trim

Audit rows per backend: `ipm_step!(W)` and `ipm_residuals!(W)` hot (`typestable`,
`noalloc`); `solve_system!`/`solve_multiplier!` hot; `refactor_weights!`, `factorize!`,
`check_termination(W, Bool)`, `solve!(W)` warm; the `:indirect` row measured. The docs state
that the IPM's per-iteration allocation is the backend's factorization. Rules: masks and
classes preallocated; every elementwise update a two-schedule function in `elementwise.jl`
style (`max_step`, `complementarity!`, `weights!`, `recover_slack_steps!`); loops over
`eachindex`, no `findall`, no broadcasting on `Vector`; `norm_inf` only; `IPMSettings{T}`
concrete; `lazy"…"` messages; verbose via `Core.stdout`. Trim entries: `solve_ipm_default`,
`solve_ipm_kkt`, `solve_ipm_unscaled`, `solve_ipm_polish`, `solve_ipm_diagonal`,
`solve_ipm_sparse_kkt`, `solve_ipm_indirect`, `solve_ipm_operator`, `setup_ipm_update`,
`derivatives_ipm`.

---

## 9. Operators under the IPM

The case the user cares about most: `P` and `A` as `LinearMap`s or SciMLOperators (through
`ProductOperator`, `operator.jl`), which supply products and nothing else. Under the IPM such a
pair has exactly one backend, `IndirectCG`, and the reduced matrix it must solve,
`P̃ + δ_p I + Ãᵀ diag(w) Ã`, has a spectrum that splits as `μ → 0` — `O(w_max‖ã‖²)` on the
active-row span, `O(λ_min(P̃) + δ_p)` on the rest (**literature**: Wright 1998). Unpreconditioned
CG on that is the failure the design has to avoid. ADMM's matrix-free path is already
1.4–5.9× ADMM's direct iteration count with a *fixed* `ρ` (`bench/results/indirect_backend.json`),
so nothing carries over by assumption.

### 9.1 Regularization: consequences of §8.5 for every backend

Proximal regularization is what makes the operator case possible, and it is the same model
the direct backends run: `δ_d` bounds `w ≤ 1/δ_d`, hence `κ(reduced) ≤ (λ_max(P̃) +
‖Ã‖²/δ_d)/(λ_min(P̃) + δ_p)`. Direct backends use the small default; `IndirectCG` uses
`reg_indirect` (provisional `1e-4`; the spike sets it). Larger `δ` costs outer iterations —
the spike measures how many. Consequences elsewhere: none for the backend interface (`w_inv`
already carries `δ_d`); the termination residuals are the original problem's, so the
reported accuracy is not the proximal subproblem's; `update_settings!(reg_indirect = …)`
changes nothing but the next weights.

### 9.2 Inner tolerance and budget

Inner tolerance per solve (**literature**: inexact Newton, Dembo–Eisenstat–Steihaug):

    atol_k = η · min(μ_k, ‖r_k‖∞),   η = cg_tol_fraction (0.1),   r_k = (r_d, r_l, r_u, r_e)

set through `set_tolerance_level!(ls, min(μ_k, ‖r_k‖∞))` before each outer iteration; the
backend floors it at `eps(T)·max(1, ‖rhs‖∞)` as today. The IPM checks the achieved reduced
residual after each solve (one operator application, three products): the first KKT block
row *is* the reduced residual and the second is zero by construction of `Δy`, so this is the
KKT residual of the step. `last_solve_converged(ls)` reports whether CG met `atol` within
`cg_max_iter`.

Budget per outer iteration: two solves × `cg_max_iter` (default `200`), no refinement.
Products per outer iteration ≈ `2·(cg_iters) · 3 + precond_rows + 3·precond_probes`.
`Solution.cg_iters` reports the total inner iterations of the solve; the verbose row prints
the per-iteration count.

Failure: `cg_fail_limit` (default `3`) consecutive solves that miss `atol` end the run
`NUMERICAL_ERROR` with the last point (never `MAX_ITER_REACHED`, which would misreport a
linear-algebra failure as a convergence budget).

### 9.3 Preconditioner interface

```julia
"""
Preconditioners approximate the inverse of the reduced matrix and are applied through
`mul!(y, M, x)` (Krylov's `M` keyword with `ldiv = false`). `update_preconditioner!(M, prob,
wt)` is called from `IndirectCG`'s `factorize!` and `refactor_weights!` with the current
weights, so `M` can follow `w` every outer iteration. A user type needs those two methods.
"""
update_preconditioner!(M, prob, wt) = M                     # identity: nothing to update

struct IdentityPreconditioner end                            # mul! is copyto!
struct JacobiPreconditioner{V}; dinv::V; end                 # reduced_diagonal!, the hook at operator.jl:218
struct ProbedWoodburyPreconditioner{T,V,M,F}                 # product-only (§9.4)
    dinv::V; V::M; Y::M; cap::M; fact::F; rows::Vector{Int}; ky::V; probe::V; probes::Int
end
```

`setup(…; preconditioner = nothing | M)`; `nothing` means the built-in choice: `Jacobi` when
both operands are materializable or the wrapped type overrides `reduced_diagonal!`
(`operator.jl:208-228` documents that hook; it is now stated to be called at every
`refactor_weights!` with the current `w`), otherwise `ProbedWoodbury`. `IndirectCG{T,V,K,M}`
carries `M` as a type parameter, so the choice is static and trim-safe. Under ADMM the default
is `Jacobi`/identity exactly as today.

### 9.4 Built-in product-only preconditioner

`ProbedWoodburyPreconditioner`: `R ≈ C + VᵀW_KV`, `C` diagonal, `V` the `k` rows of `Ã` with
the largest `w_i` (the active rows, which carry the large eigenvalues).

- Rows: `k = precond_rows` indices with the largest `w_i`, selected by an `O(mk)` loop into a
  preallocated index vector; row `i` of `Ã` is `Ãᵀeᵢ` through `mul_At!(prob, ·)` — `k` adjoint
  products per outer iteration.
- Core `C = c D² diag(P̃) + δ_p + Σ_{i∉K} w_i ã_i²`: exact through `reduced_diagonal!` when
  entries are available, otherwise estimated with `precond_probes` Hutchinson probes
  (`diag(M) ≈ mean(v ⊙ Mv)`, `v` random ±1; **literature**: Bekas–Kurzak–Saad 2007) — `3s`
  products — floored at `δ_p`.
- Woodbury: `cinv`, `Y = V C⁻¹`, `cap = W_K⁻¹ + V C⁻¹ Vᵀ` (`k×k` Cholesky), apply
  `y = C⁻¹x − Yᵀ cap⁻¹ Y x` — the arithmetic of `DiagonalLowRank.refresh_core!` and
  `solve_system!` (`lowrank.jl:111-166`), reused rather than copied.
- Cost model in products per outer iteration: `k + 3s` to build, `0` to apply (dense
  small matrix–vector products in `n` and `k`).

### 9.5 Measurement gates

Two measurements, in this order:

**Spike (before S2a; §6 S-spike)** — a bench-only prototype answers, on LinearMap-over-dense
QPs with `n ∈ {200, 500, 1000}`, `κ(A) ∈ {1e0, 1e3, 1e6}`, active fractions `{0.1, 0.5,
0.9}`: (i) CG iterations per solve versus outer iteration at `δ = 1e-8` unpreconditioned;
(ii) the same at `δ_d ∈ {1e-4, 1e-2}` and the outer-iteration cost of each; (iii) the effect
of exact Jacobi and of probed Woodbury with `k ∈ {5, 20}`; (iv) whether a restarted
"refinement" adds anything over a larger `cg_max_iter` (expected: nothing). Verdict recorded
in `bench/results/ipm_matrixfree_spike.json` and in §10.

**Ship gate (S11; `bench/ipm_matrixfree.jl`)** — operator-IPM vs dense-IPM (`FullKKT`, the
exact reference) vs ADMM-operator, LinearMap over dense matrices, `n ∈ {200, 500, 1000,
2000}`, same sweeps, BLAS single-threaded on neuromancer (timings are indicative; nothing is
gated on them), reporting outer iterations, total CG iterations, total products, wall clock,
achieved KKT residual, with each preconditioner. Gates:

- **G1 correctness**: referee `≤ 1e-5` at `eps = 1e-6` on every case with `n ≤ 1000` within
  the budget.
- **G2 bounded inner work**: median CG iterations per solve in the last three outer
  iterations `≤ 10×` those of the first three, with the built-in preconditioner.
- **G3 speed against ADMM-operator**: measured and published in the docs; decides nothing.

Outcomes: G1 or G2 fails at every `δ`/preconditioner ⇒ operator IPM does not ship and `:ipm`
refuses operators by name. G1 and G2 pass ⇒ available by `algorithm = :ipm` with an operator
pair. In v1 no automatic choice ever selects the IPM for operators; the caller chooses the
algorithm.

### 9.6 Ordering

The spike is the first thing after S0 because a negative result changes §8.5 and §9 before
any of S8–S11 is written, and it needs no refactor to run. HSD (S9b) is decided after S9's
infeasible cases, before S11, because its cost lands on the operator case.

---

## 10. Decisions (taken with the user, 2026-09-15)

1. **Regularization model:** primal–dual proximal regularization for every backend (§8.5).
2. **Operator IPM in v1:** implemented in S11; ships if G1 and G2 pass (§9.5), otherwise
   `:ipm` refuses operators by name. No automatic choice selects the IPM for operators in v1.
3. **Gates:** G1 as stated; G2 at `10×`; G3 measured and published, not a gate. The spike and
   the S11 benchmark use the §9.5 sweep (LinearMap over dense, `n` 200–2000) and run on
   neuromancer; timings there are indicative.
4. **Infeasibility:** certificate tests on directions with a stall rule (§8.7); HSD only if
   S9's random infeasible cases are not detected.
5. **`NUMERICAL_ERROR`** is added to `Status`, with the MOI mapping and docs changes.
6. **Default tolerances:** `1e-8` for `:ipm`, `1e-3` for `:admm`, documented as differing.
7. **Timing checks during the refactor (S2a):** run on neuromancer, interleaved ABBA with the
   S0 tree; indicative, not a verdict.
