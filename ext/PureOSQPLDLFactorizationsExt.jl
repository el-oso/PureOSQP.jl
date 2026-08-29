"""
    PureOSQPLDLFactorizationsExt

Factors the reduced matrix with LDLFactorizations.jl instead of CHOLMOD.

Loading LDLFactorizations changes nothing a caller can observe except speed: the same
problems select the same kind of backend, take the same iterations, and reach the same
answers. What changes is who computes the factorization, and how much of it has to be
rebuilt when `ρ` moves.

Two things make it worth the switch, both measured on the OSQP benchmark suite's own
problems, whose factors hold two to three nonzeros per column:

- The numeric factorization is 2.3–3.1× faster than CHOLMOD's and allocates nothing:
  9.1 µs against 21.4 (Lasso), 25.2 against 58.4 (Huber). A refactorization happens every
  time `ρ` is retuned, inside the solve loop, so this is not a setup-only saving.
- `L` and `D` are Julia arrays owned by the factorization. The CHOLMOD path has to extract
  them with `sparse(F.L)` and a transpose on every refactorization — 23 µs on Huber, and a
  fresh matrix each time for a pattern that never changes.

Only the factorization is delegated. The substitutions and the diagonal scaling stay in
this package: measured against LDLFactorizations' own solve they are as fast or faster
(2.74 µs against 3.02 on Lasso, 8.10 against 8.17 on Huber), and they are the code the
allocation guarantee is proved on.
"""
module PureOSQPLDLFactorizationsExt

using PureOSQP: PureOSQP
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, nonzeros
using LDLFactorizations: LDLFactorizations, ldl_analyze, ldl_factorize!

"""
    fact_L(F) -> SparseMatrixCSC
    fact_perm(F) -> Vector{Int}

The factor and the permutation, pulled out of an `LDLFactorization` once.

`LDLFactorization` reaches these through a `getproperty` that assembles them from its
internal arrays, so each access allocates and infers as `Any`. Reading them per iteration
would put both on the hot path — which is what the backends store them for, exactly as the
CHOLMOD backends store theirs rather than reaching into a foreign factor.

Annotated rather than trusted: an unannotated `getproperty` result costs the caller its
type stability whatever the field turns out to hold.
"""
fact_L(F)::SparseMatrixCSC{Float64, Int} = F.L
fact_perm(F)::Vector{Int} = F.P

"""
    SparseLDL{T,V,G,F} <: PureOSQP.LinearSystem

The reduced backend factored by LDLFactorizations, solved by this package.

`R = Lᵤ D Lᵤᵀ` with `Lᵤ` unit lower triangular. LDLFactorizations stores `Lᵤ` strictly below
the diagonal and `D` separately, which is exactly the form the substitutions want — the unit
diagonal is never loaded and never divided by.

`gram` rebuilds `R`'s values in place when `ρ` moves; `fact` holds the ordering and the
symbolic analysis, so a refactorization is numeric only.
"""
mutable struct SparseLDL{T <: Real, V <: AbstractVector{T}, G, F} <: PureOSQP.LinearSystem
    gram::G
    fact::F
    L::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    dinv::Vector{T}
    permuted::V
end

PureOSQP.backend_name(::SparseLDL) = :ldlfactorizations

"""
    PureOSQP.ldl_backend(gram, proto, n, fill_limit) -> SparseLDL or nothing

Analyse and factor `gram.R`, and take the backend if the fill stays under the limit.

`ldl_analyze` chooses its own fill-reducing ordering; the fill it reports is what decides,
exactly as the CHOLMOD path decides on `nnz(F.L)`. Returning `nothing` leaves that path to
answer instead.
"""
function PureOSQP.ldl_backend(gram, proto::AbstractVector{T}, n::Integer, fill_limit::Real) where {T <: Real}
    R = gram.R
    M = Symmetric(R, :U)
    fact = try
        ldl_analyze(M)
    catch
        return nothing
    end
    ldl_factorize!(M, fact)
    # `D` is singular exactly when the reduced matrix is, which for `P̃ + σI + Ãᵀ diag(ρ) Ã`
    # means the problem was not convex after all. Hand it back rather than divide by zero.
    any(iszero, fact.d) && return nothing
    nnz(fact.L) < fill_limit * n^2 || return nothing
    Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt).check_factor(fact.L, n)
    return SparseLDL{T, typeof(proto), typeof(gram), typeof(fact)}(
        gram, fact, fact_L(fact), fact_perm(fact), inv.(fact.d), similar(proto, T, n)
    )
end

function PureOSQP.factorize!(ls::SparseLDL{T}, ws)::Bool where {T}
    P, A = ws.P, ws.A
    Ext = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)
    if !Ext.describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries elsewhere, so both the slot map
        # and the analysis built on its pattern are stale.
        ls.gram = Ext.reduced_gram(T, P, A, ws.n)
        R = Ext.refill!(ls.gram, P, A, ws.rho_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
        ls.fact = ldl_analyze(Symmetric(R, :U))
    else
        Ext.refill!(ls.gram, P, A, ws.rho_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
    end
    ldl_factorize!(Symmetric(ls.gram.R, :U), ls.fact)
    d = ls.fact.d
    any(iszero, d) && return false
    ls.L = fact_L(ls.fact)
    ls.perm = fact_perm(ls.fact)
    Ext.check_factor(ls.L, ws.n)
    # In place: `D` has the same length every time, and a refactorization runs inside the
    # solve loop whenever `ρ` is retuned.
    length(ls.dinv) == length(d) || resize!(ls.dinv, length(d))
    ls.dinv .= inv.(d)
    return true
end


"""
    unit_forward!(x, L, N)
    unit_backward!(x, L, N)

Substitution against a unit lower triangular factor and its transpose, in place.

The diagonal is implied, so neither loop loads it and neither divides: forward scatters down
each column, backward gathers back up the same column. `L` holds only the strictly lower
entries, which is how LDLFactorizations stores it.

Both run unchecked, which [`check_factor`](@ref) is what makes safe — see there for why the
check is hoisted out of the loops and what it is worth. `@simd ivdep` is claimed only on the
forward loop: within a column the row indices are distinct, so its scattered writes carry no
dependency between iterations. The backward loop accumulates into a scalar, and vectorizing
a floating-point reduction reassociates it — which would move the iterates and cost the
property that this solver takes the same steps as the reference implementation.
"""
function unit_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        xj = x[j]
        @simd ivdep for p in colptr[j]:(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function unit_backward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in N:-1:1
        s = x[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            s -= vals[p] * x[rows[p]]
        end
        x[j] = s
    end
    return x
end

function PureOSQP.solve_system!(ls::SparseLDL{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    rhs = PureOSQP.reduced_rhs!(ws, rhs_x, rhs_z)
    perm, work, n = ls.perm, ls.permuted, ws.n
    L, dinv = ls.L, ls.dinv
    # R[perm, perm] = Lᵤ D Lᵤᵀ, so the solve is a permutation, two substitutions and the
    # diagonal, all over buffers this backend owns.
    for i in 1:n
        work[i] = rhs[perm[i]]
    end
    unit_forward!(work, L, n)
    for i in 1:n
        work[i] *= dinv[i]
    end
    unit_backward!(work, L, n)
    x = ws.xtilde
    for i in 1:n
        x[perm[i]] = work[i]
    end
    ws.m > 0 && PureOSQP.mul_A!(ws.ztilde, ws, x)
    return nothing
end

"""
    LDLKKT{T,V,F} <: PureOSQP.LinearSystem

The full quasi-definite KKT backend, factored by LDLFactorizations.

    K = ⎡P̃ + σI    Ãᵀ  ⎤
        ⎣Ã      −diag(ρ⁻¹)⎦

`K` is indefinite, which is not an obstacle: a quasi-definite matrix has an `LDLᵀ` under any
symmetric permutation, so the fill-reducing ordering can be chosen once and reused without
pivoting for stability. `D` carries the signs.

Unlike [`SparseLDL`](@ref) this recovers `z̃` from the eliminated multiplier, so the solve
needs no product against `A`.
"""
mutable struct LDLKKT{T <: Real, V <: AbstractVector{T}, G, F} <: PureOSQP.LinearSystem
    gram::G
    fact::F
    L::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    dinv::Vector{T}
    work::V
end

PureOSQP.backend_name(::LDLKKT) = :ldl_kkt

function PureOSQP.ldl_kkt_backend(
        gram, proto::AbstractVector{T}, n::Integer, m::Integer, fill_limit::Real
    ) where {T <: Real}
    M = Symmetric(gram.K, :U)
    fact = try
        ldl_analyze(M)
    catch
        return nothing
    end
    ldl_factorize!(M, fact)
    any(iszero, fact.d) && return nothing
    # Against the dense reduced factorization this replaces, whose cost is `n²`.
    nnz(fact.L) < fill_limit * n^2 || return nothing
    Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt).check_factor(fact.L, n + m)
    v = similar(proto, T, n + m)
    return LDLKKT{T, typeof(v), typeof(gram), typeof(fact)}(
        gram, fact, fact_L(fact), fact_perm(fact), inv.(fact.d), v
    )
end

function PureOSQP.factorize!(ls::LDLKKT{T}, ws)::Bool where {T}
    Ext = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)
    P, A = ws.P, ws.A
    if !Ext.describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries elsewhere, so the slot map and
        # the analysis built on its pattern are both stale.
        ls.gram = Ext.kkt_gram(T, P, A, ws.n, ws.m)
        K = Ext.refill_kkt!(ls.gram, P, A, ws.rho_inv_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
        ls.fact = ldl_analyze(Symmetric(K, :U))
    else
        Ext.refill_kkt!(ls.gram, P, A, ws.rho_inv_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
    end
    ldl_factorize!(Symmetric(ls.gram.K, :U), ls.fact)
    d = ls.fact.d
    any(iszero, d) && return false
    ls.L = fact_L(ls.fact)
    ls.perm = fact_perm(ls.fact)
    Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt).check_factor(ls.L, ws.n + ws.m)
    length(ls.dinv) == length(d) || resize!(ls.dinv, length(d))
    ls.dinv .= inv.(d)
    return true
end

function PureOSQP.solve_system!(ls::LDLKKT{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    n, m = ws.n, ws.m
    N = n + m
    perm, work = ls.perm, ls.work
    L, dinv = ls.L, ls.dinv
    # Permute straight out of the two right-hand sides: the assembled vector is never needed
    # in its own order.
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    unit_forward!(work, L, N)
    for i in 1:N
        work[i] *= dinv[i]
    end
    unit_backward!(work, L, N)
    # And scatter straight into the outputs.
    for i in 1:N
        p = perm[i]
        if p <= n
            ws.xtilde[p] = work[i]
        else
            ws.ztilde[p - n] = work[i]
        end
    end
    for i in 1:m
        ws.ztilde[i] = rhs_z[i] + ws.rho_inv_vec[i] * ws.ztilde[i]
    end
    return nothing
end

# No `trim_compat` claim: the factorization is a foreign package's, and the trim entry points
# cover the dense path.
@verify SparseLDL
@verify LDLKKT

end # module PureOSQPLDLFactorizationsExt
