"""
    PureOSQPSparseArraysExt

Sparse-aware traversals and a sparse-forming reduced backend, for `SparseMatrixCSC`.

PureOSQP's per-iteration products already go through `mul!`, which sparse matrices handle
well on their own. Equilibration and the factorization are different: they walk the
caller's matrices entry by entry, and the generic loop visits every structural zero and
reaches each one through `M[i, j]`, which on CSC is a binary search within the column. On a
400×200 matrix at 1% density that is 80 000 searches per sweep where 800 direct reads would
do, and it made equilibration ten times slower on a sparse matrix than on a dense one.

The four column traversals below walk `nzrange` instead. The extension also supplies two
reduced backends — [`SparseFormedInverse`](@ref), which forms `Ãᵀ diag(ρ) Ã` from the stored
entries but still factors densely, and [`SparseCholmod`](@ref), which also factors sparsely
when the factor stays sparse enough to pay — and a convexity test that does not densify `P`.
Nothing else in the solver needs to know the storage.
"""
module PureOSQPSparseArraysExt

using PureOSQP: PureOSQP
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric, Diagonal, LowerTriangular, UpperTriangular,
    UnitLowerTriangular, UnitUpperTriangular, I, diag,
    cholesky, cholesky!, ldlt, ldlt!, issuccess, ldiv!, transpose
using SparseArrays: SparseMatrixCSC, issparse, nnz, nzrange, rowvals, nonzeros, sparse

@inline function PureOSQP.weighted_colmax(
        ::Type{T}, M::SparseMatrixCSC, j::Integer, w::AbstractVector
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        r = max(r, w[rows[k]] * abs(T(vals[k])))
    end
    return r
end

@inline function PureOSQP.weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::SparseMatrixCSC, j::Integer,
        w::AbstractVector, s
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        v = abs(T(vals[k]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

@inline function PureOSQP.scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] = f(T(vals[k]), i)
    end
    return dest
end

@inline function PureOSQP.add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] += f(T(vals[k]), i)
    end
    return dest
end


"""
    SparseFormedInverse{T,M} <: PureOSQP.ReducedInverse

The reduced backend for a sparse `A`, which forms `Ãᵀ diag(ρ) Ã` from the stored entries
instead of through a dense product.

[`PureOSQP.ReducedCholesky`](@ref) writes the scaled `Ã` into an `m×n` dense buffer so that
one `syrk` produces the reduced matrix. That buffer is mostly zeros when `A` is sparse, and
the `syrk` does `mn²` flops to multiply them: on a 2000×4000 problem at 0.25% density it is
about 63% of a refactorization and 65% of the workspace. Accumulating over the stored
entries instead costs `Σᵢ nnzᵢ²` and needs no buffer, so this backend holds only the
inverse.

The reduced matrix itself is still dense, and everything after it is forming — the
Cholesky, the inversion, the per-iteration `symv` — is exactly what the dense backend does.
"""
struct SparseFormedInverse{T <: Real, M <: AbstractMatrix{T}} <: PureOSQP.ReducedInverse
    Rinv::M
end

"""
Density above which the dense product forms the reduced matrix faster than accumulating
over stored entries, so a `SparseMatrixCSC` is served by [`PureOSQP.ReducedCholesky`](@ref)
anyway.

Accumulation does `Σᵢ nnzᵢ²` flops against the dense product's `mn²`, but it writes into `R`
by scattered index where `syrk` streams contiguous memory, and that constant is worth about
an order of magnitude. Measured, the two cross near 20% density; this is set below that, and
matches the density above which callers are already advised to hand over dense copies.
"""
const DENSE_FORM_DENSITY = 0.1

function PureOSQP.choose_backend(
        P, A::SparseMatrixCSC, proto::AbstractVector{T}, n::Integer, m::Integer,
        D, E, c, rho_vec, sigma
    ) where {T <: Real}
    if n > 0 && m > 0 && nnz(A) > DENSE_FORM_DENSITY * m * n
        return (PureOSQP.ReducedCholesky(proto, n, m), false)
    end
    # Factoring sparsely beats inverting densely only when the factor stays sparse, which
    # is a property of the pattern and so is measured here rather than assumed. Both of
    # these factor the real matrix to find out, so a backend they return is ready to solve.
    kkt = sparse_kkt_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(kkt) || return (kkt, true)
    ldl = cholmod_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(ldl) || return (ldl, true)
    Rinv = similar(proto, T, n, n)
    return (SparseFormedInverse{T, typeof(Rinv)}(Rinv), false)
end

PureOSQP.backend_name(::SparseFormedInverse) = :sparse_formed

"""
    csr_rows(A) -> (rowptr, colind, nzval)

Group `A`'s stored entries by row, as `A[i, colind[p]] == nzval[p]` for
`p in rowptr[i]:(rowptr[i + 1] - 1)`, with `colind` ascending within each row.

The accumulation needs each row's nonzeros together and CSC stores columns, so something
has to transpose. `copy(transpose(A))` would, but it goes through the `SparseMatrixCSC`
constructor, whose dimension validation raises through a closure that formats its message —
enough runtime dispatch on a branch that never fires to cost `factorize!` the type-stability
guarantee. Three plain vectors carry everything the accumulation reads.
"""
function csr_rows(A::SparseMatrixCSC{Tv}) where {Tv}
    m, n = size(A)
    rows, vals = rowvals(A), nonzeros(A)
    rowptr = Vector{Int}(undef, m + 1)
    fill!(rowptr, 0)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    colind = Vector{Int}(undef, length(rows))
    nzval = Vector{Tv}(undef, length(vals))
    # Columns are visited in ascending order, which is what leaves `colind` ascending
    # within each row and lets `gram_upper!` take the upper triangle as `k >= j`.
    pos = copy(rowptr)
    for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            p = pos[i]
            colind[p] = j
            nzval[p] = vals[k]
            pos[i] = p + 1
        end
    end
    return (rowptr, colind, nzval)
end

"""
    gram_upper!(R, rowptr, colind, nzval, rho, E, D)

Add `Ãᵀ diag(ρ) Ã` to the upper triangle of `R`, where `Ã = diag(E) A diag(D)`.

Each row of `A` contributes the outer product of its own nonzeros, so the work is
`Σᵢ nnzᵢ²` rather than the `mn²` of a dense product against a mostly-zero buffer.

Only the upper triangle is written, which is all that is ever read: `cholesky!` is handed a
`Symmetric(R, :U)`, `potri!` writes the inverse into the same triangle, and the solve
multiplies by `Symmetric(Rinv, :U)`. The inner loop starts at `p` rather than at the row's
first entry because [`csr_rows`](@ref) leaves each row's columns ascending.
"""
function gram_upper!(
        R::AbstractMatrix{T}, rowptr::Vector{Int}, colind::Vector{Int},
        nzval::AbstractVector, rho::AbstractVector, E::AbstractVector, D::AbstractVector
    ) where {T}
    for i in eachindex(rho, E)
        ei = E[i]
        w = rho[i] * ei * ei
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop
            j = colind[p]
            wvj = w * T(nzval[p]) * D[j]
            for q in p:stop
                k = colind[q]
                R[j, k] += wvj * T(nzval[q]) * D[k]
            end
        end
    end
    return R
end

function PureOSQP.factorize!(ls::SparseFormedInverse{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    R = ls.Rinv
    fill!(R, zero(T))
    if m > 0
        # Rebuilt rather than cached: `update!` may replace `ws.A`, and a cached grouping
        # would then describe a matrix the solver no longer holds. It costs O(nnz), against
        # the O(mn²) product it replaces.
        rowptr, colind, nzval = csr_rows(ws.A)
        gram_upper!(R, rowptr, colind, nzval, ws.rho_vec, ws.E, ws.D)
    end
    c, D = ws.c, ws.D
    for j in 1:n
        dj = D[j]
        PureOSQP.add_scaled_col!(T, R, ws.P, j, (p, i) -> c * D[i] * p * dj)
    end
    for i in 1:n
        R[i, i] += ws.settings.sigma
    end
    F = cholesky!(Symmetric(R); check = false)
    issuccess(F) || return false
    PureOSQP.invert_spd!(R, F)
    return true
end


"""
    SparseKKT{T,V,F} <: PureOSQP.LinearSystem

Factors the full `(n+m)×(n+m)` quasi-definite system sparsely, with CHOLMOD's `ldlt`.

The reduced form squares `A`, so one dense row makes `R` dense however sparse the rest of it
is. The full system does not: a dense row of `A` stays one sparse row of

    K = ⎡P̃ + σI    Ãᵀ  ⎤
        ⎣Ã      −diag(ρ⁻¹)⎦

On the OSQP suite's Portfolio class — 0.9% dense `A`, one row touching every column — `R`
comes out 99% dense while `K`'s factor is 0.3% dense, and factoring `K` is 7.5× faster than
factoring `R`.

`K` is quasi-definite: `P̃ + σI` is positive definite and `−diag(ρ⁻¹)` negative definite. A
quasi-definite matrix has an `LDLᵀ` factorization under *any* symmetric permutation, which
is why a fill-reducing ordering chosen once can be reused without pivoting for stability —
the property upstream's own solver rests on.

`L`, `D⁻¹` and the permutation are extracted rather than solved through, because CHOLMOD's
`ldiv!` allocates on every call and the hot path may not. See
[`PureOSQP.reduced_rhs!`](@ref) for the reduced backends' equivalent.
"""
mutable struct SparseKKT{T <: Real, V <: AbstractVector{T}, F} <: PureOSQP.LinearSystem
    K::SparseMatrixCSC{T, Int}
    fact::F
    L::SparseMatrixCSC{T, Int}
    dinv::Vector{T}
    perm::Vector{Int}
    rhs::V
    work::V
end

PureOSQP.backend_name(::SparseKKT) = :sparse_kkt

"""
    kkt_sparse(T, P, A, rho_inv, E, D, c, sigma) -> SparseMatrixCSC

The scaled quasi-definite KKT matrix, kept sparse.

Equilibration comes out of the blocks the same way it does for the reduced matrix:
`P̃ = c·diag(D) P diag(D)` and `Ã = diag(E) A diag(D)`, so nothing is formed scaled.
"""
function kkt_sparse(::Type{T}, P, A, rho_inv, E, D, c, sigma) where {T}
    Dg = Diagonal(D)
    Pt = SparseMatrixCSC{T, Int}(c * (Dg * sparse(Symmetric(P)) * Dg) + sigma * I)
    At = SparseMatrixCSC{T, Int}(Diagonal(E) * A * Dg)
    m = size(A, 1)
    return SparseMatrixCSC{T, Int}(
        Symmetric([Pt transpose(At); At sparse(-Diagonal(rho_inv))], :L)
    )
end

function PureOSQP.factorize!(ls::SparseKKT{T}, ws)::Bool where {T}
    K = kkt_sparse(
        T, ws.P, ws.A, ws.rho_inv_vec, ws.E, ws.D, ws.c, ws.settings.sigma
    )
    if K.colptr == ls.K.colptr && K.rowval == ls.K.rowval
        # The pattern does not depend on ρ or the equilibration factors, so every
        # refactorization after the first reuses the ordering and the symbolic phase.
        ldlt!(ls.fact, K; check = false)
    else
        ls.fact = ldlt(K; check = false)
    end
    issuccess(ls.fact) || return false
    ls.K = K
    LD = sparse(ls.fact.LD)::SparseMatrixCSC{T, Int}
    d = diag(LD)
    any(iszero, d) && return false
    ls.dinv = inv.(d)
    # `LD` packs `D` on the diagonal of a unit-triangular `L`; the solve below uses
    # `UnitLowerTriangular`, which ignores the stored diagonal.
    ls.L = LD
    check_factor(LD, ws.n + ws.m)
    ls.perm = ls.fact.p::Vector{Int}
    return true
end

"""
    ldl_forward!(x, L, N)
    ldl_backward!(x, L, N)

Substitution against the unit-lower factor of an `LDLᵀ`, in place.

`L` is CHOLMOD's packed `LD`: column `j` begins with `D[j]` and its subdiagonal entries
follow, so both loops start one past `colptr[j]` and the diagonal is applied separately.

Written out rather than left to `ldiv!(UnitLowerTriangular(L), x)`, which is the exception
rather than the rule here. Measured on this backend's own factor, resetting the vector every
sample because these solves are in place, `solve_system!` costs 9.06 µs through `ldiv!` and
5.78 µs this way. [`SparseCholmod`](@ref) does the same for its `L Lᵀ` factor, and
[`llt_forward!`](@ref) records where the two paths cross over.
"""
function ldl_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        xj = x[j]
        for p in (colptr[j] + 1):(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function ldl_backward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in N:-1:1
        s = x[j]
        for p in (colptr[j] + 1):(colptr[j + 1] - 1)
            s -= vals[p] * x[rows[p]]
        end
        x[j] = s
    end
    return x
end

function PureOSQP.solve_system!(ls::SparseKKT{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    n, m = ws.n, ws.m
    N = n + m
    perm, work = ls.perm, ls.work
    # Permute straight out of the two right-hand sides: `K[perm, perm] = L D Lᵀ`, and the
    # assembled vector is never needed in its own order.
    for i in 1:N
        p = perm[i]
        work[i] = p <= n ? rhs_x[p] : rhs_z[p - n]
    end
    ldl_forward!(work, ls.L, N)
    work .*= ls.dinv
    ldl_backward!(work, ls.L, N)
    # And scatter straight into the outputs. The eliminated multiplier gives `z̃` without
    # another product with `A`.
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

"""
    sparse_kkt_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> SparseKKT or nothing

Build the full-KKT backend when the reduced form would densify and this one would not.

Both conditions are measured. The reduced matrix is rejected on the bound in
[`densest_row`](@ref); the KKT matrix is accepted on the fill CHOLMOD reports for its
pattern, against the `n²` a dense reduced factorization would cost.

The matrix is assembled from the equilibrated data, so the factorization that answers the
fill question is the one the solver goes on to use — a backend returned from here needs no
further `factorize!`.
"""
function sparse_kkt_backend(
        P, A, proto::AbstractVector{T}, n::Integer, m::Integer, D, E, c, rho_vec, sigma
    ) where {T <: Real}
    (issparse(P) && n > 0 && m > 0) || return nothing
    # Only worth considering where the reduced form loses, which is what the dense row means.
    densest_row(A)^2 < DENSE_FACTOR_FILL * n^2 && return nothing
    K = kkt_sparse(T, P, A, inv.(rho_vec), E, D, c, sigma)
    # As for the reduced matrix: a pure-Julia LDLᵀ, where one is loaded, factors this faster
    # and needs nothing extracted from a foreign factor afterwards.
    alt = PureOSQP.ldl_kkt_backend(K, proto, n, m, DENSE_FACTOR_FILL)
    isnothing(alt) || return alt
    F = ldlt(K; check = false)
    issuccess(F) || return nothing
    LD = sparse(F.LD)::SparseMatrixCSC{T, Int}
    d = diag(LD)
    any(iszero, d) && return nothing
    # Against the dense reduced factorization this replaces, whose cost is `n²`.
    nnz(LD) < DENSE_FACTOR_FILL * n^2 || return nothing
    check_factor(LD, n + m)
    v = similar(proto, T, n + m)
    return SparseKKT{T, typeof(v), typeof(F)}(
        K, F, LD, inv.(d), F.p::Vector{Int}, v, similar(v)
    )
end

@verify SparseFormedInverse trim_compat = true


"""
Fill fraction `nnz(L)/n²` above which a sparse factorization stops paying, so a sparse `R`
is factored densely anyway.

Measured at `n = 2000` over a bandwidth sweep: against the dense inverse's `symv`, the pair
of sparse triangular solves is 12.6× faster at a fill of 0.0025, 1.53× at 0.044, and loses
at 0.086 — crossing near 0.06. The limit sits below that crossing so the accepted region
wins on *both* the factorization and the per-iteration solve, which keeps the choice from
depending on how a particular run divides its time between the two. The factorization alone
favors sparse much further out, to 5.2× at a fill of 0.28.
"""
const DENSE_FACTOR_FILL = 0.05

"""
    ReducedGram{T}

The reduced matrix's upper triangle, with the map that rebuilds its values in one pass.

Every off-diagonal entry of `Ãᵀ diag(ρ) Ã` comes from a row of `A` that stores both of its
columns, so one traversal of `A` by rows enumerates every contribution; `P`'s stored entries
and `σ` supply the rest. Which slot of `nonzeros(R)` a contribution lands in depends only on
where `P` and `A` store entries — never on what they store, nor on `ρ`, `D`, `E`, `c` or `σ`
— so the slots are found once and [`refill!`](@ref) is a pass over them.

That matters because a refactorization happens every time `ρ` moves. Rebuilding the matrix
through chained sparse products instead allocates four intermediate matrices: on the OSQP
suite's Huber problem, 2.19 MB and 120 µs against 16.6 µs and nothing here.

`aperm` and `pperm` index into `nonzeros(A)` and `nonzeros(P)` rather than holding copies of
them, so a refill always reads the values the workspace currently holds. The patterns are
kept alongside, because a `P` or `A` whose pattern has changed invalidates every slot.
"""
struct ReducedGram{T}
    R::SparseMatrixCSC{T, Int}
    acolptr::Vector{Int}
    arowval::Vector{Int}
    pcolptr::Vector{Int}
    prowval::Vector{Int}
    rowptr::Vector{Int}
    colind::Vector{Int}
    aperm::Vector{Int}
    aslot::Vector{Int}
    prow::Vector{Int}
    pcol::Vector{Int}
    pperm::Vector{Int}
    pslot::Vector{Int}
    dslot::Vector{Int}
end

"""
    SparseCholmod{T,V,F} <: PureOSQP.LinearSystem

Forms the reduced matrix sparsely *and* factors it sparsely, through SparseArrays.

`cholesky(Symmetric(R))` on a `SparseMatrixCSC` is CHOLMOD, and `cholesky!(F, R)` reuses the
symbolic analysis it already did, so every refactorization after the first pays only for the
numeric phase. The factorization is `L Lᵀ`, not `L D Lᵀ`; the solve below depends on that and
[`cholmod_backend`](@ref) checks it before selecting this backend.

This cannot be a [`PureOSQP.ReducedInverse`](@ref): that stores `R⁻¹` and solves with one
`symv`, and the inverse of a sparse matrix is dense. The Cholesky factor is kept instead and
each solve is a pair of sparse triangular solves. On a banded `R` that is the better trade by
a wide margin — at `n = 4000` a refactorization goes from 1063 ms to 0.66 ms and a solve from
1430 µs to 71 µs — and on a filled-in `R` it is worse, which [`DENSE_FACTOR_FILL`](@ref)
decides.

`L` and `perm` are extracted from the factorization rather than solved through it, because
CHOLMOD's `ldiv!` allocates a result and workspace on every call — 64 KB per solve at
`n = 2000`, which the hot path may not do. Applying the permutation and the two triangular
solves over preallocated buffers allocates nothing and measures slightly faster besides.

`Lt` holds the same factor transposed, so that back-substitution can scatter rather than
gather — see [`llt_backward!`](@ref). It costs one transpose per factorization, which is
`O(nnz(L))` against the factorization's own cost.
"""
mutable struct SparseCholmod{T <: Real, V <: AbstractVector{T}, F} <: PureOSQP.LinearSystem
    gram::ReducedGram{T}
    fact::F
    L::SparseMatrixCSC{T, Int}
    Lt::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    permuted::V
end

PureOSQP.backend_name(::SparseCholmod) = :cholmod


"""
    csr_order(A) -> (rowptr, colind, aperm)

`A`'s stored entries grouped by row: entry `p` of row `i` sits at column `colind[p]` and
holds `nonzeros(A)[aperm[p]]`, for `p in rowptr[i]:(rowptr[i + 1] - 1)`, with `colind`
ascending within each row.

The positions rather than the values, so a caller can reread `A` after its numbers change.
"""
function csr_order(A::SparseMatrixCSC)
    m, n = size(A)
    rows = rowvals(A)
    rowptr = zeros(Int, m + 1)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    colind = Vector{Int}(undef, length(rows))
    aperm = Vector{Int}(undef, length(rows))
    pos = copy(rowptr)
    # Columns visited in ascending order is what leaves `colind` ascending within each row,
    # which is what lets the pair loop take `q >= p` as the upper triangle.
    for j in 1:n
        for k in nzrange(A, j)
            p = pos[rows[k]]
            colind[p] = j
            aperm[p] = k
            pos[rows[k]] = p + 1
        end
    end
    return (rowptr, colind, aperm)
end

"""
    reduced_nnz(P, A, n) -> Int

How many entries the upper triangle of the reduced matrix stores, counted from the patterns
of `P` and `A` without building it.

This is what the fill gate needs, and forming the matrix to ask is the expensive way round:
on the OSQP suite this counts in 10–92 µs where forming it through sparse products took
42–247, and on Huber it is 21.7 µs against 128.2.

Column `j` of `AᵀA` holds a row for every column any row of `A` shares with `j`, so one pass
over `A`'s columns and the rows behind them enumerates the column, and `mark` deduplicates it
in `O(1)` per candidate. `P`'s upper triangle and the diagonal `σ` lands on complete it.

Structural, so it agrees with [`reduced_gram`](@ref) exactly. Counting the formed matrix
instead would undercount: a sparse product drops an entry that cancels to zero, and whether
one cancels depends on `ρ`.
"""
function reduced_nnz(P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer)
    rowptr, colind, _ = csr_order(A)
    arows, prows = rowvals(A), rowvals(P)
    mark = zeros(Int, n)
    total = 0
    for j in 1:n
        for p in nzrange(A, j)
            k = arows[p]
            for t in rowptr[k]:(rowptr[k + 1] - 1)
                i = colind[t]
                i <= j || continue
                if mark[i] != j
                    mark[i] = j
                    total += 1
                end
            end
        end
        for t in nzrange(P, j)
            i = prows[t]
            i <= j || continue
            if mark[i] != j
                mark[i] = j
                total += 1
            end
        end
        if mark[j] != j
            mark[j] = j
            total += 1
        end
    end
    return total
end

"""
Build the reduced matrix's pattern and the slot map that refills it.

Every array is sized before it is filled: the number of contributions is
`Σᵢ nnzᵢ(nnzᵢ + 1) / 2` over `A`'s rows, plus `P`'s upper triangle, plus the `n` diagonal
entries `σ` lands on. Growing them instead would dominate, because for a problem this
backend goes on to refuse there can be an entry per pair of columns sharing a row.
"""
function reduced_gram(::Type{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, n::Integer) where {T}
    rowptr, colind, aperm = csr_order(A)
    m = length(rowptr) - 1
    npair = 0
    for i in 1:m
        k = rowptr[i + 1] - rowptr[i]
        npair += k * (k + 1) ÷ 2
    end
    prows = rowvals(P)
    nup = 0
    for j in 1:n, k in nzrange(P, j)
        nup += prows[k] <= j
    end
    N = npair + nup + n
    crow = Vector{Int}(undef, N)
    ccol = Vector{Int}(undef, N)
    t = 0
    for i in 1:m
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop, q in p:stop
            t += 1
            crow[t] = colind[p]
            ccol[t] = colind[q]
        end
    end
    prow = Vector{Int}(undef, nup)
    pcol = Vector{Int}(undef, nup)
    pperm = Vector{Int}(undef, nup)
    s = 0
    for j in 1:n, k in nzrange(P, j)
        i = prows[k]
        i <= j || continue
        s += 1
        prow[s] = i
        pcol[s] = j
        pperm[s] = k
        t += 1
        crow[t] = i
        ccol[t] = j
    end
    for j in 1:n
        t += 1
        crow[t] = j
        ccol[t] = j
    end
    R, slot = pattern_from(T, crow, ccol, n)
    return ReducedGram{T}(
        R, copy(A.colptr), copy(rowvals(A)), copy(P.colptr), copy(prows),
        rowptr, colind, aperm, slot[1:npair],
        prow, pcol, pperm, slot[(npair + 1):(npair + nup)],
        slot[(npair + nup + 1):end],
    )
end

"""
    pattern_from(T, crow, ccol, n) -> (R, slot)

An `n×n` sparse matrix holding each `(crow[t], ccol[t])` once, and for each `t` the index
into `nonzeros(R)` where that contribution accumulates.

CHOLMOD requires row indices ascending within a column, so the contributions are ordered by
row and then, stably, by column. Both passes are counting sorts over keys already known to
lie in `1:n`, which is `O(N + n)` — where a comparison sort per column costs `O(N log N)` and
measured 8× the sparse products this map replaces.
"""
function pattern_from(::Type{T}, crow::Vector{Int}, ccol::Vector{Int}, n::Integer) where {T}
    N = length(crow)
    order = counting_order(ccol, n, counting_order(crow, n, 1:N))
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, N)
    slot = Vector{Int}(undef, N)
    nz = 0
    p = 1
    for j in 1:n
        colptr[j] = nz + 1
        last_row = 0
        while p <= N && ccol[order[p]] == j
            t = order[p]
            i = crow[t]
            if i != last_row
                nz += 1
                rowval[nz] = i
                last_row = i
            end
            slot[t] = nz
            p += 1
        end
    end
    colptr[n + 1] = nz + 1
    resize!(rowval, nz)
    return (SparseMatrixCSC(n, n, colptr, rowval, zeros(T, nz)), slot)
end


"""
    counting_order(key, n, idx) -> Vector{Int}

The indices in `idx` reordered by `key`, stably, for keys in `1:n`.

Stability is what lets two passes sort by a pair of keys: ordering by row first and by column
second leaves the contributions in column-major order with rows ascending.
"""
function counting_order(key::Vector{Int}, n::Integer, idx)
    counts = zeros(Int, n + 1)
    for t in idx
        counts[key[t] + 1] += 1
    end
    counts[1] = 1
    for j in 1:n
        counts[j + 1] += counts[j]
    end
    out = Vector{Int}(undef, length(idx))
    for t in idx
        k = key[t]
        out[counts[k]] = t
        counts[k] += 1
    end
    return out
end

"Whether `g`'s slots still describe these matrices, which they do unless a pattern changed."
function describes(g::ReducedGram, P::SparseMatrixCSC, A::SparseMatrixCSC)
    return g.acolptr == A.colptr && g.arowval == rowvals(A) &&
        g.pcolptr == P.colptr && g.prowval == rowvals(P)
end

"""
    refill!(g, P, A, rho, E, D, c, sigma) -> SparseMatrixCSC

Rebuild `g.R` for the current data, in one pass over the recorded slots and without
allocating.
"""
function refill!(
        g::ReducedGram{T}, P::SparseMatrixCSC, A::SparseMatrixCSC, rho, E, D, c, sigma
    ) where {T}
    nz = nonzeros(g.R)
    fill!(nz, zero(T))
    avals = nonzeros(A)
    rowptr, colind, aperm, aslot = g.rowptr, g.colind, g.aperm, g.aslot
    t = 0
    for i in eachindex(rho, E)
        ei = E[i]
        w = rho[i] * ei * ei
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop
            wvj = w * T(avals[aperm[p]]) * D[colind[p]]
            for q in p:stop
                t += 1
                nz[aslot[t]] += wvj * T(avals[aperm[q]]) * D[colind[q]]
            end
        end
    end
    pvals = nonzeros(P)
    for k in eachindex(g.pslot)
        nz[g.pslot[k]] += c * D[g.prow[k]] * T(pvals[g.pperm[k]]) * D[g.pcol[k]]
    end
    for k in eachindex(g.dslot)
        nz[g.dslot[k]] += sigma
    end
    return g.R
end

function PureOSQP.factorize!(ls::SparseCholmod{T}, ws)::Bool where {T}
    P, A = ws.P, ws.A
    if !describes(ls.gram, P, A)
        # `update!` replaced P or A with one storing entries somewhere else, so every slot
        # the map holds is stale.
        ls.gram = reduced_gram(T, P, A, ws.n)
        R = refill!(ls.gram, P, A, ws.rho_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
        ls.fact = cholesky(Symmetric(R, :U); check = false)
    else
        # The pattern is unchanged, so the symbolic factorization still describes it and
        # only the values need redoing. This is the case every time `ρ` moves.
        R = refill!(ls.gram, P, A, ws.rho_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
        cholesky!(ls.fact, Symmetric(R, :U); check = false)
    end
    issuccess(ls.fact) || return false
    # `F.L` is defined only for an LLᵀ factorization. `choose_backend` selects this backend
    # only after checking that CHOLMOD produces one for this pattern, and the pattern is
    # what decides it, so this holds for the workspace's life.
    # Asserted, not assumed: `getproperty` on a CHOLMOD factor branches on the symbol and
    # is not inferrable, and an unannotated result costs `factorize!` type stability.
    ls.L = sparse(ls.fact.L)::SparseMatrixCSC{T, Int}
    ls.Lt = SparseMatrixCSC(transpose(ls.L))
    check_factor(ls.L, ws.n)
    check_factor(ls.Lt, ws.n)
    ls.perm = ls.fact.p::Vector{Int}
    return true
end

"""
    check_factor(L, N)

Establish that every index the substitutions will use is in range, or throw.

The substitutions index `x` by a row index read out of `L`, which no compiler can prove is
in bounds, so they are checked once here instead of on every one of the `nnz(L)` accesses —
[`unit_forward!`](@ref) and [`unit_backward!`](@ref) then run unchecked. That is worth 1.18×
to 1.32× on the OSQP suite's factors, which hold two to three nonzeros per column, where the
check is a large fraction of the work done per entry.

Called once per factorization, over `nnz(L)` entries, against a factorization that costs far
more; the guard is not on the per-iteration path.

It throws rather than returning `false` because an out-of-range index is a broken
factorization, not an unfactorizable matrix — a distinction `factorize!`'s `Bool` cannot
carry, and one a caller can do nothing about.
"""
function check_factor(L::SparseMatrixCSC, N::Integer)
    colptr, rows = L.colptr, rowvals(L)
    nz = length(rows)
    (length(colptr) > N && colptr[1] == 1 && colptr[N + 1] == nz + 1) || throw(
        ArgumentError(
            "factor has a malformed column pointer for an order-$N system: " *
                "colptr spans $(colptr[1]):$(colptr[min(N + 1, length(colptr))]) over $nz stored entries"
        )
    )
    for j in 1:N
        colptr[j] <= colptr[j + 1] || throw(
            ArgumentError("factor's column pointer decreases at column $j")
        )
    end
    for p in 1:nz
        1 <= rows[p] <= N || throw(
            ArgumentError("factor stores row index $(rows[p]) at position $p, outside 1:$N")
        )
    end
    return nothing
end

"""
    llt_forward!(x, L, N)
    llt_backward!(x, Lt, N)

Substitution against the two factors of an `L Lᵀ`, in place.

CHOLMOD stores each column's diagonal entry first, so [`llt_forward!`](@ref) takes `L[j,j]`
from `nonzeros(L)[colptr[j]]` and runs the off-diagonal entries from one past it.

[`llt_backward!`](@ref) takes `Lᵀ` rather than `L`. Against `L` the loop has to gather —
each column accumulates a dot product into a scalar, which is a serial dependency — where
against `Lᵀ` it scatters, exactly as the forward solve does. On the OSQP suite's Huber
factor, 5199 nonzeros over 1806 columns, that is 5.12 µs against 3.88 µs. Column `j` of
`Lᵀ` is row `j` of `L`, so the diagonal is its *last* entry.

Written out rather than left to `ldiv!(LowerTriangular(L), x)` for the same reason as
[`ldl_forward!`](@ref), and the reason is the factor's shape rather than the wrapper.
Measured on the OSQP suite's own factors, resetting the vector every sample because these
solves are in place: at 2.2 nonzeros per column (Lasso) the pair costs 5.00 µs through
`ldiv!` and 2.91 µs here, and at 2.9 (Huber) 11.59 µs against 9.21 µs. On a factor with 9
nonzeros per column the ordering reverses and `ldiv!` is the faster of the two — at that
density the arithmetic dominates, where here the per-column bookkeeping does, and the generic
path carries more of it.
"""
function llt_forward!(x::AbstractVector, L::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = L.colptr, rowvals(L), nonzeros(L)
    @inbounds for j in 1:N
        top = colptr[j]
        xj = x[j] / vals[top]
        x[j] = xj
        for p in (top + 1):(colptr[j + 1] - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function llt_backward!(x::AbstractVector, Lt::SparseMatrixCSC, N::Integer)
    colptr, rows, vals = Lt.colptr, rowvals(Lt), nonzeros(Lt)
    @inbounds for j in N:-1:1
        bot = colptr[j + 1] - 1
        xj = x[j] / vals[bot]
        x[j] = xj
        for p in colptr[j]:(bot - 1)
            x[rows[p]] -= vals[p] * xj
        end
    end
    return x
end

function PureOSQP.solve_system!(ls::SparseCholmod{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    rhs = PureOSQP.reduced_rhs!(ws, rhs_x, rhs_z)
    perm, work, n = ls.perm, ls.permuted, ws.n
    # R[perm, perm] = L Lᵀ, so the solve is a permutation, two triangular solves, and the
    # inverse permutation -- all over buffers this backend owns.
    for i in 1:n
        work[i] = rhs[perm[i]]
    end
    llt_forward!(work, ls.L, n)
    llt_backward!(work, ls.Lt, n)
    x = ws.xtilde
    for i in 1:n
        x[perm[i]] = work[i]
    end
    ws.m > 0 && PureOSQP.mul_A!(ws.ztilde, ws, x)
    return nothing
end

"""
    densest_row(A) -> Int

The most nonzeros any row of `A` holds.

`Ãᵀ diag(ρ) Ã` gives each row of `A` an outer product with itself, so the densest row alone
contributes a dense `nnzᵢ × nnzᵢ` block to the reduced matrix and `nnzᵢ²` is a lower bound
on `nnz(R)`. That is enough to reject a problem before forming `R` at all, which matters
because forming it is a sparse product over the whole matrix: on the OSQP suite's Portfolio
class, whose budget row `1ᵀx = 1` touches every column, that product was 27% of `setup` and
its only outcome was a rejection.
"""
function densest_row(A::SparseMatrixCSC)
    counts = zeros(Int, size(A, 1))
    for i in rowvals(A)
        counts[i] += 1
    end
    return isempty(counts) ? 0 : maximum(counts)
end


"""
    cholmod_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> SparseCholmod or nothing

Build the reduced sparse-factorization backend if it suits these matrices, and return
`nothing` if it does not.

The decision is measured rather than guessed: CHOLMOD is asked to factor the reduced matrix
and the fill it reports settles it. That costs one factorization, and none of it is wasted
when the answer is yes — the matrix is built from the equilibrated data, so the factor is
the one the solver goes on to solve against and its symbolic part is what every later
refactorization reuses. When the answer is no, [`densest_row`](@ref) usually says so before
`R` is formed at all, and `nnz(R)` catches the rest.
"""
function cholmod_backend(
        P, A, proto::AbstractVector{T}, n::Integer, m::Integer, D, E, c, rho_vec, sigma
    ) where {T <: Real}
    (issparse(P) && n > 0) || return nothing
    # A lower bound on `nnz(R)`, computed in one pass over `A`'s stored entries.
    densest_row(A)^2 < DENSE_FACTOR_FILL * n^2 || return nothing
    # Counted rather than formed: the gate wants `nnz(R)` and nothing else, and that comes
    # from the patterns. `reduced_nnz` reports one triangle where the limit is stated for the
    # whole matrix.
    2 * reduced_nnz(P, A, n) < 2 * DENSE_FACTOR_FILL * n^2 || return nothing
    gram = reduced_gram(T, P, A, n)
    R = refill!(gram, P, A, rho_vec, E, D, c, sigma)
    # A pure-Julia LDLᵀ, if one is loaded, factors this faster than CHOLMOD does and hands
    # back `L` and `D` as plain arrays, so nothing has to be extracted from a foreign factor.
    alt = PureOSQP.ldl_backend(gram, proto, n, DENSE_FACTOR_FILL)
    isnothing(alt) || return alt
    F = cholesky(Symmetric(R, :U); check = false)
    issuccess(F) || return nothing
    # An LDLᵀ factorization has no `F.L`, and this backend's solve assumes `L Lᵀ`.
    L = try
        sparse(F.L)
    catch
        return nothing
    end
    nnz(L) < DENSE_FACTOR_FILL * n^2 || return nothing
    Lt = SparseMatrixCSC(transpose(L))
    check_factor(L, n)
    check_factor(Lt, n)
    return SparseCholmod{T, typeof(proto), typeof(F)}(
        gram, F, L, Lt, F.p, similar(proto, T, n)
    )
end

# No `trim_compat` claim: the solve reaches CHOLMOD through `ccall`, and the trim entry
# points cover the dense path.
@verify SparseCholmod
@verify SparseKKT


"Whether every stored entry of `P` lies on the diagonal, in one pass over its columns."
function is_diagonal(P::SparseMatrixCSC)
    rows = rowvals(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            rows[k] == j || return false
        end
    end
    return true
end

"""
    is_convex(T, P::SparseMatrixCSC, sigma) -> Bool

The convexity test without densifying `P`. `cholesky` on a sparse matrix is CHOLMOD, which
costs `O(nnz(L))` where the generic dense test costs `O(n³)` — 0.24 ms against 22.6 ms on a
tridiagonal `P` at `n = 2000`, which was half of `setup` on a banded problem. A diagonal `P`
skips the factorization entirely.
"""
function PureOSQP.is_convex(::Type{T}, P::SparseMatrixCSC, sigma) where {T}
    isempty(P) && return true
    # A diagonal `P` needs no factorization: `P + σI` is diagonal, so it is positive definite
    # exactly when every entry clears `-σ`. This is not a corner case — an epigraph
    # reformulation leaves the objective diagonal, which is what four of the OSQP suite's
    # seven classes look like.
    if is_diagonal(P)
        vals = nonzeros(P)
        for k in eachindex(vals)
            T(vals[k]) + sigma > zero(T) || return false
        end
        return true
    end
    return issuccess(cholesky(Symmetric(SparseMatrixCSC{T, Int}(P) + sigma * I); check = false))
end

end # module PureOSQPSparseArraysExt
