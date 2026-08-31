"""
    BlockReduced{T,M,F} <: LinearSystem

The reduced system when it decouples into independent blocks, which it does for a
[`BlockDiagonal`](@ref) `P` and `A` split at the same columns.

Each block `i` carries its own

    Rᵢ = c Dᵢ Pᵢ Dᵢ + σI + Ãᵢᵀ diag(ρᵢ) Ãᵢ

and nothing couples them, so `K` Cholesky factorizations replace one and each solve is `K`
triangular solves over `Σ nᵢ` entries. Against the dense terminal that is `Σ nᵢ³` against `n³`
to factor and `Σ nᵢ²` against `n²` to store; for `K` equal blocks, `K²` and `K`.

`blocks` holds each `Rᵢ` and is overwritten by that block's *inverse*, so a solve is one `symv`
per block rather than two triangular solves; `scaled` is the scratch each block is assembled
in. Both are rebuilt whenever `ρ` moves.
"""
struct BlockReduced{T <: Real, M <: AbstractMatrix{T}} <: LinearSystem
    blocks::Vector{M}
    scaled::Vector{M}      # block `i` holds `sqrt(ρ) ⊙ E ⊙ Aᵢ ⊙ D`, so `Rᵢ` is one `syrk`
end

"""
    BlockReduced(proto::AbstractVector, sizes)

Build one square buffer per block, as `similar(proto, ...)`, following the array type of the
data it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function BlockReduced(
        proto::AbstractVector{T}, sizes::Vector{Int}, rows::Vector{Int}
    ) where {T <: Real}
    blocks = [similar(proto, T, s, s) for s in sizes]
    scaled = [similar(proto, T, r, s) for (r, s) in zip(rows, sizes)]
    for B in blocks
        fill!(B, zero(T))
    end
    return BlockReduced{T, eltype(blocks)}(blocks, scaled)
end

backend_name(::BlockReduced) = :block

# One triangle per block is what is stored, and the blocks are all there is.
function backend_info(ls::BlockReduced)
    n = sum(B -> size(B, 1), ls.blocks)
    stored = sum(B -> size(B, 1) * (size(B, 1) + 1) ÷ 2, ls.blocks)
    return BackendInfo(backend_name(ls), true, :reduced, n, stored)
end

"""
    block_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool) or nothing

Ladder rung for a reduced matrix that decouples into independent blocks. Declines unless both
operands are [`BlockDiagonal`](@ref) over the same column partition, and declines a single
block, which is the dense terminal wearing a wrapper.
"""
block_rung(P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma) = nothing

function block_rung(
        P::BlockDiagonal, A::BlockDiagonal, proto::AbstractVector{T}, n::Integer, m::Integer,
        D, E, c, rho_vec, sigma
    ) where {T <: Real}
    (nblocks(P) > 1 && same_column_partition(P, A)) || return nothing
    return (
        BlockReduced(
            proto,
            [length(colrange(A, i)) for i in 1:nblocks(A)],
            [length(rowrange(A, i)) for i in 1:nblocks(A)],
        ),
        false,
    )
end

function factorize!(ls::BlockReduced{T}, ws)::Bool where {T}
    A, P, D, E = ws.A, ws.P, ws.D, ws.E
    c, rho, sigma = ws.c, ws.rho_vec, ws.settings.sigma
    for i in eachindex(ls.blocks)
        R = ls.blocks[i]
        cols, rows = colrange(A, i), rowrange(A, i)
        Ai, Pi = A.blocks[i], P.blocks[i]
        # `sqrt(ρ) ⊙ E ⊙ Aᵢ ⊙ D` first, so the block is one `syrk` rather than a scalar triple
        # loop -- the same shape `ReducedCholesky` uses, and the reason this backend is worth
        # having at all: `K` small BLAS-3 calls, not `K` hand-rolled ones.
        W = ls.scaled[i]
        for (jj, j) in pairs(cols), (ii, i2) in pairs(rows)
            W[ii, jj] = sqrt(rho[i2]) * E[i2] * Ai[ii, jj] * D[j]
        end
        mul!(R, W', W)
        for (jj, j) in pairs(cols), (kk, k) in pairs(cols)
            R[jj, kk] += c * D[j] * Pi[jj, kk] * D[k]
        end
        for jj in axes(R, 1)
            R[jj, jj] += sigma
        end
        f = cholesky!(Symmetric(R); check = false)
        issuccess(f) || return false
        # Inverted, not kept as a factor: each iteration's block solve is then one `symv`
        # rather than two triangular solves. Both cost `2nᵢ²` flops, but a triangular solve
        # computes its entries in sequence and `symv` does not, which is the same reason
        # `ReducedCholesky` inverts. `σI` bounds the conditioning, as it does there.
        invert_spd!(R, f)
    end
    return true
end

function solve_system!(ls::BlockReduced, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    b, x = ws.work_n, ws.xtilde
    A = ws.A
    for i in eachindex(ls.blocks)
        cols = colrange(A, i)
        mul!(view(x, cols), Symmetric(ls.blocks[i], :U), view(b, cols))
    end
    ws.m > 0 && mul_A!(ws.ztilde, ws, x)
    return nothing
end

@verify BlockReduced trim_compat = true
