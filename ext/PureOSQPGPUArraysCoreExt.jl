"""
    PureOSQPGPUArraysCoreExt

Backend routing and whole-matrix traversals for GPU arrays.

The per-iteration work is already storage-generic: every elementwise update and reduction
has a schedule that does not index (see `src/elementwise.jl`), and the matrix-vector
products go through `mul!` on the caller's own matrices. What is left is choosing a backend
that can run, and reading `P` and `A` during setup without indexing them.

**GPU support is matrix-free only.** `linsys = :indirect` is the one backend whose inner
solve has a GPU counterpart: Krylov.jl's `cg!` is GPU-native, `bunchkaufman!` has no GPU
implementation at all, CHOLMOD is CPU by construction, and `potri!` reaches LAPACK. Any
other backend is refused at [`setup`](@ref) rather than left to fail inside `factorize!`.

`polish` and the derivatives stay on the host, since both build a dense `(n+k)×(n+k)`
matrix and factor it with `bunchkaufman!`.
"""
module PureOSQPGPUArraysCoreExt

using PureOSQP: PureOSQP
using GPUArraysCore: AbstractGPUMatrix, AbstractGPUVector

"""
    unsupported_backend()

What a GPU array gets when it reaches a backend that cannot run on one.

Raised at `setup`, where the caller can act on it. `JLArray` is a `StridedArray`, so without
this the dense backend would dispatch to CPU LAPACK and quietly succeed on JLArrays while
failing inside `factorize!` on a CuArray — a difference nobody would see until they had a
GPU.
"""
function unsupported_backend()
    throw(
        ArgumentError(
            "a direct backend is not available for GPU arrays: it factors a matrix, and " *
                "only the matrix-free backend has a GPU counterpart. Load Krylov and pass " *
                "`linsys = :indirect`, or move the problem to the host with `Array`."
        )
    )
end

PureOSQP.choose_backend(P, A::AbstractGPUMatrix, proto::AbstractVector, n::Integer, m::Integer) =
    unsupported_backend()
PureOSQP.choose_backend(P::AbstractGPUMatrix, A, proto::AbstractVector, n::Integer, m::Integer) =
    unsupported_backend()
PureOSQP.choose_backend(
    P::AbstractGPUMatrix, A::AbstractGPUMatrix, proto::AbstractVector, n::Integer, m::Integer
) = unsupported_backend()

"`M`'s diagonal as a vector, without indexing it."
diagonal_of(M::AbstractGPUMatrix) = vec(sum(M .* onehot(M); dims = 1))

"An identity matrix of `M`'s type, built by broadcast rather than by assignment."
onehot(M::AbstractGPUMatrix) = (1:size(M, 1)) .== permutedims(1:size(M, 2))

# Equilibration measures every column of `P` and `A` once per sweep, and the matrix-free
# backend needs the reduced matrix's diagonal. The generic paths ask column by column, which
# indexes; whole-matrix reductions answer the same questions. Both run at setup or per
# refactorization, never per iteration, so their temporaries are not on the hot path.

function PureOSQP.column_norms!(
        d::AbstractGPUVector, e::AbstractGPUVector, ::Type{T},
        pcol::AbstractGPUVector, A::AbstractGPUMatrix, D, E, c
    ) where {T}
    # `E` multiplies down each row and `D` across each column, matching what the per-column
    # traversals apply through their weight vectors.
    scaled_a = abs.(E .* A)
    e .= vec(maximum(scaled_a .* permutedims(D); dims = 2))
    d .= PureOSQP.limit_scaling.(
        max.(c .* D .* pcol, D .* vec(maximum(scaled_a; dims = 1)))
    )
    return d
end

function PureOSQP.cost_norms!(
        pcol::AbstractGPUVector, ::Type{T}, P::AbstractGPUMatrix, D::AbstractGPUVector, c, n
    ) where {T}
    pcol .= vec(maximum(abs.(D .* P); dims = 1))
    return sum(c .* D .* pcol) / n
end

function PureOSQP.reduced_diagonal!(
        dest::AbstractGPUVector, ::Type{T}, P::AbstractGPUMatrix, A::AbstractGPUMatrix,
        rho, E, D, sigma, c
    ) where {T}
    quad = vec(sum(rho .* abs2.(E .* A .* permutedims(D)); dims = 1))
    dest .= inv.(max.(c .* D .* D .* diagonal_of(P) .+ sigma .+ quad, sqrt(eps(T))))
    return dest
end

end # module PureOSQPGPUArraysCoreExt
