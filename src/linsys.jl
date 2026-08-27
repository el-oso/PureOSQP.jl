"""
    refactor!(ws)

Rebuild the factorization of the ADMM subproblem for the current `ws.rho_vec`.

The subproblem is the quasi-definite system

    ⎡P̃ + σI      Ãᵀ   ⎤ ⎡x̃⎤   ⎡rhs_x⎤
    ⎣Ã       −diag(ρ⁻¹)⎦ ⎣ν⎦ = ⎣rhs_z⎦

Eliminating `ν` gives the symmetric positive definite reduced system

    (P̃ + σI + Ãᵀ diag(ρ) Ã) x̃ = rhs_x + Ãᵀ(ρ ⊙ rhs_z),   z̃ = Ã x̃

which is `n×n` rather than `(n+m)×(n+m)` and is what this solver uses. Forming `ÃᵀρÃ`
squares the conditioning of `Ã`; equilibration keeps that acceptable to about
`cond(Ã) = 1e8`, beyond which the Cholesky fails. That failure is caught here and the
full quasi-definite system is factored with `bunchkaufman!` instead.
"""
function refactor!(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    if ws.settings.linsys === :kkt
        build_kkt!(ws)
        ws.bk = bunchkaufman!(Symmetric(ws.K, :L); check = true)
        ws.backend = :bunchkaufman
        ws.refactor_count += 1
        return ws
    end
    for j in 1:n
        dj = ws.D[j]
        for i in 1:m
            ws.W[i, j] = sqrt(ws.rho_vec[i]) * ws.E[i] * T(ws.A[i, j]) * dj
        end
    end
    if m > 0
        mul!(ws.R, ws.W', ws.W)
    else
        fill!(ws.R, zero(T))
    end
    for j in 1:n
        dj = ws.D[j]
        for i in 1:n
            ws.R[i, j] += ws.c * ws.D[i] * T(ws.P[i, j]) * dj
        end
    end
    for i in 1:n
        ws.R[i, i] += ws.settings.sigma
    end
    F = cholesky!(Symmetric(ws.R); check = false)
    if issuccess(F)
        ws.chol = F
        ws.backend = :cholesky
        ws.refactor_count += 1
    else
        isnothing(ws.bk) && throw(
            ArgumentError("the reduced matrix is not positive definite and the full-KKT fallback needs bunchkaufman! for $T, which this Julia version ($(VERSION)) does not provide. Use Float32 or Float64.")
        )
        build_kkt!(ws)
        ws.bk = bunchkaufman!(Symmetric(ws.K, :L); check = true)
        ws.backend = :bunchkaufman
        ws.refactor_count += 1
    end
    return ws
end

function build_kkt!(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    if size(ws.K, 1) != n + m
        ws.K = Matrix{T}(undef, n + m, n + m)
        ws.kkt_rhs = Vector{T}(undef, n + m)
    end
    fill!(ws.K, zero(T))
    for j in 1:n
        dj = ws.D[j]
        for i in 1:n
            ws.K[i, j] = ws.c * ws.D[i] * T(ws.P[i, j]) * dj
        end
        ws.K[j, j] += ws.settings.sigma
        for i in 1:m
            aij = ws.E[i] * T(ws.A[i, j]) * dj
            ws.K[n + i, j] = aij
            ws.K[j, n + i] = aij
        end
    end
    for i in 1:m
        ws.K[n + i, n + i] = -ws.rho_inv_vec[i]
    end
    return ws
end

"""
    solve_kkt!(ws, rhs_x, rhs_z)

Solve the ADMM subproblem, writing `x̃` into `ws.xtilde` and `z̃` into `ws.ztilde`.
"""
function solve_kkt!(ws::Workspace{T}, rhs_x::AbstractVector{T}, rhs_z::AbstractVector{T}) where {T}
    n, m = ws.n, ws.m
    if ws.backend === :cholesky
        copyto!(ws.xtilde, rhs_x)
        if m > 0
            ws.work_m .= ws.rho_vec .* rhs_z
            mul_At!(ws.work_n, ws, ws.work_m)
            ws.xtilde .+= ws.work_n
        end
        ldiv!(ws.chol, ws.xtilde)
        m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    else
        copyto!(view(ws.kkt_rhs, 1:n), rhs_x)
        copyto!(view(ws.kkt_rhs, (n + 1):(n + m)), rhs_z)
        ldiv!(ws.bk, ws.kkt_rhs)
        copyto!(ws.xtilde, view(ws.kkt_rhs, 1:n))
        for i in 1:m
            ws.ztilde[i] = rhs_z[i] + ws.rho_inv_vec[i] * ws.kkt_rhs[n + i]
        end
    end
    return ws
end
