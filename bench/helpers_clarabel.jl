# Driving Clarabel on this repository's problem form, shared by the comparisons against it.
using Clarabel, LinearAlgebra, SparseArrays

"""
    clarabel_form(P, A, l, u) -> (P, A, b)

Clarabel takes one-sided cones: the two-sided rows are stacked as `Ax ≤ u, -Ax ≤ -l`, and a
row whose bound is infinite contributes nothing on that side. It reads the upper triangle of
`P`; handing it the full matrix doubles every off-diagonal term of the objective.
"""
function clarabel_form(P, A, l, u)
    finite_l = isfinite.(l)
    finite_u = isfinite.(u)
    rows = Vector{SparseMatrixCSC{Float64, Int}}()
    bnd = Float64[]
    As = sparse(A)
    if any(finite_u)
        push!(rows, As[finite_u, :])
        append!(bnd, u[finite_u])
    end
    if any(finite_l)
        push!(rows, -As[finite_l, :])
        append!(bnd, -l[finite_l])
    end
    return sparse(triu(P)), vcat(rows...), bnd
end

"Solve `(P, q, A, l, u)` with Clarabel at gap and feasibility tolerance `tol`."
function run_clarabel(P, q, A, l, u; tol = 1.0e-8)
    Pc, Ac, bc = clarabel_form(P, A, l, u)
    settings = Clarabel.Settings(
        verbose = false, tol_gap_abs = tol, tol_gap_rel = tol, tol_feas = tol,
    )
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, Pc, q, Ac, bc, [Clarabel.NonnegativeConeT(length(bc))], settings)
    return Clarabel.solve!(solver)
end
