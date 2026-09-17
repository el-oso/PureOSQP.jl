# libosqp 1.x through `ccall`, for the benchmarks that compare against it.
#
# OSQP.jl wraps 0.6.2 and there is no Julia wrapper for 1.x, so this mirrors the three C
# structs it needs from the headers `OSQP_jll` ships beside the library.
#
# **The shipped header cannot be trusted about the scalar types, and this file does not trust
# it.** One `osqp_configure.h` covers both the double and single builds, and it defines
# `OSQP_USE_FLOAT`, which would make `OSQPFloat` a `Cfloat`. Against
# `libosqp_builtin_double.so` that is wrong. The types below were determined from the library
# instead: `osqp_set_default_settings` writes known defaults into a buffer, and the offsets
# they land on identify the widths. `verify_abi()` re-runs that determination and is called
# at load, so a rebuilt artifact that changed either width fails here rather than returning
# quiet nonsense.
using OSQP_jll: OSQP_jll, osqp_builtin_double
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals

const OSQPInt = Clonglong
const OSQPFloat = Cdouble

"""
    OSQPSettings

The 1.x settings struct, field for field. Enum members are `Cint`; the compiler pads each to
the alignment of the `OSQPInt` that follows, which is why `linsys_solver` and `cg_precond`
each carry a `_pad`.
"""
struct OSQPSettings
    device::OSQPInt
    linsys_solver::Cint
    _pad1::Cint
    allocate_solution::OSQPInt
    verbose::OSQPInt
    profiler_level::OSQPInt
    warm_starting::OSQPInt
    scaling::OSQPInt
    polishing::OSQPInt
    rho::OSQPFloat
    rho_is_vec::OSQPInt
    sigma::OSQPFloat
    alpha::OSQPFloat
    cg_max_iter::OSQPInt
    cg_tol_reduction::OSQPInt
    cg_tol_fraction::OSQPFloat
    cg_precond::Cint
    _pad2::Cint
    adaptive_rho::OSQPInt
    adaptive_rho_interval::OSQPInt
    adaptive_rho_fraction::OSQPFloat
    adaptive_rho_tolerance::OSQPFloat
    max_iter::OSQPInt
    eps_abs::OSQPFloat
    eps_rel::OSQPFloat
    eps_prim_inf::OSQPFloat
    eps_dual_inf::OSQPFloat
    scaled_termination::OSQPInt
    check_termination::OSQPInt
    check_dualgap::OSQPInt
    time_limit::OSQPFloat
    delta::OSQPFloat
    polish_refine_iter::OSQPInt
end

struct OSQPCscMatrix
    m::OSQPInt
    n::OSQPInt
    p::Ptr{OSQPInt}
    i::Ptr{OSQPInt}
    x::Ptr{OSQPFloat}
    nzmax::OSQPInt
    nz::OSQPInt
    owned::OSQPInt
end

struct OSQPSolution
    x::Ptr{OSQPFloat}
    y::Ptr{OSQPFloat}
    prim_inf_cert::Ptr{OSQPFloat}
    dual_inf_cert::Ptr{OSQPFloat}
end


"""
    OSQPInfo

What the solve reports. `status` is a fixed 32-byte C string, carried as four `UInt64` so the
struct's layout matches without a `NTuple` that Julia would align differently.

`primdual_int` is the reason this is mirrored in full rather than read at a couple of offsets:
it is libosqp's own primal-dual integral, and it is the only reference this package has for
the same quantity.
"""
struct OSQPInfo
    status_1::UInt64
    status_2::UInt64
    status_3::UInt64
    status_4::UInt64
    status_val::OSQPInt
    status_polish::OSQPInt
    obj_val::OSQPFloat
    dual_obj_val::OSQPFloat
    prim_res::OSQPFloat
    dual_res::OSQPFloat
    duality_gap::OSQPFloat
    iter::OSQPInt
    rho_updates::OSQPInt
    rho_estimate::OSQPFloat
    setup_time::OSQPFloat
    solve_time::OSQPFloat
    update_time::OSQPFloat
    polish_time::OSQPFloat
    run_time::OSQPFloat
    primdual_int::OSQPFloat
    rel_kkt_error::OSQPFloat
end

struct OSQPSolver
    settings::Ptr{OSQPSettings}
    solution::Ptr{OSQPSolution}
    info::Ptr{OSQPInfo}
    work::Ptr{Cvoid}
end

default_settings() = (
    s = Ref{OSQPSettings}();
    ccall(
        (:osqp_set_default_settings, osqp_builtin_double), Cvoid, (Ptr{OSQPSettings},), s
    );
    s[]
)

"""
    verify_abi()

Re-derive the scalar widths from the library and check this file's mirror against them.

`osqp_set_default_settings` writes documented defaults — `rho = 0.1`, `alpha = 1.6`,
`scaling = 10`, `max_iter = 4000` — so finding them at the offsets this struct puts them at
confirms both the widths and the padding. Called at load: a rebuilt artifact that changed
either width fails here instead of producing numbers that mean nothing.
"""
function verify_abi()
    # `ccall` treats an empty library name as "look in the process", so an `OSQP_jll` that
    # ships no builtin-double library does not fail here — it silently reaches whatever
    # `libosqp` some other package has already loaded, which is 0.6.2 and a different
    # struct. The width checks below cannot catch that: 0.6.2's defaults land on enough of
    # the same offsets to pass them.
    isempty(osqp_builtin_double) && error(
        "OSQP_jll v$(pkgversion(OSQP_jll)) provides no libosqp_builtin_double. " *
            "libosqp 1.x is required; add `OSQP_jll = \"100\"` to this project's [compat]."
    )
    ver = unsafe_string(ccall((:osqp_version, osqp_builtin_double), Cstring, ()))
    startswith(ver, "1.") || error("libosqp reports version $ver, not 1.x")
    s = default_settings()
    s.rho == 0.1 || error("libosqp ABI: rho is $(s.rho), not the documented default 0.1")
    s.alpha == 1.6 || error("libosqp ABI: alpha is $(s.alpha), not 1.6")
    s.scaling == 10 || error("libosqp ABI: scaling is $(s.scaling), not 10")
    s.max_iter == 4000 || error("libosqp ABI: max_iter is $(s.max_iter), not 4000")
    s.cg_tol_reduction == 10 || error("libosqp ABI: cg_tol_reduction is $(s.cg_tol_reduction)")
    # The offsets those four land on are what identify the widths, so a struct that matched
    # by luck with the wrong types would have to miss one of them.
    fieldoffset(OSQPSettings, findfirst(==(:rho), fieldnames(OSQPSettings))) == 64 ||
        error("libosqp ABI: rho moved off byte 64")
    fieldoffset(OSQPSettings, findfirst(==(:max_iter), fieldnames(OSQPSettings))) == 160 ||
        error("libosqp ABI: max_iter moved off byte 160")
    return true
end

"""
A `Ref` holding `settings` with the named fields replaced.

A name that is not a field is an error rather than a no-op: 1.x renamed several of 0.6.2's
settings (`polish` to `polishing`, `warm_start` to `warm_starting`), and silently dropping
one would leave the caller reading a comparison made under settings it did not ask for.
"""
function with_settings(base::OSQPSettings; kwargs...)
    unknown = setdiff(keys(kwargs), fieldnames(OSQPSettings))
    isempty(unknown) || error("not libosqp 1.x settings: $(join(unknown, ", "))")
    vals = map(fieldnames(OSQPSettings)) do f
        haskey(kwargs, f) ? convert(fieldtype(OSQPSettings, f), kwargs[f]) : getfield(base, f)
    end
    return Ref(OSQPSettings(vals...))
end

"Upper-triangular CSC of `P`, and CSC of `A`, in the layout libosqp expects."
function csc_arrays(M::AbstractMatrix, upper::Bool)
    m, n = size(M)
    colptr = zeros(OSQPInt, n + 1)
    rows = OSQPInt[]
    vals = OSQPFloat[]
    for j in 1:n
        colptr[j] = length(vals)
        for i in 1:(upper ? j : m)
            v = M[i, j]
            if !iszero(v)
                push!(rows, i - 1)
                push!(vals, v)
            end
        end
    end
    colptr[n + 1] = length(vals)
    return colptr, rows, vals
end

# A sparse matrix is copied from its stored entries. Reading it through `M[i, j]` looks up
# every position, which on the larger suite problems costs more than libosqp's whole solve,
# and that time would be charged to libosqp.
function csc_arrays(M::SparseMatrixCSC, upper::Bool)
    m, n = size(M)
    colptr = zeros(OSQPInt, n + 1)
    rows = sizehint!(OSQPInt[], nnz(M))
    vals = sizehint!(OSQPFloat[], nnz(M))
    rv, nz = rowvals(M), nonzeros(M)
    for j in 1:n
        colptr[j] = length(vals)
        for k in nzrange(M, j)
            i = rv[k]
            (upper && i > j) && continue
            iszero(nz[k]) && continue
            push!(rows, i - 1)
            push!(vals, nz[k])
        end
    end
    colptr[n + 1] = length(vals)
    return colptr, rows, vals
end

"""
    V1Model

A set-up libosqp solver together with the problem arrays it was built from.

`cleanup!` frees the C side and is idempotent, so the finalizer and an explicit call cannot
double-free. `arrays` is held for the lifetime of the model: the setup call is handed raw
pointers into those vectors, and rooting them here is what keeps the pointers valid.
"""
mutable struct V1Model
    solver::Ptr{OSQPSolver}
    n::Int
    m::Int
    arrays::Tuple
end

"""
    CSCData(P, A)

`P`'s upper triangle and `A`, converted once to the arrays libosqp reads.

A C caller of libosqp already holds its matrices in this form, so the benchmarks build it
outside the timed region and time `osqp_setup` and `osqp_solve` alone. Building it from a
dense matrix is an `O(mn)` scan, which on small problems is a large share of libosqp's time.
"""
struct CSCData
    n::Int
    m::Int
    P::NTuple{3, Vector}
    A::NTuple{3, Vector}
end

CSCData(P::AbstractMatrix, A::AbstractMatrix) =
    CSCData(size(A, 2), size(A, 1), csc_arrays(P, true), csc_arrays(A, false))

"""
    setup_v1(data::CSCData, q, l, u; settings...) -> V1Model
    setup_v1(P, q, A, l, u; settings...) -> V1Model

Set up libosqp 1.x. `P` is read as its upper triangle, which is what the C API expects. The
second form converts the matrices first.
"""
setup_v1(
    P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix, l::AbstractVector,
    u::AbstractVector; settings...
) = setup_v1(CSCData(P, A), q, l, u; settings...)

function setup_v1(
        data::CSCData, q::AbstractVector, l::AbstractVector, u::AbstractVector; settings...
    )
    n, m = data.n, data.m
    length(q) == n && length(l) == m && length(u) == m ||
        throw(DimensionMismatch("q, l and u do not match the CSC data's size ($m×$n)"))
    Pp, Pi, Px = data.P
    Ap, Ai, Ax = data.A
    qq = collect(OSQPFloat, q)
    ll = collect(OSQPFloat, l)
    uu = collect(OSQPFloat, u)
    Pcsc = Ref(
        OSQPCscMatrix(n, n, pointer(Pp), pointer(Pi), pointer(Px), length(Px), -1, 0)
    )
    Acsc = Ref(
        OSQPCscMatrix(m, n, pointer(Ap), pointer(Ai), pointer(Ax), length(Ax), -1, 0)
    )
    sref = with_settings(default_settings(); settings...)
    solverp = Ref{Ptr{OSQPSolver}}(C_NULL)
    arrays = (Pp, Pi, Px, Ap, Ai, Ax, qq, ll, uu)

    GC.@preserve arrays Pcsc Acsc sref begin
        rc = ccall(
            (:osqp_setup, osqp_builtin_double), OSQPInt,
            (
                Ptr{Ptr{OSQPSolver}}, Ptr{OSQPCscMatrix}, Ptr{OSQPFloat}, Ptr{OSQPCscMatrix},
                Ptr{OSQPFloat}, Ptr{OSQPFloat}, OSQPInt, OSQPInt, Ptr{OSQPSettings},
            ),
            solverp, Pcsc, qq, Acsc, ll, uu, m, n, sref
        )
        iszero(rc) || error("osqp_setup returned $rc")
    end
    model = V1Model(solverp[], n, m, arrays)
    return finalizer(cleanup!, model)
end

"Free a `V1Model`'s C-side solver. Safe to call more than once."
function cleanup!(model::V1Model)
    if model.solver != C_NULL
        ccall((:osqp_cleanup, osqp_builtin_double), OSQPInt, (Ptr{OSQPSolver},), model.solver)
        model.solver = C_NULL
    end
    return nothing
end

"""
    update_v1!(model; q, l, u) -> model

Replace `q`, `l` and `u` through `osqp_update_data_vec`, keeping the factorization, which is
upstream's own sequential re-solve path.
"""
function update_v1!(model::V1Model; q, l, u)
    model.solver == C_NULL && error("update_v1! on a cleaned-up model")
    qq, ll, uu = collect(OSQPFloat, q), collect(OSQPFloat, l), collect(OSQPFloat, u)
    rc = GC.@preserve qq ll uu ccall(
        (:osqp_update_data_vec, osqp_builtin_double), OSQPInt,
        (Ptr{OSQPSolver}, Ptr{OSQPFloat}, Ptr{OSQPFloat}, Ptr{OSQPFloat}),
        model.solver, qq, ll, uu
    )
    iszero(rc) || error("osqp_update_data_vec returned $rc")
    return model
end

"""
    solve_v1!(model) -> (; status_val, iter, obj_val, x, y, ...)

Run the ADMM loop on an already-set-up model. Calling it again warm-starts from where the
previous call left off, as libosqp does.
"""
function solve_v1!(model::V1Model)
    model.solver == C_NULL && error("solve_v1! on a cleaned-up model")
    rc = ccall((:osqp_solve, osqp_builtin_double), OSQPInt, (Ptr{OSQPSolver},), model.solver)
    iszero(rc) || error("osqp_solve returned $rc")
    solver = unsafe_load(model.solver)
    sol = unsafe_load(solver.solution)
    info = unsafe_load(solver.info)
    return (;
        status_val = Int(info.status_val),
        iter = Int(info.iter),
        obj_val = info.obj_val,
        duality_gap = info.duality_gap,
        primdual_int = info.primdual_int,
        run_time = info.run_time,
        x = copy(unsafe_wrap(Array, sol.x, model.n)),
        y = copy(unsafe_wrap(Array, sol.y, model.m)),
    )
end

"""
    solve_v1(data::CSCData, q, l, u; settings...) -> (; status_val, iter, obj_val, x, y, ...)
    solve_v1(P, q, A, l, u; settings...)

Set up, solve and free in one call. Time the first form, with `data` built beforehand; the
second also converts the matrices.
"""
function solve_v1(data::CSCData, q::AbstractVector, l::AbstractVector, u::AbstractVector; settings...)
    model = setup_v1(data, q, l, u; settings...)
    try
        return solve_v1!(model)
    finally
        cleanup!(model)
    end
end

solve_v1(
    P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix, l::AbstractVector,
    u::AbstractVector; settings...
) = solve_v1(CSCData(P, A), q, l, u; settings...)

verify_abi()
