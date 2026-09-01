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
using OSQP_jll: osqp_builtin_double

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

"A `Ref` holding `settings` with the named fields replaced."
function with_settings(base::OSQPSettings; kwargs...)
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

"""
    solve_v1(P, q, A, l, u; settings...) -> (; status, iter, obj_val, x, y)

Solve with libosqp 1.x. `P` is read as its upper triangle, which is what the C API expects.
"""
function solve_v1(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; settings...
    )
    n = length(q)
    m = length(l)
    Pp, Pi, Px = csc_arrays(P, true)
    Ap, Ai, Ax = csc_arrays(A, false)
    Pcsc = Ref(
        OSQPCscMatrix(n, n, pointer(Pp), pointer(Pi), pointer(Px), length(Px), -1, 0)
    )
    Acsc = Ref(
        OSQPCscMatrix(m, n, pointer(Ap), pointer(Ai), pointer(Ax), length(Ax), -1, 0)
    )
    sref = with_settings(default_settings(); settings...)
    solverp = Ref{Ptr{OSQPSolver}}(C_NULL)
    qq = collect(OSQPFloat, q)
    ll = collect(OSQPFloat, l)
    uu = collect(OSQPFloat, u)

    return GC.@preserve Pp Pi Px Ap Ai Ax qq ll uu Pcsc Acsc sref begin
        rc = ccall(
            (:osqp_setup, osqp_builtin_double), OSQPInt,
            (
                Ptr{Ptr{OSQPSolver}}, Ptr{OSQPCscMatrix}, Ptr{OSQPFloat}, Ptr{OSQPCscMatrix},
                Ptr{OSQPFloat}, Ptr{OSQPFloat}, OSQPInt, OSQPInt, Ptr{OSQPSettings},
            ),
            solverp, Pcsc, qq, Acsc, ll, uu, m, n, sref
        )
        iszero(rc) || error("osqp_setup returned $rc")
        try
            rc = ccall((:osqp_solve, osqp_builtin_double), OSQPInt, (Ptr{OSQPSolver},), solverp[])
            iszero(rc) || error("osqp_solve returned $rc")
            solver = unsafe_load(solverp[])
            sol = unsafe_load(solver.solution)
            info = unsafe_load(solver.info)
            return (;
                status_val = Int(info.status_val),
                iter = Int(info.iter),
                obj_val = info.obj_val,
                duality_gap = info.duality_gap,
                primdual_int = info.primdual_int,
                run_time = info.run_time,
                x = copy(unsafe_wrap(Array, sol.x, n)),
                y = copy(unsafe_wrap(Array, sol.y, m)),
            )
        finally
            ccall((:osqp_cleanup, osqp_builtin_double), OSQPInt, (Ptr{OSQPSolver},), solverp[])
        end
    end
end

verify_abi()
