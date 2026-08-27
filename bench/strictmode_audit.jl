# StrictMode gate for PureOSQP's hot path.
#
# Run:  cd bench && jl strictmode_audit.jl
# The global Stop hook runs this automatically after any turn that touches src/.
#
# `analysis = "full"` in bench/LocalPreferences.toml is deliberate: the `:fast` heuristic
# reports `admm_step!` and `update_residuals!` as allocating when AllocCheck proves they
# do not, so the cheap tier cannot be the gate for this package.
using PureOSQP
using StrictMode
using AllocCheck, JET          # the :full backends; StrictMode dispatches to them
using LinearAlgebra, Random

# A disabled audit prints exactly like a clean one. Never report a pass without this.
StrictMode.assert_enabled()

function example_workspace(backend::Symbol)
    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    ws = PureOSQP.setup(P, q, A, b .- rand(m), b .+ rand(m); linsys = backend)
    PureOSQP.solve!(ws)        # compile every specialization before analysing it
    return ws
end

const GUARANTEES = Dict(
    :hot => (:typestable, :noalloc),
    :warm => (:typestable,),
)

failures = String[]

for backend in (:auto, :kkt)
    ws = example_workspace(backend)
    W = typeof(ws)
    LS = typeof(ws.linsys)
    V = Vector{Float64}
    checks = [
        (PureOSQP.admm_step!, (W,), :hot),
        (PureOSQP.update_residuals!, (W,), :hot),
        (PureOSQP.solve_system!, (LS, W, V, V), :hot),
        (PureOSQP.check_termination, (W, Bool), :warm),
        (PureOSQP.factorize!, (LS, W), :warm),
        (PureOSQP.solve!, (W,), :warm),
    ]
    for (f, types, tier) in checks
        label = "$(nameof(f))($(join(types, ", "))) [linsys=$backend]"
        try
            StrictMode.check(f, types; guarantees = GUARANTEES[tier], mode = :full)
            println("  ✓ ", label, "  ", join(GUARANTEES[tier], ", "))
        catch e
            push!(failures, label)
            println("  ✗ ", label)
            println(sprint(showerror, e))
        end
    end
end

if isempty(failures)
    println(
        "\nStrictMode: all guarantees hold (checks_enabled=", StrictMode.checks_enabled(),
        ", mode=", StrictMode.analysis_mode(), ")."
    )
else
    println("\nStrictMode: ", length(failures), " failing guarantee(s):")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
