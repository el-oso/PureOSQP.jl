@testitem "every LinearSystem backend is --trim compatible" begin
    using LinearAlgebra, SparseArrays, BandedMatrices, LDLFactorizations, Krylov, TypeContracts
    using InteractiveUtils: subtypes

    # The core module asserts this over its own subtypes with one `@verify … subtypes = true`.
    # That sweep cannot run from an extension — it would re-register the core backends, and
    # method overwriting is forbidden during precompilation — so an extension's backend carries
    # its own `@verify` and nothing in the source compels it. This is what does, and it is the
    # only place every extension is loaded at once.
    #
    # `verified_trait` is not the signal: `@verify` seals it whether or not `trim_compat` was
    # asked for, so a backend that skipped the claim still looks verified. This runs the check.
    #
    # Guards the guard. `check_trim_compat` scans the declared method bodies and does not
    # follow calls out of them, so it is a filter and not a proof -- a backend reaching a
    # `ccall` still passes, which is why `test/trim_tests.jl` validates whole entry points
    # with the trimmer itself. This pins the depth the sweep does have: a backend whose own
    # body holds something the trimmer cannot resolve must be reported.
    #
    # `TrimUnsafeProbe` exists to be rejected, and the `@warn` it draws is the rejection being
    # reported. Matching that log is what keeps the warning out of the suite's output while
    # asserting it was emitted: the check both returns `false` and says why.
    struct TrimUnsafeProbe <: PureOSQP.LinearSystem end
    PureOSQP.factorize!(::TrimUnsafeProbe, ws) = !isempty(Base.return_types(sin, (Float64,)))
    probe = @test_logs (:warn, r"trim-unsafe") match_mode = :any TypeContracts.check_trim_compat(
        TrimUnsafeProbe
    )
    @test !probe.passed

    function concrete_subtypes!(out, T)
        for S in subtypes(T)
            isabstracttype(S) ? concrete_subtypes!(out, S) : push!(out, S)
        end
        return out
    end
    # The probe above is a `LinearSystem` like any other and would otherwise be swept as one.
    backends = filter(!=(TrimUnsafeProbe), concrete_subtypes!(Type[], PureOSQP.LinearSystem))

    # Guards the guard: if loading an extension ever stops registering its backends, the sweep
    # below would pass by checking almost nothing.
    @test length(backends) >= 10

    # `check_trim_compat` reports by warning and returns its verdict; it does not throw. The
    # verdict is the `passed` field and reading it is the whole of the check.
    failed = String[]
    for B in backends
        r = TypeContracts.check_trim_compat(B)
        r.passed || push!(failed, string(nameof(B), ": ", join(get(r.issues, PureOSQP.LinearSystem, String[]), "; ")))
    end
    # Every backend, with no exceptions. A new one whose declared methods are not trim
    # compatible lands here rather than quietly shrinking the guarantee to whatever
    # `test/trim_tests.jl` happens to list.
    @test failed == String[]
end
