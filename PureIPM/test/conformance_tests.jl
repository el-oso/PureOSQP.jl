@testitem "InteriorPoint honours the Solution and Status contract" begin
    using PureQPBase, PureIPM
    # The assertions live in PureQPBase, which owns `Solution` and `Status`, so both
    # algorithms are held to one statement of what their values mean rather than to
    # whatever each package's own suite happens to check.
    PureQPBase.conforms(InteriorPoint(); eps = 1.0e-8, slow_iters = 2)
end
