@testitem "OperatorSplitting honours the Solution and Status contract" begin
    using PureQPBase, PureOSQP
    # The assertions live in PureQPBase, which owns `Solution` and `Status`, so both
    # algorithms are held to one statement of what their values mean rather than to
    # whatever each package's own suite happens to check.
    PureQPBase.conforms(OperatorSplitting(); eps = 1.0e-9, slow_iters = 5)
end
