module PureIPMMathOptInterfaceExt

"""
    PureIPMMathOptInterfaceExt

`PureIPM.Optimizer`, the MathOptInterface wrapper around [`InteriorPoint`](@ref). The wrapper
itself is PureQPBase's and serves any algorithm; this supplies the one it runs and the name a
caller reaches it by.
"""

import MathOptInterface as MOI
import PureIPM
import PureQPBase

# PureQPBase's own extension is not a dependency of this one, so it is not loaded while this
# is precompiled and the module is looked up when an optimizer is built instead. Both
# extensions are loaded by then: each needs only MathOptInterface, which the caller has.
function PureIPM.Optimizer(; kwargs...)
    wrapper = Base.get_extension(PureQPBase, :PureQPBaseMathOptInterfaceExt)
    return wrapper.Optimizer{Float64}(PureIPM.InteriorPoint; kwargs...)
end

end
