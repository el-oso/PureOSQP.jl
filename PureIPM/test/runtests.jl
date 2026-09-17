using TestItemRunner

# Items tagged `:gpu` are not run. They exercise the GPU array path through JLArrays, and
# compiling that path crashes Julia 1.13.0 inside LLVM's loop vectorizer on AVX-512 targets.
@run_package_tests filter = ti -> !(:gpu in ti.tags)
