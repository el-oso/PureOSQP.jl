@testitem "every test file parses and declares the test items it should" begin
    # A test file with a syntax error, or one whose `@testitem` header is damaged by an
    # edit, is silently skipped by the runner rather than reported: the suite still says
    # "passed", just with fewer items. This asserts the inventory so that cannot happen
    # unnoticed. Update the counts deliberately when adding or removing a test item.
    expected = Dict(
        "c_suite_tests.jl" => 9,
        "corpus_tests.jl" => 4,
        "derivative_tests.jl" => 5,
        "gpu_tests.jl" => 3,
        "indirect_tests.jl" => 3,
        "linsys_tests.jl" => 12,
        "meta_tests.jl" => 1,
        "moi_tests.jl" => 2,
        "oracle_tests.jl" => 4,
        "polish_tests.jl" => 3,
        "scaling_tests.jl" => 7,
        "setup_tests.jl" => 10,
        "solve_tests.jl" => 21,
        "trim_tests.jl" => 1,
        "update_tests.jl" => 5,
    )
    dir = @__DIR__
    files = sort(filter(f -> endswith(f, "_tests.jl"), readdir(dir)))
    @test files == sort(collect(keys(expected)))
    for f in files
        src = read(joinpath(dir, f), String)
        @test Meta.parse("begin\n$src\nend"; raise = false) isa Expr
        n = count(m -> true, eachmatch(r"^@testitem "m, src))
        @test n == expected[f]
    end
end
