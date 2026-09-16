// Splits Clarabel.rs's time into the parts its own instrumentation exposes, for one problem
// file in the format ../../clarabel_rs_compare.jl writes.
//
// Three numbers come from outside the solver — `DefaultSolver::new` (setup), `solve()`
// (the interior-point loop), and their sum — each the minimum over repeats within a
// wall-clock budget, matching the field-wise minimum the Julia side reports.
//
// The rest comes from `Solver::timers`, which Clarabel keeps unconditionally: `solve` splits
// into `default start`, `IP iteration` and `post-process`, and `IP iteration` into
// `scale cones`, `kkt update` and `kkt solve`. Only `Timers::print` and `Timers::total_time`
// are public, so the hierarchy is printed rather than returned. Those timers are live in
// every build, so the `solve()` wall clock above already carries their overhead, as does the
// number bench/clarabel_rs_bench reports.
//
// Build (out-of-tree, as for the main driver):
//     cargo build --release --manifest-path bench/clarabel_rs/Cargo.toml --target-dir /tmp/pureosqp_clarabel_rs_target
// Run:
//     RAYON_NUM_THREADS=1 taskset -c 15 /tmp/pureosqp_clarabel_rs_target/release/split <problem.txt> [seconds] [tol]
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, NonnegativeConeT, SupportedConeT,
};

fn parse_ints(line: &str) -> Vec<usize> {
    line.split_whitespace().map(|s| s.parse().unwrap()).collect()
}

fn parse_floats(line: &str) -> Vec<f64> {
    line.split_whitespace().map(|s| s.parse().unwrap()).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: split <problem.txt> [seconds=0.5] [tol=1e-8]");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.5);
    let tol: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.0e-8);

    let file = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = BufReader::new(file).lines().map(|l| l.unwrap());
    let mut next = || lines.next().expect("truncated problem file");
    let header = parse_ints(&next());
    let (n, m) = (header[0], header[1]);
    let p_colptr = parse_ints(&next());
    let p_rowval = parse_ints(&next());
    let p_nzval = parse_floats(&next());
    let q = parse_floats(&next());
    let a_colptr = parse_ints(&next());
    let a_rowval = parse_ints(&next());
    let a_nzval = parse_floats(&next());
    let b = parse_floats(&next());

    let p = CscMatrix::new(n, n, p_colptr, p_rowval, p_nzval);
    let a = CscMatrix::new(m, n, a_colptr, a_rowval, a_nzval);
    let cones: Vec<SupportedConeT<f64>> = vec![NonnegativeConeT(m)];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .tol_feas(tol)
        .direct_solve_method("faer".to_string())
        .max_threads(1)
        .build()
        .expect("invalid Clarabel settings");

    let budget = Duration::from_secs_f64(seconds);
    let start = Instant::now();
    let (mut best_setup, mut best_solve, mut best_total) =
        (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut reps = 0usize;
    let mut solver =
        DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).expect("setup failed");
    loop {
        let t0 = Instant::now();
        let mut s = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone())
            .expect("Clarabel setup failed");
        let t1 = Instant::now();
        s.solve();
        let t2 = Instant::now();
        reps += 1;
        best_setup = best_setup.min((t1 - t0).as_secs_f64());
        best_solve = best_solve.min((t2 - t1).as_secs_f64());
        best_total = best_total.min((t2 - t0).as_secs_f64());
        solver = s;
        if start.elapsed() >= budget {
            break;
        }
    }

    let info = &solver.info;
    println!(
        "n={n} m={m} iterations={} status={:?} reps={reps}",
        info.iterations, solver.solution.status
    );
    println!(
        "linsolver={} nnzA={} nnzL={} threads={}",
        info.linsolver.name, info.linsolver.nnzA, info.linsolver.nnzL, info.linsolver.threads
    );
    println!(
        "min DefaultSolver::new = {:.3} us | min solve() = {:.3} us | min sum = {:.3} us",
        1.0e6 * best_setup,
        1.0e6 * best_solve,
        1.0e6 * best_total
    );
    println!("info.solve_time = {:.3} us (the solver's own clock over solve())", 1.0e6 * info.solve_time);
    println!("--- Solver::timers, last repeat (µs unless noted) ---");
    solver.timers.as_ref().expect("timers absent").print();
}
