use parallel_matrix_engine::{Matrix, gemm_parallel, gemm_naive};
use rand::Rng;
use std::time::Instant;

fn main() {
    // Big enough to see parallel speedups
    let n = 1024usize;
    let seed = 42u64;
    let a = Matrix::random(n, n, seed);
    let b = Matrix::random(n, n, seed + 1);

    // Warm up
    let _ = gemm_parallel(&a, &b);

    // Time parallel
    let t0 = Instant::now();
    let _c = gemm_parallel(&a, &b);
    let t_par = t0.elapsed();

    // For reference, naive for small sizes only
    // let t1 = Instant::now();
    // let _c2 = gemm_naive(&a, &b);
    // let t_naive = t1.elapsed();

    println!("GEMM parallel {}x{} took {:?}", n, n, t_par);
}
