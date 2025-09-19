use parallel_matrix_engine::{Matrix, gemm_naive, gemm_parallel};
use approx::assert_abs_diff_eq;

#[test]
fn small_matrices_match() {
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0,
                                        3.0, 4.0]);
    let b = Matrix::from_vec(2, 2, vec![5.0, 6.0,
                                        7.0, 8.0]);

    let c_naive = gemm_naive(&a, &b);
    let c_par   = gemm_parallel(&a, &b);

    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(c_naive[(i,j)], c_par[(i,j)], epsilon = 1e-6);
        }
    }
}

#[test]
fn random_matrix_consistency() {
    let n = 64;
    let a = Matrix::random(n, n, 42);
    let b = Matrix::random(n, n, 43);

    let c_naive = gemm_naive(&a, &b);
    let c_par   = gemm_parallel(&a, &b);

    for i in 0..n {
        for j in 0..n {
            assert_abs_diff_eq!(c_naive[(i,j)], c_par[(i,j)], epsilon = 1e-5);
        }
    }
}
