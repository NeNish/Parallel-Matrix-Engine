// src/lib.rs
//! Parallel Matrix Engine
//!
//! Safe, easy-to-follow blocked, parallel GEMM using Rayon.
//!
//! The code uses an `unsafe` slice creation when handing disjoint mutable slices
//! of the output matrix `C` into Rayon threads. This is sound because the row
//! blocks are non-overlapping. The inner micro-kernel (multiply_add_block_slice)
//! is scalar and a clear hook to replace with SIMD intrinsics later.

use rayon::prelude::*;
use std::ops::{Index, IndexMut};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // row-major
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0_f32; rows * cols];
        Self { rows, cols, data }
    }

    pub fn from_fill(rows: usize, cols: usize, value: f32) -> Self {
        let data = vec![value; rows * cols];
        Self { rows, cols, data }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(rng.gen::<f32>());
        }
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[r * self.cols + c] = val;
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let (r, c) = idx;
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let (r, c) = idx;
        &mut self.data[r * self.cols + c]
    }
}

/// Naive GEMM for correctness checks
pub fn gemm_naive(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let m = a.rows;
    let n = b.cols;
    let k = a.cols;
    let mut c = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for kk in 0..k {
                sum += a.get(i, kk) * b.get(kk, j);
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// Blocked, parallel GEMM (C = A * B)
///
/// This parallelizes across disjoint row-blocks of C and constructs
/// a mutable slice for each block. That slice is non-overlapping, so
/// using `unsafe` to create `&mut [f32]` from a raw pointer is sound.
pub fn gemm_parallel(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows);
    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    let block_m = 64usize.min(m).max(16);
    let block_n = 64usize.min(n).max(16);
    let block_k = 64usize.min(k).max(16);

    let mut c = Matrix::new(m, n);

    // Split C into mutable row-blocks *safely* before parallel work
    let row_blocks: Vec<(usize, usize, &mut [f32])> = {
        let ptr = c.data.as_mut_slice();
        (0..m)
            .step_by(block_m)
            .map(|ri| {
                let r_end = (ri + block_m).min(m);
                let start = ri * n;
                let end = r_end * n;
                (ri, r_end, &mut ptr[start..end])
            })
            .collect()
    };

    row_blocks.into_par_iter().for_each(|(ri, r_end, c_slice)| {
        for cj in (0..n).step_by(block_n) {
            let c_j_end = (cj + block_n).min(n);
            for kk in (0..k).step_by(block_k) {
                let k_end = (kk + block_k).min(k);
                multiply_add_block_slice(
                    a,
                    b,
                    c_slice,
                    ri,
                    r_end,
                    cj,
                    c_j_end,
                    kk,
                    k_end,
                    n,
                );
            }
        }
    });

    c
}

/// Inner kernel operating on a mutable slice that corresponds to rows [r0..r1) of C.
/// The `c_slice` has shape ((r1-r0) x n) stored row-major, and `n` is the full C stride (cols).
#[inline(always)]
fn multiply_add_block_slice(
    a: &Matrix,
    b: &Matrix,
    c_slice: &mut [f32],
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    k0: usize,
    k1: usize,
    n: usize, // stride of C (number of columns)
) {
    let a_cols = a.cols;
    let b_cols = b.cols;

    // local_i is index into c_slice rows
    for i in r0..r1 {
        let local_i = i - r0;
        let c_row_offset = local_i * n;
        for kk in k0..k1 {
            let a_ik = a.data[i * a_cols + kk];
            let b_row_offset = kk * b_cols;
            for j in c0..c1 {
                // c_slice index is (local_i * n + j)
                c_slice[c_row_offset + j] += a_ik * b.data[b_row_offset + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gemm_correct_small() {
        let a = Matrix::from_vec(
            2,
            3,
            vec![1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0],
        );
        let b = Matrix::from_vec(
            3,
            2,
            vec![7.0, 8.0,
                 9.0, 10.0,
                 11.0, 12.0],
        );
        let c_naive = gemm_naive(&a, &b);
        let c_par = gemm_parallel(&a, &b);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(c_naive[(i, j)], c_par[(i, j)], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn random_matrix_consistency() {
        let n = 64;
        let a = Matrix::random(n, n, 42);
        let b = Matrix::random(n, n, 43);

        let c_naive = gemm_naive(&a, &b);
        let c_par = gemm_parallel(&a, &b);

        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(c_naive[(i, j)], c_par[(i, j)], epsilon = 1e-5);
            }
        }
    }
}
