# 🚀 Parallel Matrix Engine

A high-performance linear algebra library written in **Rust**, designed to explore:

- ✅ **Parallel programming** with [Rayon](https://crates.io/crates/rayon)  
- ✅ **Blocked matrix multiplication (GEMM)** for cache efficiency  
- ✅ Future extensions with **SIMD acceleration**  

This project is ideal for learning about **HPC (High-Performance Computing)** in Rust.

---

## ✨ Features
- **Matrix struct (`Matrix`)** with row-major storage.  
- **Naive GEMM** for correctness verification.  
- **Parallel Blocked GEMM** using Rayon.  
- **Deterministic random matrix generation** (`StdRng`).  
- **Unit tests** with [`approx`](https://crates.io/crates/approx).  
- Ready for **Criterion benchmarks**.  

---

## 📂 Project Structure
parallel_matrix_engine/
├── Cargo.toml # Dependencies & metadata
├── src/
│ └── lib.rs # Core library: Matrix, GEMM implementations
├── benches/
│ └── gemm.rs # Criterion benchmarks (optional)
└── tests/
└── correctness.rs # Integration tests


---

## ⚡ Getting Started

### 🔧 Build
```bash
git clone https://github.com/NeNish/parallel_matrix_engine.git
cd parallel_matrix_engine
cargo build --release
```
#Run Tests
Verify correctness of parallel vs naive GEMM:
```
cargo test
```
###📊 Run Benchmarks

Run Criterion benchmarks (if benches/gemm.rs is added):
```
cargo bench
```
###🔬 Example Usage

```
use parallel_matrix_engine::{Matrix, gemm_parallel};

fn main() {
    // Create two random matrices
    let a = Matrix::random(256, 256, 42);
    let b = Matrix::random(256, 256, 1337);

    // Multiply using the parallel engine
    let c = gemm_parallel(&a, &b);

    println!("Resulting matrix C has dimensions {}x{}", c.rows, c.cols);
}
```
Compile & run:
```
cargo run --example multiply
```
### 🚀 Roadmap

 SIMD micro-kernel with AVX2/AVX-512 (std::arch intrinsics).

 Strassen’s / Winograd’s algorithm variants.

 GPU backend (CUDA / Vulkan compute).

 Sparse matrix support.

### 📜 License

Licensed under the MIT License.
Feel free to use, modify, and contribute!

### 🙌 Acknowledgements

Rayon
 for safe parallelism.

Criterion
 for benchmarking.

Rust HPC community for inspiration.
