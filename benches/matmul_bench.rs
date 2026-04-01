//! Matrix Multiplication Benchmarks
//! 
//! Compare naive, blocked, and SIMD implementations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engine_core::arena::{Arena, Section};
use engine_core::ops::{matmul_naive, matmul_blocked};
use hardware_accel::matmul_simd;

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    
    // Test different matrix sizes
    for size in [64, 128, 256, 512].iter() {
        let m = *size;
        let k = *size;
        let n = *size;
        
        // Setup arena
        let mut arena: Arena<f32> = Arena::new(
            0,
            m * k + k * n + m * n + 1000,
            0,
        );

        // Initialize matrices with random-ish data
        {
            let mut a = arena.view_mut(Section::Activations, 0, m * k, vec![m, k]);
            for (i, x) in a.data_mut().iter_mut().enumerate() {
                *x = (i % 100) as f32 / 100.0;
            }

            let mut b = arena.view_mut(Section::Activations, m * k, k * n, vec![k, n]);
            for (i, x) in b.data_mut().iter_mut().enumerate() {
                *x = (i % 100) as f32 / 100.0;
            }
        }

        // Benchmark naive implementation
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            size,
            |b, _| {
                b.iter(|| {
                    let a = arena.view(Section::Activations, 0, m * k, vec![m, k]);
                    let b_mat = arena.view(Section::Activations, m * k, k * n, vec![k, n]);
                    let c = arena.view_mut(
                        Section::Activations,
                        m * k + k * n,
                        m * n,
                        vec![m, n],
                    );
                    matmul_naive(&a, &b_mat, c, m, k, n);
                });
            },
        );

        // Benchmark blocked implementation
        group.bench_with_input(
            BenchmarkId::new("blocked", size),
            size,
            |b, _| {
                b.iter(|| {
                    let a = arena.view(Section::Activations, 0, m * k, vec![m, k]);
                    let b_mat = arena.view(Section::Activations, m * k, k * n, vec![k, n]);
                    let c = arena.view_mut(
                        Section::Activations,
                        m * k + k * n,
                        m * n,
                        vec![m, n],
                    );
                    matmul_blocked(&a, &b_mat, c, m, k, n);
                });
            },
        );

        // Benchmark SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            size,
            |b, _| {
                b.iter(|| {
                    let a = arena.view(Section::Activations, 0, m * k, vec![m, k]);
                    let b_mat = arena.view(Section::Activations, m * k, k * n, vec![k, n]);
                    let mut c = arena.view_mut(
                        Section::Activations,
                        m * k + k * n,
                        m * n,
                        vec![m, n],
                    );
                    matmul_simd(a.data(), b_mat.data(), c.data_mut(), m, k, n);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
