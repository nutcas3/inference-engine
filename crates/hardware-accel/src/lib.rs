//! SIMD-Accelerated Operations
//! 
//! Hardware-specific optimizations using AVX2/AVX-512 on x86_64
//! and NEON on ARM (Rust 1.88+ supports stable SIMD intrinsics)

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

/// SIMD-optimized matrix multiplication
/// 
/// Automatically dispatches to the best available implementation
/// based on CPU features detected at runtime
pub fn matmul_simd(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { x86::matmul_avx2(a, b, c, m, k, n) }
        } else {
            matmul_fallback(a, b, c, m, k, n);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { arm::matmul_neon(a, b, c, m, k, n) }
        } else {
            matmul_fallback(a, b, c, m, k, n);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        matmul_fallback(a, b, c, m, k, n);
    }
}

/// Fallback implementation (portable, no SIMD)
#[inline]
fn matmul_fallback(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    c.fill(0.0);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 2x3
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // 3x2
        let mut c = vec![0.0; 4];  // 2x2

        matmul_simd(&a, &b, &mut c, 2, 3, 2);

        // Expected: [22, 28], [49, 64]
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
