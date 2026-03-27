//! x86_64 SIMD Implementations
//! 
//! AVX2 and AVX-512 optimized kernels for maximum performance on Intel/AMD CPUs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized matrix multiplication
/// 
/// Uses 256-bit SIMD registers to process 8 floats at a time
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn matmul_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const BLOCK_SIZE: usize = 64;
    
    // Zero output
    c.fill(0.0);

    // Blocked algorithm with AVX2
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                let i_max = (ii + BLOCK_SIZE).min(m);
                let j_max = (jj + BLOCK_SIZE).min(n);
                let k_max = (kk + BLOCK_SIZE).min(k);

                // Process block with SIMD
                for i in ii..i_max {
                    for j in (jj..j_max).step_by(8) {
                        if j + 8 > j_max {
                            // Handle remainder with scalar code
                            for j_scalar in j..j_max {
                                let mut sum = c[i * n + j_scalar];
                                for p in kk..k_max {
                                    sum += a[i * k + p] * b[p * n + j_scalar];
                                }
                                c[i * n + j_scalar] = sum;
                            }
                            continue;
                        }

                        // Load current C values (8 floats)
                        let c_ptr = c.as_mut_ptr().add(i * n + j);
                        let mut acc = _mm256_loadu_ps(c_ptr);

                        // Compute dot product
                        for p in kk..k_max {
                            // Broadcast A[i,p] to all 8 lanes
                            let a_val = _mm256_set1_ps(a[i * k + p]);
                            
                            // Load B[p, j:j+8]
                            let b_ptr = b.as_ptr().add(p * n + j);
                            let b_vec = _mm256_loadu_ps(b_ptr);
                            
                            // FMA: acc = acc + (a_val * b_vec)
                            acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                        }

                        // Store result
                        _mm256_storeu_ps(c_ptr, acc);
                    }
                }
            }
        }
    }
}

/// AVX-512 optimized version (processes 16 floats at once)
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_avx512(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    c.fill(0.0);

    // Similar to AVX2 but with 512-bit registers (16 floats)
    for i in 0..m {
        for j in (0..n).step_by(16) {
            if j + 16 > n {
                // Scalar remainder
                for j_scalar in j..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j_scalar];
                    }
                    c[i * n + j_scalar] = sum;
                }
                continue;
            }

            let c_ptr = c.as_mut_ptr().add(i * n + j);
            let mut acc = _mm512_loadu_ps(c_ptr);

            for p in 0..k {
                let a_val = _mm512_set1_ps(a[i * k + p]);
                let b_ptr = b.as_ptr().add(p * n + j);
                let b_vec = _mm512_loadu_ps(b_ptr);
                acc = _mm512_fmadd_ps(a_val, b_vec, acc);
            }

            _mm512_storeu_ps(c_ptr, acc);
        }
    }
}

/// Optimized ReLU using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn relu_avx2(data: &mut [f32]) {
    let len = data.len();
    let chunks = len / 8;
    
    let zero = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 8);
        let vec = _mm256_loadu_ps(ptr);
        let result = _mm256_max_ps(vec, zero);
        _mm256_storeu_ps(ptr, result);
    }
    
    // Handle remainder
    for i in (chunks * 8)..len {
        data[i] = data[i].max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matmul() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        unsafe {
            matmul_avx2(&a, &b, &mut c, 2, 3, 2);
        }

        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_relu() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -5.0, 10.0];
        
        unsafe {
            relu_avx2(&mut data);
        }

        assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 10.0]);
    }
}
