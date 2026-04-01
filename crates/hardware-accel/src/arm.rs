//! ARM NEON SIMD Implementations
//! 
//! Optimized kernels for ARM processors (Apple Silicon, AWS Graviton, etc.)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized matrix multiplication
/// 
/// Uses 128-bit SIMD registers to process 4 floats at a time
#[target_feature(enable = "neon")]
pub unsafe fn matmul_neon(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const BLOCK_SIZE: usize = 32;
    
    // Zero output
    c.fill(0.0);

    // Blocked algorithm with NEON
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                let i_max = (ii + BLOCK_SIZE).min(m);
                let j_max = (jj + BLOCK_SIZE).min(n);
                let k_max = (kk + BLOCK_SIZE).min(k);

                for i in ii..i_max {
                    for j in (jj..j_max).step_by(4) {
                        if j + 4 > j_max {
                            // Scalar remainder
                            for j_scalar in j..j_max {
                                let mut sum = c[i * n + j_scalar];
                                for p in kk..k_max {
                                    sum += a[i * k + p] * b[p * n + j_scalar];
                                }
                                c[i * n + j_scalar] = sum;
                            }
                            continue;
                        }

                        // Load current C values (4 floats)
                        let c_ptr = c.as_mut_ptr().add(i * n + j);
                        let mut acc = vld1q_f32(c_ptr);

                        // Compute dot product
                        for p in kk..k_max {
                            // Broadcast A[i,p] to all 4 lanes
                            let a_val = vdupq_n_f32(a[i * k + p]);
                            
                            // Load B[p, j:j+4]
                            let b_ptr = b.as_ptr().add(p * n + j);
                            let b_vec = vld1q_f32(b_ptr);
                            
                            // FMA: acc = acc + (a_val * b_vec)
                            acc = vfmaq_f32(acc, a_val, b_vec);
                        }

                        // Store result
                        vst1q_f32(c_ptr, acc);
                    }
                }
            }
        }
    }
}

/// NEON-optimized ReLU
#[target_feature(enable = "neon")]
pub unsafe fn relu_neon(data: &mut [f32]) {
    let len = data.len();
    let chunks = len / 4;
    
    let zero = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 4);
        let vec = vld1q_f32(ptr);
        let result = vmaxq_f32(vec, zero);
        vst1q_f32(ptr, result);
    }
    
    // Handle remainder
    for i in (chunks * 4)..len {
        data[i] = data[i].max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_matmul() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            println!("NEON not available, skipping test");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 4];

        unsafe {
            matmul_neon(&a, &b, &mut c, 2, 3, 2);
        }

        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
