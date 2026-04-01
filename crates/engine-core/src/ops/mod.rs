//! Core Operations
//! 
//! Implementations of neural network operations. These are the "hot path" functions
//! that must be optimized for maximum performance.

use crate::arena::{TensorView, TensorViewMut};

/// ReLU activation: f(x) = max(0, x)
#[inline]
pub fn relu(input: &TensorView<f32>, mut output: TensorViewMut<f32>) {
    let input_data = input.data();
    let output_data = output.data_mut();
    
    assert_eq!(input_data.len(), output_data.len(), "Input/output size mismatch");
    
    // Vectorized ReLU - compiler will auto-vectorize this
    for (out, &inp) in output_data.iter_mut().zip(input_data.iter()) {
        *out = inp.max(0.0);
    }
}

/// ReLU activation (in-place)
#[inline]
pub fn relu_inplace(mut tensor: TensorViewMut<f32>) {
    let data = tensor.data_mut();
    for x in data.iter_mut() {
        *x = x.max(0.0);
    }
}

/// GELU activation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu(input: &TensorView<f32>, mut output: TensorViewMut<f32>) {
    let input_data = input.data();
    let output_data = output.data_mut();
    
    assert_eq!(input_data.len(), output_data.len(), "Input/output size mismatch");
    
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    
    for (out, &x) in output_data.iter_mut().zip(input_data.iter()) {
        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        *out = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Element-wise addition: C = A + B
#[inline]
pub fn add(a: &TensorView<f32>, b: &TensorView<f32>, mut c: TensorViewMut<f32>) {
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data_mut();
    
    assert_eq!(a_data.len(), b_data.len(), "Tensor size mismatch");
    assert_eq!(a_data.len(), c_data.len(), "Output size mismatch");
    
    for ((out, &a_val), &b_val) in c_data.iter_mut().zip(a_data.iter()).zip(b_data.iter()) {
        *out = a_val + b_val;
    }
}

/// Naive matrix multiplication (for reference/testing)
/// C = A * B where A is [m x k], B is [k x n], C is [m x n]
pub fn matmul_naive(
    a: &TensorView<f32>,
    b: &TensorView<f32>,
    mut c: TensorViewMut<f32>,
    m: usize,
    k: usize,
    n: usize,
) {
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data_mut();
    
    assert_eq!(a_data.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b_data.len(), k * n, "Matrix B size mismatch");
    assert_eq!(c_data.len(), m * n, "Matrix C size mismatch");
    
    // Zero the output
    c_data.fill(0.0);
    
    // Standard three-loop matmul (O(n³))
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
}

/// Blocked matrix multiplication (cache-friendly)
/// Uses tiling to improve cache locality
pub fn matmul_blocked(
    a: &TensorView<f32>,
    b: &TensorView<f32>,
    mut c: TensorViewMut<f32>,
    m: usize,
    k: usize,
    n: usize,
) {
    const BLOCK_SIZE: usize = 64; // Tuned for typical L1 cache
    
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data_mut();
    
    // Zero the output
    c_data.fill(0.0);
    
    // Blocked algorithm
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                // Process a block
                let i_max = (ii + BLOCK_SIZE).min(m);
                let j_max = (jj + BLOCK_SIZE).min(n);
                let k_max = (kk + BLOCK_SIZE).min(k);
                
                for i in ii..i_max {
                    for j in jj..j_max {
                        let mut sum = c_data[i * n + j];
                        for p in kk..k_max {
                            sum += a_data[i * k + p] * b_data[p * n + j];
                        }
                        c_data[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

/// Fused Linear + ReLU operation
/// Avoids writing intermediate results to memory
#[inline]
pub fn fused_linear_relu(
    input: &TensorView<f32>,
    weights: &TensorView<f32>,
    mut output: TensorViewMut<f32>,
    m: usize,
    k: usize,
    n: usize,
) {
    let input_data = input.data();
    let weights_data = weights.data();
    let output_data = output.data_mut();
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += input_data[i * k + p] * weights_data[p * n + j];
            }
            // Fused ReLU
            output_data[i * n + j] = sum.max(0.0);
        }
    }
}

/// Softmax: exp(x_i) / sum(exp(x_j))
/// Numerically stable implementation using max trick
pub fn softmax(input: &TensorView<f32>, mut output: TensorViewMut<f32>) {
    let input_data = input.data();
    let output_data = output.data_mut();
    
    let len = input_data.len();
    assert_eq!(len, output_data.len(), "Input/output size mismatch");
    
    if len == 0 {
        return;
    }
    
    // Find max for numerical stability
    let max_val = input_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for (out, &inp) in output_data.iter_mut().zip(input_data.iter()) {
        let exp_val = (inp - max_val).exp();
        *out = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    let inv_sum = 1.0 / sum;
    for out in output_data.iter_mut() {
        *out *= inv_sum;
    }
}

/// Layer Normalization
/// norm(x) = (x - mean) / sqrt(variance + epsilon)
pub fn layer_norm(input: &TensorView<f32>, mut output: TensorViewMut<f32>, epsilon: f32) {
    let input_data = input.data();
    let output_data = output.data_mut();
    
    let len = input_data.len() as f32;
    
    // Compute mean
    let mean = input_data.iter().sum::<f32>() / len;
    
    // Compute variance
    let variance = input_data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / len;
    
    // Normalize
    let inv_std = 1.0 / (variance + epsilon).sqrt();
    for (out, &inp) in output_data.iter_mut().zip(input_data.iter()) {
        *out = (inp - mean) * inv_std;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    #[test]
    fn test_relu() {
        let mut arena: Arena<f32> = Arena::new(0, 200, 0);
        
        // Setup input
        {
            let mut input = arena.view_mut(crate::arena::Section::Activations, 0, 5, vec![5]);
            input.data_mut().copy_from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        }
        
        // Run ReLU
        {
            let input = arena.view(crate::arena::Section::Activations, 0, 5, vec![5]);
            let output = arena.view_mut(crate::arena::Section::Activations, 5, 5, vec![5]);
            relu(&input, output);
        }
        
        // Check output
        {
            let output = arena.view(crate::arena::Section::Activations, 5, 5, vec![5]);
            let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
            assert_eq!(output.data(), &expected);
        }
    }

    #[test]
    fn test_matmul_small() {
        let mut arena: Arena<f32> = Arena::new(0, 500, 0);
        
        // A = [2x3], B = [3x2], C = [2x2]
        {
            let mut a = arena.view_mut(crate::arena::Section::Activations, 0, 6, vec![2, 3]);
            a.data_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            
            let mut b = arena.view_mut(crate::arena::Section::Activations, 10, 6, vec![3, 2]);
            b.data_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        
        // Compute C = A * B
        {
            let a = arena.view(crate::arena::Section::Activations, 0, 6, vec![2, 3]);
            let b = arena.view(crate::arena::Section::Activations, 10, 6, vec![3, 2]);
            let c = arena.view_mut(crate::arena::Section::Activations, 20, 4, vec![2, 2]);
            
            matmul_naive(&a, &b, c, 2, 3, 2);
        }
        
        // Check result: [22, 28], [49, 64]
        {
            let c = arena.view(crate::arena::Section::Activations, 20, 4, vec![2, 2]);
            let expected = [22.0, 28.0, 49.0, 64.0];
            assert_eq!(c.data(), &expected);
        }
    }

    #[test]
    fn test_softmax() {
        let mut arena: Arena<f32> = Arena::new(0, 200, 0);
        
        {
            let mut input = arena.view_mut(crate::arena::Section::Activations, 0, 3, vec![3]);
            input.data_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        }
        
        {
            let input = arena.view(crate::arena::Section::Activations, 0, 3, vec![3]);
            let output = arena.view_mut(crate::arena::Section::Activations, 10, 3, vec![3]);
            softmax(&input, output);
        }
        
        {
            let output = arena.view(crate::arena::Section::Activations, 10, 3, vec![3]);
            let sum: f32 = output.data().iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1.0");
        }
    }
}
