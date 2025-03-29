#ifndef ARM64_SIMD_H
#define ARM64_SIMD_H

#ifdef __aarch64__
#include <arm_neon.h>

// Optimized dot product for ARM64 NEON
inline float dot_product_neon(const float* a, const float* b, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
    }
    
    // Reduce vector sum
    float sum = vaddvq_f32(sum_vec);
    
    // Process remaining elements
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

#endif // __aarch64__

#endif // ARM64_SIMD_H
