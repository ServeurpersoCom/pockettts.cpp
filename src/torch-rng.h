#pragma once
// torch-rng.h: torch.normal_ on a CPU float tensor, reproduced.
//
// torch.manual_seed(s) seeds the CPU generator with the standard MT19937
// init (std::mt19937). A float uniform takes the low 24 bits of one 32-bit
// draw. normal_ on a contiguous float tensor of at least 16 elements fills
// it with uniforms, then turns each block of 16 into normals with
// Box-Muller: u1 = 1 - x[j], u2 = x[j + 8], r = sqrt(-2 log u1),
// x[j] = r cos(2 pi u2) * std, x[j + 8] = r sin(2 pi u2) * std (mean 0).
// A trailing partial block redraws the last 16 values. The same seed gives
// the same noise as the reference on CPU, up to libm rounding.

#include <cmath>
#include <cstdint>
#include <random>

struct TorchRng {
    std::mt19937 mt;

    explicit TorchRng(uint32_t seed) : mt(seed) {}

    float uniform() { return (float) ((double) (mt() & ((1u << 24) - 1)) * (1.0 / 16777216.0)); }

    static void fill16(float * x, float std) {
        for (int j = 0; j < 8; j++) {
            const float u1    = 1.0f - x[j];
            const float u2    = x[j + 8];
            const float r     = std::sqrt(-2.0f * std::log(u1));
            const float theta = 2.0f * 3.14159265358979323846f * u2;
            x[j]              = r * std::cos(theta) * std;
            x[j + 8]          = r * std::sin(theta) * std;
        }
    }

    // n >= 16, mean 0.
    void normal(float * x, int n, float std) {
        for (int i = 0; i < n; i++) {
            x[i] = uniform();
        }
        for (int i = 0; i + 16 <= n; i += 16) {
            fill16(x + i, std);
        }
        if (n % 16) {
            float * t = x + n - 16;
            for (int i = 0; i < 16; i++) {
                t[i] = uniform();
            }
            fill16(t, std);
        }
    }
};
