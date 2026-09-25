#pragma once
// backend.h: GGML backend initialization.
//
// Every graph runs on one backend, picked once per context: the best device
// found by ggml_backend_load_all, or the one named by GGML_BACKEND (CUDA0,
// Vulkan0, CPU, ...). There is no scheduler and no per node CPU fallback:
// an op the device lacks fails the compute.

#include "ggml-backend.h"
#include "pt-error.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

// One thread per physical core (logical / 2 for HT/SMT): GEMM shares the
// SIMD units across hyperthreads.
static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Routes ggml logs through pt_log, whole lines only, collapsing runs of
// identical lines into a count.
static void pt_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    static std::mutex  mutex;
    static std::string pending;
    static std::string last;
    static int         count = 0;

    std::lock_guard<std::mutex> lock(mutex);
    pending += text;
    size_t end;
    while ((end = pending.find('\n')) != std::string::npos) {
        const std::string line = pending.substr(0, end);
        pending.erase(0, end + 1);
        if (count > 0 && line == last) {
            count++;
            continue;
        }
        if (count > 1) {
            pt_log(PT_LOG_INFO, "[Dedup] Previous line repeated %d times total", count);
        }
        pt_log(level == GGML_LOG_LEVEL_ERROR ? PT_LOG_ERROR :
               level == GGML_LOG_LEVEL_WARN  ? PT_LOG_WARN :
                                               PT_LOG_INFO,
               "[GGML] %s", line.c_str());
        last  = line;
        count = 1;
    }
}

// Returns a fresh backend with its own memory pool, NULL on failure.
static ggml_backend_t backend_init(void) {
    static const bool loaded = [] {
        ggml_log_set(pt_ggml_log, nullptr);
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;

    ggml_backend_t backend = nullptr;
    const char *   force   = std::getenv("GGML_BACKEND");
    if (force) {
        backend = ggml_backend_init_by_name(force, nullptr);
        if (!backend) {
            std::string msg = "[Load] GGML_BACKEND=";
            msg += force;
            msg += " not found. Available:";
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                msg += ' ';
                msg += ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            pt_log(PT_LOG_ERROR, "%s", msg.c_str());
            return nullptr;
        }
    } else {
        backend = ggml_backend_init_best();
    }
    if (!backend) {
        pt_log(PT_LOG_ERROR, "[Load] no backend available");
        return nullptr;
    }

    // The CPU device ignores its init params and defaults to 4 threads:
    // set the count through the registry proc address.
    int                n_threads = backend_cpu_n_threads();
    ggml_backend_dev_t dev       = ggml_backend_get_device(backend);
    if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto               set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(backend, n_threads);
        }
        pt_log(PT_LOG_INFO, "[Load] Backend: %s (%d threads)", ggml_backend_name(backend), n_threads);
    } else {
        pt_log(PT_LOG_INFO, "[Load] Backend: %s", ggml_backend_name(backend));
    }
    return backend;
}
