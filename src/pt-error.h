#pragma once
// pt-error.h: internal helpers backing pt_last_error() and the pt_log_set
// callback routing. Not part of the public ABI; storage and readers live
// in pocket.cpp. The error slot is thread local, pt_log writes to the
// installed callback or to stderr with a trailing newline.

#include "pocket.h"

#include <cstdarg>

void pt_set_error(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

// Throws std::runtime_error formatted with printf semantics, caught at the
// ABI boundary and turned into pt_set_error.
[[noreturn]] void pt_throw(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void pt_log(enum pt_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
