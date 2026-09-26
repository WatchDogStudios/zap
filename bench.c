/* bench.c - self-test, fuzz, and speed.  usage: bench [file] [threads] */
#define ZAP_THREADS
#include "zap.h"
#include "zap_pak.h"
#undef NDEBUG /* the self-tests are asserts: keep them in release builds */
#include <assert.h>
#include <stdio.h>
#include <time.h>

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + t.tv_nsec * 1e-9; }
static zap_state st;
static zap_hc_state *hc;
static zap_dict dict;

static void fuzz(const uint8_t *c, size_t cn, size_t n, const zap_dict *d) {
    uint8_t *o = malloc(n + 1), *f = malloc(cn + 1);
    for (int i = 0; i < 100; i++) { /* corrupt input must not crash (run the ASan build) */
        memcpy(f, c, cn);
        f[rand() % cn] ^= (uint8_t)(1 + rand() % 255);
        zap_decompress(f, cn - (i & 1) * (rand() % cn), o, n, d);
        zap_frame_decode(f, cn, o, n, d);
    }
    free(o); free(f);
}

/* entropy blocks: round trip with and without caller scratch, exact-size checks, corrupt input */
static void e_roundtrip(const uint8_t *src, size_t n, const zap_dict *d, int depth) {
    size_t cap = 2 * n + 1024, sc = zap_entropy_scratch(n);
    uint8_t *c = malloc(cap), *o = malloc(n + 1), *scratch = malloc(sc), *f = malloc(cap);
    size_t cn = zap_compress_entropy(src, n, c, cap, depth ? (void *)hc : (void *)&st, depth, d);
    assert(cn > 0);
    assert(zap_decompress_entropy(c, cn, o, n, d, NULL, 0) == (ptrdiff_t)n && memcmp(src, o, n) == 0);
    memset(o, 0, n);
    assert(zap_decompress_entropy(c, cn, o, n, d, scratch, sc) == (ptrdiff_t)n && memcmp(src, o, n) == 0);
    assert(zap_decompress_entropy(c, cn, o, n + 1, d, scratch, sc) == -1);
    if (n) assert(zap_decompress_entropy(c, cn, o, n - 1, d, NULL, 0) == -1);
    assert(zap_compress_entropy(src, n, c, cn - 1, depth ? (void *)hc : (void *)&st, depth, d) == 0);
    for (int i = 0; i < 200; i++) { /* corrupt input must fail or succeed, never crash */
        memcpy(f, c, cn);
        f[rand() % cn] ^= (uint8_t)(1 + rand() % 255);
        zap_decompress_entropy(f, cn - (i & 1) * (rand() % cn), o, n, d, i & 2 ? scratch : NULL, sc);
    }
    free(c); free(o); free(scratch); free(f);
}

static size_t roundtrip(const uint8_t *src, size_t n, const zap_dict *d, int depth) {
    e_roundtrip(src, n, d, depth);
    size_t cap = zap_bound(n);
    uint8_t *c = malloc(cap), *o = malloc(n + 1);
    size_t cn = depth ? zap_compress_hc(src, n, c, cap, hc, d, depth) : zap_compress(src, n, c, cap, &st, d);
    assert(cn > 0);
    assert(zap_decompress(c, cn, o, n, d) == (ptrdiff_t)n && memcmp(src, o, n) == 0);
    assert(zap_decompress(c, cn, o, n + 1, d) == -1);                    /* wrong raw size is rejected */
    if (n) assert(zap_decompress(c, cn, o, n - 1, d) == -1);
    assert(n == 0 || (depth ? zap_compress_hc(src, n, c, cn - 1, hc, d, depth) : zap_compress(src, n, c, cn - 1, &st, d)) == 0);
    fuzz(c, cn, n, d);
    free(c); free(o);
    return cn;
}

/* incompressible blocks take the fast path inside hc and entropy mode (zap__hc_skip): still a valid round trip */
/* 1833-byte entropy block (version 1, written by zap 1.2.0) of v1_input() */
static const uint8_t kV1[] = {
    250, 6, 0, 0, 83, 0, 0, 0, 61, 0, 0, 0, 75, 0, 0, 0, 1, 215, 5, 0, 0, 102, 102, 103,
    102, 102, 103, 102, 102, 101, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 102, 102, 102, 102, 103, 102, 102,
    102, 102, 102, 102, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 160, 144,
    0, 0, 0, 0, 170, 10, 154, 8, 9, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 88, 1, 0, 0, 81, 1, 0, 0, 82, 1, 0, 0, 191, 254, 254, 249, 239, 159, 127,
    255, 247, 255, 116, 254, 207, 175, 223, 63, 254, 159, 95, 127, 253, 252, 243, 159, 159, 158, 69, 103, 2, 22, 55,
    134, 190, 224, 145, 150, 35, 90, 233, 67, 189, 141, 30, 191, 191, 204, 158, 126, 249, 45, 89, 72, 100, 151, 43,
    207, 46, 52, 3, 191, 81, 249, 81, 49, 87, 144, 199, 181, 194, 151, 93, 145, 166, 100, 155, 22, 249, 1, 39,
    129, 102, 156, 196, 51, 111, 107, 200, 35, 214, 157, 233, 17, 246, 64, 89, 126, 60, 184, 87, 83, 142, 64, 35,
    240, 222, 71, 31, 231, 195, 57, 55, 162, 178, 119, 217, 65, 222, 19, 178, 10, 80, 205, 129, 143, 239, 238, 246,
    240, 185, 17, 51, 151, 210, 180, 96, 224, 113, 159, 64, 77, 46, 104, 38, 142, 82, 80, 2, 63, 185, 191, 42,
    81, 16, 249, 181, 69, 77, 158, 73, 191, 202, 50, 129, 99, 49, 68, 225, 211, 117, 84, 242, 234, 53, 176, 175,
    53, 122, 166, 237, 6, 184, 23, 77, 211, 168, 145, 100, 162, 169, 112, 46, 10, 201, 254, 49, 217, 164, 111, 75,
    0, 10, 93, 98, 229, 64, 34, 156, 115, 231, 161, 27, 156, 62, 186, 227, 136, 147, 28, 210, 173, 137, 73, 79,
    3, 121, 190, 50, 60, 205, 139, 166, 241, 166, 70, 3, 115, 13, 172, 189, 161, 201, 85, 14, 207, 169, 222, 48,
    55, 173, 131, 237, 61, 246, 235, 51, 149, 16, 117, 133, 47, 105, 146, 160, 25, 163, 35, 90, 25, 27, 95, 120,
    217, 123, 76, 168, 27, 115, 146, 19, 44, 56, 203, 248, 213, 96, 43, 80, 225, 122, 240, 142, 155, 118, 31, 180,
    57, 117, 196, 32, 208, 217, 226, 117, 133, 192, 169, 34, 72, 232, 208, 25, 223, 164, 208, 34, 76, 229, 185, 192,
    239, 42, 243, 18, 52, 55, 153, 121, 138, 177, 94, 94, 246, 18, 195, 173, 117, 106, 65, 23, 70, 26, 209, 162,
    13, 10, 24, 86, 95, 235, 26, 236, 252, 52, 214, 122, 204, 182, 237, 0, 167, 233, 208, 184, 247, 220, 58, 96,
    197, 42, 144, 183, 59, 249, 214, 35, 115, 85, 148, 107, 110, 105, 254, 250, 112, 149, 198, 93, 19, 4, 104, 121,
    182, 173, 177, 90, 131, 119, 39, 242, 182, 141, 183, 218, 165, 63, 75, 83, 137, 3, 147, 99, 147, 224, 145, 117,
    77, 69, 31, 51, 80, 204, 16, 102, 86, 157, 81, 113, 80, 222, 142, 179, 136, 0, 226, 64, 90, 99, 210, 173,
    223, 80, 191, 187, 104, 82, 113, 197, 91, 112, 107, 228, 236, 188, 245, 15, 212, 189, 97, 246, 20, 231, 158, 195,
    80, 12, 218, 6, 212, 228, 240, 205, 178, 14, 153, 127, 111, 104, 170, 68, 140, 67, 52, 188, 48, 228, 114, 2,
    238, 24, 88, 226, 157, 44, 113, 77, 214, 187, 239, 125, 226, 25, 167, 135, 178, 176, 211, 158, 42, 42, 180, 88,
    63, 89, 230, 91, 206, 172, 233, 30, 90, 200, 22, 234, 3, 219, 154, 203, 33, 178, 146, 5, 19, 84, 181, 158,
    73, 147, 134, 78, 214, 198, 134, 213, 79, 75, 53, 138, 59, 168, 204, 120, 88, 8, 110, 95, 74, 11, 3, 79,
    129, 187, 27, 136, 188, 240, 211, 84, 165, 7, 227, 140, 107, 8, 204, 57, 15, 142, 53, 19, 38, 231, 220, 95,
    25, 15, 81, 239, 54, 91, 178, 165, 35, 83, 217, 15, 97, 222, 83, 43, 67, 29, 162, 45, 218, 56, 127, 98,
    233, 131, 169, 247, 71, 94, 123, 0, 248, 111, 251, 137, 220, 162, 93, 149, 80, 243, 171, 223, 177, 22, 152, 190,
    183, 182, 100, 88, 106, 87, 226, 195, 0, 243, 80, 109, 67, 43, 137, 109, 54, 177, 143, 237, 13, 57, 99, 11,
    225, 10, 197, 151, 28, 224, 234, 151, 63, 94, 151, 109, 154, 100, 171, 148, 247, 221, 232, 47, 30, 186, 194, 50,
    103, 1, 105, 241, 64, 237, 113, 166, 159, 3, 79, 220, 42, 78, 216, 28, 72, 64, 15, 75, 169, 87, 141, 206,
    137, 249, 177, 175, 112, 108, 74, 200, 145, 183, 126, 181, 51, 53, 73, 136, 70, 175, 241, 218, 127, 78, 30, 66,
    41, 173, 128, 226, 70, 53, 7, 125, 251, 122, 237, 248, 12, 86, 202, 239, 26, 230, 60, 146, 76, 62, 171, 23,
    158, 31, 149, 222, 98, 145, 119, 91, 208, 123, 202, 229, 169, 57, 52, 198, 206, 122, 17, 229, 123, 214, 68, 84,
    98, 190, 206, 183, 179, 82, 213, 157, 16, 130, 177, 174, 180, 119, 81, 105, 154, 196, 156, 1, 148, 120, 158, 147,
    122, 80, 58, 217, 148, 220, 139, 235, 37, 214, 241, 36, 254, 8, 92, 152, 57, 216, 72, 159, 23, 231, 42, 188,
    251, 140, 188, 130, 197, 40, 70, 145, 96, 156, 86, 81, 228, 64, 37, 237, 100, 244, 188, 16, 130, 217, 75, 191,
    175, 171, 13, 69, 140, 217, 114, 5, 196, 97, 153, 119, 121, 104, 151, 187, 246, 226, 19, 133, 255, 184, 33, 214,
    52, 236, 232, 80, 96, 105, 221, 106, 39, 2, 109, 101, 51, 25, 54, 243, 29, 22, 137, 111, 25, 125, 5, 156,
    67, 157, 83, 205, 54, 34, 227, 197, 83, 83, 251, 39, 38, 56, 230, 18, 201, 163, 91, 205, 214, 99, 90, 151,
    59, 109, 123, 64, 48, 193, 123, 137, 156, 20, 255, 100, 158, 135, 240, 54, 233, 107, 86, 23, 111, 114, 85, 107,
    117, 198, 22, 158, 206, 88, 38, 85, 127, 28, 20, 177, 90, 144, 152, 46, 199, 118, 151, 28, 127, 206, 64, 183,
    4, 125, 113, 162, 37, 93, 125, 233, 94, 194, 232, 206, 112, 226, 113, 234, 94, 215, 16, 215, 79, 168, 29, 195,
    78, 36, 44, 142, 70, 89, 5, 21, 95, 113, 178, 144, 215, 241, 91, 63, 134, 193, 189, 131, 155, 32, 140, 30,
    76, 173, 248, 5, 4, 104, 141, 252, 203, 51, 154, 75, 24, 194, 251, 254, 84, 101, 215, 155, 64, 114, 94, 198,
    27, 63, 69, 10, 25, 221, 110, 193, 104, 110, 219, 29, 219, 112, 37, 64, 61, 67, 161, 63, 55, 66, 93, 246,
    3, 210, 116, 54, 91, 58, 233, 14, 166, 44, 51, 168, 146, 52, 114, 29, 148, 112, 227, 152, 158, 245, 165, 219,
    214, 199, 67, 155, 240, 32, 63, 161, 148, 49, 170, 79, 47, 56, 149, 176, 224, 53, 122, 179, 154, 158, 115, 21,
    99, 245, 95, 104, 234, 172, 42, 114, 247, 239, 249, 174, 225, 163, 239, 131, 232, 248, 52, 118, 149, 238, 216, 72,
    21, 113, 2, 219, 86, 125, 16, 18, 34, 195, 104, 92, 32, 123, 196, 33, 218, 230, 192, 169, 230, 205, 53, 146,
    96, 93, 245, 92, 164, 91, 0, 98, 239, 58, 169, 53, 199, 94, 116, 113, 96, 131, 192, 59, 13, 17, 230, 159,
    50, 138, 156, 46, 31, 175, 229, 52, 159, 20, 246, 80, 40, 3, 152, 35, 120, 119, 35, 55, 159, 45, 155, 233,
    195, 199, 91, 123, 47, 31, 1, 125, 31, 193, 3, 141, 144, 199, 207, 206, 110, 158, 14, 42, 49, 193, 241, 99,
    100, 202, 176, 189, 177, 111, 45, 186, 134, 15, 165, 224, 216, 170, 108, 20, 157, 73, 107, 28, 25, 27, 144, 228,
    94, 250, 17, 126, 45, 210, 91, 38, 37, 152, 107, 10, 227, 104, 122, 116, 140, 9, 223, 236, 243, 139, 153, 63,
    87, 210, 21, 77, 19, 183, 135, 247, 188, 75, 142, 141, 113, 116, 101, 172, 151, 173, 229, 183, 96, 35, 185, 84,
    87, 221, 43, 1, 182, 246, 125, 170, 69, 241, 190, 37, 250, 150, 11, 93, 79, 151, 73, 54, 173, 210, 98, 178,
    36, 221, 7, 152, 208, 213, 161, 14, 226, 176, 217, 132, 94, 54, 10, 10, 190, 149, 177, 181, 143, 35, 18, 5,
    77, 97, 51, 28, 0, 83, 0, 0, 0, 253, 255, 1, 255, 1, 1, 248, 15, 243, 15, 1, 255, 1, 243, 1,
    15, 3, 255, 1, 243, 1, 15, 255, 1, 255, 1, 16, 248, 15, 0, 1, 243, 1, 15, 255, 1, 255, 1, 16,
    248, 15, 2, 243, 15, 16, 255, 1, 243, 1, 15, 255, 1, 255, 1, 1, 248, 31, 243, 15, 16, 255, 1, 243,
    1, 2, 15, 240, 79, 1, 255, 1, 1, 248, 31, 243, 15, 16, 255, 1, 243, 1, 0, 15, 0, 61, 0, 0,
    0, 8, 42, 16, 42, 11, 42, 9, 42, 9, 42, 16, 42, 2, 42, 16, 42, 9, 42, 16, 42, 11, 42, 0,
    42, 9, 42, 16, 42, 11, 42, 3, 42, 9, 42, 16, 42, 9, 42, 16, 42, 11, 42, 8, 42, 9, 42, 16,
    42, 3, 34, 16, 42, 11, 42, 8, 42, 9, 42, 16, 42, 5, 0, 83, 0, 0, 0, 5, 7, 7, 8, 7,
    8, 7, 9, 8, 7, 9, 7, 9, 8, 7, 10, 5, 7, 10, 8, 7, 10, 7, 10, 8, 7, 8, 7, 11,
    7, 5, 8, 7, 11, 7, 11, 8, 7, 8, 7, 11, 9, 8, 7, 8, 7, 11, 8, 7, 11, 7, 9, 8,
    7, 11, 7, 8, 8, 7, 8, 7, 11, 8, 7, 10, 7, 12, 7, 11, 8, 7, 11, 7, 8, 8, 7, 8,
    7, 11, 8, 7, 11, 10, 199, 205, 56, 238, 89, 172, 132, 56, 79, 78, 63, 84, 252, 56, 46, 89, 168, 144,
    197, 5, 198, 113, 230, 66, 4, 238, 66, 69, 37, 28, 66, 56, 206, 92, 164, 118, 24, 226, 204, 220, 140, 161,
    130, 67, 238, 16, 199, 205, 88, 60, 67, 156, 153, 155, 49, 84, 76, 157, 49, 192, 205, 136, 227, 102, 44, 158,
    33, 206, 204, 205, 24, 42, 108, 178, 2,
};
static void v1_input(uint8_t *s, size_t n) { uint32_t x = 12345; for (size_t i = 0; i < n; i++) { x = x * 1103515245u + 12345u; s[i] = i % 97 < 40 ? (uint8_t)"entropy v1 test vector "[i % 23] : (uint8_t)((x >> 24) & 0x3F); } }
/* entropy blocks written by zap 1.2 (version 1: one repeat offset, no low-nibble stream) must keep decoding */
static void v1_compat_test(void) {
    uint8_t s[3000], o[3000];
    v1_input(s, sizeof s);
    assert(zap_decompress_entropy(kV1, sizeof kV1, o, sizeof o, NULL, NULL, 0) == (ptrdiff_t)sizeof o && !memcmp(s, o, sizeof o));
}

/* contextual Huffman (stream method 2): data whose next byte depends on the previous one's top bits makes the
   encoder pick it for the literals; round trip, then corrupt blocks must fail cleanly */
static void ctx_huffman_test(void) {
    enum { N = 1 << 18 };
    uint8_t *s = malloc(N), *c = malloc(2 * N + 1024), *o = malloc(N), *f = malloc(2 * N + 1024);
    uint32_t x = 777;
    for (size_t i = 0; i < N; i++) {
        x = x * 1103515245u + 12345u;
        uint8_t prev = i ? s[i - 1] : 0;
        s[i] = (uint8_t)(((prev >> 4) * 16 + ((x >> 24) & 7) * 3 + (prev & 1)) & 0xFF); /* depends on prev's top nibble */
        if (i % 1024 > 1000 && i > 4096) s[i] = s[i - 3000];                          /* a few matches */
    }
    int depths[2] = { 0, 64 };
    for (int k = 0; k < 2; k++) {
        size_t cn = zap_compress_entropy(s, N, c, 2 * N + 1024, depths[k] ? (void *)hc : (void *)&st, depths[k], NULL);
        assert(cn && (zap__r32(c) >> 31) && c[20] == 2); /* version 2 block, literal stream contextual */
        assert(zap_decompress_entropy(c, cn, o, N, NULL, NULL, 0) == N && !memcmp(s, o, N));
        for (int t = 0; t < 500; t++) {
            memcpy(f, c, cn);
            f[t < 250 ? (size_t)(20 + rand() % 3000) : (size_t)rand() % cn] ^= (uint8_t)(1 + rand() % 255); /* half of them in the tables */
            zap_decompress_entropy(f, cn, o, N, NULL, NULL, 0);
        }
    }
    free(s); free(c); free(o); free(f);
}

static void incompressible_test(void) {
    enum { N = 1 << 20 };
    uint8_t *src = malloc(N), *c = malloc(zap_bound(N) + 1024), *d = malloc(N);
    for (int i = 0; i < N; i++) src[i] = (uint8_t)(rand() >> 3);
    memcpy(src + N / 3, src, 4096); /* a little redundancy the fast path must still round-trip */
    int depths[3] = { 16, 64, 64 | ZAP_FAST_DECODE };
    for (int k = 0; k < 3; k++) {
        size_t n = zap_compress_hc(src, N, c, zap_bound(N), hc, NULL, depths[k]);
        assert(n && zap_decompress(c, n, d, N, NULL) == N && !memcmp(src, d, N));
        n = zap_compress_entropy(src, N, c, zap_bound(N) + 1024, hc, depths[k], NULL);
        assert(!n || (zap_decompress_entropy(c, n, d, N, NULL, NULL, 0) == N && !memcmp(src, d, N)));
    }
    free(src); free(c); free(d);
}

static void pak_test(void) {
    enum { N = 6 };
    static uint8_t a[70000], b[5000];
    for (size_t i = 0; i < sizeof a; i++) a[i] = (uint8_t)("zap pak test "[i % 13] ^ (i / 997));
    for (size_t i = 0; i < sizeof b; i++) b[i] = (uint8_t)rand();
    zap_pak_file f[N] = { { "z/last.bin", a, sizeof a }, { "a/first.txt", a, 1000 }, { "empty", a, 0 },
                          { "noise", b, sizeof b }, { "m/mid", a + 5, 20000 }, { "m/mid2", b, 17 } };
    size_t cap = zap_pak_bound(f, N, 16384);
    uint8_t *p1 = malloc(cap), *p2 = malloc(cap), *out = malloc(sizeof a);
    int depths[3] = { 0, 16, 32 | ZAP_ENTROPY };
    for (int di = 0; di < 3; di++) {
        size_t n1 = zap_pak_write(f, N, p1, cap, 16384, depths[di], 1), n2 = zap_pak_write(f, N, p2, cap, 16384, depths[di], 4);
        assert(n1 && n1 == n2 && !memcmp(p1, p2, n1)); /* same bytes for any thread count */
        zap_pak k;
        assert(zap_pak_open(&k, p1, n1) == 0 && zap_pak_count(&k) == N);
        for (int i = 0; i < N; i++) {
            long e = zap_pak_find(&k, f[i].name);
            assert(e >= 0 && zap_pak_raw_size(&k, (size_t)e) == f[i].size);
            assert(zap_pak_read(&k, (size_t)e, out, sizeof a) == (ptrdiff_t)f[i].size && !memcmp(out, f[i].data, f[i].size));
        }
        size_t l0, l1;
        const char *n0 = zap_pak_name(&k, 0, &l0), *nl = zap_pak_name(&k, N - 1, &l1);
        assert(l0 == 11 && !memcmp(n0, "a/first.txt", 11) && l1 == 10 && !memcmp(nl, "z/last.bin", 10)); /* sorted */
        assert(zap_pak_find(&k, "m") < 0 && zap_pak_find(&k, "m/mid3") < 0 && zap_pak_find(&k, "") < 0);
        assert(zap_pak_open(&k, p1, n1 - 1) != 0 && zap_pak_write(f, N, p2, n1 - 1, 16384, depths[di], 1) == 0);
        for (int t = 0; t < 2000; t++) { /* corrupt archives must fail cleanly, never crash */
            memcpy(p2, p1, n1);
            p2[rand() % n1] ^= (uint8_t)(1 + rand() % 255);
            if (zap_pak_open(&k, p2, n1) == 0)
                for (size_t i = 0; i < zap_pak_count(&k); i++) zap_pak_read(&k, i, out, sizeof a);
        }
    }
    f[5].name = "noise"; /* duplicate names are rejected */
    assert(zap_pak_write(f, N, p1, cap, 16384, 0, 1) == 0);
    free(p1); free(p2); free(out);
}

static void selftest(void) {
    enum { N = 600000 };
    uint8_t *buf = malloc(N);
    for (int depth = 0; depth <= 32; depth += 32) {
        for (size_t n = 0; n < 300; n++) { for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(rand() % 3); roundtrip(buf, n, 0, depth); }
        memset(buf, 7, 100000); roundtrip(buf, 100000, 0, depth);             /* long run, off=1 */
        for (size_t i = 0; i < 100000; i++) buf[i] = (uint8_t)(i % 5 + i / 777);
        roundtrip(buf, 100000, 0, depth);
        for (size_t i = 0; i < N; i++) buf[i] = (uint8_t)rand();
        roundtrip(buf, 100000, 0, depth);                                      /* incompressible */
        memcpy(buf + 300000, buf, 300000);                                     /* repeat 300KB back: needs the long window */
        assert(roundtrip(buf, N, 0, depth) < N * 6 / 10);
        if (depth) roundtrip(buf, N, 0, depth | ZAP_FAST_DECODE);
        zap_dict_init(&dict, buf, 5000);                                       /* dict: packet == dict tail + noise */
        uint8_t pkt[600]; memcpy(pkt, buf + 4700, 300); memcpy(pkt + 300, buf + 100, 300); pkt[77] ^= 1;
        roundtrip(pkt, sizeof pkt, &dict, depth);
        memcpy(pkt, buf + 4990, 10); memmove(pkt + 10, pkt, 590);              /* dict match running into src */
        roundtrip(pkt, sizeof pkt, &dict, depth);
    }
    /* frames: mixed compressible / raw blocks, odd tail; version 1 and version 2 (entropy) */
    for (size_t i = 0; i < N; i++) buf[i] = i < N / 2 ? (uint8_t)(i / 100 + (i % 7 == 0) * (i >> 9)) : (uint8_t)rand();
    static const int depths[8] = { 0, 16, 32, 32 | ZAP_FAST_DECODE, ZAP_ENTROPY, 16 | ZAP_ENTROPY, 32 | ZAP_ENTROPY, 32 | ZAP_ENTROPY | ZAP_FAST_DECODE };
    for (int di = 0; di < 8; di++) {
        int depth = depths[di];
        size_t n = N - 123, cap = zap_frame_bound(n, 65536);
        uint8_t *c = malloc(cap), *o = malloc(n);
        size_t cn = zap_frame_compress(buf, n, c, cap, 65536, depth, 0);
        assert(cn && zap_frame_decode(c, cn, o, n, 0) == (ptrdiff_t)n && !memcmp(o, buf, n));
        memset(o, 0, n);
        assert(zap_frame_decode_mt(c, cn, o, n, 0, 4) == (ptrdiff_t)n && !memcmp(o, buf, n));
        assert(zap_frame_decode(c, cn, o, n - 1, 0) == -1);
        uint8_t *c2 = malloc(cap);
        for (int t = 1; t <= 8; t += 3) /* threaded compress must produce the identical frame */
            assert(zap_frame_compress_mt(buf, n, c2, cap, 65536, depth, 0, t) == cn && !memcmp(c, c2, cn));
        assert(zap_frame_compress(buf, n, c2, cap, 0, depth, 0) == 0 && zap_frame_compress_mt(buf, n, c2, cap, 0, depth, 0, 4) == 0);
        assert(zap_frame_compress_mt(buf, n, c2, cn - 1, 65536, depth, 0, 4) == 0); /* too-small cap fails cleanly */
        free(c2);
        if (depth & ZAP_ENTROPY) assert(zap__r32(c) == ZAP_FRAME_MAGIC2); else assert(zap__r32(c) == ZAP_FRAME_MAGIC);
        fuzz(c, cn, n, 0);
        free(c); free(o);
    }
    free(buf);
    v1_compat_test();
    ctx_huffman_test();
    incompressible_test();
    pak_test();
    printf("selftest ok\n");
}

/* fake game traffic: 8 message kinds with their own layout + strings, drifting values.
 * phased = kinds arrive in phases (like a real session: lobby, loadout, match, ...), else random mix. */
static size_t gen_packets(uint8_t *out, size_t count, size_t *sizes, unsigned seed, int phased) {
    uint8_t *p = out;
    srand(seed);
    for (size_t i = 0; i < count; i++) {
        uint8_t *s = p;
        int kind = phased ? (int)(i * 8 / count) : rand() % 8, ents = 2 + rand() % 6;
        *p++ = (uint8_t)(0xC0 + kind); *p++ = 0x01; *p++ = (uint8_t)i; *p++ = (uint8_t)ents;
        for (int e = 0; e < ents; e++) {
            uint32_t id = 1000 * kind + rand() % 40; float pos[3] = { (float)(id * 3 % 500) + rand() % 4, 12.5f * kind, (float)(id * 7 % 300) };
            uint16_t hp = (uint16_t)(rand() % 3 ? 100 : rand() % 100);
            memcpy(p, &id, 4); memcpy(p + 4, pos, 4 + 4 * (kind % 3)); p += 8 + 4 * (kind % 3);
            memcpy(p, &hp, 2); p += 2;
            *p++ = (uint8_t)(rand() % 4);
            if (rand() % 4 == 0) p += sprintf((char *)p, "%s_%s_%02u", (const char *[]){ "player", "npc", "projectile", "vehicle", "pickup", "door", "chat", "score" }[kind],
                                              (const char *[]){ "alpha", "bravo", "charlie", "delta", "echo" }[id % 5], id % 17);
        }
        sizes[i] = (size_t)(p - s);
    }
    return (size_t)(p - out);
}

static void bench_packets(int phased) {
    enum { COUNT = 20000 };
    uint8_t *train = malloc(COUNT * 512), *test = malloc(COUNT * 512), dbuf[16384], o[1024];
    size_t *sz = malloc(COUNT * sizeof *sz), tn = gen_packets(train, COUNT, sz, 1, phased), n = gen_packets(test, COUNT, sz, 2, 0);
    struct { const char *name; size_t len; const uint8_t *data; } cfg[3] = {
        { "no dict", 0, 0 }, { "last 16K of samples", 16384, train + tn - 16384 }, { "trained dict 16K", 0, dbuf } };
    cfg[2].len = zap_dict_train(train, tn, dbuf, sizeof dbuf);
    printf("\npackets (%s training traffic): %d, avg %zu B\n", phased ? "phased" : "uniform", COUNT, n / COUNT);
    uint8_t *all = malloc(COUNT * 1024); size_t *cs = malloc(COUNT * sizeof *cs);
    for (int k = 0; k < 3; k++) {
        zap_dict_init(&dict, cfg[k].data, cfg[k].len);
        const zap_dict *d = k ? &dict : 0;
        size_t total = 0;
        for (size_t i = 0, off = 0; i < COUNT; off += sz[i++]) {
            total += cs[i] = zap_compress(test + off, sz[i], all + i * 1024, 1024, &st, d);
            if (zap_decompress(all + i * 1024, cs[i], o, sz[i], d) != (ptrdiff_t)sz[i] || memcmp(o, test + off, sz[i])) { printf("FAIL\n"); exit(1); }
        }
        double t0 = now();
        for (int r = 0; r < 20; r++) for (size_t i = 0; i < COUNT; i++) zap_decompress(all + i * 1024, cs[i], o, sz[i], d);
        double td = (now() - t0) / 20;
        printf("  %-20s ratio %5.2f  decode %5.1f M packets/s\n", cfg[k].name, (double)n / total, COUNT / td / 1e6);
    }
    free(all); free(cs); free(train); free(test); free(sz);
}

static void bench_file(const uint8_t *src, size_t n, int threads) {
    struct { const char *name; size_t bs; int depth; } cfg[] = {
        { "fast, 256KB blocks", 256 << 10, 0 }, { "fast, 4MB blocks", 4 << 20, 0 },
        { "hc16, 4MB blocks", 4 << 20, 16 }, { "hc64, 4MB blocks", 4 << 20, 64 },
        { "fast+entropy, 4MB", 4 << 20, ZAP_ENTROPY }, { "hc16+entropy, 4MB", 4 << 20, 16 | ZAP_ENTROPY },
        { "hc64+entropy, 4MB", 4 << 20, 64 | ZAP_ENTROPY } };
    printf("\nfile: %.1f MB\n", n / 1e6);
    for (int k = 0; k < (int)(sizeof cfg / sizeof cfg[0]); k++) {
        size_t cap = zap_frame_bound(n, cfg[k].bs);
        uint8_t *c = malloc(cap), *o = malloc(n);
        double t0 = now();
        size_t cn = zap_frame_compress(src, n, c, cap, cfg[k].bs, cfg[k].depth, 0);
        double tc = now() - t0, t1 = 0, tn = 0;
        uint8_t *c2 = malloc(cap);
        t0 = now();
        size_t cn2 = zap_frame_compress_mt(src, n, c2, cap, cfg[k].bs, cfg[k].depth, 0, threads);
        double tcm = now() - t0;
        if (cn2 != cn || memcmp(c, c2, cn)) { printf("MT COMPRESS MISMATCH\n"); exit(1); }
        free(c2);
        for (int mt = 0; mt < 2; mt++) {
            int reps = 0; t0 = now();
            do {
                ptrdiff_t r = mt ? zap_frame_decode_mt(c, cn, o, n, 0, threads) : zap_frame_decode(c, cn, o, n, 0);
                if (r != (ptrdiff_t)n) { printf("FAIL\n"); exit(1); }
                reps++;
            } while (now() - t0 < 1.0);
            *(mt ? &tn : &t1) = (now() - t0) / reps;
        }
        if (memcmp(o, src, n)) { printf("MISMATCH\n"); exit(1); }
        printf("  %-18s ratio %5.3f | comp 1T %5.0f  %dT %5.0f MB/s | decode 1T %5.0f  %dT %5.0f MB/s\n", cfg[k].name,
               (double)n / cn, n / tc / 1e6, threads, n / tcm / 1e6, n / t1 / 1e6, threads, n / tn / 1e6);
        free(c); free(o);
    }
}

int main(int argc, char **argv) {
    hc = malloc(sizeof *hc);
    selftest();
    bench_packets(0);
    bench_packets(1);
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *src = malloc(n); if (fread(src, 1, n, f) != n) return 1; fclose(f);
    bench_file(src, n, argc > 2 ? atoi(argv[2]) : 8);
    return 0;
}
