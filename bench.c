/* bench.c - self-test, fuzz, and speed.  usage: bench [file] [threads] */
#define ZAP_THREADS
#include "zap.h"
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
        zap_dict_init(&dict, buf, 5000);                                       /* dict: packet == dict tail + noise */
        uint8_t pkt[600]; memcpy(pkt, buf + 4700, 300); memcpy(pkt + 300, buf + 100, 300); pkt[77] ^= 1;
        roundtrip(pkt, sizeof pkt, &dict, depth);
        memcpy(pkt, buf + 4990, 10); memmove(pkt + 10, pkt, 590);              /* dict match running into src */
        roundtrip(pkt, sizeof pkt, &dict, depth);
    }
    /* frames: mixed compressible / raw blocks, odd tail; version 1 and version 2 (entropy) */
    for (size_t i = 0; i < N; i++) buf[i] = i < N / 2 ? (uint8_t)(i / 100 + (i % 7 == 0) * (i >> 9)) : (uint8_t)rand();
    static const int depths[4] = { 0, 16, ZAP_ENTROPY, 16 | ZAP_ENTROPY };
    for (int di = 0; di < 4; di++) {
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
