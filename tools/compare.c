/* compare.c - zap vs LZ4 vs zstd on one file: same 4MB blocks, one thread, best of several runs.
 *
 * Needs LZ4 (lz4.h, lz4hc.h + library) and zstd (zstd.h + library or sources). Example (Windows, clang):
 *   clang -O3 -march=native -I. -Ipath/to/lz4/include -Ipath/to/zstd tools/compare.c <zstd .c files> path/to/lz4.lib -o compare.exe
 * Linux:  cc -O3 -march=native -I. tools/compare.c -llz4 -lzstd -o compare
 *   compare <file> [block_kb=4096]
 */
#include "zap.h"
#include <lz4.h>
#include <lz4hc.h>
#include <zstd.h>
#include <stdio.h>
#include <time.h>

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + t.tv_nsec * 1e-9; }

enum { ZAP_FAST, ZAP_HC, LZ4_FAST, LZ4_HC, ZSTD };
typedef struct { const char *name; int kind, level; } codec;

static const uint8_t *g_src;
static size_t g_n, g_bs, g_nb;
static uint8_t **g_c, *g_out;
static size_t *g_cs;
static void *g_state;
static void *g_scratch;

static size_t blk(size_t b) { return b == g_nb - 1 ? g_n - b * g_bs : g_bs; }

static void compress_all(const codec *c) {
    for (size_t b = 0; b < g_nb; b++) {
        const uint8_t *s = g_src + b * g_bs;
        size_t len = blk(b), cap = zap_bound(len) + 1024, r = 0;
        uint8_t *d = g_c[b];
        int depth = c->level & 0xFFFF, e = (c->level & ZAP_ENTROPY) != 0;
        switch (c->kind) {
        case ZAP_FAST: case ZAP_HC:
            r = e ? zap_compress_entropy(s, len, d, cap, g_state, depth, NULL)
                  : depth ? zap_compress_hc(s, len, d, cap, (zap_hc_state *)g_state, NULL, depth) : zap_compress(s, len, d, cap, (zap_state *)g_state, NULL);
            break;
        case LZ4_FAST: r = (size_t)LZ4_compress_default((const char *)s, (char *)d, (int)len, (int)cap); break;
        case LZ4_HC: r = (size_t)LZ4_compress_HC((const char *)s, (char *)d, (int)len, (int)cap, c->level); break;
        case ZSTD: r = ZSTD_compress(d, cap, s, len, c->level); break;
        }
        g_cs[b] = r;
    }
}

static int decompress_all(const codec *c) {
    for (size_t b = 0; b < g_nb; b++) {
        size_t len = blk(b);
        uint8_t *o = g_out + b * g_bs;
        long long r = -1;
        switch (c->kind) {
        case ZAP_FAST: case ZAP_HC:
            r = (c->level & ZAP_ENTROPY) ? zap_decompress_entropy(g_c[b], g_cs[b], o, len, NULL, g_scratch, zap_entropy_scratch(g_bs))
                                         : zap_decompress(g_c[b], g_cs[b], o, len, NULL);
            break;
        case LZ4_FAST: case LZ4_HC: r = LZ4_decompress_safe((const char *)g_c[b], (char *)o, (int)g_cs[b], (int)len); break;
        case ZSTD: r = (long long)ZSTD_decompress(o, len, g_c[b], g_cs[b]); break;
        }
        if (r != (long long)len) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "compare <file> [block_kb]\n"); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); g_n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *src = malloc(g_n);
    if (fread(src, 1, g_n, f) != g_n) return 1;
    fclose(f);
    g_src = src;
    g_bs = (argc > 2 ? (size_t)atoi(argv[2]) : 4096) << 10;
    g_nb = (g_n + g_bs - 1) / g_bs;
    g_c = malloc(g_nb * sizeof *g_c); g_cs = malloc(g_nb * sizeof *g_cs); g_out = malloc(g_n);
    for (size_t b = 0; b < g_nb; b++) g_c[b] = malloc(zap_bound(g_bs) + 1024);
    g_state = malloc(sizeof(zap_hc_state));
    g_scratch = malloc(zap_entropy_scratch(g_bs));
    const codec codecs[] = {
        { "lz4 1.9.4", LZ4_FAST, 0 }, { "lz4hc 9", LZ4_HC, 9 }, { "lz4hc 12", LZ4_HC, 12 },
        { "zap fast", ZAP_FAST, 0 }, { "zap hc16", ZAP_HC, 16 }, { "zap hc64", ZAP_HC, 64 },
        { "zstd 1", ZSTD, 1 }, { "zstd 3", ZSTD, 3 }, { "zstd 9", ZSTD, 9 }, { "zstd 19", ZSTD, 19 },
        { "zap fast+entropy", ZAP_FAST, ZAP_ENTROPY }, { "zap hc16+entropy", ZAP_HC, 16 | ZAP_ENTROPY }, { "zap hc64+entropy", ZAP_HC, 64 | ZAP_ENTROPY },
    };
    printf("%s: %.1f MB, %zu KB blocks, 1 thread, best of runs\n", argv[1], g_n / 1e6, g_bs >> 10);
    printf("  %-18s %6s  %10s  %12s\n", "codec", "ratio", "compress", "decompress");
    for (size_t k = 0; k < sizeof codecs / sizeof codecs[0]; k++) {
        const codec *c = &codecs[k];
        double tc = 1e30, td = 1e30;
        int reps = c->level >= 12 || (c->kind == ZAP_HC && !(c->level & ZAP_ENTROPY) && c->level > 16) ? 1 : 3;
        for (int r = 0; r < reps; r++) { double t0 = now(); compress_all(c); double t = now() - t0; if (t < tc) tc = t; }
        size_t total = 0;
        for (size_t b = 0; b < g_nb; b++) { if (!g_cs[b] || ZSTD_isError(g_cs[b])) { printf("  %s: compress failed\n", c->name); return 1; } total += g_cs[b]; }
        for (int r = 0; r < 7; r++) {
            double t0 = now();
            if (decompress_all(c)) { printf("  %s: decompress failed\n", c->name); return 1; }
            double t = now() - t0; if (t < td) td = t;
        }
        if (memcmp(g_out, src, g_n)) { printf("  %s: MISMATCH\n", c->name); return 1; }
        printf("  %-18s %6.3f  %6.0f MB/s  %8.0f MB/s\n", c->name, (double)g_n / total, g_n / tc / 1e6, g_n / td / 1e6);
    }
    return 0;
}
