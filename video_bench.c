/* video_bench.c - video self-test and benchmark.
 *   video_bench                          self-test (bit-exactness encoder vs decoder, fuzz)
 *   video_bench in.yuv w h [frames]      benchmark on raw I420 (ffmpeg -i x.mp4 -pix_fmt yuv420p -f rawvideo in.yuv);
 *                                        also encodes with 8 threads and checks the output is identical
 */
#define ZAP_THREADS
#include "zap_video.h"
#undef NDEBUG /* the self-tests are asserts: keep them in release builds */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + t.tv_nsec * 1e-9; }

static double sse(const uint8_t *a, int as, const uint8_t *b, int bs, int w, int h) {
    double s = 0;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { double d = a[y * as + x] - b[y * bs + x]; s += d * d; }
    return s;
}
static double to_psnr(double se, double n) { return se == 0 ? 99 : 10 * log10(255.0 * 255.0 * n / se); }

/* compare decoder output with encoder reconstruction (must be identical) */
static void same(const zap_video *e, const zap_video *d, int w, int h) {
    const uint8_t *p[2][3]; int ys[2], cs[2];
    zap_video_planes(e, &p[0][0], &p[0][1], &p[0][2], &ys[0], &cs[0]);
    zap_video_planes(d, &p[1][0], &p[1][1], &p[1][2], &ys[1], &cs[1]);
    assert(sse(p[0][0], ys[0], p[1][0], ys[1], w, h) == 0);
    assert(sse(p[0][1], cs[0], p[1][1], cs[1], w / 2, h / 2) == 0 && sse(p[0][2], cs[0], p[1][2], cs[1], w / 2, h / 2) == 0);
}

static int geo(void) { int z = 0; unsigned r = (unsigned)rand() | 0x10000u; while (!(r & 1)) { r >>= 1; z++; } return z; }

static void huff_test(void) {
    enum { N = 20000 };
    static uint8_t src[N], enc[N + 256], dec[N];
    for (int kind = 0; kind < 4; kind++) {
        for (int i = 0; i < N; i++) /* skewed, single symbol, uniform, very skewed (forces the length limit) */
            src[i] = (uint8_t)(kind == 0 ? rand() % (1 + rand() % 40) : kind == 1 ? 7 : kind == 2 ? rand() : (rand() % 3 ? 0 : 1 + geo() * 3 + rand() % 3));
        size_t n = zap__henc(src, N, enc, sizeof enc);
        assert(n && zap__hdec(enc, n, dec, N) == 0 && !memcmp(src, dec, N));
        assert(zap__hdec(enc, n - 1 - (n > 200), dec, N) == -1 || kind == 1); /* truncated */
        for (int i = 0; i < 200; i++) { /* garbage must fail or succeed, never crash */
            uint8_t g[300];
            for (int k = 0; k < 300; k++) g[k] = (uint8_t)rand();
            if (i & 1) memcpy(g, enc, 128);
            zap__hdec(g, sizeof g, dec, N);
        }
    }
}

/* the SIMD kernels must be bit-exact with the scalar reference (else encoder and decoder drift apart) */
static void simd_test(void) {
    static uint8_t ref[64 * 64], pred[64], a[256], b[256];
    for (int i = 0; i < 64 * 64; i++) ref[i] = (uint8_t)rand();
    for (int t = 0; t < 20000; t++) {
        int16_t q[64];
        uint8_t step[64];
        int dense = rand() % 4;
        for (int i = 0; i < 64; i++) {
            q[i] = (int16_t)(rand() % 3 == 0 || !dense ? 0 : dense == 3 ? (int16_t)(rand() % 65536 - 32768) : rand() % 64 - 32);
            step[i] = (uint8_t)(1 + rand() % 255);
        }
        if (t % 7 == 0) q[0] = (int16_t)(rand() % 2000 - 1000);
        for (int i = 0; i < 64; i++) pred[i] = (uint8_t)rand();
        const uint8_t *pp = t & 1 ? pred : NULL;
        zap__vidct_c(q, step, pp, 8, a, 8);
        zap__vidct(q, step, pp, 8, b, 8);
        assert(!memcmp(a, b, 64));
    }
    for (int t = 0; t < 2000; t++) {
        int n = t & 1 ? 16 : 8, mvx = rand() % 9 - 4, mvy = rand() % 9 - 4;
        zap__vpred_c(ref, 64, 20, 20, mvx, mvy, n, a);
        zap__vpred(ref, 64, 20, 20, mvx, mvy, n, b);
        assert(!memcmp(a, b, (size_t)(n * n)));
    }
}

static void selftest(void) {
    huff_test();
    simd_test();
    enum { W = 72, H = 40, F = 14 };
    static uint8_t yb[F][W * H], ub[F][W * H / 4], vb[F][W * H / 4];
    srand(5);
    for (int f = 0; f < F; f++) {
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int sq = x >= 10 + 3 * f && x < 30 + 3 * f && y >= 8 + f && y < 24 + f;
            yb[f][y * W + x] = (uint8_t)(sq ? 230 : (x * 3 + y * 2 + f * 4) + rand() % 6);
        }
        for (int i = 0; i < W * H / 4; i++) { ub[f][i] = (uint8_t)(100 + i % 7 + f); vb[f][i] = (uint8_t)(150 - (i / 36) * 2); }
    }
    for (int cfg = 0; cfg < 4; cfg++) {
        int quality = cfg & 1 ? 90 : 40, depth = cfg & 2 ? 8 : 0;
        zap_video *e = zap_venc_create(W, H, quality, 5, depth), *d = zap_vdec_create(W, H), *fz = zap_vdec_create(W, H);
        zap_video *et = zap_venc_create(W, H, quality, 5, depth); /* threaded: must produce the same bytes */
        assert(e && d && fz && et);
        zap_venc_threads(et, 1 + cfg);
        size_t cap = zap_video_bound(e);
        uint8_t *pkt = malloc(cap), *bad = malloc(cap), *pkt2 = malloc(cap);
        double se = 0;
        for (int f = 0; f < F; f++) {
            /* a too-small output buffer fails without advancing the encoder */
            assert(zap_venc_frame(e, yb[f], ub[f], vb[f], W, W / 2, pkt, 10) == 0);
            assert(zap_venc_frame(e, yb[f], ub[f], vb[f], W, W / 2, pkt, ZAP__VHDR + 8) == 0);
            size_t n = zap_venc_frame(e, yb[f], ub[f], vb[f], W, W / 2, pkt, cap);
            assert(n > 0);
            assert(zap_venc_frame(et, yb[f], ub[f], vb[f], W, W / 2, pkt2, cap) == n && !memcmp(pkt, pkt2, n));
            assert(zap_vdec_frame(d, pkt, n) == 0);
            same(e, d, W, H);
            const uint8_t *py, *pu, *pv; int ys, cs;
            zap_video_planes(d, &py, &pu, &pv, &ys, &cs);
            se += sse(py, ys, yb[f], W, W, H);
            for (int i = 0; i < 20; i++) { /* corrupt packets must fail cleanly */
                memcpy(bad, pkt, n);
                bad[rand() % n] ^= (uint8_t)(1 + rand() % 255);
                zap_vdec_frame(fz, bad, i & 1 ? n : (size_t)rand() % n);
            }
            /* after garbage the fuzzed decoder must resync exactly at each I-frame
               (a corrupt packet can still be well-formed, so P-frames in between may differ) */
            int r = zap_vdec_frame(fz, pkt, n);
            if (f % 5 == 0) { assert(r == 0); same(e, fz, W, H); }
        }
        assert(to_psnr(se, (double)W * H * F) > (quality > 50 ? 36 : 28));
        free(pkt); free(bad); free(pkt2);
        zap_video_destroy(et);
        zap_video_destroy(e); zap_video_destroy(d); zap_video_destroy(fz);
    }
    assert(!zap_vdec_create(3, 4) && !zap_venc_create(0, 2, 50, 1, 0)); /* odd / empty sizes rejected */
    printf("video selftest ok\n");
}

/* --psnr a.yuv b.yuv w h: compare two raw I420 files frame by frame (for measuring other codecs the same way) */
static int psnr_files(const char *a, const char *b, int w, int h) {
    size_t fsz = (size_t)w * h * 3 / 2, nf = 0;
    uint8_t *fa = malloc(fsz), *fb = malloc(fsz);
    FILE *A = fopen(a, "rb"), *B = fopen(b, "rb");
    if (!A || !B) { perror("open"); return 1; }
    double sy = 0, sa = 0;
    while (fread(fa, 1, fsz, A) == fsz && fread(fb, 1, fsz, B) == fsz) {
        double y = sse(fa, w, fb, w, w, h);
        sy += y; sa += y + sse(fa + (size_t)w * h, w / 2, fb + (size_t)w * h, w / 2, w / 2, h);
        nf++;
    }
    printf("%zu frames  PSNR-Y %.2f  PSNR-YUV %.2f\n", nf, to_psnr(sy, (double)w * h * nf), to_psnr(sa, (double)w * h * nf * 1.5));
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 6 && !strcmp(argv[1], "--psnr")) return psnr_files(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
    selftest();
    if (argc < 4) return 0;
    int w = atoi(argv[2]), h = atoi(argv[3]), maxf = argc > 4 ? atoi(argv[4]) : 1 << 30;
    size_t fsz = (size_t)w * h * 3 / 2;
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }
    uint8_t *all = NULL; int nf = 0;
    for (;;) {
        if (nf >= maxf) break;
        all = realloc(all, fsz * (size_t)(nf + 1));
        if (fread(all + fsz * (size_t)nf, 1, fsz, fi) != fsz) break;
        nf++;
    }
    fclose(fi);
    printf("\n%s  %dx%d, %d frames (bitrate assumes 30 fps)\n", argv[1], w, h, nf);
    printf("  quality   kbit/s   PSNR-Y  PSNR-YUV   encode fps   8-thread enc fps   decode fps (1 thread)\n");
    int qs[] = { 30, 50, 70, 85, 95 };
    for (int qi = 0; qi < 5; qi++) {
        zap_video *e = zap_venc_create(w, h, qs[qi], 60, 16);
        size_t cap = zap_video_bound(e), total = 0;
        uint8_t *pk = malloc(cap * (size_t)nf / 4 + cap), *p = pk;
        size_t *len = malloc(sizeof *len * (size_t)nf);
        double t0 = now();
        for (int f = 0; f < nf; f++) {
            const uint8_t *y = all + fsz * (size_t)f, *u = y + (size_t)w * h, *vv = u + (size_t)w * h / 4;
            len[f] = zap_venc_frame(e, y, u, vv, w, w / 2, p, cap);
            if (!len[f]) { printf("encode failed\n"); return 1; }
            p += len[f]; total += len[f];
        }
        double te = now() - t0;
        zap_video *et = zap_venc_create(w, h, qs[qi], 60, 16);
        zap_venc_threads(et, 8);
        uint8_t *tp = malloc(cap);
        t0 = now();
        p = pk;
        for (int f = 0; f < nf; f++) {
            const uint8_t *y = all + fsz * (size_t)f, *u = y + (size_t)w * h, *vv = u + (size_t)w * h / 4;
            size_t n = zap_venc_frame(et, y, u, vv, w, w / 2, tp, cap);
            if (n != len[f] || memcmp(tp, p, n)) { printf("threaded encode differs at frame %d\n", f); return 1; }
            p += len[f];
        }
        double tt = now() - t0;
        zap_video_destroy(et); free(tp);
        zap_video *d = zap_vdec_create(w, h);
        double sy = 0, suv = 0;
        p = pk; /* pass 1: quality */
        for (int f = 0; f < nf; f++) {
            if (zap_vdec_frame(d, p, len[f])) { printf("decode failed\n"); return 1; }
            p += len[f];
            const uint8_t *y = all + fsz * (size_t)f, *u = y + (size_t)w * h, *vv = u + (size_t)w * h / 4, *py, *pu, *pv;
            int ys, cs;
            zap_video_planes(d, &py, &pu, &pv, &ys, &cs);
            sy += sse(py, ys, y, w, w, h);
            suv += sse(pu, cs, u, w / 2, w / 2, h / 2) + sse(pv, cs, vv, w / 2, w / 2, h / 2);
        }
        t0 = now(); /* passes 2-4: speed only */
        for (int rep = 0; rep < 3; rep++) {
            p = pk;
            for (int f = 0; f < nf; f++) { zap_vdec_frame(d, p, len[f]); p += len[f]; }
        }
        double td = now() - t0;
        printf("  %5d  %9.0f   %6.2f    %6.2f   %8.1f   %12.1f       %8.1f\n", qs[qi], total * 8.0 / nf * 30 / 1000,
               to_psnr(sy, (double)w * h * nf), to_psnr(sy + suv, (double)w * h * nf * 1.5), nf / te, nf / tt, nf * 3 / td);
        zap_video_destroy(e); zap_video_destroy(d); free(pk); free(len);
    }
    return 0;
}
