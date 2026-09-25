/* tex_bench.c - texture self-test and benchmark.
 *   tex_bench                               self-test on a synthetic image
 *   tex_bench img.rgba w h                  benchmark (raw RGBA8, e.g. ffmpeg -i x.jpg -pix_fmt rgba -f rawvideo x.rgba)
 *   tex_bench img.rgba w h dds_prefix       also write <prefix>_bc1.dds etc. for cross-checking with other decoders
 */
#include "zap.h"
#include "zap_tex.h"
#undef NDEBUG /* the self-tests are asserts: keep them in release builds */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *NAMES[] = { "BC1", "BC3", "BC4", "BC5", "BC7" };

static double psnr(const uint8_t *a, const uint8_t *b, size_t npx, int chmask) {
    double se = 0; size_t n = 0;
    for (size_t i = 0; i < npx; i++)
        for (int c = 0; c < 4; c++) if (chmask >> c & 1) { double d = a[i * 4 + c] - b[i * 4 + c]; se += d * d; n++; }
    return se == 0 ? 99.0 : 10 * log10(255.0 * 255.0 * n / se);
}
static int chmask(zap_bc_format f) { return f == ZAP_BC1 ? 7 : f == ZAP_BC4 ? 1 : f == ZAP_BC5 ? 3 : 15; }

static size_t zap_size(const uint8_t *p, size_t n) {
    static zap_hc_state hc;
    uint8_t *c = malloc(zap_bound(n));
    size_t cn = zap_compress_hc(p, n, c, zap_bound(n), &hc, NULL, 64);
    free(c);
    return cn;
}

static void write_dds(const char *path, const void *bc, size_t n, int w, int h, zap_bc_format f) {
    static const char *CC[] = { "DXT1", "DXT5", "BC4U", "BC5U", "DX10" }; /* DXGI channel order (ffmpeg reads ATI2 as G,R) */
    uint8_t hd[148] = { 'D', 'D', 'S', ' ' };
    uint32_t v[] = { 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000, (uint32_t)h, (uint32_t)w, (uint32_t)n };
    memcpy(hd + 4, v, sizeof v);
    uint32_t pf[] = { 32, 0x4 }; memcpy(hd + 76, pf, 8); memcpy(hd + 84, CC[f], 4);
    uint32_t caps = 0x1000; memcpy(hd + 108, &caps, 4);
    uint32_t dx10[5] = { 98 /* DXGI_FORMAT_BC7_UNORM */, 3 /* TEXTURE2D */, 0, 1, 0 };
    memcpy(hd + 128, dx10, sizeof dx10);
    FILE *o = fopen(path, "wb"); fwrite(hd, 1, f == ZAP_BC7 ? 148 : 128, o); fwrite(bc, 1, n, o); fclose(o);
}

static void hdr_mip_test(void) {
    /* half-float conversion: every unsigned half round-trips exactly; float -> half rounds to nearest */
    for (uint32_t h = 0; h <= 0x7BFF; h++) assert(zap__f2h(zap__h2f((uint16_t)h)) == h);
    assert(zap__f2h(-1.0f) == 0 && zap__f2h(1e9f) == 0x7BFF && zap__f2h(1.0f) == 0x3C00 && zap__f2h(1.0f + 1.0f / 4096) == 0x3C00);

    /* BC6H: HDR gradients over ~6 stops plus a bright spot; relative error per channel stays small */
    enum { W = 37, H = 21 };
    static float img[W * H * 4];
    static uint16_t dec[W * H * 4];
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float *p = img + (y * W + x) * 4;
        p[0] = 0.05f * (float)(1 << (x / 6)); p[1] = 0.5f + y * 0.25f; p[2] = (x == 20 && y == 10) ? 10000.0f : 0.01f * (x + 1); p[3] = 1;
    }
    uint8_t *bc = malloc(zap_bc_size(W, H, ZAP_BC7));
    zap_bc6h_encode(img, W, H, W * 4, bc);
    zap_bc6h_decode(bc, W, H, dec, W * 4);
    double rel = 0;
    for (int i = 0; i < W * H; i++)
        for (int c = 0; c < 3; c++) {
            double a = img[i * 4 + c], b = zap__h2f(dec[i * 4 + c]);
            rel += fabs(a - b) / (a + 0.01);
        }
    rel /= W * H * 3;
    assert(rel < 0.08 && dec[3] == 0x3C00);
    for (int s = 1; s <= 5; s += 4) { /* tiny / odd sizes */
        zap_bc6h_encode(img, s, s == 1 ? 1 : 3, W * 4, bc);
        zap_bc6h_decode(bc, s, s == 1 ? 1 : 3, dec, W * 4);
    }
    free(bc);

    /* BC7 mode 5 gets picked when alpha varies independently of color */
    static uint8_t rgba[64 * 64 * 4], out7[64 * 64 * 4];
    for (int i = 0; i < 64 * 64; i++) { rgba[i * 4] = (uint8_t)(i % 64 * 4); rgba[i * 4 + 1] = (uint8_t)(i / 64 * 4); rgba[i * 4 + 2] = 90; rgba[i * 4 + 3] = (uint8_t)((255 - i % 64 * 4) ^ (i / 64 * 4)); }
    uint8_t *b7 = malloc(zap_bc_size(64, 64, ZAP_BC7));
    zap_bc_encode(rgba, 64, 64, 64 * 4, ZAP_BC7, 0, b7);
    int m5 = 0;
    for (int b = 0; b < 256; b++) m5 += (b7[b * 16] & 0x3F) == 0x20;
    zap_bc_decode(b7, 64, 64, ZAP_BC7, out7, 64 * 4);
    assert(m5 > 0 && psnr(rgba, out7, 64 * 64, 15) > 30);
    free(b7);

    /* mipmaps */
    assert(zap_mip_levels(1, 1) == 1 && zap_mip_levels(256, 128) == 9 && zap_mip_levels(5, 3) == 3);
    uint8_t bw[2 * 2 * 4] = { 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255 }, m[4];
    zap_mip_next(bw, 2, 2, 8, 0, m, 4);
    assert(m[0] == 128 && m[3] == 255);                 /* plain average */
    zap_mip_next(bw, 2, 2, 8, 1, m, 4);
    assert(m[0] == 188 && m[1] == 188 && m[3] == 255);  /* linear-light average of black and white */
    for (int v = 0; v < 256; v++) { uint8_t c4[16], o[4]; memset(c4, v, 16); zap_mip_next(c4, 2, 2, 8, 1, o, 4); assert(o[0] == v); }
    uint8_t odd[5 * 3 * 4], m2[2 * 4];
    memset(odd, 77, sizeof odd);
    zap_mip_next(odd, 5, 3, 5 * 4, 1, m2, 2 * 4);
    assert(m2[0] == 77 && m2[7] == 77);
    float fl[2 * 2 * 4] = { 1, 2, 3, 4, 3, 2, 1, 0, 5, 5, 5, 5, 7, 7, 7, 7 }, fo[4];
    zap_mip_next_f(fl, 2, 2, 8, fo, 4);
    assert(fo[0] == 4 && fo[3] == 4);
}

static void selftest(void) {
    hdr_mip_test();
    enum { W = 67, H = 45 };
    static uint8_t img[W * H * 4], out[W * H * 4];
    srand(3);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        uint8_t *p = img + (y * W + x) * 4;
        p[0] = (uint8_t)(x * 255 / W); p[1] = (uint8_t)(y * 255 / H); p[2] = (uint8_t)((x + y) * 2 + rand() % 8);
        p[3] = (uint8_t)(x < 30 ? 255 : (x * y) & 255);
    }
    for (int f = 0; f < 5; f++) for (int r = 0; r <= 1; r++) {
        size_t n = zap_bc_size(W, H, (zap_bc_format)f);
        uint8_t *bc = malloc(n), *sp = malloc(n), *mg = malloc(n);
        zap_bc_encode(img, W, H, W * 4, (zap_bc_format)f, r ? 100.0f : 0.0f, bc);
        zap_bc_decode(bc, W, H, (zap_bc_format)f, out, W * 4);
        double q = psnr(img, out, W * H, chmask((zap_bc_format)f));
        static double bc3q;
        assert(q > (r ? 28 : 32));
        if (f == ZAP_BC3 && !r) bc3q = q;
        if (f == ZAP_BC7 && !r) assert(q > bc3q); /* same RGBA image: BC7 must beat BC3 */
        if (f == ZAP_BC1) for (int i = 0; i < W * H; i++) assert(out[i * 4 + 3] == 255); /* never transparent */
        zap_bc_split(bc, n, (zap_bc_format)f, sp);
        zap_bc_merge(sp, n, (zap_bc_format)f, mg);
        assert(!memcmp(bc, mg, n));
        free(bc); free(sp); free(mg);
    }
    /* solid block and 1x1 image */
    uint8_t one[4] = { 10, 200, 30, 255 }, o1[4], b1[16];
    for (int f = 0; f < 5; f++) {
        zap_bc_encode(one, 1, 1, 4, (zap_bc_format)f, 0, b1);
        zap_bc_decode(b1, 1, 1, (zap_bc_format)f, o1, 4);
        for (int c = 0; c < 4; c++) if (chmask((zap_bc_format)f) >> c & 1) assert(abs(o1[c] - one[c]) <= 4);
    }
    printf("tex selftest ok\n");
}

int main(int argc, char **argv) {
    selftest();
    if (argc < 4) return 0;
    int w = atoi(argv[2]), h = atoi(argv[3]);
    size_t npx = (size_t)w * (size_t)h;
    uint8_t *img = malloc(npx * 4), *out = malloc(npx * 4);
    FILE *fi = fopen(argv[1], "rb");
    if (!fi || fread(img, 4, npx, fi) != npx) { fprintf(stderr, "can't read %s\n", argv[1]); return 1; }
    fclose(fi);
    /* synthetic normal map from luminance for the BC5 test */
    uint8_t *nm = malloc(npx * 4);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
#define LUM(X, Y) (img[((size_t)(Y) * w + (X)) * 4 + 1])
        int x1 = x + 1 < w ? x + 1 : x, y1 = y + 1 < h ? y + 1 : y;
        float dx = (LUM(x1, y) - LUM(x, y)) / 64.0f, dy = (LUM(x, y1) - LUM(x, y)) / 64.0f, l = sqrtf(dx * dx + dy * dy + 1);
        uint8_t *p = nm + ((size_t)y * w + x) * 4;
        p[0] = (uint8_t)((-dx / l * 0.5f + 0.5f) * 255); p[1] = (uint8_t)((-dy / l * 0.5f + 0.5f) * 255); p[2] = 255; p[3] = 255;
    }
    printf("\n%s  %dx%d\n", argv[1], w, h);
    printf("  fmt  rdo    PSNR   encode    zap(raw)  zap(split)  bits/px on disk\n");
    for (int f = 0; f < 5; f++) {
        if (f == ZAP_BC4) continue; /* BC4 is half of BC5 */
        const uint8_t *src = f == ZAP_BC5 ? nm : img;
        size_t n = zap_bc_size(w, h, (zap_bc_format)f);
        uint8_t *bc = malloc(n), *sp = malloc(n);
        float lambdas[] = { 0, 10, 30, 100, 300 };
        for (int li = 0; li < 5; li++) {
            double t0 = now();
            zap_bc_encode(src, w, h, (size_t)w * 4, (zap_bc_format)f, lambdas[li], bc);
            double te = now() - t0;
            zap_bc_decode(bc, w, h, (zap_bc_format)f, out, (size_t)w * 4);
            zap_bc_split(bc, n, (zap_bc_format)f, sp);
            size_t zr = zap_size(bc, n), zs = zap_size(sp, n);
            printf("  %s %5.0f  %6.2f  %5.0f ms  %9zu  %10zu  %5.2f\n", NAMES[f], lambdas[li], psnr(src, out, npx, chmask((zap_bc_format)f)),
                   te * 1e3, zr, zs, zs * 8.0 / npx);
            if (argc > 4 && (li == 0 || li == 3)) {
                char path[512];
                snprintf(path, sizeof path, "%s_%s_rdo%.0f.dds", argv[4], NAMES[f], lambdas[li]);
                write_dds(path, bc, n, w, h, (zap_bc_format)f);
                snprintf(path, sizeof path, "%s_%s_rdo%.0f.ref", argv[4], NAMES[f], lambdas[li]);
                FILE *o = fopen(path, "wb"); fwrite(out, 4, npx, o); fclose(o); /* our own decode, to diff */
            }
        }
        free(bc); free(sp);
    }
    return 0;
}
