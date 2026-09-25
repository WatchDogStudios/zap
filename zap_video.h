/* zap_video.h - simple fast-decoding video codec for game cutscenes / UI video, built on zap.h.
 * https://github.com/WatchDogStudios/zap   SPDX-License-Identifier: MIT   Copyright (c) 2026 WD Studios Corp.
 *
 *   zap_video *e = zap_venc_create(w, h, quality 1..100, keyint, hc_depth);   hc_depth 0 = fast stream packing
 *   size_t n = zap_venc_frame(e, y, u, v, ystride, uvstride, pkt, zap_video_bound(e));
 *   zap_video *d = zap_vdec_create(w, h);
 *   if (zap_vdec_frame(d, pkt, n) == 0) zap_video_planes(d, &y, &u, &v, &ystride, &uvstride);
 *   zap_video_destroy(e); zap_video_destroy(d);
 *
 * Input/output is 8-bit YUV 4:2:0 (I420), even width/height. Each packet is one frame; store them in
 * whatever container you have. Packets must be decoded in order; I-frames are independent.
 *
 * Design: 16x16 macroblocks, modes SKIP / INTER (half-pel motion vector + residual) / INTRA,
 * 8x8 DCT with a bit-exact integer inverse (no drift across compilers or platforms), byte-aligned
 * run/level tokens split into three streams, each stored raw, zap-compressed or canonical-Huffman coded
 * (whichever is smallest). Decode is cheap; files are larger than H.264/MPEG-4 (see README).
 *
 * The decoder validates every packet: a corrupt or hostile packet returns -1 and the decoder then
 * waits for the next I-frame instead of drifting.
 */
#ifndef ZAP_VIDEO_H
#define ZAP_VIDEO_H

#include "zap.h"

/* SSE2 (always on for x64) for the decoder's inverse transform and motion compensation, and the
   encoder's SAD. Bit-exact with the scalar code. Define ZAP_NO_SIMD to force scalar. */
#if !defined(ZAP_NO_SIMD) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define ZAP__SSE2 1
#include <emmintrin.h>
#endif

enum { ZAP__VPAD = 32, ZAP__VCPAD = 16, ZAP__VMV = 128, ZAP__VHDR = 36 }; /* VMV: max |mv| in half-pels */
enum { ZAP__SKIP = 0, ZAP__INTER = 1, ZAP__INTRA = 2 };

typedef struct {
    int w, h, mbw, mbh, cw, ch, ys, cs, nmb;
    uint8_t *mem[2][3];            /* two padded frames: reconstruction ping-pongs with reference */
    int cur, have_ref, frame_no;
    uint8_t *modes, *mvs, *coefs;  /* raw stream scratch */
    size_t cap_mvs, cap_coefs;
    int16_t *mbmv;                 /* per-MB motion vector (x, y) in half-pels, for prediction */
    /* encoder only */
    int enc, quality, keyint, depth;
    uint8_t *src[3], *tmp;
    zap_state *st;
    zap_hc_state *hc;
} zap_video;

/* ---------------- tables */

static const int16_t ZAP__IC[8][8] = { /* round(4096 * c(k) * cos((2n+1) k pi / 16)) */
    { 1448, 1448, 1448, 1448, 1448, 1448, 1448, 1448 }, { 2009, 1703, 1138, 400, -400, -1138, -1703, -2009 },
    { 1892, 784, -784, -1892, -1892, -784, 784, 1892 }, { 1703, -400, -2009, -1138, 1138, 2009, 400, -1703 },
    { 1448, -1448, -1448, 1448, 1448, -1448, -1448, 1448 }, { 1138, -2009, 400, 1703, -1703, -400, 2009, -1138 },
    { 784, -1892, 1892, -784, -784, 1892, -1892, 784 }, { 400, -1138, 1703, -2009, 2009, -1703, 1138, -400 } };

static const uint8_t ZAP__ZZ[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
    45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

static const uint8_t ZAP__QBASE[2][64] = { /* JPEG annex K luma / chroma */
    { 16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
      51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121,
      120, 101, 72, 92, 95, 98, 112, 100, 103, 99 },
    { 17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99,
      99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
      99, 99, 99, 99, 99, 99, 99, 99 } };

/* step tables: [0] intra luma, [1] intra chroma, [2] inter (flat) */
static inline void zap__vsteps(int quality, uint8_t st[3][64]) {
    int q = quality < 1 ? 1 : quality > 100 ? 100 : quality, s = q < 50 ? 5000 / q : 200 - 2 * q;
    for (int t = 0; t < 3; t++)
        for (int i = 0; i < 64; i++) {
            int v = ((t < 2 ? ZAP__QBASE[t][i] : 16) * s + 50) / 100;
            st[t][i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
        }
}

/* ---------------- setup */

static inline uint8_t *zap__vplane(const zap_video *v, int f, int p) {
    return p ? v->mem[f][p] + ZAP__VCPAD * v->cs + ZAP__VCPAD : v->mem[f][0] + ZAP__VPAD * v->ys + ZAP__VPAD;
}

static inline void zap_video_destroy(zap_video *v) {
    if (!v) return;
    for (int f = 0; f < 2; f++) for (int p = 0; p < 3; p++) free(v->mem[f][p]);
    for (int p = 0; p < 3; p++) free(v->src[p]);
    free(v->tmp);
    free(v->modes); free(v->mvs); free(v->coefs); free(v->mbmv); free(v->st); free(v->hc);
    free(v);
}

static inline zap_video *zap__vcreate(int w, int h) {
    if (w < 2 || h < 2 || (w | h) & 1 || w > 16384 || h > 16384) return NULL;
    zap_video *v = (zap_video *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->w = w; v->h = h; v->mbw = (w + 15) / 16; v->mbh = (h + 15) / 16; v->nmb = v->mbw * v->mbh;
    v->cw = v->mbw * 16; v->ch = v->mbh * 16;
    v->ys = v->cw + 2 * ZAP__VPAD; v->cs = v->cw / 2 + 2 * ZAP__VCPAD;
    size_t ysz = (size_t)v->ys * (size_t)(v->ch + 2 * ZAP__VPAD), csz = (size_t)v->cs * (size_t)(v->ch / 2 + 2 * ZAP__VCPAD);
    for (int f = 0; f < 2; f++) for (int p = 0; p < 3; p++) v->mem[f][p] = (uint8_t *)calloc(1, p ? csz : ysz);
    v->cap_mvs = 6 * (size_t)v->nmb;
    v->cap_coefs = (size_t)v->nmb * 6 * (64 * 3 + 1);
    v->modes = (uint8_t *)malloc((size_t)v->nmb);
    v->mvs = (uint8_t *)malloc(v->cap_mvs);
    v->coefs = (uint8_t *)malloc(v->cap_coefs);
    v->mbmv = (int16_t *)calloc((size_t)v->nmb, 2 * sizeof(int16_t));
    int ok = v->modes && v->mvs && v->coefs && v->mbmv;
    for (int f = 0; f < 2; f++) for (int p = 0; p < 3; p++) if (!v->mem[f][p]) ok = 0;
    if (!ok) { zap_video_destroy(v); return NULL; }
    return v;
}

static inline zap_video *zap_vdec_create(int w, int h) { return zap__vcreate(w, h); }

/* hc_depth 0 = zap fast compressor for the streams, >0 = zap hc (offline encodes) */
static inline zap_video *zap_venc_create(int w, int h, int quality, int keyint, int hc_depth) {
    zap_video *v = zap__vcreate(w, h);
    if (!v) return NULL;
    v->enc = 1; v->quality = quality; v->keyint = keyint < 1 ? 1 : keyint; v->depth = hc_depth;
    for (int p = 0; p < 3; p++) v->src[p] = (uint8_t *)malloc(p ? (size_t)(v->cw / 2) * (size_t)(v->ch / 2) : (size_t)v->cw * (size_t)v->ch);
    v->tmp = (uint8_t *)malloc(v->cap_coefs + 256);
    if (hc_depth) v->hc = (zap_hc_state *)malloc(sizeof *v->hc);
    else v->st = (zap_state *)malloc(sizeof *v->st);
    if (!v->src[0] || !v->src[1] || !v->src[2] || !v->tmp || (hc_depth ? !v->hc : !v->st)) { zap_video_destroy(v); return NULL; }
    return v;
}

static inline size_t zap_video_bound(const zap_video *v) { return ZAP__VHDR + (size_t)v->nmb + v->cap_mvs + v->cap_coefs; }

/* last decoded (or encoded) frame; planes are valid until the next zap_*_frame call */
static inline void zap_video_planes(const zap_video *v, const uint8_t **y, const uint8_t **u, const uint8_t **vp, int *ystride, int *uvstride) {
    int f = v->cur ^ 1;
    *y = zap__vplane(v, f, 0); *u = zap__vplane(v, f, 1); *vp = zap__vplane(v, f, 2);
    *ystride = v->ys; *uvstride = v->cs;
}

/* replicate edges into the padding so motion vectors may point off-frame */
static inline void zap__vextend(zap_video *v, int f) {
    for (int p = 0; p < 3; p++) {
        int pad = p ? ZAP__VCPAD : ZAP__VPAD, s = p ? v->cs : v->ys, w = p ? v->cw / 2 : v->cw, h = p ? v->ch / 2 : v->ch;
        uint8_t *o = zap__vplane(v, f, p);
        for (int y = 0; y < h; y++) { memset(o + y * s - pad, o[y * s], (size_t)pad); memset(o + y * s + w, o[y * s + w - 1], (size_t)pad); }
        for (int y = 1; y <= pad; y++) {
            memcpy(o - y * s - pad, o - pad, (size_t)(w + 2 * pad));
            memcpy(o + (h - 1 + y) * s - pad, o + (h - 1) * s - pad, (size_t)(w + 2 * pad));
        }
    }
}

/* ---------------- transform. Bit-exact integer inverse shared by encoder and decoder. */

static inline uint8_t zap__clamp8(int x) { return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x); }

/* dst = clamp(pred + idct(q * step)); pred == NULL means a flat 128. Scalar reference. */
static inline void zap__vidct_c(const int16_t *q, const uint8_t *step, const uint8_t *pred, int ps, uint8_t *dst, int ds) {
    int co[64], t[64], rows = 0;
    for (int i = 0; i < 64; i++) {
        int c = q[i] * step[i];
        co[i] = c < -4095 ? -4095 : c > 4095 ? 4095 : c;
        if (co[i]) rows |= 1 << (i >> 3);
    }
    for (int u = 0; u < 8; u++) {
        if (!(rows >> u & 1)) continue;
        for (int n = 0; n < 8; n++) {
            int s = 512;
            for (int k = 0; k < 8; k++) s += ZAP__IC[k][n] * co[u * 8 + k];
            s >>= 10;
            t[u * 8 + n] = s < -32768 ? -32768 : s > 32767 ? 32767 : s; /* = SSE2 packs saturation */
        }
    }
    for (int m = 0; m < 8; m++)
        for (int n = 0; n < 8; n++) {
            int s = 8192;
            for (int u = 0; u < 8; u++) if (rows >> u & 1) s += ZAP__IC[u][m] * t[u * 8 + n];
            dst[m * ds + n] = zap__clamp8((pred ? pred[m * ps + n] : 128) + (s >> 14));
        }
}

#ifdef ZAP__SSE2
/* same math: rows are 8 x int16, pairs of taps go through pmaddwd, saturation = the scalar clamps */
static inline void zap__vidct_sse2(const int16_t *q, const uint8_t *step, const uint8_t *pred, int ps, uint8_t *dst, int ds) {
    __m128i co[8], t[8];
    const __m128i lim = _mm_set1_epi16(4095), nlim = _mm_set1_epi16(-4095), zero = _mm_setzero_si128();
    for (int u = 0; u < 8; u++) { /* dequantize: 32-bit product, saturate to int16, clamp to +-4095 */
        __m128i qq = _mm_loadu_si128((const __m128i *)(q + 8 * u));
        __m128i st = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(step + 8 * u)), zero);
        __m128i lo = _mm_mullo_epi16(qq, st), hi = _mm_mulhi_epi16(qq, st);
        __m128i c = _mm_packs_epi32(_mm_unpacklo_epi16(lo, hi), _mm_unpackhi_epi16(lo, hi));
        co[u] = _mm_max_epi16(_mm_min_epi16(c, lim), nlim);
    }
    /* row pass: t[u][n] = sum_k C[k][n] co[u][k] */
    const __m128i r512 = _mm_set1_epi32(512), r8192 = _mm_set1_epi32(8192);
    __m128i ckl[4], ckh[4]; /* taps k, k+1 interleaved per output column */
    for (int k = 0; k < 8; k += 2) {
        __m128i r0 = _mm_loadu_si128((const __m128i *)ZAP__IC[k]), r1 = _mm_loadu_si128((const __m128i *)ZAP__IC[k + 1]);
        ckl[k / 2] = _mm_unpacklo_epi16(r0, r1); ckh[k / 2] = _mm_unpackhi_epi16(r0, r1);
    }
    for (int u = 0; u < 8; u++) {
        __m128i p0 = _mm_shuffle_epi32(co[u], 0x00), p1 = _mm_shuffle_epi32(co[u], 0x55); /* (co[k], co[k+1]) pairs */
        __m128i p2 = _mm_shuffle_epi32(co[u], 0xAA), p3 = _mm_shuffle_epi32(co[u], 0xFF);
        __m128i a = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ckl[0], p0), _mm_madd_epi16(ckl[1], p1)),
                                  _mm_add_epi32(_mm_madd_epi16(ckl[2], p2), _mm_madd_epi16(ckl[3], p3)));
        __m128i b = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(ckh[0], p0), _mm_madd_epi16(ckh[1], p1)),
                                  _mm_add_epi32(_mm_madd_epi16(ckh[2], p2), _mm_madd_epi16(ckh[3], p3)));
        t[u] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(a, r512), 10), _mm_srai_epi32(_mm_add_epi32(b, r512), 10));
    }
    /* column pass: x[m][n] = sum_u C[u][m] t[u][n] */
    __m128i tl[4], th[4];
    for (int u = 0; u < 8; u += 2) { tl[u / 2] = _mm_unpacklo_epi16(t[u], t[u + 1]); th[u / 2] = _mm_unpackhi_epi16(t[u], t[u + 1]); }
    for (int m = 0; m < 8; m++) {
        __m128i a = r8192, b = r8192;
        for (int u = 0; u < 8; u += 2) {
            __m128i cm = _mm_set1_epi32((int)((uint32_t)(uint16_t)ZAP__IC[u][m] | (uint32_t)(uint16_t)ZAP__IC[u + 1][m] << 16));
            a = _mm_add_epi32(a, _mm_madd_epi16(tl[u / 2], cm));
            b = _mm_add_epi32(b, _mm_madd_epi16(th[u / 2], cm));
        }
        __m128i x = _mm_packs_epi32(_mm_srai_epi32(a, 14), _mm_srai_epi32(b, 14));
        __m128i pr = pred ? _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(pred + m * ps)), zero) : _mm_set1_epi16(128);
        _mm_storel_epi64((__m128i *)(dst + m * ds), _mm_packus_epi16(_mm_adds_epi16(pr, x), x));
    }
}
#endif

static inline void zap__vidct(const int16_t *q, const uint8_t *step, const uint8_t *pred, int ps, uint8_t *dst, int ds) {
    int dc_only = 1;
    for (int i = 1; i < 64 && dc_only; i++) dc_only = q[i] == 0;
    if (dc_only) { /* the common case: exactly what both full paths compute for a lone DC */
        int c = q[0] * step[0];
        c = c < -4095 ? -4095 : c > 4095 ? 4095 : c;
        int x = (8192 + ZAP__IC[0][0] * ((512 + ZAP__IC[0][0] * c) >> 10)) >> 14;
        for (int m = 0; m < 8; m++) for (int n = 0; n < 8; n++) dst[m * ds + n] = zap__clamp8((pred ? pred[m * ps + n] : 128) + x);
        return;
    }
#ifdef ZAP__SSE2
    zap__vidct_sse2(q, step, pred, ps, dst, ds);
#else
    zap__vidct_c(q, step, pred, ps, dst, ds);
#endif
}

static inline void zap__vcopy(uint8_t *d, int ds, const uint8_t *s, int ss, int n) {
    for (int y = 0; y < n; y++) memcpy(d + y * ds, s + y * ss, (size_t)n);
}
static inline void zap__vfill(uint8_t *d, int ds, int n) { for (int y = 0; y < n; y++) memset(d + y * ds, 128, (size_t)n); }

static inline int zap__vclampi(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }
static inline int zap__floor2(int x) { return (x - (x < 0)) / 2; }

/* legal half-pel motion vector range for the MB at luma (x, y): the interpolated block, and the chroma
   block it implies, stay inside the padding */
static inline void zap__vmvrange(const zap_video *v, int x, int y, int *lx, int *hx, int *ly, int *hy) {
    *lx = 2 * (2 - ZAP__VPAD - x); *hx = 2 * (v->cw + ZAP__VPAD - 19 - x);
    *ly = 2 * (2 - ZAP__VPAD - y); *hy = 2 * (v->ch + ZAP__VPAD - 19 - y);
    if (*lx < -ZAP__VMV) *lx = -ZAP__VMV;
    if (*hx > ZAP__VMV) *hx = ZAP__VMV;
    if (*ly < -ZAP__VMV) *ly = -ZAP__VMV;
    if (*hy > ZAP__VMV) *hy = ZAP__VMV;
}

/* n x n prediction from plane ref at (x, y) displaced by a half-pel vector, bilinear, into o (stride n) */
static inline void zap__vpred_c(const uint8_t *ref, int rs, int x, int y, int mvx, int mvy, int n, uint8_t *o) {
    int ix = zap__floor2(mvx), iy = zap__floor2(mvy), fx = mvx - 2 * ix, fy = mvy - 2 * iy;
    const uint8_t *p = ref + (y + iy) * rs + x + ix;
    for (int r = 0; r < n; r++, p += rs, o += n) {
        if (!fx && !fy) memcpy(o, p, (size_t)n);
        else if (!fy) for (int c = 0; c < n; c++) o[c] = (uint8_t)((p[c] + p[c + 1] + 1) >> 1);
        else if (!fx) for (int c = 0; c < n; c++) o[c] = (uint8_t)((p[c] + p[c + rs] + 1) >> 1);
        else for (int c = 0; c < n; c++) o[c] = (uint8_t)((p[c] + p[c + 1] + p[c + rs] + p[c + rs + 1] + 2) >> 2);
    }
}

#ifdef ZAP__SSE2
static inline __m128i zap__vld(const uint8_t *p, int n) { return n == 16 ? _mm_loadu_si128((const __m128i *)p) : _mm_loadl_epi64((const __m128i *)p); }
static inline void zap__vst(uint8_t *p, __m128i v, int n) { if (n == 16) _mm_storeu_si128((__m128i *)p, v); else _mm_storel_epi64((__m128i *)p, v); }
static inline void zap__vpred_sse2(const uint8_t *ref, int rs, int x, int y, int mvx, int mvy, int n, uint8_t *o) {
    int ix = zap__floor2(mvx), iy = zap__floor2(mvy), fx = mvx - 2 * ix, fy = mvy - 2 * iy;
    const uint8_t *p = ref + (y + iy) * rs + x + ix;
    const __m128i z = _mm_setzero_si128(), two = _mm_set1_epi16(2);
    for (int r = 0; r < n; r++, p += rs, o += n) {
        __m128i a = zap__vld(p, n), v;
        if (!fx && !fy) v = a;
        else if (!fy) v = _mm_avg_epu8(a, zap__vld(p + 1, n)); /* (a + b + 1) >> 1 */
        else if (!fx) v = _mm_avg_epu8(a, zap__vld(p + rs, n));
        else { /* (a + b + c + d + 2) >> 2 in 16 bits */
            __m128i b = zap__vld(p + 1, n), c = zap__vld(p + rs, n), d = zap__vld(p + rs + 1, n);
            __m128i lo = _mm_add_epi16(_mm_add_epi16(_mm_unpacklo_epi8(a, z), _mm_unpacklo_epi8(b, z)), _mm_add_epi16(_mm_unpacklo_epi8(c, z), _mm_unpacklo_epi8(d, z)));
            __m128i hi = _mm_add_epi16(_mm_add_epi16(_mm_unpackhi_epi8(a, z), _mm_unpackhi_epi8(b, z)), _mm_add_epi16(_mm_unpackhi_epi8(c, z), _mm_unpackhi_epi8(d, z)));
            v = _mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(lo, two), 2), _mm_srli_epi16(_mm_add_epi16(hi, two), 2));
        }
        zap__vst(o, v, n);
    }
}
#endif

static inline void zap__vpred(const uint8_t *ref, int rs, int x, int y, int mvx, int mvy, int n, uint8_t *o) {
#ifdef ZAP__SSE2
    zap__vpred_sse2(ref, rs, x, y, mvx, mvy, n, o);
#else
    zap__vpred_c(ref, rs, x, y, mvx, mvy, n, o);
#endif
}

/* luma 16x16 + both chroma 8x8 predictions for a macroblock */
static inline void zap__vpredmb(const zap_video *v, uint8_t *const ref[3], int x, int y, int mvx, int mvy,
                                uint8_t py[256], uint8_t pu[64], uint8_t pv[64]) {
    int cmx = zap__floor2(mvx), cmy = zap__floor2(mvy);
    zap__vpred(ref[0], v->ys, x, y, mvx, mvy, 16, py);
    zap__vpred(ref[1], v->cs, x / 2, y / 2, cmx, cmy, 8, pu);
    zap__vpred(ref[2], v->cs, x / 2, y / 2, cmx, cmy, 8, pv);
}

/* motion vector deltas: one signed byte, or 0x80 + int16 */
static inline uint8_t *zap__vputmv(uint8_t *p, int d) {
    if (d >= -127 && d <= 127) *p++ = (uint8_t)(d & 0xFF);
    else { *p++ = 0x80; *p++ = (uint8_t)(d & 0xFF); *p++ = (uint8_t)((d >> 8) & 0xFF); }
    return p;
}
static inline int zap__vgetmv(const uint8_t **pp, const uint8_t *e, int *d) {
    const uint8_t *p = *pp;
    if (p >= e) return -1;
    if (*p != 0x80) { *d = (int8_t)*p; *pp = p + 1; return 0; }
    if (e - p < 3) return -1;
    *d = (int16_t)(uint16_t)(p[1] | p[2] << 8); *pp = p + 3;
    return 0;
}

/* ---------------- tokens: 0x00-0x7F = run(3b) level(-8..8, !=0); 0x80 = end of block; 0x81+run, int16 level */

static inline uint8_t *zap__vput(uint8_t *p, const int16_t *q) {
    int run = 0;
    for (int i = 0; i < 64; i++) {
        int c = q[ZAP__ZZ[i]];
        if (!c) { run++; continue; }
        if (run < 8 && c >= -8 && c <= 8) *p++ = (uint8_t)(run << 4 | (c < 0 ? c + 8 : c + 7));
        else { *p++ = (uint8_t)(0x81 + run); *p++ = (uint8_t)(c & 0xFF); *p++ = (uint8_t)((c >> 8) & 0xFF); }
        run = 0;
    }
    *p++ = 0x80;
    return p;
}

static inline int zap__vget(const uint8_t **pp, const uint8_t *end, int16_t *q) {
    const uint8_t *p = *pp;
    memset(q, 0, 64 * sizeof *q);
    for (int pos = 0;;) {
        if (p >= end) return -1;
        int t = *p++, run, c;
        if (t == 0x80) break;
        if (t < 0x80) { run = t >> 4; c = (t & 15) < 8 ? (t & 15) - 8 : (t & 15) - 7; }
        else { if (end - p < 2) return -1; run = t - 0x81; c = (int16_t)(uint16_t)(p[0] | p[1] << 8); p += 2; }
        pos += run;
        if (pos > 63) return -1;
        q[ZAP__ZZ[pos++]] = (int16_t)c;
    }
    *pp = p;
    return 0;
}

/* ---------------- encoder */

static inline int zap__vsad(const uint8_t *a, int as, const uint8_t *b, int bs, int n) {
#ifdef ZAP__SSE2
    if (n == 16) {
        __m128i acc = _mm_setzero_si128();
        for (int y = 0; y < 16; y++)
            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadu_si128((const __m128i *)(a + y * as)), _mm_loadu_si128((const __m128i *)(b + y * bs))));
        return _mm_cvtsi128_si32(acc) + _mm_cvtsi128_si32(_mm_srli_si128(acc, 8));
    }
#endif
    int s = 0;
    for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) { int d = a[y * as + x] - b[y * bs + x]; s += d < 0 ? -d : d; }
    return s;
}

/* float forward DCT (encoder only) + quantize with precomputed 1/(4096^2 * step). Returns nonzero count.
   Encoder-side only, so it needn't be bit-exact across platforms: the decoder reconstructs from the levels. */
static const float ZAP__FCT[8][8] = { /* ZAP__IC transposed */
    { 1448.f, 2009.f, 1892.f, 1703.f, 1448.f, 1138.f, 784.f, 400.f }, { 1448.f, 1703.f, 784.f, -400.f, -1448.f, -2009.f, -1892.f, -1138.f },
    { 1448.f, 1138.f, -784.f, -2009.f, -1448.f, 400.f, 1892.f, 1703.f }, { 1448.f, 400.f, -1892.f, -1138.f, 1448.f, 1703.f, -784.f, -2009.f },
    { 1448.f, -400.f, -1892.f, 1138.f, 1448.f, -1703.f, -784.f, 2009.f }, { 1448.f, -1138.f, -784.f, 2009.f, -1448.f, -400.f, 1892.f, -1703.f },
    { 1448.f, -1703.f, 784.f, 400.f, -1448.f, 2009.f, -1892.f, 1138.f }, { 1448.f, -2009.f, 1892.f, -1703.f, 1448.f, -1138.f, 784.f, -400.f } };

static inline int zap__vfdctq(const int *res, const float *inv, float round, int16_t *q) {
    float t[64], o[64];
#ifdef ZAP__SSE2
    for (int m = 0; m < 8; m++) { /* t[m][:] = sum_n x[m][n] * C[:][n] */
        __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
        for (int n = 0; n < 8; n++) {
            __m128 x = _mm_set1_ps((float)res[m * 8 + n]);
            lo = _mm_add_ps(lo, _mm_mul_ps(x, _mm_loadu_ps(ZAP__FCT[n]))); hi = _mm_add_ps(hi, _mm_mul_ps(x, _mm_loadu_ps(ZAP__FCT[n] + 4)));
        }
        _mm_storeu_ps(t + m * 8, lo); _mm_storeu_ps(t + m * 8 + 4, hi);
    }
    for (int u = 0; u < 8; u++) { /* o[u][:] = sum_m C[u][m] * t[m][:], pre-scaled */
        __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
        for (int m = 0; m < 8; m++) {
            __m128 c = _mm_set1_ps((float)ZAP__IC[u][m]);
            lo = _mm_add_ps(lo, _mm_mul_ps(c, _mm_loadu_ps(t + m * 8))); hi = _mm_add_ps(hi, _mm_mul_ps(c, _mm_loadu_ps(t + m * 8 + 4)));
        }
        _mm_storeu_ps(o + u * 8, _mm_mul_ps(lo, _mm_loadu_ps(inv + u * 8))); _mm_storeu_ps(o + u * 8 + 4, _mm_mul_ps(hi, _mm_loadu_ps(inv + u * 8 + 4)));
    }
#else
    for (int m = 0; m < 8; m++)
        for (int vv = 0; vv < 8; vv++) { float a = 0; for (int n = 0; n < 8; n++) a += ZAP__FCT[n][vv] * (float)res[m * 8 + n]; t[m * 8 + vv] = a; }
    for (int u = 0; u < 8; u++)
        for (int vv = 0; vv < 8; vv++) { float a = 0; for (int m = 0; m < 8; m++) a += ZAP__IC[u][m] * t[m * 8 + vv]; o[u * 8 + vv] = a * inv[u * 8 + vv]; }
#endif
    int nz = 0;
    for (int i = 0; i < 64; i++) {
        float v = o[i];
        int l = (int)(v < 0 ? -(-v + round) : v + round);
        l = l < -32767 ? -32767 : l > 32767 ? 32767 : l;
        q[i] = (int16_t)l;
        nz += l != 0;
    }
    return nz;
}

/* predictive + logarithmic full-pel search, then half-pel refinement. Vectors in half-pels. */
static inline int zap__vsearch(const zap_video *v, const uint8_t *sb, const uint8_t *ref, int x, int y, int px, int py,
                               int tx, int ty, int *bx, int *by) {
    int lx, hx, ly, hy;
    zap__vmvrange(v, x, y, &lx, &hx, &ly, &hy);
    int flx = -zap__floor2(-lx), fhx = zap__floor2(hx), fly = -zap__floor2(-ly), fhy = zap__floor2(hy); /* full-pel range */
    int cx[3] = { 0, zap__floor2(px), zap__floor2(tx) }, cy[3] = { 0, zap__floor2(py), zap__floor2(ty) };
    int best = 1 << 30, fx = 0, fy = 0;
    for (int i = 0; i < 3; i++) {
        int mx = zap__vclampi(cx[i], flx, fhx), my = zap__vclampi(cy[i], fly, fhy);
        int s = zap__vsad(sb, v->cw, ref + (y + my) * v->ys + x + mx, v->ys, 16);
        if (s < best) { best = s; fx = mx; fy = my; }
    }
    static const int DX[8] = { 1, -1, 0, 0, 1, 1, -1, -1 }, DY[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
    for (int step = 8; step >= 1; step >>= 1)
        for (int moved = 1, it = 0; moved && it < 4; it++) {
            moved = 0;
            int ox = fx, oy = fy;
            for (int d = 0; d < 4; d++) {
                int mx = ox + DX[d] * step, my = oy + DY[d] * step;
                if (mx < flx || mx > fhx || my < fly || my > fhy) continue;
                int s = zap__vsad(sb, v->cw, ref + (y + my) * v->ys + x + mx, v->ys, 16);
                if (s < best) { best = s; fx = mx; fy = my; moved = 1; }
            }
        }
    *bx = 2 * fx; *by = 2 * fy;
    int cx0 = *bx, cy0 = *by;
    uint8_t tmp[256];
    for (int d = 0; d < 8; d++) {
        int mx = cx0 + DX[d], my = cy0 + DY[d];
        if (mx < lx || mx > hx || my < ly || my > hy) continue;
        zap__vpred(ref, v->ys, x, y, mx, my, 16, tmp);
        int s = zap__vsad(sb, v->cw, tmp, 16, 16);
        if (s < best) { best = s; *bx = mx; *by = my; }
    }
    return best;
}

/* append a stream as raw (0), zap (1) or Huffman (2), whichever is smallest. slot: method byte + (raw, stored) sizes */
static inline size_t zap__vpack(zap_video *v, const uint8_t *s, size_t n, uint8_t *out, size_t cap, uint8_t *method, uint8_t *sizes) {
    size_t lim = cap < n ? cap : n, best = n, z = 0, h = 0;
    int m = 0;
    if (n > 16) z = v->depth ? zap_compress_hc(s, n, out, lim ? lim - 1 : 0, v->hc, NULL, v->depth)
                             : zap_compress(s, n, out, lim ? lim - 1 : 0, v->st, NULL);
    if (z && z < best) { best = z; m = 1; }
    if (n > 256) h = zap__henc(s, n, v->tmp, best - 1 < v->cap_coefs + 256 ? best - 1 : v->cap_coefs + 256);
    if (h && h < best && h <= cap) { best = h; m = 2; memcpy(out, v->tmp, h); }
    if (m == 0) { if (cap < n) return (size_t)-1; memcpy(out, s, n); }
    *method = (uint8_t)m;
    zap__w32(sizes, (uint32_t)n); zap__w32(sizes + 4, (uint32_t)best);
    return best;
}

/* returns packet size, 0 on failure (cap too small). Planes: y is w x h, u/v are w/2 x h/2. */
static inline size_t zap_venc_frame(zap_video *v, const uint8_t *y, const uint8_t *u, const uint8_t *vp, int ystride,
                                    int uvstride, void *out_, size_t cap) {
    uint8_t *out = (uint8_t *)out_;
    if (!v->enc || cap < ZAP__VHDR) return 0;
    const uint8_t *in[3] = { y, u, vp };
    for (int p = 0; p < 3; p++) { /* copy in, replicating edges up to the macroblock grid */
        int w = p ? v->w / 2 : v->w, h = p ? v->h / 2 : v->h, cw = p ? v->cw / 2 : v->cw, ch = p ? v->ch / 2 : v->ch, is = p ? uvstride : ystride;
        for (int r = 0; r < ch; r++) {
            const uint8_t *s = in[p] + (size_t)(r < h ? r : h - 1) * (size_t)is;
            memcpy(v->src[p] + (size_t)r * cw, s, (size_t)w);
            memset(v->src[p] + (size_t)r * cw + w, s[w - 1], (size_t)(cw - w));
        }
    }
    int key = !v->have_ref || v->frame_no % v->keyint == 0;
    uint8_t steps[3][64];
    float inv[3][64];
    zap__vsteps(v->quality, steps);
    for (int t = 0; t < 3; t++) for (int i = 0; i < 64; i++) inv[t][i] = 1.0f / (4096.0f * 4096.0f * (float)steps[t][i]);
    uint8_t *rec[3], *ref[3], *pm = v->modes, *pv = v->mvs, *pc = v->coefs;
    for (int p = 0; p < 3; p++) { rec[p] = zap__vplane(v, v->cur, p); ref[p] = zap__vplane(v, v->cur ^ 1, p); }
    int ccw = v->cw / 2;
    for (int my = 0; my < v->mbh; my++)
        for (int mx = 0; mx < v->mbw; mx++) {
            int i = my * v->mbw + mx, x = mx * 16, yy = my * 16, mode = ZAP__INTRA, mvx = 0, mvy = 0, cbp = 0;
            int px = mx ? v->mbmv[2 * (i - 1)] : 0, py = mx ? v->mbmv[2 * (i - 1) + 1] : 0;
            const uint8_t *sb = v->src[0] + (size_t)yy * v->cw + x;
            if (!key) {
                int tx = my ? v->mbmv[2 * (i - v->mbw)] : 0, ty = my ? v->mbmv[2 * (i - v->mbw) + 1] : 0;
                int sad = zap__vsearch(v, sb, ref[0], x, yy, px, py, tx, ty, &mvx, &mvy), mean = 0, act = 0;
                for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) mean += sb[r * v->cw + c];
                mean = (mean + 128) >> 8;
                for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) { int d = sb[r * v->cw + c] - mean; act += d < 0 ? -d : d; }
                mode = act + 512 < sad ? ZAP__INTRA : ZAP__INTER;
            }
            if (mode == ZAP__INTRA) mvx = mvy = 0;
            uint8_t pry[256], pru[64], prv[64];
            if (mode == ZAP__INTER) zap__vpredmb(v, ref, x, yy, mvx, mvy, pry, pru, prv);
            int16_t q[6][64];
            const uint8_t *pred[6] = { 0 };
            uint8_t *dst[6];
            int ds[6], pst[6];
            for (int b = 0; b < 6; b++) {
                int p = b < 4 ? 0 : b - 3, bx = b < 4 ? x + (b & 1) * 8 : x / 2, by = b < 4 ? yy + (b >> 1) * 8 : yy / 2;
                int ss = p ? ccw : v->cw, rs = p ? v->cs : v->ys, res[64];
                const uint8_t *s = v->src[p] + (size_t)by * ss + bx;
                dst[b] = rec[p] + by * rs + bx; ds[b] = rs; pst[b] = p ? 8 : 16;
                if (mode == ZAP__INTER) pred[b] = p == 0 ? pry + (b >> 1) * 128 + (b & 1) * 8 : p == 1 ? pru : prv;
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++) res[r * 8 + c] = s[r * ss + c] - (pred[b] ? pred[b][r * pst[b] + c] : 128);
                if (zap__vfdctq(res, inv[mode == ZAP__INTER ? 2 : p ? 1 : 0], mode == ZAP__INTER ? 0.3f : 0.5f, q[b])) cbp |= 1 << b;
            }
            if (mode == ZAP__INTER && !cbp && !mvx && !mvy) mode = ZAP__SKIP;
            *pm++ = (uint8_t)(mode | cbp << 2);
            if (mode == ZAP__INTER) { pv = zap__vputmv(pv, mvx - px); pv = zap__vputmv(pv, mvy - py); }
            v->mbmv[2 * i] = (int16_t)mvx; v->mbmv[2 * i + 1] = (int16_t)mvy;
            for (int b = 0; b < 6; b++) { /* reconstruct exactly like the decoder */
                const uint8_t *st = mode == ZAP__INTRA ? steps[b < 4 ? 0 : 1] : steps[2];
                if (cbp >> b & 1) { pc = zap__vput(pc, q[b]); zap__vidct(q[b], st, pred[b], pst[b], dst[b], ds[b]); }
                else if (pred[b]) zap__vcopy(dst[b], ds[b], pred[b], pst[b], 8);
                else zap__vfill(dst[b], ds[b], 8);
            }
        }
    zap__vextend(v, v->cur);
    /* header: "ZV" type quality w h, 3 method bytes, pad, then (raw, stored) sizes for modes / mvs / coefs */
    out[0] = 'Z'; out[1] = 'V'; out[2] = (uint8_t)(key ? 'I' : 'P'); out[3] = (uint8_t)v->quality;
    out[4] = (uint8_t)v->w; out[5] = (uint8_t)(v->w >> 8); out[6] = (uint8_t)v->h; out[7] = (uint8_t)(v->h >> 8);
    out[11] = 0;
    size_t pos = ZAP__VHDR, c;
    const uint8_t *streams[3] = { v->modes, v->mvs, v->coefs };
    size_t lens[3] = { (size_t)(pm - v->modes), (size_t)(pv - v->mvs), (size_t)(pc - v->coefs) };
    for (int s = 0; s < 3; s++) {
        if ((c = zap__vpack(v, streams[s], lens[s], out + pos, cap - pos, out + 8 + s, out + 12 + 8 * s)) == (size_t)-1) return 0;
        pos += c;
    }
    v->cur ^= 1; v->have_ref = 1; v->frame_no++;
    return pos;
}

/* ---------------- decoder */

static inline int zap__vdec(zap_video *v, const void *pkt_, size_t n) {
    const uint8_t *pkt = (const uint8_t *)pkt_;
    if (n < ZAP__VHDR || pkt[0] != 'Z' || pkt[1] != 'V' || (pkt[2] != 'I' && pkt[2] != 'P')) return -1;
    if ((pkt[4] | pkt[5] << 8) != v->w || (pkt[6] | pkt[7] << 8) != v->h) return -1;
    int key = pkt[2] == 'I';
    if (!key && !v->have_ref) return -1;
    uint8_t *bufs[3] = { v->modes, v->mvs, v->coefs };
    size_t caps[3] = { (size_t)v->nmb, v->cap_mvs, v->cap_coefs }, lens[3], pos = ZAP__VHDR;
    for (int s = 0; s < 3; s++) {
        size_t raw = zap__r32(pkt + 12 + 8 * s), c = zap__r32(pkt + 16 + 8 * s);
        int m = pkt[8 + s];
        if (raw > caps[s] || c > n - pos || (s == 0 && raw != (size_t)v->nmb)) return -1;
        if (m == 0) { if (c != raw) return -1; memcpy(bufs[s], pkt + pos, raw); }
        else if (m == 1) { if (c >= raw || zap_decompress(pkt + pos, c, bufs[s], raw, NULL) != (ptrdiff_t)raw) return -1; }
        else if (m == 2) { if (zap__hdec(pkt + pos, c, bufs[s], raw)) return -1; }
        else return -1;
        lens[s] = raw; pos += c;
    }
    uint8_t steps[3][64];
    zap__vsteps(pkt[3], steps);
    uint8_t *rec[3], *ref[3];
    for (int p = 0; p < 3; p++) { rec[p] = zap__vplane(v, v->cur, p); ref[p] = zap__vplane(v, v->cur ^ 1, p); }
    const uint8_t *pv = v->mvs, *pve = v->mvs + lens[1], *pc = v->coefs, *pce = v->coefs + lens[2];
    for (int my = 0; my < v->mbh; my++)
        for (int mx = 0; mx < v->mbw; mx++) {
            int i = my * v->mbw + mx, x = mx * 16, yy = my * 16, mode = v->modes[i] & 3, cbp = v->modes[i] >> 2, mvx = 0, mvy = 0;
            if (mode == 3 || (key && mode != ZAP__INTRA) || (mode == ZAP__SKIP && cbp)) return -1;
            if (mode == ZAP__INTER) {
                int lx, hx, ly, hy, dx, dy;
                if (zap__vgetmv(&pv, pve, &dx) || zap__vgetmv(&pv, pve, &dy)) return -1;
                zap__vmvrange(v, x, yy, &lx, &hx, &ly, &hy);
                mvx = zap__vclampi((mx ? v->mbmv[2 * (i - 1)] : 0) + dx, lx, hx);
                mvy = zap__vclampi((mx ? v->mbmv[2 * (i - 1) + 1] : 0) + dy, ly, hy);
            }
            v->mbmv[2 * i] = (int16_t)mvx; v->mbmv[2 * i + 1] = (int16_t)mvy;
            uint8_t pry[256], pru[64], prv[64];
            if (mode != ZAP__INTRA) zap__vpredmb(v, ref, x, yy, mvx, mvy, pry, pru, prv);
            for (int b = 0; b < 6; b++) {
                int p = b < 4 ? 0 : b - 3, bx = b < 4 ? x + (b & 1) * 8 : x / 2, by = b < 4 ? yy + (b >> 1) * 8 : yy / 2;
                int rs = p ? v->cs : v->ys, ps = p ? 8 : 16;
                uint8_t *dst = rec[p] + by * rs + bx;
                const uint8_t *pred = mode == ZAP__INTRA ? NULL : p == 0 ? pry + (b >> 1) * 128 + (b & 1) * 8 : p == 1 ? pru : prv;
                if (cbp >> b & 1) {
                    int16_t q[64];
                    if (zap__vget(&pc, pce, q)) return -1;
                    zap__vidct(q, mode == ZAP__INTRA ? steps[p ? 1 : 0] : steps[2], pred, ps, dst, rs);
                } else if (pred) zap__vcopy(dst, rs, pred, ps, 8);
                else zap__vfill(dst, rs, 8);
            }
        }
    if (pv != pve || pc != pce) return -1;
    zap__vextend(v, v->cur);
    v->cur ^= 1;
    return 0;
}

/* 0 ok, -1 corrupt (the decoder then rejects P-frames until the next I-frame) */
static inline int zap_vdec_frame(zap_video *v, const void *pkt, size_t n) {
    int r = v->enc ? -1 : zap__vdec(v, pkt, n);
    v->have_ref = r == 0;
    return r;
}

#endif
