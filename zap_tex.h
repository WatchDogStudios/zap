/* zap_tex.h - GPU texture block compression (BC1/BC3/BC4/BC5/BC7) tuned to pair with zap.h.
 * https://github.com/WatchDogStudios/zap   SPDX-License-Identifier: MIT   Copyright (c) 2026 WD Studios Corp.
 *
 *   zap_bc_encode(rgba, w, h, stride, fmt, rdo, out)   RGBA8 in -> BCn blocks out (zap_bc_size bytes)
 *   zap_bc_decode(bc, w, h, fmt, rgba, stride)         BCn -> RGBA8 (for tools / verification)
 *   zap_bc_split / zap_bc_merge                         lossless: regroup block fields into planes before zap
 *
 * rdo = 0: best quality. rdo > 0 (try 5..200): rate-distortion optimisation - blocks may reuse the
 * endpoints / indices of recent blocks when the extra error is worth the bytes zap saves. Output is
 * still plain BCn: the GPU never knows.
 *
 * Pipeline for shipping:  encode(rdo) -> split -> zap_compress_hc  ...  zap_decompress -> merge -> upload
 *
 * Formats: BC1 = RGB (alpha ignored), BC3 = RGBA, BC4 = R, BC5 = RG (normal maps), BC7 = RGBA high quality.
 * BC7: the decoder handles all 8 modes; the encoder uses mode 6 (smooth / alpha) and mode 1 (2 partitions,
 * opaque edges). BC7 RDO reuses whole blocks, or the endpoint / index bytes of recent mode-6 blocks.
 * Decoded BC4 is (r,0,0,255) and BC5 is (r,g,0,255), like GPU sampling.
 * Any width/height; partial edge blocks replicate the edge pixels.
 */
#ifndef ZAP_TEX_H
#define ZAP_TEX_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum { ZAP_BC1, ZAP_BC3, ZAP_BC4, ZAP_BC5, ZAP_BC7 } zap_bc_format;

static inline size_t zap_bc_block_bytes(zap_bc_format f) { return f == ZAP_BC1 || f == ZAP_BC4 ? 8 : 16; }
static inline size_t zap_bc_size(int w, int h, zap_bc_format f) {
    return (size_t)((w + 3) / 4) * (size_t)((h + 3) / 4) * zap_bc_block_bytes(f);
}

/* ---------------- BC1 color sub-block: u16 c0, u16 c1, u32 2-bit indices (pixel 0 = low bits) */

static inline void zap__565(uint16_t c, int *r, int *g, int *b) {
    int R = (c >> 11) & 31, G = (c >> 5) & 63, B = c & 31;
    *r = (R << 3) | (R >> 2); *g = (G << 2) | (G >> 4); *b = (B << 3) | (B >> 2);
}
static inline uint16_t zap__to565(float r, float g, float b) {
    int R = (int)(r * 31.0f / 255.0f + 0.5f), G = (int)(g * 63.0f / 255.0f + 0.5f), B = (int)(b * 31.0f / 255.0f + 0.5f);
    R = R < 0 ? 0 : R > 31 ? 31 : R; G = G < 0 ? 0 : G > 63 ? 63 : G; B = B < 0 ? 0 : B > 31 ? 31 : B;
    return (uint16_t)((R << 11) | (G << 5) | B);
}

/* palette; ncol = 4, or 3 when (c0 <= c1 && !force4): index 3 is transparent black, never chosen by us */
static inline int zap__bc1_pal(uint16_t c0, uint16_t c1, int force4, int pal[4][3]) {
    zap__565(c0, &pal[0][0], &pal[0][1], &pal[0][2]);
    zap__565(c1, &pal[1][0], &pal[1][1], &pal[1][2]);
    int four = force4 || c0 > c1;
    for (int k = 0; k < 3; k++) {
        int a = pal[0][k], b = pal[1][k];
        pal[2][k] = four ? (2 * a + b + 1) / 3 : (a + b + 1) / 2;
        pal[3][k] = four ? (a + 2 * b + 1) / 3 : 0;
    }
    return four ? 4 : 3;
}

static inline int zap__sq3(const uint8_t *p, const int *q) {
    int dr = p[0] - q[0], dg = p[1] - q[1], db = p[2] - q[2];
    return dr * dr + dg * dg + db * db;
}

/* best indices for fixed endpoints; returns SSE */
static inline int zap__bc1_fit(const uint8_t px[16][4], uint16_t c0, uint16_t c1, int force4, uint32_t *idx) {
    int pal[4][3], n = zap__bc1_pal(c0, c1, force4, pal), err = 0;
    uint32_t bits = 0;
    for (int i = 0; i < 16; i++) {
        int best = 0, be = zap__sq3(px[i], pal[0]);
        for (int k = 1; k < n; k++) { int e = zap__sq3(px[i], pal[k]); if (e < be) { be = e; best = k; } }
        bits |= (uint32_t)best << (2 * i); err += be;
    }
    *idx = bits;
    return err;
}

static inline int zap__bc1_eval(const uint8_t px[16][4], uint16_t c0, uint16_t c1, int force4, uint32_t idx) {
    int pal[4][3], err = 0;
    zap__bc1_pal(c0, c1, force4, pal);
    for (int i = 0; i < 16; i++) err += zap__sq3(px[i], pal[(idx >> (2 * i)) & 3]);
    return err;
}

/* SSE of an endpoint pair, canonicalised in place to 4-color order (c0 > c1), or solid (c0 == c1, idx 0) */
static inline int zap__bc1_try(const uint8_t px[16][4], uint16_t *c0, uint16_t *c1, uint32_t *idx) {
    if (*c0 < *c1) { uint16_t t = *c0; *c0 = *c1; *c1 = t; }
    if (*c0 == *c1) { *idx = 0; return zap__bc1_eval(px, *c0, *c1, 1, 0); }
    return zap__bc1_fit(px, *c0, *c1, 1, idx);
}

/* quality BC1: principal axis (2 starts) + least squares + endpoint hill climb. Returns SSE. */
static inline int zap__bc1_encode(const uint8_t px[16][4], uint16_t *oc0, uint16_t *oc1, uint32_t *oidx) {
    float mean[3] = { 0, 0, 0 }, cov[6] = { 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 16; i++) for (int k = 0; k < 3; k++) mean[k] += px[i][k] / 16.0f;
    for (int i = 0; i < 16; i++) {
        float r = px[i][0] - mean[0], g = px[i][1] - mean[1], b = px[i][2] - mean[2];
        cov[0] += r * r; cov[1] += r * g; cov[2] += r * b; cov[3] += g * g; cov[4] += g * b; cov[5] += b * b;
    }
    float ax[3] = { 1, 1, 1 };
    for (int it = 0; it < 6; it++) { /* power iteration */
        float x = cov[0] * ax[0] + cov[1] * ax[1] + cov[2] * ax[2];
        float y = cov[1] * ax[0] + cov[3] * ax[1] + cov[4] * ax[2];
        float z = cov[2] * ax[0] + cov[4] * ax[1] + cov[5] * ax[2];
        float m = x < 0 ? -x : x, my = y < 0 ? -y : y, mz = z < 0 ? -z : z; /* scale by max |component|: no sqrt */
        if (my > m) m = my;
        if (mz > m) m = mz;
        if (m < 1e-6f) break;
        ax[0] = x / m; ax[1] = y / m; ax[2] = z / m;
    }
    float n2 = ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2];
    float lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < 16; i++) {
        float t = (px[i][0] - mean[0]) * ax[0] + (px[i][1] - mean[1]) * ax[1] + (px[i][2] - mean[2]) * ax[2];
        if (t < lo) lo = t;
        if (t > hi) hi = t;
    }
    uint16_t best0 = 0, best1 = 0;
    uint32_t bidx = 0;
    int best = 1 << 30;
    for (int start = 0; start < 2; start++) { /* principal axis with / without inset, each refined */
        float in = start ? 0.0f : (hi - lo) / 16.0f, l = (lo + in) / n2, u = (hi - in) / n2;
        uint16_t c0 = zap__to565(mean[0] + ax[0] * u, mean[1] + ax[1] * u, mean[2] + ax[2] * u);
        uint16_t c1 = zap__to565(mean[0] + ax[0] * l, mean[1] + ax[1] * l, mean[2] + ax[2] * l);
        uint32_t idx = 0;
        int e = zap__bc1_try(px, &c0, &c1, &idx);
        for (int it = 0; it < 3 && e > 0; it++) { /* least squares: x_i ~ w_i*a + (1-w_i)*b */
            static const float W[4] = { 1.0f, 0.0f, 2.0f / 3.0f, 1.0f / 3.0f };
            float aa = 0, ab = 0, bb = 0, ax_[3] = { 0, 0, 0 }, bx[3] = { 0, 0, 0 };
            for (int i = 0; i < 16; i++) {
                float w = W[(idx >> (2 * i)) & 3], v = 1.0f - w;
                aa += w * w; ab += w * v; bb += v * v;
                for (int k = 0; k < 3; k++) { ax_[k] += w * px[i][k]; bx[k] += v * px[i][k]; }
            }
            float det = aa * bb - ab * ab;
            if (det < 1e-6f && det > -1e-6f) break;
            float A[3], B[3];
            for (int k = 0; k < 3; k++) { A[k] = (ax_[k] * bb - bx[k] * ab) / det; B[k] = (bx[k] * aa - ax_[k] * ab) / det; }
            uint16_t n0 = zap__to565(A[0], A[1], A[2]), n1 = zap__to565(B[0], B[1], B[2]);
            uint32_t ni = 0;
            int ne = zap__bc1_try(px, &n0, &n1, &ni);
            if (ne >= e) break;
            e = ne; c0 = n0; c1 = n1; idx = ni;
        }
        if (e < best) { best = e; best0 = c0; best1 = c1; bidx = idx; }
    }
    /* hill climb: nudge each 565 channel of each endpoint by +-1 while it helps */
    static const int SH[3] = { 11, 5, 0 }, MX[3] = { 31, 63, 31 };
    for (int pass = 0, improved = 1; improved && pass < 8 && best > 0; pass++) {
        improved = 0;
        for (int ep = 0; ep < 2; ep++) for (int ch = 0; ch < 3; ch++) for (int d = -1; d <= 1; d += 2) {
            uint16_t c = ep ? best1 : best0;
            int v = (c >> SH[ch]) & MX[ch];
            if (v + d < 0 || v + d > MX[ch]) continue;
            c = (uint16_t)((c & ~(MX[ch] << SH[ch])) | ((v + d) << SH[ch]));
            uint16_t n0 = ep ? best0 : c, n1 = ep ? c : best1;
            uint32_t ni = 0;
            int ne = zap__bc1_try(px, &n0, &n1, &ni);
            if (ne < best) { best = ne; best0 = n0; best1 = n1; bidx = ni; improved = 1; }
        }
    }
    *oc0 = best0; *oc1 = best1; *oidx = bidx;
    return best;
}

/* ---------------- BC4 channel sub-block: u8 a0, u8 a1, 48 bits of 3-bit indices */

static inline void zap__bc4_pal(int a0, int a1, int pal[8]) {
    pal[0] = a0; pal[1] = a1;
    if (a0 > a1) for (int i = 2; i < 8; i++) pal[i] = ((8 - i) * a0 + (i - 1) * a1 + 3) / 7;
    else { for (int i = 2; i < 6; i++) pal[i] = ((6 - i) * a0 + (i - 1) * a1 + 2) / 5; pal[6] = 0; pal[7] = 255; }
}

static inline int zap__bc4_fit(const uint8_t v[16], int a0, int a1, uint64_t *idx) {
    int pal[8], err = 0;
    uint64_t bits = 0;
    zap__bc4_pal(a0, a1, pal);
    for (int i = 0; i < 16; i++) {
        int best = 0, be = (v[i] - pal[0]) * (v[i] - pal[0]);
        for (int k = 1; k < 8; k++) { int e = (v[i] - pal[k]) * (v[i] - pal[k]); if (e < be) { be = e; best = k; } }
        bits |= (uint64_t)best << (3 * i); err += be;
    }
    *idx = bits;
    return err;
}

static inline int zap__bc4_eval(const uint8_t v[16], int a0, int a1, uint64_t idx) {
    int pal[8], err = 0;
    zap__bc4_pal(a0, a1, pal);
    for (int i = 0; i < 16; i++) { int d = v[i] - pal[(idx >> (3 * i)) & 7]; err += d * d; }
    return err;
}

static inline int zap__bc4_encode(const uint8_t v[16], uint8_t *oa0, uint8_t *oa1, uint64_t *oidx) {
    int lo = 255, hi = 0;
    for (int i = 0; i < 16; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    int best = 1 << 30, b0 = hi, b1 = lo;
    uint64_t bi = 0;
    for (int d0 = 0; d0 <= 2; d0++) for (int d1 = 0; d1 <= 2; d1++) { /* small inset search */
        int a0 = hi - d0, a1 = lo + d1;
        if (a0 <= a1) { a0 = hi; a1 = lo; }
        uint64_t idx = 0;
        int e = a0 == a1 ? zap__bc4_eval(v, a0, a1, 0) : zap__bc4_fit(v, a0, a1, &idx);
        if (a0 == a1) idx = 0;
        if (e < best) { best = e; b0 = a0; b1 = a1; bi = idx; }
    }
    *oa0 = (uint8_t)b0; *oa1 = (uint8_t)b1; *oidx = bi;
    return best;
}

/* ---------------- block (de)serialisation, little-endian */

static inline uint64_t zap__ld(const uint8_t *p, int n) { uint64_t v = 0; for (int i = n - 1; i >= 0; i--) v = v << 8 | p[i]; return v; }
static inline void zap__st(uint8_t *p, uint64_t v, int n) { for (int i = 0; i < n; i++) { p[i] = (uint8_t)v; v >>= 8; } }

static inline void zap__put_bc1(uint8_t *o, uint16_t c0, uint16_t c1, uint32_t idx) { zap__st(o, c0, 2); zap__st(o + 2, c1, 2); zap__st(o + 4, idx, 4); }
static inline void zap__put_bc4(uint8_t *o, uint8_t a0, uint8_t a1, uint64_t idx) { o[0] = a0; o[1] = a1; zap__st(o + 2, idx, 6); }

/* ---------------- RDO
 * Rate model for zap on split planes: a field costs 0 bytes if it equals the field at the same match
 * distance as the previous block (the LZ match just continues), ~3 if it equals any field in the
 * window (new match), else its size in literals. */
#define ZAP_TEX_WINDOW 16

typedef struct { int dist[2]; } zap__rdo_state; /* per sub-block kind: current match distance for each field */

static inline int zap__field_eq(const uint8_t *a, const uint8_t *b, int n) { return memcmp(a, b, (size_t)n) == 0; }

/* cost of placing field bytes f (size n) at block `cur`, with fields of block j at base + j*stride + off */
static inline float zap__field_rate(const uint8_t *f, int n, const uint8_t *base, size_t cur, size_t stride, size_t off,
                                    int dist, int *newdist) {
    *newdist = dist;
    if (dist > 0 && (size_t)dist <= cur && zap__field_eq(f, base + (cur - dist) * stride + off, n)) return 0.0f;
    for (size_t j = 1; j <= ZAP_TEX_WINDOW && j <= cur; j++)
        if (zap__field_eq(f, base + (cur - j) * stride + off, n)) { *newdist = (int)j; return n < 3 ? (float)n : 3.0f; }
    return (float)n;
}

/* encode one BC1 sub-block at out (= base + cur*stride + off) with RDO against previous blocks */
static inline void zap__bc1_rdo(const uint8_t px[16][4], int force4, float lambda, uint8_t *base, size_t cur, size_t stride,
                                size_t off, zap__rdo_state *rs) {
    uint16_t c0, c1; uint32_t idx;
    int e0 = zap__bc1_encode(px, &c0, &c1, &idx);
    uint8_t best[8], cand[8];
    zap__put_bc1(best, c0, c1, idx);
    if (lambda <= 0) { memcpy(base + cur * stride + off, best, 8); return; }
    int d0, d1, bd0 = 0, bd1 = 0;
    float bj = (float)e0 + lambda * (zap__field_rate(best, 4, base, cur, stride, off, rs->dist[0], &bd0) +
                                     zap__field_rate(best + 4, 4, base, cur, stride, off + 4, rs->dist[1], &bd1));
    for (size_t j = 1; j <= ZAP_TEX_WINDOW && j <= cur; j++) {
        const uint8_t *pb = base + (cur - j) * stride + off;
        uint16_t p0 = (uint16_t)zap__ld(pb, 2), p1 = (uint16_t)zap__ld(pb + 2, 2);
        uint32_t pi = (uint32_t)zap__ld(pb + 4, 4), fi;
        for (int kind = 0; kind < 3; kind++) {
            int e;
            if (kind == 0) { e = zap__bc1_fit(px, p0, p1, force4, &fi); zap__put_bc1(cand, p0, p1, fi); } /* reuse endpoints */
            else if (kind == 1) { e = zap__bc1_eval(px, c0, c1, force4, pi); zap__put_bc1(cand, c0, c1, pi); } /* reuse indices */
            else { e = zap__bc1_eval(px, p0, p1, force4, pi); memcpy(cand, pb, 8); } /* reuse block */
            if (!force4 && zap__ld(cand, 2) <= zap__ld(cand + 2, 2)) { /* 3-color mode: index 3 = transparent, not allowed */
                uint32_t ci = (uint32_t)zap__ld(cand + 4, 4); int bad = 0;
                for (int i = 0; i < 16; i++) bad |= ((ci >> (2 * i)) & 3) == 3;
                if (bad) continue;
            }
            float jj = (float)e;
            if (jj >= bj) continue;
            jj += lambda * (zap__field_rate(cand, 4, base, cur, stride, off, rs->dist[0], &d0) +
                            zap__field_rate(cand + 4, 4, base, cur, stride, off + 4, rs->dist[1], &d1));
            if (jj < bj) { bj = jj; memcpy(best, cand, 8); bd0 = d0; bd1 = d1; }
        }
    }
    rs->dist[0] = bd0; rs->dist[1] = bd1;
    memcpy(base + cur * stride + off, best, 8);
}

static inline void zap__bc4_rdo(const uint8_t v[16], float lambda, uint8_t *base, size_t cur, size_t stride, size_t off,
                                zap__rdo_state *rs) {
    uint8_t a0, a1; uint64_t idx;
    int e0 = zap__bc4_encode(v, &a0, &a1, &idx);
    uint8_t best[8], cand[8];
    zap__put_bc4(best, a0, a1, idx);
    if (lambda <= 0) { memcpy(base + cur * stride + off, best, 8); return; }
    int d0, d1, bd0 = 0, bd1 = 0;
    float bj = (float)e0 + lambda * (zap__field_rate(best, 2, base, cur, stride, off, rs->dist[0], &bd0) +
                                     zap__field_rate(best + 2, 6, base, cur, stride, off + 2, rs->dist[1], &bd1));
    for (size_t j = 1; j <= ZAP_TEX_WINDOW && j <= cur; j++) {
        const uint8_t *pb = base + (cur - j) * stride + off;
        uint64_t pi = zap__ld(pb + 2, 6), fi;
        for (int kind = 0; kind < 3; kind++) {
            int e;
            if (kind == 0) { e = zap__bc4_fit(v, pb[0], pb[1], &fi); zap__put_bc4(cand, pb[0], pb[1], fi); }
            else if (kind == 1) { e = zap__bc4_eval(v, a0, a1, pi); zap__put_bc4(cand, a0, a1, pi); }
            else { e = zap__bc4_eval(v, pb[0], pb[1], pi); memcpy(cand, pb, 8); }
            float jj = (float)e;
            if (jj >= bj) continue;
            jj += lambda * (zap__field_rate(cand, 2, base, cur, stride, off, rs->dist[0], &d0) +
                            zap__field_rate(cand + 2, 6, base, cur, stride, off + 2, rs->dist[1], &d1));
            if (jj < bj) { bj = jj; memcpy(best, cand, 8); bd0 = d0; bd1 = d1; }
        }
    }
    rs->dist[0] = bd0; rs->dist[1] = bd1;
    memcpy(base + cur * stride + off, best, 8);
}

/* ---------------- BC7 */

typedef struct { uint8_t ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2; } zap__bc7mode;
static const zap__bc7mode ZAP__BC7M[8] = {
    { 3, 4, 0, 0, 4, 0, 1, 0, 3, 0 }, { 2, 6, 0, 0, 6, 0, 0, 1, 3, 0 }, { 3, 6, 0, 0, 5, 0, 0, 0, 2, 0 },
    { 2, 6, 0, 0, 7, 0, 1, 0, 2, 0 }, { 1, 0, 2, 1, 5, 6, 0, 0, 2, 3 }, { 1, 0, 2, 0, 7, 8, 0, 0, 2, 2 },
    { 1, 0, 0, 0, 7, 7, 1, 0, 4, 0 }, { 2, 6, 0, 0, 5, 5, 1, 0, 2, 0 } };

/* 2-subset partitions: bit i set = pixel i in subset 1 */
static const uint16_t ZAP__BC7P2[64] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
    0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
    0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
    0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C, 0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22 };

/* 3-subset partitions, 2 bits per pixel (pixel i at bits 2i) */
static const uint32_t ZAP__BC7P3[64] = {
#define ZAP__P3(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,q) ((uint32_t)(a)|(uint32_t)(b)<<2|(uint32_t)(c)<<4|(uint32_t)(d)<<6|(uint32_t)(e)<<8|(uint32_t)(f)<<10|(uint32_t)(g)<<12|(uint32_t)(h)<<14|(uint32_t)(i)<<16|(uint32_t)(j)<<18|(uint32_t)(k)<<20|(uint32_t)(l)<<22|(uint32_t)(m)<<24|(uint32_t)(n)<<26|(uint32_t)(o)<<28|(uint32_t)(q)<<30)
    ZAP__P3(0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2), ZAP__P3(0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1), ZAP__P3(0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1), ZAP__P3(0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1),
    ZAP__P3(0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2), ZAP__P3(0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2), ZAP__P3(0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1), ZAP__P3(0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1),
    ZAP__P3(0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2), ZAP__P3(0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2), ZAP__P3(0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2), ZAP__P3(0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2),
    ZAP__P3(0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2), ZAP__P3(0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2), ZAP__P3(0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2), ZAP__P3(0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0),
    ZAP__P3(0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2), ZAP__P3(0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0), ZAP__P3(0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2), ZAP__P3(0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1),
    ZAP__P3(0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2), ZAP__P3(0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1), ZAP__P3(0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2), ZAP__P3(0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0),
    ZAP__P3(0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0), ZAP__P3(0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2), ZAP__P3(0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0), ZAP__P3(0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1),
    ZAP__P3(0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2), ZAP__P3(0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2), ZAP__P3(0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1), ZAP__P3(0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1),
    ZAP__P3(0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2), ZAP__P3(0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1), ZAP__P3(0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2), ZAP__P3(0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0),
    ZAP__P3(0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0), ZAP__P3(0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0), ZAP__P3(0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0), ZAP__P3(0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1),
    ZAP__P3(0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1), ZAP__P3(0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2), ZAP__P3(0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1), ZAP__P3(0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2),
    ZAP__P3(0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1), ZAP__P3(0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1), ZAP__P3(0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1), ZAP__P3(0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1),
    ZAP__P3(0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2), ZAP__P3(0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1), ZAP__P3(0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2), ZAP__P3(0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2),
    ZAP__P3(0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2), ZAP__P3(0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2), ZAP__P3(0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2), ZAP__P3(0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2),
    ZAP__P3(0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2), ZAP__P3(0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2), ZAP__P3(0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2), ZAP__P3(0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2),
    ZAP__P3(0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1), ZAP__P3(0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2), ZAP__P3(0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2), ZAP__P3(0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0)
#undef ZAP__P3
};

/* anchor (fix-up) pixel of the second subset (2-subset), and of subsets 1 and 2 (3-subset) */
static const uint8_t ZAP__BC7A2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15 };
static const uint8_t ZAP__BC7A3a[64] = {
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3 };
static const uint8_t ZAP__BC7A3b[64] = {
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8 };

static const uint8_t ZAP__BC7W2[4] = { 0, 21, 43, 64 }, ZAP__BC7W3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 },
                     ZAP__BC7W4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
static inline int zap__bc7w(int bits, int i) { return bits == 2 ? ZAP__BC7W2[i] : bits == 3 ? ZAP__BC7W3[i] : ZAP__BC7W4[i]; }

static inline int zap__bc7_subset(int ns, int part, int i) {
    return ns == 1 ? 0 : ns == 2 ? (ZAP__BC7P2[part] >> i) & 1 : (int)(ZAP__BC7P3[part] >> (2 * i)) & 3;
}
static inline int zap__bc7_anchor(int ns, int part, int sub) {
    return sub == 0 ? 0 : ns == 2 ? ZAP__BC7A2[part] : sub == 1 ? ZAP__BC7A3a[part] : ZAP__BC7A3b[part];
}

static inline unsigned zap__bits(const uint8_t *b, int *pos, int n) {
    unsigned v = 0;
    for (int i = 0; i < n; i++, (*pos)++) v |= (unsigned)(b[*pos >> 3] >> (*pos & 7) & 1) << i;
    return v;
}
static inline void zap__putbits(uint8_t *b, int *pos, unsigned v, int n) {
    for (int i = 0; i < n; i++, (*pos)++) if (v >> i & 1) b[*pos >> 3] |= (uint8_t)(1 << (*pos & 7));
}
static inline int zap__bc7x(int v, int n) { v <<= 8 - n; return v | v >> n; } /* n-bit -> 8-bit */

/* full decoder: all 8 modes (reserved mode -> transparent black, like the GPU) */
static inline void zap__bc7_dec(const uint8_t *b, uint8_t px[16][4]) {
    int mode = 0;
    while (mode < 8 && !(b[0] >> mode & 1)) mode++;
    if (mode == 8) { memset(px, 0, 64); return; }
    const zap__bc7mode *m = &ZAP__BC7M[mode];
    int pos = mode + 1, part = (int)zap__bits(b, &pos, m->pb), rot = (int)zap__bits(b, &pos, m->rb);
    int isb = (int)zap__bits(b, &pos, m->isb), ne = 2 * m->ns, ep[6][4], cb = m->cb, ab = m->ab;
    for (int c = 0; c < 3; c++) for (int e = 0; e < ne; e++) ep[e][c] = (int)zap__bits(b, &pos, m->cb);
    for (int e = 0; e < ne; e++) ep[e][3] = ab ? (int)zap__bits(b, &pos, ab) : 255;
    if (m->epb || m->spb) {
        for (int e = 0; e < ne; e++) { /* one p-bit per endpoint (epb), or per subset shared by both endpoints (spb) */
            if (m->epb || !(e & 1)) {
                int pb = (int)zap__bits(b, &pos, 1);
                for (int k = e; k <= (m->epb ? e : e + 1); k++) {
                    for (int c = 0; c < 3; c++) ep[k][c] = ep[k][c] << 1 | pb;
                    if (ab) ep[k][3] = ep[k][3] << 1 | pb;
                }
            }
        }
        cb++; if (ab) ab++;
    }
    for (int e = 0; e < ne; e++) {
        for (int c = 0; c < 3; c++) ep[e][c] = zap__bc7x(ep[e][c], cb);
        if (ab) ep[e][3] = zap__bc7x(ep[e][3], ab);
    }
    int i1[16], i2[16];
    for (int i = 0; i < 16; i++) {
        int anchor = i == zap__bc7_anchor(m->ns, part, zap__bc7_subset(m->ns, part, i));
        i1[i] = (int)zap__bits(b, &pos, m->ib - anchor);
    }
    for (int i = 0; i < 16 && m->ib2; i++) i2[i] = (int)zap__bits(b, &pos, m->ib2 - (i == 0));
    for (int i = 0; i < 16; i++) {
        int sb = zap__bc7_subset(m->ns, part, i), *e0 = ep[2 * sb], *e1 = ep[2 * sb + 1];
        int cw = m->ib2 && isb ? zap__bc7w(m->ib2, i2[i]) : zap__bc7w(m->ib, i1[i]);
        int aw = !m->ib2 ? cw : isb ? zap__bc7w(m->ib, i1[i]) : zap__bc7w(m->ib2, i2[i]);
        for (int c = 0; c < 3; c++) px[i][c] = (uint8_t)(((64 - cw) * e0[c] + cw * e1[c] + 32) >> 6);
        px[i][3] = (uint8_t)(((64 - aw) * e0[3] + aw * e1[3] + 32) >> 6);
        if (rot) { uint8_t t = px[i][3]; px[i][3] = px[i][rot - 1]; px[i][rot - 1] = t; }
    }
}

static inline int zap__sq4(const uint8_t *a, const int *b, int nc) {
    int e = 0;
    for (int c = 0; c < nc; c++) { int d = a[c] - b[c]; e += d * d; }
    return e;
}

/* quantize an endpoint to n bits total where the low bit is the p-bit p; returns 8-bit expanded values */
static inline int zap__bc7q(const float *v, int nc, int n, int p, int *q, int *out) {
    int err = 0, mx = (1 << (n - 1)) - 1;
    for (int c = 0; c < nc; c++) {
        int best = 0, be = 1 << 30, g = (int)((v[c] / 255.0f * (float)((1 << n) - 1) - (float)p) / 2.0f + 0.5f);
        for (int t = g - 1; t <= g + 1; t++) {
            int tq = t < 0 ? 0 : t > mx ? mx : t, x = zap__bc7x(tq << 1 | p, n);
            float d = (float)x - v[c];
            if ((int)(d * d) < be) { be = (int)(d * d); best = tq; }
        }
        q[c] = best; out[c] = zap__bc7x(best << 1 | p, n); err += be;
    }
    return err;
}

/* principal-axis endpoints of a pixel subset (nc = 3 or 4 channels) */
static inline void zap__bc7_pca(const uint8_t px[16][4], const int *sel, int cnt, int nc, float lo[4], float hi[4]) {
    float mean[4] = { 0, 0, 0, 0 }, cov[4][4] = { { 0 } }, ax[4] = { 1, 1, 1, 1 };
    for (int k = 0; k < cnt; k++) for (int c = 0; c < nc; c++) mean[c] += px[sel[k]][c] / (float)cnt;
    for (int k = 0; k < cnt; k++)
        for (int a = 0; a < nc; a++) for (int b2 = 0; b2 < nc; b2++) cov[a][b2] += (px[sel[k]][a] - mean[a]) * (px[sel[k]][b2] - mean[b2]);
    for (int it = 0; it < 6; it++) {
        float nv[4] = { 0, 0, 0, 0 }, m = 0;
        for (int a = 0; a < nc; a++) { for (int b2 = 0; b2 < nc; b2++) nv[a] += cov[a][b2] * ax[b2]; if ((nv[a] < 0 ? -nv[a] : nv[a]) > m) m = nv[a] < 0 ? -nv[a] : nv[a]; }
        if (m < 1e-6f) break;
        for (int a = 0; a < nc; a++) ax[a] = nv[a] / m;
    }
    float n2 = 0, tl = 1e9f, th = -1e9f;
    for (int a = 0; a < nc; a++) n2 += ax[a] * ax[a];
    for (int k = 0; k < cnt; k++) {
        float t = 0;
        for (int c = 0; c < nc; c++) t += (px[sel[k]][c] - mean[c]) * ax[c];
        if (t < tl) tl = t;
        if (t > th) th = t;
    }
    for (int c = 0; c < nc; c++) { lo[c] = mean[c] + ax[c] * tl / n2; hi[c] = mean[c] + ax[c] * th / n2; }
    for (int c = nc; c < 4; c++) { lo[c] = hi[c] = 255; }
}

/* one subset: endpoints (float) -> quantized endpoints + best indices. shared_p: one p-bit for both endpoints.
   Returns SSE; writes q (quantized, 2 x nc), p-bits and indices for the selected pixels. */
static inline int zap__bc7_fit(const uint8_t px[16][4], const int *sel, int cnt, int nc, int n, int ib, int shared_p,
                               const float lo[4], const float hi[4], int q[2][4], int pb[2], int *idx) {
    int e[2][4], best = 1 << 30;
    int bq[2][4], bp[2] = { 0, 0 };
    for (int p0 = 0; p0 < 2; p0++) for (int p1 = 0; p1 < 2; p1++) {
        if (shared_p && p0 != p1) continue;
        int tq[2][4];
        int err = zap__bc7q(lo, nc, n, p0, tq[0], e[0]) + zap__bc7q(hi, nc, n, p1, tq[1], e[1]);
        if (err < best) { best = err; memcpy(bq, tq, sizeof bq); bp[0] = p0; bp[1] = p1; }
    }
    memcpy(q, bq, sizeof bq); pb[0] = bp[0]; pb[1] = bp[1];
    for (int k = 0; k < 2; k++) for (int c = 0; c < nc; c++) e[k][c] = zap__bc7x(q[k][c] << 1 | pb[k], n);
    int pal[16][4], np = 1 << ib, err = 0;
    for (int j = 0; j < np; j++) for (int c = 0; c < nc; c++) { int w = zap__bc7w(ib, j); pal[j][c] = ((64 - w) * e[0][c] + w * e[1][c] + 32) >> 6; }
    for (int k = 0; k < cnt; k++) {
        int bi = 0, be = 1 << 30;
        for (int j = 0; j < np; j++) { int d = zap__sq4(px[sel[k]], pal[j], nc); if (d < be) { be = d; bi = j; } }
        idx[k] = bi; err += be;
    }
    return err;
}

/* PCA endpoints, fit, then least-squares refinement on the chosen indices */
static inline int zap__bc7_subset_enc(const uint8_t px[16][4], const int *sel, int cnt, int nc, int n, int ib, int shared_p,
                                      int q[2][4], int pb[2], int *idx) {
    float lo[4], hi[4];
    zap__bc7_pca(px, sel, cnt, nc, lo, hi);
    int err = zap__bc7_fit(px, sel, cnt, nc, n, ib, shared_p, lo, hi, q, pb, idx);
    for (int it = 0; it < 2 && err > 0; it++) {
        float aa = 0, ab = 0, bb = 0, ax[4] = { 0, 0, 0, 0 }, bx[4] = { 0, 0, 0, 0 };
        for (int k = 0; k < cnt; k++) {
            float w = zap__bc7w(ib, idx[k]) / 64.0f, v = 1 - w;
            aa += v * v; ab += v * w; bb += w * w;
            for (int c = 0; c < nc; c++) { ax[c] += v * px[sel[k]][c]; bx[c] += w * px[sel[k]][c]; }
        }
        float det = aa * bb - ab * ab;
        if (det < 1e-6f && det > -1e-6f) break;
        float nlo[4], nhi[4];
        for (int c = 0; c < nc; c++) {
            nlo[c] = (ax[c] * bb - bx[c] * ab) / det; nhi[c] = (bx[c] * aa - ax[c] * ab) / det;
            nlo[c] = nlo[c] < 0 ? 0 : nlo[c] > 255 ? 255 : nlo[c]; nhi[c] = nhi[c] < 0 ? 0 : nhi[c] > 255 ? 255 : nhi[c];
        }
        for (int c = nc; c < 4; c++) nlo[c] = nhi[c] = 255;
        int nq[2][4], np[2], ni[16];
        int ne = zap__bc7_fit(px, sel, cnt, nc, n, ib, shared_p, nlo, nhi, nq, np, ni);
        if (ne >= err) break;
        err = ne; memcpy(q, nq, sizeof nq); pb[0] = np[0]; pb[1] = np[1]; memcpy(idx, ni, sizeof(int) * (size_t)cnt);
    }
    return err;
}

/* mode 6: one subset, RGBA 7 bits + unique p-bit, 4-bit indices.
   Layout: bits 0-6 mode, 7-62 endpoints, 63-64 p-bits, 65-127 indices -> bytes 0-7 are pure mode+endpoints,
   bytes 9-15 pure indices, which is what the RDO below exploits. */
static inline void zap__bc7_pack6(int q[2][4], const int pb[2], const int *idx, uint8_t out[16]) {
    int pos = 0;
    memset(out, 0, 16);
    zap__putbits(out, &pos, 1 << 6, 7);
    for (int c = 0; c < 4; c++) for (int e = 0; e < 2; e++) zap__putbits(out, &pos, (unsigned)q[e][c], 7);
    zap__putbits(out, &pos, (unsigned)pb[0], 1); zap__putbits(out, &pos, (unsigned)pb[1], 1);
    for (int i = 0; i < 16; i++) zap__putbits(out, &pos, (unsigned)idx[i], i ? 4 : 3);
}

static inline int zap__bc7_is6(const uint8_t *b) { return (b[0] & 0x7F) == 0x40; }

static inline void zap__bc7_unpack6(const uint8_t *b, int q[2][4], int pb[2], int idx[16]) {
    int pos = 7;
    for (int c = 0; c < 4; c++) for (int e = 0; e < 2; e++) q[e][c] = (int)zap__bits(b, &pos, 7);
    pb[0] = (int)zap__bits(b, &pos, 1); pb[1] = (int)zap__bits(b, &pos, 1);
    for (int i = 0; i < 16; i++) idx[i] = (int)zap__bits(b, &pos, i ? 4 : 3);
}

static inline void zap__bc7_pal6(int q[2][4], const int pb[2], int pal[16][4]) {
    int e[2][4];
    for (int k = 0; k < 2; k++) for (int c = 0; c < 4; c++) e[k][c] = zap__bc7x(q[k][c] << 1 | pb[k], 8);
    for (int j = 0; j < 16; j++) for (int c = 0; c < 4; c++) pal[j][c] = ((64 - ZAP__BC7W4[j]) * e[0][c] + ZAP__BC7W4[j] * e[1][c] + 32) >> 6;
}

/* best indices for fixed endpoints; pixel 0 limited to 0..7 so the block stays valid without swapping */
static inline int zap__bc7_fit6_idx(const uint8_t px[16][4], int q[2][4], const int pb[2], int idx[16]) {
    int pal[16][4], err = 0;
    zap__bc7_pal6(q, pb, pal);
    for (int i = 0; i < 16; i++) {
        int bi = 0, be = 1 << 30;
        for (int j = 0; j < (i ? 16 : 8); j++) { int d = zap__sq4(px[i], pal[j], 4); if (d < be) { be = d; bi = j; } }
        idx[i] = bi; err += be;
    }
    return err;
}

/* best endpoints for fixed indices (least squares, then quantize); returns SSE with those indices */
static inline int zap__bc7_fit6_ep(const uint8_t px[16][4], const int idx[16], int q[2][4], int pb[2]) {
    float aa = 0, ab = 0, bb = 0, ax[4] = { 0, 0, 0, 0 }, bx[4] = { 0, 0, 0, 0 }, lo[4], hi[4];
    for (int i = 0; i < 16; i++) {
        float w = ZAP__BC7W4[idx[i]] / 64.0f, v = 1 - w;
        aa += v * v; ab += v * w; bb += w * w;
        for (int c = 0; c < 4; c++) { ax[c] += v * px[i][c]; bx[c] += w * px[i][c]; }
    }
    float det = aa * bb - ab * ab;
    for (int c = 0; c < 4; c++) {
        if (det < 1e-6f && det > -1e-6f) lo[c] = hi[c] = (ax[c] + bx[c]) / 16.0f; /* one index everywhere: flat */
        else { lo[c] = (ax[c] * bb - bx[c] * ab) / det; hi[c] = (bx[c] * aa - ax[c] * ab) / det; }
        lo[c] = lo[c] < 0 ? 0 : lo[c] > 255 ? 255 : lo[c]; hi[c] = hi[c] < 0 ? 0 : hi[c] > 255 ? 255 : hi[c];
    }
    int e0[4], e1[4], t0[4], t1[4], b0 = 1 << 30, b1 = 1 << 30;
    for (int p = 0; p < 2; p++) { /* unique p-bits: pick each endpoint's independently */
        int x = zap__bc7q(lo, 4, 8, p, t0, e0), y = zap__bc7q(hi, 4, 8, p, t1, e1);
        if (x < b0) { b0 = x; memcpy(q[0], t0, sizeof t0); pb[0] = p; }
        if (y < b1) { b1 = y; memcpy(q[1], t1, sizeof t1); pb[1] = p; }
    }
    int pal[16][4], err = 0;
    zap__bc7_pal6(q, pb, pal);
    for (int i = 0; i < 16; i++) err += zap__sq4(px[i], pal[idx[i]], 4);
    return err;
}

static inline int zap__bc7_m6(const uint8_t px[16][4], uint8_t out[16]) {
    int sel[16], q[2][4], pb[2], idx[16];
    for (int i = 0; i < 16; i++) sel[i] = i;
    int err = zap__bc7_subset_enc(px, sel, 16, 4, 8, 4, 0, q, pb, idx);
    if (idx[0] & 8) { /* anchor index must have MSB 0: swap endpoints, invert indices */
        for (int c = 0; c < 4; c++) { int t = q[0][c]; q[0][c] = q[1][c]; q[1][c] = t; }
        int t = pb[0]; pb[0] = pb[1]; pb[1] = t;
        for (int i = 0; i < 16; i++) idx[i] = 15 - idx[i];
    }
    zap__bc7_pack6(q, pb, idx, out);
    return err;
}

/* mode 1: two subsets, RGB 6 bits + shared p-bit per subset, 3-bit indices. Opaque blocks only. */
static inline int zap__bc7_m1(const uint8_t px[16][4], int part, uint8_t out[16]) {
    int sel[2][16], cnt[2] = { 0, 0 }, q[2][2][4], pb[2][2], idx[2][16], err = 0, full[16];
    for (int i = 0; i < 16; i++) { int s2 = (ZAP__BC7P2[part] >> i) & 1; sel[s2][cnt[s2]++] = i; }
    for (int s2 = 0; s2 < 2; s2++) err += zap__bc7_subset_enc(px, sel[s2], cnt[s2], 3, 7, 3, 1, q[s2], pb[s2], idx[s2]);
    if (!out) return err;
    for (int s2 = 0; s2 < 2; s2++) {
        for (int k = 0; k < cnt[s2]; k++) full[sel[s2][k]] = idx[s2][k];
        int anchor = s2 ? ZAP__BC7A2[part] : 0;
        if (full[anchor] & 4) {
            for (int c = 0; c < 3; c++) { int t = q[s2][0][c]; q[s2][0][c] = q[s2][1][c]; q[s2][1][c] = t; }
            for (int k = 0; k < cnt[s2]; k++) full[sel[s2][k]] = 7 - full[sel[s2][k]];
        }
    }
    int pos = 0;
    memset(out, 0, 16);
    zap__putbits(out, &pos, 2, 2);
    zap__putbits(out, &pos, (unsigned)part, 6);
    for (int c = 0; c < 3; c++) for (int s2 = 0; s2 < 2; s2++) for (int e = 0; e < 2; e++) zap__putbits(out, &pos, (unsigned)q[s2][e][c], 6);
    zap__putbits(out, &pos, (unsigned)pb[0][0], 1); zap__putbits(out, &pos, (unsigned)pb[1][0], 1);
    for (int i = 0; i < 16; i++) zap__putbits(out, &pos, (unsigned)full[i], 3 - (i == 0 || i == ZAP__BC7A2[part]));
    return err;
}

static inline int zap__bc7_err(const uint8_t px[16][4], const uint8_t *blk) {
    uint8_t d[16][4];
    int err = 0;
    zap__bc7_dec(blk, d);
    for (int i = 0; i < 16; i++) for (int c = 0; c < 4; c++) { int x = px[i][c] - d[i][c]; err += x * x; }
    return err;
}

/* best of mode 6 and (for opaque blocks) mode 1 over all 64 partitions */
static inline int zap__bc7_encode(const uint8_t px[16][4], uint8_t out[16]) {
    int err = zap__bc7_m6(px, out), opaque = 1;
    for (int i = 0; i < 16; i++) opaque &= px[i][3] == 255;
    if (!opaque || err <= 16 * 4) return err; /* smooth block: mode 6 is already near-exact */
    int bp = -1, be = err;
    for (int part = 0; part < 64; part++) { int e = zap__bc7_m1(px, part, NULL); if (e < be) { be = e; bp = part; } }
    if (bp >= 0) {
        uint8_t m1[16];
        zap__bc7_m1(px, bp, m1);
        int e1 = zap__bc7_err(px, m1);
        if (e1 < err) { memcpy(out, m1, 16); err = e1; }
    }
    return err;
}

/* BC7 RDO. Candidates from recent blocks: the whole block, or (mode 6) its endpoint bytes 0-7 with refit
   indices, or its index bytes 9-15 with re-solved endpoints. Byte 8 straddles both and is always paid for. */
static inline float zap__bc7_rate(const uint8_t *c, const uint8_t *base, size_t cur, const zap__rdo_state *rs, int d[2]) {
    return zap__field_rate(c, 8, base, cur, 16, 0, rs->dist[0], &d[0]) + zap__field_rate(c + 9, 7, base, cur, 16, 9, rs->dist[1], &d[1]) + 1;
}

static inline void zap__bc7_rdo(const uint8_t px[16][4], float lambda, uint8_t *base, size_t cur, zap__rdo_state *rs) {
    uint8_t best[16], cand[16];
    int e0 = zap__bc7_encode(px, best), bd[2] = { 0, 0 }, d[2];
    if (lambda > 0) {
        float bj = (float)e0 + lambda * zap__bc7_rate(best, base, cur, rs, bd);
        for (size_t j = 1; j <= ZAP_TEX_WINDOW && j <= cur; j++) {
            const uint8_t *pb = base + (cur - j) * 16;
            int pq[2][4], ppb[2], pidx[16], nq[2][4], npb[2], nidx[16];
            int six = zap__bc7_is6(pb);
            if (six) zap__bc7_unpack6(pb, pq, ppb, pidx);
            for (int kind = 0; kind < (six ? 3 : 1); kind++) {
                int e;
                if (kind == 0) { e = zap__bc7_err(px, pb); memcpy(cand, pb, 16); }                       /* whole block */
                else if (kind == 1) { e = zap__bc7_fit6_idx(px, pq, ppb, nidx); zap__bc7_pack6(pq, ppb, nidx, cand); } /* its endpoints */
                else { e = zap__bc7_fit6_ep(px, pidx, nq, npb); zap__bc7_pack6(nq, npb, pidx, cand); }    /* its indices */
                if ((float)e >= bj) continue;
                float jj = (float)e + lambda * zap__bc7_rate(cand, base, cur, rs, d);
                if (jj < bj) { bj = jj; memcpy(best, cand, 16); bd[0] = d[0]; bd[1] = d[1]; }
            }
        }
        rs->dist[0] = bd[0]; rs->dist[1] = bd[1];
    }
    memcpy(base + cur * 16, best, 16);
}

/* ---------------- public API */

/* rdo: 0 = off. lambda in squared-error units per byte saved; 5..200 is the useful range. */
static inline void zap_bc_encode(const uint8_t *rgba, int w, int h, size_t stride, zap_bc_format f, float rdo, void *out_) {
    uint8_t *out = (uint8_t *)out_;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    size_t bs = zap_bc_block_bytes(f);
    zap__rdo_state rs[2] = { { { 1, 1 } }, { { 1, 1 } } };
    /* ponytail: raster block order; a Morton/tiled order would give RDO more nearby candidates */
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            uint8_t px[16][4], ch[2][16];
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x >= w) x = w - 1;
                if (y >= h) y = h - 1;
                memcpy(px[i], rgba + (size_t)y * stride + (size_t)x * 4, 4);
            }
            size_t cur = (size_t)by * (size_t)bw + (size_t)bx;
            if (f == ZAP_BC1) zap__bc1_rdo((const uint8_t(*)[4])px, 0, rdo, out, cur, bs, 0, &rs[0]);
            else if (f == ZAP_BC4) {
                for (int i = 0; i < 16; i++) ch[0][i] = px[i][0];
                zap__bc4_rdo(ch[0], rdo, out, cur, bs, 0, &rs[0]);
            } else if (f == ZAP_BC3) {
                for (int i = 0; i < 16; i++) ch[0][i] = px[i][3];
                zap__bc4_rdo(ch[0], rdo, out, cur, bs, 0, &rs[0]);
                zap__bc1_rdo((const uint8_t(*)[4])px, 1, rdo, out, cur, bs, 8, &rs[1]);
            } else if (f == ZAP_BC5) {
                for (int i = 0; i < 16; i++) { ch[0][i] = px[i][0]; ch[1][i] = px[i][1]; }
                zap__bc4_rdo(ch[0], rdo, out, cur, bs, 0, &rs[0]);
                zap__bc4_rdo(ch[1], rdo, out, cur, bs, 8, &rs[1]);
            } else {
                zap__bc7_rdo((const uint8_t(*)[4])px, rdo, out, cur, &rs[0]);
            }
        }
}

static inline void zap__bc1_dec(const uint8_t *b, int force4, uint8_t px[16][4]) {
    int pal[4][3];
    int n = zap__bc1_pal((uint16_t)zap__ld(b, 2), (uint16_t)zap__ld(b + 2, 2), force4, pal);
    uint32_t idx = (uint32_t)zap__ld(b + 4, 4);
    for (int i = 0; i < 16; i++) {
        int k = (idx >> (2 * i)) & 3;
        px[i][0] = (uint8_t)pal[k][0]; px[i][1] = (uint8_t)pal[k][1]; px[i][2] = (uint8_t)pal[k][2];
        px[i][3] = (uint8_t)(n == 3 && k == 3 ? 0 : 255);
    }
}

static inline void zap__bc4_dec(const uint8_t *b, uint8_t v[16]) {
    int pal[8];
    zap__bc4_pal(b[0], b[1], pal);
    uint64_t idx = zap__ld(b + 2, 6);
    for (int i = 0; i < 16; i++) v[i] = (uint8_t)pal[(idx >> (3 * i)) & 7];
}

static inline void zap_bc_decode(const void *bc_, int w, int h, zap_bc_format f, uint8_t *rgba, size_t stride) {
    const uint8_t *bc = (const uint8_t *)bc_;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    size_t bs = zap_bc_block_bytes(f);
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            const uint8_t *b = bc + ((size_t)by * (size_t)bw + (size_t)bx) * bs;
            uint8_t px[16][4], v0[16], v1[16];
            if (f == ZAP_BC1) zap__bc1_dec(b, 0, px);
            else if (f == ZAP_BC7) zap__bc7_dec(b, px);
            else if (f == ZAP_BC3) { zap__bc1_dec(b + 8, 1, px); zap__bc4_dec(b, v0); for (int i = 0; i < 16; i++) px[i][3] = v0[i]; }
            else {
                zap__bc4_dec(b, v0);
                if (f == ZAP_BC5) zap__bc4_dec(b + 8, v1); else memset(v1, 0, 16);
                for (int i = 0; i < 16; i++) { px[i][0] = v0[i]; px[i][1] = v1[i]; px[i][2] = 0; px[i][3] = 255; }
            }
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x < w && y < h) memcpy(rgba + (size_t)y * stride + (size_t)x * 4, px[i], 4);
            }
        }
}

/* ---------------- split / merge: plane k = field k of every block. Same size in and out. */

static inline int zap__bc_fields(zap_bc_format f, int sz[4]) {
    static const int F[5][5] = { { 2, 4, 4 }, { 4, 2, 6, 4, 4 }, { 2, 2, 6 }, { 4, 2, 6, 2, 6 }, { 3, 8, 1, 7 } };
    for (int i = 0; i < F[f][0]; i++) sz[i] = F[f][i + 1];
    return F[f][0];
}

static inline void zap__bc_shuffle(const uint8_t *in, uint8_t *out, size_t n, zap_bc_format f, int split) {
    int sz[4], nf = zap__bc_fields(f, sz);
    size_t bs = zap_bc_block_bytes(f), nb = n / bs, plane = 0, off = 0;
    for (int k = 0; k < nf; k++) {
        for (size_t b = 0; b < nb; b++) {
            const uint8_t *s = split ? in + b * bs + off : in + plane + b * (size_t)sz[k];
            uint8_t *d = split ? out + plane + b * (size_t)sz[k] : out + b * bs + off;
            memcpy(d, s, (size_t)sz[k]);
        }
        plane += nb * (size_t)sz[k]; off += (size_t)sz[k];
    }
    memcpy(out + nb * bs, in + nb * bs, n - nb * bs); /* stray tail bytes, if any */
}
static inline void zap_bc_split(const void *bc, size_t n, zap_bc_format f, void *out) { zap__bc_shuffle((const uint8_t *)bc, (uint8_t *)out, n, f, 1); }
static inline void zap_bc_merge(const void *planes, size_t n, zap_bc_format f, void *out) { zap__bc_shuffle((const uint8_t *)planes, (uint8_t *)out, n, f, 0); }

#endif
