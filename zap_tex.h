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
 * BC7: the decoder handles all 8 modes; the encoder uses mode 6 (smooth / alpha), mode 5 (independent alpha,
 * channel rotation) and mode 1 (2 partitions, opaque edges). BC7 RDO reuses whole blocks, or the endpoint /
 * index bytes of recent mode-6 blocks.
 *
 *   zap_bc6h_encode(rgba_float, w, h, stride, out) / zap_bc6h_decode   HDR (BC6H unsigned, mode 11)
 *   zap_astc_encode(rgba, w, h, stride, out) / zap_astc_decode           ASTC 4x4 LDR (mobile), 16 bytes per block
 *   zap_mip_levels(w, h), zap_mip_next(src, w, h, stride, srgb, dst, dst_stride), zap_mip_next_f(...)   mipmaps
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

/* mode 5: one subset, channel rotation, RGB 7 bits + A 8 bits (no p-bits), separate 2-bit color and alpha
   indices. Best for blocks whose alpha (or one color channel, via rotation) varies independently. */
static inline int zap__bc7_fit2(const uint8_t r[16][4], int c0, int nc, const int *e0, const int *e1, int idx[16]) {
    int pal[4][4], err = 0;
    for (int j = 0; j < 4; j++) for (int c = 0; c < nc; c++) pal[j][c] = ((64 - ZAP__BC7W2[j]) * e0[c] + ZAP__BC7W2[j] * e1[c] + 32) >> 6;
    for (int i = 0; i < 16; i++) {
        int bi = 0, be = 1 << 30;
        for (int j = 0; j < 4; j++) {
            int d = 0;
            for (int c = 0; c < nc; c++) { int x = r[i][c0 + c] - pal[j][c]; d += x * x; }
            if (d < be) { be = d; bi = j; }
        }
        idx[i] = bi; err += be;
    }
    return err;
}

/* quantize endpoint channels to n bits (no p-bit): q = stored value, e = expanded 8-bit value */
static inline void zap__bc7_qn(const float *v, int nc, int n, int *q, int *e) {
    int mx = (1 << n) - 1;
    for (int c = 0; c < nc; c++) {
        int g = (int)(v[c] * (float)mx / 255.0f + 0.5f), bq = 0, bd = 1 << 30;
        for (int t = g - 1; t <= g + 1; t++) {
            int tq = t < 0 ? 0 : t > mx ? mx : t, x = zap__bc7x(tq, n);
            int d = (int)((float)x - v[c] < 0 ? v[c] - (float)x : (float)x - v[c]);
            if (d < bd) { bd = d; bq = tq; }
        }
        q[c] = bq; e[c] = zap__bc7x(bq, n);
    }
}

/* least-squares endpoints for fixed 2-bit indices on channels [c0, c0 + nc) */
static inline int zap__bc7_ls2(const uint8_t r[16][4], int c0, int nc, const int idx[16], float lo[4], float hi[4]) {
    float aa = 0, ab = 0, bb = 0, ax[4] = { 0, 0, 0, 0 }, bx[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < 16; i++) {
        float w = ZAP__BC7W2[idx[i]] / 64.0f, v = 1 - w;
        aa += v * v; ab += v * w; bb += w * w;
        for (int c = 0; c < nc; c++) { ax[c] += v * r[i][c0 + c]; bx[c] += w * r[i][c0 + c]; }
    }
    float det = aa * bb - ab * ab;
    if (det < 1e-6f && det > -1e-6f) return 0;
    for (int c = 0; c < nc; c++) {
        lo[c] = (ax[c] * bb - bx[c] * ab) / det; hi[c] = (bx[c] * aa - ax[c] * ab) / det;
        lo[c] = lo[c] < 0 ? 0 : lo[c] > 255 ? 255 : lo[c]; hi[c] = hi[c] < 0 ? 0 : hi[c] > 255 ? 255 : hi[c];
    }
    return 1;
}

/* fit one part (color: 3 channels at 7 bits, or alpha: 1 channel at 8 bits) from a start, then refine */
static inline int zap__bc7_part5(const uint8_t r[16][4], int c0, int nc, int n, float lo[4], float hi[4], int q[2][3], int idx[16]) {
    int e0[3], e1[3], ti[16], tq[2][3], te0[3], te1[3];
    zap__bc7_qn(lo, nc, n, q[0], e0); zap__bc7_qn(hi, nc, n, q[1], e1);
    int err = zap__bc7_fit2(r, c0, nc, e0, e1, idx);
    for (int it = 0; it < 2 && err > 0; it++) {
        float nlo[4], nhi[4];
        if (!zap__bc7_ls2(r, c0, nc, idx, nlo, nhi)) break;
        zap__bc7_qn(nlo, nc, n, tq[0], te0); zap__bc7_qn(nhi, nc, n, tq[1], te1);
        int e = zap__bc7_fit2(r, c0, nc, te0, te1, ti);
        if (e >= err) break;
        err = e; memcpy(q, tq, sizeof tq); memcpy(idx, ti, sizeof ti);
    }
    return err;
}

static inline int zap__bc7_m5(const uint8_t px[16][4], uint8_t out[16]) {
    int best = 1 << 30, brot = 0, bcq[2][3] = { { 0 } }, baq[2][3] = { { 0 } }, bci[16] = { 0 }, bai[16] = { 0 };
    for (int rot = 0; rot < 4; rot++) {
        uint8_t r[16][4];
        int sel[16], cq[2][3], aq[2][3], ci[16], ai[16];
        memcpy(r, px, sizeof r);
        for (int i = 0; i < 16; i++) { sel[i] = i; if (rot) { uint8_t t = r[i][3]; r[i][3] = r[i][rot - 1]; r[i][rot - 1] = t; } }
        float lo[4], hi[4], alo[4] = { 255 }, ahi[4] = { 0 };
        zap__bc7_pca((const uint8_t(*)[4])r, sel, 16, 3, lo, hi);
        int err = zap__bc7_part5((const uint8_t(*)[4])r, 0, 3, 7, lo, hi, cq, ci);
        for (int i = 0; i < 16; i++) { if (r[i][3] < alo[0]) alo[0] = r[i][3]; if (r[i][3] > ahi[0]) ahi[0] = r[i][3]; }
        err += zap__bc7_part5((const uint8_t(*)[4])r, 3, 1, 8, alo, ahi, aq, ai);
        if (err < best) { best = err; brot = rot; memcpy(bcq, cq, sizeof cq); memcpy(baq, aq, sizeof aq); memcpy(bci, ci, sizeof ci); memcpy(bai, ai, sizeof ai); }
    }
    if (bci[0] & 2) { for (int c = 0; c < 3; c++) { int t = bcq[0][c]; bcq[0][c] = bcq[1][c]; bcq[1][c] = t; } for (int i = 0; i < 16; i++) bci[i] = 3 - bci[i]; }
    if (bai[0] & 2) { int t = baq[0][0]; baq[0][0] = baq[1][0]; baq[1][0] = t; for (int i = 0; i < 16; i++) bai[i] = 3 - bai[i]; }
    int pos = 0;
    memset(out, 0, 16);
    zap__putbits(out, &pos, 1 << 5, 6);
    zap__putbits(out, &pos, (unsigned)brot, 2);
    for (int c = 0; c < 3; c++) for (int e = 0; e < 2; e++) zap__putbits(out, &pos, (unsigned)bcq[e][c], 7);
    for (int e = 0; e < 2; e++) zap__putbits(out, &pos, (unsigned)baq[e][0], 8);
    for (int i = 0; i < 16; i++) zap__putbits(out, &pos, (unsigned)bci[i], i ? 2 : 1);
    for (int i = 0; i < 16; i++) zap__putbits(out, &pos, (unsigned)bai[i], i ? 2 : 1);
    return best;
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

/* best of mode 6, mode 5 and (for opaque blocks) mode 1 over all 64 partitions */
static inline int zap__bc7_encode(const uint8_t px[16][4], uint8_t out[16]) {
    int err = zap__bc7_m6(px, out), opaque = 1;
    for (int i = 0; i < 16; i++) opaque &= px[i][3] == 255;
    if (err <= 16 * 4) return err; /* smooth block: mode 6 is already near-exact */
    uint8_t m5[16];
    zap__bc7_m5(px, m5);
    int e5 = zap__bc7_err(px, m5); /* actual decoded error */
    if (e5 < err) { memcpy(out, m5, 16); err = e5; }
    if (!opaque) return err;
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

/* ---------------- BC6H: HDR, unsigned half floats. Mode 11 only: one region, 10-bit endpoints (no delta
 * transform), 4-bit indices. The decoder reads mode 11 (what the encoder writes); other modes decode to 0. */

static inline uint16_t zap__f2h(float f) { /* float -> unsigned half bits, clamped to [0, 65504] */
    uint32_t x;
    if (!(f > 0)) return 0; /* negatives and NaN */
    memcpy(&x, &f, 4);
    if (x >= 0x477FE000u) return 0x7BFF;
    if (x < 0x38800000u) return (uint16_t)(f * 16777216.0f + 0.5f); /* half subnormal: f * 2^24 */
    return (uint16_t)(((x - 0x38000000u) + 0x1000u) >> 13);        /* rebias, round mantissa to 10 bits */
}
static inline float zap__h2f(uint16_t h) {
    uint32_t e = (h >> 10) & 31, m = h & 1023u, x;
    float f;
    if (e == 0) return (float)m * (1.0f / 16777216.0f);
    x = ((e + 112u) << 23) | (m << 13);
    memcpy(&f, &x, 4);
    return f;
}

static inline int zap__bc6_unq(int e) { return e == 0 ? 0 : e == 1023 ? 0xFFFF : ((e << 16) + 0x8000) >> 10; }
static inline int zap__bc6_fin(int u) { return (u * 31) >> 6; }

/* nearest 10-bit endpoint for a pre-finish value u */
static inline int zap__bc6_q(float u) {
    int g = (int)((u - 32.0f) / 64.0f + 0.5f), best = 0;
    float bd = 1e30f;
    for (int t = g - 1; t <= g + 1; t++) {
        int e = t < 0 ? 0 : t > 1023 ? 1023 : t;
        float d = (float)zap__bc6_unq(e) - u;
        d = d < 0 ? -d : d;
        if (d < bd) { bd = d; best = e; }
    }
    return best;
}

/* squared error in half-bit (roughly log) space for endpoints e[2][3]; writes best 4-bit indices */
static inline double zap__bc6_fit(const uint16_t t[16][3], const int e[2][3], int idx[16]) {
    int pal[16][3];
    double err = 0;
    for (int j = 0; j < 16; j++)
        for (int c = 0; c < 3; c++)
            pal[j][c] = zap__bc6_fin(((64 - ZAP__BC7W4[j]) * zap__bc6_unq(e[0][c]) + ZAP__BC7W4[j] * zap__bc6_unq(e[1][c]) + 32) >> 6);
    for (int i = 0; i < 16; i++) {
        double be = 1e300;
        int bi = 0;
        for (int j = 0; j < 16; j++) {
            double d = 0;
            for (int c = 0; c < 3; c++) { double x = (double)t[i][c] - pal[j][c]; d += x * x; }
            if (d < be) { be = d; bi = j; }
        }
        idx[i] = bi; err += be;
    }
    return err;
}

static inline double zap__bc6_block(const uint16_t t[16][3], uint8_t out[16]) {
    float U[16][3], mean[3] = { 0, 0, 0 }, cov[6] = { 0, 0, 0, 0, 0, 0 }, ax[3] = { 1, 1, 1 };
    for (int i = 0; i < 16; i++) for (int c = 0; c < 3; c++) { U[i][c] = t[i][c] * 64.0f / 31.0f; mean[c] += U[i][c] / 16.0f; }
    for (int i = 0; i < 16; i++) {
        float r = U[i][0] - mean[0], g = U[i][1] - mean[1], b = U[i][2] - mean[2];
        cov[0] += r * r; cov[1] += r * g; cov[2] += r * b; cov[3] += g * g; cov[4] += g * b; cov[5] += b * b;
    }
    for (int it = 0; it < 6; it++) {
        float x = cov[0] * ax[0] + cov[1] * ax[1] + cov[2] * ax[2], y = cov[1] * ax[0] + cov[3] * ax[1] + cov[4] * ax[2];
        float z = cov[2] * ax[0] + cov[4] * ax[1] + cov[5] * ax[2], m = x < 0 ? -x : x;
        if ((y < 0 ? -y : y) > m) m = y < 0 ? -y : y;
        if ((z < 0 ? -z : z) > m) m = z < 0 ? -z : z;
        if (m < 1e-6f) break;
        ax[0] = x / m; ax[1] = y / m; ax[2] = z / m;
    }
    float n2 = ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2], tl = 1e30f, th = -1e30f, bl[3] = { 1e30f, 1e30f, 1e30f }, bh[3] = { 0, 0, 0 };
    for (int i = 0; i < 16; i++) {
        float d = (U[i][0] - mean[0]) * ax[0] + (U[i][1] - mean[1]) * ax[1] + (U[i][2] - mean[2]) * ax[2];
        if (d < tl) tl = d;
        if (d > th) th = d;
        for (int c = 0; c < 3; c++) { if (U[i][c] < bl[c]) bl[c] = U[i][c]; if (U[i][c] > bh[c]) bh[c] = U[i][c]; }
    }
    double best = 1e300;
    int be[2][3] = { { 0 } }, bidx[16] = { 0 };
    for (int start = 0; start < 2; start++) { /* principal axis, and the per-channel bounding box */
        int e[2][3], idx[16];
        for (int c = 0; c < 3; c++) {
            float lo = start ? bl[c] : mean[c] + ax[c] * tl / n2, hi = start ? bh[c] : mean[c] + ax[c] * th / n2;
            e[0][c] = zap__bc6_q(lo); e[1][c] = zap__bc6_q(hi);
        }
        double err = zap__bc6_fit(t, (const int(*)[3])e, idx);
        for (int it = 0; it < 3 && err > 0; it++) { /* least squares on the pre-finish values */
            float aa = 0, ab = 0, bb = 0, xa[3] = { 0, 0, 0 }, xb[3] = { 0, 0, 0 };
            for (int i = 0; i < 16; i++) {
                float w = ZAP__BC7W4[idx[i]] / 64.0f, v = 1 - w;
                aa += v * v; ab += v * w; bb += w * w;
                for (int c = 0; c < 3; c++) { xa[c] += v * U[i][c]; xb[c] += w * U[i][c]; }
            }
            float det = aa * bb - ab * ab;
            if (det < 1e-6f && det > -1e-6f) break;
            int ne[2][3], ni[16];
            for (int c = 0; c < 3; c++) { ne[0][c] = zap__bc6_q((xa[c] * bb - xb[c] * ab) / det); ne[1][c] = zap__bc6_q((xb[c] * aa - xa[c] * ab) / det); }
            double e2 = zap__bc6_fit(t, (const int(*)[3])ne, ni);
            if (e2 >= err) break;
            err = e2; memcpy(e, ne, sizeof ne); memcpy(idx, ni, sizeof ni);
        }
        if (err < best) { best = err; memcpy(be, e, sizeof e); memcpy(bidx, idx, sizeof idx); }
    }
    if (bidx[0] & 8) { /* anchor index needs MSB 0 */
        for (int c = 0; c < 3; c++) { int x = be[0][c]; be[0][c] = be[1][c]; be[1][c] = x; }
        for (int i = 0; i < 16; i++) bidx[i] = 15 - bidx[i];
    }
    int pos = 0;
    memset(out, 0, 16);
    zap__putbits(out, &pos, 3, 5); /* mode 11 */
    for (int k = 0; k < 2; k++) for (int c = 0; c < 3; c++) zap__putbits(out, &pos, (unsigned)be[k][c], 10);
    for (int i = 0; i < 16; i++) zap__putbits(out, &pos, (unsigned)bidx[i], i ? 4 : 3);
    return best;
}

static inline void zap__bc6_dec(const uint8_t *b, uint16_t px[16][4]) {
    int pos = 0, e[2][3];
    memset(px, 0, 16 * 4 * sizeof(uint16_t));
    if (zap__bits(b, &pos, 5) != 3) return; /* not mode 11 */
    for (int k = 0; k < 2; k++) for (int c = 0; c < 3; c++) e[k][c] = zap__bc6_unq((int)zap__bits(b, &pos, 10));
    for (int i = 0; i < 16; i++) {
        int w = ZAP__BC7W4[zap__bits(b, &pos, i ? 4 : 3)];
        for (int c = 0; c < 3; c++) px[i][c] = (uint16_t)zap__bc6_fin(((64 - w) * e[0][c] + w * e[1][c] + 32) >> 6);
        px[i][3] = 0x3C00; /* 1.0 */
    }
}

/* rgba: 4 floats per pixel, stride in floats per row; alpha ignored, negatives clamp to 0. out: zap_bc_size(w, h, ZAP_BC7) */
static inline void zap_bc6h_encode(const float *rgba, int w, int h, size_t stride, void *out_) {
    uint8_t *out = (uint8_t *)out_;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            uint16_t t[16][3];
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x >= w) x = w - 1;
                if (y >= h) y = h - 1;
                for (int c = 0; c < 3; c++) t[i][c] = zap__f2h(rgba[(size_t)y * stride + (size_t)x * 4 + c]);
            }
            zap__bc6_block((const uint16_t(*)[3])t, out + ((size_t)by * (size_t)bw + (size_t)bx) * 16);
        }
}

/* to RGBA half-float bits (alpha 1.0), stride in uint16 per row */
static inline void zap_bc6h_decode(const void *bc_, int w, int h, uint16_t *rgba, size_t stride) {
    const uint8_t *bc = (const uint8_t *)bc_;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            uint16_t px[16][4];
            zap__bc6_dec(bc + ((size_t)by * (size_t)bw + (size_t)bx) * 16, px);
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x < w && y < h) memcpy(rgba + (size_t)y * stride + (size_t)x * 4, px[i], 8);
            }
        }
}

/* ---------------- ASTC 4x4, LDR, linear (UNORM) decode. 4x4 weight grid, no dual plane, 1 or 2 partitions.
 * Each block tries these and keeps the best (the endpoint range isn't stored: decoders infer it from the bits
 * left over, so it's whatever the spec's rule picks):
 *   1 partition, opaque (CEM 8, RGB direct):   weights 16 levels / endpoints 192 | 8 / 256 | 32 / 32
 *   1 partition, alpha  (CEM 12, RGBA direct): weights 16 / endpoints 48 | 8 / 192 | 4 / 256
 *   2 partitions, opaque (CEM 8): weights 8 / endpoints 16 | 4 / 40; the partition is picked by 2-means
 *     clustering, then the best-matching of the 1024 partition patterns are fitted fully
 * zap_astc_decode reads exactly this subset plus constant-color (void-extent) blocks; anything else decodes to
 * magenta. Verified bit-exact against ARM astcenc's decoder. */

typedef struct { uint16_t levels; uint8_t trit, quint, bits; } zap__iser;
static const zap__iser ZAP__ISE[21] = {
    { 2, 0, 0, 1 }, { 3, 1, 0, 0 }, { 4, 0, 0, 2 }, { 5, 0, 1, 0 }, { 6, 1, 0, 1 }, { 8, 0, 0, 3 }, { 10, 0, 1, 1 },
    { 12, 1, 0, 2 }, { 16, 0, 0, 4 }, { 20, 0, 1, 2 }, { 24, 1, 0, 3 }, { 32, 0, 0, 5 }, { 40, 0, 1, 3 }, { 48, 1, 0, 4 },
    { 64, 0, 0, 6 }, { 80, 0, 1, 4 }, { 96, 1, 0, 5 }, { 128, 0, 0, 7 }, { 160, 0, 1, 5 }, { 192, 1, 0, 6 }, { 256, 0, 0, 8 } };

static inline int zap__ise_bits(const zap__iser *r, int n) { return n * r->bits + (r->trit ? (8 * n + 4) / 5 : r->quint ? (7 * n + 2) / 3 : 0); }

/* the spec's rule: the largest color range (>= 6 levels) whose encoding fits in the bits left */
static inline const zap__iser *zap__astc_crange(int nvals, int avail) {
    for (int i = 20; i >= 4; i--) if (zap__ise_bits(&ZAP__ISE[i], nvals) <= avail) return &ZAP__ISE[i];
    return NULL;
}

static inline int zap__rep(int v, int n, int to) { /* bit replication of an n-bit value to `to` bits */
    int x = 0;
    for (int sh = to - n; sh > -n; sh -= n) x |= sh >= 0 ? v << sh : v >> -sh;
    return x & ((1 << to) - 1);
}

/* color endpoint unquantization to 0..255 */
static inline int zap__astc_cunq(const zap__iser *r, int v) {
    if (!r->trit && !r->quint) return zap__rep(v, r->bits, 8);
    int n = r->bits, m = v & ((1 << n) - 1), D = v >> n;
    int a = m & 1, b = m >> 1 & 1, c = m >> 2 & 1, d = m >> 3 & 1, e = m >> 4 & 1, f = m >> 5 & 1, A = a ? 0x1FF : 0, B = 0, C = 0;
    switch (r->levels) {
    case 6: C = 204; break;
    case 10: C = 113; break;
    case 12: B = b << 8 | b << 4 | b << 2 | b << 1; C = 93; break;
    case 20: B = b << 8 | b << 3 | b << 2; C = 54; break;
    case 24: B = c << 8 | b << 7 | c << 3 | b << 2 | c << 1 | b; C = 44; break;
    case 40: B = c << 8 | b << 7 | c << 2 | b << 1 | c; C = 26; break;
    case 48: B = d << 8 | c << 7 | b << 6 | d << 2 | c << 1 | b; C = 22; break;
    case 80: B = d << 8 | c << 7 | b << 6 | d << 1 | c; C = 13; break;
    case 96: B = e << 8 | d << 7 | c << 6 | b << 5 | e << 1 | d; C = 11; break;
    case 160: B = e << 8 | d << 7 | c << 6 | b << 5 | e; C = 6; break;
    case 192: B = f << 8 | e << 7 | d << 6 | c << 5 | b << 4 | f; C = 5; break;
    }
    int T = (D * C + B) ^ A;
    return (A & 0x80) | (T >> 2);
}

/* weights (power-of-two ranges only): n-bit index -> 0..64 */
static inline int zap__astc_wunq(int w, int n) { int x = zap__rep(w, n, 6); return x > 32 ? x + 1 : x; }

static inline void zap__trits(int T, int t[5]) { /* 8 bits -> 5 trits (spec C.2.12) */
    int C;
    if ((T >> 2 & 7) == 7) { C = (T >> 5 & 7) << 2 | (T & 3); t[4] = 2; t[3] = 2; }
    else {
        C = T & 31;
        if ((T >> 5 & 3) == 3) { t[4] = 2; t[3] = T >> 7 & 1; } else { t[4] = T >> 7 & 1; t[3] = T >> 5 & 3; }
    }
    if ((C & 3) == 3) { t[2] = 2; t[1] = C >> 4 & 1; t[0] = (C >> 3 & 1) << 1 | ((C >> 2 & 1) & ~(C >> 3) & 1); }
    else if ((C >> 2 & 3) == 3) { t[2] = 2; t[1] = 2; t[0] = C & 3; }
    else { t[2] = C >> 4 & 1; t[1] = C >> 2 & 3; t[0] = (C >> 1 & 1) << 1 | ((C & 1) & ~(C >> 1) & 1); }
}

static inline void zap__quints(int Q, int q[3]) { /* 7 bits -> 3 quints (spec C.2.12) */
    if ((Q >> 1 & 3) == 3 && (Q >> 5 & 3) == 0) {
        q[2] = (Q & 1) << 2 | ((Q >> 4 & 1) & ~Q & 1) << 1 | ((Q >> 3 & 1) & ~Q & 1); q[1] = 4; q[0] = 4;
        return;
    }
    int C;
    if ((Q >> 1 & 3) == 3) { q[2] = 4; C = (Q >> 3 & 3) << 3 | (~(Q >> 5) & 3) << 1 | (Q & 1); }
    else { q[2] = Q >> 5 & 3; C = Q & 31; }
    if ((C & 7) == 5) { q[1] = 4; q[0] = C >> 3 & 3; } else { q[1] = C >> 3 & 3; q[0] = C & 7; }
}

static const int ZAP__TBITS[5] = { 2, 2, 1, 2, 1 }; /* trit block bits after each value: T[1:0] T[3:2] T[4] T[6:5] T[7] */
static const int ZAP__QBITS[3] = { 3, 2, 2 };       /* quint block bits after each value: Q[2:0] Q[4:3] Q[6:5] */

/* ISE tables: trit/quint combination -> packed code (inverse of the decoders above) */
typedef struct { uint8_t t[243], q[125]; } zap__isetab;
static inline void zap__isetab_init(zap__isetab *k) {
    for (int T = 255; T >= 0; T--) { int t[5]; zap__trits(T, t); k->t[t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4]] = (uint8_t)T; }
    for (int Q = 127; Q >= 0; Q--) { int q[3]; zap__quints(Q, q); k->q[q[0] + 5 * q[1] + 25 * q[2]] = (uint8_t)Q; }
}

static inline void zap__ise_get(const uint8_t *b, int *pos, const zap__iser *r, int n, int *v) {
    int gs = r->trit ? 5 : r->quint ? 3 : 1;
    const int *xb = r->trit ? ZAP__TBITS : ZAP__QBITS;
    for (int g = 0; g < n; g += gs) {
        int cnt = n - g < gs ? n - g : gs, m[5], X = 0, xp = 0, t[5];
        for (int k = 0; k < cnt; k++) {
            m[k] = (int)zap__bits(b, pos, r->bits);
            if (gs > 1) { X |= (int)zap__bits(b, pos, xb[k]) << xp; xp += xb[k]; }
        }
        if (r->trit) zap__trits(X, t);
        else if (r->quint) zap__quints(X, t);
        for (int k = 0; k < cnt; k++) v[g + k] = gs > 1 ? t[k] << r->bits | m[k] : m[k];
    }
}

static inline void zap__ise_put(uint8_t *b, int *pos, const zap__iser *r, int n, const int *v, const zap__isetab *tab) {
    int gs = r->trit ? 5 : r->quint ? 3 : 1, base = r->trit ? 3 : 5;
    const int *xb = r->trit ? ZAP__TBITS : ZAP__QBITS;
    for (int g = 0; g < n; g += gs) {
        int cnt = n - g < gs ? n - g : gs, X = 0, xp = 0;
        if (gs > 1) {
            int idx = 0;
            for (int k = gs - 1; k >= 0; k--) idx = idx * base + (k < cnt ? v[g + k] >> r->bits : 0);
            X = r->trit ? tab->t[idx] : tab->q[idx];
        }
        for (int k = 0; k < cnt; k++) {
            zap__putbits(b, pos, (unsigned)(v[g + k] & ((1 << r->bits) - 1)), r->bits);
            if (gs > 1) { zap__putbits(b, pos, (unsigned)(X >> xp), xb[k]); xp += xb[k]; }
        }
    }
}

/* partition of texel (x, y) for a 2..4 partition pattern `seed` (spec C.2.21; 4x4 is a "small block") */
static inline uint32_t zap__hash52(uint32_t p) {
    p ^= p >> 15; p -= p << 17; p += p << 7; p += p << 4; p ^= p >> 5; p += p << 16; p ^= p >> 7; p ^= p >> 3; p ^= p << 6; p ^= p >> 17;
    return p;
}
static inline int zap__astc_part(int seed, int x, int y, int count) {
    x <<= 1; y <<= 1; /* fewer than 31 texels */
    seed += (count - 1) * 1024;
    uint32_t rnum = zap__hash52((uint32_t)seed);
    int s[12] = { (int)(rnum & 15), (int)(rnum >> 4 & 15), (int)(rnum >> 8 & 15), (int)(rnum >> 12 & 15), (int)(rnum >> 16 & 15), (int)(rnum >> 20 & 15),
                  (int)(rnum >> 24 & 15), (int)(rnum >> 28 & 15), (int)(rnum >> 18 & 15), (int)(rnum >> 22 & 15), (int)(rnum >> 26 & 15), (int)((rnum >> 30 | rnum << 2) & 15) };
    for (int i = 0; i < 12; i++) s[i] *= s[i];
    int sh1, sh2;
    if (seed & 1) { sh1 = seed & 2 ? 4 : 5; sh2 = count == 3 ? 6 : 5; }
    else { sh1 = count == 3 ? 6 : 5; sh2 = seed & 2 ? 4 : 5; }
    int sh3 = seed & 0x10 ? sh1 : sh2;
    for (int i = 0; i < 8; i++) s[i] >>= i & 1 ? sh2 : sh1;
    for (int i = 8; i < 12; i++) s[i] >>= sh3;
    int a = (s[0] * x + s[1] * y + (int)(rnum >> 14)) & 0x3F, b = (s[2] * x + s[3] * y + (int)(rnum >> 10)) & 0x3F;
    int c = (s[4] * x + s[5] * y + (int)(rnum >> 6)) & 0x3F, d = (s[6] * x + s[7] * y + (int)(rnum >> 2)) & 0x3F;
    if (count < 4) d = 0;
    if (count < 3) c = 0;
    return a >= b && a >= c && a >= d ? 0 : b >= c && b >= d ? 1 : c >= d ? 2 : 3;
}

static inline void zap__astc_bluec(int *e) { e[0] = (e[0] + e[2]) >> 1; e[1] = (e[1] + e[2]) >> 1; } /* blue contraction */

/* decoded 8-bit RGBA endpoints of a CEM 8 / 12 partition from its unquantized values v (order r0 r1 g0 g1 b0 b1 [a0 a1]) */
static inline void zap__astc_endpoints(int cem, const int *v, int e0[4], int e1[4]) {
    int a0 = cem == 12 ? v[6] : 255, a1 = cem == 12 ? v[7] : 255;
    if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4]) { e0[0] = v[0]; e0[1] = v[2]; e0[2] = v[4]; e0[3] = a0; e1[0] = v[1]; e1[1] = v[3]; e1[2] = v[5]; e1[3] = a1; }
    else { e0[0] = v[1]; e0[1] = v[3]; e0[2] = v[5]; e0[3] = a1; e1[0] = v[0]; e1[1] = v[2]; e1[2] = v[4]; e1[3] = a0; zap__astc_bluec(e0); zap__astc_bluec(e1); }
}

static inline int zap__astc_lerp(int a, int b, int w) { /* UNORM decode: 16-bit endpoints, 8-bit result */
    return (((a * 257) * (64 - w) + (b * 257) * w + 32) >> 6) >> 8;
}

static inline void zap__astc_dec(const uint8_t *b, uint8_t px[16][4]) {
    int pos = 0, mode = (int)zap__bits(b, &pos, 11);
    for (int i = 0; i < 16; i++) { px[i][0] = 255; px[i][1] = 0; px[i][2] = 255; px[i][3] = 255; } /* unsupported: magenta */
    if ((mode & 0x1FF) == 0x1FC) { /* void extent: constant color, 16-bit UNORM at bits 64..127 */
        int p2 = 64;
        for (int c = 0; c < 4; c++) { int x = (int)zap__bits(b, &p2, 16) >> 8; for (int i = 0; i < 16; i++) px[i][c] = (uint8_t)x; }
        return;
    }
    int R = (mode & 1) << 1 | (mode >> 1 & 1) << 2 | (mode >> 4 & 1), H = mode >> 9 & 1;
    if (!(mode & 3) || (mode >> 2 & 3) || (mode >> 5 & 3) != 2 || (mode >> 7 & 3) || (mode >> 10 & 1)) return; /* not a 4x4 grid */
    static const int WL[2][8] = { { 0, 0, 2, 3, 4, 5, 6, 8 }, { 0, 0, 10, 12, 16, 20, 24, 32 } };
    int levels = WL[H][R], wb = levels == 2 ? 1 : levels == 4 ? 2 : levels == 8 ? 3 : levels == 16 ? 4 : levels == 32 ? 5 : 0;
    int parts = (int)zap__bits(b, &pos, 2) + 1, seed = 0, cem;
    if (!wb || parts > 2) return; /* power-of-two weight ranges, 1 or 2 partitions */
    if (parts == 2) {
        seed = (int)zap__bits(b, &pos, 10);
        int cf = (int)zap__bits(b, &pos, 6);
        if (cf & 3) return; /* both partitions must share one endpoint mode */
        cem = cf >> 2;
    } else cem = (int)zap__bits(b, &pos, 4);
    int nv = cem == 8 ? 6 : cem == 12 ? 8 : 0, v[16], e[2][2][4];
    if (!nv) return;
    const zap__iser *r = zap__astc_crange(nv * parts, 128 - pos - 16 * wb);
    if (!r) return;
    zap__ise_get(b, &pos, r, nv * parts, v);
    for (int i = 0; i < nv * parts; i++) v[i] = zap__astc_cunq(r, v[i]);
    for (int k = 0; k < parts; k++) zap__astc_endpoints(cem, v + k * nv, e[k][0], e[k][1]);
    for (int i = 0; i < 16; i++) {
        int w = 0, k = parts == 2 ? zap__astc_part(seed, i & 3, i >> 2, 2) : 0;
        for (int j = 0; j < wb; j++) { int bit = 127 - (i * wb + j); w |= (b[bit >> 3] >> (bit & 7) & 1) << j; }
        w = zap__astc_wunq(w, wb);
        for (int c = 0; c < 4; c++) px[i][c] = (uint8_t)zap__astc_lerp(e[k][0][c], e[k][1][c], w);
    }
}

typedef struct { const zap__iser *r; int wb, cem, nv, parts; uint8_t q[256]; } zap__astc_cfg; /* q: 8-bit target -> best ISE value */

static inline void zap__astc_cfg_init(zap__astc_cfg *c, int cem, int wb, int parts) {
    c->cem = cem; c->wb = wb; c->nv = cem == 8 ? 6 : 8; c->parts = parts;
    c->r = zap__astc_crange(c->nv * parts, 128 - (parts == 1 ? 17 : 29) - 16 * wb);
    for (int x = 0; x < 256; x++) {
        int best = 0, bd = 1 << 30;
        for (int v = 0; v < c->r->levels; v++) { int d = zap__astc_cunq(c->r, v) - x; d = d < 0 ? -d : d; if (d < bd) { bd = d; best = v; } }
        c->q[x] = (uint8_t)best;
    }
}

/* float endpoints (lo -> v0/v2/v4/v6, hi -> v1/v3/v5/v7) quantized; hi keeps the larger RGB sum so no blue contraction */
static inline void zap__astc_quant(const zap__astc_cfg *c, const float lo[4], const float hi[4], int *ise, int *uv) {
    float s0 = lo[0] + lo[1] + lo[2], s1 = hi[0] + hi[1] + hi[2];
    const float *a = s1 >= s0 ? lo : hi, *b = s1 >= s0 ? hi : lo;
    for (int k = 0; k < c->nv / 2; k++) {
        int x0 = (int)(a[k] + 0.5f), x1 = (int)(b[k] + 0.5f);
        x0 = x0 < 0 ? 0 : x0 > 255 ? 255 : x0; x1 = x1 < 0 ? 0 : x1 > 255 ? 255 : x1;
        ise[2 * k] = c->q[x0]; ise[2 * k + 1] = c->q[x1];
        uv[2 * k] = zap__astc_cunq(c->r, ise[2 * k]); uv[2 * k + 1] = zap__astc_cunq(c->r, ise[2 * k + 1]);
    }
    if (uv[1] + uv[3] + uv[5] < uv[0] + uv[2] + uv[4]) /* quantization flipped the order: swap endpoints back */
        for (int k = 0; k < c->nv / 2; k++) { int t = ise[2 * k]; ise[2 * k] = ise[2 * k + 1]; ise[2 * k + 1] = t; t = uv[2 * k]; uv[2 * k] = uv[2 * k + 1]; uv[2 * k + 1] = t; }
}

/* weights for fixed endpoints (part[i] picks each texel's partition); returns SSE over nc channels */
static inline int zap__astc_fitw(const uint8_t px[16][4], const zap__astc_cfg *c, const int *uv, const int *part, int nc, int w[16]) {
    int L = 1 << c->wb, pal[2][32][4], err = 0;
    for (int k = 0; k < c->parts; k++) {
        int e0[4], e1[4];
        zap__astc_endpoints(c->cem, uv + k * c->nv, e0, e1);
        for (int j = 0; j < L; j++) { int wt = zap__astc_wunq(j, c->wb); for (int ch = 0; ch < 4; ch++) pal[k][j][ch] = zap__astc_lerp(e0[ch], e1[ch], wt); }
    }
    for (int i = 0; i < 16; i++) {
        int bj = 0, be = 1 << 30, k = part[i];
        for (int j = 0; j < L; j++) { int d = 0; for (int ch = 0; ch < nc; ch++) { int x = px[i][ch] - pal[k][j][ch]; d += x * x; } if (d < be) { be = d; bj = j; } }
        w[i] = bj; err += be;
    }
    return err;
}

static inline int zap__astc_try(const uint8_t px[16][4], const zap__astc_cfg *c, const int *part, int ise[16], int w[16]) {
    int nc = c->cem == 12 ? 4 : 3, uv[16], ti[16], tuv[16], tw[16];
    for (int k = 0; k < c->parts; k++) {
        int sel[16], cnt = 0;
        float lo[4], hi[4];
        for (int i = 0; i < 16; i++) if (part[i] == k) sel[cnt++] = i;
        if (!cnt) { sel[0] = 0; cnt = 1; }
        zap__bc7_pca(px, sel, cnt, nc, lo, hi);
        zap__astc_quant(c, lo, hi, ise + k * c->nv, uv + k * c->nv);
    }
    int err = zap__astc_fitw(px, c, uv, part, nc, w);
    for (int it = 0; it < 3 && err > 0; it++) { /* least squares per partition on the chosen weights */
        memcpy(ti, ise, sizeof ti); memcpy(tuv, uv, sizeof tuv);
        for (int k = 0; k < c->parts; k++) {
            float aa = 0, ab = 0, bb = 0, xa[4] = { 0, 0, 0, 0 }, xb[4] = { 0, 0, 0, 0 }, nlo[4], nhi[4];
            for (int i = 0; i < 16; i++) {
                if (part[i] != k) continue;
                float f = zap__astc_wunq(w[i], c->wb) / 64.0f, g = 1 - f;
                aa += g * g; ab += g * f; bb += f * f;
                for (int ch = 0; ch < nc; ch++) { xa[ch] += g * px[i][ch]; xb[ch] += f * px[i][ch]; }
            }
            float det = aa * bb - ab * ab;
            if (det < 1e-6f && det > -1e-6f) continue;
            for (int ch = 0; ch < nc; ch++) { nlo[ch] = (xa[ch] * bb - xb[ch] * ab) / det; nhi[ch] = (xb[ch] * aa - xa[ch] * ab) / det; }
            for (int ch = nc; ch < 4; ch++) nlo[ch] = nhi[ch] = 255;
            zap__astc_quant(c, nlo, nhi, ti + k * c->nv, tuv + k * c->nv);
        }
        int e = zap__astc_fitw(px, c, tuv, part, nc, tw);
        if (e >= err) break;
        err = e; memcpy(ise, ti, sizeof ti); memcpy(uv, tuv, sizeof tuv); memcpy(w, tw, sizeof tw);
    }
    return err;
}

static inline void zap__astc_pack(const zap__astc_cfg *c, int seed, const int *ise, const int w[16], const zap__isetab *tab, uint8_t out[16]) {
    static const int RH[6][2] = { { 0, 0 }, { 0, 0 }, { 4, 0 }, { 7, 0 }, { 4, 1 }, { 7, 1 } }; /* weight bits -> (R, H) */
    int R = RH[c->wb][0], H = RH[c->wb][1], pos = 0;
    unsigned mode = (unsigned)((R >> 1 & 1) | (R >> 2 & 1) << 1 | (R & 1) << 4 | 2 << 5 | H << 9);
    uint8_t wbits[16] = { 0 };
    memset(out, 0, 16);
    zap__putbits(out, &pos, mode, 11);
    zap__putbits(out, &pos, (unsigned)(c->parts - 1), 2);
    if (c->parts == 2) { zap__putbits(out, &pos, (unsigned)seed, 10); zap__putbits(out, &pos, (unsigned)c->cem << 2, 6); }
    else zap__putbits(out, &pos, (unsigned)c->cem, 4);
    zap__ise_put(out, &pos, c->r, c->nv * c->parts, ise, tab);
    int wp = 0;
    for (int i = 0; i < 16; i++) zap__putbits(wbits, &wp, (unsigned)w[i], c->wb);
    for (int i = 0; i < wp; i++) if (wbits[i >> 3] >> (i & 7) & 1) out[(127 - i) >> 3] |= (uint8_t)(1 << ((127 - i) & 7)); /* weights run down from bit 127 */
}

/* 2-means on the block's colors: bit i set = texel i in cluster 1 */
static inline unsigned zap__astc_2means(const uint8_t px[16][4]) {
    float c[2][3];
    int lo = 0, hi = 0;
    for (int i = 1; i < 16; i++) { int s = px[i][0] + px[i][1] + px[i][2]; if (s < px[lo][0] + px[lo][1] + px[lo][2]) lo = i; if (s > px[hi][0] + px[hi][1] + px[hi][2]) hi = i; }
    for (int k = 0; k < 3; k++) { c[0][k] = px[lo][k]; c[1][k] = px[hi][k]; }
    unsigned m = 0;
    for (int it = 0; it < 4; it++) {
        float sum[2][3] = { { 0 } };
        int n[2] = { 0, 0 };
        m = 0;
        for (int i = 0; i < 16; i++) {
            float d0 = 0, d1 = 0;
            for (int k = 0; k < 3; k++) { float a = px[i][k] - c[0][k], b = px[i][k] - c[1][k]; d0 += a * a; d1 += b * b; }
            int s = d1 < d0;
            m |= (unsigned)s << i; n[s]++;
            for (int k = 0; k < 3; k++) sum[s][k] += px[i][k];
        }
        for (int s = 0; s < 2; s++) if (n[s]) for (int k = 0; k < 3; k++) c[s][k] = sum[s][k] / n[s];
    }
    return m;
}

static inline int zap__popc16(unsigned x) { int n = 0; while (x) { n += x & 1; x >>= 1; } return n; }

/* RGBA8 -> ASTC 4x4 blocks (16 bytes each, zap_bc_size(w, h, ZAP_BC7) bytes). Decode as UNORM (not sRGB). */
static inline void zap_astc_encode(const uint8_t *rgba, int w, int h, size_t stride, void *out_) {
    uint8_t *out = (uint8_t *)out_;
    zap__isetab tab;
    zap__astc_cfg cfg[8];
    uint16_t pmask[1024]; /* 2-partition patterns: bit i set = texel i in partition 1 */
    zap__isetab_init(&tab);
    zap__astc_cfg_init(&cfg[0], 8, 4, 1); zap__astc_cfg_init(&cfg[1], 8, 3, 1); zap__astc_cfg_init(&cfg[2], 8, 5, 1);
    zap__astc_cfg_init(&cfg[3], 12, 4, 1); zap__astc_cfg_init(&cfg[4], 12, 3, 1); zap__astc_cfg_init(&cfg[5], 12, 2, 1);
    zap__astc_cfg_init(&cfg[6], 8, 3, 2); zap__astc_cfg_init(&cfg[7], 8, 2, 2);
    for (int sd = 0; sd < 1024; sd++) { unsigned m = 0; for (int i = 0; i < 16; i++) m |= (unsigned)zap__astc_part(sd, i & 3, i >> 2, 2) << i; pmask[sd] = (uint16_t)m; }
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            uint8_t px[16][4];
            int opaque = 1, part0[16] = { 0 };
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x >= w) x = w - 1;
                if (y >= h) y = h - 1;
                memcpy(px[i], rgba + (size_t)y * stride + (size_t)x * 4, 4);
                opaque &= px[i][3] == 255;
            }
            int best = 1 << 30, bc = 0, bseed = 0, bi[16], bwt[16], ise[16], wt[16];
            for (int k = opaque ? 0 : 3; k < (opaque ? 3 : 6); k++) {
                int e = zap__astc_try((const uint8_t(*)[4])px, &cfg[k], part0, ise, wt);
                if (e < best) { best = e; bc = k; memcpy(bi, ise, sizeof ise); memcpy(bwt, wt, sizeof wt); }
            }
            if (opaque && best > 16 * 3 * 4) { /* 2 partitions: patterns closest to the 2-means split, fully fitted */
                unsigned km = zap__astc_2means((const uint8_t(*)[4])px);
                int cand[4] = { -1, -1, -1, -1 }, cd[4] = { 99, 99, 99, 99 };
                for (int sd = 0; sd < 1024; sd++) {
                    int pc = zap__popc16(pmask[sd]);
                    if (pc == 0 || pc == 16) continue; /* degenerate pattern */
                    int dd = zap__popc16(pmask[sd] ^ km), di = 16 - dd;
                    int dist = dd < di ? dd : di;
                    for (int j = 0; j < 4; j++) if (dist < cd[j]) { for (int t = 3; t > j; t--) { cd[t] = cd[t - 1]; cand[t] = cand[t - 1]; } cd[j] = dist; cand[j] = sd; break; }
                }
                for (int j = 0; j < 4 && cand[j] >= 0; j++) {
                    int part[16];
                    for (int i = 0; i < 16; i++) part[i] = pmask[cand[j]] >> i & 1;
                    for (int k = 6; k < 8; k++) {
                        int e = zap__astc_try((const uint8_t(*)[4])px, &cfg[k], part, ise, wt);
                        if (e < best) { best = e; bc = k; bseed = cand[j]; memcpy(bi, ise, sizeof ise); memcpy(bwt, wt, sizeof wt); }
                    }
                }
            }
            zap__astc_pack(&cfg[bc], bseed, bi, bwt, &tab, out + ((size_t)by * (size_t)bw + (size_t)bx) * 16);
        }
}

static inline void zap_astc_decode(const void *bc_, int w, int h, uint8_t *rgba, size_t stride) {
    const uint8_t *bc = (const uint8_t *)bc_;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            uint8_t px[16][4];
            zap__astc_dec(bc + ((size_t)by * (size_t)bw + (size_t)bx) * 16, px);
            for (int i = 0; i < 16; i++) {
                int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x < w && y < h) memcpy(rgba + (size_t)y * stride + (size_t)x * 4, px[i], 4);
            }
        }
}

/* ---------------- mipmaps: 2x2 box filter; sRGB data is averaged in linear light. Odd sizes drop the last
 * row/column (ponytail: box filter; a Kaiser/Lanczos filter would keep more detail in the smaller mips). */
static const float ZAP__SRGB_LIN[256] = {
    0.0f, 0.000303526984f, 0.000607053967f, 0.000910580951f, 0.00121410793f, 0.00151763492f, 0.0018211619f, 0.00212468888f,
    0.00242821587f, 0.00273174285f, 0.00303526984f, 0.00334653576f, 0.00367650732f, 0.00402471702f, 0.00439144204f, 0.00477695348f,
    0.0051815167f, 0.00560539162f, 0.00604883302f, 0.00651209079f, 0.00699541019f, 0.00749903204f, 0.00802319299f, 0.00856812562f,
    0.0091340587f, 0.00972121732f, 0.010329823f, 0.010960094f, 0.0116122452f, 0.0122864884f, 0.0129830323f, 0.013702083f,
    0.0144438436f, 0.0152085144f, 0.0159962934f, 0.0168073758f, 0.0176419545f, 0.0185002201f, 0.019382361f, 0.0202885631f,
    0.0212190104f, 0.0221738848f, 0.0231533662f, 0.0241576324f, 0.0251868596f, 0.0262412219f, 0.0273208916f, 0.0284260395f,
    0.0295568344f, 0.0307134437f, 0.0318960331f, 0.0331047666f, 0.0343398068f, 0.0356013149f, 0.0368894504f, 0.0382043716f,
    0.0395462353f, 0.0409151969f, 0.0423114106f, 0.0437350293f, 0.0451862044f, 0.0466650863f, 0.0481718242f, 0.049706566f,
    0.0512694584f, 0.052860647f, 0.0544802764f, 0.05612849f, 0.0578054302f, 0.0595112382f, 0.0612460542f, 0.0630100177f,
    0.0648032667f, 0.0666259386f, 0.0684781698f, 0.0703600957f, 0.0722718507f, 0.0742135684f, 0.0761853815f, 0.0781874218f,
    0.0802198203f, 0.0822827071f, 0.0843762115f, 0.086500462f, 0.0886555863f, 0.0908417112f, 0.0930589628f, 0.0953074666f,
    0.0975873471f, 0.0998987282f, 0.102241733f, 0.104616484f, 0.107023103f, 0.109461711f, 0.111932428f, 0.114435374f,
    0.116970668f, 0.119538428f, 0.122138772f, 0.124771818f, 0.12743768f, 0.130136477f, 0.132868322f, 0.13563333f,
    0.138431615f, 0.141263291f, 0.144128471f, 0.147027266f, 0.14995979f, 0.152926152f, 0.155926464f, 0.158960835f,
    0.162029376f, 0.165132195f, 0.1682694f, 0.171441101f, 0.174647404f, 0.177888416f, 0.181164244f, 0.184474995f,
    0.187820772f, 0.191201683f, 0.19461783f, 0.19806932f, 0.201556254f, 0.205078736f, 0.20863687f, 0.212230757f,
    0.2158605f, 0.2195262f, 0.223227957f, 0.226965874f, 0.230740049f, 0.234550582f, 0.238397574f, 0.242281122f,
    0.246201327f, 0.250158285f, 0.254152094f, 0.258182853f, 0.262250658f, 0.266355605f, 0.270497791f, 0.274677312f,
    0.278894263f, 0.28314874f, 0.287440838f, 0.29177065f, 0.296138271f, 0.300543794f, 0.304987314f, 0.309468923f,
    0.313988713f, 0.318546778f, 0.323143209f, 0.327778098f, 0.332451536f, 0.337163615f, 0.341914425f, 0.346704056f,
    0.3515326f, 0.356400144f, 0.36130678f, 0.366252596f, 0.37123768f, 0.376262123f, 0.381326011f, 0.386429434f,
    0.391572478f, 0.396755231f, 0.40197778f, 0.407240212f, 0.412542613f, 0.417885071f, 0.42326767f, 0.428690497f,
    0.434153636f, 0.439657174f, 0.445201195f, 0.450785783f, 0.456411023f, 0.462077f, 0.467783796f, 0.473531496f,
    0.479320183f, 0.48514994f, 0.49102085f, 0.496932995f, 0.502886458f, 0.508881321f, 0.514917665f, 0.520995573f,
    0.527115126f, 0.533276404f, 0.539479489f, 0.545724461f, 0.552011402f, 0.55834039f, 0.564711506f, 0.571124829f,
    0.57758044f, 0.584078418f, 0.590618841f, 0.597201788f, 0.603827339f, 0.610495571f, 0.617206562f, 0.623960392f,
    0.630757136f, 0.637596874f, 0.644479682f, 0.651405637f, 0.658374817f, 0.665387298f, 0.672443157f, 0.67954247f,
    0.686685312f, 0.693871761f, 0.701101892f, 0.70837578f, 0.715693501f, 0.723055129f, 0.73046074f, 0.737910409f,
    0.74540421f, 0.752942217f, 0.760524505f, 0.768151147f, 0.775822218f, 0.783537792f, 0.79129794f, 0.799102738f,
    0.806952258f, 0.814846572f, 0.822785754f, 0.830769877f, 0.838799012f, 0.846873232f, 0.854992608f, 0.863157213f,
    0.871367119f, 0.879622397f, 0.887923118f, 0.896269353f, 0.904661174f, 0.913098652f, 0.921581856f, 0.930110858f,
    0.938685728f, 0.947306537f, 0.955973353f, 0.964686248f, 0.97344529f, 0.98225055f, 0.991102097f, 1.0f
};
static const float ZAP__SRGB_THR[255] = {
    0.000151763492f, 0.000455290475f, 0.000758817459f, 0.00106234444f, 0.00136587143f, 0.00166939841f, 0.00197292539f, 0.00227645238f,
    0.00257997936f, 0.00288350634f, 0.0031883009f, 0.00350925935f, 0.00384831493f, 0.00420574803f, 0.00458183274f, 0.00497683725f,
    0.00539102416f, 0.00582465078f, 0.00627796943f, 0.00675122763f, 0.00724466842f, 0.0077585305f, 0.00829304845f, 0.00884845295f,
    0.00942497089f, 0.0100228256f, 0.0106422369f, 0.0112834213f, 0.0119465921f, 0.0126319598f, 0.0133397316f, 0.014070112f,
    0.0148233028f, 0.0155995031f, 0.0163989095f, 0.0172217161f, 0.0180681146f, 0.0189382945f, 0.0198324428f, 0.0207507446f,
    0.0216933829f, 0.0226605384f, 0.0236523902f, 0.024669115f, 0.0257108881f, 0.0267778826f, 0.0278702702f, 0.0289882206f,
    0.0301319019f, 0.0313014806f, 0.0324971216f, 0.0337189882f, 0.0349672424f, 0.0362420443f, 0.037543553f, 0.0388719259f,
    0.0402273192f, 0.0416098877f, 0.0430197848f, 0.0444571628f, 0.0459221727f, 0.047414964f, 0.0489356854f, 0.0504844842f,
    0.0520615066f, 0.0536668976f, 0.0553008013f, 0.0569633604f, 0.0586547169f, 0.0603750115f, 0.0621243839f, 0.0639029729f,
    0.0657109163f, 0.0675483509f, 0.0694154125f, 0.0713122362f, 0.0732389559f, 0.0751957047f, 0.077182615f, 0.0791998181f,
    0.0812474446f, 0.0833256241f, 0.0854344855f, 0.087574157f, 0.0897447658f, 0.0919464383f, 0.0941793004f, 0.096443477f,
    0.0987390924f, 0.10106627f, 0.103425133f, 0.105815802f, 0.108238401f, 0.110693048f, 0.113179865f, 0.11569897f,
    0.118250482f, 0.12083452f, 0.1234512f, 0.12610064f, 0.128782955f, 0.131498261f, 0.134246673f, 0.137028306f,
    0.139843272f, 0.142691686f, 0.14557366f, 0.148489305f, 0.151438734f, 0.154422057f, 0.157439385f, 0.160490827f,
    0.163576493f, 0.166696492f, 0.169850932f, 0.17303992f, 0.176263564f, 0.179521971f, 0.182815248f, 0.186143498f,
    0.189506829f, 0.192905345f, 0.196339151f, 0.19980835f, 0.203313045f, 0.20685334f, 0.210429338f, 0.21404114f,
    0.217688849f, 0.221372565f, 0.225092389f, 0.228848422f, 0.232640764f, 0.236469515f, 0.240334772f, 0.244236636f,
    0.248175205f, 0.252150577f, 0.256162849f, 0.260212118f, 0.264298482f, 0.268422037f, 0.272582879f, 0.276781103f,
    0.281016805f, 0.285290081f, 0.289601024f, 0.293949728f, 0.298336289f, 0.302760799f, 0.307223352f, 0.31172404f,
    0.316262956f, 0.320840192f, 0.325455841f, 0.330109993f, 0.33480274f, 0.339534173f, 0.344304382f, 0.349113458f,
    0.353961491f, 0.35884857f, 0.363774785f, 0.368740224f, 0.373744977f, 0.378789131f, 0.383872775f, 0.388995998f,
    0.394158885f, 0.399361525f, 0.404604005f, 0.409886411f, 0.41520883f, 0.420571347f, 0.42597405f, 0.431417022f,
    0.43690035f, 0.442424119f, 0.447988412f, 0.453593316f, 0.459238914f, 0.46492529f, 0.470652528f, 0.476420711f,
    0.482229923f, 0.488080246f, 0.493971763f, 0.499904557f, 0.505878709f, 0.511894303f, 0.517951419f, 0.524050139f,
    0.530190544f, 0.536372716f, 0.542596734f, 0.54886268f, 0.555170635f, 0.561520677f, 0.567912887f, 0.574347344f,
    0.580824128f, 0.587343319f, 0.593904994f, 0.600509233f, 0.607156115f, 0.613845717f, 0.620578117f, 0.627353395f,
    0.634171626f, 0.641032889f, 0.647937261f, 0.654884819f, 0.66187564f, 0.668909801f, 0.675987377f, 0.683108445f,
    0.690273081f, 0.697481362f, 0.704733362f, 0.712029156f, 0.719368822f, 0.726752432f, 0.734180063f, 0.741651788f,
    0.749167683f, 0.756727821f, 0.764332277f, 0.771981125f, 0.779674438f, 0.787412289f, 0.795194753f, 0.803021903f,
    0.810893811f, 0.81881055f, 0.826772194f, 0.834778813f, 0.842830482f, 0.850927271f, 0.859069253f, 0.867256499f,
    0.875489082f, 0.883767073f, 0.892090542f, 0.900459561f, 0.908874202f, 0.917334534f, 0.925840628f, 0.934392556f,
    0.942990386f, 0.95163419f, 0.960324036f, 0.969059996f, 0.977842139f, 0.986670534f, 0.99554525f
}; /* linear value where sRGB k rounds up to k + 1 */

static inline uint8_t zap__lin2srgb(float v) {
    int lo = 0, hi = 255; /* first k with v < threshold[k] */
    while (lo < hi) { int mid = (lo + hi) >> 1; if (v < ZAP__SRGB_THR[mid]) hi = mid; else lo = mid + 1; }
    return (uint8_t)lo;
}

static inline int zap_mip_levels(int w, int h) {
    int n = 1;
    while (w > 1 || h > 1) { w = w > 1 ? w / 2 : 1; h = h > 1 ? h / 2 : 1; n++; }
    return n;
}

/* RGBA8 level -> next level (max(1, w/2) x max(1, h/2)). srgb: RGB is sRGB-encoded (alpha is always linear). */
static inline void zap_mip_next(const uint8_t *src, int w, int h, size_t stride, int srgb, uint8_t *dst, size_t dst_stride) {
    int nw = w > 1 ? w / 2 : 1, nh = h > 1 ? h / 2 : 1;
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++) {
            int x0 = 2 * x, x1 = 2 * x + 1 < w ? 2 * x + 1 : w - 1, y0 = 2 * y, y1 = 2 * y + 1 < h ? 2 * y + 1 : h - 1;
            if (x0 >= w) x0 = w - 1;
            if (y0 >= h) y0 = h - 1;
            const uint8_t *p[4] = { src + (size_t)y0 * stride + (size_t)x0 * 4, src + (size_t)y0 * stride + (size_t)x1 * 4,
                                    src + (size_t)y1 * stride + (size_t)x0 * 4, src + (size_t)y1 * stride + (size_t)x1 * 4 };
            uint8_t *o = dst + (size_t)y * dst_stride + (size_t)x * 4;
            for (int c = 0; c < 4; c++) {
                if (srgb && c < 3) o[c] = zap__lin2srgb((ZAP__SRGB_LIN[p[0][c]] + ZAP__SRGB_LIN[p[1][c]] + ZAP__SRGB_LIN[p[2][c]] + ZAP__SRGB_LIN[p[3][c]]) * 0.25f);
                else o[c] = (uint8_t)((p[0][c] + p[1][c] + p[2][c] + p[3][c] + 2) >> 2);
            }
        }
}

/* float RGBA level (linear) -> next level; strides in floats per row */
static inline void zap_mip_next_f(const float *src, int w, int h, size_t stride, float *dst, size_t dst_stride) {
    int nw = w > 1 ? w / 2 : 1, nh = h > 1 ? h / 2 : 1;
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++) {
            int x0 = 2 * x < w ? 2 * x : w - 1, x1 = 2 * x + 1 < w ? 2 * x + 1 : w - 1, y0 = 2 * y < h ? 2 * y : h - 1, y1 = 2 * y + 1 < h ? 2 * y + 1 : h - 1;
            for (int c = 0; c < 4; c++)
                dst[(size_t)y * dst_stride + (size_t)x * 4 + c] = 0.25f * (src[(size_t)y0 * stride + (size_t)x0 * 4 + c] + src[(size_t)y0 * stride + (size_t)x1 * 4 + c] +
                                                                           src[(size_t)y1 * stride + (size_t)x0 * 4 + c] + src[(size_t)y1 * stride + (size_t)x1 * 4 + c]);
        }
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
