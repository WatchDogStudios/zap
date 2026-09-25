/* zap.h - single-header LZ77 byte codec for games (packaging + networking).
 * https://github.com/WatchDogStudios/zap   SPDX-License-Identifier: MIT   Copyright (c) 2026 WD Studios Corp.
 *
 * Blocks (raw API, you store the sizes):
 *   size_t    zap_compress   (src, n, dst, cap, zap_state*, dict);          // fast, ~250+ MB/s. 0 = didn't fit
 *   size_t    zap_compress_hc(src, n, dst, cap, zap_hc_state*, dict, depth); // slow, smaller, same format
 *   ptrdiff_t zap_decompress (src, n, dst, raw_size, dict);                 // -1 = corrupt / wrong size
 *
 * Frames (packaging: self-describing, independent blocks, parallel decode):
 *   zap_frame_compress(src, n, dst, cap, block_size, depth /0 = fast/, dict)
 *   zap_frame_open + zap_frame_decode_block(f, i, dst, dict)  -> call from your job system
 *   zap_frame_decode(...)  single thread
 *   #define ZAP_THREADS for zap_frame_compress_mt / zap_frame_decode_mt (pthreads, or C11 threads on Windows)
 *
 * Dictionaries (networking: small packets):
 *   zap_dict_train(samples, n, dict_out, cap)  -> bytes; then zap_dict_init(&d, dict_out, size) on both ends.
 *
 * - Decoder bounds-checks every read and write: safe on untrusted network data.
 *   It writes only inside [dst, dst+raw_size) and must produce exactly raw_size bytes.
 * - Endian-neutral format; compressed output is identical on LE and BE hosts.
 *
 * Block format: [token: lit<<4 | (mlen-4)][lit ext 255..][literals][offset][mlen ext 255..]
 *   offset: u16 LE < 0x8000, or 3 bytes (u16 with top bit set + 1 byte << 15) for up to 8MB back.
 * Last sequence is literals only and ends exactly at end of input.
 */
#ifndef ZAP_H
#define ZAP_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ZAP_VERSION "1.0.0"

#define ZAP_HLOG 16
#define ZAP_HC_HLOG 17
#define ZAP_WINDOW_LOG 23
#define ZAP_MAX_DIST (((size_t)1 << ZAP_WINDOW_LOG) - 1)
#define ZAP_FRAME_MAGIC 0x3150415Au /* "ZAP1" */

typedef struct { uint32_t table[1 << ZAP_HLOG]; } zap_state; /* 256KB, reuse per thread */
typedef struct { uint32_t head[1 << ZAP_HC_HLOG]; uint32_t prev[1 << ZAP_WINDOW_LOG]; } zap_hc_state; /* ~32.5MB, heap it */
typedef struct { const uint8_t *data; size_t len; uint32_t table[1 << ZAP_HLOG]; } zap_dict;

static inline size_t zap_bound(size_t n) { return n + n / 255 + 16; }

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define ZAP_BIG_ENDIAN 1
#endif

static inline uint32_t zap__r32(const uint8_t *p) { /* little-endian load */
    uint32_t v; memcpy(&v, p, 4);
#ifdef ZAP_BIG_ENDIAN
    v = __builtin_bswap32(v);
#endif
    return v;
}
static inline uint64_t zap__r64n(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; } /* native */
static inline void zap__w32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static inline uint32_t zap__hash(uint32_t v, int hlog) { return (v * 2654435761u) >> (32 - hlog); }

/* index of the first differing byte in memory order, given x = load(a) ^ load(b) != 0 */
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline unsigned zap__nb(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (unsigned)i >> 3; }
#elif defined(ZAP_BIG_ENDIAN)
static inline unsigned zap__nb(uint64_t x) { return (unsigned)__builtin_clzll(x) >> 3; }
#else
static inline unsigned zap__nb(uint64_t x) { return (unsigned)__builtin_ctzll(x) >> 3; }
#endif

/* common prefix length of a and b, a bounded by aend */
static inline size_t zap__count(const uint8_t *a, const uint8_t *b, const uint8_t *aend) {
    const uint8_t *s = a;
    while (aend - a >= 8) {
        uint64_t x = zap__r64n(a) ^ zap__r64n(b);
        if (x) return (size_t)(a - s) + zap__nb(x);
        a += 8; b += 8;
    }
    while (a < aend && *a == *b) { a++; b++; }
    return (size_t)(a - s);
}

/* far offsets cost a byte more, so they need a longer match to pay off */
static inline size_t zap__minml(size_t off) { return off < 0x8000 ? 4 : 5; }
static inline size_t zap__score(size_t ml, size_t off) { return ml - (off >= 0x8000); }

static inline uint8_t *zap__len(uint8_t *op, size_t v) {
    while (v >= 255) { *op++ = 255; v -= 255; }
    *op++ = (uint8_t)v;
    return op;
}

static inline uint8_t *zap__emit(uint8_t *op, uint8_t *oend, const uint8_t *lit, size_t ll, size_t off, size_t ml) {
    if ((size_t)(oend - op) < ll + ll / 255 + ml / 255 + 9) return NULL;
    size_t mc = ml - 4;
    *op++ = (uint8_t)(((ll < 15 ? ll : 15) << 4) | (mc < 15 ? mc : 15));
    if (ll >= 15) op = zap__len(op, ll - 15);
    memcpy(op, lit, ll); op += ll;
    if (off < 0x8000) { *op++ = (uint8_t)off; *op++ = (uint8_t)(off >> 8); }
    else { *op++ = (uint8_t)off; *op++ = (uint8_t)(((off >> 8) & 0x7F) | 0x80); *op++ = (uint8_t)(off >> 15); }
    if (mc >= 15) op = zap__len(op, mc - 15);
    return op;
}

static inline size_t zap__finish(uint8_t *op, uint8_t *oend, const uint8_t *anchor, const uint8_t *iend, uint8_t *dst) {
    size_t ll = (size_t)(iend - anchor);
    if ((size_t)(oend - op) < ll + ll / 255 + 2) return 0;
    *op++ = (uint8_t)((ll < 15 ? ll : 15) << 4);
    if (ll >= 15) op = zap__len(op, ll - 15);
    memcpy(op, anchor, ll); op += ll;
    return (size_t)(op - dst);
}

/* dict: only the last 8MB is reachable. Keep dict memory alive while in use. */
static inline void zap_dict_init(zap_dict *d, const void *data, size_t len) {
    if (len > ZAP_MAX_DIST) { data = (const uint8_t *)data + len - ZAP_MAX_DIST; len = ZAP_MAX_DIST; }
    d->data = (const uint8_t *)data; d->len = len < 4 ? 0 : len;
    memset(d->table, 0, sizeof d->table);
    for (size_t i = 0; i + 4 <= d->len; i++) d->table[zap__hash(zap__r32(d->data + i), ZAP_HLOG)] = (uint32_t)i;
}

/* dict candidate at ip: match length (0 if none), offset in *off */
static inline size_t zap__dmatch(const zap_dict *d, const uint8_t *ip, const uint8_t *src, const uint8_t *iend, size_t *off) {
    uint32_t seq = zap__r32(ip);
    size_t dc = d->table[zap__hash(seq, ZAP_HLOG)];
    if (dc + 4 > d->len) return 0;
    *off = (size_t)(ip - src) + d->len - dc;
    if (*off > ZAP_MAX_DIST || zap__r32(d->data + dc) != seq) return 0;
    size_t rest = d->len - dc - 4, avail = (size_t)(iend - ip - 4);
    size_t ml = zap__count(ip + 4, d->data + dc + 4, ip + 4 + (rest < avail ? rest : avail));
    if (ml == rest) ml += zap__count(ip + 4 + ml, src, iend); /* match runs off dict end into src */
    return ml + 4;
}

static inline size_t zap_compress(const void *src_, size_t n, void *dst_, size_t cap,
                                  zap_state *st, const zap_dict *d) {
    const uint8_t *src = (const uint8_t *)src_, *ip = src, *anchor = src, *iend = src + n;
    uint8_t *op = (uint8_t *)dst_, *oend = op + cap;
    uint32_t *t = st->table;
    int hlog = 8;
    while (hlog < ZAP_HLOG && ((size_t)1 << hlog) < n) hlog++; /* small packets: small table to clear */
    memset(t, 0, sizeof(uint32_t) << hlog);

    if (n >= 13) {
        const uint8_t *mflimit = iend - 12;
        size_t misses = 0;
        while (ip < mflimit) {
            uint32_t seq = zap__r32(ip), h = zap__hash(seq, hlog);
            size_t pos = (size_t)(ip - src), cand = t[h], off = 0, ml = 0;
            t[h] = (uint32_t)pos;
            if (cand < pos && pos - cand <= ZAP_MAX_DIST && zap__r32(src + cand) == seq) {
                off = pos - cand;
                ml = 4 + zap__count(ip + 4, src + cand + 4, iend);
            } else if (d) {
                ml = zap__dmatch(d, ip, src, iend, &off);
            }
            if (ml < 4 || ml < zap__minml(off)) {
                ip += 1 + (misses++ >> 6); /* skip faster through incompressible data */
                continue;
            }
            if (off <= pos) { const uint8_t *m = ip - off; while (ip > anchor && m > src && ip[-1] == m[-1]) { ip--; m--; ml++; } }
            if (!(op = zap__emit(op, oend, anchor, (size_t)(ip - anchor), off, ml))) return 0;
            ip += ml; anchor = ip; misses = 0;
            if (ip < mflimit) t[zap__hash(zap__r32(ip - 2), hlog)] = (uint32_t)(ip - 2 - src);
        }
    }
    return zap__finish(op, oend, anchor, iend, (uint8_t *)dst_);
}

/* best match at ip via hash chains (inserts every position up to ip first) */
static inline size_t zap__hc_find(zap_hc_state *s, const uint8_t *src, const uint8_t *ip, const uint8_t *iend,
                                  const zap_dict *d, int depth, size_t *next, size_t *off_out) {
    const size_t mask = ((size_t)1 << ZAP_WINDOW_LOG) - 1, pos = (size_t)(ip - src);
    for (size_t p = *next; p < pos; p++) {
        uint32_t h = zap__hash(zap__r32(src + p), ZAP_HC_HLOG);
        s->prev[p & mask] = s->head[h]; s->head[h] = (uint32_t)p;
    }
    if (pos > *next) *next = pos;
    uint32_t seq = zap__r32(ip);
    size_t best = 0, boff = 0, bscore = 0, c = s->head[zap__hash(seq, ZAP_HC_HLOG)];
    while (c < pos && pos - c <= ZAP_MAX_DIST && depth-- > 0) {
        const uint8_t *m = src + c;
        if (m[best] == ip[best] && zap__r32(m) == seq) {
            size_t ml = 4 + zap__count(ip + 4, m + 4, iend), o = pos - c;
            if (ml >= zap__minml(o) && zap__score(ml, o) > bscore) {
                best = ml; boff = o; bscore = zap__score(ml, o);
                if (ip + ml == iend) break; /* can't beat it, and ip[best] would overrun */
            }
        }
        c = s->prev[c & mask];
    }
    if (d) {
        size_t o, ml = zap__dmatch(d, ip, src, iend, &o);
        if (ml >= 4 && ml >= zap__minml(o) && zap__score(ml, o) > bscore) { best = ml; boff = o; }
    }
    *off_out = boff;
    return best;
}

/* depth: chain steps per position. 16 = quick, 64 = good default, 1024+ = max effort */
static inline size_t zap_compress_hc(const void *src_, size_t n, void *dst_, size_t cap,
                                     zap_hc_state *s, const zap_dict *d, int depth) {
    const uint8_t *src = (const uint8_t *)src_, *ip = src, *anchor = src, *iend = src + n;
    uint8_t *op = (uint8_t *)dst_, *oend = op + cap;
    size_t next = 0;
    memset(s->head, 0xFF, sizeof s->head);
    if (n >= 13) {
        const uint8_t *mflimit = iend - 12;
        while (ip < mflimit) {
            size_t off, ml = zap__hc_find(s, src, ip, iend, d, depth, &next, &off);
            if (!ml) { ip++; continue; }
            /* lazy: a better match one byte later is worth one literal */
            while (ip + 1 < mflimit) {
                size_t off2, ml2 = zap__hc_find(s, src, ip + 1, iend, d, depth, &next, &off2);
                if (!ml2 || zap__score(ml2, off2) <= zap__score(ml, off)) break;
                ip++; ml = ml2; off = off2;
            }
            size_t pos = (size_t)(ip - src);
            if (off <= pos) { const uint8_t *m = ip - off; while (ip > anchor && m > src && ip[-1] == m[-1]) { ip--; m--; ml++; } }
            if (!(op = zap__emit(op, oend, anchor, (size_t)(ip - anchor), off, ml))) return 0;
            ip += ml; anchor = ip;
        }
    }
    return zap__finish(op, oend, anchor, iend, (uint8_t *)dst_);
}

/* copy ml bytes from op-off (may overlap) to op; op+ml <= oend guaranteed by caller */
static inline void zap__match_copy(uint8_t *op, size_t off, size_t ml, uint8_t *oend) {
    const uint8_t *m = op - off;
    if (off < 16) { /* short period: prime bytewise, then copy with a period multiple >= 16 */
        size_t p = off;
        while (p < 16) p += off;
        size_t pre = p - off < ml ? p - off : ml;
        for (size_t i = 0; i < pre; i++) op[i] = m[i];
        op += pre; ml -= pre; m = op - p;
    }
    if ((size_t)(oend - op) >= ml + 16) {
        uint8_t *e = op + ml;
        while (op < e) { memcpy(op, m, 16); op += 16; m += 16; }
    } else {
        while (ml--) *op++ = *m++;
    }
}

static inline size_t zap__ext(const uint8_t **ip, const uint8_t *iend, int *err) {
    size_t v = 0; uint8_t b;
    do {
        if (*ip >= iend) { *err = 1; return 0; }
        b = *(*ip)++; v += b;
    } while (b == 255);
    return v;
}

/* match that may start in the dict and continue into output. Validated by caller. */
static inline void zap__dict_copy(uint8_t *op, uint8_t *ostart, uint8_t *oend, const zap_dict *d, size_t off, size_t ml) {
    size_t have = (size_t)(op - ostart);
    if (off > have) {
        size_t back = off - have, n1 = back < ml ? back : ml;
        memcpy(op, d->data + d->len - back, n1);
        op += n1; ml -= n1;
    }
    if (ml) zap__match_copy(op, off, ml, oend);
}

/* raw_size must be the exact decompressed size. Returns raw_size, or -1. */
static inline ptrdiff_t zap_decompress(const void *src_, size_t n, void *dst_, size_t raw_size, const zap_dict *d) {
    const uint8_t *ip = (const uint8_t *)src_, *iend = ip + n;
    uint8_t *op = (uint8_t *)dst_, *ostart = op, *oend = op + raw_size;
    size_t dlen = d ? d->len : 0;
    int err = 0;
    for (;;) {
        if (ip >= iend) return -1;
        unsigned tok = *ip++;
        size_t lit = tok >> 4;
        /* hot path: short literals + short match, lots of headroom -> fixed-size copies, no loops */
        if (lit < 15 && (tok & 15) < 15 && iend - ip >= 32 && oend - op >= 32) {
            memcpy(op, ip, 16);
            op += lit; ip += lit;
            size_t off = (size_t)ip[0] | ((size_t)ip[1] << 8), ml = (tok & 15) + 4;
            if (off & 0x8000) { off = (off & 0x7FFF) | ((size_t)ip[2] << 15); ip += 3; } else ip += 2;
            size_t have = (size_t)(op - ostart);
            if (off >= 16 && off <= have) {
                memcpy(op, op - off, 16);
                memcpy(op + 16, op - off + 16, 2);
            } else if (off == 0 || off > have + dlen) {
                return -1;
            } else if (off >= have + 18) { /* whole match inside dict */
                memcpy(op, d->data + dlen - (off - have), 18);
            } else {
                zap__dict_copy(op, ostart, oend, d, off, ml);
            }
            op += ml;
            continue;
        }
        if (lit == 15) { lit += zap__ext(&ip, iend, &err); if (err) return -1; }
        if (lit > (size_t)(iend - ip) || lit > (size_t)(oend - op)) return -1;
        if ((size_t)(iend - ip) >= lit + 16 && (size_t)(oend - op) >= lit + 16) {
            const uint8_t *s = ip; uint8_t *o = op, *e = op + lit;
            while (o < e) { memcpy(o, s, 16); o += 16; s += 16; }
        } else {
            memcpy(op, ip, lit);
        }
        op += lit; ip += lit;
        if (ip == iend) break;

        if (iend - ip < 2) return -1;
        size_t off = (size_t)ip[0] | ((size_t)ip[1] << 8);
        ip += 2;
        if (off & 0x8000) { if (ip >= iend) return -1; off = (off & 0x7FFF) | ((size_t)*ip++ << 15); }
        size_t ml = tok & 15;
        if (ml == 15) { ml += zap__ext(&ip, iend, &err); if (err) return -1; }
        ml += 4;
        if (off == 0 || off > (size_t)(op - ostart) + dlen || ml > (size_t)(oend - op)) return -1;
        zap__dict_copy(op, ostart, oend, d, off, ml);
        op += ml;
    }
    return op == oend ? op - ostart : -1;
}

/* ---------------- frames: [magic][block_size u32][raw_size u64][end offset u64 per block][blocks]
 * A block whose stored size equals its raw size is stored uncompressed. */

typedef struct { const uint8_t *table, *blocks; size_t raw, bs, nb, blocks_size; } zap_frame;

static inline size_t zap_frame_bound(size_t n, size_t bs) { return 16 + 8 * ((n + bs - 1) / bs) + n; }

static inline void zap__w64(uint8_t *p, uint64_t v) { zap__w32(p, (uint32_t)v); zap__w32(p + 4, (uint32_t)(v >> 32)); }

/* one frame block: compressed size (< len), or 0 = store raw. lim = output room. */
static inline size_t zap__block(const uint8_t *src, size_t len, uint8_t *dst, size_t lim, void *st, int depth, const zap_dict *d) {
    if (len < 2) return 0;
    if (lim > len - 1) lim = len - 1;
    return depth ? zap_compress_hc(src, len, dst, lim, (zap_hc_state *)st, d, depth)
                 : zap_compress(src, len, dst, lim, (zap_state *)st, d);
}

static inline size_t zap__frame_hdr(uint8_t *dst, size_t cap, size_t n, size_t bs, size_t *nb) {
    if (!bs || bs > 0x80000000u) return 0;
    size_t hdr = 16 + 8 * (*nb = (n + bs - 1) / bs);
    if (cap < hdr) return 0;
    zap__w32(dst, ZAP_FRAME_MAGIC); zap__w32(dst + 4, (uint32_t)bs); zap__w64(dst + 8, n);
    return hdr;
}

/* depth 0 = fast compressor, >0 = hc chain depth. block_size <= 2GB. Returns frame size, 0 on failure. */
static inline size_t zap_frame_compress(const void *src_, size_t n, void *dst_, size_t cap, size_t bs, int depth, const zap_dict *d) {
    const uint8_t *src = (const uint8_t *)src_;
    uint8_t *dst = (uint8_t *)dst_;
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb), hdr = pos;
    if (!pos) return 0;
    void *st = malloc(depth ? sizeof(zap_hc_state) : sizeof(zap_state));
    if (!st) return 0;
    for (size_t b = 0; b < nb; b++) {
        size_t len = b == nb - 1 ? n - b * bs : bs, room = cap - pos;
        size_t c = zap__block(src + b * bs, len, dst + pos, room, st, depth, d);
        if (!c) { /* incompressible: store raw */
            if (room < len) { free(st); return 0; }
            memcpy(dst + pos, src + b * bs, len); c = len;
        }
        pos += c;
        zap__w64(dst + 16 + 8 * b, pos - hdr);
    }
    free(st);
    return pos;
}

static inline uint64_t zap__r64le(const uint8_t *p) { return zap__r32(p) | ((uint64_t)zap__r32(p + 4) << 32); }

/* validates the header and block table. 0 ok, -1 corrupt. */
static inline int zap_frame_open(zap_frame *f, const void *src_, size_t n) {
    const uint8_t *src = (const uint8_t *)src_;
    if (n < 16 || zap__r32(src) != ZAP_FRAME_MAGIC) return -1;
    uint64_t raw = zap__r64le(src + 8);
    f->bs = zap__r32(src + 4);
    if (!f->bs || f->bs > 0x80000000u || raw > SIZE_MAX) return -1;
    f->raw = (size_t)raw;
    f->nb = f->raw / f->bs + (f->raw % f->bs != 0);
    if (f->nb > (n - 16) / 8) return -1;
    f->table = src + 16; f->blocks = f->table + 8 * f->nb; f->blocks_size = n - 16 - 8 * f->nb;
    uint64_t prev = 0;
    for (size_t b = 0; b < f->nb; b++) {
        uint64_t end = zap__r64le(f->table + 8 * b), len = b == f->nb - 1 ? f->raw - b * f->bs : f->bs;
        if (end < prev || end - prev > len || end > f->blocks_size) return -1;
        prev = end;
    }
    return 0;
}

/* decode block i into dst (the whole raw_size output buffer). Thread-safe: call from any job. 0 ok, -1 corrupt. */
static inline int zap_frame_decode_block(const zap_frame *f, size_t i, void *dst, const zap_dict *d) {
    size_t start = i ? (size_t)zap__r64le(f->table + 8 * (i - 1)) : 0, end = (size_t)zap__r64le(f->table + 8 * i);
    size_t len = i == f->nb - 1 ? f->raw - i * f->bs : f->bs;
    uint8_t *o = (uint8_t *)dst + i * f->bs;
    if (end - start == len) { memcpy(o, f->blocks + start, len); return 0; }
    return zap_decompress(f->blocks + start, end - start, o, len, d) == (ptrdiff_t)len ? 0 : -1;
}

/* returns raw size, or -1 if corrupt / cap too small */
static inline ptrdiff_t zap_frame_decode(const void *src, size_t n, void *dst, size_t cap, const zap_dict *d) {
    zap_frame f;
    if (zap_frame_open(&f, src, n) || f.raw > cap) return -1;
    for (size_t b = 0; b < f.nb; b++) if (zap_frame_decode_block(&f, b, dst, d)) return -1;
    return (ptrdiff_t)f.raw;
}

#ifdef ZAP_THREADS
/* run fn(ctx, t, T) for t in [0, T) on T threads (t = 0 on the caller) */
typedef struct { void (*fn)(void *, int, int); void *ctx; int t, n; } zap__task;
#ifdef _WIN32
#include <threads.h> /* MSVC 17.8+ / UCRT */
static inline int zap__tramp(void *p) { zap__task *k = (zap__task *)p; k->fn(k->ctx, k->t, k->n); return 0; }
typedef thrd_t zap__thread;
#define ZAP__START(th, k) (thrd_create(&(th), zap__tramp, (k)) == thrd_success)
#define ZAP__JOIN(th) thrd_join((th), NULL)
#else
#include <pthread.h>
static inline void *zap__tramp(void *p) { zap__task *k = (zap__task *)p; k->fn(k->ctx, k->t, k->n); return NULL; }
typedef pthread_t zap__thread;
#define ZAP__START(th, k) (pthread_create(&(th), NULL, zap__tramp, (k)) == 0)
#define ZAP__JOIN(th) pthread_join((th), NULL)
#endif
enum { ZAP__MAXT = 64 };

static inline int zap__threads(int want, size_t jobs) {
    if (want > ZAP__MAXT) want = ZAP__MAXT;
    if ((size_t)want > jobs) want = (int)jobs;
    return want < 1 ? 1 : want;
}

static inline void zap__par(int n, void (*fn)(void *, int, int), void *ctx) {
    zap__thread th[ZAP__MAXT]; zap__task k[ZAP__MAXT]; int ok[ZAP__MAXT];
    for (int t = 0; t < n; t++) {
        k[t].fn = fn; k[t].ctx = ctx; k[t].t = t; k[t].n = n;
        ok[t] = t && ZAP__START(th[t], &k[t]);
    }
    for (int t = 0; t < n; t++) {
        if (ok[t]) ZAP__JOIN(th[t]);
        else fn(ctx, t, n); /* thread 0, or a thread that failed to start: run it here */
    }
}

/* ponytail: static block striding, fine for equal-size blocks; use your job system + decode_block for anything fancier */
typedef struct { const zap_frame *f; uint8_t *dst; const zap_dict *d; int err[ZAP__MAXT]; } zap__djob;
static inline void zap__dwork(void *p, int t, int n) {
    zap__djob *j = (zap__djob *)p;
    for (size_t b = (size_t)t; b < j->f->nb; b += (size_t)n) if (zap_frame_decode_block(j->f, b, j->dst, j->d)) j->err[t] = 1;
}

static inline ptrdiff_t zap_frame_decode_mt(const void *src, size_t n, void *dst, size_t cap, const zap_dict *d, int threads) {
    zap_frame f;
    if (zap_frame_open(&f, src, n) || f.raw > cap) return -1;
    zap__djob j = { &f, (uint8_t *)dst, d, { 0 } };
    threads = zap__threads(threads, f.nb);
    zap__par(threads, zap__dwork, &j);
    for (int t = 0; t < threads; t++) if (j.err[t]) return -1;
    return (ptrdiff_t)f.raw;
}

/* blocks compress into their own slot of tmp (each result < block size, so slots never overflow) */
typedef struct { const uint8_t *src; uint8_t *tmp; size_t n, bs, nb, *cs; int depth; const zap_dict *d; } zap__cjob;
static inline void zap__cwork(void *p, int t, int n) {
    zap__cjob *j = (zap__cjob *)p;
    void *st = malloc(j->depth ? sizeof(zap_hc_state) : sizeof(zap_state));
    if (!st) return; /* its blocks keep cs == SIZE_MAX -> caller fails */
    for (size_t b = (size_t)t; b < j->nb; b += (size_t)n) {
        size_t len = b == j->nb - 1 ? j->n - b * j->bs : j->bs;
        j->cs[b] = zap__block(j->src + b * j->bs, len, j->tmp + b * j->bs, len, st, j->depth, j->d);
    }
    free(st);
}

/* same output as zap_frame_compress. Memory: threads x state (32.5MB each for hc) + n bytes scratch. */
static inline size_t zap_frame_compress_mt(const void *src_, size_t n, void *dst_, size_t cap, size_t bs, int depth,
                                           const zap_dict *d, int threads) {
    const uint8_t *src = (const uint8_t *)src_;
    uint8_t *dst = (uint8_t *)dst_;
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb), hdr = pos;
    if (!pos) return 0;
    zap__cjob j = { src, (uint8_t *)malloc(n ? n : 1), n, bs, nb, (size_t *)malloc((nb ? nb : 1) * sizeof(size_t)), depth, d };
    if (!j.tmp || !j.cs) { free(j.tmp); free(j.cs); return 0; }
    for (size_t b = 0; b < nb; b++) j.cs[b] = SIZE_MAX;
    zap__par(zap__threads(threads, nb), zap__cwork, &j);
    for (size_t b = 0; b < nb; b++) {
        size_t len = b == nb - 1 ? n - b * bs : bs, c = j.cs[b];
        const uint8_t *from = c ? j.tmp + b * bs : src + b * bs;
        if (!c) c = len; /* store raw */
        if (c == SIZE_MAX || cap - pos < c) { pos = 0; break; }
        memcpy(dst + pos, from, c);
        pos += c;
        zap__w64(dst + 16 + 8 * b, pos - hdr);
    }
    free(j.tmp); free(j.cs);
    return pos;
}
#endif

/* ---------------- dictionary training (COVER-style greedy segment picking)
 * samples: concatenated representative packets (a few MB is plenty). Writes up to cap bytes to dict,
 * most useful segments last (shortest offsets). Returns dict size, 0 on alloc failure. */
static inline size_t zap_dict_train(const void *samples_, size_t n, void *dict_, size_t cap) {
    enum { K = 8, SEG = 64, FLOG = 20 };
    const uint8_t *s = (const uint8_t *)samples_;
    uint8_t *dict = (uint8_t *)dict_;
    if (cap > ZAP_MAX_DIST) cap = ZAP_MAX_DIST;
    if (n < SEG * 2 || cap < SEG) { size_t c = n < cap ? n : cap; memcpy(dict, s + n - c, c); return c; }
    uint32_t *freq = (uint32_t *)calloc((size_t)1 << FLOG, 4);
    size_t nseg = cap / SEG, *pick = (size_t *)malloc(nseg * 2 * sizeof(size_t)), got = 0;
    if (!freq || !pick) { free(freq); free(pick); return 0; }
#define ZAP__HK(p) ((uint32_t)(((zap__r32(s + (p)) | ((uint64_t)zap__r32(s + (p) + 4) << 32)) * 0x9E3779B97F4A7C15ull) >> (64 - FLOG)))
    for (size_t p = 0; p + K <= n; p++) freq[ZAP__HK(p)]++;
    size_t es = n / nseg < SEG ? SEG : n / nseg;
    /* ponytail: segments may straddle sample boundaries; split by sample if that hurts */
    for (size_t lo = 0; lo + SEG <= n && got < nseg; lo += es) {
        size_t hi = (lo + es < n ? lo + es : n) - SEG, best = lo;
        uint64_t sc = 0, bsc;
        for (size_t q = lo; q <= lo + SEG - K; q++) sc += freq[ZAP__HK(q)];
        bsc = sc;
        for (size_t p = lo + 1; p <= hi; p++) {
            sc += freq[ZAP__HK(p + SEG - K)]; sc -= freq[ZAP__HK(p - 1)];
            if (sc > bsc) { bsc = sc; best = p; }
        }
        if (!bsc) continue;
        for (size_t q = best; q <= best + SEG - K; q++) freq[ZAP__HK(q)] = 0; /* don't pick the same content twice */
        pick[2 * got] = best; pick[2 * got + 1] = (size_t)bsc; got++;
    }
#undef ZAP__HK
    for (size_t i = 1; i < got; i++) /* insertion sort by score ascending: best segments end up nearest the data */
        for (size_t j = i; j && pick[2 * j - 1] > pick[2 * j + 1]; j--) {
            size_t a = pick[2 * j], b = pick[2 * j + 1];
            pick[2 * j] = pick[2 * j - 2]; pick[2 * j + 1] = pick[2 * j - 1]; pick[2 * j - 2] = a; pick[2 * j - 1] = b;
        }
    for (size_t i = 0; i < got; i++) memcpy(dict + i * SEG, s + pick[2 * i], SEG);
    free(freq); free(pick);
    return got * SEG;
}

#endif
