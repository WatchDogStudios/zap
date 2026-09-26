/* zap.h - single-header LZ77 byte codec for games (packaging + networking).
 * https://github.com/WatchDogStudios/zap   SPDX-License-Identifier: MIT   Copyright (c) 2026 WD Studios Corp.
 *
 * Blocks (raw API, you store the sizes):
 *   size_t    zap_compress   (src, n, dst, cap, zap_state*, dict);          // fast, ~250+ MB/s. 0 = didn't fit
 *   size_t    zap_compress_hc(src, n, dst, cap, zap_hc_state*, dict, depth); // slow, smaller, same format
 *   ptrdiff_t zap_decompress (src, n, dst, raw_size, dict);                 // -1 = corrupt / wrong size
 *
 * Entropy mode (packaging: smaller files, Huffman-coded streams, blocks >= ~16KB):
 *   size_t    zap_compress_entropy  (src, n, dst, cap, state, depth, dict);   // state: zap_state (depth 0) or zap_hc_state
 *   ptrdiff_t zap_decompress_entropy(src, n, dst, raw_size, dict, scratch, scratch_cap); // scratch may be NULL (mallocs)
 *
 * Frames (packaging: self-describing, independent blocks, parallel decode):
 *   zap_frame_compress(src, n, dst, cap, block_size, depth /0 = fast/ [| ZAP_ENTROPY], dict)
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

#define ZAP_VERSION "1.3.0-dev"

#define ZAP_HLOG 16
#define ZAP_HC_HLOG 17
#ifndef ZAP_WINDOW_LOG
#define ZAP_WINDOW_LOG 23 /* max match distance 2^23 - 1 (the format allows 23; smaller = compressor-side limit) */
#endif
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
static inline uint64_t zap__r64(const uint8_t *p) { /* little-endian load */
    uint64_t v = zap__r64n(p);
#ifdef ZAP_BIG_ENDIAN
    v = __builtin_bswap64(v);
#endif
    return v;
}
static inline void zap__w32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static inline uint32_t zap__hash(uint32_t v, int hlog) { return (v * 2654435761u) >> (32 - hlog); }

/* index of the first differing byte in memory order, given x = load(a) ^ load(b) != 0 */
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int zap__log2(uint32_t v) { unsigned long i; _BitScanReverse(&i, v); return (int)i; } /* v > 0 */
static inline unsigned zap__nb(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (unsigned)i >> 3; }
#else
static inline int zap__log2(uint32_t v) { return 31 - __builtin_clz(v); } /* v > 0 */
#if defined(ZAP_BIG_ENDIAN)
static inline unsigned zap__nb(uint64_t x) { return (unsigned)__builtin_clzll(x) >> 3; }
#else
static inline unsigned zap__nb(uint64_t x) { return (unsigned)__builtin_ctzll(x) >> 3; }
#endif
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

/* ---------------- optimal parse (zap_compress_hc with depth >= ZAP_OPT_DEPTH)
 * Shortest path over a window of positions: every match length the match finder offers, priced by what it
 * costs to encode (bytes for the plain format, estimated Huffman bits for entropy mode), plus a small
 * per-sequence penalty so ties go to fewer, longer matches - which also decode faster. */
#define ZAP_OPT_DEPTH 32
#define ZAP_FAST_DECODE (1 << 17) /* OR into depth (>= ZAP_OPT_DEPTH): ~2 bytes per sequence penalty -> ~15% faster decode, ~3.5% bigger */
#define ZAP__SEQ_PENALTY(depth) ((depth) & ZAP_FAST_DECODE ? 256u : 8u)
#ifndef ZAP__OPTN
#define ZAP__OPTN 2048
#endif
#ifndef ZAP__SUFF
#define ZAP__SUFF 256 /* a match this long is taken outright instead of being priced at every length */
#endif
#ifndef ZAP__E_PASSES
#define ZAP__E_PASSES 2 /* entropy mode: optimal parses re-priced from the previous parse's Huffman code lengths */
#endif
enum { ZAP__MAXC = 24 };

typedef struct { uint32_t len, off; } zap__match;
typedef struct { uint32_t price, len, off, lit, rep, rep1, rep2; } zap__opt;

/* prices in 1/16 bit */
typedef struct {
    int entropy;
    uint32_t lit[256], tok[256], oc[26], low[16], lenb, seq; /* oc: entropy v2 offset codes (0-2 repeats, 3 + log2) */
} zap__cost;

static inline void zap__cost_plain(zap__cost *c, uint32_t seq) {
    memset(c, 0, sizeof *c);
    for (int i = 0; i < 256; i++) { c->lit[i] = 128; c->tok[i] = 128; }
    c->lenb = 128; c->seq = seq;
}

/* matches at ip with strictly increasing length (repeat offset first); inserts every position up to ip */
static inline int zap__hc_all(zap_hc_state *s, const uint8_t *src, const uint8_t *ip, const uint8_t *iend, const zap_dict *d,
                              int depth, size_t *next, size_t rep, zap__match *m) {
    const size_t mask = ((size_t)1 << ZAP_WINDOW_LOG) - 1, pos = (size_t)(ip - src);
    for (size_t p = *next; p < pos; p++) {
        uint32_t h = zap__hash(zap__r32(src + p), ZAP_HC_HLOG);
        s->prev[p & mask] = s->head[h]; s->head[h] = (uint32_t)p;
    }
    if (pos > *next) *next = pos;
    uint32_t seq = zap__r32(ip);
    size_t best = 3, c = s->head[zap__hash(seq, ZAP_HC_HLOG)];
    int n = 0;
    if (rep && rep <= pos && zap__r32(ip - rep) == seq) {
        best = 4 + zap__count(ip + 4, ip - rep + 4, iend);
        m[n].len = (uint32_t)best; m[n].off = (uint32_t)rep; n++;
    }
    while (c != 0xFFFFFFFFu && c >= pos) c = s->prev[c & mask]; /* position revisited: skip newer entries */
    while (c < pos && pos - c <= ZAP_MAX_DIST && depth-- > 0 && ip + best < iend) {
        const uint8_t *q = src + c;
        if (q[best] == ip[best] && zap__r32(q) == seq) {
            size_t l = 4 + zap__count(ip + 4, q + 4, iend);
            if (l > best) {
                best = l;
                if (n == ZAP__MAXC) n--;
                m[n].len = (uint32_t)l; m[n].off = (uint32_t)(pos - c); n++;
            }
        }
        c = s->prev[c & mask];
    }
    if (d) {
        size_t o, l = zap__dmatch(d, ip, src, iend, &o);
        if (l > best) { if (n == ZAP__MAXC) n--; m[n].len = (uint32_t)l; m[n].off = (uint32_t)o; n++; }
    }
    return n;
}

/* Binary-tree match finder for the optimal parser (LZMA's bt4 idea). Every position is a node in a tree ordered
   by the bytes that follow it; a lookup walks from the hash bucket's newest position toward the ones sharing the
   longest prefix, and inserting re-links the tree around the new position on the way. Hash chains visit
   candidates newest-first and give up after `depth` steps, which on structured data misses the long matches.
   The two child links per position live in the hc state's prev array (so the tree window is half the chain
   window: 4 MB by default). Walks stop at ZAP__BT_NICE bytes, which also keeps long runs from degrading the tree. */
#ifndef ZAP__BT_NICE
#define ZAP__BT_NICE 256
#endif
static inline int zap__bt(zap_hc_state *s, const uint8_t *src, size_t pos, const uint8_t *iend, int depth, int insert, size_t best,
                          zap__match *m, int n) {
    const size_t wmask = sizeof s->prev / sizeof s->prev[0] / 2 - 1;
    uint32_t *son = s->prev;
    const uint8_t *ip = src + pos;
    size_t avail = (size_t)(iend - ip), limit = avail < ZAP__BT_NICE ? avail : ZAP__BT_NICE, len0 = 0, len1 = 0;
    uint32_t *head = &s->head[zap__hash(zap__r32(ip), ZAP_HC_HLOG)], dummy[2];
    size_t cur = *head;
    uint32_t *p1 = insert ? son + 2 * (pos & wmask) : dummy, *p0 = p1 + 1;
    if (insert) *head = (uint32_t)pos;
    while (cur < pos && pos - cur <= wmask && depth-- > 0) {
        uint32_t *pair = son + 2 * (cur & wmask);
        const uint8_t *q = src + cur;
        size_t len = len0 < len1 ? len0 : len1;
        if (q[len] == ip[len]) {
            len += 1 + zap__count(ip + len + 1, q + len + 1, ip + limit);
            if (len > best && m) {
                best = len;
                if (n == ZAP__MAXC) n--;
                m[n].len = (uint32_t)len; m[n].off = (uint32_t)(pos - cur); n++;
            }
            if (len >= limit) { *p1 = pair[0]; *p0 = pair[1]; return n; } /* equal: the new node takes cur's place */
        }
        if (q[len] < ip[len]) { *p1 = (uint32_t)cur; p1 = pair + 1; cur = *p1; len1 = len; }
        else { *p0 = (uint32_t)cur; p0 = pair; cur = *p0; len0 = len; }
        if (!insert) { p1 = dummy; p0 = dummy + 1; }
    }
    *p0 = *p1 = 0xFFFFFFFFu;
    return n;
}

/* the optimal parser's match list at pos (strictly increasing lengths, repeat offset first): inserts any
   positions the parser skipped, then searches pos (inserting it the first time it's visited) */
static inline int zap__bt_all(zap_hc_state *s, const uint8_t *src, const uint8_t *ip, const uint8_t *iend, const zap_dict *d,
                              int depth, size_t *next, size_t rep, zap__match *m) {
    size_t pos = (size_t)(ip - src), best = 3;
    for (size_t p = *next; p < pos; p++) zap__bt(s, src, p, iend, depth < 8 ? depth : 8, 1, 0, NULL, 0); /* skipped: shallow insert */
    int n = 0;
    if (rep && rep <= pos && zap__r32(ip - rep) == zap__r32(ip)) {
        best = 4 + zap__count(ip + 4, ip - rep + 4, iend);
        m[n].len = (uint32_t)best; m[n].off = (uint32_t)rep; n++;
    }
    n = zap__bt(s, src, pos, iend, depth, pos >= *next, best, m, n);
    if (pos >= *next) *next = pos + 1;
    if (d) {
        size_t o, l = zap__dmatch(d, ip, src, iend, &o), b = n ? m[n - 1].len : 3;
        if (l > b) { if (n == ZAP__MAXC) n--; m[n].len = (uint32_t)l; m[n].off = (uint32_t)o; n++; }
    }
    return n;
}

static inline uint32_t zap__ext_bytes(size_t v) { return v < 15 ? 0 : (uint32_t)((v - 15) / 255 + 1); }

/* entropy v2 offset coding: which repeat slot (0-2) an offset hits, or -1 */
static inline int zap__rep_slot(size_t off, const uint32_t r[3]) { return off == r[0] ? 0 : off == r[1] ? 1 : off == r[2] ? 2 : -1; }
/* repeat-offset update after a match at `off` (LZMA-style move to front) */
static inline void zap__rep_push(uint32_t r[3], size_t off) {
    uint32_t o = (uint32_t)off;
    if (o == r[0]) return;
    if (o != r[1]) r[2] = r[1];
    r[1] = r[0]; r[0] = o;
}

static inline uint32_t zap__match_price(const zap__cost *c, size_t len, size_t off, size_t lit, const uint32_t rep[3]) {
    uint32_t p = c->seq + zap__ext_bytes(len - 4) * c->lenb;
    if (!c->entropy) return p + c->tok[0] + (off < 0x8000 ? 256u : 384u);
    p += c->tok[(lit < 15 ? lit : 15) << 4 | (len - 4 < 15 ? len - 4 : 15)];
    int slot = zap__rep_slot(off, rep);
    if (slot >= 0) return p + c->oc[slot];
    int k = zap__log2((uint32_t)off);
    return p + c->oc[k + 3] + (k >= 4 ? c->low[off & 15] + 16u * (uint32_t)(k - 4) : 16u * (uint32_t)k);
}

static inline size_t zap__compress_opt(const uint8_t *src, size_t n, uint8_t *dst, size_t cap, zap_hc_state *s, const zap_dict *d,
                                       int depth, const zap__cost *cm) {
    const uint8_t *iend = src + n;
    uint8_t *op = dst, *oend = dst + cap;
    zap__opt *opt = (zap__opt *)malloc(sizeof(zap__opt) * (ZAP__OPTN + ZAP__SUFF + 2));
    zap__match m[ZAP__MAXC];
    size_t next = 0, anchor = 0, start = 0, limit = n >= 13 ? n - 12 : 0;
    uint32_t reps[3] = { 0, 0, 0 };
    if (!opt) return 0;
    memset(s->head, 0xFF, sizeof s->head);
    while (start < limit) {
        size_t last = 0, p, fl = 0, fo = 0;
        opt[0].price = 0; opt[0].len = 0; opt[0].lit = (uint32_t)(start - anchor);
        opt[0].rep = reps[0]; opt[0].rep1 = reps[1]; opt[0].rep2 = reps[2];
        for (p = 0; p <= last && p < ZAP__OPTN && start + p < limit; p++) {
            zap__opt o = opt[p];
            size_t pos = start + p;
            /* one more literal */
            uint32_t lp = o.price + cm->lit[src[pos]] + (zap__ext_bytes(o.lit + 1) - zap__ext_bytes(o.lit)) * cm->lenb;
            if (p + 1 > last) { opt[p + 1].price = 0xFFFFFFFFu; last = p + 1; }
            if (lp < opt[p + 1].price) { opt[p + 1] = o; opt[p + 1].price = lp; opt[p + 1].len = 0; opt[p + 1].lit = o.lit + 1; }
            int nm = zap__bt_all(s, src, src + pos, iend, d, depth, &next, o.rep, m);
            if (nm && m[nm - 1].len >= ZAP__SUFF) { fl = m[nm - 1].len; fo = m[nm - 1].off; break; } /* long match: just take it */
            const uint32_t orep[3] = { o.rep, o.rep1, o.rep2 };
            if (cm->entropy) /* repeat offsets 1 and 2: cheap to code, so worth trying even when shorter than chain matches */
                for (int r = 1; r < 3; r++) {
                    size_t ro = orep[r];
                    if (!ro || ro > pos || ro == orep[0] || (r == 2 && ro == orep[1]) || zap__r32(src + pos - ro) != zap__r32(src + pos)) continue;
                    size_t rl = 4 + zap__count(src + pos + 4, src + pos - ro + 4, iend);
                    if (rl >= ZAP__SUFF) rl = ZAP__SUFF - 1;
                    for (size_t L = 4; L <= rl; L++) {
                        uint32_t mp = o.price + zap__match_price(cm, L, ro, o.lit, orep);
                        while (last < p + L) opt[++last].price = 0xFFFFFFFFu;
                        if (mp < opt[p + L].price) {
                            uint32_t nr[3] = { orep[0], orep[1], orep[2] };
                            zap__rep_push(nr, ro);
                            opt[p + L].price = mp; opt[p + L].len = (uint32_t)L; opt[p + L].off = (uint32_t)ro; opt[p + L].lit = 0;
                            opt[p + L].rep = nr[0]; opt[p + L].rep1 = nr[1]; opt[p + L].rep2 = nr[2];
                        }
                    }
                }
            for (int i = 0; i < nm; i++) {
                size_t lo = i ? m[i - 1].len + 1 : 4;
                uint32_t nr[3] = { orep[0], orep[1], orep[2] };
                zap__rep_push(nr, m[i].off);
                for (size_t L = lo; L <= m[i].len; L++) {
                    uint32_t mp = o.price + zap__match_price(cm, L, m[i].off, o.lit, orep);
                    while (last < p + L) opt[++last].price = 0xFFFFFFFFu;
                    if (mp < opt[p + L].price) {
                        opt[p + L].price = mp; opt[p + L].len = (uint32_t)L; opt[p + L].off = m[i].off; opt[p + L].lit = 0;
                        opt[p + L].rep = nr[0]; opt[p + L].rep1 = nr[1]; opt[p + L].rep2 = nr[2];
                    }
                }
            }
        }
        /* end the window at the furthest settled position reached by a match (or the forced long match) */
        size_t end = fl ? p : 0;
        if (!fl) for (size_t e = p < last ? p : last; e > 0; e--) if (opt[e].len) { end = e; break; }
        if (!end && !fl) { start += p ? p : 1; continue; } /* nothing matched: the literals carry into the next window */
        /* backtrack: matches along the path to end, then emit them in order */
        size_t cnt = 0, t = end;
        while (t > 0) { if (opt[t].len) { cnt++; t -= opt[t].len; } else t--; }
        size_t *seqs = (size_t *)malloc((cnt + 1) * 3 * sizeof(size_t)), k = cnt;
        if (!seqs) { free(opt); return 0; }
        for (t = end; t > 0;) {
            if (opt[t].len) { k--; seqs[3 * k] = start + t - opt[t].len; seqs[3 * k + 1] = opt[t].len; seqs[3 * k + 2] = opt[t].off; t -= opt[t].len; }
            else t--;
        }
        if (fl) { seqs[3 * cnt] = start + p; seqs[3 * cnt + 1] = fl; seqs[3 * cnt + 2] = fo; cnt++; }
        for (k = 0; k < cnt; k++) {
            if (!(op = zap__emit(op, oend, src + anchor, seqs[3 * k] - anchor, seqs[3 * k + 2], seqs[3 * k + 1]))) { free(seqs); free(opt); return 0; }
            anchor = seqs[3 * k] + seqs[3 * k + 1];
            zap__rep_push(reps, seqs[3 * k + 2]);
        }
        free(seqs);
        start = anchor;
    }
    free(opt);
    return zap__finish(op, oend, src + anchor, iend, dst);
}

/* depth: chain steps per position. 16 = quick lazy parse; >= ZAP_OPT_DEPTH (32) = optimal parse (64 = good default) */
/* On incompressible data (already-compressed audio, images, archives) the hc search walks full chains of hash
   collisions, a cache miss each, for nothing: a 2.4 MB .tar.gz took 0.8 s at depth 16. Probe three 128 KB samples
   with the fast compressor (the hc state's prev table is the scratch); if none saves 1/64, the caller uses the fast
   compressor for the block. Compressible blocks are unaffected.
   ponytail: three samples can miss a compressible stretch between them; probe more spots if that shows up in real content. */
static inline int zap__hc_skip(const uint8_t *src, size_t n, zap_hc_state *s) {
    enum { P = 128 << 10 };
    if (n < 4 * (size_t)P || sizeof s->prev < sizeof(zap_state) + 2 * (size_t)P) return 0;
    zap_state *t = (zap_state *)(void *)s->prev;
    uint8_t *o = (uint8_t *)(s->prev + (1 << ZAP_HLOG));
    for (int k = 0; k < 3; k++) {
        size_t c = zap_compress(src + (n - P) / 2 * (size_t)k, P, o, zap_bound(P), t, NULL);
        if (c && c < P - P / 64) return 0;
    }
    return 1;
}

static inline size_t zap_compress_hc(const void *src_, size_t n, void *dst_, size_t cap,
                                     zap_hc_state *s, const zap_dict *d, int depth) {
    const uint8_t *src = (const uint8_t *)src_, *ip = src, *anchor = src, *iend = src + n;
    uint8_t *op = (uint8_t *)dst_, *oend = op + cap;
    if (zap__hc_skip(src, n, s)) return zap_compress(src, n, dst_, cap, (zap_state *)(void *)s->prev, d);
    size_t next = 0;
    uint32_t seq = ZAP__SEQ_PENALTY(depth);
    depth &= 0xFFFF;
    if (depth >= ZAP_OPT_DEPTH) {
        zap__cost cm;
        zap__cost_plain(&cm, seq);
        return zap__compress_opt(src, n, op, cap, s, d, depth, &cm);
    }
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
        /* hot path: short literals with headroom -> fixed-size copies */
        if (lit < 15 && iend - ip >= 32 && oend - op >= 32) {
            memcpy(op, ip, 16);
            op += lit; ip += lit;
            /* branch-free 2/3-byte offset: near/far offsets mix unpredictably in big blocks */
            size_t v = (size_t)ip[0] | ((size_t)ip[1] << 8), far = v >> 15, ml = tok & 15;
            size_t off = (v & 0x7FFF) | (((size_t)ip[2] << 15) & ((size_t)0 - far)), have;
            ip += 2 + far;
            have = (size_t)(op - ostart);
            if (ml < 15) { /* short match: one 18-byte copy */
                ml += 4;
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
            ml += 4 + zap__ext(&ip, iend, &err); /* long match: 16-byte chunks while there's room */
            if (err || off == 0 || off > have + dlen || ml > (size_t)(oend - op)) return -1;
            if (off >= 16 && off <= have && (size_t)(oend - op) >= ml + 16) {
                const uint8_t *m = op - off;
                uint8_t *e = op + ml;
                do { memcpy(op, m, 16); op += 16; m += 16; } while (op < e);
                op = e;
            } else {
                zap__dict_copy(op, ostart, oend, d, off, ml);
                op += ml;
            }
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

/* ---------------- canonical Huffman (order-0, max code length 11, LSB-first bits)
 * layout: 128 bytes of 4-bit code lengths (symbol 2k in the low nibble), then the bitstream */
enum { ZAP__HMAX = 11 };

static inline void zap__hlens(const uint32_t *freq, uint8_t len[256]) {
    uint32_t f[256], w[512];
    int sym[256], parent[512], d[512];
    memcpy(f, freq, sizeof f);
    for (;;) {
        int n = 0;
        memset(len, 0, 256);
        for (int i = 0; i < 256; i++) if (f[i]) sym[n++] = i;
        if (n == 0) return;
        if (n == 1) { len[sym[0]] = 1; return; }
        for (int i = 1; i < n; i++) /* sort leaves by weight */
            for (int j = i; j && f[sym[j - 1]] > f[sym[j]]; j--) { int t = sym[j]; sym[j] = sym[j - 1]; sym[j - 1] = t; }
        for (int i = 0; i < n; i++) w[i] = f[sym[i]];
        int li = 0, ii = n, next = n; /* two-queue construction: internal nodes come out in weight order */
        for (int k = 0; k < n - 1; k++) {
            int ab[2];
            for (int t = 0; t < 2; t++) ab[t] = li < n && (ii >= next || w[li] <= w[ii]) ? li++ : ii++;
            w[next] = w[ab[0]] + w[ab[1]]; parent[ab[0]] = parent[ab[1]] = next; next++;
        }
        int maxd = 0;
        d[next - 1] = 0;
        for (int i = next - 2; i >= 0; i--) { d[i] = d[parent[i]] + 1; if (i < n && d[i] > maxd) maxd = d[i]; }
        if (maxd <= ZAP__HMAX) { for (int i = 0; i < n; i++) len[sym[i]] = (uint8_t)d[i]; return; }
        for (int i = 0; i < 256; i++) if (f[i]) f[i] = (f[i] >> 1) | 1; /* flatten and retry */
    }
}

/* LSB-first codes: canonical code, bit-reversed */
static inline void zap__hcodes(const uint8_t len[256], uint16_t code[256]) {
    int count[16] = { 0 }, next[16] = { 0 };
    for (int i = 0; i < 256; i++) count[len[i]]++;
    count[0] = 0;
    for (int l = 1, c = 0; l < 16; l++) { c = (c + count[l - 1]) << 1; next[l] = c; }
    for (int i = 0; i < 256; i++) {
        if (!len[i]) continue;
        int c = next[len[i]]++, r = 0;
        for (int b = 0; b < len[i]; b++) r |= ((c >> b) & 1) << (len[i] - 1 - b);
        code[i] = (uint16_t)r;
    }
}

/* returns coded size, 0 if it doesn't fit in cap */
static inline size_t zap__henc(const uint8_t *src, size_t n, uint8_t *out, size_t cap) {
    uint32_t freq[256] = { 0 };
    uint8_t len[256];
    uint16_t code[256];
    if (cap < 128) return 0;
    for (size_t i = 0; i < n; i++) freq[src[i]]++;
    zap__hlens(freq, len);
    zap__hcodes(len, code);
    for (int i = 0; i < 128; i++) out[i] = (uint8_t)(len[2 * i] | len[2 * i + 1] << 4);
    uint8_t *o = out + 128, *oe = out + cap;
    uint64_t acc = 0;
    int nb = 0;
    for (size_t i = 0; i < n; i++) {
        acc |= (uint64_t)code[src[i]] << nb; nb += len[src[i]];
        while (nb >= 8) { if (o >= oe) return 0; *o++ = (uint8_t)acc; acc >>= 8; nb -= 8; }
    }
    if (nb) { if (o >= oe) return 0; *o++ = (uint8_t)acc; }
    return (size_t)(o - out);
}

/* decode table from the 128-byte length header: entry = symbol | length << 8, 0 = unused code. 0 ok, -1 corrupt. */
static inline int zap__htable(const uint8_t *in, uint16_t table[1 << ZAP__HMAX]) {
    uint8_t len[256];
    uint16_t code[256];
    uint32_t kraft = 0;
    for (int i = 0; i < 128; i++) { len[2 * i] = in[i] & 15; len[2 * i + 1] = in[i] >> 4; }
    for (int i = 0; i < 256; i++) {
        if (len[i] > ZAP__HMAX) return -1;
        if (len[i]) kraft += 1u << (ZAP__HMAX - len[i]);
    }
    if (kraft > 1u << ZAP__HMAX) return -1; /* over-subscribed code */
    zap__hcodes(len, code);
    memset(table, 0, sizeof(uint16_t) << ZAP__HMAX);
    for (int i = 0; i < 256; i++)
        if (len[i]) for (int j = code[i]; j < 1 << ZAP__HMAX; j += 1 << len[i]) table[j] = (uint16_t)(i | len[i] << 8);
    return 0;
}

/* bit reader. Invariant: bit nb of acc is bit 0 of *p. */
typedef struct { const uint8_t *p, *e; uint64_t acc; int nb; } zap__hbits;

/* careful path: byte refills, bounds-checked; used for stream tails */
static inline int zap__hrun(const uint16_t *table, zap__hbits *b, uint8_t *out, size_t cnt) {
    const unsigned mask = (1u << ZAP__HMAX) - 1;
    for (size_t i = 0; i < cnt; i++) {
        while (b->nb <= 56 && b->p < b->e) { b->acc |= (uint64_t)*b->p++ << b->nb; b->nb += 8; }
        int t = table[b->acc & mask], l = t >> 8;
        if (!l || l > b->nb) return -1; /* unused code, or ran past the end */
        out[i] = (uint8_t)t; b->acc >>= l; b->nb -= l;
    }
    return 0;
}

/* one 64-bit refill (needs 8 readable bytes): at least 56 valid bits, enough for 4 symbols */
#define ZAP__HREFILL(b) do { (b).acc |= zap__r64((b).p) << (b).nb; (b).p += (63 - (b).nb) >> 3; (b).nb |= 56; } while (0)
/* fast-path symbol: no validity check. An unused code (length 0) just repeats a symbol without consuming bits;
   output stays in bounds, and the caller's structural checks reject the result. Tails are fully checked. */
#define ZAP__HSYM(b, dst) do { int t_ = table[(b).acc & mask], l_ = t_ >> 8; \
    (dst) = (uint8_t)t_; (b).acc >>= l_; (b).nb -= l_; } while (0)

/* single-stream Huffman (video streams); decodes exactly raw symbols. 0 ok, -1 corrupt. */
static inline int zap__hdec(const uint8_t *in, size_t n, uint8_t *out, size_t raw) {
    uint16_t table[1 << ZAP__HMAX];
    const unsigned mask = (1u << ZAP__HMAX) - 1;
    if (n < 128 || zap__htable(in, table)) return -1;
    zap__hbits b = { in + 128, in + n, 0, 0 };
    size_t i = 0;
    while (raw - i >= 4 && b.e - b.p >= 8) {
        ZAP__HREFILL(b);
        for (int k = 0; k < 4; k++) ZAP__HSYM(b, out[i + k]);
        i += 4;
    }
    return zap__hrun(table, &b, out + i, raw - i);
}

/* 4-stream Huffman (entropy mode): symbols split into 4 contiguous quarters, each its own bitstream, so the
 * decoder runs 4 independent dependency chains. Layout: 128-byte lengths, u32 size of streams 0..2, 4 streams. */
static inline size_t zap__henc4(const uint8_t *src, size_t n, uint8_t *out, size_t cap) {
    uint32_t freq[256] = { 0 };
    uint8_t len[256];
    uint16_t code[256];
    if (cap < 140) return 0;
    for (size_t i = 0; i < n; i++) freq[src[i]]++;
    zap__hlens(freq, len);
    zap__hcodes(len, code);
    for (int i = 0; i < 128; i++) out[i] = (uint8_t)(len[2 * i] | len[2 * i + 1] << 4);
    uint8_t *o = out + 140, *oe = out + cap;
    size_t q = (n + 3) / 4;
    for (int k = 0; k < 4; k++) {
        size_t lo = (size_t)k * q < n ? (size_t)k * q : n, hi = lo + q < n ? lo + q : n;
        uint8_t *start = o;
        uint64_t acc = 0;
        int nb = 0;
        for (size_t i = lo; i < hi; i++) {
            acc |= (uint64_t)code[src[i]] << nb; nb += len[src[i]];
            while (nb >= 8) { if (o >= oe) return 0; *o++ = (uint8_t)acc; acc >>= 8; nb -= 8; }
        }
        if (nb) { if (o >= oe) return 0; *o++ = (uint8_t)acc; }
        if (k < 3) zap__w32(out + 128 + 4 * k, (uint32_t)(o - start));
    }
    return (size_t)(o - out);
}

static inline int zap__hdec4(const uint8_t *in, size_t n, uint8_t *out, size_t raw) {
    uint16_t table[1 << ZAP__HMAX];
    const unsigned mask = (1u << ZAP__HMAX) - 1;
    if (n < 140 || zap__htable(in, table)) return -1;
    size_t sz[3] = { zap__r32(in + 128), zap__r32(in + 132), zap__r32(in + 136) }, avail = n - 140;
    if (sz[0] > avail || sz[1] > avail - sz[0] || sz[2] > avail - sz[0] - sz[1]) return -1;
    const uint8_t *s0 = in + 140, *s1 = s0 + sz[0], *s2 = s1 + sz[1], *s3 = s2 + sz[2];
    zap__hbits b0 = { s0, s1, 0, 0 }, b1 = { s1, s2, 0, 0 }, b2 = { s2, s3, 0, 0 }, b3 = { s3, in + n, 0, 0 };
    size_t q = (raw + 3) / 4, c3 = raw > 3 * q ? raw - 3 * q : 0, i = 0; /* quarters 0-2 have q symbols, 3 has c3 <= q */
    uint8_t *o0 = out, *o1 = out + q, *o2 = out + 2 * q, *o3 = out + 3 * q;
    if (raw < 4 * q && raw < 3 * q) { /* tiny input: some quarters are short or empty */
        zap__hbits *bs[4] = { &b0, &b1, &b2, &b3 };
        for (int k = 0; k < 4; k++) {
            size_t lo = (size_t)k * q < raw ? (size_t)k * q : raw, hi = lo + q < raw ? lo + q : raw;
            if (zap__hrun(table, bs[k], out + lo, hi - lo)) return -1;
        }
        return 0;
    }
    while (c3 - i >= 5 && b0.e - b0.p >= 8 && b1.e - b1.p >= 8 && b2.e - b2.p >= 8 && b3.e - b3.p >= 8) {
        ZAP__HREFILL(b0); ZAP__HREFILL(b1); ZAP__HREFILL(b2); ZAP__HREFILL(b3);
        for (int k = 0; k < 5; k++) { /* 5 x 11 bits <= 56 */
            ZAP__HSYM(b0, o0[i + k]); ZAP__HSYM(b1, o1[i + k]); ZAP__HSYM(b2, o2[i + k]); ZAP__HSYM(b3, o3[i + k]);
        }
        i += 5;
    }
    if (zap__hrun(table, &b0, o0 + i, q - i) || zap__hrun(table, &b1, o1 + i, q - i) ||
        zap__hrun(table, &b2, o2 + i, q - i) || zap__hrun(table, &b3, o3 + i, c3 - i)) return -1;
    return 0;
}
#undef ZAP__HREFILL
#undef ZAP__HSYM

/* ---------------- entropy mode: the LZ parse re-coded as Huffman streams. For packaging blocks (>= ~16KB);
 * small packets should stay on the plain format (5 x 128-byte code tables would outweigh the gain).
 * Block layout (little-endian):
 *   u32 n_lits, u32 n_seq, u32 n_lens, u32 n_extra
 *   4 streams - literals, tokens, length bytes, offset codes - each: u8 method (0 raw, 1 4-way Huffman) + u32 size + data
 *   n_extra bytes of offset extra bits (LSB-first)
 * Sequence i: token = literal nibble | (match - 4) nibble, 15 continues as 255-runs in the length stream;
 * offset code 0 = repeat the previous offset, c in 1..23 = offset in [2^(c-1), 2^c) plus c - 1 extra bits.
 * Literals left over after the last sequence end the block. */
#define ZAP_ENTROPY (1 << 16) /* OR into a frame's depth: entropy blocks, version-2 frame */

/* decoder scratch that always suffices for a block of raw bytes */
static inline size_t zap_entropy_scratch(size_t raw) { return 2 * raw + raw / 255 + 128; }

static inline uint8_t *zap__e_stream(uint8_t *op, uint8_t *oend, const uint8_t *s, size_t n) {
    if (!op || oend - op < 5) return NULL;
    size_t room = (size_t)(oend - op) - 5, h = 0;
    if (n > 64) h = zap__henc4(s, n, op + 5, room < n - 1 ? room : n - 1);
    if (h) { *op = 1; zap__w32(op + 1, (uint32_t)h); return op + 5 + h; }
    if (room < n) return NULL;
    *op = 0; zap__w32(op + 1, (uint32_t)n); memcpy(op + 5, s, n);
    return op + 5 + n;
}

/* re-code a (trusted, self-produced) plain block of raw bytes; 0 = doesn't fit in cap */
/* Entropy block, version 2 (flagged by the top bit of the first word):
     u32 n_literals | 1 << 31, u32 n_sequences, u32 n_length_bytes, u32 n_extra_bytes, u32 n_low
     5 streams (method + size + data): literals, tokens, length bytes, offset codes, offset low nibbles
     offset extra bits (LSB-first)
   Offset code 0-2: repeat offset 0-2 (move-to-front). Code 3 + k: an offset in [2^k, 2^(k+1)); for k >= 4 its low
   4 bits come from the low-nibble stream (two nibbles per byte, first in the low half, n_low counts nibbles) and the k - 4 bits above them are extra bits, else k extra bits. Aligned
   data (DXT blocks, vertex strides, PCM frames) makes the low nibbles very predictable. */
static inline size_t zap__e_encode(const uint8_t *lz, size_t lzn, size_t raw, uint8_t *dst, size_t cap) {
    size_t maxseq = raw / 4 + 2, nl = 0, ns = 0, nlen = 0, nlow = 0;
    uint32_t rep[3] = { 0, 0, 0 };
    uint8_t *buf = (uint8_t *)malloc(raw + 6 * maxseq + lzn + 16);
    if (!buf || cap < 20) { free(buf); return 0; }
    uint8_t *lits = buf, *toks = lits + raw, *offc = toks + maxseq, *low = offc + maxseq, *xb = low + maxseq, *lens = xb + 3 * maxseq + 16, *xp = xb;
    uint64_t acc = 0;
    int nb = 0;
    for (const uint8_t *ip = lz, *ie = lz + lzn;;) {
        unsigned tok = *ip++;
        size_t ll = tok >> 4, mark = nlen;
        if (ll == 15) { uint8_t b; do { b = *ip++; lens[nlen++] = b; ll += b; } while (b == 255); }
        memcpy(lits + nl, ip, ll); nl += ll; ip += ll;
        if (ip >= ie) { nlen = mark; break; } /* trailing literals: implied by n_lits */
        size_t off = (size_t)ip[0] | ((size_t)ip[1] << 8);
        ip += 2;
        if (off & 0x8000) off = (off & 0x7FFF) | ((size_t)*ip++ << 15);
        if ((tok & 15) == 15) { uint8_t b; do { b = *ip++; lens[nlen++] = b; } while (b == 255); }
        toks[ns] = (uint8_t)tok;
        int slot = zap__rep_slot(off, rep);
        if (slot >= 0) offc[ns] = (uint8_t)slot;
        else {
            int k = zap__log2((uint32_t)off), xk = k >= 4 ? k - 4 : k;
            size_t x = off - ((size_t)1 << k);
            offc[ns] = (uint8_t)(k + 3);
            if (k >= 4) { if (nlow & 1) low[nlow >> 1] |= (uint8_t)((x & 15) << 4); else low[nlow >> 1] = (uint8_t)(x & 15); nlow++; x >>= 4; }
            acc |= (uint64_t)x << nb; nb += xk;
            while (nb >= 8) { *xp++ = (uint8_t)acc; acc >>= 8; nb -= 8; }
        }
        zap__rep_push(rep, off);
        ns++;
    }
    if (nb) *xp++ = (uint8_t)acc;
    size_t nx = (size_t)(xp - xb);
    uint8_t *op = dst, *oend = dst + cap;
    zap__w32(op, (uint32_t)nl | 0x80000000u); zap__w32(op + 4, (uint32_t)ns); zap__w32(op + 8, (uint32_t)nlen); zap__w32(op + 12, (uint32_t)nx);
    zap__w32(op + 16, (uint32_t)nlow);
    op = zap__e_stream(op + 20, oend, lits, nl);
    op = zap__e_stream(op, oend, toks, ns);
    op = zap__e_stream(op, oend, lens, nlen);
    op = zap__e_stream(op, oend, offc, ns);
    op = zap__e_stream(op, oend, low, (nlow + 1) / 2);
    size_t r = 0;
    if (op && (size_t)(oend - op) >= nx) { memcpy(op, xb, nx); r = (size_t)(op + nx - dst); }
    free(buf);
    return r;
}

/* entropy-mode prices from a finished parse: Huffman code lengths of each stream, unseen symbols priced high */
static inline void zap__cost_from_lz(const uint8_t *lz, size_t lzn, zap__cost *c, uint32_t seq) {
    uint32_t fl[256] = { 0 }, ft[256] = { 0 }, fo[256] = { 0 }, flow[256] = { 0 }, fx = 0, nx = 0;
    uint8_t len[256];
    uint32_t rep[3] = { 0, 0, 0 };
    for (const uint8_t *ip = lz, *ie = lz + lzn;;) {
        unsigned tok = *ip++;
        size_t ll = tok >> 4;
        if (ll == 15) { uint8_t b; do { b = *ip++; ll += b; fx++; } while (b == 255); }
        for (size_t i = 0; i < ll; i++) fl[ip[i]]++;
        ip += ll;
        if (ip >= ie) break;
        size_t off = (size_t)ip[0] | ((size_t)ip[1] << 8);
        ip += 2;
        if (off & 0x8000) off = (off & 0x7FFF) | ((size_t)*ip++ << 15);
        if ((tok & 15) == 15) { uint8_t b; do { b = *ip++; fx++; } while (b == 255); }
        ft[tok]++; nx++;
        int slot = zap__rep_slot(off, rep), k = zap__log2((uint32_t)off);
        if (slot >= 0) fo[slot]++;
        else { fo[k + 3]++; if (k >= 4) flow[off & 15]++; }
        zap__rep_push(rep, off);
    }
    memset(c, 0, sizeof *c);
    c->entropy = 1;
    zap__hlens(fl, len); for (int i = 0; i < 256; i++) c->lit[i] = 16u * (len[i] ? len[i] : ZAP__HMAX + 2);
    zap__hlens(ft, len); for (int i = 0; i < 256; i++) c->tok[i] = 16u * (len[i] ? len[i] : ZAP__HMAX + 2);
    zap__hlens(fo, len); for (int i = 0; i < 26; i++) c->oc[i] = 16u * (len[i] ? len[i] : ZAP__HMAX + 2);
    zap__hlens(flow, len); for (int i = 0; i < 16; i++) c->low[i] = 16u * (len[i] ? len[i] : ZAP__HMAX + 2);
    c->lenb = 16 * 7; c->seq = seq;
    (void)fx; (void)nx;
}

/* depth 0: fast parse (state = zap_state*), else hc chain depth (state = zap_hc_state*). 0 = doesn't fit in cap.
   depth >= ZAP_OPT_DEPTH: two optimal parses, the second priced with the first one's Huffman code lengths. */
static inline size_t zap_compress_entropy(const void *src, size_t n, void *dst, size_t cap, void *state, int depth, const zap_dict *d) {
    size_t lcap = zap_bound(n), r = 0;
    uint8_t *lz = (uint8_t *)malloc(lcap);
    if (!lz) return 0;
    uint32_t seq = ZAP__SEQ_PENALTY(depth);
    int fd = depth & ZAP_FAST_DECODE;
    depth &= 0xFFFF;
    int skip = depth && zap__hc_skip((const uint8_t *)src, n, (zap_hc_state *)state); /* incompressible: fast parse, Huffman still applies */
    size_t ln = skip ? zap_compress(src, n, lz, lcap, (zap_state *)(void *)((zap_hc_state *)state)->prev, d)
              : depth ? zap_compress_hc(src, n, lz, lcap, (zap_hc_state *)state, d, depth | fd) : zap_compress(src, n, lz, lcap, (zap_state *)state, d);
    for (int pass = 0; ln && depth >= ZAP_OPT_DEPTH && !skip && pass < ZAP__E_PASSES; pass++) {
        zap__cost cm;
        zap__cost_from_lz(lz, ln, &cm, seq);
        ln = zap__compress_opt((const uint8_t *)src, n, lz, lcap, (zap_hc_state *)state, d, depth, &cm);
    }
    if (ln) r = zap__e_encode(lz, ln, n, (uint8_t *)dst, cap);
    free(lz);
    return r;
}

/* raw_size must be exact. scratch: zap_entropy_scratch(raw_size) bytes, or NULL to malloc. Returns raw_size or -1. */
static inline ptrdiff_t zap_decompress_entropy(const void *src_, size_t n, void *dst_, size_t raw_size, const zap_dict *d, void *scratch, size_t scratch_cap) {
    const uint8_t *ip = (const uint8_t *)src_, *iend = ip + n;
    if (n < 16) return -1;
    int v2 = (zap__r32(ip) >> 31) != 0;
    size_t nl = zap__r32(ip) & 0x7FFFFFFFu, ns = zap__r32(ip + 4), nlen = zap__r32(ip + 8), nx = zap__r32(ip + 12), nlow = 0;
    ip += 16;
    if (v2) { if (n < 20) return -1; nlow = zap__r32(ip); ip += 4; }
    if (nl > raw_size || ns > raw_size / 4 + 1 || nlen > 2 * ns + raw_size / 255 + 1 || nx > 3 * ns + 8 || nlow > ns) return -1;
    int nstreams = v2 ? 5 : 4;
    size_t need = nl + 2 * ns + nlen + (nlow + 1) / 2, cnt[5] = { nl, ns, nlen, ns, (nlow + 1) / 2 };
    uint8_t *mem = (uint8_t *)scratch, *own = NULL;
    if (!mem || scratch_cap < need) { if (!(mem = own = (uint8_t *)malloc(need ? need : 1))) return -1; }
    const uint8_t *st[5] = { NULL, NULL, NULL, NULL, NULL };
    uint8_t *w = mem;
    ptrdiff_t r = -1;
    for (int k = 0; k < nstreams; k++) {
        if (iend - ip < 5) goto out;
        int m = ip[0];
        size_t sz = zap__r32(ip + 1);
        ip += 5;
        if (sz > (size_t)(iend - ip)) goto out;
        if (m == 0) { if (sz != cnt[k]) goto out; st[k] = ip; }
        else if (m == 1) { if (zap__hdec4(ip, sz, w, cnt[k])) goto out; st[k] = w; w += cnt[k]; }
        else goto out;
        ip += sz;
    }
    if ((size_t)(iend - ip) != nx) goto out;
    {
        const uint8_t *lp = st[0], *le = lp + nl, *tp = st[1], *lnp = st[2], *lne = lnp + nlen, *oc = st[3], *lwb = st[4], *xp = ip;
        size_t lwi = 0; /* low nibbles used */
        uint8_t *op = (uint8_t *)dst_, *ostart = op, *oend = op + raw_size;
        size_t rep = 0, rep1 = 0, rep2 = 0, dlen = d ? d->len : 0;
        uint64_t xacc = 0;
        int xnb = 0, err = 0;
        const unsigned base = v2 ? 3 : 1, cmax = v2 ? 25 : 23;
        for (size_t i = 0; i < ns; i++) {
            unsigned tok = tp[i], c = oc[i];
            size_t ll = tok >> 4, ml = tok & 15, off;
            if (c < base) { /* repeat offset (v1: only slot 0) */
                if (c == 0) off = rep;
                else if (c == 1) { off = rep1; rep1 = rep; rep = off; }
                else { off = rep2; rep2 = rep1; rep1 = rep; rep = off; }
                if (!off) goto out;
            } else {
                if (c > cmax) goto out;
                int k = (int)(c - base), xk = v2 && k >= 4 ? k - 4 : k;
                if (xnb < xk) { /* 64-bit refill while 8 bytes remain, bytewise at the end */
                    if (iend - xp >= 8) { xacc |= zap__r64(xp) << xnb; xp += (63 - xnb) >> 3; xnb |= 56; }
                    else while (xnb < xk) { if (xp >= iend) goto out; xacc |= (uint64_t)*xp++ << xnb; xnb += 8; }
                }
                size_t x = (size_t)(xacc & (((uint64_t)1 << xk) - 1));
                xacc >>= xk; xnb -= xk;
                if (v2 && k >= 4) { if (lwi >= nlow) goto out; x = x << 4 | (size_t)(lwb[lwi >> 1] >> (lwi & 1) * 4 & 15); lwi++; }
                off = ((size_t)1 << k) + x;
                if (v2) { if (off != rep1) rep2 = rep1; rep1 = rep; }
                rep = off;
            }
            /* hot path, as in zap_decompress: short literals + short match with headroom -> fixed-size copies */
            if (ll < 15 && ml < 15 && le - lp >= 16 && oend - op >= 32) {
                memcpy(op, lp, 16);
                op += ll; lp += ll; ml += 4;
                size_t have = (size_t)(op - ostart);
                if (off >= 16 && off <= have) { memcpy(op, op - off, 16); memcpy(op + 16, op - off + 16, 2); }
                else if (off > have + dlen) goto out;
                else zap__dict_copy(op, ostart, oend, d, off, ml);
                op += ml;
                continue;
            }
            if (ll == 15) { ll += zap__ext(&lnp, lne, &err); if (err) goto out; }
            if (ll > (size_t)(le - lp) || ll > (size_t)(oend - op)) goto out;
            if ((size_t)(le - lp) >= ll + 16 && (size_t)(oend - op) >= ll + 16) {
                const uint8_t *s = lp; uint8_t *o = op, *e = op + ll;
                while (o < e) { memcpy(o, s, 16); o += 16; s += 16; }
            } else {
                memcpy(op, lp, ll);
            }
            op += ll; lp += ll;
            if (ml == 15) { ml += zap__ext(&lnp, lne, &err); if (err) goto out; }
            ml += 4;
            if (off > (size_t)(op - ostart) + dlen || ml > (size_t)(oend - op)) goto out;
            zap__dict_copy(op, ostart, oend, d, off, ml);
            op += ml;
        }
        if (lwi != nlow || (nlow & 1 && lwb[nlow >> 1] >> 4)) goto out; /* every nibble used; the pad nibble is 0 */
        if ((size_t)(le - lp) != (size_t)(oend - op)) goto out; /* trailing literals finish the block exactly */
        memcpy(op, lp, (size_t)(le - lp));
        r = (ptrdiff_t)raw_size;
    }
out:
    free(own);
    return r;
}

/* ---------------- frames: [magic][block_size u32][raw_size u64][end offset u64 per block][blocks]
 * "ZAP1": a block whose stored size equals its raw size is stored uncompressed, else it's a plain block.
 * "ZAP2" (written when depth has ZAP_ENTROPY): each block starts with a method byte: 0 raw, 1 plain, 2 entropy. */
#define ZAP_FRAME_MAGIC2 0x3250415Au /* "ZAP2" */

typedef struct { const uint8_t *table, *blocks; size_t raw, bs, nb, blocks_size; int v; } zap_frame;

static inline size_t zap_frame_bound(size_t n, size_t bs) { return 16 + 9 * ((n + bs - 1) / bs) + n; }

static inline void zap__w64(uint8_t *p, uint64_t v) { zap__w32(p, (uint32_t)v); zap__w32(p + 4, (uint32_t)(v >> 32)); }

/* one frame block: compressed size (< len), or 0 = store raw. lim = output room. */
static inline size_t zap__block(const uint8_t *src, size_t len, uint8_t *dst, size_t lim, void *st, int depth, const zap_dict *d) {
    if (len < 2) return 0;
    if (lim > len - 1) lim = len - 1;
    return depth ? zap_compress_hc(src, len, dst, lim, (zap_hc_state *)st, d, depth)
                 : zap_compress(src, len, dst, lim, (zap_state *)st, d);
}

/* version-2 frame block: method byte + payload, smallest of raw / plain / entropy. 0 = no room or out of memory. */
static inline size_t zap__block2(const uint8_t *src, size_t len, uint8_t *dst, size_t lim, void *st, int depth, const zap_dict *d) {
    size_t lcap = zap_bound(len), ln = 0, en = 0, best = len;
    int m = 0;
    uint8_t *lz = (uint8_t *)malloc(lcap), *e = (uint8_t *)malloc(len + 1);
    if (!lz || !e) { free(lz); free(e); return 0; }
    if (len >= 2) {
        ln = depth ? zap_compress_hc(src, len, lz, lcap, (zap_hc_state *)st, d, depth) : zap_compress(src, len, lz, lcap, (zap_state *)st, d);
        if (ln && ln < best) { best = ln; m = 1; }
        if ((depth & 0xFFFF) >= ZAP_OPT_DEPTH && best > 1) en = zap_compress_entropy(src, len, e, best - 1, st, depth, d); /* its own, entropy-priced parse */
        else if (ln && best > 1) en = zap__e_encode(lz, ln, len, e, best - 1);
        if (en && en < best) { best = en; m = 2; }
    }
    size_t r = 0;
    if (lim >= best + 1) { dst[0] = (uint8_t)m; memcpy(dst + 1, m == 0 ? src : m == 1 ? lz : e, best); r = best + 1; }
    free(lz); free(e);
    return r;
}

static inline size_t zap__frame_hdr(uint8_t *dst, size_t cap, size_t n, size_t bs, size_t *nb, int v2) {
    if (!bs || bs > 0x80000000u) return 0;
    size_t hdr = 16 + 8 * (*nb = (n + bs - 1) / bs);
    if (cap < hdr) return 0;
    zap__w32(dst, v2 ? ZAP_FRAME_MAGIC2 : ZAP_FRAME_MAGIC); zap__w32(dst + 4, (uint32_t)bs); zap__w64(dst + 8, n);
    return hdr;
}

/* depth 0 = fast compressor, >0 = hc chain depth; | ZAP_ENTROPY for entropy blocks (version-2 frame).
 * block_size <= 2GB. Returns frame size, 0 on failure. */
static inline size_t zap_frame_compress(const void *src_, size_t n, void *dst_, size_t cap, size_t bs, int depth, const zap_dict *d) {
    const uint8_t *src = (const uint8_t *)src_;
    uint8_t *dst = (uint8_t *)dst_;
    int v2 = (depth & ZAP_ENTROPY) != 0;
    depth = depth & 0xFFFF ? depth & ~ZAP_ENTROPY : 0;
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb, v2), hdr = pos;
    if (!pos) return 0;
    void *st = malloc(depth ? sizeof(zap_hc_state) : sizeof(zap_state));
    if (!st) return 0;
    for (size_t b = 0; b < nb; b++) {
        size_t len = b == nb - 1 ? n - b * bs : bs, room = cap - pos;
        size_t c = v2 ? zap__block2(src + b * bs, len, dst + pos, room, st, depth, d) : zap__block(src + b * bs, len, dst + pos, room, st, depth, d);
        if (!c && v2) { free(st); return 0; }
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
    if (n < 16 || (zap__r32(src) != ZAP_FRAME_MAGIC && zap__r32(src) != ZAP_FRAME_MAGIC2)) return -1;
    f->v = zap__r32(src) == ZAP_FRAME_MAGIC2 ? 2 : 1;
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
        if (end < prev || end - prev > len + (f->v == 2) || (f->v == 2 && end == prev) || end > f->blocks_size) return -1;
        prev = end;
    }
    return 0;
}

/* decode block i into dst (the whole raw_size output buffer). Thread-safe: call from any job. 0 ok, -1 corrupt.
 * scratch: zap_entropy_scratch(block_size) bytes reused across calls for entropy blocks, or NULL to malloc. */
static inline int zap_frame_decode_block_ex(const zap_frame *f, size_t i, void *dst, const zap_dict *d, void *scratch, size_t scratch_cap) {
    size_t start = i ? (size_t)zap__r64le(f->table + 8 * (i - 1)) : 0, end = (size_t)zap__r64le(f->table + 8 * i);
    size_t len = i == f->nb - 1 ? f->raw - i * f->bs : f->bs, sz = end - start;
    const uint8_t *p = f->blocks + start;
    uint8_t *o = (uint8_t *)dst + i * f->bs;
    int m = sz == len ? 0 : 1;
    if (f->v == 2) { m = *p++; sz--; }
    if (m == 0) { if (sz != len) return -1; memcpy(o, p, len); return 0; }
    if (m == 1) return zap_decompress(p, sz, o, len, d) == (ptrdiff_t)len ? 0 : -1;
    if (m == 2) return zap_decompress_entropy(p, sz, o, len, d, scratch, scratch_cap) == (ptrdiff_t)len ? 0 : -1;
    return -1;
}

static inline int zap_frame_decode_block(const zap_frame *f, size_t i, void *dst, const zap_dict *d) {
    return zap_frame_decode_block_ex(f, i, dst, d, NULL, 0);
}

/* returns raw size, or -1 if corrupt / cap too small */
static inline ptrdiff_t zap_frame_decode(const void *src, size_t n, void *dst, size_t cap, const zap_dict *d) {
    zap_frame f;
    if (zap_frame_open(&f, src, n) || f.raw > cap) return -1;
    size_t sc = f.v == 2 ? zap_entropy_scratch(f.bs) : 0;
    void *scratch = sc ? malloc(sc) : NULL;
    ptrdiff_t r = (ptrdiff_t)f.raw;
    for (size_t b = 0; b < f.nb && r >= 0; b++) if (zap_frame_decode_block_ex(&f, b, dst, d, scratch, scratch ? sc : 0)) r = -1;
    free(scratch);
    return r;
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
typedef mtx_t zap__mtx;
typedef cnd_t zap__cnd;
#define ZAP__MINIT(m) (mtx_init(&(m), mtx_plain) == thrd_success)
#define ZAP__CINIT(c) (cnd_init(&(c)) == thrd_success)
#define ZAP__MFREE(m) mtx_destroy(&(m))
#define ZAP__CFREE(c) cnd_destroy(&(c))
#define ZAP__LOCK(m) mtx_lock(&(m))
#define ZAP__UNLOCK(m) mtx_unlock(&(m))
#define ZAP__WAIT(c, m) cnd_wait(&(c), &(m))
#define ZAP__WAKE(c) cnd_signal(&(c))
#define ZAP__WAKEALL(c) cnd_broadcast(&(c))
#else
#include <pthread.h>
static inline void *zap__tramp(void *p) { zap__task *k = (zap__task *)p; k->fn(k->ctx, k->t, k->n); return NULL; }
typedef pthread_t zap__thread;
#define ZAP__START(th, k) (pthread_create(&(th), NULL, zap__tramp, (k)) == 0)
#define ZAP__JOIN(th) pthread_join((th), NULL)
typedef pthread_mutex_t zap__mtx;
typedef pthread_cond_t zap__cnd;
#define ZAP__MINIT(m) (pthread_mutex_init(&(m), NULL) == 0)
#define ZAP__CINIT(c) (pthread_cond_init(&(c), NULL) == 0)
#define ZAP__MFREE(m) pthread_mutex_destroy(&(m))
#define ZAP__CFREE(c) pthread_cond_destroy(&(c))
#define ZAP__LOCK(m) pthread_mutex_lock(&(m))
#define ZAP__UNLOCK(m) pthread_mutex_unlock(&(m))
#define ZAP__WAIT(c, m) pthread_cond_wait(&(c), &(m))
#define ZAP__WAKE(c) pthread_cond_signal(&(c))
#define ZAP__WAKEALL(c) pthread_cond_broadcast(&(c))
#endif
enum { ZAP__MAXT = 64 };

static inline int zap__threads(int want, size_t jobs) {
    if (want > ZAP__MAXT) want = ZAP__MAXT;
    if ((size_t)want > jobs) want = (int)jobs;
    return want < 1 ? 1 : want;
}

/* Persistent workers for work that repeats every few hundred microseconds (video frames), where starting threads
   per call costs more than it saves. zap__pool_run(p, fn, ctx) runs fn(ctx, t, n) for t in [0, n), t = 0 on the
   caller, and returns when all are done. */
typedef struct zap__pool zap__pool;
typedef struct { zap__pool *p; int t; } zap__pw;
struct zap__pool {
    int n, started, quit, pending;
    unsigned gen;
    void (*fn)(void *, int, int);
    void *ctx;
    zap__mtx m;
    zap__cnd go, done;
    zap__thread th[ZAP__MAXT];
    zap__pw w[ZAP__MAXT];
    zap__task k[ZAP__MAXT];
};
static inline void zap__pool_worker(void *arg, int t, int n) {
    zap__pw *w = (zap__pw *)arg;
    zap__pool *p = w->p;
    unsigned seen = 0;
    (void)t; (void)n;
    for (;;) {
        ZAP__LOCK(p->m);
        while (p->gen == seen && !p->quit) ZAP__WAIT(p->go, p->m);
        if (p->quit) { ZAP__UNLOCK(p->m); return; }
        seen = p->gen;
        void (*fn)(void *, int, int) = p->fn;
        void *ctx = p->ctx;
        ZAP__UNLOCK(p->m);
        fn(ctx, w->t, p->n);
        ZAP__LOCK(p->m);
        if (--p->pending == 0) ZAP__WAKE(p->done);
        ZAP__UNLOCK(p->m);
    }
}
static inline void zap__pool_free(zap__pool *p) {
    if (!p) return;
    ZAP__LOCK(p->m); p->quit = 1; ZAP__WAKEALL(p->go); ZAP__UNLOCK(p->m);
    for (int t = 1; t < p->started; t++) ZAP__JOIN(p->th[t]);
    ZAP__MFREE(p->m); ZAP__CFREE(p->go); ZAP__CFREE(p->done);
    free(p);
}
/* NULL if n < 2 or on failure: callers then run the work on one thread */
static inline zap__pool *zap__pool_new(int n) {
    if (n > ZAP__MAXT) n = ZAP__MAXT;
    if (n < 2) return NULL;
    zap__pool *p = (zap__pool *)calloc(1, sizeof *p);
    if (!p) return NULL;
    if (!ZAP__MINIT(p->m)) { free(p); return NULL; }
    if (!ZAP__CINIT(p->go)) { ZAP__MFREE(p->m); free(p); return NULL; }
    if (!ZAP__CINIT(p->done)) { ZAP__CFREE(p->go); ZAP__MFREE(p->m); free(p); return NULL; }
    p->started = 1;
    for (int t = 1; t < n; t++) {
        p->w[t].p = p; p->w[t].t = t;
        p->k[t].fn = zap__pool_worker; p->k[t].ctx = &p->w[t]; p->k[t].t = t; p->k[t].n = n;
        if (!ZAP__START(p->th[t], &p->k[t])) break;
        p->started = t + 1;
    }
    p->n = p->started;
    if (p->n < 2) { zap__pool_free(p); return NULL; }
    return p;
}
static inline void zap__pool_run(zap__pool *p, void (*fn)(void *, int, int), void *ctx) {
    ZAP__LOCK(p->m);
    p->fn = fn; p->ctx = ctx; p->pending = p->n - 1; p->gen++;
    ZAP__WAKEALL(p->go);
    ZAP__UNLOCK(p->m);
    fn(ctx, 0, p->n);
    ZAP__LOCK(p->m);
    while (p->pending) ZAP__WAIT(p->done, p->m);
    ZAP__UNLOCK(p->m);
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
    size_t sc = j->f->v == 2 ? zap_entropy_scratch(j->f->bs) : 0;
    void *scratch = sc ? malloc(sc) : NULL; /* one per thread, reused across blocks */
    for (size_t b = (size_t)t; b < j->f->nb; b += (size_t)n)
        if (zap_frame_decode_block_ex(j->f, b, j->dst, j->d, scratch, scratch ? sc : 0)) j->err[t] = 1;
    free(scratch);
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
typedef struct { const uint8_t *src; uint8_t *tmp; size_t n, bs, nb, *cs, slot; int depth, v2; const zap_dict *d; } zap__cjob;
static inline void zap__cwork(void *p, int t, int n) {
    zap__cjob *j = (zap__cjob *)p;
    void *st = malloc(j->depth ? sizeof(zap_hc_state) : sizeof(zap_state));
    if (!st) return; /* its blocks keep cs == SIZE_MAX -> caller fails */
    for (size_t b = (size_t)t; b < j->nb; b += (size_t)n) {
        size_t len = b == j->nb - 1 ? j->n - b * j->bs : j->bs;
        if (j->v2) { size_t c = zap__block2(j->src + b * j->bs, len, j->tmp + b * j->slot, len + 1, st, j->depth, j->d); if (c) j->cs[b] = c; }
        else j->cs[b] = zap__block(j->src + b * j->bs, len, j->tmp + b * j->slot, len, st, j->depth, j->d);
    }
    free(st);
}

/* same output as zap_frame_compress. Memory: threads x state (32.5MB each for hc) + n bytes scratch. */
static inline size_t zap_frame_compress_mt(const void *src_, size_t n, void *dst_, size_t cap, size_t bs, int depth,
                                           const zap_dict *d, int threads) {
    const uint8_t *src = (const uint8_t *)src_;
    uint8_t *dst = (uint8_t *)dst_;
    int v2 = (depth & ZAP_ENTROPY) != 0;
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb, v2), hdr = pos;
    if (!pos) return 0;
    size_t slot = v2 ? (nb == 1 ? n : bs) + 1 : bs;
    zap__cjob j = { src, (uint8_t *)malloc(v2 ? nb * slot + 1 : n + 1), n, bs, nb, (size_t *)malloc((nb ? nb : 1) * sizeof(size_t)), slot, depth & 0xFFFF ? depth & ~ZAP_ENTROPY : 0, v2, d };
    if (!j.tmp || !j.cs) { free(j.tmp); free(j.cs); return 0; }
    for (size_t b = 0; b < nb; b++) j.cs[b] = SIZE_MAX;
    zap__par(zap__threads(threads, nb), zap__cwork, &j);
    for (size_t b = 0; b < nb; b++) {
        size_t len = b == nb - 1 ? n - b * bs : bs, c = j.cs[b];
        const uint8_t *from = c && c != SIZE_MAX ? j.tmp + b * slot : src + b * bs;
        if (!c) c = len; /* store raw (version 1 only: version 2 blocks always carry a method byte) */
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
