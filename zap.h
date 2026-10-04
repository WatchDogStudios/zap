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
 *   depth >= 32 without a dict writes entropy v3, the Kraken tier (multi-arrival parse; per thread ~4 MB/s at
 *   depth 32, ~3 at 48, ~1 at 64, ~0.5 at 128; ~85 MB per 4 MB block); | ZAP_FAST_DECODE keeps literal chunks raw unless
 *   Huffman saves > 4% (~10% faster decode, ~0.6% larger). Blocks written by 1.x versions (v1, v2) still decode.
 *
 * Frames (packaging: self-describing, independent blocks, parallel decode):
 *   zap_frame_compress(src, n, dst, cap, block_size, depth /0 = fast/ [| ZAP_ENTROPY or ZAP_TURBO], dict)
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

#define ZAP_VERSION "1.3.0"

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
    /* turbo blocks: tri[kind][lit][ml] = the block's command table has it (kind 0 repeat, else offset bytes);
       pesc / pfar: decode-time penalties for escapes and for offsets >= farlim (cache misses) */
    int turbo;
    uint8_t tri[4][17][33];
    uint32_t pesc, pfar, farlim;
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
static inline int zap__bt(zap_hc_state *s, uint32_t *heads, int hlog, const uint8_t *src, size_t pos, const uint8_t *iend, int depth,
                          int insert, size_t best, zap__match *m, int n) {
    const size_t wmask = sizeof s->prev / sizeof s->prev[0] / 2 - 1;
    uint32_t *son = s->prev;
    const uint8_t *ip = src + pos;
    size_t avail = (size_t)(iend - ip), limit = avail < ZAP__BT_NICE ? avail : ZAP__BT_NICE, len0 = 0, len1 = 0;
    uint32_t *head = &heads[zap__hash(zap__r32(ip), hlog)], dummy[2];
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
    for (size_t p = *next; p < pos; p++) zap__bt(s, s->head, ZAP_HC_HLOG, src, p, iend, depth < 8 ? depth : 8, 1, 0, NULL, 0); /* skipped: shallow insert */
    int n = 0;
    if (rep && rep <= pos && zap__r32(ip - rep) == zap__r32(ip)) {
        best = 4 + zap__count(ip + 4, ip - rep + 4, iend);
        m[n].len = (uint32_t)best; m[n].off = (uint32_t)rep; n++;
    }
    n = zap__bt(s, s->head, ZAP_HC_HLOG, src, pos, iend, depth, pos >= *next, best, m, n);
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

#define ZAP__T_SPLIT_LEN 96 /* turbo: matches of 33..96 bytes (offset >= 16) go out as up to 3 table commands */
#ifndef ZAP__T_MAXOFF
/* turbo: farthest offset the parse may use (ZAP_MAX_DIST = the whole window). Decode speed is bound by match-source loads;
   sources within ~384KB mostly stay in L2 (512KB on Zen 2/3, the consoles included), farther ones come from L3.
   On a game pak sample the cap costs 0.9% ratio and gains ~25% decode speed. */
#define ZAP__T_MAXOFF (384u << 10)
#endif
static inline uint32_t zap__match_price(const zap__cost *c, size_t len, size_t off, size_t lit, const uint32_t rep[3]) {
    if (c->turbo) { /* token byte + offset bytes; an escape adds its two length bytes (4 each when >= 255) */
        if (off > ZAP__T_MAXOFF && off != rep[0]) return 0x3FFFFFFFu; /* out of reach: never chosen */
        int kind = off == rep[0] ? 0 : off < 256 ? 1 : off < 65536 ? 2 : 3;
        uint32_t p = c->seq + 128u * (1u + (unsigned)kind) + (kind && off >= c->farlim ? c->pfar : 0u);
        size_t first = len;
        if (len > 32 && len <= ZAP__T_SPLIT_LEN && off >= 16) { /* split into repeat commands: a token (and command) each */
            size_t more = (len - 1) / 32;
            p += (c->seq + 128u) * (uint32_t)more;
            first = len - 32 * more >= 4 ? 32 : len - 4;
        }
        if (lit > 16 || first > 32 || off < 16 || !c->tri[kind][lit][first])
            p += (first <= 32 ? c->pesc : 0u) + 256u + (lit >= 255 ? 384u : 0u) + (first >= 259 ? 384u : 0u);
        return p;
    }
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

static inline void zap__hlens_l(const uint32_t *freq, uint8_t len[256], int maxlen) {
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
        if (maxd <= maxlen) { for (int i = 0; i < n; i++) len[sym[i]] = (uint8_t)d[i]; return; }
        for (int i = 0; i < 256; i++) if (f[i]) f[i] = (f[i] >> 1) | 1; /* flatten and retry */
    }
}
static inline void zap__hlens(const uint32_t *freq, uint8_t len[256]) { zap__hlens_l(freq, len, ZAP__HMAX); }

/* LSB-first codes: canonical code, bit-reversed */
static inline void zap__hcodes(const uint8_t len[256], uint16_t code[256]) {
    int count[16] = { 0 }, next[16] = { 0 };
    for (int i = 0; i < 256; i++) count[len[i]]++;
    count[0] = 0;
    for (int l = 1, c = 0; l < 16; l++) { c = (c + count[l - 1]) << 1; next[l] = c; }
    for (int i = 0; i < 256; i++) {
        if (!len[i]) continue;
        unsigned c = (unsigned)next[len[i]]++; /* reverse 16 bits, keep the top len */
        c = (c >> 1 & 0x5555u) | (c & 0x5555u) << 1; c = (c >> 2 & 0x3333u) | (c & 0x3333u) << 2;
        c = (c >> 4 & 0x0F0Fu) | (c & 0x0F0Fu) << 4; c = (c >> 8 & 0x00FFu) | (c & 0x00FFu) << 8;
        code[i] = (uint16_t)(c >> (16 - len[i]));
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
    if (raw >= 256 && (n - 140) * 16 <= raw * 88) { /* short codes (<= 5.5 bits on average): two symbols per lookup */
        uint32_t t2[1 << ZAP__HMAX]; /* sym1 | sym2 << 8 | count << 16 | total length << 20 */
        for (unsigned bi = 0; bi <= mask; bi++) {
            unsigned e1 = table[bi], l1 = e1 >> 8, e2 = l1 ? table[(bi >> l1) & mask] : 0, l2 = e2 >> 8;
            /* codes are LSB-first, so the second code only needs its own l2 low bits of what follows the first */
            t2[bi] = l1 && l2 && l1 + l2 <= ZAP__HMAX ? (e1 & 255) | (e2 & 255) << 8 | 2u << 16 | (l1 + l2) << 20 : (e1 & 255) | 1u << 16 | l1 << 20;
        }
        size_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        zap__hbits r0 = b0, r1 = b1, r2 = b2, r3 = b3;
        /* 5 lookups (<= 55 bits) per refill, up to 10 symbols per quarter: runs while every quarter has 10 left */
        while (c3 - i3 >= 10 && q - i0 >= 10 && q - i1 >= 10 && q - i2 >= 10 && r0.e - r0.p >= 8 && r1.e - r1.p >= 8 && r2.e - r2.p >= 8 && r3.e - r3.p >= 8) {
            ZAP__HREFILL(r0); ZAP__HREFILL(r1); ZAP__HREFILL(r2); ZAP__HREFILL(r3);
#define ZAP__HSYM2(b, o, i) do { uint32_t e_ = t2[(b).acc & mask], l_ = e_ >> 20; (o)[i] = (uint8_t)e_; (o)[(i) + 1] = (uint8_t)(e_ >> 8); \
                                 (i) += e_ >> 16 & 3; (b).acc >>= l_; (b).nb -= (int)l_; } while (0)
            for (int k = 0; k < 5; k++) { ZAP__HSYM2(r0, o0, i0); ZAP__HSYM2(r1, o1, i1); ZAP__HSYM2(r2, o2, i2); ZAP__HSYM2(r3, o3, i3); }
#undef ZAP__HSYM2
        }
        b0 = r0; b1 = r1; b2 = r2; b3 = r3;
        if (zap__hrun(table, &b0, o0 + i0, q - i0) || zap__hrun(table, &b1, o1 + i1, q - i1) ||
            zap__hrun(table, &b2, o2 + i2, q - i2) || zap__hrun(table, &b3, o3 + i3, c3 - i3)) return -1;
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

/* Contextual 4-way Huffman (stream method 2): symbol i is coded with the table of group grp[previous symbol in the
   same quarter] (group 0 at each quarter's start), so the four quarters still decode independently. On game data
   this is -1.1% for literals grouped by the previous literal's top 4 bits and -1.2% for tokens grouped by the
   previous token's literal and match lengths, net of the extra tables.
   Layout: u8 groups (1..ZAP__HCTX), groups x 128 bytes of 4-bit code lengths, u32 sizes of quarters 0-2, 4 bitstreams. */
enum { ZAP__HCTX = 16 };
static inline size_t zap__henc4c(const uint8_t *src, size_t n, const uint8_t grp[256], int ng, uint8_t *out, size_t cap) {
    size_t hdr = 1 + 128 * (size_t)ng + 12, q = (n + 3) / 4;
    if (cap < hdr || ng < 1 || ng > ZAP__HCTX) return 0;
    uint32_t (*freq)[256] = (uint32_t(*)[256])calloc((size_t)ng, sizeof *freq);
    uint8_t (*len)[256] = (uint8_t(*)[256])malloc((size_t)ng * sizeof *len);
    uint16_t (*code)[256] = (uint16_t(*)[256])malloc((size_t)ng * sizeof *code);
    size_t r = 0;
    if (!freq || !len || !code) goto done;
    for (size_t i = 0; i < n; i++) freq[i % q ? grp[src[i - 1]] : 0][src[i]]++;
    out[0] = (uint8_t)ng;
    for (int g = 0; g < ng; g++) {
        zap__hlens(freq[g], len[g]);
        zap__hcodes(len[g], code[g]);
        for (int i = 0; i < 128; i++) out[1 + 128 * g + i] = (uint8_t)(len[g][2 * i] | len[g][2 * i + 1] << 4);
    }
    {
        uint8_t *o = out + hdr, *oe = out + cap;
        for (int k = 0; k < 4; k++) {
            size_t lo = (size_t)k * q < n ? (size_t)k * q : n, hi = lo + q < n ? lo + q : n;
            uint8_t *start = o;
            uint64_t acc = 0;
            int nb = 0, g = 0;
            for (size_t i = lo; i < hi; i++) {
                acc |= (uint64_t)code[g][src[i]] << nb; nb += len[g][src[i]];
                g = grp[src[i]];
                while (nb >= 8) { if (o >= oe) goto done; *o++ = (uint8_t)acc; acc >>= 8; nb -= 8; }
            }
            if (nb) { if (o >= oe) goto done; *o++ = (uint8_t)acc; }
            if (k < 3) zap__w32(out + 1 + 128 * (size_t)ng + 4 * (size_t)k, (uint32_t)(o - start));
        }
        r = (size_t)(o - out);
    }
done:
    free(freq); free(len); free(code);
    return r;
}

/* tables: ZAP__HCTX << ZAP__HMAX entries of scratch */
static inline int zap__hdec4c(const uint8_t *in, size_t n, uint8_t *out, size_t raw, const uint8_t grp[256], uint16_t *tables) {
    const unsigned mask = (1u << ZAP__HMAX) - 1;
    if (n < 1) return -1;
    int ng = in[0];
    size_t hdr = 1 + 128 * (size_t)ng + 12;
    if (ng < 1 || ng > ZAP__HCTX || n < hdr) return -1;
    for (int v = 0; v < 256; v++) if (grp[v] >= ng) return -1; /* a group without a table */
    for (int g = 0; g < ng; g++) {
        uint16_t *t = tables + ((size_t)g << ZAP__HMAX);
        if (zap__htable(in + 1 + 128 * g, t)) return -1;
        /* entries are symbol | length << 8 (length <= 11); the spare top 4 bits get the symbol's group, so the next
           table is known from this entry alone: one load per symbol in the dependency chain instead of two */
        for (int j = 0; j < 1 << ZAP__HMAX; j++) t[j] = (uint16_t)(t[j] | grp[t[j] & 255] << 12);
    }
    const uint8_t *sz = in + 1 + 128 * (size_t)ng;
    size_t s0 = zap__r32(sz), s1 = zap__r32(sz + 4), s2 = zap__r32(sz + 8), avail = n - hdr;
    if (s0 > avail || s1 > avail - s0 || s2 > avail - s0 - s1) return -1;
    const uint8_t *p[4] = { in + hdr, in + hdr + s0, in + hdr + s0 + s1, in + hdr + s0 + s1 + s2 }, *e[4] = { p[1], p[2], p[3], in + n };
    size_t q = (raw + 3) / 4, lo[4], hi[4], i = 0;
    for (int k = 0; k < 4; k++) { lo[k] = (size_t)k * q < raw ? (size_t)k * q : raw; hi[k] = lo[k] + q < raw ? lo[k] + q : raw; }
    zap__hbits b0 = { p[0], e[0], 0, 0 }, b1 = { p[1], e[1], 0, 0 }, b2 = { p[2], e[2], 0, 0 }, b3 = { p[3], e[3], 0, 0 };
    unsigned g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    /* fast part: the four quarters interleaved (independent dependency chains), while the shortest (the last)
       has 5 symbols left and every reader 8 bytes */
#define ZAP__HCSYM(b, g, dst) do { unsigned t_ = tables[((g) << ZAP__HMAX) | (unsigned)((b).acc & mask)], l_ = t_ >> 8 & 15; \
    (dst) = (uint8_t)t_; (b).acc >>= l_; (b).nb -= (int)l_; (g) = t_ >> 12; } while (0)
#define ZAP__HCREFILL(b) do { (b).acc |= zap__r64((b).p) << (b).nb; (b).p += (63 - (b).nb) >> 3; (b).nb |= 56; } while (0)
    while (hi[3] - lo[3] - i >= 5 && b0.e - b0.p >= 8 && b1.e - b1.p >= 8 && b2.e - b2.p >= 8 && b3.e - b3.p >= 8) {
        ZAP__HCREFILL(b0); ZAP__HCREFILL(b1); ZAP__HCREFILL(b2); ZAP__HCREFILL(b3);
        for (int j = 0; j < 5; j++) {
            ZAP__HCSYM(b0, g0, out[lo[0] + i + j]); ZAP__HCSYM(b1, g1, out[lo[1] + i + j]);
            ZAP__HCSYM(b2, g2, out[lo[2] + i + j]); ZAP__HCSYM(b3, g3, out[lo[3] + i + j]);
        }
        i += 5;
    }
#undef ZAP__HCSYM
#undef ZAP__HCREFILL
    zap__hbits *bs[4] = { &b0, &b1, &b2, &b3 };
    unsigned gs[4] = { g0, g1, g2, g3 };
    for (int k = 0; k < 4; k++) { /* careful tails, bounds-checked */
        zap__hbits *b = bs[k];
        unsigned g = gs[k];
        for (size_t x = lo[k] + i; x < hi[k]; x++) {
            while (b->nb <= 56 && b->p < b->e) { b->acc |= (uint64_t)*b->p++ << b->nb; b->nb += 8; }
            int t = tables[(g << ZAP__HMAX) | (unsigned)(b->acc & mask)], l = t >> 8 & 15;
            if (!l || l > b->nb) return -1;
            out[x] = (uint8_t)t; b->acc >>= l; b->nb -= l; g = (unsigned)t >> 12;
        }
    }
    return 0;
}

/* the fixed groupings: literals by the previous literal's top 4 bits, tokens by the previous token's literal length
   (0, 1, 2-3, 4+) x match nibble (0-1, 2-4, 5-14, 15) */
static inline void zap__grp_lit(uint8_t g[256]) { for (int v = 0; v < 256; v++) g[v] = (uint8_t)(v >> 4); }
static inline void zap__grp_tok(uint8_t g[256]) {
    for (int v = 0; v < 256; v++) {
        int l = v >> 4, m = v & 15;
        g[v] = (uint8_t)((l == 0 ? 0 : l <= 1 ? 1 : l <= 3 ? 2 : 3) * 4 + (m <= 1 ? 0 : m <= 4 ? 1 : m < 15 ? 2 : 3));
    }
}

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
static inline size_t zap_entropy_scratch(size_t raw) { return 5 * raw + raw / 255 + (2u << 17) + 8192; } /* streams + tables (v1-v3) */

static inline uint8_t *zap__e_stream(uint8_t *op, uint8_t *oend, const uint8_t *s, size_t n, const uint8_t *grp) {
    if (!op || oend - op < 5) return NULL;
    size_t room = (size_t)(oend - op) - 5, h = 0, hc = 0;
    if (n > 64) h = zap__henc4(s, n, op + 5, room < n - 1 ? room : n - 1);
    if (grp && n > 4096) { /* contextual Huffman: kept only if smaller (it overwrites the plain encoding) */
        size_t lim = (h ? h : n) - 1, cap2 = room < lim ? room : lim;
        uint8_t *tmp = (uint8_t *)malloc(cap2 ? cap2 : 1);
        if (tmp && (hc = zap__henc4c(s, n, grp, ZAP__HCTX, tmp, cap2)) != 0) { *op = 2; zap__w32(op + 1, (uint32_t)hc); memcpy(op + 5, tmp, hc); }
        free(tmp);
        if (hc) return op + 5 + hc;
    }
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
    uint8_t glit[256], gtok[256];
    zap__grp_lit(glit); zap__grp_tok(gtok);
    op = zap__e_stream(op + 20, oend, lits, nl, glit);
    op = zap__e_stream(op, oend, toks, ns, gtok);
    op = zap__e_stream(op, oend, lens, nlen, NULL);
    op = zap__e_stream(op, oend, offc, ns, NULL);
    op = zap__e_stream(op, oend, low, (nlow + 1) / 2, NULL);
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

static inline size_t zap__k_compress(const uint8_t *src, size_t n, uint8_t *dst, size_t cap, zap_hc_state *hc, int depth, int rawpct);
static inline ptrdiff_t zap__k_decompress(const uint8_t *ip, size_t n, uint8_t *dst, size_t raw_size, void *scratch, size_t scratch_cap);

/* depth 0: fast parse (state = zap_state*), else hc chain depth (state = zap_hc_state*). 0 = doesn't fit in cap.
   depth >= ZAP_OPT_DEPTH without a dictionary: entropy v3 (the Kraken tier: multi-arrival parses, see below);
   otherwise v2: two optimal parses, the second priced with the first one's Huffman code lengths. */
static inline size_t zap_compress_entropy(const void *src, size_t n, void *dst, size_t cap, void *state, int depth, const zap_dict *d) {
    if (!d && (depth & 0xFFFF) >= ZAP_OPT_DEPTH && !zap__hc_skip((const uint8_t *)src, n, (zap_hc_state *)state))
        return zap__k_compress((const uint8_t *)src, n, (uint8_t *)dst, cap, (zap_hc_state *)state, depth & 0xFFFF, depth & ZAP_FAST_DECODE ? 4 : 0);
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
    if ((zap__r32(ip) >> 30) == 3) return d ? -1 : zap__k_decompress(ip, n, (uint8_t *)dst_, raw_size, scratch, scratch_cap); /* v3 */
    int v2 = (zap__r32(ip) >> 31) != 0;
    size_t nl = zap__r32(ip) & 0x7FFFFFFFu, ns = zap__r32(ip + 4), nlen = zap__r32(ip + 8), nx = zap__r32(ip + 12), nlow = 0;
    ip += 16;
    if (v2) { if (n < 20) return -1; nlow = zap__r32(ip); ip += 4; }
    if (nl > raw_size || ns > raw_size / 4 + 1 || nlen > 2 * ns + raw_size / 255 + 1 || nx > 3 * ns + 8 || nlow > ns) return -1;
    int nstreams = v2 ? 5 : 4;
    size_t tabbytes = sizeof(uint16_t) * ((size_t)ZAP__HCTX << ZAP__HMAX);
    size_t need = nl + 2 * ns + nlen + (nlow + 1) / 2 + tabbytes + 16, cnt[5] = { nl, ns, nlen, ns, (nlow + 1) / 2 };
    uint8_t grps[2][256];
    zap__grp_lit(grps[0]); zap__grp_tok(grps[1]);
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
        else if (m == 2 && v2 && k < 2) { /* contextual Huffman: literals and tokens only; tables at the end of scratch */
            uint8_t *tb = mem + nl + 2 * ns + nlen + (nlow + 1) / 2; /* past everything the streams decode into */
            uint16_t *tab = (uint16_t *)(void *)(tb + ((16 - ((uintptr_t)tb & 15)) & 15));
            if (zap__hdec4c(ip, sz, w, cnt[k], grps[k], tab)) goto out;
            st[k] = w; w += cnt[k];
        }
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
        if (v2 && !d) { /* version 2 without a dictionary (the packaging case): a specialised loop */
            size_t xbit = 0; /* extra-bit position */
            const uint8_t *nw = lwb, *nwe = lwb + (nlow + 1) / 2;
            uint64_t nbuf = 0;
            int nn = 0; /* nibbles left in nbuf */
            for (size_t i = 0; i < ns; i++) {
                unsigned tok = tp[i], c = oc[i];
                size_t ll = tok >> 4, ml = tok & 15, off;
                if (c >= 3) {
                    if (c > 25) goto out;
                    unsigned k = c - 3, xk = k >= 4 ? k - 4 : k;
                    /* position-based bit reader: only the bit position carries over, so the loads don't chain */
                    size_t x, by = xbit >> 3;
                    if (by + 8 <= nx) x = (size_t)((zap__r64(xp + by) >> (xbit & 7)) & (((uint64_t)1 << xk) - 1));
                    else { /* the last 8 bytes: bytewise, bounds-checked */
                        uint64_t v = 0;
                        for (size_t q = 0; q < 8 && by + q < nx; q++) v |= (uint64_t)xp[by + q] << 8 * q;
                        if (xbit + xk > 8 * nx) goto out;
                        x = (size_t)((v >> (xbit & 7)) & (((uint64_t)1 << xk) - 1));
                    }
                    xbit += xk;
                    if (k >= 4) {
                        if (!nn) { /* 16 nibbles at a time; bytewise at the end */
                            if (nwe - nw >= 8) { nbuf = zap__r64(nw); nw += 8; nn = 16; }
                            else { nbuf = 0; for (int b = 0; nw < nwe; b++) { nbuf |= (uint64_t)*nw++ << 8 * b; nn += 2; } if (!nn) goto out; }
                        }
                        x = x << 4 | (size_t)(nbuf & 15); nbuf >>= 4; nn--;
                    }
                    off = ((size_t)1 << k) + x;
                    if (off != rep1) rep2 = rep1;
                    rep1 = rep; rep = off;
                } else if (c == 0) { off = rep; if (!off) goto out; }
                else if (c == 1) { off = rep1; rep1 = rep; rep = off; if (!off) goto out; }
                else { off = rep2; rep2 = rep1; rep1 = rep; rep = off; if (!off) goto out; }
                if (ll < 15 && ml < 15 && le - lp >= 16 && oend - op >= 32 && off >= 16 && off <= (size_t)(op - ostart) + ll) {
                    memcpy(op, lp, 16);
                    op += ll; lp += ll;
                    memcpy(op, op - off, 16); memcpy(op + 16, op - off + 16, 2);
                    op += ml + 4;
                    continue;
                }
                if (ll == 15) { ll += zap__ext(&lnp, lne, &err); if (err) goto out; }
                if (ll > (size_t)(le - lp) || ll > (size_t)(oend - op)) goto out;
                memcpy(op, lp, ll);
                op += ll; lp += ll;
                if (ml == 15) { ml += zap__ext(&lnp, lne, &err); if (err) goto out; }
                ml += 4;
                if (off > (size_t)(op - ostart) || ml > (size_t)(oend - op)) goto out;
                zap__match_copy(op, off, ml, oend);
                op += ml;
            }
            /* every nibble used (the pad nibble of an odd count is 0) */
            if ((size_t)(nw - lwb) * 2 - (size_t)nn != nlow) goto out; /* fetched minus still buffered = used */
            if (nlow & 1 && lwb[nlow >> 1] >> 4) goto out;
            if ((size_t)(le - lp) != (size_t)(oend - op)) goto out;
            memcpy(op, lp, (size_t)(le - lp));
            r = (ptrdiff_t)raw_size;
            goto out;
        }
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

/* ---------------- turbo blocks: the fastest decode (version-2 frame method 3; no dictionary)
 * [u32 ntok][u32 noff][u32 nlen][u8 ntab][ntab x u16: lit | (ml - 4) << 5 | kind << 10]
 * [tokens: one byte per command][offsets][lengths][literals: the rest of the block]
 * A token below ntab is that table entry: lit (0-16) literals, then ml (4-32) bytes from an offset given by kind:
 * 0 = the previous command's offset, else kind bytes (LE) from the offset stream (1: < 256, 2: < 64K, 3). Tokens 252 + kind are
 * escapes: lit and ml - 4 come from the lengths stream, one byte each (255 + 3 bytes LE for values >= 255).
 * Each block picks its own table: its 252 most frequent (kind, lit, ml) commands. Offsets < 16 are always escapes,
 * so a table command is always one 16-byte literal copy and two 16-byte match copies. The parse is priced in bytes
 * plus decode-time penalties for commands, escapes and far offsets. */
#define ZAP_TURBO (1 << 18) /* OR into a frame's depth: turbo blocks (version-2 frame); depth >= ZAP_OPT_DEPTH to price the parse */
#ifndef ZAP__T_SEQ
#define ZAP__T_SEQ 64 /* per command (1/16 bit) */
#endif
#ifndef ZAP__T_ESC
#define ZAP__T_ESC 128 /* per escape */
#endif
#ifndef ZAP__T_FAR
#define ZAP__T_FAR 0 /* per offset >= ZAP__T_FARLIM */
#endif
#ifndef ZAP__T_FARLIM
#define ZAP__T_FARLIM (256u << 10)
#endif
#define ZAP__T_ESC0 252u
#define ZAP__T_NTRI (4 * 17 * 33)

/* a command's offset kind: 0 = repeat the previous offset, else how many offset bytes it takes */
static inline size_t zap__t_kind(size_t off, size_t last) { return off == last ? 0 : off < 256 ? 1 : off < 65536 ? 2 : 3; }

typedef struct { const uint8_t *p, *e; } zap__lzcur;
/* next sequence of a plain block made by this compressor: its literals in lp, ll; 0 = the trailing literals */
static inline int zap__lznext(zap__lzcur *c, const uint8_t **lp, size_t *ll, size_t *off, size_t *ml) {
    unsigned tok = *c->p++;
    size_t l = tok >> 4, m = tok & 15;
    if (l == 15) { uint8_t b; do { b = *c->p++; l += b; } while (b == 255); }
    *lp = c->p; *ll = l; c->p += l;
    if (c->p >= c->e) return 0;
    size_t v = (size_t)c->p[0] | (size_t)c->p[1] << 8;
    c->p += 2;
    *off = v & 0x7FFF;
    if (v & 0x8000) *off |= (size_t)*c->p++ << 15;
    if (m == 15) { uint8_t b; do { b = *c->p++; m += b; } while (b == 255); }
    *ml = m + 4;
    return 1;
}

/* the commands one sequence becomes: a match of 33..ZAP__T_SPLIT_LEN bytes (offset >= 16) splits into up to 3 table
   commands, the later ones with no literals and the same offset (so a repeat): a token byte each instead of an
   escape's two length bytes, and no escape to decode */
static inline int zap__t_split(size_t ll, size_t ml, size_t off, size_t cl[3], size_t cm[3]) {
    int n = 0;
    if (ml > 32 && ml <= ZAP__T_SPLIT_LEN && off >= 16)
        while (ml > 32) { size_t c = ml - 32 >= 4 ? 32 : ml - 4; cl[n] = n ? 0 : ll; cm[n++] = c; ml -= c; }
    cl[n] = n ? 0 : ll; cm[n++] = ml;
    return n;
}

/* the triple index of a command, or -1 when it can only be an escape */
static inline int zap__t_tri(size_t kind, size_t ll, size_t off, size_t ml) {
    return ll <= 16 && ml <= 32 && off >= 16 ? (int)((kind * 17 + ll) * 33 + ml) : -1;
}

/* the block's command table: its 252 most frequent triples, as tri[] flags and (if code) the code of each triple */
static inline int zap__t_table(const uint8_t *lz, size_t lzn, uint8_t tri[ZAP__T_NTRI], int16_t *code) {
    uint64_t *h = (uint64_t *)calloc(ZAP__T_NTRI, sizeof *h);
    if (!h) return -1;
    zap__lzcur c = { lz, lz + lzn };
    const uint8_t *lp;
    size_t ll, off, ml, last = 0;
    while (zap__lznext(&c, &lp, &ll, &off, &ml)) {
        size_t cl[3], cm[3];
        for (int j = 0, nc = zap__t_split(ll, ml, off, cl, cm); j < nc; j++) {
            int i = zap__t_tri(zap__t_kind(off, last), cl[j], off, cm[j]);
            last = off;
            if (i >= 0) h[i] += (uint64_t)1 << 16;
        }
    }
    for (int i = 0; i < ZAP__T_NTRI; i++) h[i] |= (uint64_t)i;
    for (int i = 0; i < (int)ZAP__T_ESC0; i++) { /* partial selection sort: the 252 largest counts, ties by index */
        int b = i;
        for (int j = i + 1; j < ZAP__T_NTRI; j++) if (h[j] > h[b]) b = j;
        uint64_t x = h[i]; h[i] = h[b]; h[b] = x;
    }
    memset(tri, 0, ZAP__T_NTRI);
    if (code) for (int i = 0; i < ZAP__T_NTRI; i++) code[i] = -1;
    int n = 0;
    for (; n < (int)ZAP__T_ESC0 && h[n] >> 16; n++) {
        tri[h[n] & 0xFFFF] = 1;
        if (code) code[h[n] & 0xFFFF] = (int16_t)n;
    }
    free(h);
    return n;
}

static inline uint8_t *zap__t_ext(uint8_t *p, size_t v) {
    if (v < 255) { *p++ = (uint8_t)v; return p; }
    v -= 255; p[0] = 255; p[1] = (uint8_t)v; p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)(v >> 16);
    return p + 4;
}

/* a plain block (this compressor's output, no dictionary) re-coded as a turbo block. 0 = doesn't fit in cap. */
static inline size_t zap__t_encode(const uint8_t *lz, size_t lzn, uint8_t *dst, size_t cap) {
    uint8_t *tri = (uint8_t *)malloc(ZAP__T_NTRI);
    int16_t *code = (int16_t *)malloc(ZAP__T_NTRI * sizeof *code);
    size_t r = 0;
    int ntab = tri && code ? zap__t_table(lz, lzn, tri, code) : -1;
    if (ntab < 0) goto done;
    { /* stream sizes first, so each stream can be written in place */
        zap__lzcur c = { lz, lz + lzn };
        const uint8_t *lp;
        size_t ll, off, ml, last = 0, ntok = 0, noff = 0, nlen = 0, nlit = 0;
        int more;
        do {
            more = zap__lznext(&c, &lp, &ll, &off, &ml);
            nlit += ll;
            if (!more) break;
            size_t cl[3], cm[3];
            for (int j = 0, nc = zap__t_split(ll, ml, off, cl, cm); j < nc; j++) {
                size_t kind = zap__t_kind(off, last);
                last = off;
                int i = zap__t_tri(kind, cl[j], off, cm[j]);
                ntok++; noff += kind;
                if (i < 0 || code[i] < 0) nlen += (cl[j] < 255 ? 1 : 4) + (cm[j] - 4 < 255 ? 1 : 4);
            }
        } while (more);
        size_t hdr = 13 + 2 * (size_t)ntab, total = hdr + ntok + noff + nlen + nlit;
        if (total > cap || ntok > 0xFFFFFFFFu || noff > 0xFFFFFFFFu || nlen > 0xFFFFFFFFu) goto done;
        zap__w32(dst, (uint32_t)ntok); zap__w32(dst + 4, (uint32_t)noff); zap__w32(dst + 8, (uint32_t)nlen); dst[12] = (uint8_t)ntab;
        for (int i = 0; i < ZAP__T_NTRI; i++) if (code[i] >= 0) {
            unsigned kind = (unsigned)i / (17 * 33), li = (unsigned)i / 33 % 17, mi = (unsigned)i % 33, w = li | (mi - 4) << 5 | kind << 10;
            dst[13 + 2 * code[i]] = (uint8_t)w; dst[14 + 2 * code[i]] = (uint8_t)(w >> 8);
        }
        uint8_t *tp = dst + hdr, *op = tp + ntok, *lnp = op + noff, *litp = lnp + nlen;
        c.p = lz; last = 0;
        do {
            more = zap__lznext(&c, &lp, &ll, &off, &ml);
            memcpy(litp, lp, ll); litp += ll;
            if (!more) break;
            size_t cl[3], cm[3];
            for (int j = 0, nc = zap__t_split(ll, ml, off, cl, cm); j < nc; j++) {
                size_t kind = zap__t_kind(off, last);
                last = off;
                int i = zap__t_tri(kind, cl[j], off, cm[j]), k = i >= 0 ? code[i] : -1;
                if (k < 0) { k = (int)(ZAP__T_ESC0 + kind); lnp = zap__t_ext(zap__t_ext(lnp, cl[j]), cm[j] - 4); }
                *tp++ = (uint8_t)k;
                for (size_t b = 0; b < kind; b++) *op++ = (uint8_t)(off >> 8 * b);
            }
        } while (more);
        r = total;
    }
done:
    free(tri); free(code);
    return r;
}

/* the block's parse, priced for turbo blocks: a first pass that assumes every short command is in the table, then
   one priced with the table that pass would get. Output: a plain block for zap__t_encode. */
static inline size_t zap__t_parse(const uint8_t *src, size_t n, uint8_t *lz, size_t cap, zap_hc_state *s, int depth) {
    zap__cost *cm = (zap__cost *)malloc(sizeof *cm);
    if (!cm) return 0;
    zap__cost_plain(cm, ZAP__T_SEQ);
    cm->lenb = 0; cm->turbo = 1; cm->pesc = ZAP__T_ESC; cm->pfar = ZAP__T_FAR; cm->farlim = ZAP__T_FARLIM;
    for (int i = 0; i < ZAP__T_NTRI; i++) { size_t ll = (size_t)i / 33 % 17, ml = (size_t)i % 33; ((uint8_t *)cm->tri)[i] = ml >= 4 && ll <= 16; }
    size_t r = zap__compress_opt(src, n, lz, cap, s, NULL, depth, cm);
    if (r && zap__t_table(lz, r, (uint8_t *)cm->tri, NULL) >= 0) r = zap__compress_opt(src, n, lz, cap, s, NULL, depth, cm);
    free(cm);
    return r;
}

static inline int zap__t_len(const uint8_t **p, const uint8_t *e, size_t *v) {
    if (*p >= e) return -1;
    uint8_t b = *(*p)++;
    if (b < 255) { *v = b; return 0; }
    if (e - *p < 3) return -1;
    *v = 255 + ((size_t)(*p)[0] | (size_t)(*p)[1] << 8 | (size_t)(*p)[2] << 16);
    *p += 3;
    return 0;
}

/* match copy for escapes and the block tail: 1 <= off <= op - start and ml <= oend - op checked by the caller */
static inline void zap__t_copy(uint8_t *op, size_t off, size_t ml, uint8_t *oend) {
    const uint8_t *m = op - off;
    if (off >= 16 && (size_t)(oend - op) >= ml + 16) { size_t j = 0; do { memcpy(op + j, m + j, 16); j += 16; } while (j < ml); }
    else if (off >= 8 && (size_t)(oend - op) >= ml + 8) { size_t j = 0; do { memcpy(op + j, m + j, 8); j += 8; } while (j < ml); }
    else for (size_t j = 0; j < ml; j++) op[j] = m[j];
}

/* 16 bytes from s to d, which may overlap: through a temporary, so neither memcpy overlaps (memmove would be the
   obvious call, but MSVC doesn't inline it); compilers make this one 16-byte load and store */
static inline void zap__cp16(uint8_t *d, const uint8_t *s) { uint8_t t[16]; memcpy(t, s, 16); memcpy(d, t, 16); }

/* table entry: bits 0-7 lit, 8-15 ml, 16-23 offset bytes, bit 31 escape; bits 32-63 offset mask (0 = repeat the
   previous offset). An unused code is ZAP__T_BAD (escape bit set, caught on the escape path). */
#define ZAP__T_BAD 0xFFFFFFFFu
#define ZAP__T_CMD(ESCAPE)                                                                                           \
    do {                                                                                                             \
        uint64_t e = tab[*t++];                                                                                      \
        size_t nb = (e >> 16) & 0xFF, nv = zap__r32(o) & (uint32_t)(e >> 32);                                        \
        o += nb;                                                                                                     \
        off = nb ? nv : off;                                                                                         \
        if ((int32_t)(uint32_t)e < 0) goto ESCAPE;                                                                   \
        size_t ll = e & 0xFF;                                                                                        \
        memcpy(op, lp, 16);                                                                                          \
        op += ll; lp += ll;                                                                                          \
        uintptr_t m = (uintptr_t)op - off; /* integer math: the check reuses the address the copy needs */           \
        if (m < (uintptr_t)dst) return -1;                                                                           \
        zap__cp16(op, (const uint8_t *)m); /* off < 16 only in a corrupt block: may overlap */                       \
        zap__cp16(op + 16, (const uint8_t *)m + 16);                                                                 \
        op += (e >> 8) & 0xFF;                                                                                       \
    } while (0)

static inline ptrdiff_t zap__t_decode(const uint8_t *src, size_t n, uint8_t *dst, size_t raw) {
    static const uint32_t kmask[4] = { 0, 0xFFu, 0xFFFFu, 0xFFFFFFu };
    if (n < 13) return -1;
    size_t ntab = src[12];
    if (ntab > ZAP__T_ESC0 || n - 13 < 2 * ntab) return -1;
    uint64_t tab[256];
    for (int i = 0; i < 256; i++) tab[i] = ZAP__T_BAD;
    for (size_t i = 0; i < ntab; i++) {
        unsigned w = src[13 + 2 * i] | (unsigned)src[14 + 2 * i] << 8, ll = w & 31, ml = (w >> 5 & 31) + 4, kind = w >> 10;
        if (ll > 16 || ml > 32 || kind > 3) return -1;
        tab[i] = ll | ml << 8 | (uint64_t)kind << 16 | (uint64_t)kmask[kind] << 32;
    }
    for (unsigned k = 0; k < 4; k++) tab[ZAP__T_ESC0 + k] = 0x80000000u | (uint64_t)k << 16 | (uint64_t)kmask[k] << 32;
    const uint8_t *t = src + 13 + 2 * ntab, *end = src + n;
    uint64_t ntok = zap__r32(src), noff = zap__r32(src + 4), nlen = zap__r32(src + 8);
    if (ntok + noff + nlen > (uint64_t)(end - t)) return -1;
    const uint8_t *tend = t + ntok, *o = tend, *oe = o + noff, *ln = oe, *le = ln + nlen, *lp = le;
    uint8_t *op = dst, *oend = dst + raw;
    size_t off = 0;
    for (;;) { /* batches of table commands; each reads <= 16 literal and 4 offset bytes and writes <= 48 output bytes */
        size_t kl = (size_t)(end - lp) >> 4, ko = (size_t)(oend - op) >> 6, kt = (size_t)(tend - t), ro = (size_t)(end - o);
        size_t kf = ro >= 4 ? (ro - 4) / 3 : 0, k = kl < ko ? kl : ko;
        k = k < kt ? k : kt; k = k < kf ? k : kf;
        if (k < 2) break;
        const uint8_t *tlim = t + (k & ~(size_t)1);
        while (t < tlim) {
            ZAP__T_CMD(esc);
            ZAP__T_CMD(esc);
            continue;
        esc: { /* its offset (if any) is already in off; the lengths decide what fits, so the batch ends here */
                size_t ll, ml;
                if (tab[t[-1]] == ZAP__T_BAD || zap__t_len(&ln, le, &ll) || zap__t_len(&ln, le, &ml)) return -1;
                ml += 4;
                if (ll > (size_t)(end - lp) || ll > (size_t)(oend - op)) return -1;
                if (ll + 16 <= (size_t)(end - lp) && ll + 16 <= (size_t)(oend - op)) { size_t j = 0; do { memcpy(op + j, lp + j, 16); j += 16; } while (j < ll); }
                else memcpy(op, lp, ll);
                op += ll; lp += ll;
                if (off - 1 >= (size_t)(op - dst) || ml > (size_t)(oend - op)) return -1;
                zap__t_copy(op, off, ml, oend);
                op += ml;
                break;
            }
        }
    }
    if (o > oe) return -1;
    while (t < tend) { /* the last commands, exactly */
        uint64_t e = tab[*t++];
        size_t nb = (e >> 16) & 0xFF, ll = e & 0xFF, ml = (e >> 8) & 0xFF;
        if (e == ZAP__T_BAD || nb > (size_t)(oe - o)) return -1;
        if (nb) { off = 0; for (size_t j = 0; j < nb; j++) off |= (size_t)o[j] << 8 * j; o += nb; }
        if ((uint32_t)e & 0x80000000u) { if (zap__t_len(&ln, le, &ll) || zap__t_len(&ln, le, &ml)) return -1; ml += 4; }
        if (ll > (size_t)(end - lp) || ll > (size_t)(oend - op)) return -1;
        memcpy(op, lp, ll); op += ll; lp += ll;
        if (off - 1 >= (size_t)(op - dst) || ml > (size_t)(oend - op)) return -1;
        zap__t_copy(op, off, ml, oend);
        op += ml;
    }
    if (o != oe || ln != le || (size_t)(end - lp) != (size_t)(oend - op)) return -1;
    memcpy(op, lp, (size_t)(end - lp));
    return (ptrdiff_t)raw;
}

/* turbo block API (you store the sizes). depth >= ZAP_OPT_DEPTH: speed-priced optimal parse (s required);
   1..31: lazy hc parse (s required); 0: fast parse (s may be NULL). 0 = didn't fit / out of memory. */
static inline size_t zap_compress_turbo(const void *src_, size_t n, void *dst, size_t cap, zap_hc_state *s, int depth) {
    const uint8_t *src = (const uint8_t *)src_;
    size_t lcap = zap_bound(n), ln = 0, r = 0;
    uint8_t *lz = (uint8_t *)malloc(lcap);
    zap_state *fs = !(depth & 0xFFFF) ? (zap_state *)malloc(sizeof *fs) : NULL;
    depth &= 0xFFFF;
    if (lz && (depth || fs) && (!depth || s))
        ln = depth >= ZAP_OPT_DEPTH ? zap__t_parse(src, n, lz, lcap, s, depth) : depth ? zap_compress_hc(src, n, lz, lcap, s, NULL, depth) : zap_compress(src, n, lz, lcap, fs, NULL);
    if (ln) r = zap__t_encode(lz, ln, (uint8_t *)dst, cap);
    free(lz); free(fs);
    return r;
}
static inline ptrdiff_t zap_decompress_turbo(const void *src, size_t n, void *dst, size_t raw_size) {
    return zap__t_decode((const uint8_t *)src, n, (uint8_t *)dst, raw_size);
}

/* ---------------- entropy v3: the Kraken tier (zap_compress_entropy at depth >= ZAP_OPT_DEPTH, no dictionary)
 * Multi-arrival optimal parse priced exactly as coded; byte-coded offsets; contextual tokens; literals per 128 KB
 * chunk of output (raw, plain, or contextual); Huffman streams split in 6 or 8 parts decoded symbol-interleaved.
 * Block (tag 3 in the top two bits of the first word; v2 blocks have 2, v1 blocks 0 or 1):
 *   u32 n_lits | 3 << 30, u32 n_seq, u32 n_len, u32 n_near, u32 n_far, u32 n_chunks | scale << 24,
 *   n_chunks x u32 literal counts (bit 31: delta chunk), 8 streams, each u8 method | L << 4 (0 raw, 1 plain Huffman, 2 contextual Huffman;
 *   15 the literal records), u32 size, data: literals, tokens, length bytes, offset symbols, near low bytes, far low,
 *   far mid, far high bytes.
 * Sequence i: token = literal nibble | (match - 3) nibble, 15 continuing as 255-runs in the length stream; offset
 * symbol 0-2 = repeat offset 0-2 (move-to-front), else a new offset from a value v: near v = h << 8 | (next near low
 * byte), far v = next far low | mid << 8 | high << 16. Scale 1: symbol 3 + h near (h <= 250), 255 far, offset v.
 * Scale S > 1 (strided data: pixels, vertices): 3 + h near (h < 125) and 254 far give offset v * S; 128 + h near
 * (h < 125) and 255 far give offset v. Leftover literals end the block.
 * Huffman data: [u8 ntables][ntables x 128 bytes of 4-bit code lengths][u32 sizes of parts 0..W-2][W parts]; part
 * k holds symbols [k q, (k + 1) q), q = ceil(n / W), LSB-first; contextual: a part's table is the group of its
 * previous symbol (group 0 at its start). Tokens: 6 parts, 16 groups (zap__grp_tok), L11. Plain streams: 8 parts, L11.
 * Literal stream: [u8 sets][16 x 128 contextual tables, L10, per set: bit 0 literals, bit 1 deltas] then per chunk
 * [u8 mode][u32 bytes][payload]: mode 0 raw, 1 plain [128-byte table][u32 x 7][8 parts], 2 contextual [u32 x 5][6
 * parts] (groups: previous symbol >> 4). A delta chunk's symbols are literal - the byte at the last match's offset
 * (0 before the first match); its contextual chunks take the second set. Chunks start at sequences: a literal run
 * is in one chunk. */
#if defined(_MSC_VER) && !defined(__clang__)
#define ZAP__NOINLINE __declspec(noinline)
#if defined(__AVX2__) /* /arch:AVX2: Haswell-class, LZCNT and BMI2 */
#define ZAP__BMI2 1
static inline unsigned zap__clz64(uint64_t x) { return (unsigned)__lzcnt64(x); }
#else
static inline unsigned zap__clz64(uint64_t x) { unsigned long i; _BitScanReverse64(&i, x); return 63u - (unsigned)i; } /* x != 0 */
#endif
#else
#define ZAP__NOINLINE __attribute__((noinline))
static inline unsigned zap__clz64(uint64_t x) { return (unsigned)__builtin_clzll(x); } /* x != 0 */
#endif
#if defined(__BMI2__) && !defined(ZAP__BMI2)
#define ZAP__BMI2 1
#endif
#ifndef ZAP__BMI2
#define ZAP__BMI2 0
#endif
#if defined(__SSSE3__) || defined(__AVX__)
#include <immintrin.h>
#define ZAP__PSHUFB 1
#else
#define ZAP__PSHUFB 0
#endif
#if defined(_M_X64) || defined(__x86_64__)
#define ZAP__X86 1
#else
#define ZAP__X86 0
#endif
/* Huffman decoders come in an X variant (BMI2 shrx + LZCNT; length-low table entries) and, unless the build targets
   BMI2, a B variant (plain shifts; symbol-low entries). On x86-64 without BMI2 in the build, CPUID picks one. */
#define ZAP__KXV (ZAP__BMI2 || ZAP__X86)
#if ZAP__BMI2
#define ZAP__KXATTR
#define ZAP__KXSHR(v, c) ((v) >> (c))
#define ZAP__KXCLZ(v) zap__clz64(v)
#elif defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#define ZAP__KXATTR
#define ZAP__KXSHR(v, c) _shrx_u64((v), (unsigned)(c))
#define ZAP__KXCLZ(v) ((unsigned)__lzcnt64(v))
#elif ZAP__X86
#define ZAP__KXATTR __attribute__((target("bmi2,lzcnt")))
#define ZAP__KXSHR(v, c) ((v) >> (c))
#define ZAP__KXCLZ(v) ((unsigned)__builtin_clzll(v))
#endif
#if ZAP__X86 && !(ZAP__BMI2 && ZAP__PSHUFB)
#if defined(_MSC_VER) /* MSVC, and clang targeting it (<cpuid.h>'s __cpuid macro would break a later <intrin.h>) */
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
static inline int zap__k_cpu(void) { /* bit 0: BMI2 and LZCNT (the X decoders), bit 1: SSSE3 */
#if ZAP__BMI2 && ZAP__PSHUFB
    return 3;
#elif ZAP__X86
    static int f = -1; /* benign race: every thread computes the same */
    if (f < 0) {
#if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, 7, 0);
        int bmi2 = (r[1] >> 8) & 1;
        __cpuid(r, (int)0x80000001);
        int lz = (r[2] >> 5) & 1;
        __cpuid(r, 1);
        f = (bmi2 && lz) | ((r[2] >> 9) & 1) << 1;
#else
        unsigned a, b, c, d, bmi2 = 0, lz = 0, s3 = 0;
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) bmi2 = (b >> 8) & 1;
        if (__get_cpuid(0x80000001u, &a, &b, &c, &d)) lz = (c >> 5) & 1;
        if (__get_cpuid(1, &a, &b, &c, &d)) s3 = (c >> 9) & 1;
        f = (int)((bmi2 && lz) | s3 << 1);
#endif
#if ZAP__BMI2
        f |= 1;
#endif
#if ZAP__PSHUFB
        f |= 2;
#endif
    }
    return f;
#else
    return 0;
#endif
}
static inline int zap__k_x(void) { return zap__k_cpu() & 1; } /* use the X decoders */
#define ZAP__K_CHUNK (128u << 10) /* literal chunks: output bytes */
#define ZAP__K_NEAR (251u << 8)   /* scale 1: offsets below are near (high byte in the offset symbol) */
#define ZAP__K_N 512              /* sequences per offset-stage batch */

/* ---- Huffman streams in parts */
static inline void zap__k_head(uint8_t *o, const uint8_t (*len)[256], int nt) {
    *o++ = (uint8_t)nt;
    for (int g = 0; g < nt; g++) for (int i = 0; i < 128; i++) *o++ = (uint8_t)(len[g][2 * i] | len[g][2 * i + 1] << 4);
}
/* [u32 sizes of parts 0..W-2][W parts] of s[0..n); grp: chain context (NULL = one table). NULL if it doesn't fit */
static inline uint8_t *zap__k_parts(uint8_t *o, uint8_t *oe, const uint8_t *s, size_t n, int W, const uint8_t *grp, const uint8_t (*len)[256]) {
    uint16_t (*code)[256] = (uint16_t(*)[256])malloc(sizeof(uint16_t) * 256 * (grp ? 16 : 1));
    uint8_t *szp = o;
    size_t q = (n + W - 1) / W;
    if (!code || oe - o < 4 * W) { free(code); return NULL; }
    for (int g = 0; g < (grp ? 16 : 1); g++) zap__hcodes(len[g], code[g]);
    o += 4 * (W - 1);
    for (int k = 0; k < W && o; k++) {
        size_t lo = (size_t)k * q < n ? (size_t)k * q : n, hi = lo + q < n ? lo + q : n;
        uint8_t *start = o;
        uint64_t acc = 0;
        int nb = 0;
        unsigned g = 0;
        for (size_t i = lo; i < hi; i++) {
            acc |= (uint64_t)code[g][s[i]] << nb; nb += len[g][s[i]];
            if (grp) g = grp[s[i]];
            while (nb >= 8) { if (o >= oe) { o = NULL; break; } *o++ = (uint8_t)acc; acc >>= 8; nb -= 8; }
            if (!o) break;
        }
        if (o && nb) { if (o >= oe) o = NULL; else *o++ = (uint8_t)acc; }
        if (o && k < W - 1) zap__w32(szp + 4 * k, (uint32_t)(o - start));
    }
    free(code);
    return o;
}
/* code lengths for s[0..n) in W parts (grp: per previous-symbol group, 16 tables) */
static inline void zap__k_lens(const uint8_t *s, size_t n, int W, const uint8_t *grp, int L, uint8_t (*len)[256]) {
    uint32_t (*f)[256] = (uint32_t(*)[256])calloc(grp ? 16 : 1, sizeof *f);
    if (!f) { memset(len, 8, 256 * (grp ? 16 : 1)); return; }
    size_t q = (n + W - 1) / W;
    for (size_t i = 0; i < n; i++) f[grp && q && i % q ? grp[s[i - 1]] : 0][s[i]]++;
    for (int g = 0; g < (grp ? 16 : 1); g++) zap__hlens_l(f[g], len[g], L);
    free(f);
}
/* a stream: raw, or Huffman when that saves >= 1% (plain: 8 parts L11; grp: contextual 16 groups, 6 parts, L) */
static inline uint8_t *zap__k_stream(uint8_t *op, uint8_t *oend, const uint8_t *s, size_t n, const uint8_t *grp, int L) {
    if (!op || oend - op < 5) return NULL;
    uint8_t *o = NULL;
    int nt = grp ? 16 : 1;
    if (n >= 64 && (size_t)(oend - op) > 5 + 1 + 128 * (size_t)nt + 32) {
        uint8_t (*len)[256] = (uint8_t(*)[256])malloc(256 * (size_t)nt);
        if (len) {
            zap__k_lens(s, n, grp ? 6 : 8, grp, grp ? L : 11, len);
            zap__k_head(op + 5, (const uint8_t(*)[256])len, nt);
            o = zap__k_parts(op + 6 + 128 * nt, oend, s, n, grp ? 6 : 8, grp, (const uint8_t(*)[256])len);
            free(len);
        }
        if (o && (size_t)(o - op - 5) >= n - n / 100) o = NULL;
    }
    if (o) { op[0] = (uint8_t)((grp ? 2 : 1) | (grp ? L : 11) << 4); zap__w32(op + 1, (uint32_t)(o - op - 5)); return o; }
    if ((size_t)(oend - op) - 5 < n) return NULL;
    op[0] = 0; zap__w32(op + 1, (uint32_t)n); memcpy(op + 5, s, n);
    return op + 5 + n;
}
/* table entries: X len | group << 4 | sym << 8 (shrx shifts by the entry itself); B sym | len << 8 | group << 12 (the
   symbol stores straight from the entry; shifts go through CL anyway) */
#define ZAP__KENTRY(x, s, l, g) ((x) ? ((l) | (g) << 4 | (s) << 8) : ((s) | (l) << 8 | (g) << 12))
#define ZAP__KSYM(x, e) ((uint8_t)((x) ? (e) >> 8 : (e)))
#define ZAP__KLEN(x, e) ((x) ? (e) & 15 : (e) >> 8 & 15)
#define ZAP__KGRP(x, e) ((x) ? (e) >> 4 & 15 : (e) >> 12)
/* decode tables from nt x 128 nibbles (group = grp[sym], 0 without grp) */
static inline int zap__k_tables(const uint8_t *in, int nt, const uint8_t *grp, int L, uint16_t *T, int x) {
    for (int g = 0; g < nt; g++) {
        uint8_t len[256];
        uint16_t code[256];
        uint32_t kraft = 0;
        for (int i = 0; i < 128; i++) { len[2 * i] = in[128 * g + i] & 15; len[2 * i + 1] = in[128 * g + i] >> 4; }
        for (int i = 0; i < 256; i++) { if (len[i] > L) return -1; if (len[i]) kraft += 1u << (L - len[i]); }
        if (kraft > 1u << L) return -1;
        zap__hcodes(len, code);
        uint16_t *t = T + ((size_t)g << L);
        memset(t, 0, sizeof(uint16_t) << L);
        for (int s = 0; s < 256; s++) {
            unsigned c = grp ? grp[s] : 0;
            if (c >= (unsigned)nt) return -1;
            if (len[s]) for (int j = code[s]; j < 1 << L; j += 1 << len[s]) t[j] = (uint16_t)ZAP__KENTRY(x, s, len[s], c);
        }
    }
    return 0;
}
/* careful tail of one part: cnt symbols from bit b of s (part ends at byte e), context g */
static inline int zap__k_tail(const uint8_t *s, size_t e, size_t b, uint8_t *o, size_t cnt, const uint16_t *T, int L, unsigned g, int ctx, int x) {
    const unsigned M = (1u << L) - 1;
    for (size_t i = 0; i < cnt; i++) {
        size_t by = b >> 3;
        uint64_t v = 0;
        if (by + 8 <= e) v = zap__r64(s + by);
        else for (size_t t = 0; t < 8 && by + t < e; t++) v |= (uint64_t)s[by + t] << 8 * t;
        v >>= b & 7;
        unsigned en = T[(g << L) | (unsigned)(v & M)];
        if (!ZAP__KLEN(x, en) || b + ZAP__KLEN(x, en) > 8 * e) return -1;
        o[i] = ZAP__KSYM(x, en); b += ZAP__KLEN(x, en);
        if (ctx) g = ZAP__KGRP(x, en);
    }
    return 0;
}
/* decoders: in = [u32 sizes][W parts], tables built. A step decodes 5 symbols of every part (one 8-byte load each:
   >= 57 bits); all parts' dependency chains sit side by side so the core overlaps them; a marker bit gives each
   part's consumed bits (clz). Symbols go to one row per part of a stack buffer (a fixed stride, so no output pointer
   per part has to live in a register) and are copied out per slice. Tails go through zap__k_tail. */
#define ZAP__KSL 1020  /* symbols per part per slice (a multiple of 5) */
#define ZAP__KROW 1088 /* row stride */
#define ZAP__KRB(k) uint64_t v##k = (zap__r64(s + (b[k] >> 3)) >> (b[k] & 7)) | (1ull << 63)
#define ZAP__KDB(k) b[k] += zap__clz64(v##k)
#define ZAP__KPB(k, j) do { unsigned e_ = T[v##k & M]; rp[(k) * ZAP__KROW + (j)] = (uint8_t)e_; v##k >>= e_ >> 8; } while (0)
#define ZAP__KCB(k, j) do { unsigned e_ = T[g##k | (unsigned)(v##k & M)]; rp[(k) * ZAP__KROW + (j)] = (uint8_t)e_; v##k >>= (e_ >> 8 & 15); g##k = (e_ >> 12) << L; } while (0)
#if ZAP__KXV
#define ZAP__KRX(k) uint64_t v##k = ZAP__KXSHR(zap__r64(s + (b[k] >> 3)), b[k] & 7) | (1ull << 63)
#define ZAP__KDX(k) b[k] += ZAP__KXCLZ(v##k)
#define ZAP__KPX(k, j) do { unsigned e_ = T[v##k & M]; rp[(k) * ZAP__KROW + (j)] = (uint8_t)(e_ >> 8); v##k = ZAP__KXSHR(v##k, e_ & 63); } while (0)
#define ZAP__KCX(k, j) do { unsigned e_ = T[g##k | (unsigned)(v##k & M)]; rp[(k) * ZAP__KROW + (j)] = (uint8_t)(e_ >> 8); v##k = ZAP__KXSHR(v##k, e_ & 15); g##k = (e_ & 0xF0) << (L - 4); } while (0)
#endif
#define ZAP__KSETUP(W)                                                                                                     \
    const unsigned M = (1u << L) - 1;                                                                                      \
    size_t hdr = 4 * (W - 1), q = (raw + W - 1) / W, b[W], e[W], cnt[W], lo[W], at = 0, i = 0;                             \
    uint8_t row[W * ZAP__KROW];                                                                                            \
    if (n < hdr) return -1;                                                                                                \
    const uint8_t *s = in + hdr;                                                                                           \
    for (int k = 0; k < W; k++) {                                                                                          \
        size_t sz = k < W - 1 ? zap__r32(in + 4 * (size_t)k) : n - hdr - at;                                               \
        if (sz > n - hdr - at) return -1;                                                                                  \
        b[k] = 8 * at; e[k] = at + sz; at += sz;                                                                           \
        lo[k] = (size_t)k * q < raw ? (size_t)k * q : raw; cnt[k] = lo[k] + q < raw ? q : raw - lo[k];                     \
    }                                                                                                                      \
    unsigned gg[W] = { 0 };
#define ZAP__KITERS(W)                                                                                                     \
    size_t it = (cnt[W - 1] - i) / 5;                                                                                      \
    for (int k = 0; k < W; k++) { size_t by = b[k] >> 3, m_ = e[k] >= by + 8 ? (e[k] - by - 8) / 7 + 1 : 0; if (m_ < it) it = m_; } \
    if (it > ZAP__KSL / 5) it = ZAP__KSL / 5;
#define ZAP__KOUT(W)                                                                                                       \
    for (int k = 0; k < W; k++) memcpy(out + lo[k] + i, row + (size_t)k * ZAP__KROW, x);                                  \
    i += x;
#define ZAP__KTAILS(W, CTX, XL)                                                                                            \
    for (int k = 0; k < W; k++) {                                                                                          \
        size_t done = i < cnt[k] ? i : cnt[k];                                                                             \
        if (zap__k_tail(s, e[k], b[k], out + lo[k] + done, cnt[k] - done, T, L, gg[k], CTX, XL)) return -1;                \
    }                                                                                                                      \
    return 0;
#define ZAP__KDEC8(NAME, CTX, L_, V, ATTR)                                                                                          \
    ZAP__NOINLINE ATTR static int NAME(const uint8_t *in, size_t n, uint8_t *out, size_t raw, const uint16_t *T) {         \
        enum { L = L_ };                                                                                                   \
        ZAP__KSETUP(8)                                                                                                     \
        for (;;) {                                                                                                         \
            ZAP__KITERS(8)                                                                                                 \
            if (!it) break;                                                                                                \
            unsigned g0 = gg[0] << L, g1 = gg[1] << L, g2 = gg[2] << L, g3 = gg[3] << L, g4 = gg[4] << L, g5 = gg[5] << L, g6 = gg[6] << L, g7 = gg[7] << L; \
            uint8_t *rp = row, *rpe = row + 5 * it; /* row pointer and end: a register fewer than index, count, limit */\
            for (; rp < rpe; rp += 5) {                                                                                \
                ZAP__KR##V(0); ZAP__KR##V(1); ZAP__KR##V(2); ZAP__KR##V(3); ZAP__KR##V(4); ZAP__KR##V(5); ZAP__KR##V(6); ZAP__KR##V(7);            \
                if (CTX) { for (int j = 0; j < 5; j++) { ZAP__KC##V(0, j); ZAP__KC##V(1, j); ZAP__KC##V(2, j); ZAP__KC##V(3, j); ZAP__KC##V(4, j); ZAP__KC##V(5, j); ZAP__KC##V(6, j); ZAP__KC##V(7, j); } } \
                else { for (int j = 0; j < 5; j++) { ZAP__KP##V(0, j); ZAP__KP##V(1, j); ZAP__KP##V(2, j); ZAP__KP##V(3, j); ZAP__KP##V(4, j); ZAP__KP##V(5, j); ZAP__KP##V(6, j); ZAP__KP##V(7, j); } } \
                ZAP__KD##V(0); ZAP__KD##V(1); ZAP__KD##V(2); ZAP__KD##V(3); ZAP__KD##V(4); ZAP__KD##V(5); ZAP__KD##V(6); ZAP__KD##V(7);            \
            }                                                                                                              \
            gg[0] = g0 >> L; gg[1] = g1 >> L; gg[2] = g2 >> L; gg[3] = g3 >> L; gg[4] = g4 >> L; gg[5] = g5 >> L; gg[6] = g6 >> L; gg[7] = g7 >> L; \
            size_t x = (size_t)(rp - row);                                                                             \
            ZAP__KOUT(8)                                                                                               \
        }                                                                                                                  \
        ZAP__KTAILS(8, CTX, ZAP__KXL##V)                                                                                  \
    }
#define ZAP__KDEC6(NAME, CTX, L_, V, ATTR)                                                                                          \
    ZAP__NOINLINE ATTR static int NAME(const uint8_t *in, size_t n, uint8_t *out, size_t raw, const uint16_t *T) {         \
        enum { L = L_ };                                                                                                   \
        ZAP__KSETUP(6)                                                                                                     \
        for (;;) {                                                                                                         \
            ZAP__KITERS(6)                                                                                                 \
            if (!it) break;                                                                                                \
            unsigned g0 = gg[0] << L, g1 = gg[1] << L, g2 = gg[2] << L, g3 = gg[3] << L, g4 = gg[4] << L, g5 = gg[5] << L; \
            uint8_t *rp = row, *rpe = row + 5 * it; /* row pointer and end: a register fewer than index, count, limit */\
            for (; rp < rpe; rp += 5) {                                                                                \
                ZAP__KR##V(0); ZAP__KR##V(1); ZAP__KR##V(2); ZAP__KR##V(3); ZAP__KR##V(4); ZAP__KR##V(5);                                    \
                if (CTX) { for (int j = 0; j < 5; j++) { ZAP__KC##V(0, j); ZAP__KC##V(1, j); ZAP__KC##V(2, j); ZAP__KC##V(3, j); ZAP__KC##V(4, j); ZAP__KC##V(5, j); } } \
                else { for (int j = 0; j < 5; j++) { ZAP__KP##V(0, j); ZAP__KP##V(1, j); ZAP__KP##V(2, j); ZAP__KP##V(3, j); ZAP__KP##V(4, j); ZAP__KP##V(5, j); } } \
                ZAP__KD##V(0); ZAP__KD##V(1); ZAP__KD##V(2); ZAP__KD##V(3); ZAP__KD##V(4); ZAP__KD##V(5);                                    \
            }                                                                                                              \
            gg[0] = g0 >> L; gg[1] = g1 >> L; gg[2] = g2 >> L; gg[3] = g3 >> L; gg[4] = g4 >> L; gg[5] = g5 >> L;         \
            size_t x = (size_t)(rp - row);                                                                             \
            ZAP__KOUT(6)                                                                                               \
        }                                                                                                                  \
        ZAP__KTAILS(6, CTX, ZAP__KXL##V)                                                                                  \
    }
#define ZAP__KXLX 1
#define ZAP__KXLB 0
#if ZAP__KXV
ZAP__KDEC8(zap__k_p8X, 0, 11, X, ZAP__KXATTR)
ZAP__KDEC6(zap__k_c6X, 1, 10, X, ZAP__KXATTR)
ZAP__KDEC6(zap__k_c6lX, 1, 11, X, ZAP__KXATTR)
#endif
#if !ZAP__BMI2
ZAP__KDEC8(zap__k_p8B, 0, 11, B, )
ZAP__KDEC6(zap__k_c6B, 1, 10, B, )
ZAP__KDEC6(zap__k_c6lB, 1, 11, B, )
#endif
#if ZAP__BMI2
#define ZAP__KCALL(x, f) f##X
#elif ZAP__KXV
#define ZAP__KCALL(x, f) ((x) ? f##X : f##B)
#else
#define ZAP__KCALL(x, f) f##B
#endif

/* a whole stream (tokens, lengths, offset bytes): method from the header byte, tables into T */
static inline int zap__k_dstream(int m, int L, const uint8_t *in, size_t sz, uint8_t *out, size_t cnt, const uint8_t *grp, uint16_t *T) {
    int nt = m == 2 ? 16 : 1;
    if ((m != 1 && m != 2) || L != 11 || !cnt || (m == 2 && !grp) || sz < 1 + 128 * (size_t)nt || in[0] != nt) return -1;
    int x = zap__k_x();
    if (zap__k_tables(in + 1, nt, m == 2 ? grp : NULL, L, T, x)) return -1;
    return m == 2 ? ZAP__KCALL(x, zap__k_c6l)(in + 1 + 128 * nt, sz - 1 - 128 * nt, out, cnt, T) : ZAP__KCALL(x, zap__k_p8)(in + 1 + 128 * nt, sz - 1 - 128 * nt, out, cnt, T);
}

/* ---- literals per chunk: raw, plain (own table), or contextual (the block's 16 tables), whichever is smallest */
/* coded bytes of s[0..n) in W parts, or (size_t)-1 if a symbol has no code (a table set built from other chunks) */
static inline size_t zap__k_bits(const uint8_t *s, size_t n, const uint8_t (*len)[256], const uint8_t *grp, int W) {
    uint64_t b = 0;
    size_t q = (n + W - 1) / W;
    for (size_t i = 0; i < n; i++) { unsigned g = grp && i % q ? grp[s[i - 1]] : 0; if (!len[g][s[i]]) return (size_t)-1; b += len[g][s[i]]; }
    return (size_t)((b + 7) / 8) + (size_t)W;
}
/* plain-chunk bytes of s[0..n), its code lengths in lp ((size_t)-1: too short to pay) */
static inline size_t zap__k_pcost(const uint8_t *s, size_t n, uint8_t lp[1][256]) {
    uint32_t fp[256] = { 0 };
    for (size_t i = 0; i < n; i++) fp[s[i]]++;
    zap__hlens_l(fp, lp[0], 11);
    return n >= 64 ? zap__k_bits(s, n, (const uint8_t(*)[256])lp, NULL, 8) + 128 + 28 : (size_t)-1; /* own table: every symbol has a code */
}
/* Each chunk is raw, plain (own table) or contextual (a block table set), of its literals or of their deltas (dls:
   literal - the byte at the last match offset; strided data - heightmaps, vertices, nav meshes - makes those small).
   A mode pays for its decode time: plain Huffman (~1.6 ticks a literal) must save ~1% of the chunk over raw (a
   memcpy), contextual (~2.5) another 0.5% over plain, deltas 0.2% more; about the exchange rate Kraken's parse
   settles on. On near-incompressible literals (BC7 textures: Huffman saved 0.04%) chunks stay raw: ~10x faster.
   rawpct: raw unless Huffman saves more than this many percent too (ZAP_FAST_DECODE: 4). dflag: the delta chunks. */
static inline uint8_t *zap__k_lits(uint8_t *op, uint8_t *oend, const uint8_t *lits, const uint8_t *dls, const size_t *cb, size_t nch, int rawpct, uint8_t *dflag) {
    uint8_t glit[256], *mode = (uint8_t *)malloc(nch + 1), lp[1][256];
    uint32_t (*fq)[16][256] = (uint32_t(*)[16][256])malloc(sizeof(uint32_t) * 2 * 16 * 256);
    uint8_t (*lc)[16][256] = (uint8_t(*)[16][256])malloc(2 * 16 * 256);
    uint8_t *o = NULL;
    int64_t gain[2] = { 0, 0 };
    zap__grp_lit(glit);
    if (!mode || !fq || !lc || !op || oend - op < 1) goto done;
    for (size_t c = 0; c < nch; c++) { mode[c] = 2; dflag[c] = 2; } /* first round: both sets from every chunk */
    for (int it = 0; it < 3; it++) { /* the block's table sets from the contextual chunks, then decide again */
        memset(fq, 0, sizeof(uint32_t) * 2 * 16 * 256);
        for (size_t c = 0; c < nch; c++) if (mode[c] == 2)
            for (int d = 0; d < 2; d++) if (dflag[c] == d || dflag[c] == 2) {
                size_t n = cb[c + 1] - cb[c], q = (n + 5) / 6;
                const uint8_t *t = (d ? dls : lits) + cb[c];
                for (size_t i = 0; i < n; i++) fq[d][i % q ? glit[t[i - 1]] : 0][t[i]]++;
            }
        for (int d = 0; d < 2; d++) for (int g = 0; g < 16; g++) zap__hlens_l(fq[d][g], lc[d][g], 10);
        gain[0] = gain[1] = 0;
        for (size_t c = 0; c < nch; c++) {
            size_t n = cb[c + 1] - cb[c], best = n - n * (size_t)rawpct / 100, np = best;
            int bm = 0, bd = 0;
            for (int d = 0; d < 2; d++) {
                const uint8_t *t = (d ? dls : lits) + cb[c];
                size_t pen = n / 100 + (size_t)d * n / 512, sp = zap__k_pcost(t, n, lp), sc = n >= 256 ? zap__k_bits(t, n, (const uint8_t(*)[256])lc[d], glit, 6) : (size_t)-1;
                if (sc != (size_t)-1) sc += 20;
                if (sp != (size_t)-1 && sp + pen < np) np = sp + pen;
                if (sp != (size_t)-1 && sp + pen < best) { best = sp + pen; bm = 1; bd = d; }
                if (sc != (size_t)-1 && sc + pen + n / 200 < best) { best = sc + pen + n / 200; bm = 2; bd = d; }
            }
            mode[c] = (uint8_t)bm; dflag[c] = (uint8_t)bd;
            if (bm == 2) gain[bd] += (int64_t)np - (int64_t)best;
        }
    }
    for (size_t c = 0; c < nch; c++) if (mode[c] == 2 && gain[dflag[c]] <= 16 * 128) { /* a set that costs more than it saves: raw or plain */
        size_t n = cb[c + 1] - cb[c], best = n - n * (size_t)rawpct / 100;
        mode[c] = 0; dflag[c] = 0;
        for (int d = 0; d < 2; d++) {
            size_t sp = zap__k_pcost((d ? dls : lits) + cb[c], n, lp);
            if (sp != (size_t)-1 && sp + n / 100 + (size_t)d * n / 512 < best) { best = sp + n / 100 + (size_t)d * n / 512; mode[c] = 1; dflag[c] = (uint8_t)d; }
        }
    }
    o = op;
    {
        int sets = (gain[0] > 16 * 128) | (gain[1] > 16 * 128) << 1;
        if ((size_t)(oend - o) < 1 + (size_t)((sets & 1) + (sets >> 1)) * 16 * 128) { o = NULL; goto done; }
        *o++ = (uint8_t)sets;
        for (int d = 0; d < 2; d++) if (sets >> d & 1) for (int g = 0; g < 16; g++) for (int i = 0; i < 128; i++) *o++ = (uint8_t)(lc[d][g][2 * i] | lc[d][g][2 * i + 1] << 4);
    }
    for (size_t c = 0; c < nch && o; c++) {
        size_t n = cb[c + 1] - cb[c];
        const uint8_t *t = (dflag[c] ? dls : lits) + cb[c];
        uint8_t *rec = o;
        if (oend - o < 5) { o = NULL; break; }
        o += 5;
        if (mode[c] == 0) { if ((size_t)(oend - o) < n) { o = NULL; break; } memcpy(o, t, n); o += n; }
        else if (mode[c] == 1) {
            zap__k_pcost(t, n, lp);
            if (oend - o < 128) { o = NULL; break; }
            for (int i = 0; i < 128; i++) *o++ = (uint8_t)(lp[0][2 * i] | lp[0][2 * i + 1] << 4);
            o = zap__k_parts(o, oend, t, n, 8, NULL, (const uint8_t(*)[256])lp);
        } else o = zap__k_parts(o, oend, t, n, 6, glit, (const uint8_t(*)[256])lc[dflag[c]]);
        if (o) { rec[0] = mode[c]; zap__w32(rec + 1, (uint32_t)(o - rec - 5)); }
    }
done:
    free(mode); free(fq); free(lc);
    return o;
}
static inline int zap__k_dlits(const uint8_t *in, size_t sz, const uint8_t *chunks, size_t nch, uint8_t *out, size_t nl, uint16_t *Tc, uint16_t *Tp) {
    const uint8_t *ip, *ie = in + sz;
    uint8_t glit[256];
    size_t done = 0;
    zap__grp_lit(glit);
    int x = zap__k_x();
    if (sz < 1 || in[0] > 3) return -1;
    int sets = in[0];
    ip = in + 1;
    for (int d = 0; d < 2; d++) if (sets >> d & 1) { /* literal and delta table sets */
        if ((size_t)(ie - ip) < 16 * 128 || zap__k_tables(ip, 16, glit, 10, Tc + ((size_t)d << 14), x)) return -1;
        ip += 16 * 128;
    }
    for (size_t c = 0; c < nch; c++) {
        size_t w = zap__r32(chunks + 4 * c), n = w & 0x7FFFFFFFu, d = w >> 31;
        if (ie - ip < 5 || n > nl - done) return -1;
        int m = ip[0];
        size_t rs = zap__r32(ip + 1);
        ip += 5;
        if (rs > (size_t)(ie - ip)) return -1;
        if (m == 0) { if (rs != n) return -1; memcpy(out + done, ip, n); }
        else if (!n) return -1;
        else if (m == 1) { if (rs < 128 || zap__k_tables(ip, 1, NULL, 11, Tp, x) || ZAP__KCALL(x, zap__k_p8)(ip + 128, rs - 128, out + done, n, Tp)) return -1; }
        else if (m == 2) { if (!(sets >> d & 1) || ZAP__KCALL(x, zap__k_c6)(ip, rs, out + done, n, Tc + (d << 14))) return -1; }
        else return -1;
        ip += rs; done += n;
    }
    return ip == ie && done == nl ? 0 : -1;
}

/* ---- encoder. A parse is a list of sequences: a literal run, then a match; the last one is literals only (ml 0).
   In a sampled parse (statistics only), offset 0 with ml > 0 is a gap: ml bytes not parsed. */
typedef struct { uint32_t ll, ml, off; } zap__kseq;
static inline unsigned zap__k_tok(size_t ll, size_t ml) { return (unsigned)((ll < 15 ? ll : 15) << 4 | (ml - 3 < 15 ? ml - 3 : 15)); }
/* new offset -> offset symbol and the value it codes (v: near low byte, or the 3 far bytes), for the block's scale */
typedef struct { size_t S; uint64_t M; } zap__ksc; /* M: off / S as (off * M) >> 32, exact for off < 2^26 (S <= 64) */
static inline void zap__ksc_set(zap__ksc *sc, unsigned S) { sc->S = S; sc->M = ((uint64_t)1 << 32) / S + 1; }
static inline unsigned zap__k_ocls(const zap__ksc *sc, size_t off, size_t *v) {
    if (sc->S == 1) { *v = off; return off < ZAP__K_NEAR ? 3u + (unsigned)(off >> 8) : 255u; }
    size_t q = (size_t)((off * sc->M) >> 32);
    int d = q * sc->S == off;
    *v = d ? q : off;
    return *v < (125u << 8) ? (d ? 3u : 128u) + (unsigned)(*v >> 8) : d ? 254u : 255u;
}
/* Huffman-coded bytes of a histogram, table included (an estimate for the encoder's choices) */
static inline size_t zap__k_hbytes(const uint32_t *h) {
    uint8_t len[256];
    uint64_t b = 0, t = 0;
    for (int i = 0; i < 256; i++) t += h[i];
    if (!t) return 0;
    zap__hlens(h, len);
    for (int i = 0; i < 256; i++) b += (uint64_t)h[i] * len[i];
    return (size_t)(b / 8) + 128;
}
/* the offset scale (1..ZAP__K_MAXS) under which the parse's new offsets code smallest: a stride that divides most
   of them (3 for RGB pixels, a vertex size) takes log2(S) bits off each, and more of them stay near */
#define ZAP__K_MAXS 64
static inline unsigned zap__k_scale(const zap__kseq *q, size_t nq) {
    uint32_t *o = (uint32_t *)malloc(sizeof(uint32_t) * (nq + 1)), rep[3] = { 0, 0, 0 }, reps[3] = { 0, 0, 0 };
    uint32_t (*h)[256] = (uint32_t(*)[256])malloc(5 * sizeof *h);
    size_t no = 0, bc = 0;
    unsigned best = 1;
    if (!o || !h) { free(o); free(h); return 1; }
    for (size_t i = 0; i < nq && q[i].ml; i++) {
        if (!q[i].off) { rep[0] = rep[1] = rep[2] = 0; continue; } /* a gap */
        int slot = zap__rep_slot(q[i].off, rep);
        if (slot >= 0) reps[slot]++; else o[no++] = q[i].off;
        zap__rep_push(rep, q[i].off);
    }
    for (unsigned S = 1; S <= ZAP__K_MAXS; S++) {
        zap__ksc sc;
        zap__ksc_set(&sc, S);
        if (S > 1) { /* a scale needs most offsets divisible */
            size_t dv = 0;
            for (size_t i = 0; i < no; i++) dv += (size_t)((o[i] * sc.M) >> 32) * S == o[i];
            if (dv * 2 < no) continue;
        }
        memset(h, 0, 5 * sizeof *h);
        for (int k = 0; k < 3; k++) h[0][k] = reps[k];
        for (size_t i = 0; i < no; i++) {
            size_t v;
            unsigned c = zap__k_ocls(&sc, o[i], &v);
            h[0][c]++;
            if (c < 254) h[1][v & 255]++;
            else { h[2][v & 255]++; h[3][(v >> 8) & 255]++; h[4][(v >> 16) & 255]++; }
        }
        size_t c = 0;
        for (int k = 0; k < 5; k++) c += zap__k_hbytes(h[k]);
        if (S == 1 || c < bc - bc / 512) { bc = c; best = S; }
    }
    free(o); free(h);
    return best;
}
static inline size_t zap__k_encode(const uint8_t *src, const zap__kseq *q, size_t nq, unsigned S, size_t raw, uint8_t *dst, size_t cap, int rawpct) {
    size_t maxseq = nq + 1, nl = 0, ns = 0, nlen = 0, nn = 0, nf = 0, nch = 0, maxch = raw / ZAP__K_CHUNK + 2, pos = 0, cend = 0;
    uint32_t rep[3] = { 0, 0, 0 };
    uint8_t *buf = (uint8_t *)malloc(2 * raw + 9 * maxseq + raw / 255 + maxch + 16), gtok[256];
    size_t *cb = (size_t *)malloc(sizeof(size_t) * (maxch + 1)), lo = 0; /* lo: the last match's offset */
    zap__ksc sc;
    zap__ksc_set(&sc, S);
    if (!buf || !cb || cap < 24 + 4 * maxch) { free(buf); free(cb); return 0; }
    uint8_t *lits = buf, *dls = lits + raw, *toks = dls + raw, *offc = toks + maxseq, *nlo = offc + maxseq, *flo = nlo + maxseq,
            *fmi = flo + maxseq, *fhi = fmi + maxseq, *dfl = fhi + maxseq, *lens = dfl + maxch;
    for (size_t i = 0; i < nq; i++) {
        if (!nch || pos >= cend) { cb[nch++] = nl; cend = (pos / ZAP__K_CHUNK + 1) * ZAP__K_CHUNK; }
        size_t ll = q[i].ll, ml = q[i].ml, off = q[i].off, v;
        for (size_t t = 0; t < ll; t++) { lits[nl + t] = src[pos + t]; dls[nl + t] = (uint8_t)(src[pos + t] - (lo ? src[pos + t - lo] : 0)); }
        nl += ll; pos += ll;
        if (!ml) break;
        lo = off;
        if (ll >= 15) { for (v = ll - 15; v >= 255; v -= 255) lens[nlen++] = 255; lens[nlen++] = (uint8_t)v; }
        if (ml >= 18) { for (v = ml - 18; v >= 255; v -= 255) lens[nlen++] = 255; lens[nlen++] = (uint8_t)v; }
        toks[ns] = (uint8_t)zap__k_tok(ll, ml);
        int slot = zap__rep_slot(off, rep);
        if (slot >= 0) offc[ns] = (uint8_t)slot;
        else {
            unsigned c = zap__k_ocls(&sc, off, &v);
            offc[ns] = (uint8_t)c;
            if (c < 254) nlo[nn++] = (uint8_t)v;
            else { flo[nf] = (uint8_t)v; fmi[nf] = (uint8_t)(v >> 8); fhi[nf] = (uint8_t)(v >> 16); nf++; }
        }
        zap__rep_push(rep, off);
        ns++; pos += ml;
    }
    cb[nch] = nl;
    uint8_t *op = dst, *oend = dst + cap;
    zap__w32(op, (uint32_t)nl | 0xC0000000u); zap__w32(op + 4, (uint32_t)ns); zap__w32(op + 8, (uint32_t)nlen);
    zap__w32(op + 12, (uint32_t)nn); zap__w32(op + 16, (uint32_t)nf); zap__w32(op + 20, (uint32_t)nch | (uint32_t)S << 24);
    op += 24;
    uint8_t *ls = op + 4 * nch;
    op = zap__k_lits(ls + 5, oend, lits, dls, cb, nch, rawpct, dfl);
    if (op) {
        ls[0] = 0x0F; zap__w32(ls + 1, (uint32_t)(op - ls - 5));
        for (size_t c = 0; c < nch; c++) zap__w32(ls - 4 * nch + 4 * c, (uint32_t)(cb[c + 1] - cb[c]) | (uint32_t)dfl[c] << 31);
    }
    zap__grp_tok(gtok);
    op = zap__k_stream(op, oend, toks, ns, gtok, 11);
    op = zap__k_stream(op, oend, lens, nlen, NULL, 11);
    op = zap__k_stream(op, oend, offc, ns, NULL, 11);
    op = zap__k_stream(op, oend, nlo, nn, NULL, 11);
    op = zap__k_stream(op, oend, flo, nf, NULL, 11);
    op = zap__k_stream(op, oend, fmi, nf, NULL, 11);
    op = zap__k_stream(op, oend, fhi, nf, NULL, 11);
    free(buf); free(cb);
    return op ? (size_t)(op - dst) : 0;
}

/* ---- the parse: prices (1/16 bit) as the block codes them */
typedef struct {
    uint32_t tok[16][256], lit[16][256], dlit[16][256]; /* by group of the previous token / previous literal or delta */
    uint32_t oc[256], nlo[256], flo[256], fmi[256], fhi[256], lenb, seq, pfar, farat;
    zap__ksc sc;
    uint8_t *dreg; /* per 128 KB region: its literals priced as deltas (the encoder's likely chunk mode) */
    size_t nreg;
} zap__kc;
/* Huffman-coded bits of a histogram (the delta / literal choice per region) */
static inline uint64_t zap__k_hbits(const uint32_t *h) {
    uint8_t len[256];
    uint64_t b = 0;
    zap__hlens(h, len);
    for (int i = 0; i < 256; i++) b += (uint64_t)h[i] * len[i];
    return b;
}
static inline void zap__k_costs(const uint8_t *src, const zap__kseq *q, size_t nq, unsigned S, zap__kc *c, uint32_t seq) {
    uint32_t (*f)[256] = (uint32_t(*)[256])calloc(54, sizeof *f); /* 16 tok, 16 lit, oc, nlo, flo, fmi, fhi, -, 16 delta */
    uint8_t len[256], gt[256];
    size_t n = 0;
    for (size_t i = 0; i < nq; i++) n += q[i].ll + q[i].ml;
    uint32_t (*h)[2][256] = (uint32_t(*)[2][256])calloc(n / ZAP__K_CHUNK + 1, sizeof *h);
    uint8_t *dreg = (uint8_t *)realloc(c->dreg, n / ZAP__K_CHUNK + 1);
    if (!f || !h || !dreg) { free(f); free(h); return; }
    c->dreg = dreg; c->nreg = n / ZAP__K_CHUNK + 1;
    zap__grp_tok(gt);
    zap__ksc_set(&c->sc, S);
    uint32_t rep[3] = { 0, 0, 0 };
    unsigned pg = 0, pl = 0;
    for (size_t i = 0, pos = 0, lo = 0; i < nq; pos += q[i].ll + q[i].ml, lo = q[i].off, i++) /* each region: literals or deltas, whichever is smaller */
        for (size_t k = 0; k < q[i].ll; k++) { size_t p = pos + k; h[p / ZAP__K_CHUNK][0][src[p]]++; h[p / ZAP__K_CHUNK][1][(uint8_t)(src[p] - (lo ? src[p - lo] : 0))]++; }
    for (size_t r = 0; r < c->nreg; r++) { uint64_t br = zap__k_hbits(h[r][0]); dreg[r] = zap__k_hbits(h[r][1]) + br / 80 < br; } /* deltas 1.2% smaller */
    free(h);
    for (size_t i = 0, pos = 0, lo = 0; i < nq; i++) {
        for (size_t k = 0; k < q[i].ll; k++) {
            size_t p = pos + k;
            unsigned v = dreg[p / ZAP__K_CHUNK] ? (uint8_t)(src[p] - (lo ? src[p - lo] : 0)) : src[p];
            f[dreg[p / ZAP__K_CHUNK] ? 38 + pl : 16 + pl][v]++; pl = v >> 4;
        }
        pos += q[i].ll;
        if (!q[i].ml) break;
        if (!q[i].off) { pos += q[i].ml; rep[0] = rep[1] = rep[2] = 0; lo = 0; continue; } /* a gap */
        lo = q[i].off;
        size_t off = q[i].off, v;
        unsigned tok = zap__k_tok(q[i].ll, q[i].ml);
        f[pg][tok]++; pg = gt[tok];
        int slot = zap__rep_slot(off, rep);
        if (slot >= 0) f[32][slot]++;
        else {
            unsigned s = zap__k_ocls(&c->sc, off, &v);
            f[32][s]++;
            if (s < 254) f[33][v & 255]++; else { f[34][v & 255]++; f[35][(v >> 8) & 255]++; f[36][(v >> 16) & 255]++; }
        }
        zap__rep_push(rep, off);
        pos += q[i].ml;
    }
#define ZAP__KL(fr, to) do { zap__hlens(fr, len); for (int i_ = 0; i_ < 256; i_++) (to)[i_] = 16u * (len[i_] ? len[i_] : ZAP__HMAX + 2); } while (0)
    for (int g = 0; g < 16; g++) { ZAP__KL(f[g], c->tok[g]); ZAP__KL(f[16 + g], c->lit[g]); ZAP__KL(f[38 + g], c->dlit[g]); }
    ZAP__KL(f[32], c->oc); ZAP__KL(f[33], c->nlo); ZAP__KL(f[34], c->flo); ZAP__KL(f[35], c->fmi); ZAP__KL(f[36], c->fhi);
#undef ZAP__KL
    c->lenb = 16 * 7; c->seq = seq; c->pfar = 32; c->farat = 256u << 10; /* ~2 bits against far (cache-missing) sources */
    free(f);
}
/* the offset's part of a match price (repeat slot or new offset), and the length / token part */
static inline uint32_t zap__k_oprice(const zap__kc *c, size_t off, const uint32_t rep[3]) {
    int slot = zap__rep_slot(off, rep);
    uint32_t far = off >= c->farat ? c->pfar : 0u;
    if (slot >= 0) return far + c->oc[slot];
    if (off >= ((size_t)1 << 24)) return 0x3FFFFFFFu;
    size_t v;
    unsigned s = zap__k_ocls(&c->sc, off, &v);
    return far + c->oc[s] + (s < 254 ? c->nlo[v & 255] : c->flo[v & 255] + c->fmi[(v >> 8) & 255] + c->fhi[v >> 16]);
}
static inline uint32_t zap__k_lprice(const zap__kc *c, size_t len, size_t lit, unsigned pg) {
    return c->seq + zap__ext_bytes(len - 3) * c->lenb + c->tok[pg][zap__k_tok(lit, len)];
}

/* multi-arrival optimal parse: each position keeps up to K arrivals (the cheapest per distinct repeat-offset state),
   so a slightly dearer path whose repeat offsets enable a long repeat match later survives. Arrivals also carry
   the previous token's and previous literal's groups for the contextual prices. */
typedef struct { uint32_t price, off, lit, rep[3]; uint16_t len; uint8_t from, pg, pl, pad_[3]; } zap__arr; /* 32 bytes */
/* w: the position's worst kept price once it holds K arrivals (else ~0), for the callers' early exits */
static inline void zap__ma_insert(zap__arr *a, uint8_t *na, uint32_t *w, int K, const zap__arr *x) {
    int n = *na;
    if (K == 1) { if (!n || x->price < a[0].price) { a[0] = *x; *na = 1; *w = x->price; } return; }
    for (int i = 0; i < n; i++)
        if (a[i].rep[0] == x->rep[0] && a[i].rep[1] == x->rep[1] && a[i].rep[2] == x->rep[2]) {
            if (x->price >= a[i].price) return;
            for (int j = i; j + 1 < n; j++) a[j] = a[j + 1];
            n--;
            break;
        }
    if (n == K && x->price >= a[K - 1].price) { *na = (uint8_t)n; return; }
    int j = n < K ? n : K - 1;
    while (j > 0 && a[j - 1].price > x->price) { a[j] = a[j - 1]; j--; }
    a[j] = *x;
    if (n < K) n++;
    *na = (uint8_t)n;
    *w = n == K ? a[K - 1].price : 0xFFFFFFFFu;
}
#define ZAP__K_H3LIM (1u << 18) /* 3-byte matches: offsets below this (farther ones don't pay for their offset) */
#define ZAP__K_H3LOG 20         /* their hash: 4 MB; 2^16 buckets lost 0.09% to collisions */
#if defined(__GNUC__) || defined(__clang__)
#define ZAP__PREFETCH(p) __builtin_prefetch(p)
#elif defined(_MSC_VER) && ZAP__X86
#define ZAP__PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define ZAP__PREFETCH(p) ((void)(p))
#endif
/* The block's matches, found once for every pass: per position the binary tree's list (strictly increasing lengths),
   led by the nearest earlier 3 bytes when there are some (a 3-byte hash; a collision just misses one). Entries are
   offset << 8 | (length - 3); ~12 bytes per input byte. Searching every position in order also finds matches where
   a per-pass search couldn't: positions a window revisits, and those a long match skipped. */
typedef struct { uint32_t *idx, *m; } zap__kmt;
static inline void zap__kmt_free(zap__kmt *t) { free(t->idx); free(t->m); t->idx = t->m = NULL; }
#define ZAP__K_HLOG 20 /* the table's tree roots: 4 MB of heads, fewer 4-byte values per tree than the hc state's (-25% time) */
static inline int zap__kmt_build(zap__kmt *t, const uint8_t *src, size_t n, zap_hc_state *s, int depth) {
    const uint8_t *iend = src + n;
    size_t limit = n >= 13 ? n - 12 : 0, cap = limit + limit / 2 + 64, cnt = 0;
    uint32_t *h3 = (uint32_t *)malloc(sizeof(uint32_t) << ZAP__K_H3LOG), *heads = (uint32_t *)malloc(sizeof(uint32_t) << ZAP__K_HLOG);
    zap__match mm[ZAP__MAXC];
    t->idx = (uint32_t *)malloc(sizeof(uint32_t) * (limit + 1));
    t->m = (uint32_t *)malloc(sizeof(uint32_t) * cap);
    if (!h3 || !heads || !t->idx || !t->m) { free(h3); free(heads); zap__kmt_free(t); return -1; }
    memset(h3, 0xFF, sizeof(uint32_t) << ZAP__K_H3LOG);
    memset(heads, 0xFF, sizeof(uint32_t) << ZAP__K_HLOG);
    for (size_t pos = 0; pos < limit; pos++) {
        uint32_t v = zap__r32(src + pos) & 0xFFFFFF, h = (v * 2654435761u) >> (32 - ZAP__K_H3LOG), c = h3[h];
        if (pos + 16 < limit) { /* the tree walks are chains of cache misses: fetch the root 8 positions ahead */
            uint32_t r = heads[zap__hash(zap__r32(src + pos + 8), ZAP__K_HLOG)];
            ZAP__PREFETCH(&heads[zap__hash(zap__r32(src + pos + 16), ZAP__K_HLOG)]);
            if (r < pos) { ZAP__PREFETCH(s->prev + 2 * (r & (sizeof s->prev / sizeof s->prev[0] / 2 - 1))); ZAP__PREFETCH(src + r); }
        }
        int nm = zap__bt(s, heads, ZAP__K_HLOG, src, pos, iend, depth, 1, 3, mm, 0);
        h3[h] = (uint32_t)pos;
        t->idx[pos] = (uint32_t)cnt;
        if (cnt + (size_t)nm + 1 > cap) {
            uint32_t *g = (uint32_t *)realloc(t->m, sizeof(uint32_t) * (cap += cap / 2));
            if (!g) { free(h3); free(heads); zap__kmt_free(t); return -1; }
            t->m = g;
        }
        if (c < pos && pos - c < ZAP__K_H3LIM && (zap__r32(src + c) & 0xFFFFFF) == v) t->m[cnt++] = (uint32_t)(pos - c) << 8;
        for (int i = 0; i < nm; i++) t->m[cnt++] = mm[i].off << 8 | (mm[i].len - 3);
    }
    t->idx[limit] = (uint32_t)cnt;
    free(h3); free(heads);
    return 0;
}
/* the first parse, only for the first pass's statistics: lazy over the table (the longest match of 4+ bytes, unless
   the next position has a longer one) */
static inline size_t zap__k_lazy(size_t n, const zap__kmt *t, zap__kseq *out) {
    size_t limit = n >= 13 ? n - 12 : 0, nq = 0, anchor = 0, pos = 0;
    while (pos < limit) {
        uint32_t b = t->idx[pos + 1];
        size_t len = b > t->idx[pos] ? (t->m[b - 1] & 255) + 3 : 0;
        if (len < 4 || (pos + 1 < limit && t->idx[pos + 2] > b && (t->m[t->idx[pos + 2] - 1] & 255) + 3 > len + 1)) { pos++; continue; }
        out[nq].ll = (uint32_t)(pos - anchor); out[nq].ml = (uint32_t)len; out[nq].off = t->m[b - 1] >> 8; nq++;
        pos += len; anchor = pos;
    }
    out[nq].ll = (uint32_t)(n - anchor); out[nq].ml = 0; out[nq].off = 0;
    return nq + 1;
}
/* the parse, as sequences into out (room for n / 3 + 2); returns the count, 0 = out of memory */
static inline size_t zap__compress_ma(const uint8_t *src, size_t n, size_t base, zap__kseq *out, const zap__kmt *mt, const zap__kc *cm, int K) {
    const uint8_t *iend = src + n;
    uint8_t gtk[256];
    size_t slots = ZAP__OPTN + ZAP__SUFF + 2, anchor = 0, start = 0, limit = n >= 13 ? n - 12 : 0, nq = 0;
    zap__arr *arr = (zap__arr *)malloc(sizeof(zap__arr) * slots * (size_t)K);
    uint8_t *na = (uint8_t *)malloc(slots);
    uint32_t *wp = (uint32_t *)malloc(sizeof(uint32_t) * slots); /* worst kept price per position (early exits) */
    size_t *seqs = (size_t *)malloc(sizeof(size_t) * 3 * (slots + 1));
    zap__match m[ZAP__MAXC + 1];
    uint32_t reps[3] = { 0, 0, 0 }, lastpg = 0;
    zap__grp_tok(gtk);
    if (!arr || !na || !wp || !seqs) { free(arr); free(na); free(wp); free(seqs); return 0; }
#define ZAP__A(p) (arr + (size_t)(p) * (size_t)K)
#define ZAP__G(l, L) gtk[zap__k_tok(l, L)]
    while (start < limit) {
        size_t last = 0, p, fl = 0, fo = 0;
        zap__arr a0 = { 0, 0, (uint32_t)(start - anchor), { reps[0], reps[1], reps[2] }, 0, 0, (uint8_t)lastpg, (uint8_t)(start ? src[start - 1] >> 4 : 0), { 0, 0, 0 } };
        na[0] = 1; ZAP__A(0)[0] = a0; wp[0] = K == 1 ? 0 : 0xFFFFFFFFu;
        for (p = 0; p <= last && p < ZAP__OPTN && start + p < limit; p++) {
            if (!na[p]) continue;
            size_t pos = start + p;
            const uint32_t *me = mt->m + mt->idx[pos];
            int nm = (int)(mt->idx[pos + 1] - mt->idx[pos]); /* every match: arrivals differ in repeats */
            for (int i = 0; i < nm; i++) { m[i].len = (me[i] & 255) + 3; m[i].off = me[i] >> 8; if (m[i].len > n - pos) m[i].len = (uint32_t)(n - pos); } /* a table built for a longer span */
            if (nm && m[nm - 1].len >= ZAP__SUFF) { /* long match: just take it, all of it (the table stops at 256) */
                fo = m[nm - 1].off; fl = m[nm - 1].len + zap__count(src + pos + m[nm - 1].len, src + pos - fo + m[nm - 1].len, iend);
                break;
            }
            int cnt = na[p];
            for (int ai = 0; ai < cnt; ai++) {
                zap__arr o = ZAP__A(p)[ai];
                size_t lt = (o.lit < 15 ? o.lit : 15) << 4;
                const uint32_t *tr = cm->tok[o.pg] + lt; /* this arrival's token prices by match length, and groups */
                const uint8_t *gr = gtk + lt;
#define ZAP__LP(L) (cm->seq + ((L) < 18 ? tr[(L) - 3] : tr[15] + zap__ext_bytes((L) - 3) * cm->lenb))
#define ZAP__LG(L) gr[(L) < 18 ? (L) - 3 : 15]
                if (p + 1 > last) { na[p + 1] = 0; wp[p + 1] = 0xFFFFFFFFu; last = p + 1; }
                zap__arr x = o;
                size_t rg = (base + pos) / ZAP__K_CHUNK;
                int dr = rg < cm->nreg && cm->dreg[rg]; /* a delta region: the literal minus the byte at the last match offset */
                unsigned lv = dr ? (uint8_t)(src[pos] - (o.rep[0] ? *(src + pos - o.rep[0]) : 0)) : src[pos]; /* the reference may precede a sampled view */
                x.price = o.price + (dr ? cm->dlit : cm->lit)[o.pl][lv] + (zap__ext_bytes(o.lit + 1) - zap__ext_bytes(o.lit)) * cm->lenb;
                x.len = 0; x.off = 0; x.lit = o.lit + 1; x.from = (uint8_t)ai; x.pl = (uint8_t)(lv >> 4);
                zap__ma_insert(ZAP__A(p + 1), &na[p + 1], &wp[p + 1], K, &x);
                for (int r = 0; r < 3; r++) { /* this arrival's repeat offsets */
                    size_t ro = o.rep[r];
                    if (!ro || ro > pos || (r >= 1 && ro == o.rep[0]) || (r == 2 && ro == o.rep[1]) || ((zap__r32(src + pos - ro) ^ zap__r32(src + pos)) & 0xFFFFFF)) continue;
                    size_t rl = 3 + zap__count(src + pos + 3, src + pos - ro + 3, iend);
                    if (rl >= ZAP__SUFF) { if (!ai) { fl = rl; fo = ro; goto window_end; } rl = ZAP__SUFF - 1; } /* the cheapest path's long repeat: take it */
                    uint32_t nr[3] = { o.rep[0], o.rep[1], o.rep[2] }, op_ = o.price + zap__k_oprice(cm, ro, o.rep);
                    zap__rep_push(nr, ro);
                    while (last < p + rl) { na[++last] = 0; wp[last] = 0xFFFFFFFFu; }
                    for (size_t L = 3; L <= rl; L++) {
                        uint32_t rp = op_ + ZAP__LP(L);
                        if (rp >= wp[p + L]) continue; /* can't place (the insert would say so too) */
                        zap__arr y = { rp, (uint32_t)ro, 0, { nr[0], nr[1], nr[2] }, (uint16_t)L, (uint8_t)ai, ZAP__LG(L), o.pl, { 0, 0, 0 } };
                        zap__ma_insert(ZAP__A(p + L), &na[p + L], &wp[p + L], K, &y);
                    }
                }
                for (int i = 0; i < nm; i++) { /* a 3-byte match first, then the tree's from 4 */
                    size_t lo = i ? m[i - 1].len + 1 : m[0].len < 4 ? 3 : 4;
                    uint32_t nr[3] = { o.rep[0], o.rep[1], o.rep[2] }, op_ = o.price + zap__k_oprice(cm, m[i].off, o.rep);
                    zap__rep_push(nr, m[i].off);
                    while (last < p + m[i].len) { na[++last] = 0; wp[last] = 0xFFFFFFFFu; }
                    for (size_t L = ai ? m[i].len : lo; L <= m[i].len; L++) { /* dearer arrivals (kept for their repeats): whole matches only, -12% time, ratio +-0.05% */
                        uint32_t mp = op_ + ZAP__LP(L);
                        if (mp >= wp[p + L]) continue;
                        zap__arr y = { mp, m[i].off, 0, { nr[0], nr[1], nr[2] }, (uint16_t)L, (uint8_t)ai, ZAP__LG(L), o.pl, { 0, 0, 0 } };
                        zap__ma_insert(ZAP__A(p + L), &na[p + L], &wp[p + L], K, &y);
                    }
                }
#undef ZAP__LP
#undef ZAP__LG
            }
        }
    window_end:;
        size_t end = fl ? p : 0;
        if (!fl) for (size_t e = p < last ? p : last; e > 0; e--) if (na[e] && ZAP__A(e)[0].len) { end = e; break; }
        if (!end && !fl) { start += p ? p : 1; continue; } /* nothing matched: the literals carry into the next window */
        size_t cnt = 0, t = end, ai = 0; /* backtrack from the cheapest arrival at end */
        while (t > 0) {
            zap__arr *x = &ZAP__A(t)[ai];
            if (x->len) { seqs[3 * cnt] = start + t - x->len; seqs[3 * cnt + 1] = x->len; seqs[3 * cnt + 2] = x->off; cnt++; ai = x->from; t -= x->len; }
            else { ai = x->from; t--; }
        }
        for (size_t i = 0, j = cnt ? cnt - 1 : 0; i < j; i++, j--) for (int q = 0; q < 3; q++) { size_t tmp = seqs[3 * i + q]; seqs[3 * i + q] = seqs[3 * j + q]; seqs[3 * j + q] = tmp; }
        if (fl) { seqs[3 * cnt] = start + p; seqs[3 * cnt + 1] = fl; seqs[3 * cnt + 2] = fo; cnt++; }
        for (size_t k = 0; k < cnt; k++) {
            out[nq].ll = (uint32_t)(seqs[3 * k] - anchor); out[nq].ml = (uint32_t)seqs[3 * k + 1]; out[nq].off = (uint32_t)seqs[3 * k + 2]; nq++;
            lastpg = ZAP__G(seqs[3 * k] - anchor, seqs[3 * k + 1]);
            anchor = seqs[3 * k] + seqs[3 * k + 1];
            zap__rep_push(reps, seqs[3 * k + 2]);
        }
        start = anchor;
    }
#undef ZAP__A
#undef ZAP__G
    out[nq].ll = (uint32_t)(n - anchor); out[nq].ml = 0; out[nq].off = 0; nq++;
    free(arr); free(na); free(wp); free(seqs);
    return nq;
}

/* ---- decoder: offsets, then copies, per batch of ZAP__K_N sequences */
/* offset symbol -> (h << 8) | multiplier << 16 for new offsets (symbols >= 254 are far); 0 = invalid (offset 0) */
static inline void zap__k_otab(uint32_t T[256], unsigned S) {
    memset(T, 0, 256 * sizeof *T);
    if (S == 1) { for (unsigned h = 0; h <= 250; h++) T[3 + h] = h << 8 | 1u << 16; }
    else { for (unsigned h = 0; h < 125; h++) { T[3 + h] = h << 8 | S << 16; T[128 + h] = h << 8 | 1u << 16; } T[254] = S << 16; }
    T[255] = 1u << 16;
}
typedef struct { const uint8_t *pa; const uint32_t *pb; size_t r0, r1, r2; } zap__ko;
ZAP__NOINLINE static void zap__k_offs(zap__ko *s, const uint8_t *cb, const uint32_t *T, uint32_t *O, size_t cnt) {
    const uint8_t *pa = s->pa;
    const uint32_t *pb = s->pb;
    size_t r0 = s->r0, r1 = s->r1, r2 = s->r2;
    for (size_t j = 0; j < cnt; j++) {
        unsigned c = cb[j];
        if (c < 3) { size_t o = c == 0 ? r0 : c == 1 ? r1 : r2; r2 = c <= 1 ? r2 : r1; r1 = c == 0 ? r1 : r0; r0 = o; } /* repeats: ~15% */
        else {
            size_t t = T[c], isf = c >= 254, nv = (t & 0xFFFF) | *pa, fv = *pb;
            r2 = r1; r1 = r0; r0 = (isf ? fv : nv) * (t >> 16); pa += isf ^ 1; pb += isf;
        }
        O[j] = (uint32_t)r0;
    }
    s->pa = pa; s->pb = pb; s->r0 = r0; s->r1 = r1; s->r2 = r2;
}
/* match with offset 1..15: a pattern of the period (two 16-byte stores via pshufb), then copies at a multiple >= 16;
   the caller guarantees ml + 32 bytes of room. Without SSSE3: the byte-priming copy. */
#if ZAP__PSHUFB || ZAP__X86
#if !ZAP__PSHUFB && !(defined(_MSC_VER) && !defined(__clang__))
#include <tmmintrin.h> /* clang for Windows leaves it out of <immintrin.h> without -mssse3 */
#define ZAP__S3ATTR __attribute__((target("ssse3")))
#else
#define ZAP__S3ATTR
#endif
static const uint8_t zap__k_pat[16][33] = {
#define ZAP__PR(o) { 0%o,1%o,2%o,3%o,4%o,5%o,6%o,7%o,8%o,9%o,10%o,11%o,12%o,13%o,14%o,15%o, 16%o,17%o,18%o,19%o,20%o,21%o,22%o,23%o,24%o,25%o,26%o,27%o,28%o,29%o,30%o,31%o, (o)*((16+(o)-1)/(o)) }
    { 0 }, ZAP__PR(1), ZAP__PR(2), ZAP__PR(3), ZAP__PR(4), ZAP__PR(5), ZAP__PR(6), ZAP__PR(7), ZAP__PR(8), ZAP__PR(9), ZAP__PR(10), ZAP__PR(11), ZAP__PR(12), ZAP__PR(13), ZAP__PR(14), ZAP__PR(15)
#undef ZAP__PR
};
ZAP__S3ATTR static inline void zap__k_short3(uint8_t *op, size_t off, size_t ml) {
    const uint8_t *t = zap__k_pat[off];
    __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(op - off));
    _mm_storeu_si128((__m128i *)(void *)op, _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i *)(const void *)t)));
    _mm_storeu_si128((__m128i *)(void *)(op + 16), _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i *)(const void *)(t + 16))));
    if (ml > 32) { size_t p = t[32]; for (uint8_t *d = op + 32, *e = op + ml; d < e; d += 16) zap__cp16(d, d - p); }
}
#endif
/* delta literals: out = sym + the byte r back (r: the last match's offset, 0 before the first). k_add16 writes 16
   bytes for ll <= 15 (the tail is overwritten after); byte lanes add without carries (SWAR), 8 at a time for r >= 8 */
static inline uint64_t zap__k_ld8(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void zap__k_st8(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static inline uint64_t zap__k_add8(uint64_t a, uint64_t b) { return ((a & 0x7F7F7F7F7F7F7F7Full) + (b & 0x7F7F7F7F7F7F7F7Full)) ^ ((a ^ b) & 0x8080808080808080ull); }
static inline void zap__k_addn(uint8_t *op, const uint8_t *lp, size_t ll, size_t r) {
    size_t t = 0;
    if (!r) { memcpy(op, lp, ll); return; }
    if (r >= 8) for (; t + 8 <= ll; t += 8) zap__k_st8(op + t, zap__k_add8(zap__k_ld8(lp + t), zap__k_ld8(op + t - r)));
    for (const uint8_t *m = op - r; t < ll; t++) op[t] = (uint8_t)(lp[t] + m[t]);
}
#if ZAP__X86
#include <emmintrin.h>
#endif
static inline void zap__k_add16(uint8_t *op, const uint8_t *lp, size_t ll, size_t r) {
#if ZAP__X86
    if (r >= 16) { _mm_storeu_si128((__m128i *)(void *)op, _mm_add_epi8(_mm_loadu_si128((const __m128i *)(const void *)lp), _mm_loadu_si128((const __m128i *)(const void *)(op - r)))); return; }
#endif
    if (r >= 8) { zap__k_st8(op, zap__k_add8(zap__k_ld8(lp), zap__k_ld8(op - r))); zap__k_st8(op + 8, zap__k_add8(zap__k_ld8(lp + 8), zap__k_ld8(op + 8 - r))); }
    else if (!r) zap__cp16(op, lp);
    else for (size_t t = 0; t < ll; t++) op[t] = (uint8_t)(lp[t] + (op - r)[t]);
}
static inline void zap__k_short(uint8_t *op, size_t off, size_t ml, uint8_t *oend, int s3) {
#if ZAP__PSHUFB
    (void)oend; (void)s3;
    zap__k_short3(op, off, ml);
#elif ZAP__X86
    if (s3) zap__k_short3(op, off, ml); else zap__match_copy(op, off, ml, oend);
#else
    (void)s3;
    zap__match_copy(op, off, ml, oend);
#endif
}
/* fast sequences: each reads <= 16 literals, writes <= 16 + 32 bytes (matches <= 17); D: the chunk's literals are
   deltas from the previous match's offset (Ob[j] = O[j - 1]) */
#define ZAP__KFAST(P, D)                                                                                                   \
            for (; j < lim; j++) {                                                                                         \
                unsigned tok = tb[j];                                                                                      \
                size_t ll = tok >> 4, ml = tok & 15, off = O[j];                                                           \
                if (ll == 15) goto P##slow;                                                                                \
                if (D) zap__k_add16(op, lp, ll, Ob[j]); else zap__cp16(op, lp);                                          \
                op += ll; lp += ll;                                                                                        \
                if (ml == 15) goto P##long_match;                                                                          \
                if (off < 16 || off > (size_t)(op - dst)) goto P##slow_match;                                              \
                zap__cp16(op, op - off); zap__cp16(op + 16, op - off + 16);                                                \
                op += ml + 3;                                                                                              \
                continue;                                                                                                  \
            P##slow_match:                                                                                                 \
                if (off - 1 >= (size_t)(op - dst)) return -1;                                                              \
                zap__k_short(op, off, ml + 3, oend, s3);                                                                   \
                op += ml + 3;                                                                                              \
                continue;                                                                                                  \
            P##long_match: /* 18..272 bytes from 16+ back, room for the overcopy: 16-byte steps, the batch trimmed */    \
                if (lnp < lne && *lnp < 255 && off >= 16 && off <= (size_t)(op - dst) && (size_t)(oend - op) >= 34 + (size_t)*lnp) { \
                    uint8_t *d = op, *e = op + 18 + *lnp++;                                                                \
                    do { zap__cp16(d, d - off); d += 16; } while (d < e);                                                  \
                    op = e;                                                                                                \
                    if (lim - j - 1 > (size_t)(oend - op) >> 6) lim = j + 1 + ((size_t)(oend - op) >> 6); /* rest fits */ \
                    continue;                                                                                              \
                }                                                                                                          \
                goto P##slow_len;                                                                                          \
            P##slow:                                                                                                       \
                ll += zap__ext(&lnp, lne, &err);                                                                           \
                if (err || ll > (size_t)(le - lp) || ll > (size_t)(oend - op)) return -1;                                  \
                if (D) zap__k_addn(op, lp, ll, Ob[j]); else memcpy(op, lp, ll);                                          \
                op += ll; lp += ll;                                                                                        \
            P##slow_len:                                                                                                   \
                if (ml == 15) { ml += zap__ext(&lnp, lne, &err); if (err) return -1; }                                     \
                ml += 3;                                                                                                   \
                if (off - 1 >= (size_t)(op - dst) || ml > (size_t)(oend - op)) return -1;                                  \
                if (off < 16 && (size_t)(oend - op) >= ml + 32) zap__k_short(op, off, ml, oend, s3);                       \
                else zap__match_copy(op, off, ml, oend);                                                                   \
                op += ml;                                                                                                  \
                j++;                                                                                                       \
                break;                                                                                                     \
            }
static inline ptrdiff_t zap__k_lz(const uint8_t *lits, size_t nl, const uint8_t *chunks, size_t nch, const uint8_t *tp, size_t ns, const uint8_t *lnp,
                                  const uint8_t *lne, const uint8_t *oc, const uint32_t *T, const uint8_t *NL, size_t nn, const uint32_t *FAR, size_t nf,
                                  uint8_t *dst, size_t raw) {
    uint32_t Ob[ZAP__K_N + 1], *O = Ob + 1; /* Ob[0]: the previous batch's last offset (delta literals' reference) */
    zap__ko s = { NL, FAR, 0, 0, 0 };
    const uint8_t *lp = lits, *le = lits, *lend = lits + nl; /* le: the current chunk's literals end */
    uint8_t *op = dst, *oend = dst + raw;
    size_t ci = 0; /* next chunk */
    int err = 0, s3 = zap__k_cpu() >> 1 & 1, dl = 0;
    Ob[0] = 0;
#define ZAP__KCHUNK() while (lp == le && ci < nch) { size_t w_ = zap__r32(chunks + 4 * ci++); le += w_ & 0x7FFFFFFFu; dl = (int)(w_ >> 31); }
    for (size_t i = 0; i < ns;) {
        size_t cnt = ns - i < ZAP__K_N ? ns - i : ZAP__K_N, j;
        if (s.pa > NL + nn || s.pb > FAR + nf) return -1; /* a batch reads at most ZAP__K_N entries past these (padding) */
        const uint8_t *tb = tp + i;
        zap__k_offs(&s, oc + i, T, O, cnt);
        for (j = 0; j < cnt;) {
            size_t kl = (size_t)(le - lp) >> 4, ko = (size_t)(oend - op) >> 6, k = kl < ko ? kl : ko, lim = cnt - j < k ? cnt : j + k;
            if (dl) { ZAP__KFAST(d_, 1) }
            else { ZAP__KFAST(r_, 0) }
            if (j < cnt && k == 0) { /* near a chunk's end, or the output's: checked copies */
                unsigned tok = tb[j];
                size_t ll = tok >> 4, ml = tok & 15, off = O[j];
                if (ll == 15) { ll += zap__ext(&lnp, lne, &err); if (err) return -1; }
                if (ll) ZAP__KCHUNK();
                if (ll > (size_t)(le - lp) || ll > (size_t)(oend - op)) return -1;
                if (dl) zap__k_addn(op, lp, ll, Ob[j]); else memcpy(op, lp, ll);
                op += ll; lp += ll;
                if (ml == 15) { ml += zap__ext(&lnp, lne, &err); if (err) return -1; }
                ml += 3;
                if (off - 1 >= (size_t)(op - dst) || ml > (size_t)(oend - op)) return -1;
                zap__match_copy(op, off, ml, oend);
                op += ml;
                j++;
            }
        }
        Ob[0] = O[cnt - 1];
        i += cnt;
    }
    if (s.pa != NL + nn || s.pb != FAR + nf || lnp != lne) return -1;
    ZAP__KCHUNK(); /* the trailing literals: one chunk's */
    if (le != lend || ci != nch || (size_t)(le - lp) != (size_t)(oend - op)) return -1;
    if (dl) zap__k_addn(op, lp, (size_t)(le - lp), Ob[0]); else memcpy(op, lp, (size_t)(le - lp));
    return (ptrdiff_t)raw;
#undef ZAP__KCHUNK
}
#undef ZAP__KFAST

static inline ptrdiff_t zap__k_decompress(const uint8_t *ip, size_t n, uint8_t *dst, size_t raw_size, void *scratch, size_t scratch_cap) {
    const uint8_t *iend = ip + n;
    if (n < 24) return -1;
    size_t nl = zap__r32(ip) & 0x3FFFFFFFu, ns = zap__r32(ip + 4), nlen = zap__r32(ip + 8), nn = zap__r32(ip + 12), nf = zap__r32(ip + 16),
           nch = zap__r32(ip + 20) & 0xFFFFFF, S = ip[23];
    ip += 24;
    if (nl > raw_size || ns > raw_size / 3 + 1 || nlen > 2 * ns + raw_size / 255 + 1 || nn > ns || nf > ns - nn || !nch || nch > raw_size / 16 + 1 ||
        !S || (size_t)(iend - ip) < 4 * (size_t)nch) return -1;
    const uint8_t *chunks = ip;
    ip += 4 * nch;
    size_t pad = ZAP__K_N + 16, cnt[8] = { nl, ns, nlen, ns, nn, nf, nf, nf }, bytes = nl + 2 * ns + nlen + nn + 3 * nf + 2 * pad + 64;
    size_t need = bytes + 4 * (nf + pad) + sizeof(uint16_t) * (2u << 15) + 64;
    uint8_t gtok[256], *mem = (uint8_t *)scratch, *own = NULL;
    zap__grp_tok(gtok);
    if (!mem || scratch_cap < need) { if (!(mem = own = (uint8_t *)malloc(need))) return -1; }
    const uint8_t *st[8] = { NULL };
    uint8_t *w = mem;
    uint32_t *FAR = (uint32_t *)(void *)(mem + ((bytes + 15) & ~(size_t)15));
    uint8_t *tb = (uint8_t *)(FAR + nf + pad);
    uint16_t *tab = (uint16_t *)(void *)(tb + ((16 - ((uintptr_t)tb & 15)) & 15)), *tab2 = tab + (1u << 15);
    ptrdiff_t r = -1;
    for (int k = 0; k < 8; k++) {
        if (iend - ip < 5) goto out;
        int m = ip[0] & 15, L = ip[0] >> 4;
        size_t sz = zap__r32(ip + 1);
        ip += 5;
        if (sz > (size_t)(iend - ip)) goto out;
        if (k == 0) { if (ip[-5] != 0x0F || zap__k_dlits(ip, sz, chunks, nch, w, nl, tab, tab2)) goto out; st[0] = w; w += nl; }
        else if (m == 0) {
            if (L || sz != cnt[k]) goto out;
            if (k == 4) { memcpy(w, ip, sz); st[k] = w; w += cnt[k] + pad; } else st[k] = ip; /* near low bytes are read past their end */
        } else {
            if (zap__k_dstream(m, L, ip, sz, w, cnt[k], k == 1 ? gtok : NULL, tab)) goto out;
            st[k] = w; w += cnt[k] + (k == 4 ? pad : 0);
        }
        if (k == 4) memset((uint8_t *)st[4] + nn, 0, pad);
        ip += sz;
    }
    if (ip != iend) goto out;
    for (size_t j = 0; j < nf; j++) FAR[j] = (uint32_t)st[5][j] | (uint32_t)st[6][j] << 8 | (uint32_t)st[7][j] << 16;
    memset(FAR + nf, 0, 4 * pad);
    uint32_t T[256];
    zap__k_otab(T, (unsigned)S);
    r = zap__k_lz(st[0], nl, chunks, nch, st[1], ns, st[2], st[2] + nlen, st[3], T, st[4], nn, FAR, nf, dst, raw_size);
out:
    free(own);
    return r;
}

/* Regions (128 KB of output) where the parse's matches save under 1% of the region, against storing its bytes, are
   stored: their matches become literals (which then stay raw). Near-random data (BC7 textures) otherwise keeps
   hundreds of barely-paying 3-byte matches per region, each a sequence to decode (~20 ticks) where a memcpy would
   do; Kraken stores such quanta. Returns the new sequence count. */
static inline size_t zap__k_store(zap__kseq *q, size_t nq, const zap__kc *cm) {
    size_t n = 0, nr, w = 0, carry = 0;
    for (size_t i = 0; i < nq; i++) n += q[i].ll + q[i].ml;
    nr = n / ZAP__K_CHUNK + 1;
    int64_t *save = (int64_t *)calloc(nr, sizeof *save); /* 1/16 bits */
    uint8_t gt[256];
    if (!save) return nq;
    zap__grp_tok(gt);
    uint32_t rep[3] = { 0, 0, 0 };
    unsigned pg = 0;
    for (size_t i = 0, pos = 0; i < nq && q[i].ml; i++) {
        pos += q[i].ll;
        save[pos / ZAP__K_CHUNK] += 128 * (int64_t)q[i].ml - (int64_t)zap__k_lprice(cm, q[i].ml, q[i].ll, pg) - (int64_t)zap__k_oprice(cm, q[i].off, rep);
        pg = gt[zap__k_tok(q[i].ll, q[i].ml)];
        zap__rep_push(rep, q[i].off);
        pos += q[i].ml;
    }
    for (size_t i = 0, pos = 0; i < nq; i++) {
        size_t at = pos + q[i].ll;
        pos = at + q[i].ml;
        if (q[i].ml && save[at / ZAP__K_CHUNK] < (int64_t)ZAP__K_CHUNK * 128 / 100) { carry += q[i].ll + q[i].ml; continue; } /* 8 bits a byte: 1% */
        q[w] = q[i]; q[w].ll += (uint32_t)carry; carry = 0; w++;
    }
    free(save);
    return w;
}
/* The first pass is only there for its statistics: every other 64 KB is enough (-0.03%, half the work). The
   segments' parses are joined by gaps. */
#define ZAP__K_SEG (64u << 10)
static inline size_t zap__k_sampled(const uint8_t *src, size_t n, zap__kseq *out, const zap__kmt *mt, const zap__kc *cm, int K) {
    size_t nq = 0, at = 0;
    do { /* at least one segment, even an empty one */
        size_t sl = n - at < ZAP__K_SEG ? n - at : ZAP__K_SEG, gap = n - at - sl < ZAP__K_SEG ? n - at - sl : ZAP__K_SEG, k;
        zap__kmt v = { mt->idx + at, mt->m }; /* the table from here on */
        if (!(k = zap__compress_ma(src + at, sl, at, out + nq, &v, cm, K))) return 0;
        nq += k;
        out[nq - 1].ml = (uint32_t)gap; /* the segment's trailing literals, then the skipped one (none: the end) */
    } while ((at += 2 * ZAP__K_SEG) < n);
    return nq;
}
/* the match table, a lazy parse over it, then optimal parses with K arrivals, each priced from the previous parse
   (the first one sampled). Arrivals per pass: depth 32-47: 1, 1; 48-63: 1, 2; 64-127: 1, 2, 2, 4; 128+: 1, 2, 4, 8
   (more passes pay more than more arrivals: 1, 2, 2, 4 beats 4, 4, 4 at less work). The table's search depth is 128
   or more (deeper finds more, at about the same speed). */
static inline size_t zap__k_compress(const uint8_t *src, size_t n, uint8_t *dst, size_t cap, zap_hc_state *hc, int depth, int rawpct) {
    static const uint8_t sched[4][4] = { { 1, 1 }, { 1, 2 }, { 1, 2, 2, 4 }, { 1, 2, 4, 8 } };
    int lv = depth >= 128 ? 3 : depth >= 64 ? 2 : depth >= 48 ? 1 : 0;
    size_t r = 0, nq = 0;
    zap__kseq *q = (zap__kseq *)malloc(sizeof(zap__kseq) * (n / 3 + 2));
    zap__kc *cm = (zap__kc *)malloc(sizeof *cm);
    zap__kmt mt = { NULL, NULL };
    if (cm) { cm->dreg = NULL; cm->nreg = 0; }
    if (q && cm && !zap__kmt_build(&mt, src, n, hc, depth < 128 ? 128 : depth)) {
        nq = zap__k_lazy(n, &mt, q);
        for (int pass = 0; nq && pass < 4 && sched[lv][pass]; pass++) {
            zap__k_costs(src, q, nq, zap__k_scale(q, nq), cm, 24); /* sequence penalty 1.5 bits: fewer, longer sequences; ratio-neutral */
            nq = pass ? zap__compress_ma(src, n, 0, q, &mt, cm, sched[lv][pass]) : zap__k_sampled(src, n, q, &mt, cm, sched[lv][pass]);
        }
        if (nq) nq = zap__k_store(q, nq, cm);
        if (nq) r = zap__k_encode(src, q, nq, zap__k_scale(q, nq), n, dst, cap, rawpct);
    }
    zap__kmt_free(&mt); free(q);
    if (cm) free(cm->dreg);
    free(cm);
    return r;
}

/* ---------------- frames: [magic][block_size u32][raw_size u64][end offset u64 per block][blocks]
 * "ZAP1": a block whose stored size equals its raw size is stored uncompressed, else it's a plain block.
 * "ZAP2" (written when depth has ZAP_ENTROPY or ZAP_TURBO): each block starts with a method byte: 0 raw, 1 plain, 2 entropy, 3 turbo. */
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

/* version-2 frame block: method byte + payload, smallest of raw / plain / entropy (or raw / turbo with ZAP_TURBO and
   no dictionary). 0 = no room or out of memory. */
static inline size_t zap__block2(const uint8_t *src, size_t len, uint8_t *dst, size_t lim, void *st, int depth, const zap_dict *d) {
    size_t lcap = zap_bound(len), ln = 0, en = 0, best = len;
    int m = 0, hc = depth & 0xFFFF;
    uint8_t *lz = (uint8_t *)malloc(lcap), *e = (uint8_t *)malloc(len + 1);
    if (!lz || !e) { free(lz); free(e); return 0; }
    if (len >= 2 && (depth & ZAP_TURBO) && !d) {
        ln = hc >= ZAP_OPT_DEPTH ? zap__t_parse(src, len, lz, lcap, (zap_hc_state *)st, hc)
                                 : hc ? zap_compress_hc(src, len, lz, lcap, (zap_hc_state *)st, NULL, hc) : zap_compress(src, len, lz, lcap, (zap_state *)st, NULL);
        if (ln && best > 1) en = zap__t_encode(lz, ln, e, best - 1);
        if (en && en < best) { best = en; m = 3; }
    } else if (len >= 2 && !d && hc >= ZAP_OPT_DEPTH) {
        /* v3 parses on its own: the plain parse (~15% of the time) only runs as a fallback, or on small blocks where
           v3's stream headers can lose to it */
        en = zap_compress_entropy(src, len, e, len - 1, st, depth, NULL);
        if (en) { best = en; m = 2; }
        if (!en || len < (64u << 10)) {
            ln = zap_compress_hc(src, len, lz, lcap, (zap_hc_state *)st, NULL, depth);
            if (ln && ln < best) { best = ln; m = 1; }
        }
    } else if (len >= 2) {
        ln = hc ? zap_compress_hc(src, len, lz, lcap, (zap_hc_state *)st, d, depth) : zap_compress(src, len, lz, lcap, (zap_state *)st, d);
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
    int v2 = (depth & (ZAP_ENTROPY | ZAP_TURBO)) != 0;
    depth = (depth & 0xFFFF ? depth & ~ZAP_ENTROPY : 0) | (depth & ZAP_TURBO);
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb, v2), hdr = pos;
    if (!pos) return 0;
    void *st = malloc(depth & 0xFFFF ? sizeof(zap_hc_state) : sizeof(zap_state));
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
    if (m == 3) return zap__t_decode(p, sz, o, len) == (ptrdiff_t)len ? 0 : -1;
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
    void *st = malloc(j->depth & 0xFFFF ? sizeof(zap_hc_state) : sizeof(zap_state));
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
    int v2 = (depth & (ZAP_ENTROPY | ZAP_TURBO)) != 0;
    size_t nb, pos = zap__frame_hdr(dst, cap, n, bs, &nb, v2), hdr = pos;
    if (!pos) return 0;
    size_t slot = v2 ? (nb == 1 ? n : bs) + 1 : bs;
    zap__cjob j = { src, (uint8_t *)malloc(v2 ? nb * slot + 1 : n + 1), n, bs, nb, (size_t *)malloc((nb ? nb : 1) * sizeof(size_t)), slot, (depth & 0xFFFF ? depth & ~ZAP_ENTROPY : 0) | (depth & ZAP_TURBO), v2, d };
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
    size_t es = n / nseg < SEG ? (size_t)SEG : n / nseg;
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
