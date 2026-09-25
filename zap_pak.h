/* zap_pak.h - .zappak archives: many named files, each stored as a zap frame, plus a sorted table of contents.
 * https://github.com/WatchDogStudios/zap   SPDX-License-Identifier: MIT   Copyright (c) 2026 WD Studios Corp.
 *
 *   zap_pak_file in[] = { { "textures/wall.dds", data, size }, ... };
 *   size_t cap = zap_pak_bound(in, n, block), len = zap_pak_write(in, n, out, cap, block, depth, threads);
 *
 *   zap_pak k;
 *   if (zap_pak_open(&k, buf, len) == 0) {
 *       long i = zap_pak_find(&k, "textures/wall.dds");
 *       if (i >= 0) zap_pak_read(&k, (size_t)i, dst, zap_pak_raw_size(&k, (size_t)i));
 *   }
 *
 * block and depth are zap_frame_compress's (depth may include ZAP_ENTROPY / ZAP_FAST_DECODE). threads > 1 compresses
 * files in parallel when ZAP_THREADS is defined; the archive is byte-identical for any thread count.
 *
 * Layout (little-endian):
 *   header  32 bytes: "ZPAK", u32 version (1), u32 count, u32 0, u64 toc offset, u64 toc size
 *   data    one zap frame per file, back to back
 *   toc     count x 32-byte entries { u64 offset, u64 stored size, u64 raw size, u32 name offset, u32 name length },
 *           sorted by name (bytewise), then the names (UTF-8, '/' separators, not NUL-terminated)
 * zap_pak_open checks every entry against the buffer, so a corrupt archive fails there or in the frame decoder.
 */
#ifndef ZAP_PAK_H
#define ZAP_PAK_H

#include "zap.h"

typedef struct { const char *name; const void *data; size_t size; } zap_pak_file;
typedef struct { const uint8_t *p, *ent, *names; size_t n, names_size; uint32_t count; } zap_pak;

enum { ZAP__PAKHDR = 32, ZAP__PAKENT = 32 };

static inline size_t zap_pak_bound(const zap_pak_file *f, size_t n, size_t block) {
    size_t s = ZAP__PAKHDR + ZAP__PAKENT * n;
    for (size_t i = 0; i < n; i++) s += zap_frame_bound(f[i].size, block) + strlen(f[i].name);
    return s;
}

/* ---------------- writer */

typedef struct { const zap_pak_file *f; size_t n, block, *cs; uint8_t **buf; int depth; } zap__pjob;
static inline void zap__pwork(void *p, int t, int nt) {
    zap__pjob *j = (zap__pjob *)p;
    for (size_t i = (size_t)t; i < j->n; i += (size_t)nt) {
        size_t cap = zap_frame_bound(j->f[i].size, j->block);
        j->buf[i] = (uint8_t *)malloc(cap ? cap : 1);
        j->cs[i] = j->buf[i] ? zap_frame_compress(j->f[i].data, j->f[i].size, j->buf[i], cap, j->block, j->depth, NULL) : 0;
    }
}

static const zap_pak_file *zap__pak_sortbase;
static inline int zap__pak_cmp(const void *a, const void *b) {
    return strcmp(zap__pak_sortbase[*(const size_t *)a].name, zap__pak_sortbase[*(const size_t *)b].name);
}

/* returns the archive size, or 0 on failure (cap too small, duplicate names, out of memory) */
static inline size_t zap_pak_write(const zap_pak_file *f, size_t n, void *dst_, size_t cap, size_t block, int depth, int threads) {
    uint8_t *dst = (uint8_t *)dst_;
    if (cap < ZAP__PAKHDR || n > 0xFFFFFFFFu) return 0;
    size_t *ord = (size_t *)malloc((n ? n : 1) * sizeof(size_t)), *cs = (size_t *)calloc(n ? n : 1, sizeof(size_t)), pos = ZAP__PAKHDR, ok = 1;
    uint8_t **buf = (uint8_t **)calloc(n ? n : 1, sizeof(uint8_t *));
    if (!ord || !cs || !buf) { free(ord); free(cs); free(buf); return 0; }
    for (size_t i = 0; i < n; i++) ord[i] = i;
    zap__pak_sortbase = f; /* ponytail: qsort has no context pointer in C99; don't call zap_pak_write from two threads at once */
    qsort(ord, n, sizeof *ord, zap__pak_cmp);
    for (size_t i = 1; i < n; i++) if (!strcmp(f[ord[i - 1]].name, f[ord[i]].name)) ok = 0;
    zap__pjob j = { f, n, block, cs, buf, depth };
    (void)threads;
    if (ok) {
#ifdef ZAP_THREADS
        zap__par(zap__threads(threads, n), zap__pwork, &j);
#else
        zap__pwork(&j, 0, 1);
#endif
    }
    size_t names = 0;
    for (size_t k = 0; ok && k < n; k++) { /* frames in name order */
        size_t i = ord[k];
        if (!cs[i] || cap - pos < cs[i]) { ok = 0; break; }
        memcpy(dst + pos, buf[i], cs[i]);
        pos += cs[i];
        names += strlen(f[i].name);
    }
    size_t toc = pos, tsz = ZAP__PAKENT * n + names;
    if (ok && (cap - pos < tsz || names > 0xFFFFFFFFu)) ok = 0;
    for (size_t k = 0, off = ZAP__PAKHDR, no = 0; ok && k < n; k++) {
        size_t i = ord[k], nl = strlen(f[i].name);
        uint8_t *e = dst + toc + ZAP__PAKENT * k;
        zap__w64(e, off); zap__w64(e + 8, cs[i]); zap__w64(e + 16, f[i].size);
        zap__w32(e + 24, (uint32_t)no); zap__w32(e + 28, (uint32_t)nl);
        memcpy(dst + toc + ZAP__PAKENT * n + no, f[i].name, nl);
        off += cs[i]; no += nl;
    }
    for (size_t i = 0; i < n; i++) free(buf[i]);
    free(ord); free(cs); free(buf);
    if (!ok) return 0;
    memcpy(dst, "ZPAK", 4); zap__w32(dst + 4, 1); zap__w32(dst + 8, (uint32_t)n); zap__w32(dst + 12, 0);
    zap__w64(dst + 16, toc); zap__w64(dst + 24, tsz);
    return toc + tsz;
}

/* ---------------- reader */

static inline int zap_pak_open(zap_pak *k, const void *p_, size_t n) {
    const uint8_t *p = (const uint8_t *)p_;
    if (n < ZAP__PAKHDR || memcmp(p, "ZPAK", 4) || zap__r32(p + 4) != 1) return -1;
    uint64_t cnt = zap__r32(p + 8), toc = zap__r64(p + 16), tsz = zap__r64(p + 24);
    if (toc < ZAP__PAKHDR || toc > n || tsz != n - toc || tsz / ZAP__PAKENT < cnt) return -1;
    k->p = p; k->n = n; k->count = (uint32_t)cnt;
    k->ent = p + toc; k->names = k->ent + ZAP__PAKENT * cnt; k->names_size = (size_t)(tsz - ZAP__PAKENT * cnt);
    for (uint64_t i = 0; i < cnt; i++) {
        const uint8_t *e = k->ent + ZAP__PAKENT * i;
        uint64_t off = zap__r64(e), st = zap__r64(e + 8), no = zap__r32(e + 24), nl = zap__r32(e + 28);
        if (off < ZAP__PAKHDR || off > toc || st > toc - off || no > k->names_size || nl > k->names_size - no) return -1;
    }
    return 0;
}

static inline size_t zap_pak_count(const zap_pak *k) { return k->count; }
static inline size_t zap_pak_raw_size(const zap_pak *k, size_t i) { return (size_t)zap__r64(k->ent + ZAP__PAKENT * i + 16); }
static inline size_t zap_pak_stored_size(const zap_pak *k, size_t i) { return (size_t)zap__r64(k->ent + ZAP__PAKENT * i + 8); }
/* name of entry i: not NUL-terminated */
static inline const char *zap_pak_name(const zap_pak *k, size_t i, size_t *len) {
    const uint8_t *e = k->ent + ZAP__PAKENT * i;
    *len = zap__r32(e + 28);
    return (const char *)k->names + zap__r32(e + 24);
}
/* the entry's zap frame, for zap_frame_decode_mt or per-block decoding */
static inline const void *zap_pak_frame(const zap_pak *k, size_t i, size_t *len) {
    const uint8_t *e = k->ent + ZAP__PAKENT * i;
    *len = (size_t)zap__r64(e + 8);
    return k->p + zap__r64(e);
}

static inline long zap_pak_find(const zap_pak *k, const char *name) {
    size_t lo = 0, hi = k->count, nl = strlen(name);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2, ml;
        const char *m = zap_pak_name(k, mid, &ml);
        int c = memcmp(m, name, ml < nl ? ml : nl);
        if (!c) c = ml < nl ? -1 : ml > nl;
        if (!c) return (long)mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return -1;
}

/* decode entry i into dst; returns the raw size, or -1 (corrupt, or cap too small) */
static inline ptrdiff_t zap_pak_read(const zap_pak *k, size_t i, void *dst, size_t cap) {
    size_t len;
    const void *fr = zap_pak_frame(k, i, &len);
    ptrdiff_t r = zap_frame_decode(fr, len, dst, cap, NULL);
    return r == (ptrdiff_t)zap_pak_raw_size(k, i) ? r : -1;
}

#endif
