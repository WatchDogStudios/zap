/* zap.c - command line tool.
 *   zap c [-e] [-x] [-l depth] [-b block_kb] [-t threads] [-D dict] in out   pack (depth 0 = fast, >= 32 optimal parse, default 64; -e entropy mode, -x faster decode)
 *   zap d [-t threads] [-D dict] in out                            unpack
 *   zap train [-s dict_bytes] out samples...                       train a packet dictionary (default 16384)
 *   zap tex [-f bc1|bc3|bc4|bc5|bc7|astc] [-r rdo] [-m] [-S] w h in.rgba out.dds|.astc  GPU block-compress raw RGBA8 to DDS (-m mips, -S sRGB)
 *   zap venc [-q quality] [-k keyint] [-F fps] [-t threads] [-C 601|709|601full|709full] w h in.yuv|- out.zapvid   encode raw I420 (- = stdin)
 *   zap vdec [-t threads] in.zapvid out.yuv|-|null                decode to raw I420 (- = stdout, null = time only)
 *   zap pak [-e] [-x] [-l depth] [-b block_kb] [-t threads] out.zappak files/folders...   package
 *   zap unpak in.zappak outdir                                     extract a package
 *   zap ls file                                                    describe a .zappak, .zapvid or zap frame
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#define ZAP_THREADS
#include "zap.h"
#include "zap_tex.h"
#include "zap_video.h"
#include "zap_pak.h"
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#define zap_mkdir(p) _mkdir(p)
#else
#include <dirent.h>
#include <sys/stat.h>
#define zap_mkdir(p) mkdir((p), 0777)
#endif

#ifdef _WIN32
#define zap_ftell _ftelli64
#define zap_fseek _fseeki64
#else
#define zap_ftell ftello
#define zap_fseek fseeko
#endif

static uint8_t *load(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    zap_fseek(f, 0, SEEK_END); *n = (size_t)zap_ftell(f); zap_fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(*n + 1);
    if (!p || fread(p, 1, *n, f) != *n) { fprintf(stderr, "%s: read failed\n", path); exit(1); }
    fclose(f);
    return p;
}

static FILE *create(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    return f;
}

static void save(const char *path, const void *p, size_t n) {
    FILE *f = create(path);
    if (fwrite(p, 1, n, f) != n || fclose(f)) { perror(path); exit(1); }
}

static int usage(void) {
    fprintf(stderr, "zap c [-e] [-x] [-l depth] [-b block_kb] [-t threads] [-D dict] in out\n"
                    "zap d [-t threads] [-D dict] in out\n"
                    "zap train [-s bytes] out samples...\n"
                    "zap tex [-f bc1|bc3|bc4|bc5|bc7|astc] [-r rdo] [-m] [-S] w h in.rgba out.dds|.astc\n"
                    "zap venc [-q quality] [-k keyint] [-F fps|num/den] [-t threads] [-C 601|709|601full|709full] w h in.yuv|- out.zapvid\n"
                    "zap vdec [-t threads] in.zapvid out.yuv|-|null\n"
                    "zap pak [-e] [-x] [-l depth] [-b block_kb] [-t threads] out.zappak files/folders...\n"
                    "zap unpak in.zappak outdir\n"
                    "zap ls file\n");
    return 1;
}

static double now(void) { struct timespec t; timespec_get(&t, TIME_UTC); return t.tv_sec + t.tv_nsec * 1e-9; }

/* raw I420 in (file or stdin) -> .zapvid: header, packets, index; the header is patched once the count is known */
static int venc(int w, int h, int quality, int keyint, uint32_t fps_num, uint32_t fps_den, int depth, int threads, int color, const char *in, const char *outp) {
    FILE *fi = strcmp(in, "-") ? fopen(in, "rb") : stdin, *fo = create(outp);
    if (!fi) { perror(in); return 1; }
#ifdef _WIN32
    if (fi == stdin) _setmode(_fileno(stdin), _O_BINARY);
#endif
    zap_video *e = zap_venc_create(w, h, quality, keyint, depth);
    if (!e) { fprintf(stderr, "bad size (width and height must be even, 2..16384)\n"); return 1; }
    zap_venc_threads(e, threads);
    size_t fsz = (size_t)w * h * 3 / 2, cap = zap_video_bound(e), total = 0, nf = 0, icap = 0;
    uint8_t *frame = malloc(fsz), *pkt = malloc(cap), hdr[ZAP_VID_HEADER] = { 0 }, *index = NULL;
    if (!frame || !pkt) { fprintf(stderr, "out of memory\n"); return 1; }
    fwrite(hdr, 1, sizeof hdr, fo); /* placeholder */
    uint64_t off = ZAP_VID_HEADER;
    double t0 = now();
    while (fread(frame, 1, fsz, fi) == fsz) {
        size_t n = zap_venc_frame(e, frame, frame + (size_t)w * h, frame + (size_t)w * h * 5 / 4, w, w / 2, pkt, cap);
        if (!n || fwrite(pkt, 1, n, fo) != n) { fprintf(stderr, "encode/write failed\n"); return 1; }
        if (nf == icap && !(index = realloc(index, (icap = icap ? icap * 2 : 256) * ZAP_VID_ENTRY))) { fprintf(stderr, "out of memory\n"); return 1; }
        zap_vid_write_entry(index + nf * ZAP_VID_ENTRY, off, (uint32_t)n, pkt[2] == 'I');
        off += n; total += n; nf++;
    }
    double t = now() - t0;
    if (nf > 0xFFFFFFFFu || (nf && fwrite(index, 1, nf * ZAP_VID_ENTRY, fo) != nf * ZAP_VID_ENTRY)) { fprintf(stderr, "write failed\n"); return 1; }
    if (color < 0) color = h >= 720 ? ZAP_VID_BT709 : 0; /* ffmpeg's convention for untagged video */
    zap_vid_write_header(hdr, w, h, fps_num, fps_den, (uint32_t)nf, (uint32_t)keyint, off, color);
    if (zap_fseek(fo, 0, SEEK_SET) || fwrite(hdr, 1, sizeof hdr, fo) != sizeof hdr || fclose(fo)) { perror(outp); return 1; }
    double fps = (double)fps_num / fps_den;
    printf("%zu frames %dx%d -> %zu bytes (%.0f kbit/s at %.3f fps), encoded at %.1f fps\n", nf, w, h, total + ZAP_VID_HEADER + nf * ZAP_VID_ENTRY,
           nf ? total * 8.0 * fps / nf / 1000 : 0.0, fps, nf / (t > 0 ? t : 1e-9));
    zap_video_destroy(e);
    free(frame); free(pkt); free(index);
    if (fi != stdin) fclose(fi);
    return 0;
}

static int vdec(const char *in, const char *outp, int threads) {
    size_t n;
    uint8_t *f = load(in, &n);
    zap_vid v;
    if (zap_vid_open(&v, f, n)) { fprintf(stderr, "%s: not a valid .zapvid file\n", in); return 1; }
    zap_video *d = zap_vdec_create(v.w, v.h);
    if (!d) { fprintf(stderr, "bad header\n"); return 1; }
    zap_vdec_threads(d, threads);
    int null = !strcmp(outp, "null"); /* decode only, for timing */
    FILE *fo = null ? NULL : strcmp(outp, "-") ? create(outp) : stdout;
#ifdef _WIN32
    if (fo == stdout) _setmode(_fileno(stdout), _O_BINARY);
#endif
    double td = 0;
    for (uint32_t i = 0; i < v.frames; i++) {
        size_t len;
        const void *pk = zap_vid_frame(&v, i, &len, NULL);
        double t0 = now();
        if (zap_vdec_frame(d, pk, len)) { fprintf(stderr, "corrupt frame %u\n", i); return 1; }
        td += now() - t0;
        if (null) continue;
        const uint8_t *p[3]; int ys, cs;
        zap_video_planes(d, &p[0], &p[1], &p[2], &ys, &cs);
        for (int pl = 0; pl < 3; pl++)
            for (int y = 0; y < (pl ? v.h / 2 : v.h); y++) fwrite(p[pl] + (size_t)y * (pl ? cs : ys), 1, (size_t)(pl ? v.w / 2 : v.w), fo);
    }
    if (fo && fo != stdout && fclose(fo)) { perror(outp); return 1; }
    fprintf(fo == stdout ? stderr : stdout, "%u frames, %dx%d, decoded at %.0f fps (%d thread%s)\n", v.frames, v.w, v.h, v.frames / (td > 0 ? td : 1e-9),
            d->threads, d->threads > 1 ? "s" : "");
    zap_video_destroy(d);
    free(f);
    return 0;
}

/* ---------------- packages */

typedef struct { char *path, *name; } pak_in;
static pak_in *g_in;
static size_t g_nin, g_cin;
static void pak_push(const char *path, const char *name) {
    if (g_nin == g_cin && !(g_in = realloc(g_in, (g_cin = g_cin ? g_cin * 2 : 64) * sizeof *g_in))) { fprintf(stderr, "out of memory\n"); exit(1); }
    size_t pl = strlen(path) + 1, nl = strlen(name) + 1;
    g_in[g_nin].path = malloc(pl); g_in[g_nin].name = malloc(nl);
    memcpy(g_in[g_nin].path, path, pl); memcpy(g_in[g_nin].name, name, nl);
    for (char *c = g_in[g_nin].name; *c; c++) if (*c == '\\') *c = '/';
    g_nin++;
}
/* ponytail: ANSI directory APIs on Windows, so non-ASCII file names outside the code page aren't found; use the
   wide-char APIs if content needs them */
static void pak_walk(const char *dir, const char *name) {
    char p[4096], nm[4096];
#ifdef _WIN32
    struct _finddata_t fd;
    snprintf(p, sizeof p, "%s/*", dir);
    intptr_t h = _findfirst(p, &fd);
    if (h == -1) return;
    do {
        if (!strcmp(fd.name, ".") || !strcmp(fd.name, "..")) continue;
        snprintf(p, sizeof p, "%s/%s", dir, fd.name); snprintf(nm, sizeof nm, "%s/%s", name, fd.name);
        if (fd.attrib & _A_SUBDIR) pak_walk(p, nm); else pak_push(p, nm);
    } while (_findnext(h, &fd) == 0);
    _findclose(h);
#else
    DIR *d = opendir(dir);
    if (!d) return;
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name); snprintf(nm, sizeof nm, "%s/%s", name, e->d_name);
        if (stat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) pak_walk(p, nm); else if (S_ISREG(st.st_mode)) pak_push(p, nm);
    }
    closedir(d);
#endif
}
static int is_dir(const char *p) {
#ifdef _WIN32
    struct _finddata_t fd;
    intptr_t h = _findfirst(p, &fd);
    if (h == -1) return 0;
    _findclose(h);
    return (fd.attrib & _A_SUBDIR) != 0;
#else
    struct stat st;
    return !stat(p, &st) && S_ISDIR(st.st_mode);
#endif
}

/* folders keep their own name as the top level ("levels/e1m1.bsp"); files are stored under their base name */
static int pak(const char *outp, char **items, int n, size_t bs, int depth, int threads) {
    for (int k = 0; k < n; k++) {
        char item[4096];
        snprintf(item, sizeof item, "%s", items[k]);
        size_t l = strlen(item);
        while (l > 1 && (item[l - 1] == '/' || item[l - 1] == '\\')) item[--l] = 0;
        const char *base = item + l;
        while (base > item && base[-1] != '/' && base[-1] != '\\' && base[-1] != ':') base--;
        if (is_dir(item)) pak_walk(item, base); else pak_push(item, base);
    }
    zap_pak_file *f = malloc((g_nin ? g_nin : 1) * sizeof *f);
    size_t raw = 0;
    for (size_t i = 0; i < g_nin; i++) {
        size_t sz;
        f[i].name = g_in[i].name; f[i].data = load(g_in[i].path, &sz); f[i].size = sz; raw += sz;
    }
    size_t cap = zap_pak_bound(f, g_nin, bs);
    uint8_t *out = malloc(cap);
    double t0 = now();
    size_t len = out ? zap_pak_write(f, g_nin, out, cap, bs, depth, threads) : 0;
    double t = now() - t0;
    if (!len) { fprintf(stderr, "packing failed (duplicate names, or out of memory)\n"); return 1; }
    save(outp, out, len);
    printf("%zu files, %zu -> %zu bytes (%.3fx) in %.2f s\n", g_nin, raw, len, (double)raw / len, t);
    for (size_t i = 0; i < g_nin; i++) { free((void *)f[i].data); free(g_in[i].path); free(g_in[i].name); }
    free(f); free(out); free(g_in);
    return 0;
}

/* a stored name is only written if it stays inside outdir: relative, no "..", no drive letters or backslashes */
static int safe_name(const char *n, size_t l) {
    if (!l || n[0] == '/') return 0;
    for (size_t i = 0; i < l; i++) if (n[i] == '\\' || n[i] == ':' || !n[i]) return 0;
    for (size_t i = 0; i < l;) {
        size_t j = i;
        while (j < l && n[j] != '/') j++;
        if (j == i || (j - i == 2 && n[i] == '.' && n[i + 1] == '.') || (j - i == 1 && n[i] == '.')) return 0;
        i = j + 1;
    }
    return 1;
}
static int unpak(const char *in, const char *outdir) {
    size_t n;
    uint8_t *p = load(in, &n);
    zap_pak k;
    if (zap_pak_open(&k, p, n)) { fprintf(stderr, "%s: not a valid .zappak\n", in); return 1; }
    zap_mkdir(outdir);
    size_t cnt = zap_pak_count(&k), big = 1, total = 0, written = 0;
    for (size_t i = 0; i < cnt; i++) if (zap_pak_raw_size(&k, i) > big) big = zap_pak_raw_size(&k, i);
    uint8_t *buf = malloc(big);
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
    double td = 0;
    for (size_t i = 0; i < cnt; i++) {
        size_t nl;
        const char *nm = zap_pak_name(&k, i, &nl);
        if (!safe_name(nm, nl) || strlen(outdir) + nl + 2 > 4096) { fprintf(stderr, "skipping unsafe name %.*s\n", (int)nl, nm); continue; }
        char path[4096];
        size_t ol = strlen(outdir);
        memcpy(path, outdir, ol); path[ol] = '/'; memcpy(path + ol + 1, nm, nl); path[ol + 1 + nl] = 0;
        for (char *c = path + strlen(outdir) + 1; *c; c++) if (*c == '/') { *c = 0; zap_mkdir(path); *c = '/'; } /* parents */
        double t0 = now();
        ptrdiff_t r = zap_pak_read(&k, i, buf, big);
        td += now() - t0;
        if (r < 0) { fprintf(stderr, "corrupt entry %.*s\n", (int)nl, nm); return 1; }
        save(path, buf, (size_t)r);
        total += (size_t)r; written++;
    }
    printf("%zu of %zu files, %zu bytes, decoded at %.0f MB/s (one thread)\n", written, cnt, total, total / 1e6 / (td > 0 ? td : 1e-9));
    free(buf); free(p);
    return written == cnt ? 0 : 2; /* 2: some names were refused */
}

static int ls(const char *in) {
    size_t n;
    uint8_t *p = load(in, &n);
    zap_pak k;
    zap_vid v;
    zap_frame fr;
    if (zap_pak_open(&k, p, n) == 0) {
        size_t raw = 0;
        printf("%12s %12s %7s  name\n", "size", "stored", "ratio");
        for (size_t i = 0; i < zap_pak_count(&k); i++) {
            size_t nl, r = zap_pak_raw_size(&k, i), st = zap_pak_stored_size(&k, i);
            const char *nm = zap_pak_name(&k, i, &nl);
            printf("%12zu %12zu %7.3f  %.*s\n", r, st, st ? (double)r / st : 0.0, (int)nl, nm);
            raw += r;
        }
        printf(".zappak: %zu files, %zu bytes -> %zu (%.3fx)\n", zap_pak_count(&k), raw, n, (double)raw / n);
    } else if (zap_vid_open(&v, p, n) == 0) {
        uint32_t keys = 0;
        for (uint32_t i = 0; i < v.frames; i++) { int key; size_t len; zap_vid_frame(&v, i, &len, &key); keys += key; }
        double fps = (double)v.fps_num / v.fps_den, secs = v.frames / fps;
        printf(".zapvid: %dx%d, %u frames, %u/%u fps (%.3f), %.2f s, %u keyframes (interval %u), %.0f kbit/s, %s %s range\n", v.w, v.h, v.frames,
               v.fps_num, v.fps_den, fps, secs, keys, v.keyint, secs > 0 ? n * 8.0 / secs / 1000 : 0.0, v.color & ZAP_VID_BT709 ? "BT.709" : "BT.601",
               v.color & ZAP_VID_FULL_RANGE ? "full" : "limited");
    } else if (zap_frame_open(&fr, p, n) == 0) {
        printf("zap frame (version %d): %llu bytes -> %zu (%.3fx), %zu blocks of %zu\n", fr.v, (unsigned long long)fr.raw, n, (double)fr.raw / n, (size_t)fr.nb, (size_t)fr.bs);
    } else {
        fprintf(stderr, "%s: not a zap file\n", in);
        free(p);
        return 1;
    }
    free(p);
    return 0;
}

/* mips: full chain (linear-light filtering when srgb). srgb also picks the *_SRGB DXGI format (BC1/BC3/BC7). */
static int tex(int w, int h, const char *fmt, float rdo, int mips, int srgb, const char *in, const char *outp) {
    static const char *NAMES[] = { "bc1", "bc3", "bc4", "bc5", "bc7" }, *CC[] = { "DXT1", "DXT5", "BC4U", "BC5U", "DX10" };
    static const uint32_t DXGI[5] = { 71, 77, 80, 83, 98 }; /* BC1..BC7 UNORM; +1 = _SRGB for BC1, BC3, BC7 */
    int f = 0;
    if (!strcmp(fmt, "astc")) { /* .astc container (one level, 4x4 blocks, UNORM) */
        size_t n, bn = zap_bc_size(w, h, ZAP_BC7);
        uint8_t *img = load(in, &n), *bc = malloc(bn), hd[16] = { 0x13, 0xAB, 0xA1, 0x5C, 4, 4, 1 };
        if (n < (size_t)w * h * 4) { fprintf(stderr, "%s: expected %dx%d RGBA8\n", in, w, h); return 1; }
        if (mips) fprintf(stderr, "note: .astc holds one level; -m ignored\n");
        zap_astc_encode(img, w, h, (size_t)w * 4, bc);
        for (int k = 0; k < 3; k++) { hd[7 + k] = (uint8_t)(w >> (8 * k)); hd[10 + k] = (uint8_t)(h >> (8 * k)); }
        hd[13] = 1; /* depth 1 */
        FILE *fo = create(outp);
        if (fwrite(hd, 1, 16, fo) != 16 || fwrite(bc, 1, bn, fo) != bn || fclose(fo)) { perror(outp); return 1; }
        printf("%dx%d astc 4x4 -> %zu bytes\n", w, h, bn + 16);
        (void)rdo; (void)srgb;
        return 0;
    }
    while (f < 5 && strcmp(fmt, NAMES[f])) f++;
    if (f == 5 || w < 1 || h < 1) return usage();
    if (f == ZAP_BC4 || f == ZAP_BC5) srgb = 0; /* channel data, never sRGB */
    int levels = mips ? zap_mip_levels(w, h) : 1, dx10 = f == ZAP_BC7 || srgb;
    size_t n, hn = dx10 ? 148 : 128, total = 0;
    for (int l = 0, lw = w, lh = h; l < levels; l++, lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1) total += zap_bc_size(lw, lh, (zap_bc_format)f);
    uint8_t *img = load(in, &n), hdr[148] = { 'D', 'D', 'S', ' ' };
    if (n < (size_t)w * h * 4) { fprintf(stderr, "%s: expected %dx%d RGBA8\n", in, w, h); return 1; }
    uint8_t *bc = malloc(total), *next = malloc((size_t)(w > 1 ? w / 2 : 1) * (size_t)(h > 1 ? h / 2 : 1) * 4 + 4), *p = bc;
    for (int l = 0, lw = w, lh = h; l < levels; l++) {
        zap_bc_encode(img, lw, lh, (size_t)lw * 4, (zap_bc_format)f, rdo, p);
        p += zap_bc_size(lw, lh, (zap_bc_format)f);
        if (l + 1 < levels) {
            zap_mip_next(img, lw, lh, (size_t)lw * 4, srgb, next, (size_t)(lw > 1 ? lw / 2 : 1) * 4);
            lw = lw > 1 ? lw / 2 : 1; lh = lh > 1 ? lh / 2 : 1;
            memcpy(img, next, (size_t)lw * lh * 4);
        }
    }
    zap__w32(hdr + 4, 124); zap__w32(hdr + 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | (mips ? 0x20000 : 0));
    zap__w32(hdr + 12, (uint32_t)h); zap__w32(hdr + 16, (uint32_t)w); zap__w32(hdr + 20, (uint32_t)zap_bc_size(w, h, (zap_bc_format)f));
    zap__w32(hdr + 28, (uint32_t)levels);
    zap__w32(hdr + 76, 32); zap__w32(hdr + 80, 0x4); memcpy(hdr + 84, dx10 ? "DX10" : CC[f], 4);
    zap__w32(hdr + 108, 0x1000 | (mips ? 0x400008 : 0)); /* TEXTURE (+ COMPLEX | MIPMAP) */
    zap__w32(hdr + 128, DXGI[f] + (uint32_t)(srgb != 0)); zap__w32(hdr + 132, 3); zap__w32(hdr + 140, 1); /* DX10: format, 2D, 1 element */
    FILE *fo = create(outp);
    if (fwrite(hdr, 1, hn, fo) != hn || fwrite(bc, 1, total, fo) != total || fclose(fo)) { perror(outp); return 1; }
    printf("%dx%d %s%s, %d level%s -> %zu bytes\n", w, h, NAMES[f], srgb ? " sRGB" : "", levels, levels > 1 ? "s" : "", total + hn);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    const char *mode = argv[1], *dpath = 0, *fmt = "bc1";
    int depth = 64, threads = 8, quality = 70, keyint = 60, i = 2;
    uint32_t fps_num = 30, fps_den = 1;
    size_t bs = 4 << 20, dsize = 16384;
    float rdo = 0;
    int entropy = 0, mips = 0, srgb = 0, fastdec = 0, color = -1;
    for (; i + 1 < argc && argv[i][0] == '-'; i += 2) {
        if (argv[i][1] == 'e' || argv[i][1] == 'm' || argv[i][1] == 'S' || argv[i][1] == 'x') { /* flags without a value */
            if (argv[i][1] == 'e') entropy = 1; else if (argv[i][1] == 'm') mips = 1; else if (argv[i][1] == 'x') fastdec = 1; else srgb = 1;
            i--; continue;
        }
        switch (argv[i][1]) {
        case 'l': depth = atoi(argv[i + 1]); break;
        case 'b': bs = (size_t)atoi(argv[i + 1]) << 10; break;
        case 't': threads = atoi(argv[i + 1]); break;
        case 's': dsize = (size_t)atoi(argv[i + 1]); break;
        case 'D': dpath = argv[i + 1]; break;
        case 'q': quality = atoi(argv[i + 1]); break;
        case 'k': keyint = atoi(argv[i + 1]); break;
        case 'F': { /* 30, 29.97 or 30000/1001 */
            const char *v = argv[i + 1], *sl = strchr(v, '/');
            if (sl) { fps_num = (uint32_t)atoi(v); fps_den = (uint32_t)atoi(sl + 1); }
            else { double x = atof(v); fps_num = (uint32_t)(x * 1000 + 0.5); fps_den = 1000; }
            if (!fps_num || !fps_den) return usage();
            break;
        }
        case 'f': fmt = argv[i + 1]; break;
        case 'r': rdo = (float)atof(argv[i + 1]); break;
        case 'C': /* 601, 709, 601full, 709full */
            color = (strstr(argv[i + 1], "709") ? ZAP_VID_BT709 : 0) | (strstr(argv[i + 1], "full") ? ZAP_VID_FULL_RANGE : 0);
            break;
        default: return usage();
        }
    }
    if (!strcmp(mode, "tex")) return i + 4 == argc ? tex(atoi(argv[i]), atoi(argv[i + 1]), fmt, rdo, mips, srgb, argv[i + 2], argv[i + 3]) : usage();
    if (!strcmp(mode, "venc"))
        return i + 4 == argc ? venc(atoi(argv[i]), atoi(argv[i + 1]), quality, keyint, fps_num, fps_den, depth > 16 ? 16 : depth, threads, color, argv[i + 2], argv[i + 3]) : usage();
    if (!strcmp(mode, "vdec")) return i + 2 == argc ? vdec(argv[i], argv[i + 1], threads) : usage();
    if (!strcmp(mode, "pak"))
        return i + 2 <= argc ? pak(argv[i], argv + i + 1, argc - i - 1, bs, depth | (entropy ? ZAP_ENTROPY : 0) | (fastdec ? ZAP_FAST_DECODE : 0), threads) : usage();
    if (!strcmp(mode, "unpak")) return i + 2 == argc ? unpak(argv[i], argv[i + 1]) : usage();
    if (!strcmp(mode, "ls")) return i + 1 == argc ? ls(argv[i]) : usage();

    static zap_dict dict;
    const zap_dict *d = 0;
    if (dpath) { size_t dn; uint8_t *dd = load(dpath, &dn); zap_dict_init(&dict, dd, dn); d = &dict; }

    if (!strcmp(mode, "train")) {
        if (i + 2 > argc) return usage();
        size_t total = 0, n;
        uint8_t *all = 0, *out = malloc(dsize ? dsize : 1);
        for (int k = i + 1; k < argc; k++) { /* concatenate every sample file */
            uint8_t *p = load(argv[k], &n);
            if (!(all = realloc(all, total + n + 1))) { fprintf(stderr, "out of memory\n"); return 1; }
            memcpy(all + total, p, n); total += n; free(p);
        }
        if (!out) { fprintf(stderr, "out of memory\n"); return 1; }
        size_t got = zap_dict_train(all, total, out, dsize);
        save(argv[i], out, got);
        printf("%zu bytes of samples -> %zu byte dict\n", total, got);
        return 0;
    }
    if (i + 2 != argc) return usage();
    size_t n;
    uint8_t *in = load(argv[i], &n);
    if (!strcmp(mode, "c")) {
        size_t cap = zap_frame_bound(n, bs ? bs : 1);
        uint8_t *out = malloc(cap);
        size_t cn = out ? zap_frame_compress_mt(in, n, out, cap, bs, depth | (entropy ? ZAP_ENTROPY : 0) | (fastdec ? ZAP_FAST_DECODE : 0), d, threads) : 0;
        if (!cn) { fprintf(stderr, "compress failed (bad block size or out of memory)\n"); return 1; }
        save(argv[i + 1], out, cn);
        printf("%zu -> %zu (%.3fx)\n", n, cn, (double)n / cn);
    } else if (!strcmp(mode, "d")) {
        zap_frame f;
        if (zap_frame_open(&f, in, n)) { fprintf(stderr, "not a zap frame\n"); return 1; }
        uint8_t *out = malloc(f.raw + 1);
        if (!out || zap_frame_decode_mt(in, n, out, f.raw, d, threads) < 0) { fprintf(stderr, "corrupt input\n"); return 1; }
        save(argv[i + 1], out, f.raw);
    } else {
        return usage();
    }
    return 0;
}
