/* zap.c - command line tool.
 *   zap c [-e] [-l depth] [-b block_kb] [-t threads] [-D dict] in out   pack (depth 0 = fast, default 64; -e entropy mode)
 *   zap d [-t threads] [-D dict] in out                            unpack
 *   zap train [-s dict_bytes] out samples...                       train a packet dictionary (default 16384)
 *   zap tex [-f bc1|bc3|bc4|bc5|bc7] [-r rdo] w h in.rgba out.dds  GPU block-compress raw RGBA8 to DDS
 *   zap venc [-q quality] [-k keyint] [-F fps] w h in.yuv out.zv   encode raw I420 video
 *   zap vdec in.zv out.yuv                                         decode to raw I420
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#define ZAP_THREADS
#include "zap.h"
#include "zap_tex.h"
#include "zap_video.h"
#include <stdio.h>

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
    fprintf(stderr, "zap c [-e] [-l depth] [-b block_kb] [-t threads] [-D dict] in out\n"
                    "zap d [-t threads] [-D dict] in out\n"
                    "zap train [-s bytes] out samples...\n"
                    "zap tex [-f bc1|bc3|bc4|bc5|bc7] [-r rdo] w h in.rgba out.dds\n"
                    "zap venc [-q quality] [-k keyint] [-F fps] w h in.yuv out.zv\n"
                    "zap vdec in.zv out.yuv\n");
    return 1;
}

/* .zv container: "ZAPV" u16 w u16 h u32 fps, then per frame: u32 packet size + packet */
static int venc(int w, int h, int quality, int keyint, int fps, int depth, const char *in, const char *outp) {
    FILE *fi = fopen(in, "rb"), *fo = create(outp);
    if (!fi) { perror(in); return 1; }
    zap_video *e = zap_venc_create(w, h, quality, keyint, depth);
    if (!e) { fprintf(stderr, "bad size (width and height must be even)\n"); return 1; }
    size_t fsz = (size_t)w * h * 3 / 2, cap = zap_video_bound(e), total = 0, nf = 0;
    uint8_t *frame = malloc(fsz), *pkt = malloc(cap), hdr[12] = { 'Z', 'A', 'P', 'V' };
    hdr[4] = (uint8_t)w; hdr[5] = (uint8_t)(w >> 8); hdr[6] = (uint8_t)h; hdr[7] = (uint8_t)(h >> 8); zap__w32(hdr + 8, (uint32_t)fps);
    fwrite(hdr, 1, sizeof hdr, fo);
    while (fread(frame, 1, fsz, fi) == fsz) {
        size_t n = zap_venc_frame(e, frame, frame + (size_t)w * h, frame + (size_t)w * h * 5 / 4, w, w / 2, pkt, cap);
        uint8_t len[4];
        zap__w32(len, (uint32_t)n);
        if (!n || fwrite(len, 1, 4, fo) != 4 || fwrite(pkt, 1, n, fo) != n) { fprintf(stderr, "encode/write failed\n"); return 1; }
        total += n + 4; nf++;
    }
    if (fclose(fo)) { perror(outp); return 1; }
    printf("%zu frames -> %zu bytes (%.0f kbit/s at %d fps)\n", nf, total, nf ? total * 8.0 * fps / nf / 1000 : 0.0, fps);
    zap_video_destroy(e);
    return 0;
}

static int vdec(const char *in, const char *outp) {
    size_t n, pos = 12, nf = 0;
    uint8_t *f = load(in, &n);
    if (n < 12 || memcmp(f, "ZAPV", 4)) { fprintf(stderr, "not a .zv file\n"); return 1; }
    int w = f[4] | f[5] << 8, h = f[6] | f[7] << 8;
    zap_video *d = zap_vdec_create(w, h);
    if (!d) { fprintf(stderr, "bad header\n"); return 1; }
    FILE *fo = create(outp);
    while (n - pos >= 4) {
        size_t len = zap__r32(f + pos);
        pos += 4;
        if (len > n - pos || zap_vdec_frame(d, f + pos, len)) { fprintf(stderr, "corrupt frame %zu\n", nf); return 1; }
        pos += len; nf++;
        const uint8_t *p[3]; int ys, cs;
        zap_video_planes(d, &p[0], &p[1], &p[2], &ys, &cs);
        for (int pl = 0; pl < 3; pl++)
            for (int y = 0; y < (pl ? h / 2 : h); y++) fwrite(p[pl] + (size_t)y * (pl ? cs : ys), 1, (size_t)(pl ? w / 2 : w), fo);
    }
    if (fclose(fo)) { perror(outp); return 1; }
    printf("%zu frames, %dx%d\n", nf, w, h);
    zap_video_destroy(d);
    return 0;
}

static int tex(int w, int h, const char *fmt, float rdo, const char *in, const char *outp) {
    static const char *NAMES[] = { "bc1", "bc3", "bc4", "bc5", "bc7" }, *CC[] = { "DXT1", "DXT5", "BC4U", "BC5U", "DX10" };
    int f = 0;
    while (f < 5 && strcmp(fmt, NAMES[f])) f++;
    if (f == 5 || w < 1 || h < 1) return usage();
    size_t n, bn = zap_bc_size(w, h, (zap_bc_format)f), hn = f == ZAP_BC7 ? 148 : 128;
    uint8_t *img = load(in, &n), hdr[148] = { 'D', 'D', 'S', ' ' };
    if (n < (size_t)w * h * 4) { fprintf(stderr, "%s: expected %dx%d RGBA8\n", in, w, h); return 1; }
    uint8_t *bc = malloc(bn);
    zap_bc_encode(img, w, h, (size_t)w * 4, (zap_bc_format)f, rdo, bc);
    zap__w32(hdr + 4, 124); zap__w32(hdr + 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000);
    zap__w32(hdr + 12, (uint32_t)h); zap__w32(hdr + 16, (uint32_t)w); zap__w32(hdr + 20, (uint32_t)bn);
    zap__w32(hdr + 76, 32); zap__w32(hdr + 80, 0x4); memcpy(hdr + 84, CC[f], 4); zap__w32(hdr + 108, 0x1000);
    zap__w32(hdr + 128, 98); zap__w32(hdr + 132, 3); zap__w32(hdr + 140, 1); /* DX10 header: BC7_UNORM, 2D, 1 element */
    FILE *fo = create(outp);
    if (fwrite(hdr, 1, hn, fo) != hn || fwrite(bc, 1, bn, fo) != bn || fclose(fo)) { perror(outp); return 1; }
    printf("%dx%d %s -> %zu bytes\n", w, h, NAMES[f], bn + hn);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    const char *mode = argv[1], *dpath = 0, *fmt = "bc1";
    int depth = 64, threads = 8, quality = 70, keyint = 60, fps = 30, i = 2;
    size_t bs = 4 << 20, dsize = 16384;
    float rdo = 0;
    int entropy = 0;
    for (; i + 1 < argc && argv[i][0] == '-'; i += 2) {
        if (argv[i][1] == 'e') { entropy = 1; i--; continue; } /* flag without a value */
        switch (argv[i][1]) {
        case 'l': depth = atoi(argv[i + 1]); break;
        case 'b': bs = (size_t)atoi(argv[i + 1]) << 10; break;
        case 't': threads = atoi(argv[i + 1]); break;
        case 's': dsize = (size_t)atoi(argv[i + 1]); break;
        case 'D': dpath = argv[i + 1]; break;
        case 'q': quality = atoi(argv[i + 1]); break;
        case 'k': keyint = atoi(argv[i + 1]); break;
        case 'F': fps = atoi(argv[i + 1]); break;
        case 'f': fmt = argv[i + 1]; break;
        case 'r': rdo = (float)atof(argv[i + 1]); break;
        default: return usage();
        }
    }
    if (!strcmp(mode, "tex")) return i + 4 == argc ? tex(atoi(argv[i]), atoi(argv[i + 1]), fmt, rdo, argv[i + 2], argv[i + 3]) : usage();
    if (!strcmp(mode, "venc"))
        return i + 4 == argc ? venc(atoi(argv[i]), atoi(argv[i + 1]), quality, keyint, fps, depth > 16 ? 16 : depth, argv[i + 2], argv[i + 3]) : usage();
    if (!strcmp(mode, "vdec")) return i + 2 == argc ? vdec(argv[i], argv[i + 1]) : usage();

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
        size_t cn = out ? zap_frame_compress_mt(in, n, out, cap, bs, depth | (entropy ? ZAP_ENTROPY : 0), d, threads) : 0;
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
