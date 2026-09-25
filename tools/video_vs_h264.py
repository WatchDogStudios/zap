#!/usr/bin/env python3
"""zap video vs H.264: same frames, same bitrate, then decode speed on one thread and on all threads.

    python tools/video_vs_h264.py clip.mp4 [--zap build/Release/zap] [--frames 300] [--quality 70] [--rounds 5]

Needs ffmpeg (with libx264) on PATH and a built `zap` CLI. The clip is decoded once to raw I420; zap encodes it
at --quality, x264 (two-pass, preset medium) encodes it at zap's bitrate, and PSNR is measured by ffmpeg for
both. Decode timing alternates zap and H.264 for --rounds rounds and keeps each one's best, so background load
affects both alike. ffmpeg's number includes demuxing; zap's is decode only.
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, time


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout + r.stderr


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path])
    w, h, rate = out.strip().split("\n")[0].split(",")[:3]
    return int(w), int(h), rate


def psnr(ref_yuv, test, w, h):
    if not test.endswith(".yuv"):  # compare raw against raw, so frames pair up by position rather than timestamp
        run(["ffmpeg", "-hide_banner", "-y", "-i", test, "-f", "rawvideo", "-pix_fmt", "yuv420p", test + ".yuv"])
        test += ".yuv"
    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30"]
    out = run(["ffmpeg", "-hide_banner", *raw, "-i", test, *raw, "-i", ref_yuv, "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
    m = re.search(r"y:([\d.inf]+).*average:([\d.inf]+)", out)
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--zap", default=os.path.join("build", "Release", "zap.exe") if os.name == "nt" else "build/zap")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--preset", default="medium")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    w, h, rate = probe(a.clip)
    w &= ~1; h &= ~1
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den)
    threads = os.cpu_count() or 1
    tmp = tempfile.mkdtemp(prefix="zapvs")
    raw, zv, mkv = (os.path.join(tmp, n) for n in ("src.yuv", "clip.zapvid", "clip.mkv"))
    run(["ffmpeg", "-hide_banner", "-y", "-i", a.clip, "-frames:v", str(a.frames), "-vf", f"crop={w}:{h}:0:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", raw])
    n = os.path.getsize(raw) // (w * h * 3 // 2)
    secs = n / fps
    out = run([a.zap, "venc", "-q", str(a.quality), "-F", f"{num}/{den}", "-t", str(threads), str(w), str(h), raw, zv])
    zenc = float(re.search(r"encoded at ([\d.]+) fps", out).group(1))
    kbit = os.path.getsize(zv) * 8 / secs / 1000
    passlog = os.path.join(tmp, "x264")
    for p in (1, 2):
        t0 = time.perf_counter()
        run(["ffmpeg", "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", f"{num}/{den}", "-i", raw,
             "-c:v", "libx264", "-preset", a.preset, "-b:v", f"{kbit:.0f}k", "-pass", str(p), "-passlogfile", passlog,
             *(["-f", "null", os.devnull] if p == 1 else [mkv])])
        henc = n / (time.perf_counter() - t0)  # the second pass: what a one-pass encode at this preset costs
    hkbit = os.path.getsize(mkv) * 8 / secs / 1000
    zdec_yuv = os.path.join(tmp, "zap.yuv")
    run([a.zap, "vdec", zv, zdec_yuv])
    zy, zall = psnr(raw, zdec_yuv, w, h)
    hy, hall = psnr(raw, mkv, w, h)
    best = {}
    for _ in range(a.rounds):  # interleaved, best of each
        for t in (1, threads):
            o = run([a.zap, "vdec", "-t", str(t), zv, "null"])
            f = float(re.search(r"decoded at ([\d.]+) fps", o).group(1))
            best[("zap", t)] = max(best.get(("zap", t), 0), f)
            o = run(["ffmpeg", "-hide_banner", "-threads", str(0 if t > 1 else 1), "-benchmark", "-i", mkv, "-f", "null", "-"])
            f = n / float(re.search(r"rtime=([\d.]+)s", o).group(1))
            best[("h264", t)] = max(best.get(("h264", t), 0), f)
    print(f"{os.path.basename(a.clip)}: {w}x{h}, {n} frames at {fps:.3f} fps, {threads} hardware threads\n")
    print(f"| Codec | kbit/s | PSNR-Y | PSNR-YUV | Decode fps, 1 thread | Decode fps, {threads} threads |")
    print("|---|---|---|---|---|---|")
    print(f"| zap q{a.quality} | {kbit:.0f} | {zy:.2f} dB | {zall:.2f} dB | {best[('zap', 1)]:.0f} | {best[('zap', threads)]:.0f} |")
    print(f"| H.264 (x264 {a.preset}) | {hkbit:.0f} | {hy:.2f} dB | {hall:.2f} dB | {best[('h264', 1)]:.0f} | {best[('h264', threads)]:.0f} |")
    print(f"\nzap decodes {best[('zap', 1)] / best[('h264', 1)]:.1f}x faster on one thread and "
          f"{best[('zap', threads)] / best[('h264', threads)]:.1f}x faster on all threads.\n"
          f"Encode ({threads} threads): zap {zenc:.0f} fps, x264 {a.preset} {henc:.0f} fps (second pass, includes reading the raw input).")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
