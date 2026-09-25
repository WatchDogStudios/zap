// zap D3D11 + Dear ImGui showcase.
//
//   zap_viewer [image | video]      GUI. Texture tab: split view of two encodings of an image (original, BC1,
//                                   BC3, BC7, each with RDO) with quality / size / speed stats. Video tab: any file
//                                   Media Foundation can decode (MP4, MOV, MKV, ...) is transcoded live through zap;
//                                   split view source vs zap with bitrate, PSNR and decode-time stats.
//   zap_viewer --verify             GPU conformance: decode every BC format (BC1-BC7, BC6H) on the GPU and diff against zap
//   zap_viewer --shot out [image] [--video file]   render out_texture(_zoom|_diff).png and out_video.png, then exit
//
// Mouse: wheel zoom, right-drag pan, left-drag moves the split line. Drag & drop files onto the window.
#define _CRT_SECURE_NO_WARNINGS
#define ZAP_THREADS
#include "../../zap.h"
#include "../../zap_tex.h"
#include "../../zap_video.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wincodec.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <wrl/client.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { fprintf(stderr, "%s failed: 0x%08lx (line %d)\n", #x, (unsigned long)hr_, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- shaders

static const char kShaders[] = R"(
cbuffer C : register(b0) { float4 xf; float4 params; float4 yuv; } // xf: screen uv -> image uv; params: split, diff; yuv: bt709, full range
Texture2D T0 : register(t0); Texture2D T1 : register(t1); Texture2D T2 : register(t2);
Texture2D T3 : register(t3); Texture2D T4 : register(t4); Texture2D T5 : register(t5);
SamplerState S : register(s0);
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD; };
V vs(uint id : SV_VertexID) {
    V o; o.uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
static const float4 kBack = float4(0.08, 0.08, 0.09, 1);
float4 pick(float2 uv, float3 a, float3 b) {
    if (params.y > 0.5) return float4(saturate(abs(a - b) * 8), 1);
    if (abs(uv.x - params.x) < 0.0012) return float4(1, 0.75, 0.1, 1);
    return float4(uv.x < params.x ? a : b, 1);
}
float4 ps_tex(V i) : SV_Target {
    float2 t = i.uv * xf.xy + xf.zw;
    if (any(t < 0) || any(t > 1)) return kBack;
    return pick(i.uv, T0.Sample(S, t).rgb, T1.Sample(S, t).rgb);
}
float3 rgb(float y, float u, float v) {
    if (yuv.y < 0.5) { y = (y - 16.0 / 255) * 1.164; u = (u - 0.5) * 1.138; v = (v - 0.5) * 1.138; } else { u -= 0.5; v -= 0.5; }
    return saturate(yuv.x > 0.5 ? float3(y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u)
                                : float3(y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u));
}
float4 ps_video(V i) : SV_Target {
    float2 t = i.uv * xf.xy + xf.zw;
    if (any(t < 0) || any(t > 1)) return kBack;
    return pick(i.uv, rgb(T0.Sample(S, t).r, T1.Sample(S, t).r, T2.Sample(S, t).r), rgb(T3.Sample(S, t).r, T4.Sample(S, t).r, T5.Sample(S, t).r));
}
float4 ps_load(V i) : SV_Target { return T0.Load(int3(i.pos.xy, 0)); }
)";

struct Gfx {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<IDXGISwapChain> swap;
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps_tex, ps_video, ps_load;
    ComPtr<ID3D11Buffer> cb;
    ComPtr<ID3D11SamplerState> point, linear;
};

static ComPtr<ID3DBlob> compile(const char *entry, const char *target) {
    ComPtr<ID3DBlob> code, err;
    if (FAILED(D3DCompile(kShaders, sizeof kShaders - 1, "viewer.hlsl", nullptr, nullptr, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        fprintf(stderr, "shader %s: %s\n", entry, err ? (const char *)err->GetBufferPointer() : "?");
        exit(1);
    }
    return code;
}

static void init_pipeline(Gfx &g) {
    auto v = compile("vs", "vs_5_0");
    CHECK(g.dev->CreateVertexShader(v->GetBufferPointer(), v->GetBufferSize(), nullptr, &g.vs));
    const char *names[3] = { "ps_tex", "ps_video", "ps_load" };
    ComPtr<ID3D11PixelShader> *outs[3] = { &g.ps_tex, &g.ps_video, &g.ps_load };
    for (int i = 0; i < 3; i++) {
        auto p = compile(names[i], "ps_5_0");
        CHECK(g.dev->CreatePixelShader(p->GetBufferPointer(), p->GetBufferSize(), nullptr, outs[i]->GetAddressOf()));
    }
    D3D11_BUFFER_DESC bd = { 48, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
    CHECK(g.dev->CreateBuffer(&bd, nullptr, &g.cb));
    D3D11_SAMPLER_DESC sd = {};
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    CHECK(g.dev->CreateSamplerState(&sd, &g.point));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    CHECK(g.dev->CreateSamplerState(&sd, &g.linear));
}

static ComPtr<ID3D11ShaderResourceView> make_tex(ID3D11Device *dev, DXGI_FORMAT fmt, int w, int h, const void *data, UINT pitch,
                                                 ComPtr<ID3D11Texture2D> *tex_out = nullptr) {
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = (UINT)w; td.Height = (UINT)h; td.MipLevels = 1; td.ArraySize = 1; td.Format = fmt; td.SampleDesc.Count = 1;
    td.Usage = tex_out ? D3D11_USAGE_DEFAULT : D3D11_USAGE_IMMUTABLE;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = { data, pitch, 0 };
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11ShaderResourceView> srv;
    CHECK(dev->CreateTexture2D(&td, data ? &sd : nullptr, &tex));
    CHECK(dev->CreateShaderResourceView(tex.Get(), nullptr, &srv));
    if (tex_out) *tex_out = tex;
    return srv;
}

static void draw(Gfx &g, ID3D11PixelShader *ps, ID3D11ShaderResourceView *const *srvs, int nsrv, const float cb[12], int w, int h,
                 ID3D11RenderTargetView *rtv, ID3D11SamplerState *smp) {
    D3D11_VIEWPORT vp = { 0, 0, (float)w, (float)h, 0, 1 };
    g.ctx->UpdateSubresource(g.cb.Get(), 0, nullptr, cb, 0, 0);
    g.ctx->OMSetRenderTargets(1, &rtv, nullptr);
    g.ctx->RSSetViewports(1, &vp);
    g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g.ctx->IASetInputLayout(nullptr);
    g.ctx->VSSetShader(g.vs.Get(), nullptr, 0);
    g.ctx->PSSetShader(ps, nullptr, 0);
    g.ctx->PSSetConstantBuffers(0, 1, g.cb.GetAddressOf());
    g.ctx->PSSetShaderResources(0, (UINT)nsrv, srvs);
    g.ctx->PSSetSamplers(0, 1, &smp);
    g.ctx->Draw(3, 0);
}

static const DXGI_FORMAT kDxgi[5] = { DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC7_UNORM };
static UINT bc_pitch(int w, zap_bc_format f) { return (UINT)((w + 3) / 4 * zap_bc_block_bytes(f)); }

// ---------------------------------------------------------------- --verify (hardware GPU)

static int verify() {
    Gfx g;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    CHECK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &g.dev, nullptr, &g.ctx));
    init_pipeline(g);
    {
        ComPtr<IDXGIDevice> dx; ComPtr<IDXGIAdapter> ad; DXGI_ADAPTER_DESC d;
        g.dev.As(&dx); dx->GetAdapter(&ad); ad->GetDesc(&d);
        printf("GPU: %ls\n", d.Description);
    }
    const int W = 256, H = 256;
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> rt, staging;
    ComPtr<ID3D11RenderTargetView> rtv;
    CHECK(g.dev->CreateTexture2D(&td, nullptr, &rt));
    CHECK(g.dev->CreateRenderTargetView(rt.Get(), nullptr, &rtv));
    td.BindFlags = 0; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    CHECK(g.dev->CreateTexture2D(&td, nullptr, &staging));

    std::vector<uint8_t> img(W * H * 4), ours(W * H * 4), gpu(W * H * 4);
    srand(11);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { // gradients, hard edges, noise, alpha
        uint8_t *p = &img[(y * W + x) * 4];
        bool edge = ((x / 13) + (y / 17)) & 1;
        p[0] = (uint8_t)(edge ? x : 255 - y); p[1] = (uint8_t)((x * y) >> 8); p[2] = (uint8_t)(rand() % 256 * (y > 192) + (y <= 192) * (x ^ y));
        p[3] = (uint8_t)(x < 128 ? 255 : (y * 3 + (rand() & 15)));
    }
    auto gpu_decode = [&](ID3D11ShaderResourceView *srv) {
        float cb[12] = { 1, 1, 0, 0 };
        draw(g, g.ps_load.Get(), &srv, 1, cb, W, H, rtv.Get(), g.point.Get());
        g.ctx->CopyResource(staging.Get(), rt.Get());
        D3D11_MAPPED_SUBRESOURCE m;
        CHECK(g.ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m));
        for (int y = 0; y < H; y++) memcpy(&gpu[y * W * 4], (uint8_t *)m.pData + y * m.RowPitch, W * 4);
        g.ctx->Unmap(staging.Get(), 0);
    };
    auto compare = [&](const char *what, int tolerance, int chmask) {
        int maxd = 0; size_t bad = 0;
        for (int i = 0; i < W * H * 4; i++) {
            if (!(chmask >> (i & 3) & 1)) continue;
            int d = abs(gpu[i] - ours[i]);
            if (d > maxd) maxd = d;
            bad += d > tolerance;
        }
        printf("  %-26s max |gpu - zap| = %3d  %s\n", what, maxd, bad ? "FAIL" : "ok");
        return bad ? 1 : 0;
    };
    static const char *names[5] = { "BC1", "BC3", "BC4", "BC5", "BC7" };
    static const int kMask[5] = { 7, 15, 1, 3, 15 };
    int fails = 0;
    for (int f = 0; f < 5; f++)
        for (float rdo : { 0.0f, 30.0f }) {
            std::vector<uint8_t> bc(zap_bc_size(W, H, (zap_bc_format)f));
            zap_bc_encode(img.data(), W, H, W * 4, (zap_bc_format)f, rdo, bc.data());
            zap_bc_decode(bc.data(), W, H, (zap_bc_format)f, ours.data(), W * 4);
            gpu_decode(make_tex(g.dev.Get(), kDxgi[f], W, H, bc.data(), bc_pitch(W, (zap_bc_format)f)).Get());
            char what[64];
            snprintf(what, sizeof what, "%s encoder, rdo %.0f", names[f], rdo);
            fails += compare(what, f == ZAP_BC7 ? 0 : 2, kMask[f]); // BC1-5 interpolation rounding is implementation-defined
        }
    std::vector<uint8_t> bc(W * H); // BC7 decoder on random blocks of every mode (all partitions / rotations / index modes)
    for (int m = 0; m <= 8; m++) {
        for (size_t i = 0; i < bc.size(); i++) bc[i] = (uint8_t)rand();
        for (size_t b = 0; b < bc.size(); b += 16) bc[b] = m == 8 ? 0 : (uint8_t)((bc[b] << (m + 1)) | (1 << m));
        zap_bc_decode(bc.data(), W, H, ZAP_BC7, ours.data(), W * 4);
        gpu_decode(make_tex(g.dev.Get(), DXGI_FORMAT_BC7_UNORM, W, H, bc.data(), bc_pitch(W, ZAP_BC7)).Get());
        char what[64];
        snprintf(what, sizeof what, m == 8 ? "BC7 decoder, reserved mode" : "BC7 decoder, random mode %d", m);
        fails += compare(what, 0, 15);
    }
    { // BC6H (unsigned half floats): zap's mode-11 encoder output, then random mode-11 blocks; exact match expected
        D3D11_TEXTURE2D_DESC hd = {};
        hd.Width = W; hd.Height = H; hd.MipLevels = 1; hd.ArraySize = 1; hd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; hd.SampleDesc.Count = 1;
        hd.BindFlags = D3D11_BIND_RENDER_TARGET;
        ComPtr<ID3D11Texture2D> hrt, hst;
        ComPtr<ID3D11RenderTargetView> hrtv;
        CHECK(g.dev->CreateTexture2D(&hd, nullptr, &hrt));
        CHECK(g.dev->CreateRenderTargetView(hrt.Get(), nullptr, &hrtv));
        hd.BindFlags = 0; hd.Usage = D3D11_USAGE_STAGING; hd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        CHECK(g.dev->CreateTexture2D(&hd, nullptr, &hst));
        std::vector<float> hdr(W * H * 4);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            float *p = &hdr[(y * W + x) * 4];
            p[0] = 0.02f * (float)(1 << (x / 20)); p[1] = (float)((x * y) % 97) * 0.37f; p[2] = ((x / 13 + y / 17) & 1) ? 4000.0f * y / H : 0.001f * x; p[3] = 1;
        }
        std::vector<uint16_t> zh(W * H * 4), gh(W * H * 4);
        std::vector<uint8_t> b6(zap_bc_size(W, H, ZAP_BC7));
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 0) zap_bc6h_encode(hdr.data(), W, H, W * 4, b6.data());
            else { for (auto &x : b6) x = (uint8_t)rand(); for (size_t b = 0; b < b6.size(); b += 16) b6[b] = (uint8_t)((b6[b] & 0xE0) | 3); }
            zap_bc6h_decode(b6.data(), W, H, zh.data(), W * 4);
            auto tex6 = make_tex(g.dev.Get(), DXGI_FORMAT_BC6H_UF16, W, H, b6.data(), bc_pitch(W, ZAP_BC7));
            ID3D11ShaderResourceView *srv = tex6.Get();
            float cb[12] = { 1, 1, 0, 0 };
            draw(g, g.ps_load.Get(), &srv, 1, cb, W, H, hrtv.Get(), g.point.Get());
            g.ctx->CopyResource(hst.Get(), hrt.Get());
            D3D11_MAPPED_SUBRESOURCE m;
            CHECK(g.ctx->Map(hst.Get(), 0, D3D11_MAP_READ, 0, &m));
            for (int y = 0; y < H; y++) memcpy(&gh[y * W * 4], (uint8_t *)m.pData + y * m.RowPitch, W * 8);
            g.ctx->Unmap(hst.Get(), 0);
            int maxd = 0; size_t bad = 0;
            for (int i = 0; i < W * H * 4; i++) { int d = abs((int)gh[i] - (int)zh[i]); if (d > maxd) maxd = d; bad += d != 0; }
            printf("  %-26s max |gpu - zap| = %3d half ulp  %s\n", pass ? "BC6H decoder, random mode 11" : "BC6H encoder (mode 11)", maxd, bad ? "FAIL" : "ok");
            fails += bad ? 1 : 0;
        }
    }
    printf(fails ? "FAILED (%d)\n" : "all formats match the GPU\n", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------- files

static bool load_image(const std::wstring &path, std::vector<uint8_t> &rgba, int &w, int &h) {
    ComPtr<IWICImagingFactory> f;
    ComPtr<IWICBitmapDecoder> dec;
    ComPtr<IWICBitmapFrameDecode> fr;
    ComPtr<IWICFormatConverter> cv;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) return false;
    if (FAILED(f->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) return false;
    if (FAILED(dec->GetFrame(0, &fr)) || FAILED(f->CreateFormatConverter(&cv))) return false;
    if (FAILED(cv->Initialize(fr.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) return false;
    UINT W, H;
    cv->GetSize(&W, &H);
    rgba.resize((size_t)W * H * 4);
    if (FAILED(cv->CopyPixels(nullptr, W * 4, (UINT)rgba.size(), rgba.data()))) return false;
    w = (int)W; h = (int)H;
    return true;
}

static bool save_png(const std::wstring &path, const uint8_t *rgba, int w, int h) {
    ComPtr<IWICImagingFactory> f;
    ComPtr<IWICStream> st;
    ComPtr<IWICBitmapEncoder> enc;
    ComPtr<IWICBitmapFrameEncode> fr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) return false;
    if (FAILED(f->CreateStream(&st)) || FAILED(st->InitializeFromFilename(path.c_str(), GENERIC_WRITE))) return false;
    if (FAILED(f->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) || FAILED(enc->Initialize(st.Get(), WICBitmapEncoderNoCache))) return false;
    if (FAILED(enc->CreateNewFrame(&fr, nullptr)) || FAILED(fr->Initialize(nullptr)) || FAILED(fr->SetSize((UINT)w, (UINT)h))) return false;
    WICPixelFormatGUID pf = GUID_WICPixelFormat32bppBGRA; // what the PNG encoder takes
    if (FAILED(fr->SetPixelFormat(&pf)) || pf != GUID_WICPixelFormat32bppBGRA) return false;
    std::vector<uint8_t> bgra(rgba, rgba + (size_t)w * h * 4);
    for (size_t i = 0; i < bgra.size(); i += 4) std::swap(bgra[i], bgra[i + 2]);
    if (FAILED(fr->WritePixels((UINT)h, (UINT)w * 4, (UINT)bgra.size(), bgra.data()))) return false;
    return SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
}

static std::wstring pick_file(HWND owner, bool video) {
    ComPtr<IFileOpenDialog> d;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&d)))) return {};
    COMDLG_FILTERSPEC f[2] = { { L"Images", L"*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.gif;*.webp;*.heic;*.jxr" }, { L"All files", L"*.*" } };
    if (video) f[0] = { L"Video", L"*.mp4;*.m4v;*.mov;*.mkv;*.webm;*.avi;*.wmv;*.ts;*.mts" };
    d->SetFileTypes(2, f);
    ComPtr<IShellItem> it;
    PWSTR p = nullptr;
    if (FAILED(d->Show(owner)) || FAILED(d->GetResult(&it)) || FAILED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) return {};
    std::wstring s = p;
    CoTaskMemFree(p);
    return s;
}

static bool is_video_path(const std::wstring &p) {
    static const wchar_t *ext[] = { L".mp4", L".m4v", L".mov", L".mkv", L".webm", L".avi", L".wmv", L".ts", L".mts" };
    for (const wchar_t *e : ext) { size_t n = wcslen(e); if (p.size() > n && _wcsicmp(p.c_str() + p.size() - n, e) == 0) return true; }
    return false;
}

static std::string file_name(const std::wstring &w) {
    std::string s(WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr), 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), (int)s.size(), nullptr, nullptr);
    if (!s.empty()) s.pop_back();
    size_t slash = s.find_last_of("\\/");
    return slash == std::string::npos ? s : s.substr(slash + 1);
}

// ---------------------------------------------------------------- texture tab

static const char *kTexFmt[4] = { "Original RGBA8", "BC1", "BC3", "BC7" };
static const zap_bc_format kTexBC[4] = { ZAP_BC1, ZAP_BC1, ZAP_BC3, ZAP_BC7 };

struct TexResult { int gen = 0, w = 0, h = 0, fmt = 0; std::vector<uint8_t> data; double psnr = 0, ms = 0; size_t gpu = 0, disk = 0; };

struct TexSide {
    int fmt = 3;
    float rdo = 0;
    int want_fmt = -1; float want_rdo = -1; // settings of the last job started
    int gen = 0;
    bool busy = false;
    TexResult shown;
    ComPtr<ID3D11ShaderResourceView> srv;
    std::shared_ptr<TexResult> ready;
};

// ---------------------------------------------------------------- video tab

struct Series {
    float v[240] = {};
    int n = 0, at = 0;
    void push(float x) { v[at] = x; at = (at + 1) % 240; if (n < 240) n++; }
    float avg(int last) const { float s = 0; int k = last < n ? last : n; for (int i = 0; i < k; i++) s += v[(at - 1 - i + 240) % 240]; return k ? s / k : 0; }
};

struct Video {
    ComPtr<IMFSourceReader> rd;
    std::string name, codec, status;
    int w = 0, h = 0, fh = 0, bt709 = 0, full = 0;
    double fps = 30, src_kbps = 0, duration_s = 0;
    std::vector<uint8_t> y, u, v;              // source frame, I420
    zap_video *enc = nullptr, *dec = nullptr;
    int quality = 70, keyint = 60, packing = 0; // packing: 0 fast, 1 hc 8, 2 hc 16
    bool rebuild = true, paused = false;
    std::vector<uint8_t> pkt;
    ComPtr<ID3D11Texture2D> tex[6];
    ComPtr<ID3D11ShaderResourceView> srv[6];
    Series src_ms, enc_ms, dec_ms, kbps, psnr;
    size_t frames = 0;
    Clock::time_point next;
};

// ---------------------------------------------------------------- app

static struct App {
    Gfx g;
    HWND hwnd = nullptr;
    int ww = 1600, wh = 900, tab = 0, want_tab = -1;
    float zoom = 1, panx = 0, pany = 0, split = 0.5f;
    bool diff = false, linear = false, drag_split = false, drag_pan = false;
    int mx = 0, my = 0;
    // texture
    std::string img_name;
    int iw = 0, ih = 0;
    std::shared_ptr<const std::vector<uint8_t>> img;
    TexSide side[2];
    std::mutex mtx;
    std::vector<std::thread> jobs;
    // video
    Video vid;
} A;

/* blocks are independent (RDO only looks back ZAP_TEX_WINDOW blocks), so encode horizontal bands in parallel */
static void bc_encode_mt(const uint8_t *img, int w, int h, zap_bc_format f, float rdo, uint8_t *out) {
    int rows = h / 4, n = (int)std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 16u);
    size_t row_bytes = (size_t)(w / 4) * zap_bc_block_bytes(f);
    std::vector<std::thread> th;
    for (int t = 0; t < n; t++) {
        int b0 = rows * t / n, b1 = rows * (t + 1) / n;
        if (b1 > b0) th.emplace_back([=] { zap_bc_encode(img + (size_t)b0 * 4 * w * 4, w, (b1 - b0) * 4, (size_t)w * 4, f, rdo, out + b0 * row_bytes); });
    }
    for (auto &x : th) x.join();
}

static void tex_job(std::shared_ptr<const std::vector<uint8_t>> img, int w, int h, int fmt, float rdo, int gen, int s) {
    auto r = std::make_shared<TexResult>();
    r->gen = gen; r->w = w; r->h = h; r->fmt = fmt;
    zap_hc_state *hc = (zap_hc_state *)malloc(sizeof(zap_hc_state));
    if (fmt == 0) { // original: raw RGBA8, and what plain zap makes of it
        r->data = *img; r->gpu = img->size(); r->psnr = INFINITY;
        std::vector<uint8_t> z(zap_bound(img->size()));
        auto t0 = Clock::now();
        r->disk = zap_compress_hc(img->data(), img->size(), z.data(), z.size(), hc, nullptr, 32);
        r->ms = ms_since(t0);
    } else {
        zap_bc_format f = kTexBC[fmt];
        size_t n = zap_bc_size(w, h, f);
        std::vector<uint8_t> dec((size_t)w * h * 4), planes(n), z(zap_bound(n));
        r->data.resize(n);
        auto t0 = Clock::now();
        bc_encode_mt(img->data(), w, h, f, rdo, r->data.data());
        r->ms = ms_since(t0);
        zap_bc_decode(r->data.data(), w, h, f, dec.data(), (size_t)w * 4);
        double se = 0;
        for (size_t k = 0; k < dec.size(); k++) if ((k & 3) != 3) { double d = (double)dec[k] - (*img)[k]; se += d * d; }
        r->psnr = se > 0 ? 10 * log10(255.0 * 255.0 * (double)w * h * 3 / se) : INFINITY;
        zap_bc_split(r->data.data(), n, f, planes.data());
        r->gpu = n;
        r->disk = zap_compress_hc(planes.data(), n, z.data(), z.size(), hc, nullptr, 32);
    }
    free(hc);
    std::lock_guard<std::mutex> lk(A.mtx);
    if (gen == A.side[s].gen) A.side[s].ready = r;
}

static void tex_update() {
    if (!A.img) return;
    for (int s = 0; s < 2; s++) {
        TexSide &t = A.side[s];
        float rdo = t.fmt ? t.rdo : 0;
        if (t.fmt != t.want_fmt || rdo != t.want_rdo) {
            t.want_fmt = t.fmt; t.want_rdo = rdo; t.busy = true;
            int gen;
            { std::lock_guard<std::mutex> lk(A.mtx); gen = ++t.gen; }
            A.jobs.emplace_back(tex_job, A.img, A.iw, A.ih, t.fmt, rdo, gen, s);
        }
        std::shared_ptr<TexResult> r;
        { std::lock_guard<std::mutex> lk(A.mtx); r.swap(t.ready); }
        if (r) {
            zap_bc_format f = kTexBC[r->fmt];
            t.srv = r->fmt ? make_tex(A.g.dev.Get(), kDxgi[f], r->w, r->h, r->data.data(), bc_pitch(r->w, f))
                           : make_tex(A.g.dev.Get(), DXGI_FORMAT_R8G8B8A8_UNORM, r->w, r->h, r->data.data(), (UINT)r->w * 4);
            r->data.clear(); r->data.shrink_to_fit();
            t.shown = std::move(*r);
            t.busy = t.gen != t.shown.gen;
        }
    }
}

static bool open_image(const std::wstring &path) {
    std::vector<uint8_t> img;
    int w, h;
    if (!load_image(path, img, w, h)) return false;
    while (w > 4096 || h > 4096) { // keep encodes interactive: halve huge images
        int nw = w / 2, nh = h / 2;
        std::vector<uint8_t> o((size_t)nw * nh * 4);
        for (int y = 0; y < nh; y++) for (int x = 0; x < nw; x++) for (int c = 0; c < 4; c++) {
            const uint8_t *p = &img[((size_t)(2 * y) * w + 2 * x) * 4 + c];
            o[((size_t)y * nw + x) * 4 + c] = (uint8_t)((p[0] + p[4] + p[(size_t)w * 4] + p[(size_t)w * 4 + 4] + 2) / 4);
        }
        img.swap(o); w = nw; h = nh;
    }
    int cw = w & ~3, ch = h & ~3; // BC textures need multiple-of-4 sizes: crop
    if (cw < 4 || ch < 4) return false;
    auto crop = std::make_shared<std::vector<uint8_t>>((size_t)cw * ch * 4);
    for (int y = 0; y < ch; y++) memcpy(&(*crop)[(size_t)y * cw * 4], &img[(size_t)y * w * 4], (size_t)cw * 4);
    A.img = crop; A.iw = cw; A.ih = ch; A.img_name = file_name(path);
    for (TexSide &t : A.side) { t.want_fmt = -1; t.srv.Reset(); t.shown = TexResult(); }
    A.zoom = 1; A.panx = A.pany = 0;
    return true;
}

// ---------------------------------------------------------------- video: Media Foundation source -> zap -> display

static std::string codec_name(const GUID &g) {
    struct { const GUID *g; const char *n; } k[] = { { &MFVideoFormat_H264, "H.264" }, { &MFVideoFormat_HEVC, "HEVC" }, { &MFVideoFormat_VP90, "VP9" },
        { &MFVideoFormat_AV1, "AV1" }, { &MFVideoFormat_MPEG2, "MPEG-2" }, { &MFVideoFormat_MP4V, "MPEG-4 Part 2" }, { &MFVideoFormat_WMV3, "WMV9" },
        { &MFVideoFormat_MJPG, "Motion JPEG" }, { &MFVideoFormat_H263, "H.263" }, { &MFVideoFormat_VP80, "VP8" } };
    for (auto &e : k) if (*e.g == g) return e.n;
    char cc[5] = { (char)(g.Data1 & 255), (char)(g.Data1 >> 8 & 255), (char)(g.Data1 >> 16 & 255), (char)(g.Data1 >> 24), 0 };
    return cc;
}

static void video_close() {
    Video &v = A.vid;
    zap_video_destroy(v.enc); zap_video_destroy(v.dec);
    v = Video();
}

static bool open_video(const std::wstring &path) {
    video_close();
    Video &v = A.vid;
    ComPtr<IMFAttributes> attr;
    MFCreateAttributes(&attr, 1);
    attr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE); // lets any decoder's output be converted to NV12
    if (FAILED(MFCreateSourceReaderFromURL(path.c_str(), attr.Get(), &v.rd))) { v.status = "Media Foundation can't open " + file_name(path); return false; }
    v.rd->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
    v.rd->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
    ComPtr<IMFMediaType> native, want, cur;
    if (FAILED(v.rd->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &native))) { v.status = "no video stream"; v.rd.Reset(); return false; }
    GUID sub = {};
    native->GetGUID(MF_MT_SUBTYPE, &sub);
    v.codec = codec_name(sub);
    UINT32 num = 0, den = 0;
    if (SUCCEEDED(MFGetAttributeRatio(native.Get(), MF_MT_FRAME_RATE, &num, &den)) && num && den) v.fps = (double)num / den;
    if (v.fps < 1 || v.fps > 240) v.fps = 30;
    v.src_kbps = MFGetAttributeUINT32(native.Get(), MF_MT_AVG_BITRATE, 0) / 1000.0;
    v.bt709 = MFGetAttributeUINT32(native.Get(), MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709) == MFVideoTransferMatrix_BT709;
    v.full = MFGetAttributeUINT32(native.Get(), MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235) == MFNominalRange_0_255;
    PROPVARIANT pv;
    PropVariantInit(&pv);
    if (SUCCEEDED(v.rd->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &pv))) v.duration_s = pv.uhVal.QuadPart / 1e7;
    PropVariantClear(&pv);
    if (v.src_kbps == 0 && v.duration_s > 0 && SUCCEEDED(v.rd->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE, MF_PD_TOTAL_FILE_SIZE, &pv)))
        v.src_kbps = pv.uhVal.QuadPart * 8.0 / v.duration_s / 1000.0; // whole container (includes audio)
    PropVariantClear(&pv);
    MFCreateMediaType(&want);
    want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
    if (FAILED(v.rd->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, want.Get())) ||
        FAILED(v.rd->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur))) { v.status = "no decoder for " + v.codec + " (codec extension missing?)"; v.rd.Reset(); return false; }
    UINT32 fw = 0, fh = 0;
    MFGetAttributeSize(cur.Get(), MF_MT_FRAME_SIZE, &fw, &fh);
    MFVideoArea area = {};
    int w = (int)fw, h = (int)fh;
    if (SUCCEEDED(cur->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE, (UINT8 *)&area, sizeof area, nullptr)) && area.Area.cx && area.Area.cy) { w = (int)area.Area.cx; h = (int)area.Area.cy; }
    v.w = w & ~1; v.h = h & ~1; v.fh = (int)fh;
    if (v.w < 16 || v.h < 16) { v.status = "video too small"; v.rd.Reset(); return false; }
    v.y.resize((size_t)v.w * v.h); v.u.resize((size_t)v.w * v.h / 4); v.v.resize(v.u.size());
    for (int i = 0; i < 6; i++) v.srv[i] = make_tex(A.g.dev.Get(), DXGI_FORMAT_R8_UNORM, i % 3 ? v.w / 2 : v.w, i % 3 ? v.h / 2 : v.h, nullptr, 0, &v.tex[i]);
    v.dec = zap_vdec_create(v.w, v.h);
    v.name = file_name(path);
    v.next = Clock::now();
    A.zoom = 1; A.panx = A.pany = 0;
    return true;
}

static bool read_source_frame() {
    Video &v = A.vid;
    for (int tries = 0; tries < 64; tries++) {
        DWORD idx, flags = 0;
        LONGLONG ts;
        ComPtr<IMFSample> s;
        if (FAILED(v.rd->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &idx, &flags, &ts, &s))) { v.status = "source decode error"; return false; }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { // loop
            PROPVARIANT p;
            PropVariantInit(&p); p.vt = VT_I8; p.hVal.QuadPart = 0;
            v.rd->SetCurrentPosition(GUID_NULL, p);
            continue;
        }
        if (!s) continue;
        ComPtr<IMFMediaBuffer> buf;
        if (FAILED(s->ConvertToContiguousBuffer(&buf))) return false;
        ComPtr<IMF2DBuffer> b2;
        BYTE *data = nullptr;
        LONG pitch = 0;
        DWORD len = 0;
        bool locked2d = SUCCEEDED(buf.As(&b2)) && SUCCEEDED(b2->Lock2D(&data, &pitch));
        if (!locked2d) { if (FAILED(buf->Lock(&data, nullptr, &len))) return false; pitch = (LONG)(len * 2 / 3 / (DWORD)v.fh); }
        if (pitch < v.w) { if (locked2d) b2->Unlock2D(); else buf->Unlock(); v.status = "unexpected frame layout"; return false; }
        for (int y = 0; y < v.h; y++) memcpy(&v.y[(size_t)y * v.w], data + (size_t)y * pitch, (size_t)v.w);
        const BYTE *uv = data + (size_t)pitch * v.fh; // NV12: interleaved chroma follows the (padded) luma plane
        for (int y = 0; y < v.h / 2; y++)
            for (int x = 0; x < v.w / 2; x++) { v.u[(size_t)y * v.w / 2 + x] = uv[(size_t)y * pitch + 2 * x]; v.v[(size_t)y * v.w / 2 + x] = uv[(size_t)y * pitch + 2 * x + 1]; }
        if (locked2d) b2->Unlock2D(); else buf->Unlock();
        return true;
    }
    return false;
}

static void video_step() {
    Video &v = A.vid;
    if (!v.rd) return;
    if (v.rebuild) {
        static const int depth[3] = { 0, 8, 16 };
        zap_video_destroy(v.enc);
        v.enc = zap_venc_create(v.w, v.h, v.quality, v.keyint, depth[v.packing]);
        v.pkt.resize(zap_video_bound(v.enc));
        v.rebuild = false;
    }
    auto t0 = Clock::now();
    if (!read_source_frame()) return;
    v.src_ms.push((float)ms_since(t0));
    t0 = Clock::now();
    size_t n = zap_venc_frame(v.enc, v.y.data(), v.u.data(), v.v.data(), v.w, v.w / 2, v.pkt.data(), v.pkt.size());
    v.enc_ms.push((float)ms_since(t0));
    t0 = Clock::now();
    if (!n || zap_vdec_frame(v.dec, v.pkt.data(), n)) { v.status = "zap round trip failed"; return; }
    v.dec_ms.push((float)ms_since(t0));
    v.kbps.push((float)(n * 8.0 * v.fps / 1000.0));
    const uint8_t *zy, *zu, *zv;
    int ys, cs;
    zap_video_planes(v.dec, &zy, &zu, &zv, &ys, &cs);
    double se = 0;
    for (int y = 0; y < v.h; y++) for (int x = 0; x < v.w; x++) { int d = zy[(size_t)y * ys + x] - v.y[(size_t)y * v.w + x]; se += d * d; }
    v.psnr.push((float)(se > 0 ? 10 * log10(255.0 * 255.0 * v.w * v.h / se) : 99));
    const uint8_t *planes[6] = { v.y.data(), v.u.data(), v.v.data(), zy, zu, zv };
    const int pitches[6] = { v.w, v.w / 2, v.w / 2, ys, cs, cs };
    for (int i = 0; i < 6; i++) A.g.ctx->UpdateSubresource(v.tex[i].Get(), 0, nullptr, planes[i], (UINT)pitches[i], 0);
    v.frames++;
}

// ---------------------------------------------------------------- view + input

static void view_xf(int iw, int ih, float out[4]) {
    float wa = (float)A.ww / A.wh, ia = (float)iw / ih, sx = 1, sy = 1;
    if (wa > ia) sx = wa / ia; else sy = ia / wa;
    sx /= A.zoom; sy /= A.zoom;
    out[0] = sx; out[1] = sy;
    out[2] = 0.5f - sx * 0.5f + A.panx; out[3] = 0.5f - sy * 0.5f + A.pany;
}
static int cur_w() { return A.tab ? A.vid.w : A.iw; }
static int cur_h() { return A.tab ? A.vid.h : A.ih; }

static void resize(int w, int h) {
    if (!A.g.swap || w <= 0 || h <= 0) return;
    A.ww = w; A.wh = h;
    A.g.rtv.Reset();
    CHECK(A.g.swap->ResizeBuffers(0, (UINT)w, (UINT)h, DXGI_FORMAT_UNKNOWN, 0));
    ComPtr<ID3D11Texture2D> bb;
    CHECK(A.g.swap->GetBuffer(0, IID_PPV_ARGS(&bb)));
    CHECK(A.g.dev->CreateRenderTargetView(bb.Get(), nullptr, &A.g.rtv));
}

static void open_any(const std::wstring &p) {
    if (is_video_path(p)) { open_video(p); A.want_tab = 1; }
    else if (open_image(p)) A.want_tab = 0;
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(h, msg, wp, lp)) return 1;
    bool ui = ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
    switch (msg) {
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_SIZE: if (wp != SIZE_MINIMIZED) resize(LOWORD(lp), HIWORD(lp)); return 0;
    case WM_DROPFILES: {
        wchar_t p[MAX_PATH];
        if (DragQueryFileW((HDROP)wp, 0, p, MAX_PATH)) open_any(p);
        DragFinish((HDROP)wp);
        return 0;
    }
    case WM_LBUTTONDOWN: if (!ui) { A.drag_split = true; A.split = (float)GET_X_LPARAM(lp) / A.ww; SetCapture(h); } return 0;
    case WM_RBUTTONDOWN: if (!ui) { A.drag_pan = true; A.mx = GET_X_LPARAM(lp); A.my = GET_Y_LPARAM(lp); SetCapture(h); } return 0;
    case WM_LBUTTONUP: case WM_RBUTTONUP: A.drag_split = A.drag_pan = false; ReleaseCapture(); return 0;
    case WM_MOUSEMOVE: {
        int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp);
        if (A.drag_split) A.split = std::min(1.0f, std::max(0.0f, (float)x / A.ww));
        if (A.drag_pan && cur_w()) {
            float xf[4];
            view_xf(cur_w(), cur_h(), xf);
            A.panx -= (float)(x - A.mx) / A.ww * xf[0]; A.pany -= (float)(y - A.my) / A.wh * xf[1];
            A.mx = x; A.my = y;
        }
        return 0;
    }
    case WM_MOUSEWHEEL: { // zoom around the cursor
        if (ui || !cur_w()) return 0;
        POINT p = { GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
        ScreenToClient(h, &p);
        float xf[4], u = (float)p.x / A.ww, v = (float)p.y / A.wh;
        view_xf(cur_w(), cur_h(), xf);
        float tx = u * xf[0] + xf[2], ty = v * xf[1] + xf[3];
        A.zoom = std::min(64.0f, std::max(1.0f, A.zoom * (GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1.25f : 0.8f)));
        view_xf(cur_w(), cur_h(), xf);
        A.panx += tx - (u * xf[0] + xf[2]); A.pany += ty - (v * xf[1] + xf[3]);
        return 0;
    }
    }
    return DefWindowProcW(h, msg, wp, lp);
}

// ---------------------------------------------------------------- GUI

static void ui_view_controls() {
    ImGui::Separator();
    ImGui::Checkbox("Difference x8", &A.diff);
    ImGui::SameLine();
    ImGui::Checkbox("Linear filter", &A.linear);
    ImGui::SameLine();
    if (ImGui::Button("Reset view")) { A.zoom = 1; A.panx = A.pany = 0; A.split = 0.5f; }
    ImGui::TextDisabled("wheel: zoom   right-drag: pan   left-drag: split   drop files here");
}

static void label_col(const char *s) { ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted(s); }

static void ui_texture() {
    if (ImGui::Button("Open image...")) { std::wstring p = pick_file(A.hwnd, false); if (!p.empty() && !open_image(p)) MessageBoxW(A.hwnd, L"Can't load that image.", L"zap", MB_OK); }
    ImGui::SameLine();
    ImGui::Text("%s  %dx%d", A.img_name.c_str(), A.iw, A.ih);
    if (ImGui::BeginTable("sides", 3, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 9);
        ImGui::TableSetupColumn("Left");
        ImGui::TableSetupColumn("Right");
        ImGui::TableHeadersRow();
        label_col("Format");
        for (int s = 0; s < 2; s++) { ImGui::TableNextColumn(); ImGui::PushID(s); ImGui::SetNextItemWidth(-1); ImGui::Combo("##fmt", &A.side[s].fmt, kTexFmt, 4); ImGui::PopID(); }
        label_col("RDO");
        for (int s = 0; s < 2; s++) {
            ImGui::TableNextColumn(); ImGui::PushID(s + 10); ImGui::SetNextItemWidth(-1);
            if (A.side[s].fmt) ImGui::SliderFloat("##rdo", &A.side[s].rdo, 0, 200, A.side[s].rdo > 0 ? "%.0f" : "off", ImGuiSliderFlags_Logarithmic);
            else ImGui::TextDisabled("-");
            ImGui::PopID();
        }
        const char *rows[] = { "PSNR (RGB)", "GPU memory", "On disk (zap)", "Disk bits/pixel", "Encode time" };
        for (int r = 0; r < 5; r++) {
            label_col(rows[r]);
            for (int s = 0; s < 2; s++) {
                ImGui::TableNextColumn();
                const TexSide &t = A.side[s];
                if (t.busy || !t.shown.w) { ImGui::TextDisabled("encoding..."); continue; }
                const TexResult &x = t.shown;
                double px = (double)x.w * x.h;
                if (r == 0) { if (std::isinf(x.psnr)) ImGui::TextUnformatted("lossless"); else ImGui::Text("%.2f dB", x.psnr); }
                if (r == 1) ImGui::Text("%.2f MB", x.gpu / 1e6);
                if (r == 2) ImGui::Text("%.2f MB (%.1f:1)", x.disk / 1e6, px * 4 / (double)(x.disk ? x.disk : 1));
                if (r == 3) ImGui::Text("%.2f", x.disk * 8.0 / px);
                if (r == 4) ImGui::Text(x.fmt ? "%.0f ms (all cores)" : "%.0f ms (zap hc)", x.ms);
            }
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("On disk: BC blocks split + zap hc. Original: RGBA8 + zap hc (lossless).");
    ImGui::TextDisabled("Ratios are against uncompressed RGBA8.");
    ui_view_controls();
}

static void plot(const char *label, const Series &s, float lo, float hi, const char *fmt) {
    char overlay[64];
    snprintf(overlay, sizeof overlay, fmt, s.avg(30));
    ImGui::PlotLines(label, s.v, s.n, s.n < 240 ? 0 : s.at, overlay, lo, hi, ImVec2(ImGui::GetFontSize() * 18, ImGui::GetFontSize() * 3));
}

static void ui_video() {
    Video &v = A.vid;
    if (ImGui::Button("Open video...")) { std::wstring p = pick_file(A.hwnd, true); if (!p.empty()) open_video(p); }
    if (!v.rd) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", v.status.empty() ? "MP4 / MOV / MKV / AVI / WMV: anything Media Foundation decodes" : v.status.c_str());
        return;
    }
    ImGui::SameLine();
    if (ImGui::Button(v.paused ? "Play" : "Pause")) v.paused = !v.paused;
    ImGui::SameLine();
    ImGui::Text("%s", v.name.c_str());
    ImGui::TextDisabled("%dx%d  %.2f fps  %.0f s  %s %s range", v.w, v.h, v.fps, v.duration_s, v.bt709 ? "BT.709" : "BT.601", v.full ? "full" : "limited");
    if (!v.status.empty()) ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "%s", v.status.c_str());
    ImGui::SeparatorText("zap encoder (live)");
    if (ImGui::SliderInt("Quality", &v.quality, 1, 100)) v.rebuild = true;
    if (ImGui::SliderInt("Keyframe every", &v.keyint, 1, 300, "%d frames")) v.rebuild = true;
    const char *packs[3] = { "fast", "hc depth 8", "hc depth 16" };
    if (ImGui::Combo("Stream packing", &v.packing, packs, 3)) v.rebuild = true;
    double zk = v.kbps.avg((int)v.fps), dec = v.dec_ms.avg(30), src = v.src_ms.avg(30), enc = v.enc_ms.avg(30);
    if (ImGui::BeginTable("vstats", 3, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 9);
        ImGui::TableSetupColumn("Left: source");
        ImGui::TableSetupColumn("Right: zap");
        ImGui::TableHeadersRow();
        label_col("Codec");
        ImGui::TableNextColumn(); ImGui::TextUnformatted(v.codec.c_str());
        ImGui::TableNextColumn(); ImGui::Text("zap_video q%d", v.quality);
        label_col("Bitrate");
        ImGui::TableNextColumn(); if (v.src_kbps > 0) ImGui::Text("%.0f kbit/s", v.src_kbps); else ImGui::TextDisabled("unknown");
        ImGui::TableNextColumn(); ImGui::Text("%.0f kbit/s", zk); if (v.src_kbps > 0) { ImGui::SameLine(); ImGui::TextDisabled("(%.2fx)", zk / v.src_kbps); }
        label_col("PSNR-Y vs source");
        ImGui::TableNextColumn(); ImGui::TextDisabled("reference");
        ImGui::TableNextColumn(); ImGui::Text("%.2f dB", v.psnr.avg(30));
        label_col("Decode / frame");
        ImGui::TableNextColumn(); ImGui::Text("%.2f ms", src);
        ImGui::TableNextColumn(); ImGui::Text("%.2f ms (%.0f fps)", dec, dec > 0 ? 1000 / dec : 0);
        label_col("Encode / frame");
        ImGui::TableNextColumn(); ImGui::TextDisabled("-");
        ImGui::TableNextColumn(); ImGui::Text("%.2f ms", enc);
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Source decode: Media Foundation (may use several threads, includes demux).");
    ImGui::TextDisabled("zap decode: 1 thread, SSE2.");
    double budget = 1000.0 / v.fps, pipe = src + enc + dec;
    ImGui::TextColored(pipe <= budget ? ImVec4(0.4f, 1, 0.5f, 1) : ImVec4(1, 0.7f, 0.3f, 1), "Live pipeline %.1f ms/frame, budget %.1f ms%s", pipe, budget,
                       pipe <= budget ? "" : " (slower than real time)");
    plot("zap decode", v.dec_ms, 0, (float)std::max(2.0, dec * 3), "%.2f ms");
    plot("source decode", v.src_ms, 0, (float)std::max(2.0, src * 3), "%.2f ms");
    plot("zap bitrate", v.kbps, 0, (float)std::max(100.0, zk * 3), "%.0f kbit/s");
    plot("zap PSNR-Y", v.psnr, 20, 60, "%.2f dB");
    ui_view_controls();
}

static void ui() {
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::Begin("zap", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::BeginTabBar("tabs")) {
        int prev = A.tab;
        if (ImGui::BeginTabItem("Texture", nullptr, A.want_tab == 0 ? ImGuiTabItemFlags_SetSelected : 0)) { A.tab = 0; ui_texture(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Video", nullptr, A.want_tab == 1 ? ImGuiTabItemFlags_SetSelected : 0)) { A.tab = 1; ui_video(); ImGui::EndTabItem(); }
        if (A.tab != prev) { A.zoom = 1; A.panx = A.pany = 0; }
        ImGui::EndTabBar();
    }
    A.want_tab = -1;
    ImGui::End();
    if (!cur_w() || A.diff) return;
    std::string l, r; // labels either side of the split line
    if (A.tab == 0) {
        for (int s = 0; s < 2; s++) (s ? r : l) = std::string(kTexFmt[A.side[s].fmt]) + (A.side[s].fmt && A.side[s].rdo > 0 ? " + RDO" : "");
    } else {
        l = "source " + A.vid.codec; r = "zap q" + std::to_string(A.vid.quality);
    }
    ImDrawList *dl = ImGui::GetForegroundDrawList();
    float x = A.split * A.ww, pad = ImGui::GetFontSize() * 0.4f, y = A.wh - ImGui::GetFontSize() * 2.2f;
    ImVec2 ls = ImGui::CalcTextSize(l.c_str()), rs = ImGui::CalcTextSize(r.c_str());
    dl->AddRectFilled(ImVec2(x - ls.x - 3 * pad, y - pad), ImVec2(x - pad, y + ls.y + pad), IM_COL32(0, 0, 0, 170), pad);
    dl->AddText(ImVec2(x - ls.x - 2 * pad, y), IM_COL32(255, 255, 255, 255), l.c_str());
    dl->AddRectFilled(ImVec2(x + pad, y - pad), ImVec2(x + rs.x + 3 * pad, y + rs.y + pad), IM_COL32(0, 0, 0, 170), pad);
    dl->AddText(ImVec2(x + 2 * pad, y), IM_COL32(255, 255, 255, 255), r.c_str());
}

static void render_scene(ID3D11RenderTargetView *rtv) {
    const float clear[4] = { 0.08f, 0.08f, 0.09f, 1 };
    A.g.ctx->ClearRenderTargetView(rtv, clear);
    float cb[12] = { 0 };
    ID3D11SamplerState *smp = A.linear ? A.g.linear.Get() : A.g.point.Get();
    cb[4] = A.split; cb[5] = A.diff ? 1.0f : 0.0f;
    if (A.tab == 0 && A.side[0].srv && A.side[1].srv) {
        view_xf(A.iw, A.ih, cb);
        ID3D11ShaderResourceView *srv[2] = { A.side[0].srv.Get(), A.side[1].srv.Get() };
        draw(A.g, A.g.ps_tex.Get(), srv, 2, cb, A.ww, A.wh, rtv, smp);
    } else if (A.tab == 1 && A.vid.rd && A.vid.frames) {
        view_xf(A.vid.w, A.vid.h, cb);
        cb[8] = (float)A.vid.bt709; cb[9] = (float)A.vid.full;
        ID3D11ShaderResourceView *srv[6];
        for (int i = 0; i < 6; i++) srv[i] = A.vid.srv[i].Get();
        draw(A.g, A.g.ps_video.Get(), srv, 6, cb, A.ww, A.wh, rtv, smp);
    }
}

static void frame(ID3D11RenderTargetView *rtv) {
    tex_update();
    if (A.tab == 1 && A.vid.rd && !A.vid.paused && Clock::now() >= A.vid.next) {
        video_step();
        A.vid.next += std::chrono::microseconds((long long)(1e6 / A.vid.fps));
        if (Clock::now() - A.vid.next > std::chrono::milliseconds(250)) A.vid.next = Clock::now(); // fell behind: don't spiral
    }
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    ui();
    ImGui::Render();
    render_scene(rtv);
    A.g.ctx->OMSetRenderTargets(1, &rtv, nullptr);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

static void shots(const std::wstring &prefix, bool have_video) {
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = (UINT)A.ww; td.Height = (UINT)A.wh; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> rt, staging;
    ComPtr<ID3D11RenderTargetView> rtv;
    CHECK(A.g.dev->CreateTexture2D(&td, nullptr, &rt));
    CHECK(A.g.dev->CreateRenderTargetView(rt.Get(), nullptr, &rtv));
    td.BindFlags = 0; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    CHECK(A.g.dev->CreateTexture2D(&td, nullptr, &staging));
    std::vector<uint8_t> px((size_t)A.ww * A.wh * 4);
    auto grab = [&](const wchar_t *name) {
        for (int i = 0; i < 3; i++) frame(rtv.Get()); // let the UI settle (auto-resize takes a frame)
        A.g.ctx->CopyResource(staging.Get(), rt.Get());
        D3D11_MAPPED_SUBRESOURCE m;
        CHECK(A.g.ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m));
        for (int y = 0; y < A.wh; y++) memcpy(&px[(size_t)y * A.ww * 4], (uint8_t *)m.pData + (size_t)y * m.RowPitch, (size_t)A.ww * 4);
        A.g.ctx->Unmap(staging.Get(), 0);
        std::wstring path = prefix + name;
        printf("%ls: %s\n", path.c_str(), save_png(path, px.data(), A.ww, A.wh) ? "saved" : "FAILED");
    };
    auto settle = [&] { frame(rtv.Get()); for (auto &j : A.jobs) j.join(); A.jobs.clear(); };
    A.side[0].fmt = 1; A.side[1].fmt = 3; // BC1 | BC7
    A.want_tab = 0;
    settle();
    grab(L"_texture.png");
    A.zoom = 8; A.panx = 0.15f; A.pany = -0.3f; // up close on fine detail, point sampled: BC blocks visible
    grab(L"_texture_zoom.png");
    A.zoom = 1; A.panx = A.pany = 0;
    A.side[0].fmt = 0; A.side[1].fmt = 1; A.diff = true; // |original - BC1| x8
    settle();
    grab(L"_texture_diff.png");
    A.diff = false;
    if (have_video) {
        A.want_tab = 1;
        frame(rtv.Get());
        for (int i = 0; i < 90; i++) video_step();
        A.vid.paused = true;
        grab(L"_video.png");
    }
}

// ----------------------------------------------------------------

int wmain(int argc, wchar_t **argv) {
    std::wstring path, shot, video;
    for (int i = 1; i < argc; i++) {
        std::wstring a = argv[i];
        if (a == L"--verify") return verify();
        if (a == L"--shot" && i + 1 < argc) shot = argv[++i];
        else if (a == L"--video" && i + 1 < argc) video = argv[++i];
        else path = a;
    }
    CHECK(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE));
    CHECK(MFStartup(MF_VERSION, MFSTARTUP_LITE));
    ImGui_ImplWin32_EnableDpiAwareness();
    float dpi = ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY));
    A.ww = (int)(1600 * dpi); A.wh = (int)(900 * dpi);

    WNDCLASSW wc = {};
    wc.lpfnWndProc = wndproc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"zap_viewer";
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512) /* IDC_ARROW */);
    RegisterClassW(&wc);
    RECT r = { 0, 0, A.ww, A.wh };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    A.hwnd = CreateWindowW(L"zap_viewer", L"zap viewer", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                           nullptr, nullptr, wc.hInstance, nullptr);
    DragAcceptFiles(A.hwnd, TRUE);
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = A.hwnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    CHECK(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &sd, &A.g.swap, &A.g.dev, nullptr, &A.g.ctx));
    init_pipeline(A.g);
    RECT cr;
    GetClientRect(A.hwnd, &cr);
    resize(cr.right - cr.left, cr.bottom - cr.top);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGuiStyle &st = ImGui::GetStyle();
    st.ScaleAllSizes(dpi);
    st.FontScaleDpi = dpi;
    st.WindowRounding = 6 * dpi;
    ImGui_ImplWin32_Init(A.hwnd);
    ImGui_ImplDX11_Init(A.g.dev.Get(), A.g.ctx.Get());

    if (!path.empty() && is_video_path(path)) { video = path; path.clear(); }
    if (path.empty() || !open_image(path)) open_image(L"C:\\Windows\\Web\\Wallpaper\\Windows\\img0.jpg");
    if (!video.empty()) { open_video(video); A.want_tab = 1; }

    if (!shot.empty()) shots(shot, A.vid.rd != nullptr);
    else {
        ShowWindow(A.hwnd, SW_SHOW);
        for (bool quit = false; !quit;) {
            MSG m;
            while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
                if (m.message == WM_QUIT) quit = true;
                TranslateMessage(&m); DispatchMessageW(&m);
            }
            if (quit) break;
            if (IsIconic(A.hwnd)) { Sleep(16); continue; }
            frame(A.g.rtv.Get());
            A.g.swap->Present(1, 0);
        }
    }
    for (auto &j : A.jobs) j.join();
    video_close();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    MFShutdown();
    return 0;
}
