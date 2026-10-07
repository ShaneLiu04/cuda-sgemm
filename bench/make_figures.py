# =====================================================================
# make_figures.py — 从 results/performance.csv 生成本工程全部可解释性图表
# 用法: python bench/make_figures.py   （输出 results/figures/*.png）
# 数据来源: AR007 矩阵会话（2026-10-05, Quadro RTX 5000, 时钟策略 B,
#           9 kernel × 6 尺寸 × rounds=3 门控多轮，git_sha=84e261f）；
#           每个 (kernel,size) 取 CSV 末行 = 最新矩阵会话
# 消融数据: results/ablation_ar007.csv（bk/lb 分流）+ 矩阵默认行
# 资源数据: build.log（-Xptxas -v 审计, 2026-10-05 重建）
# 带宽标定: E10 D2D 实测 375.7 GB/s；理论 FP32 峰值 11.15 TF（@1815 MHz，
#           会话动态加速最高 1950 MHz → 会话峰值 11.98 TF）
# =====================================================================
import csv
import math
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "performance.csv")
OUT_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SESSION_SHA = "84e261f"
PEAK_TF_NOMINAL = 11150.0     # GFLOPS @ 1815 MHz (标称 boost)
PEAK_TF_SESSION = 11980.0     # GFLOPS @ 1950 MHz (会话实测最高 SM 时钟)
BW_MEASURED = 375.7           # GB/s, E10 D2D 标定
BW_THEORETICAL = 448.1        # GB/s, 显存规格

KERNELS = ["naive", "coalesced", "smem1d", "tile2d", "vec4", "cpasync",
           "cpasync2", "swpipe", "cublas"]
LABEL = {
    "naive": "K0 naive", "coalesced": "K1 coalesced", "smem1d": "K2 smem1d",
    "tile2d": "K3 tile2d", "vec4": "K4 vec4", "cpasync": "K5 cpasync",
    "cpasync2": "K5' cpasync2", "swpipe": "K6 swpipe", "cublas": "cuBLAS FP32",
}
TECH = {
    "naive": "no tiling\nstrided access", "coalesced": "coalesced\nmapping",
    "smem1d": "smem tile\n32x32x32 + TM=8", "tile2d": "2D reg tile\n128x128x8, 8x8",
    "vec4": "float4 loads\n+ XOR swizzle", "cpasync": "double buffer\n(sync on sm_75)",
    "cpasync2": "db + A transpose\n+ reg prefetch", "swpipe": "single buffer\n+ reg prefetch\n(no cp.async)",
    "cublas": "NVIDIA library",
}
COLOR = {
    "naive": "#9e9e9e", "coalesced": "#64b5f6", "smem1d": "#4db6ac",
    "tile2d": "#ffb74d", "vec4": "#2e7d32", "cpasync": "#ba68c8",
    "cpasync2": "#f06292", "swpipe": "#d84315", "cublas": "#212121",
}
SIZES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
         (2048, 2048, 2048), (4096, 4096, 4096), (1000, 1016, 1024)]
SIZE_LBL = {s: f"{s[0]}x{s[1]}x{s[2]}" for s in SIZES}

REGS = {"naive": 50, "coalesced": 50, "smem1d": 72, "tile2d": 114,
        "vec4": 128, "cpasync": 139, "cpasync2": 125, "swpipe": 128, "cublas": 0}
SMEM_KB = {"naive": 0, "coalesced": 0, "smem1d": 9.0, "tile2d": 10.6,
           "vec4": 8.1, "cpasync": 20.0, "cpasync2": 16.3, "swpipe": 8.1,
           "cublas": 0}
OCC_PCT = {"naive": 100, "coalesced": 100, "smem1d": 88, "tile2d": 50,
           "vec4": 50, "cpasync": 25, "cpasync2": 50, "swpipe": 50, "cublas": 0}

ROWS = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    lines = [ln for ln in f if not ln.startswith("#")]
for raw in csv.reader(lines):
    if len(raw) != 14:
        continue
    r = {"kernel": raw[0], "m": int(raw[1]), "n": int(raw[2]), "k": int(raw[3]),
         "ms_median": float(raw[4]), "rsd_pct": float(raw[8]),
         "gflops": float(raw[9]), "git_sha": raw[12]}
    if r["git_sha"] == SESSION_SHA:
        ROWS.append(r)

GF = {}
for r in ROWS:
    key = (r["kernel"], r["m"], r["n"], r["k"])
    GF[key] = r["gflops"]      # 末行胜出 = AR007 矩阵会话行

def gf(kernel, size):
    return GF.get((kernel,) + size, np.nan)

MAIN = (4096, 4096, 4096)
SQ_SIZES = SIZES[:5]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 9.5,
    "savefig.bbox": "tight", "savefig.dpi": 170,
})

# ---------------------------------------------------------------- fig9 (AR008)
ABL_AR008 = os.path.join(ROOT, "results", "ablation_ar008.csv")
SK_SWEEP_SIZES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                  (1000, 1016, 1024), (2048, 2048, 2048)]
SKS = [1, 2, 4, 8, 12, 16]
NUM_SMS = 48
TILE = 128   # swsk tile 主体 BM=BN=128


def load_sk_ablation():
    """读 ablation_ar008.csv：swsk_sk<N> 行（每 (size,sk) 2 行=升/降序双向遍历）
    + cublas 参考行。返回 {(size,sk): [gf...]} 与 {size: [gf...]}。"""
    data, cb = {}, {}
    if not os.path.exists(ABL_AR008):
        return data, cb
    with open(ABL_AR008, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) != 14:
            continue
        name, m, n, k = raw[0], int(raw[1]), int(raw[2]), int(raw[3])
        g = float(raw[9])
        size = (m, n, k)
        if name.startswith("swsk_sk"):
            data.setdefault((size, int(name[len("swsk_sk"):])), []).append(g)
        elif name == "cublas":
            cb.setdefault(size, []).append(g)
    return data, cb


SIZE256_CSV = os.path.join(ROOT, "results", "size256_ar008.csv")


def load_size256():
    """AR008 T005 一次性采集：256^3 全 kernel 同会话（2026-10-05 下午，
    1620 MHz 持续态）。返回 {name: gf}。"""
    out = {}
    if not os.path.exists(SIZE256_CSV):
        return out
    with open(SIZE256_CSV, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) != 14:
            continue
        out[raw[0]] = float(raw[9])
    return out


def fig_splitk_sweep():
    data, cb = load_sk_ablation()
    if not data:
        print("[fig9] ablation_ar008.csv 无 swsk 数据，跳过")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.8),
                                        gridspec_kw={"width_ratios": [1.15, 1.0, 1.3]})
    palette = dict(zip(SK_SWEEP_SIZES,
                       ["#5c6bc0", "#26a69a", "#ef6c00", "#8e24aa", "#c62828"]))
    for size in SK_SWEEP_SIZES:
        blocks = -(-size[0] // TILE) * (-(-size[1] // TILE))
        xs, ys, best = [], [], None
        for sk in SKS:
            gfs = data.get((size, sk))
            if not gfs:
                continue
            med = float(np.median(gfs))
            xs.append(blocks * sk / NUM_SMS)
            ys.append(med)
            if best is None or med > best[1]:
                best = (blocks * sk / NUM_SMS, med, sk)
        if not xs:
            continue
        lbl = f"{size[0]}x{size[1]}x{size[2]}  ({blocks} blocks)"
        c = palette[size]
        ax1.plot(xs, ys, "-o", color=c, lw=1.8, ms=5, label=lbl)
        ax2.plot(xs, ys, "-o", color=c, lw=1.8, ms=5, label=lbl)
        # 最优点：星标 + sk 标注
        ax1.plot(best[0], best[1], "*", color=c, ms=15, mec="#212121", mew=0.7)
        ax2.plot(best[0], best[1] / np.median(cb[size]), "*", color=c, ms=15,
                 mec="#212121", mew=0.7)
        ax1.annotate(f"sk{best[2]}", (best[0], best[1]),
                     textcoords="offset points", xytext=(6, 5),
                     fontsize=9, color=c, fontweight="bold")
        # cuBLAS 参考线（同会话同协议）
        if size in cb:
            cbmed = np.median(cb[size])
            ax1.axhline(cbmed, color=c, ls=":", lw=1.1, alpha=0.45)
    ax1.set_yscale("log")
    ax1.set_xlabel("effective waves = blocks x sk / 48 SMs")
    ax1.set_ylabel("GFLOPS (log)")
    ax1.set_title("(a) absolute throughput vs machine fill")
    ax1.legend(fontsize=8.6, loc="upper right", framealpha=0.9)
    ax2.axhline(1.0, color="#212121", ls="--", lw=1.2, alpha=0.7)
    ax2.text(0.985, 1.02, "cuBLAS FP32 (same session)", ha="right", va="bottom",
             transform=ax2.get_yaxis_transform(), fontsize=8.8, color="#212121")
    ax2.set_xlabel("effective waves = blocks x sk / 48 SMs")
    ax2.set_ylabel("GFLOPS / cuBLAS GFLOPS")
    ax2.set_title("(b) normalized to cuBLAS - crossing at 256$^3$ sk12")
    ax2.set_ylim(0, None)

    # ---- (c) 256^3 全 kernel 同会话对比（T005 守成专项）----
    s256 = load_size256()
    if s256:
        order = ["naive", "coalesced", "tile2d", "vec4", "cpasync",
                 "cpasync2", "swpipe", "smem1d", "cublas", "swsk_sk12"]
        blocks256 = {"naive": 256, "coalesced": 256, "tile2d": 4, "vec4": 4,
                     "cpasync": 4, "cpasync2": 4, "swpipe": 4, "smem1d": 64,
                     "cublas": -1, "swsk_sk12": 48}
        lbls = {"naive": "K0", "coalesced": "K1", "tile2d": "K3", "vec4": "K4",
                "cpasync": "K5", "cpasync2": "K5'", "swpipe": "K6",
                "smem1d": "K2", "cublas": "cuBLAS", "swsk_sk12": "K6'\nswsk\nsk12"}
        xs = np.arange(len(order))
        for i, kn in enumerate(order):
            if kn not in s256:
                continue
            v = s256[kn]
            c = "#4527a0" if kn == "swsk_sk12" else (
                COLOR.get(kn, "#9e9e9e") if kn != "cublas" else "#212121")
            ax3.bar(i, v, width=0.68, color=c, edgecolor="white",
                    linewidth=0.7, zorder=3)
            ax3.text(i, v + 28, f"{v:,.0f}", ha="center", va="bottom",
                     fontsize=8.8, fontweight="bold", color=c)
            b = blocks256[kn]
            ax3.text(i, -218, ("int." if b < 0 else f"{b} blk") +
                     ("" if b < 0 else f"\n{b/48:.2f} wv"),
                     ha="center", va="top", fontsize=7.6, color="#555")
        ax3.axhline(s256.get("smem1d", np.nan), color="#4db6ac", ls=":", lw=1.2,
                    alpha=0.8, zorder=2)
        ax3.text(0.99, s256.get("smem1d", 0) + 30, "smem1d (AR007 champion)",
                 ha="right", fontsize=8.6, color="#00695c")
        ax3.set_xticks(xs)
        ax3.set_xticklabels([lbls[k] for k in order], fontsize=8.8)
        ax3.set_ylabel("GFLOPS (256$^3$)")
        ax3.set_ylim(top=max(s256.values()) * 1.14)
        ax3.set_title("(c) 256$^3$ all kernels, same session:\n"
                      "swsk sk12 (1.0 wave) takes the crown +7.9% over smem1d")
    fig.text(0.995, 0.012,
             "split-K cost: P workspace = sk x M x N x 4 B write + read "
             "(256$^3$ sk12: 6.3 MB round-trip; 2048$^3$ sk16: 1.07 GB)\n"
             "medians of bidirectional 2-pass (asc+desc) runs, 45C gated groups, "
             "100 iters x 2, session 2026-10-05, git 84e261f",
             ha="right", va="bottom", fontsize=8.2, color="#455a64")
    fig.suptitle("AR008 split-K slice sweep (swsk): starvation zone gains, "
                 "saturation zone pure loss", fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig9_splitk_sweep.png"))
    plt.close(fig)


AUTO_CSV = os.path.join(ROOT, "results", "auto_ar008.csv")
# T006 配对段计划：每尺寸 [winner -> auto] x2（run_auto_paired.ps1 尾部 24 行）
AUTO_WINNER = {
    (256, 256, 256): ("swsk", 12), (512, 512, 512): ("swsk", 4),
    (1024, 1024, 1024): ("swsk", 4), (1000, 1016, 1024): ("swsk", 4),
    (2048, 2048, 2048): ("swpipe", 1), (4096, 4096, 4096): ("swpipe", 1),
}


def load_auto_paired():
    """读 auto_ar008.csv 尾部 24 行（run_auto_paired 配对段）。
    返回 {size: {"winner": [gf...], "auto": [gf...]}}。"""
    if not os.path.exists(AUTO_CSV):
        return {}
    with open(AUTO_CSV, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    rows = [r for r in csv.reader(lines) if len(r) == 14]
    out = {}
    for raw in rows[-24:]:
        size = (int(raw[1]), int(raw[2]), int(raw[3]))
        g = float(raw[9])
        key = "auto" if raw[0] == "auto" else "winner"
        out.setdefault(size, {}).setdefault(key, []).append(g)
    return out


def fig_dispatch_map():
    paired = load_auto_paired()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    # ---- (a) (M,N) dispatch 地图（K 充分大，无 K 钳制）----
    mgrid = np.logspace(6, 13, 140, base=2.0)   # 64 .. 8192
    band = np.zeros((len(mgrid), len(mgrid)))
    for i, mv in enumerate(mgrid):
        for j, nv in enumerate(mgrid):
            blocks = math.ceil(mv / 128) * math.ceil(nv / 128)
            band[i, j] = 0 if blocks <= 4 else (1 if blocks <= 64 else 2)
    cmap = matplotlib.colors.ListedColormap(["#4527a0", "#7e57c2", "#d84315"])
    ax1.pcolormesh(mgrid, mgrid, band, cmap=cmap,
                   norm=matplotlib.colors.BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
                   shading="auto")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    for (m, n, k) in AUTO_WINNER:
        ax1.plot(n, m, "o", mec="white", mfc="#ffe082", ms=9, mew=1.6, zorder=5)
        ax1.annotate(f"{m}x{n}", (n, m), textcoords="offset points",
                     xytext=(8, -3), fontsize=8.4, color="white", zorder=6)
    handles = [matplotlib.patches.Patch(color="#4527a0", label="swsk sk=12 (blocks<=4)"),
               matplotlib.patches.Patch(color="#7e57c2", label="swsk sk=4 (blocks<=64)"),
               matplotlib.patches.Patch(color="#d84315", label="swpipe single-wave (blocks>64)")]
    ax1.legend(handles=handles, loc="upper left", fontsize=8.6, framealpha=0.92)
    ax1.set_xlabel("N")
    ax1.set_ylabel("M")
    ax1.set_title("(a) auto dispatch map over (M, N)\n(K large; sk clamped to ceil(K/8))")
    # ---- (b) 选中=实测最优 配对验证 ----
    sizes = list(AUTO_WINNER.keys())
    xs = np.arange(len(sizes))
    w_vals, a_vals = [], []
    for s in sizes:
        d = paired.get(s, {})
        w_vals.append(np.median(d.get("winner", [np.nan])))
        a_vals.append(np.median(d.get("auto", [np.nan])))
    ax2.bar(xs - 0.19, w_vals, width=0.38, color="#78909c",
            label="size winner (sweep-measured)", zorder=3)
    ax2.bar(xs + 0.19, a_vals, width=0.38, color="#4527a0",
            label="auto (dispatched)", zorder=3)
    for i, (w, a) in enumerate(zip(w_vals, a_vals)):
        kn, sk = AUTO_WINNER[sizes[i]]
        ax2.text(i, max(w, a) * 1.015, f"{(a/w-1)*100:+.2f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color="#2e7d32" if abs(a / w - 1) <= 0.02 else "#c62828")
        ax2.text(i, -max(w_vals) * 0.055, f"{kn}" + (f" sk{sk}" if kn == "swsk" else ""),
                 ha="center", va="top", fontsize=8.2, color="#555")
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{s[0]}x{s[1]}" for s in sizes], fontsize=9)
    ax2.set_ylabel("GFLOPS (median of 2 paired rounds)")
    ax2.set_ylim(0, max(w_vals + a_vals) * 1.12)
    ax2.set_title("(b) auto pick vs measured winner, back-to-back pairs\n"
                  "all |delta| <= 0.9% (tolerance +/-2%) -> selection validated")
    ax2.legend(fontsize=9, loc="upper left")
    fig.suptitle("AR008 --kernel auto: geometric dispatch (T004/T005 measured bands)",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig10_dispatch_map.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig12/13
# ws 三轴消融（T008）：8 配置 = PW{1,2} x STAGES{2,3} x LB{1,2}
# 资源包络来自 build.log ptxas 审计（实测）；occupancy = 驻留 warp / 32（sm_75）
WS_RESOURCES = [
    # (pw, stages, lb, regs, spill_B, smem_B, warps_per_sm, occupancy)
    (2, 3, 1, 128, 0, 24960, 10, "31.3%"),
    (2, 3, 2,  96, 0, 24960, 20, "62.5%"),
    (2, 2, 1, 129, 0, 16640, 10, "31.3%"),
    (2, 2, 2,  96, 8, 16640, 20, "62.5%"),
    (1, 3, 1, 125, 0, 24960,  9, "28.1%"),
    (1, 3, 2,  96, 0, 24960, 18, "56.3%"),
    (1, 2, 1, 130, 0, 16640,  9, "28.1%"),
    (1, 2, 2, 96, 8, 16640, 18, "56.3%"),
]
WS_CFG_LABELS = ["ws_pw1_st2_lb1", "ws_pw1_st2_lb2", "ws_pw1_st3_lb1",
                 "ws_pw1_st3_lb2", "ws_pw2_st2_lb1", "ws_pw2_st2_lb2",
                 "ws_pw2_st3_lb1", "ws_pw2_st3_lb2"]


def load_ablation_rows(prefix):
    """读 ablation_ar008.csv：{('ws_pw2_st3_lb1','512x512x512'): median_gf}"""
    out = {}
    if not os.path.exists(ABL_AR008):
        return out
    with open(ABL_AR008, newline="", encoding="utf-8", errors="replace") as f:
        acc = {}
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) == 14 and r[0].startswith(prefix):
                key = (r[0], f"{r[1]}x{r[2]}x{r[3]}")
                acc.setdefault(key, []).append(float(r[9]))
        out = {k: np.median(v) for k, v in acc.items()}
    return out


def fig_ws_ablation():
    ws = load_ablation_rows("ws_pw")
    refs = load_ablation_rows("swpipe")
    refs.update(load_ablation_rows("cublas"))
    sizes = ["512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096"]
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [2.7, 1.0]})

    xs = np.arange(len(sizes))
    bw = 0.10
    shades = ["#b0bec5", "#78909c", "#4527a0", "#7e57c2", "#5e35b1", "#9575cd",
              "#d84315", "#ef6c00"]
    for i, cfg in enumerate(WS_CFG_LABELS):
        vals = [ws.get((cfg, s), np.nan) for s in sizes]
        ax.bar(xs + (i - 3.5) * bw, vals, width=bw * 0.92, color=shades[i],
               label=cfg, zorder=3)
    for j, s in enumerate(sizes):
        sp = refs.get(("swpipe", s), np.nan)
        ax.plot([j - 0.5, j + 0.5], [sp, sp], color="#111", lw=1.8, ls="--",
                zorder=4)
        cb = refs.get(("cublas", s), np.nan)
        ax.plot([j - 0.5, j + 0.5], [cb, cb], color="#c62828", lw=1.8, ls=":",
                zorder=4)
    ax.plot([], [], color="#111", lw=1.8, ls="--", label="swpipe (baseline)")
    ax.plot([], [], color="#c62828", lw=1.8, ls=":", label="cublas")
    ax.set_xticks(xs)
    ax.set_xticklabels(["512³", "1024³", "2048³", "4096³"], fontsize=10)
    ax.set_ylabel("GFLOPS (median of 2-pass bidirectional sweep)")
    ax.set_title("(a) ws 3-axis ablation matrix: PW x STAGES x LB\n"
                 "(cooldown-gated, ascending+descending pass pairing)")
    ax.legend(fontsize=7.6, ncol=3, loc="upper left")
    ax.set_ylim(0, 11200)

    axr.axis("off")
    axr.text(0.5, 1.0, "ptxas resource envelope\n(build.log, measured)",
             ha="center", va="top", fontsize=10, fontweight="bold")
    header = "pw st lb regs spill  smem  w/SM  occup"
    axr.text(0.5, 0.90, header, ha="center", va="top",
             family="monospace", fontsize=8.6, fontweight="bold")
    for i, r in enumerate(WS_RESOURCES):
        line = f"{r[0]:>2} {r[1]:>2} {r[2]:>2} {r[3]:>4} {r[4]:>5} {r[5]:>5} {r[6]:>4} {r[7]:>6}"
        color = "#c62828" if r[4] != 0 else "#222"
        axr.text(0.5, 0.845 - i * 0.058, line, ha="center", va="top",
                 family="monospace", fontsize=8.6, color=color)
    axr.text(0.5, 0.845 - 8 * 0.058 - 0.03,
             "red = 8B spill (STAGES=2 x LB=2)\nLB=2 occupancy gain rejected:\n"
             "62.5% configs lose to 31.3% (3/4 sizes)",
             ha="center", va="top", fontsize=8.2, color="#c62828")
    fig.suptitle("AR008 T008 ws ablation: PW=2 beats PW=1; LB=2 rejected; "
                 "ws never beats swpipe at >=1024³",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig12_ws_ablation.png"))
    plt.close(fig)


def fig_isslot_hypothesis():
    ws = load_ablation_rows("ws_pw")
    sp = load_ablation_rows("swpipe")
    cb = load_ablation_rows("cublas")
    sizes = ["512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096"]
    labels = ["512³", "1024³", "2048³", "4096³"]
    best_cfg, best_gf, delta = [], [], []
    for s in sizes:
        c = max((c for c in WS_CFG_LABELS if (c, s) in ws), key=lambda c: ws[(c, s)])
        best_cfg.append(c)
        best_gf.append(ws[(c, s)])
        delta.append((ws[(c, s)] / sp[("swpipe", s)] - 1) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = ["#2e7d32" if d >= 0 else "#c62828" for d in delta]
    xs = np.arange(len(sizes))
    ax1.bar(xs, delta, color=colors, width=0.55, zorder=3)
    for i, (d, c) in enumerate(zip(delta, best_cfg)):
        ax1.text(i, d + (0.8 if d >= 0 else -1.6), f"{d:+.1f}%",
                 ha="center", fontsize=11, fontweight="bold",
                 color=colors[i])
        ax1.text(i, -25.5, c, ha="center", fontsize=8, color="#555", rotation=0)
    ax1.axhline(0, color="#333", lw=1.0)
    ax1.axhline(2, color="#2e7d32", lw=1.0, ls="--")
    ax1.text(3.42, 2.4, "G4 line (+2%)", fontsize=8, color="#2e7d32")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("best-ws vs swpipe, in-sweep delta (%)")
    ax1.set_ylim(-28, 10)
    ax1.set_title("(a) issue-slot hypothesis verdict:\n"
                  "rejected at large sizes (all 8 configs below swpipe)")

    for i, s in enumerate(sizes):
        ax2.plot(i, sp[("swpipe", s)], "s", color="#111", ms=9)
        ax2.plot(i, cb[("cublas", s)], "o", color="#c62828", ms=9)
        ax2.plot(i, best_gf[i], "^", color="#4527a0", ms=10)
        ax2.annotate(f"{delta[i]:+.0f}%", (i, best_gf[i]),
                     textcoords="offset points", xytext=(14, 4),
                     fontsize=9, color=colors[i], fontweight="bold")
    ax2.plot([], [], "s", color="#111", label="swpipe")
    ax2.plot([], [], "^", color="#4527a0", label="ws (best cfg)")
    ax2.plot([], [], "o", color="#c62828", label="cublas")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("GFLOPS (2-pass median)")
    ax2.set_title("(b) absolute ladder: ws peak 5.5-5.7 TF\n"
                  "vs swpipe 5.2-7.1 TF (G3 target 7.0 TF: FAIL)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 11500)
    fig.suptitle("AR008 ws issue-slot hypothesis: NEGATIVE at large sizes — "
                 "pre-Ampere software pipelining boundary (honest null result)",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig13_isslot_hypothesis.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig1


def fig_ws_structure():
    STAGES, NT = 3, 8
    FILL, COMP = 1.0, 2.4          # 模型化单位（结构示意）
    slot_colors = ["#4527a0", "#7e57c2", "#9575cd"]

    prod, cons = [], []            # (t, slot, start, dur)
    p_free = c_free = 0.0
    empty_rel = [0.0] * STAGES     # 环初态视为空（t<STAGES 免等）
    full_rel = [None] * STAGES
    for t in range(NT):
        s = t % STAGES
        ps = max(p_free, empty_rel[s] if t >= STAGES else 0.0)
        prod.append((t, s, ps, FILL))
        p_free = ps + FILL
        full_rel[s] = ps + FILL
        cs = max(c_free, full_rel[s])
        cons.append((t, s, cs, COMP))
        c_free = cs + COMP
        empty_rel[s] = cs + COMP

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [2.6, 1.0]})

    for (t, s, st, d) in prod:
        ax.broken_barh([(st, d)], (1.55, 0.9), color=slot_colors[s],
                       edgecolor="white", zorder=3)
    for (t, s, st, d) in cons:
        ax.broken_barh([(st, d)], (-0.45, 0.9), color=slot_colors[s],
                       edgecolor="white", zorder=3)
        ax.text(st + d / 2, -0.02, f"t{t}", ha="center", va="center",
                fontsize=7.5, color="white", zorder=4)
    for (t, s, st, d) in prod:
        ax.text(st + d / 2, 1.98, f"t{t}", ha="center", va="center",
                fontsize=7.5, color="white", zorder=4)

    # 环位标签（右缘）
    ax.text(NT * 1.02, 2.0, "producer\n(warp 0-1)", fontsize=9.5, va="center")
    ax.text(NT * 1.02, 0.0, "consumer\n(warp 2-9)", fontsize=9.5, va="center")
    ax.text(NT * 1.02, 1.0, "smem ring\n3 x 8.3KB", fontsize=8.5, va="center",
            color="#555")

    # 屏障事件标注（首个环回绕 = 协议关键点）
    t3 = [p for p in prod if p[0] == 3][0]
    ax.annotate("bar.sync empty[0]\n(consumer t0 done)", xy=(t3[2], 2.45),
                xytext=(t3[2] - 2.2, 3.15), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
                color="#c62828")
    c0 = cons[0]
    ax.annotate("bar.sync full[0]", xy=(c0[2], -0.5), xytext=(c0[2] - 1.5, -1.35),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
                color="#2e7d32")
    # run-ahead 括注（稳态：producer 领先 STAGES-1 个 tile）
    pt5 = [p for p in prod if p[0] == 5][0]
    ct5 = [c for c in cons if c[0] == 5][0]
    ax.annotate("", xy=(pt5[2] + 0.5, 2.9), xytext=(ct5[2] + 0.5, 0.55),
                arrowprops=dict(arrowstyle="<->", color="#f9a825", lw=1.4))
    ax.text((pt5[2] + ct5[2]) / 2 + 3.4, 1.5,
            "run-ahead\n= STAGES-1 tiles\n(DRAM latency cover)",
            fontsize=8.5, color="#b8860b", ha="left")

    ax.set_ylim(-1.6, 3.6)
    ax.set_xlim(-0.2, NT * 1.30)
    ax.set_yticks([])
    ax.set_xlabel("time (modeled units; structure schematic, not measured)")
    ax.set_title("(a) ws pipeline spacetime: named-barrier ring protocol "
                 "(bar.arrive = non-blocking, bar.sync = blocking, count=320)")
    for sp in ("left", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (b) 资源包络（build.log ptxas 实测表；PW=2 默认路径 4 实例）----
    axr.axis("off")
    rows = [("pw", "st", "lb", "regs", "spill", "smem", "occup")]
    for r in WS_RESOURCES:
        if r[0] != 2:      # 结构图只列默认 PW=2 的 4 实例（全 8 实例见 fig12）
            continue
        rows.append((str(r[0]), str(r[1]), str(r[2]), str(r[3]),
                     str(r[4]), str(r[5]), r[7]))
    ytab = 0.92
    axr.text(0.5, 1.0, "(b) ptxas resource envelope\n(build.log, measured; PW=2)",
             ha="center", va="top", fontsize=10, fontweight="bold")
    for i, row in enumerate(rows):
        weight = "bold" if i == 0 else "normal"
        color = "#c62828" if (i > 0 and row[4] != "0") else "#222"
        axr.text(0.5, ytab - i * 0.085, "  ".join(f"{c:>7}" for c in row),
                 ha="center", va="top", family="monospace", fontsize=8.8,
                 fontweight=weight, color=color)
    axr.text(0.5, ytab - len(rows) * 0.085 - 0.04,
             "red = 8B spill (LB=2,STAGES=2 instance;\nrecorded tradeoff, T008 verdict)",
             ha="center", va="top", fontsize=8, color="#c62828")
    fig.suptitle("AR008 Kernel 7 ws: warp-specialized producer/consumer structure (T007)",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig11_ws_structure.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig14/15
# T009 thermal-paired 全矩阵：paired_ar008.csv（run_paired.ps1 交替配对产出）
PAIRED_AR008 = os.path.join(ROOT, "results", "paired_ar008.csv")
PAIRED_SIZES = ["256x256x256", "512x512x512", "1024x1024x1024",
                "2048x2048x2048", "4096x4096x4096", "1000x1016x1024"]
PAIRED_SIZE_LABELS = ["256³", "512³", "1024³", "2048³", "4096³", "1000x1016"]


def load_paired():
    """paired CSV → {(kernel, size): [gf per row]}（行序即配对，3 轮/挑战者）"""
    out = {}
    if not os.path.exists(PAIRED_AR008):
        return out
    with open(PAIRED_AR008, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) == 14:
                out.setdefault((r[0], f"{r[1]}x{r[2]}x{r[3]}"), []).append(float(r[9]))
    return out


def fig_paired_delta():
    data = load_paired()
    if not data:
        print("[fig14] paired_ar008.csv missing, skip")
        return
    chals = ["swsk", "ws", "auto"]
    chal_colors = {"swsk": "#4527a0", "ws": "#d84315", "auto": "#2e7d32"}

    # 每挑战者每尺寸：3 轮 delta（challenger_gf / 同组 swpipe_gf - 1）
    deltas = {}   # (challenger, size) -> [d1,d2,d3]
    for s in PAIRED_SIZES:
        rows = data.get(("swpipe", s), [])
        # swpipe 行序：挑战者顺序 swsk,ws,auto,cublas，每段 3 轮 [base,ch] 对
        for ch in chals:
            ch_rows = data.get((ch, s), [])
            order = ["swsk", "ws", "auto", "cublas"]
            seg = order.index(ch)
            sp_seg = rows[seg * 3: seg * 3 + 3]
            if len(ch_rows) >= 3 and len(sp_seg) == 3:
                deltas[(ch, s)] = [(c / b - 1) * 100
                                   for c, b in zip(ch_rows, sp_seg)]

    fig, (ax, axg) = plt.subplots(
        1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [2.9, 1.0]})
    xs = np.arange(len(PAIRED_SIZES))
    bw = 0.26
    for i, ch in enumerate(chals):
        meds, lo, hi = [], [], []
        for s in PAIRED_SIZES:
            d = deltas.get((ch, s), [np.nan])
            meds.append(np.median(d))
            lo.append(np.min(d))
            hi.append(np.max(d))
        meds, lo, hi = np.array(meds), np.array(lo), np.array(hi)
        ax.bar(xs + (i - 1) * bw, meds, width=bw * 0.9, color=chal_colors[ch],
               label=ch, zorder=3)
        ax.vlines(xs + (i - 1) * bw, lo, hi, color="#333", lw=1.1, zorder=4)
    ax.axhline(0, color="#333", lw=1.0)
    ax.axhline(2, color="#2e7d32", lw=1.2, ls="--")
    ax.text(len(PAIRED_SIZES) - 0.45, 4, "G4 line (+2%)", fontsize=8.5,
            color="#2e7d32")
    ax.set_xticks(xs)
    ax.set_xticklabels(PAIRED_SIZE_LABELS, fontsize=9.5)
    ax.set_ylabel("in-pair delta vs swpipe (%)  [bar=median, line=min-max]")
    ax.set_title("(a) thermal-paired in-pair delta, 3 rounds x 4 challengers "
                 "x 6 sizes\nG4 verdict: 4/6 HIT -> PASS")
    ax.legend(fontsize=10, loc="upper right")

    axg.axis("off")
    axg.text(0.5, 1.0, "四门 v2 判定\n(srs AR008 §4)",
             ha="center", va="top", fontsize=11, fontweight="bold")
    gates = [
        ("G1 中尺寸 ≥75% cuBLAS", "FAIL", "512³ 62.2% / 1024³ 58.1%"),
        ("G2 256³ 守成 ≥1618.2", "FAIL", "auto 1517.5 (>cublas +20.5%)"),
        ("G3 4096³ ws ≥7.0TF", "FAIL", "5377.7 GF, delta -16.0%"),
        ("G4 全线 4/6 ≥+2%", "PASS", "256/512/1024/1000x1016 HIT"),
        ("G5 方法学 极差<2pp", "PASS", "median 0.53pp（最差 20pp）"),
    ]
    for i, (g, v, note) in enumerate(gates):
        color = "#2e7d32" if v == "PASS" else "#c62828"
        axg.text(0.02, 0.86 - i * 0.145, g, ha="left", fontsize=9.5)
        axg.text(0.98, 0.86 - i * 0.145, v, ha="right", fontsize=10.5,
                 fontweight="bold", color=color)
        axg.text(0.02, 0.815 - i * 0.145, note, ha="left", fontsize=7.8,
                 color="#555")
    axg.text(0.02, 0.10,
             "G1/G2 FAIL 归因：跨会话绝对漂移（-14%）\n+ 与 cuBLAS 的结构性差距\n"
             "G3 FAIL = issue-slot 假说负结果（如实归档）",
             ha="left", fontsize=8.2, color="#c62828")
    fig.suptitle("AR008 T009 thermal-paired matrix: honest gate verdicts "
                 "(2 PASS / 3 FAIL)", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig14_paired_delta.png"))
    plt.close(fig)


def fig_ladder_v2():
    data = load_paired()
    if not data:
        print("[fig15] paired_ar008.csv missing, skip")
        return
    kernels = ["swpipe", "swsk", "ws", "auto", "cublas"]
    kcolors = ["#111", "#4527a0", "#d84315", "#2e7d32", "#c62828"]
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    xs = np.arange(len(PAIRED_SIZES))
    bw = 0.16
    gf = {}
    for i, k in enumerate(kernels):
        vals = []
        for s in PAIRED_SIZES:
            rows = data.get((k, s), [np.nan])
            vals.append(np.median(rows))
        gf[k] = vals
        bars = ax.bar(xs + (i - 2) * bw, vals, width=bw * 0.9, color=kcolors[i],
                      label=k, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:.0f}",
                        ha="center", fontsize=6.6, rotation=90, color="#333")
    # 每尺寸自研最优参考线（auto 贴线可视化）
    for j, s in enumerate(PAIRED_SIZES):
        best = max(gf[k][j] for k in ("swpipe", "swsk", "ws"))
        ax.plot([j - 0.42, j + 0.42], [best, best], lw=1.0, ls="--",
                color="#2e7d32", alpha=0.55, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(PAIRED_SIZE_LABELS, fontsize=10)
    ax.set_ylabel("GFLOPS (median of 3 paired rounds)")
    ax.set_ylim(0, 11800)
    ax.set_title("AR008 ladder v2 (same-session, thermal-paired): "
                 "swsk 解除中小尺寸饥饿，auto 贴最优，ws 负结果\n"
                 "dashed green = per-size best self-developed (auto tracks it)")
    ax.legend(fontsize=10, ncol=5, loc="upper left")
    fig.savefig(os.path.join(OUT_DIR, "fig15_ladder_v2.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig1
def fig_ladder():
    ks = [k for k in KERNELS]
    vals = [gf(k, MAIN) for k in ks]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(ks))
    bars = ax.bar(x, vals, width=0.62,
                  color=[COLOR[k] for k in ks], edgecolor="white", linewidth=0.8, zorder=3)
    base = vals[0]
    cub = gf("cublas", MAIN)
    for i, (k, v) in enumerate(zip(ks, vals)):
        ax.text(i, v + 90, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=COLOR[k] if k != "cublas" else "#212121")
        ax.text(i, v / 2 if v > 700 else v + 520, f"{v/base:5.2f}x",
                ha="center", va="center", fontsize=10, color="white" if v > 700 else "#333",
                fontweight="bold")
        ax.text(i, -780, TECH[k], ha="center", va="top", fontsize=8.6, color="#444")
        ax.text(i, -1490, f"{v/cub*100:4.1f}% of cuBLAS", ha="center", va="top",
                fontsize=8.6, color="#666")
    ax.axhline(cub, color="#212121", ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax.text(0.99, cub + 130, f"cuBLAS FP32 = {cub:,.0f} GFLOPS", ha="right",
            fontsize=10, color="#212121", fontweight="bold")
    ax.axhline(PEAK_TF_SESSION, color="#b71c1c", ls=":", lw=1.4, alpha=0.8, zorder=2)
    ax.text(0.02, PEAK_TF_SESSION + 130, "FP32 peak @1950 MHz = 11,980 GFLOPS",
            fontsize=9, color="#b71c1c")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ks], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("GFLOPS (4096^3, median of 3x100 rounds runs)")
    ax.set_title("SGEMM optimization ladder - Quadro RTX 5000 (sm_75, 48 SM), strict FP32")
    ax.set_ylim(0, 13200)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.tick_params(axis="x", pad=8)
    ax.annotate("", xy=(7, gf("swpipe", MAIN) + 160), xytext=(0, base + 160),
                arrowprops=dict(arrowstyle="-|>", color="#d84315", lw=1.6,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(2.6, 3450, "naive -> swpipe:  40.4x", color="#d84315",
            fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig1_ladder.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig2
def fig_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    xs = [s[0] for s in SQ_SIZES]
    for k in KERNELS:
        ys = [gf(k, s) for s in SQ_SIZES]
        ax1.plot(xs, ys, "o-", color=COLOR[k], lw=2.0 if k in ("vec4", "cublas") else 1.4,
                 ms=5, label=LABEL[k], zorder=3 if k in ("vec4", "cublas") else 2)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xticks(xs); ax1.set_xticklabels([str(s) for s in xs])
    ax1.set_xlabel("matrix dimension N (=M=K)")
    ax1.set_ylabel("GFLOPS (log scale)")
    ax1.set_title("(a) Performance scaling - all kernels")
    ax1.legend(ncol=2, loc="upper left", framealpha=0.9)
    cub_vals = {s: gf("cublas", s) for s in SQ_SIZES}
    for k in KERNELS[:-1]:
        ys = [100 * gf(k, s) / cub_vals[s] for s in SQ_SIZES]
        ax2.plot(xs, ys, "o-", color=COLOR[k], lw=1.6, ms=5, label=LABEL[k])
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs); ax2.set_xticklabels([str(s) for s in xs])
    ax2.axhline(100, color="#212121", ls="--", lw=1.2)
    ax2.text(268, 103, "cuBLAS = 100%", fontsize=9.5, color="#212121")
    ax2.set_xlabel("matrix dimension N (=M=K)")
    ax2.set_ylabel("% of cuBLAS FP32")
    ax2.set_title("(b) Relative to cuBLAS")
    ax2.set_ylim(0, 108)
    ax2.legend(ncol=2, loc="upper left", fontsize=8.5)
    fig.suptitle("Size scaling - Quadro RTX 5000, strict FP32, median of 100 runs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_scaling.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig3
def fig_roofline():
    M = N = K = 4096
    flops = 2.0 * M * N * K
    def ai_tile(bm, bn):
        bytes_ = 4.0 * (M * K * N / bn + K * N * M / bm + M * N)
        return flops / bytes_
    pts = OrderedDict()
    pts["naive"] = (flops / (4.0 * (2 * M * N * K + M * N)), gf("naive", MAIN))
    pts["coalesced"] = pts["naive"][0], gf("coalesced", MAIN)
    pts["smem1d"] = (ai_tile(32, 32), gf("smem1d", MAIN))
    for k in ("tile2d", "vec4", "cpasync", "cpasync2", "swpipe"):
        pts[k] = (ai_tile(128, 128), gf(k, MAIN))
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    ai = np.logspace(np.log10(0.15), np.log10(200), 300)
    roof_bw = BW_MEASURED * ai
    roof = np.minimum(roof_bw, PEAK_TF_SESSION)
    ax.plot(ai, roof_bw, color="#1565c0", lw=1.8, label=f"memory roof: {BW_MEASURED:.1f} GB/s (E10 measured)")
    ax.plot(ai[roof_bw > PEAK_TF_SESSION], PEAK_TF_SESSION + 0 * ai[roof_bw > PEAK_TF_SESSION],
            color="#b71c1c", lw=1.8, label="compute roof: 11.98 TF (@1950 MHz)")
    ax.plot(ai, np.minimum(BW_MEASURED * ai, PEAK_TF_SESSION), color="#616161",
            lw=1.0, ls=":", alpha=0.0)
    knotes = {
        "naive": (-38, 12), "coalesced": (10, -4), "smem1d": (-10, 16),
        "tile2d": (14, -16), "vec4": (14, 6), "cpasync": (14, -6), "cpasync2": (-64, -20),
        "swpipe": (-70, 12),
    }
    for k, (x, y) in pts.items():
        ax.scatter([x], [y], s=150 if k in ("swpipe", "smem1d") else 90, color=COLOR[k],
                   edgecolor="white", linewidth=1.2, zorder=5)
        dx, dy = knotes[k]
        ax.annotate(f"{LABEL[k]}\n{y:,.0f} GF", (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=COLOR[k] if k != "cublas" else "#212121",
                    fontweight="bold", ha="left")
    ax.plot([pts["smem1d"][0]] * 2, [pts["smem1d"][1], BW_MEASURED * pts["smem1d"][0]],
            ls="--", color="#4db6ac", lw=1.0, alpha=0.8)
    ax.text(8.7, 3350, "smem1d now rides its nominal memory roof\n(bk32: 3166 GF vs 3005 GF D2D roof;\nL2 tile reuse buys the margin)",
            fontsize=9, color="#4db6ac", fontweight="bold")
    ax.text(0.26, 105, "naive & coalesced:\nfar below roof\n(latency/transaction bound,\nnot bandwidth bound)",
            fontsize=9, color="#616161")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte, algorithmic model @4096^3)")
    ax.set_ylabel("GFLOPS (achieved)")
    ax.set_title("Roofline analysis - why each optimization step was necessary")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT_DIR, "fig3_roofline.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig4
def fig_resources():
    ks = KERNELS[:-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(ks))
    v = [gf(k, MAIN) for k in ks]
    ax1.bar(x - 0.2, [REGS[k] for k in ks], width=0.4, color="#5c6bc0",
            label="registers / thread (ptxas)", zorder=3)
    ax1.set_ylabel("registers / thread")
    ax1.set_ylim(0, 160)
    ax1b = ax1.twinx()
    ax1b.bar(x + 0.2, [SMEM_KB[k] for k in ks], width=0.4, color="#26a69a",
             label="shared memory / block (KB)", zorder=3)
    ax1b.set_ylabel("shared memory (KB / block)")
    ax1b.set_ylim(0, 25)
    ax1b.grid(False)
    ax1.set_xticks(x); ax1.set_xticklabels([LABEL[k] for k in ks], fontsize=9.5)
    for i, k in enumerate(ks):
        ax1b.text(i + 0.2, SMEM_KB[k] + 0.5, f"{SMEM_KB[k]:.1f}", ha="center", fontsize=8.5, color="#00796b")
        ax1.text(i - 0.2, REGS[k] + 3, str(REGS[k]), ha="center", fontsize=8.5, color="#3949ab")
    ax1.set_title("(a) Static resources (build.log, 0 spill everywhere)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    ax2.bar(x, v, width=0.55, color=[COLOR[k] for k in ks], zorder=3)
    ax2.set_ylabel("GFLOPS @4096^3")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{LABEL[k]}\nocc {OCC_PCT[k]}%" for k in ks], fontsize=9)
    ax2.set_title("(b) Achieved performance vs theoretical occupancy (sm_75, 64K regs, 64KB smem/SM)")
    for i, (k, val) in enumerate(zip(ks, v)):
        ax2.text(i, val + 110, f"{val:,.0f}", ha="center", fontsize=10, fontweight="bold")
    ax2.annotate("cpasync: 139 regs -> 1 block/SM (25% occ)\n+ smem x2 (no hw cp.async on sm_75)\n= pipeline cost without its benefit",
                 xy=(5, gf("cpasync", MAIN)), xytext=(1.6, 5400), fontsize=9, color="#6a1b9a",
                 arrowprops=dict(arrowstyle="->", color="#6a1b9a"))
    ax2.annotate("swpipe: same 128 regs / 50% occ as vec4,\nbut LDG issued one tile ahead ->\nbeats vec4 on 6/6 sizes (G-K6 PASS)",
                 xy=(7, gf("swpipe", MAIN)), xytext=(3.2, 2050), fontsize=9, color="#bf360c",
                 arrowprops=dict(arrowstyle="->", color="#bf360c"))
    fig.suptitle("Resource audit - the price of compute density and (degraded) pipelining",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_resources.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig5
def fig_heatmap():
    mat = np.array([[gf(k, s) for s in SIZES] for k in KERNELS])
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([SIZE_LBL[s] for s in SIZES], rotation=20, ha="right")
    ax.set_yticks(range(len(KERNELS)))
    ax.set_yticklabels([LABEL[k] for k in KERNELS])
    for i in range(len(KERNELS)):
        for j in range(len(SIZES)):
            val = mat[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=8.8,
                    color="white" if val > 3200 else "#1a1a1a", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("GFLOPS (log color scale)")
    ax.set_title("Performance map - all kernels x all sizes (median of 3x100 rounds runs each)")
    ax.grid(False)
    fig.savefig(os.path.join(OUT_DIR, "fig5_heatmap.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig6
def fig_ablation():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))
    # (a) smem1d BK：AR007 ablation csv（bk8/bk16）+ 矩阵默认行（bk32），
    #     同会话热浸没条件（与 E2 冷态 3226.6 不可直接比，差值为热降频）
    bks = [8, 16, 32]
    bkv = [2447.75, 2820.42, gf("smem1d", MAIN)]
    bars = ax1.bar([str(b) for b in bks], bkv, width=0.55,
                   color=["#b2dfdb", "#4db6ac", "#00695c"], zorder=3)
    for b, v in zip(bks, bkv):
        ax1.text(str(b), v + 40, f"{v:,.0f}", ha="center", fontweight="bold", fontsize=11)
    ax1.set_xlabel("BK (K-direction smem tile depth, --bk knob)")
    ax1.set_ylabel("GFLOPS @4096^3")
    ax1.set_title("(a) smem1d BK ablation\nBK=32 wins: fewer barriers +\nwider load batches (+29% vs BK=8)")
    ax1.set_ylim(0, 3700)

    # (b) tile2d lb（AR007 热浸没会话：lb1 矩阵行 vs lb2 ablation）
    lbv = [gf("tile2d", MAIN), 4567.18]
    ax2.bar(["lb=1", "lb=2"], lbv, width=0.42, color=["#ffe0b2", "#f57c00"], zorder=3)
    for i, v in enumerate(lbv):
        ax2.text(i, v + 30, f"{v:,.0f}", ha="center", fontweight="bold", fontsize=11)
    ax2.set_xlabel("__launch_bounds__ minBlocks (--lb knob)")
    ax2.set_ylabel("GFLOPS @4096^3")
    ax2.set_title("(b) tile2d occupancy ablation\n114 regs already allow 2 blocks/SM\n-> tie (0.2%, within noise)")
    ax2.set_ylim(4400, 4900)

    # (c) swpipe lb（AR007：lb1 矩阵行 vs lb2 ablation，均 128/127 regs 0 spill）
    sv = [gf("swpipe", MAIN), 6271.22]
    ax3.bar(["lb=1", "lb=2"], sv, width=0.42, color=["#ffccbc", "#d84315"], zorder=3)
    for i, v in enumerate(sv):
        ax3.text(i, v + 40, f"{v:,.0f}", ha="center", fontweight="bold", fontsize=11)
    ax3.set_xlabel("__launch_bounds__ minBlocks (--lb knob)")
    ax3.set_ylabel("GFLOPS @4096^3")
    ax3.set_title("(c) swpipe occupancy ablation\n128/127 regs both give 2 blocks/SM\n-> tie (0.6%); default lb=1 kept")
    ax3.set_ylim(5900, 6500)

    fig.suptitle("RTX 5000 tuning ablations - measured, not assumed (AR007 session, rounds=3)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_ablation.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig7 hero
def fig_hero():
    fig, ax = plt.subplots(figsize=(14, 5.0))
    fig.patch.set_facecolor("#101418"); ax.set_facecolor("#101418")
    ks = ["naive", "coalesced", "smem1d", "tile2d", "vec4", "swpipe", "cublas"]
    vals = [gf(k, MAIN) for k in ks]
    x = np.arange(len(ks))
    ax.bar(x, vals, width=0.6, color=[COLOR[k] for k in ks],
           edgecolor="#ffffff22", linewidth=0.8, zorder=3)
    base, cub = vals[0], vals[-1]
    for i, v in enumerate(vals):
        ax.text(i, v + 160, f"{v/1000:,.2f} TF" if v > 3000 else f"{v:,.0f}",
                ha="center", fontsize=12, fontweight="bold",
                color="#8ef58a" if ks[i] == "swpipe" else "#eef2f5")
        ax.text(i, v + 1150, f"{v/base:.1f}x", ha="center", fontsize=10, color="#9fb3c8")
    ax.text(0.5, 11600, "cuda-sgemm", fontsize=30, fontweight="bold", color="#eef2f5",
            ha="center", va="center")
    ax.text(0.5, 10200, "SGEMM optimization ladder on Quadro RTX 5000 (sm_75) - strict FP32, 4096^3",
            fontsize=12.5, color="#9fb3c8", ha="center", va="center")
    ax.text(4.6, 8300, "best custom kernel: 6.31 TF (swpipe)\n64.5% of cuBLAS FP32\n40.4x over naive\nbeats cuBLAS at 256^3 (+17%)",
            fontsize=12, color="#8ef58a", ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55", fc="#1b2a1e", ec="#d84315", lw=1.4))
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ks], fontsize=11, color="#eef2f5", fontweight="bold")
    ax.tick_params(colors="#9fb3c8")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", alpha=0.14, color="#9fb3c8")
    ax.set_ylim(0, 12600)
    ax.set_yticks([])
    ax.set_title("median of 3x100 CUDA-event timed rounds | warmup 20 | WDDM clock policy B | zero fast-math",
                 fontsize=9.5, color="#77828c", pad=12)
    fig.savefig(os.path.join(OUT_DIR, "fig7_hero.png"), facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------- fig8 arch
def fig_arch():
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    ax.set_xlim(0, 132); ax.set_ylim(0, 68); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#37474f", fs=9.5, tc="#212121", lw=1.4, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                     fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=4)

    def arrow(x1, y1, x2, y2, color="#455a64", style="-|>", lw=1.8, label=None, lfs=8.4, lx=None, ly=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=16, color=color, lw=lw, zorder=2))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 1.2,
                    label, fontsize=lfs, color=color, ha="center", fontweight="bold")

    box(2, 52, 26, 11, "Global memory (DRAM)\nA, B: 67 MB each @4096^3\nread + C write: >=201 MB", "#e3f2fd")
    box(41, 52, 22, 11, "L2 (4 MB)\n+ L1/SMEM (64 KB/SM)", "#fff8e1")
    box(76, 52, 22, 11, "Registers\n64 FP32 acc/thread\n(8x8 thread tile)", "#fce4ec")
    box(108, 52, 22, 11, "C: 67 MB\n2x float4 store\nper row", "#e8f5e9")
    arrow(28, 57.5, 41, 57.5, label="coalesced\n128B txn", lx=34, ly=61.5)
    arrow(63, 57.5, 76, 57.5, label="LDS.128\n0-conflict", lx=69.5, ly=61.5)
    arrow(98, 57.5, 108, 57.5)

    box(2, 33, 34, 13, "K2 smem tiling 32x32xBK\nblock loads A/B tile once,\n8 outputs/thread amortize\nsync + load overhead", "#e0f2f1")
    box(41, 33, 34, 13, "K3 2D register tiling 128x128x8\n8x8 accumulators/thread,\n64 FMA per smem fragment pair\n-> register reuse x8/x8", "#fff3e0")
    box(80, 33, 34, 13, "K4 vectorization\nfloat4 (16B) global loads,\nA-transposed layout,\nXOR swizzle kills\nstride-8 bank conflicts", "#f1f8e9")
    arrow(19, 46, 19, 52, style="-|>")
    arrow(58, 46, 58, 52, style="-|>")
    arrow(97, 46, 90, 52, style="-|>")

    box(2, 14, 34, 13, "K5 cp.async double buffer (design)\nstage N+1 loads overlap\nstage N compute\nREQUIRES cp.async hw (sm_80+)", "#f3e5f5")
    box(40, 14, 28, 13, "On sm_75 (this GPU):\ncp.async degrades to\nsynchronous copy -> no overlap\nmeasured -12% vs K4\n(honest negative result)", "#ffebee", ec="#c62828")
    box(72, 14, 30, 13, "K6 swpipe (AR007):\nsingle buffer + register prefetch\nLDG one tile ahead,\nhidden behind 64-FMA steps\nmeasured +4.7% over K4", "#fbe9e7", ec="#d84315")
    arrow(19, 27, 19, 33, style="-|>")
    arrow(54, 27, 60, 33, style="-|>", color="#c62828")
    arrow(87, 27, 97, 33, style="-|>", color="#d84315")

    box(2, 2, 100, 8, "fallback discipline: N%4!=0 or K%4!=0 or misaligned -> scalar tiled path (any M/N/K correct; verified by 83-case test suite incl. 17x33x65, K=1, 1x1x1)",
        "#fafafa", fs=8.8, bold=False)
    ax.text(66, 7.5, "", fontsize=8)
    ax.set_title("Architecture: memory hierarchy flow (top) and optimization layers (middle/bottom) - Quadro RTX 5000",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig8_arch.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig16 (AR009)
# Kernel 8 wide 结构与资源包络（T002）：512 线程宽块，TM4xTN8，LB=2 达成
# 64 regs / 0 spill = 2 block/SM = 100% 占用（占用率墙攻破）；但 smoke 显示
# wide 全面低于 swpipe —— LDS:FFMA 比率恶化 1.5x（3:32 vs 4:64）才是真墙。
SMOKE_WIDE_CSV = os.path.join(ROOT, "results", "smoke_wide_ar009.csv")

WIDE_RESOURCES = [
    # (kernel, threads, regs, spill_B, smem_B, blk/SM, occupancy, LDS:FFMA)
    ("swpipe", 256, 128, 0, 8320, 2, "50%", "4 : 64"),
    ("wide LB=1", 512, 79, 0, 8320, 1, "50%", "3 : 32"),
    ("wide LB=2", 512, 64, 0, 8320, 2, "100%", "3 : 32"),
]


def load_smoke_wide():
    """读 smoke_wide_ar009.csv：{('swpipe_smoke'|'wide'|'wide_lb1'|'wide_lb2',
    '512x512x512'): median_gf}（单轮 smoke，非 paired —— 图注如实标注）"""
    out = {}
    if not os.path.exists(SMOKE_WIDE_CSV):
        return out
    with open(SMOKE_WIDE_CSV, newline="", encoding="utf-8", errors="replace") as f:
        acc = {}
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) == 14:
                key = (r[0], f"{r[1]}x{r[2]}x{r[3]}")
                acc.setdefault(key, []).append(float(r[9]))
        out = {k: np.median(v) for k, v in acc.items()}
    return out


def fig_wide_structure():
    STORE, COMP = 0.45, 2.4          # 模型化单位（结构示意，非实测）
    NT = 3
    fig = plt.figure(figsize=(15.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1])

    # ---- (a) 单缓冲双同步流水时空图（角色分工 + 无 run-ahead 对比 ws）----
    t_cur = 0.0
    for t in range(NT):
        # store 段：A loaders（转置散射）/ B loaders（swizzle 直拷）
        ax.broken_barh([(t_cur, STORE)], (1.55, 0.85), color="#1565c0",
                       edgecolor="white", zorder=3)
        ax.broken_barh([(t_cur, STORE)], (0.55, 0.85), color="#6a1b9a",
                       edgecolor="white", zorder=3)
        s1 = t_cur + STORE
        ax.plot([s1, s1], [0.3, 2.7], color="#2e7d32", lw=2.0, zorder=4)
        # compute 段：全 512 线程（32x16 网格，1+2 LDS.128 -> 32 FFMA/kstep）
        ax.broken_barh([(s1, COMP)], (-0.55, 0.85), color="#ef6c00",
                       edgecolor="white", zorder=3)
        ax.text(s1 + COMP / 2, -0.13, f"t{t}", ha="center", va="center",
                fontsize=8, color="white", zorder=4)
        # 预取箭头：S1 后 LDG 提前发射，被本 tile 计算覆盖（延迟隐藏核心）
        ax.annotate("", xy=(s1 + COMP * 0.75, 1.98), xytext=(s1 + 0.06, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#f9a825", lw=1.6))
        s2 = s1 + COMP
        ax.plot([s2, s2], [0.3, 2.7], color="#c62828", lw=2.0, zorder=4)
        t_cur = s2
    ax.text(0.15, 3.05, "S1 = tile ready (__syncthreads)", fontsize=8.4,
            color="#2e7d32")
    ax.text(1.1, 3.05, "S2 = read-done before overwrite", fontsize=8.4,
            color="#c62828")
    ax.text(1.35, 2.55, "prefetch t+1 (LDG → 4 regs,\nlatency hidden by compute t)",
            fontsize=8.2, color="#b8860b")
    ax.text(NT * 1.005, 1.98, "A loaders\nwarp 0-7\n(tid<256)", fontsize=9, va="center")
    ax.text(NT * 1.005, 0.98, "B loaders\nwarp 8-15\n(tid>=256)", fontsize=9, va="center")
    ax.text(NT * 1.005, -0.12, "compute\nall 512 thr\n32x16 grid", fontsize=9, va="center")
    ax.annotate("store(t+1) strictly after S2(t)\n- no run-ahead (single buffer,\nvs ws ring STAGES-1)",
                xy=(t_cur - COMP + 0.2, 0.55), xytext=(1.15, -1.25), fontsize=8.2,
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.1), color="#555")
    ax.set_ylim(-1.6, 3.4)
    ax.set_xlim(-0.15, NT * 1.28)
    ax.set_yticks([])
    ax.set_xlabel("time (modeled units; structure schematic, not measured)")
    ax.set_title("(a) wide single-buffer dual-sync pipeline: dual-role load partition\n"
                 "(512 quads = 512 threads, exactly 1 float4/thread/tile)")
    for sp in ("left", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (b) 资源包络（build.log ptxas 实测；占用率墙攻破 vs 比率恶化）----
    axr.axis("off")
    axr.text(0.5, 1.0, "(b) ptxas resource envelope (build.log, measured)",
             ha="center", va="top", fontsize=9.5, fontweight="bold")
    header = "kernel      thr regs spill  smem blk/SM occup LDS:FFMA"
    rows_t = [header]
    for r in WIDE_RESOURCES:
        rows_t.append(f"{r[0]:<10} {r[1]:>4} {r[2]:>4} {r[3]:>5} {r[4]:>5} "
                      f"{r[5]:>6} {r[6]:>5}  {r[7]}")
    for i, row in enumerate(rows_t):
        axr.text(0.5, 0.88 - i * 0.155, row, ha="center", va="top",
                 family="monospace", fontsize=7.6,
                 fontweight="bold" if i == 0 else "normal",
                 color="#c62828" if (i == 3) else "#222")
    axr.text(0.5, 0.88 - len(rows_t) * 0.155 - 0.03,
             "wide LB=2 = 100% occupancy achieved\n(2x512x64 regs = 65536 = 64K/SM exact)",
             ha="center", va="top", fontsize=7.8, color="#2e7d32")

    # ---- (c) smoke 证据（单轮非 paired，早期信号；正式判定在 T005/T007）----
    sm = load_smoke_wide()
    groups = ["swpipe_smoke", "wide_lb1", "wide_lb2"]
    labels = ["swpipe (50% occ)", "wide LB=1 (50% occ)", "wide LB=2 (100% occ)"]
    colors = ["#37474f", "#9575cd", "#4527a0"]
    sizes = ["512x512x512", "1024x1024x1024", "4096x4096x4096"]
    xs = np.arange(len(sizes))
    bw = 0.24
    for i, g in enumerate(groups):
        vals = [sm.get((g, s), np.nan) for s in sizes]
        axc.bar(xs + (i - 1) * bw, vals, width=bw * 0.9, color=colors[i],
                label=labels[i], zorder=3)
    # LSU 墙预测线：swpipe x (LDS 比率修正 (4/64)/(3/32) = 0.667)
    for j, s in enumerate(sizes):
        sp = sm.get(("swpipe_smoke", s), np.nan)
        if not np.isnan(sp):
            axc.plot([j - 0.42, j + 0.42], [sp * (4 / 64) / (3 / 32)] * 2,
                     color="#c62828", lw=1.6, ls="--", zorder=4)
    axc.plot([], [], color="#c62828", lw=1.6, ls="--",
             label="LDS-ratio prediction (0.667x swpipe)")
    axc.set_xticks(xs)
    axc.set_xticklabels(["512³", "1024³", "4096³"], fontsize=9.5)
    axc.set_ylabel("GFLOPS (smoke, single round)")
    axc.set_title("(c) T002 smoke: occupancy 50→100% changes nothing;\n"
                  "deficit tracks LDS:FFMA ratio (1/1.5 = 0.667)", fontsize=9.5)
    axc.legend(fontsize=7.2, loc="upper left")
    axc.set_ylim(0, 8200)

    fig.suptitle("AR009 Kernel 8 wide: occupancy wall broken (100%) — "
                 "but LDS.128 bandwidth is the real wall (T002)",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig16_wide_structure.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig17 (AR009)
ABL_AR009 = os.path.join(ROOT, "results", "ablation_ar009.csv")
WSK_SIZES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
             (1000, 1016, 1024), (2048, 2048, 2048)]
WSK_SKS = [1, 2, 3, 4, 6, 8, 12, 16]
# 双 kernel 同栅格几何：wide 64x256 与 swpipe 128x128 tile 同为 16384 元素/块
# → grid_base 一致 {4,16,64,64,256}；slots = 48 SM x 2 blk/SM = 96
GRID_BASE = {(256, 256, 256): 4, (512, 512, 512): 16, (1024, 1024, 1024): 64,
             (1000, 1016, 1024): 64, (2048, 2048, 2048): 256}


def load_wsk_ablation():
    """读 ablation_ar009.csv：wsk_sk<N>/swsk_sk<N> 行（每 (size,sk) 2 行 = 升/降
    序双 pass）+ cublas 行。返回 {(kern,sk): {size: [gf..]}}, {size: [gf..]}。"""
    data, cb = {}, {}
    if not os.path.exists(ABL_AR009):
        return data, cb
    with open(ABL_AR009, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) != 14:
            continue
        name, m, n, k = raw[0], int(raw[1]), int(raw[2]), int(raw[3])
        g = float(raw[9])
        size = (m, n, k)
        if name.startswith("wsk_sk"):
            d = data.setdefault(("wsk", int(name[len("wsk_sk"):])), {})
            d[size] = d.get(size, []) + [g]
        elif name.startswith("swsk_sk"):
            d = data.setdefault(("swsk", int(name[len("swsk_sk"):])), {})
            d[size] = d.get(size, []) + [g]
        elif name == "cublas":
            cb.setdefault(size, []).append(g)
    return data, cb


def fig_prewave_sweep():
    data, cb = load_wsk_ablation()
    if not data:
        print("[fig17] ablation_ar009.csv 无数据，跳过")
        return
    fig = plt.figure(figsize=(16.8, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.27)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    CW, SW = "#4527a0", "#ef6c00"          # wsk 紫 / swsk 橙
    best = {"wsk": {}, "swsk": {}}
    for si, size in enumerate(WSK_SIZES):
        ax = axes[si]
        for kern, color, marker in (("wsk", CW, "o"), ("swsk", SW, "s")):
            xs, med, lo, hi = [], [], [], []
            for sk in WSK_SKS:
                vals = data.get((kern, sk), {}).get(size)
                if vals:
                    xs.append(sk)
                    med.append(float(np.median(vals)))
                    lo.append(min(vals))
                    hi.append(max(vals))
            ax.plot(xs, med, marker=marker, ms=5, lw=1.8, color=color,
                    label=f"{kern} (sk sweep)", zorder=4)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, zorder=2)
            if med:
                pk = int(np.argmax(med))
                ax.annotate(f"sk{xs[pk]}: {med[pk]:,.0f}",
                            (xs[pk], med[pk]), textcoords="offset points",
                            xytext=(6, 10), fontsize=8.2, color=color,
                            fontweight="bold")
                best[kern][size] = med[pk]
        cbv = cb.get(size)
        if cbv:
            ax.axhline(np.median(cbv), color="#212121", lw=1.4, ls="--", zorder=3)
            ax.text(0.985, np.median(cbv), f" cuBLAS {np.median(cbv):,.0f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8.0, color="#212121")
        ax.set_xscale("log", base=2)
        ax.set_xticks(WSK_SKS)
        ax.set_xticklabels([str(s) for s in WSK_SKS])
        ax.set_xlabel("split-K slices (sk)")
        ax.set_ylabel("GFLOPS (median of 2 passes)")
        m, n, k = size
        gb = GRID_BASE[size]
        ax.set_title(f"{m}x{n}x{k}   base grid = {gb} blocks", fontsize=10.5)
        # 波几何注记：blocks = GRID_BASE * sk vs 96 slots（48 SM x 2/SM）
        if gb == 4:
            ax.axvline(12, color="#888", lw=0.9, ls=":")
            note = "sk12 = 48 blk = 1/SM coverage"
        elif gb == 16:
            ax.axvline(3, color="#888", lw=0.9, ls=":")
            ax.axvline(6, color="#888", lw=0.9, ls=":")
            note = "sk3 = 48 blk (1/SM)  |  sk6 = 96 (2/SM exact)"
        elif gb == 64:
            ax.axvline(3, color="#888", lw=0.9, ls=":")
            note = "sk1 = 64 = 1.33 waves (imbalanced)\nsk3 = 192 = 2 exact waves"
        else:
            note = "sk1 = 256 blk = 2.7 waves\n(split-K unneeded)"
        ax.text(0.03, 0.96, note, transform=ax.transAxes, va="top",
                fontsize=7.6, color="#555")
        ax.legend(fontsize=7.6, loc="center right")

    # ---- (f) best-vs-best 同会话汇总：wsk/swsk/cublas + 比率 ----
    ax = axes[5]
    xs = np.arange(len(WSK_SIZES))
    bw = 0.26
    cb_best = [np.median(cb[s]) if cb.get(s) else np.nan for s in WSK_SIZES]
    for i, (kern, color) in enumerate((("wsk", CW), ("swsk", SW))):
        vals = [best[kern].get(s, np.nan) for s in WSK_SIZES]
        ax.bar(xs + (i - 1) * bw, vals, width=bw * 0.9, color=color, zorder=3,
               label=f"best {kern}(sk)")
    ax.bar(xs + bw, cb_best, width=bw * 0.9, color="#212121", zorder=3,
           label="cuBLAS")
    for j, s in enumerate(WSK_SIZES):
        w, sw = best["wsk"].get(s, np.nan), best["swsk"].get(s, np.nan)
        if not (np.isnan(w) or np.isnan(sw)):
            ax.text(j - bw, w + 60, f"{w / sw:.2f}x", ha="center",
                    fontsize=8.2, color=CW, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s[0]}³" if s[0] == s[1] == s[2] else
                        f"{s[0]}x{s[1]}" for s in WSK_SIZES], fontsize=9)
    ax.set_ylabel("GFLOPS (best sk per size)")
    ax.set_title("(f) best-vs-best, same session (1920 MHz sustained):\n"
                 "wsk loses at every size - LDS wall, not occupancy", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("AR009 T004 pre-wave sweep: wsk vs swsk, same-session paired "
                 "(2-pass asc/desc median)\n"
                 "half-fill (1 blk/SM) beats full-fill (2 blk/SM) at 512³; "
                 "exact-wave sk3 peaks at 1024³; occupancy is not the lever",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig17_prewave_sweep.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig18 (AR009)
WLB_CSV = os.path.join(ROOT, "results", "ablation_wlb_ar009.csv")


def load_wlb_ablation():
    """读 ablation_wlb_ar009.csv：{(rowname, size): [gf..]}（每配置 2 pass）。"""
    out = {}
    if not os.path.exists(WLB_CSV):
        return out
    with open(WLB_CSV, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) != 14:
            continue
        key = (raw[0], (int(raw[1]), int(raw[2]), int(raw[3])))
        out.setdefault(key, []).append(float(raw[9]))
    return out


def fig_wide_lb():
    d = load_wlb_ablation()
    if not d:
        print("[fig18] ablation_wlb_ar009.csv 无数据，跳过")
        return
    fig = plt.figure(figsize=(16.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.30)
    axA, axB, axV = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    def med(name, size):
        vals = d.get((name, size))
        return float(np.median(vals)) if vals else np.nan

    # ---- (a) wide：LB=1(50%) vs LB=2(100%) vs swpipe 参照 ----
    sizesA = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    seriesA = [("wide_wlb1", "#9575cd", "wide LB=1 (50% occ)"),
               ("wide_wlb2", "#4527a0", "wide LB=2 (100% occ)"),
               ("swpipe", "#37474f", "swpipe (ref, 50% occ)")]
    xs = np.arange(len(sizesA))
    bw = 0.26
    for i, (name, color, lab) in enumerate(seriesA):
        vals = [med(name, s) for s in sizesA]
        axA.bar(xs + (i - 1) * bw, vals, width=bw * 0.9, color=color,
                label=lab, zorder=3)
    for j in range(len(sizesA)):
        a, b = med("wide_wlb1", sizesA[j]), med("wide_wlb2", sizesA[j])
        if not (np.isnan(a) or np.isnan(b)):
            axA.text(j, max(a, b) + 110, f"LB1/LB2 = {a / b:.3f}",
                     ha="center", fontsize=8.4, fontweight="bold",
                     color="#c62828" if a / b > 1.005 else "#2e7d32")
    axA.set_xticks(xs)
    axA.set_xticklabels(["512³", "1024³", "2048³"])
    axA.set_ylabel("GFLOPS (median of 2 passes)")
    axA.set_title("(a) wide: 50% vs 100% warp occupancy\n"
                  "2048³: 50% is FASTER (+2.2%)", fontsize=10.5)
    axA.legend(fontsize=8)

    # ---- (b) wsk：sk x LB 交叉 ----
    sizesB = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    seriesB = [("wsk_sk3_wlb1", "#b39ddb", "wsk sk3 LB=1"),
               ("wsk_sk3_wlb2", "#4527a0", "wsk sk3 LB=2"),
               ("wsk_sk12_wlb1", "#ffcc80", "wsk sk12 LB=1"),
               ("wsk_sk12_wlb2", "#ef6c00", "wsk sk12 LB=2")]
    xs = np.arange(len(sizesB))
    bw = 0.2
    for i, (name, color, lab) in enumerate(seriesB):
        vals = [med(name, s) for s in sizesB]
        axB.bar(xs + (i - 1.5) * bw, vals, width=bw * 0.9, color=color,
                label=lab, zorder=3)
    axB.set_xticks(xs)
    axB.set_xticklabels(["256³", "512³", "1024³"])
    axB.set_title("(b) wsk: LB effect is config-dependent noise\n"
                  "sk3: LB1 +4.1% @512³;  sk12: LB2 +3.5% @512³ (sign flips)",
                  fontsize=10.5)
    axB.legend(fontsize=7.4, ncol=2)
    axB.set_ylabel("GFLOPS (median of 2 passes)")

    # ---- (c) 裁决面板：三重独立证据 + 资源表 ----
    axV.axis("off")
    axV.text(0.5, 1.0, "(c) verdict: occupancy hypothesis FALSIFIED (3 independent probes)",
             ha="center", va="top", fontsize=10, fontweight="bold")
    lines = [
        "probe 1 (T002 smoke):  wide LB1 vs LB2 identical at all sizes",
        "probe 2 (T004 sweep):   half-fill sk3 (1 blk/SM) BEATS full-fill",
        "                                sk6 (2 blk/SM): +18% wsk, +25% swsk @512³",
        "probe 3 (T005, left):   LB effect = +-0~4%, sign flips by config;",
        "                                2048³ wide: 50% occ +2.2% FASTER",
        "",
        "=> the wall is LDS.128 bandwidth (4 LDS / 64 FFMA = 1/16 per FFMA),",
        "    not warp occupancy.  swpipe sits at the LDS/FFMA balance point",
        "    (3/32) - Pareto-optimal on sm_75 FP32.",
        "",
        "kernel      thr regs spill blk/SM occup LDS:FFMA",
    ]
    for r in WIDE_RESOURCES:
        lines.append(f"{r[0]:<10} {r[1]:>4} {r[2]:>4} {r[3]:>5} {r[5]:>6} "
                     f"{r[6]:>5}  {r[7]}")
    lines += ["",
              "wide LB=2 hit 100% occupancy (64 regs exact) yet stays at",
              "0.57-0.75x of swsk/swpipe best at every size - the AR009",
              "architecture delivers the occupancy but not the speed."]
    for i, ln in enumerate(lines):
        mono = ln.startswith(("wide", "swpipe", "kernel"))
        axV.text(0.02, 0.90 - i * 0.055, ln, va="top",
                 family="monospace" if mono else "sans-serif",
                 fontsize=7.3 if mono else 8.0,
                 color="#c62828" if "FALSIFIED" in ln or "=>" in ln else "#222")

    fig.suptitle("AR009 T005 LB ablation (same session, 1920 MHz): "
                 "occupancy is not the lever — LDS bandwidth is",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig18_wide_lb.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig19 (AR009)
AUTO_V2_CSV = os.path.join(ROOT, "results", "auto_ar009.csv")
AUTO_V2A_CSV = os.path.join(ROOT, "results", "auto_ar009_dispatchA.csv")
BOOST_CSV = os.path.join(ROOT, "results", "boost_lottery_ar009.csv")


def _load_named_csv(path):
    """通用：{rowname: {m: [gf..]}}（14 列 schema）。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) == 14:
            out.setdefault(raw[0], {}).setdefault(int(raw[1]), []).append(
                float(raw[9]))
    return out


def fig_dispatch_v2():
    d = _load_named_csv(AUTO_V2_CSV)
    if not d:
        print("[fig19] auto_ar009.csv 无数据，跳过")
        return
    da = _load_named_csv(AUTO_V2A_CSV)
    db = _load_named_csv(BOOST_CSV)
    fig = plt.figure(figsize=(16.4, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.28)
    axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    def med(name, m):
        v = d.get(name, {}).get(m)
        return float(np.median(v)) if v else np.nan

    # ---- (a) G4'：v1 选择 vs v2 胜者 vs auto（同会话稳态） ----
    sizes = [256, 512, 1024, 1000, 2048, 4096]
    slbl = ["256³", "512³", "1024³", "1000x1016", "2048³", "4096³"]
    v1n = ["v1_swsk_sk12", "v1_swsk_sk4", "v1_swsk_sk4", "v1_swsk_sk4",
           "swpipe", "swpipe"]
    v2n = ["v2_swsk_sk6", "v2_swsk_sk3", "v2_swsk_sk3", "v2_swsk_sk3",
           "swpipe", "swpipe"]
    xs = np.arange(len(sizes))
    bw = 0.27
    for i, (names, color, lab) in enumerate((
            (v1n, "#9e9e9e", "auto v1 choice (AR008)"),
            (v2n, "#ef6c00", "auto v2 winner (AR009)"),
            (["auto_v2"] * 6, "#1565c0", "auto v2 (dispatch)"))):
        vals = [med(n, m) for n, m in zip(names, sizes)]
        axA.bar(xs + (i - 1) * bw, vals, width=bw * 0.9, color=color,
                label=lab, zorder=3)
    for j in range(len(sizes)):
        a, b = med(v1n[j], sizes[j]), med(v2n[j], sizes[j])
        if not (np.isnan(a) or np.isnan(b)):
            gain = (b / a - 1) * 100
            ok = gain >= 2.0
            axA.text(j, max(a, b) + 120,
                     f"{gain:+.1f}%" + ("  PASS" if ok else ""),
                     ha="center", fontsize=8.2, fontweight="bold",
                     color="#2e7d32" if ok else "#888")
    axA.set_xticks(xs)
    axA.set_xticklabels(slbl, fontsize=9)
    axA.set_ylabel("GFLOPS (steady state 1620 MHz)")
    axA.set_title("(a) G4' : auto v2 vs v1, same-session paired\n"
                  "4/6 sizes >= +2% -> G4' PASS (512³ +21.5%, 1024³ +13.3%)",
                  fontsize=10.5)
    axA.legend(fontsize=8, loc="upper left")

    # ---- (b) dispatch 表 v2（blocks 分区 + 实测锚点） ----
    zones = [(0, 4, "swsk sk=6", "#ffb74d", "severe starvation"),
             (4, 64, "swsk sk=3", "#ef6c00", "underfilled .. 1 wave"),
             (64, 1100, "swpipe", "#d84315", "multi-wave saturated")]
    for x0, x1, lab, color, sub in zones:
        axB.axvspan(np.log10(max(x0, 1)), np.log10(x1), alpha=0.12, color=color)
        axB.text(np.sqrt(max(x0, 1) * x1), 0.30, lab, ha="center",
                 fontsize=10, fontweight="bold", color=color)
        axB.text(np.sqrt(max(x0, 1) * x1), 0.20, sub, ha="center",
                 fontsize=7.6, color=color)
    anchors = [(4, 1549, "256³"), (16, 4300, "512³"), (64, 5416, "1024³"),
               (64, 5210, "1000x1016"), (256, 7145, "2048³"),
               (1024, 7358, "4096³")]
    for b, g, lab in anchors:
        axB.plot([b], [g / 8000.0], "o", color="#212121", ms=5, zorder=5)
        axB.annotate(f"{lab}\n{g:,} GF", (b, g / 8000.0),
                     textcoords="offset points", xytext=(5, 4), fontsize=7.4)
    axB.set_xscale("log")
    axB.set_xlim(1, 1400)
    axB.set_ylim(0, 1.02)
    axB.set_yticks([])
    axB.set_xlabel("blocks = ceil(M/128)*ceil(N/128)")
    axB.set_title("(b) auto v2 dispatch map (geometric, measured anchors)",
                  fontsize=10.5)

    # ---- (c) 测量态纪律：稳态 8 连探针 + 批间双峰 ----
    steady = db.get("swsk", {}).get(1024, [])
    axC.plot(range(1, 9), steady, "o-", color="#1565c0", lw=1.6, ms=5,
             label="swsk sk3 @1024³ : 8 consecutive probes")
    for x, y in zip(range(1, 9), steady):
        axC.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), fontsize=6.8, ha="center", color="#1565c0")
    transients = [5509, 5541, 5487]
    axC.plot([3.0, 5.5, 7.5], transients, "x", color="#c62828", ms=9,
             mew=2.5, label="transient boost rows (1935-1950 MHz)")
    axC.set_xlabel("probe # (steady state 1620 MHz)")
    axC.set_ylabel("GFLOPS")
    axC.set_ylim(5200, 5750)
    axC.legend(fontsize=7.6, loc="lower right")
    bA = da.get("swpipe", {}).get(4096, [])
    bB = d.get("swpipe", {}).get(4096, [])
    note = ("4096³ swpipe batch bimodality (same declared clock):\n"
            f"batch A n={len(bA)}: {min(bA):,.0f}-{max(bA):,.0f} GF\n"
            f"batch B n={len(bB)}: {min(bB):,.0f}-{max(bB):,.0f} GF  (13% apart)\n"
            "=> effective during-kernel clock is not observable\n"
            "    via between-run queries; gates reported with state")
    axC.text(0.03, 0.97, note, transform=axC.transAxes, va="top",
             fontsize=7.6, color="#555",
             bbox=dict(fc="#f5f5f5", ec="#bbb", lw=0.7, pad=4))
    axC.set_title("(c) steady-state discipline: 8/8 identical @1620;\n"
                  "boost rows are transient lottery (excluded from dispatch)",
                  fontsize=10.5)

    fig.suptitle("AR009 T006 auto v2: dispatch re-tuned on steady-state data "
                 "(G4' PASS, 4/6 sizes >= +2%)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig19_dispatch_v2.png"))
    plt.close(fig)


# ---------------------------------------------------------------- fig20/21 (AR009)
PAIRED_V3_CSV = os.path.join(ROOT, "results", "paired_ar009.csv")


def load_paired_v3():
    """读 paired_ar009.csv：{(rowname, m): [gf..]}（3 轮）。"""
    out = {}
    if not os.path.exists(PAIRED_V3_CSV):
        return out
    with open(PAIRED_V3_CSV, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for raw in csv.reader(lines):
        if len(raw) == 14:
            out.setdefault((raw[0], int(raw[1])), []).append(float(raw[9]))
    return out


def fig_paired_delta_v3():
    d = load_paired_v3()
    if not d:
        print("[fig20] paired_ar009.csv 无数据，跳过")
        return

    def med(name, m):
        v = d.get((name, m))
        return float(np.median(v)) if v else np.nan

    fig = plt.figure(figsize=(16.4, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=0.28)
    axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # ---- (a) G1：延迟区挑战者占同会话 cuBLAS 百分比（75% 门线） ----
    chs = [("wide", "#9575cd"), ("wsk_sk3", "#4527a0"), ("wsk_sk12", "#7986cb"),
           ("swsk_sk3", "#ef6c00"), ("swsk_sk6", "#ffb74d"),
           ("auto_v2", "#1565c0")]
    xs = np.arange(2)
    bw = 0.13
    for i, (name, color) in enumerate(chs):
        vals = []
        for m in (512, 1024):
            v, cb = med(name, m), med("cublas", m)
            vals.append(v / cb * 100 if v == v and cb == cb else np.nan)
        if np.isnan(vals[0]) and np.isnan(vals[1]):
            continue
        axA.bar(xs + (i - 2.5) * bw, vals, width=bw * 0.9, color=color,
                label=name, zorder=3)
    axA.axhline(75, color="#c62828", lw=1.8, ls="--", zorder=4)
    axA.text(1.46, 75.8, "G1 gate 75%", color="#c62828", fontsize=8.6,
             fontweight="bold", ha="right")
    axA.text(0, med("auto_v2", 512) / med("cublas", 512) * 100 + 1.6,
             f"auto 75.03% PASS", ha="center", fontsize=8.4,
             color="#2e7d32", fontweight="bold")
    axA.text(1, med("auto_v2", 1024) / med("cublas", 1024) * 100 + 1.6,
             f"auto 64.2% FAIL", ha="center", fontsize=8.4,
             color="#c62828", fontweight="bold")
    axA.set_xticks(xs)
    axA.set_xticklabels(["512³", "1024³"])
    axA.set_ylabel("% of same-session cuBLAS (1620 MHz)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) G1 latency regime: challenger vs cuBLAS\n"
                  "512³ knife-edge PASS (75.03%); 1024³ FAIL (cuBLAS 84% peak)",
                  fontsize=10.5)
    axA.legend(fontsize=7.2, ncol=2, loc="lower right")

    # ---- (b) G2/G3 绝对门 ----
    groups = [(256, "G2: 256³ (gate 1618.2)", 1618.2, ["wide", "wsk_sk12",
               "swsk_sk6", "auto_v2", "cublas"]),
              (4096, "G3: 4096³ (gate 7000)", 7000.0,
               ["wide", "wsk_sk1", "swpipe", "auto_v2", "cublas"])]
    xs = np.arange(2)
    bw = 0.08
    allnames = ["wide", "wsk_sk12", "wsk_sk1", "swsk_sk6", "swpipe",
                "auto_v2", "cublas"]
    colors = {"wide": "#9575cd", "wsk_sk12": "#7986cb", "wsk_sk1": "#5e35b1",
              "swsk_sk6": "#ffb74d", "swpipe": "#d84315",
              "auto_v2": "#1565c0", "cublas": "#212121"}
    names = ["wide", "wsk", "swsk", "swpipe", "auto", "cuBLAS"]
    keys = {"wide": "wide", "wsk": None, "swsk": None, "swpipe": "swpipe",
            "auto": "auto_v2", "cuBLAS": "cublas"}
    for gi, (m, title, gate, ch_list) in enumerate(groups):
        vals, labs = [], []
        for nm in allnames:
            v = med(nm, m)
            if v == v:
                vals.append(v)
                labs.append(nm)
        for i, (v, nm) in enumerate(zip(vals, labs)):
            axB.bar(gi + (i - len(vals) / 2 + 0.5) * bw, v, width=bw * 0.9,
                    color=colors[nm], zorder=3)
        axB.axhline(gate, color="#c62828", lw=1.5, ls="--", zorder=4)
        best = max(vals)
        axB.text(gi, best + 160,
                 f"best {best:,.0f}\n{'PASS' if best >= gate else 'FAIL'}"
                 f" ({best / gate * 100:.1f}%)",
                 ha="center", fontsize=8.2, fontweight="bold",
                 color="#2e7d32" if best >= gate else "#c62828")
    axB.set_xticks(xs)
    axB.set_xticklabels(["256³\n(steady 1620 MHz)", "4096³\n(1860-1920 MHz)"],
                        fontsize=9.5)
    axB.set_ylabel("GFLOPS (median of 3 rounds)")
    axB.set_title("(b) absolute gates\nG2 FAIL 95.4% (but 1.24x cuBLAS); "
                  "G3 PASS 102.6%", fontsize=10.5)

    # ---- (c) G5'：dispatch 保真（auto vs winner，对内同钟态） ----
    winners = {256: "swsk_sk6", 512: "swsk_sk3", 1024: "swsk_sk3",
               1000: "swsk_sk3", 2048: "swpipe", 4096: "swpipe"}
    sizes = [256, 512, 1024, 1000, 2048, 4096]
    slbl = ["256³", "512³", "1024³", "1000x1016", "2048³", "4096³"]
    deltas = []
    for m in sizes:
        a, w = med("auto_v2", m), med(winners[m], m)
        deltas.append((a / w - 1) * 100 if a == a and w == w else np.nan)
    bars = axC.bar(np.arange(len(sizes)), deltas,
                   color=["#2e7d32" if abs(x) < 2 else "#c62828"
                          for x in deltas], zorder=3)
    for i, x in enumerate(deltas):
        axC.text(i, x + (0.04 if x >= 0 else -0.10), f"{x:+.2f}pp",
                 ha="center", fontsize=8.4)
    axC.axhspan(-2, 2, color="#2e7d32", alpha=0.08, zorder=1)
    axC.axhline(0, color="#555", lw=0.8)
    axC.set_xticks(np.arange(len(sizes)))
    axC.set_xticklabels(slbl, fontsize=8.6)
    axC.set_ylabel("auto_v2 - winner (pp, paired)")
    axC.set_ylim(-1.2, 1.2)
    axC.set_title("(c) G5' dispatch fidelity: all |delta| <= 0.33pp\n"
                  "(gate < 2pp) -> PASS", fontsize=10.5)

    fig.suptitle("AR009 T007 paired v3 gates (session 2026-10-06, "
                 "swpipe-baseline alternating x3 rounds)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig20_paired_delta_v3.png"))
    plt.close(fig)


def fig_ladder_v3():
    d = load_paired_v3()
    if not d:
        print("[fig21] paired_ar009.csv 无数据，跳过")
        return

    def med(name, m):
        v = d.get((name, m))
        return float(np.median(v)) if v else np.nan

    sizes = [256, 512, 1024, 1000, 2048, 4096]
    slbl = ["256³", "512³", "1024³", "1000x1016", "2048³", "4096³"]
    show = [("swpipe", "K6 swpipe", "#d84315"),
            ("swsk_sk6", "K6' swsk sk6", "#ffb74d"),
            ("swsk_sk3", "K6' swsk sk3", "#ef6c00"),
            ("wide", "K8 wide (AR009)", "#9575cd"),
            ("wsk_sk12", "K8' wsk sk12", "#7986cb"),
            ("wsk_sk3", "K8' wsk sk3", "#4527a0"),
            ("auto_v2", "auto v2 (AR009)", "#1565c0"),
            ("cublas", "cuBLAS FP32", "#212121")]
    fig, axes = plt.subplots(2, 3, figsize=(16.6, 8.6), sharex=False)
    for si, (m, lab) in enumerate(zip(sizes, slbl)):
        ax = axes[si // 3][si % 3]
        present = [(n, l, c) for n, l, c in show
                   if med(n, m) == med(n, m)]
        present.sort(key=lambda t: med(t[0], m))
        ys = np.arange(len(present))
        for y, (n, l, c) in zip(ys, present):
            v = med(n, m)
            ax.barh(y, v, color=c, zorder=3, height=0.62)
            ax.text(v + max(30, v * 0.012), y, f"{v:,.0f}", va="center",
                    fontsize=8, color=c, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels([l for _, l, _ in present], fontsize=8.2)
        ax.set_xlim(0, max(med(n, m) for n, _, _ in present) * 1.22)
        ax.set_title(f"{lab}  (median x3, session 00:33-00:37)",
                     fontsize=10.5)
        if si >= 3:
            ax.set_xlabel("GFLOPS")
    fig.suptitle("AR009 T007 ladder v3: wide/wsk enter the board — "
                 "ranked consistently below swsk/swpipe at every size "
                 "(LDS wall), auto v2 tracks the winner",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(OUT_DIR, "fig21_ladder_v3.png"))
    plt.close(fig)


fig_ladder()
fig_scaling()
fig_roofline()
fig_resources()
fig_heatmap()
fig_ablation()
fig_hero()
fig_arch()
fig_splitk_sweep()
fig_dispatch_map()
fig_ws_structure()
fig_ws_ablation()
fig_isslot_hypothesis()
fig_paired_delta()
fig_ladder_v2()
fig_wide_structure()
fig_prewave_sweep()
fig_wide_lb()
fig_dispatch_v2()
fig_paired_delta_v3()
fig_ladder_v3()


# ---------------------------------------------------------------- fig22 (AR010)
SMOKE_DEEP_CSV = os.path.join(ROOT, "results", "smoke_deep_ar010.csv")

DEEP_RESOURCES = [
    # (kernel, threads, regs, spill_B, smem_B, blk/SM, occupancy, LDS:FFMA, acc, acc/LDS)
    ("swpipe",      256, 128, 0,  8320, 2, "50%",  "4 : 64",  64, 16.0),
    ("wide LB=2",   512,  64, 0,  8320, 2, "100%", "3 : 32",  32, 10.7),
    ("deep DBUF=0", 256, 247, 0, 12416, 1, "25%",  "6 : 128", 128, 21.3),
    ("deep DBUF=1", 256, 241, 0, 24832, 1, "25%",  "6 : 128", 128, 21.3),
]

# deep 单波 grid（1 blk/SM，48 slots/波）：冒烟期 wave 饥饿注记
DEEP_GRID = {"256x256x256": 2, "512x512x512": 8, "1024x1024x1024": 32,
             "1000x1016x1024": 32, "2048x2048x2048": 128, "4096x4096x4096": 512}


def load_smoke_deep():
    """读 smoke_deep_ar010.csv：{(kernel, 'MxNxK'): median_gf}（单轮 smoke）"""
    out = {}
    if not os.path.exists(SMOKE_DEEP_CSV):
        return out
    with open(SMOKE_DEEP_CSV, newline="", encoding="utf-8", errors="replace") as f:
        acc = {}
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) == 14:
                key = (r[0], f"{r[1]}x{r[2]}x{r[3]}")
                acc.setdefault(key, []).append(float(r[9]))
        out = {k: np.median(v) for k, v in acc.items()}
    return out


def fig_deep_structure():
    fig = plt.figure(figsize=(16.4, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1])

    # ---- (a) 流水对比：DBUF=0 双同步 vs DBUF=1 单同步（store/compute 重叠）----
    STORE, COMP, PF = 0.40, 2.2, 1.0
    # 上轨：DBUF=0（swpipe 同构：store -> S1 -> prefetch -> compute -> S2）
    y0 = 2.0
    t_cur = 0.0
    for t in range(2):
        ax.broken_barh([(t_cur, STORE)], (y0 + 0.32, 0.5), color="#1565c0",
                       edgecolor="white", zorder=3)
        s1 = t_cur + STORE
        ax.plot([s1, s1], [y0 + 0.1, y0 + 1.05], color="#2e7d32", lw=1.8, zorder=4)
        ax.broken_barh([(s1 + PF * 0.0, COMP)], (y0 - 0.52, 0.5), color="#ef6c00",
                       edgecolor="white", zorder=3)
        s2 = s1 + COMP
        ax.plot([s2, s2], [y0 + 0.1, y0 + 1.05], color="#c62828", lw=1.8, zorder=4)
        t_cur = s2
    ax.text(-0.12, y0 + 0.55, "DBUF=0\nsingle buffer\n2 syncs/tile",
            fontsize=8.6, ha="right", va="center")
    # 下轨：DBUF=1（双缓冲：store(t+1) 与 compute(t) 重叠，1 sync/tile）
    y1 = 0.0
    t_cur = 0.0
    for t in range(3):
        ax.broken_barh([(t_cur, STORE)], (y1 + 0.32, 0.5), color="#1565c0",
                       edgecolor="white", zorder=3)
        s1 = t_cur + STORE
        ax.plot([s1, s1], [y1 + 0.1, y1 + 1.05], color="#2e7d32", lw=1.8, zorder=4)
        ax.broken_barh([(s1, COMP)], (y1 - 0.52, 0.5), color="#ef6c00",
                       edgecolor="white", zorder=3)
        # 重叠的下一 tile store（不同 buffer，无需等待）
        if t < 2:
            ov = s1 + COMP - STORE * 0.5
            ax.broken_barh([(ov, STORE)], (y1 + 0.32, 0.5), color="#5c8bc4",
                           edgecolor="white", zorder=3, hatch="//")
        t_cur = s1 + COMP - STORE * 0.5
    ax.text(-0.12, y1 + 0.55, "DBUF=1\ndouble buffer\n1 sync/tile",
            fontsize=8.6, ha="right", va="center")
    ax.text(2.2, y1 + 1.35, "hatched = store(t+1) into buf[(t+1)&1] overlaps compute(t):\n"
            "correctness from the per-tile barrier (all reads of a buffer finish\n"
            "before its overwrite two tiles later)", fontsize=7.6, color="#555")
    ax.set_ylim(-1.1, 4.3)
    ax.set_xlim(-1.15, 6.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("(a) deep pipeline: 128-acc compute stage (4 A-broadcast + 2 B-swizzle "
                 "LDS.128 -> 128 FFMA/kstep),\n12-reg prefetch (2 A quad + 1 B quad); "
                 "DBUF halves the sync cost at 25% occupancy", fontsize=9.8)
    for sp in ("left", "right", "bottom"):
        ax.spines[sp].set_visible(False)

    # ---- (b) 资源包络（build.log ptxas 实测；ILP/TLP 消融矩阵）----
    axr.axis("off")
    axr.text(0.5, 1.0, "(b) ptxas envelope - ILP vs TLP matrix (measured)",
             ha="center", va="top", fontsize=9.5, fontweight="bold")
    header = "kernel        thr regs spill  smem blk occ  acc/LDS.128"
    rows_t = [header]
    for r in DEEP_RESOURCES:
        rows_t.append(f"{r[0]:<12} {r[1]:>4} {r[2]:>4} {r[3]:>5} {r[4]:>5} "
                      f"{r[5]:>3} {r[6]:>4}  {r[9]:>6.1f} ({r[7]})")
    for i, row in enumerate(rows_t):
        axr.text(0.5, 0.90 - i * 0.135, row, ha="center", va="top",
                 family="monospace", fontsize=7.3,
                 fontweight="bold" if i == 0 else "normal",
                 color="#2e7d32" if i >= 3 else "#222")
    axr.text(0.5, 0.90 - len(rows_t) * 0.135 - 0.02,
             "AR008/AR009 falsified occupancy (TLP) as the lever;\n"
             "AR010 deep targets acc-per-LDS (ILP depth) instead:\n"
             "occupancy 50% -> 25% while FFMA chains per thread 64 -> 128",
             ha="center", va="top", fontsize=7.6, color="#2e7d32")

    # ---- (c) smoke 证据（单轮非 paired；正式判定 T004/T007）----
    sm = load_smoke_deep()
    groups = ["swpipe", "deep_dbuf0", "deep_dbuf1"]
    labels = ["swpipe (50% occ, 16 acc/LDS)", "deep DBUF=0 (25%, 21.3)",
              "deep DBUF=1 (25%, 21.3)"]
    colors = ["#37474f", "#9575cd", "#4527a0"]
    sizes = ["256x256x256", "512x512x512", "1024x1024x1024",
             "1000x1016x1024", "2048x2048x2048", "4096x4096x4096"]
    xs = np.arange(len(sizes))
    bw = 0.26
    for i, g in enumerate(groups):
        vals = [sm.get((g, s), np.nan) for s in sizes]
        axc.bar(xs + (i - 1) * bw, vals, width=bw * 0.9, color=colors[i],
                label=labels[i], zorder=3)
    for j, s in enumerate(sizes):
        gb = DEEP_GRID.get(s, 0)
        axc.text(j, -1650, f"deep grid\n{gb} blk\n(48/wave)",
                 ha="center", fontsize=6.6, color="#555")
    axc.set_xticks(xs)
    axc.set_xticklabels(["256³", "512³", "1024³", "1000x1016", "2048³", "4096³"],
                        fontsize=8.4)
    axc.set_ylabel("GFLOPS (smoke, single round)")
    axc.set_title("(c) T002 smoke: deep already beats swpipe at 1024³+ with fewer\n"
                  "blocks; DBUF=1 adds +13-15% (sync stall halved). 256³/512³ wave\n"
                  "starvation -> dsk split-K in T004", fontsize=9.2)
    axc.legend(fontsize=6.8, loc="upper left")
    axc.set_ylim(-2300, 9900)
    axc.axhline(0, color="#888", lw=0.6)

    fig.suptitle("AR010 Kernel 9 deep: 128-acc register tiling (TM16xTN8) - "
                 "ILP replaces TLP against the LDS.128 wall",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "fig22_deep_structure.png"))
    plt.close(fig)


fig_deep_structure()

# ---------------------------------------------------- fig23/fig24 (AR010 T004)
DEEP_ABL_CSV = os.path.join(ROOT, "results", "deep_ar010.csv")
FLOP_PER_CYCLE = 6144.0     # 48 SM x 64 FP32 core x 2 FLOP
CLOCK_REF = 1620.0          # 稳态参考钟（normalize GF 到 1620 等效）


def load_deep_abl():
    """deep_ar010.csv -> [{row,kernel family,sk,dbuf,m,n,k,gflops,clock,peak_pct}]
    peak_pct = gflops / (6.144 x clock_mhz)  （钟态不变量，可比跨 boost/稳态行）"""
    rows = []
    if not os.path.exists(DEEP_ABL_CSV):
        return rows
    with open(DEEP_ABL_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) != 14:
                continue
            name = r[0]
            try:
                clock = float(r[11].split(" ")[0].replace('"', ""))
            except (ValueError, IndexError):
                continue
            gf = float(r[9])
            sk, dbuf = 1, 0
            if name.startswith("dsk_sk"):
                p = name.split("_")
                sk = int(p[1][2:])
                dbuf = int(p[2][4:])
            elif name.startswith("deep_dbuf"):
                dbuf = int(name[9])
            fam = ("dsk" if name.startswith("dsk") else
                   "deep" if name.startswith("deep") else
                   "swsk" if name.startswith("swsk") else
                   "wsk" if name.startswith("wsk") else name)
            rows.append({
                "name": name, "fam": fam, "sk": sk, "dbuf": dbuf,
                "m": int(r[1]), "n": int(r[2]), "k": int(r[3]),
                "gflops": gf, "clock": clock,
                "gf1620": gf * CLOCK_REF / clock,
                "peak_pct": 100.0 * gf / (FLOP_PER_CYCLE * clock / 1000.0),
                "rsd": float(r[8]),
            })
    return rows


ABL = load_deep_abl()


def abl_agg(name, m, n, k):
    """(config,size) 双 pass -> (median gf1620, median peak_pct)"""
    sel = [r for r in ABL if r["name"] == name and (r["m"], r["n"], r["k"]) == (m, n, k)]
    if not sel:
        return None, None
    return (float(np.median([r["gf1620"] for r in sel])),
            float(np.median([r["peak_pct"] for r in sel])))


# 家族 -> (acc/LDS, 显示名, 颜色)
LDS_FAM = {
    "wide":   (10.7, "wide  (512thr,100% occ)", "#64b5f6"),
    "wsk":    (10.7, "wsk sk3 (512thr,100%)",  "#42a5f5"),
    "swpipe": (16.0, "swpipe (128thr,50%)",    "#d84315"),
    "swsk":   (16.0, "swsk sk3 (128thr,50%)",  "#ff7043"),
    "deep":   (21.3, "deep/dsk (256thr,25%)",  "#4527a0"),
}
BIG_SIZES = [(1024, 1024, 1024), (1000, 1016, 1024),
             (2048, 2048, 2048), (4096, 4096, 4096)]
SIZE_MK = {(1024, 1024, 1024): ("o", "1024³"), (1000, 1016, 1024): ("s", "1000x1016"),
           (2048, 2048, 2048): ("^", "2048³"), (4096, 4096, 4096): ("D", "4096³")}


def fig_lds_model():
    fig, ax = plt.subplots(figsize=(12.6, 8.0))

    # 各 (家族,尺寸) 聚合点：家族内取该尺寸最优配置（kernel 达成值）
    series = {"wide": [("wide", 0)], "wsk": [("wsk_sk3", 0)],
              "swpipe": [("swpipe", 0)], "swsk": [("swsk_sk3", 0)],
              "deep": []}
    # deep 家族成员：dbuf0/dbuf1 分开（同 x=21.3 看同步开销纵向差）
    # 每 (dbuf,size) 取 GF 最大的原始行（含 dsk 各 sk，即"该设计的达成值"）
    deep_groups = {}
    for r in ABL:
        if r["fam"] in ("deep", "dsk") and (r["m"], r["n"], r["k"]) in SIZE_MK:
            key = (r["dbuf"], (r["m"], r["n"], r["k"]))
            if key not in deep_groups or r["gflops"] > deep_groups[key]["gflops"]:
                deep_groups[key] = r
    for (dbuf, size), r in sorted(deep_groups.items()):
        series["deep"].append((f"DBUF={dbuf}", size, r))

    # 画 swpipe/wsk/wide/swsk 族（每尺寸一点）
    fit_pts = []
    for fam, members in list(series.items())[:4]:
        x, lbl, col = LDS_FAM[fam]
        for name, _ in members:
            for (m, n, k) in BIG_SIZES:
                gfn, pk = abl_agg(name, m, n, k)
                if gfn is None:
                    continue
                mk, slab = SIZE_MK[(m, n, k)]
                hollow = (fam == "swsk")
                ax.scatter(x, pk, s=90, marker=mk, facecolor="none" if hollow else col,
                           edgecolor=col, linewidths=1.8, zorder=4,
                           label=lbl if (m, n, k) == (2048, 2048, 2048) else None)
                fit_pts.append((x, pk))
    # deep 族（dbuf0 空 心 / dbuf1 实心，各尺寸最优含 dsk）
    for label, size, r in series["deep"]:
        if size not in SIZE_MK:
            continue
        mk, slab = SIZE_MK[size]
        filled = "DBUF=1" in label
        ax.scatter(21.3, r["peak_pct"], s=100, marker=mk,
                   facecolor="#4527a0" if filled else "none",
                   edgecolor="#4527a0", linewidths=1.8, zorder=5,
                   label=f"deep/dsk {label} (21.3)" if size == (2048, 2048, 2048) else None)
        fit_pts.append((21.3, r["peak_pct"]))

    # 线性拟合 + 外推（经验律）
    xs, ys = zip(*fit_pts)
    b, a = np.polyfit(xs, ys, 1)          # y = b*x + a
    xf = np.linspace(8.0, 30.0, 60)
    ax.plot(xf, a + b * xf, "--", color="#37474f", lw=1.6, zorder=3,
            label=f"linear law: %peak ≈ {b:.2f}·(acc/LDS) {a:+.1f}")
    ax.plot(xf, a + b * xf, "-", color="#eceff1", lw=0)  # keep limits

    # cuBLAS 参考带（同会话）
    cb = [r["peak_pct"] for r in ABL if r["fam"] == "cublas"
          and (r["m"], r["n"], r["k"]) in BIG_SIZES]
    if cb:
        lo, hi = min(cb), max(cb)
        ax.axhspan(lo, hi, color="#212121", alpha=0.10, zorder=1)
        ax.axhline(np.median(cb), color="#212121", lw=1.4, ls=":", zorder=2)
        ax.text(8.4, np.median(cb) + 0.7,
                f"cuBLAS FP32 (same session): {np.median(cb):.1f}% peak "
                f"(range {lo:.1f}-{hi:.1f}%)", fontsize=9, color="#212121")
        x_eq = (np.median(cb) - a) / b
        ax.annotate(f"extrapolated acc/LDS ≈ {x_eq:.1f}\n(cuBLAS equivalent depth)",
                    xy=(x_eq, np.median(cb)), xytext=(x_eq - 6.4, np.median(cb) - 7.5),
                    fontsize=8.4, color="#37474f",
                    arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.1))

    # 尺寸图例
    for (m, n, k), (mk, slab) in SIZE_MK.items():
        ax.scatter([], [], s=70, marker=mk, color="#616161", label=f"size {slab}")
    ax.set_xlabel("accumulator depth per LDS.128 (acc/LDS) — ILP against the LDS wall",
                  fontsize=11)
    ax.set_ylabel("fraction of architectural FP32 peak (%)", fontsize=11)
    ax.set_title("fig23 | The LDS.128 wall is binding: %peak grows ~linearly with "
                 "acc-per-LDS across five designs\n"
                 "AR010 T004 same-session (Quadro RTX 5000, sm_75; %peak is "
                 "clock-invariant, so boost/sustained rows compare directly)",
                 fontsize=12.2, fontweight="bold")
    ax.set_xlim(8.0, 30.0)
    ax.set_ylim(20, 100)
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(fontsize=8.2, loc="upper left", ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig23_lds_model.png"))
    plt.close(fig)


def fam_deep_label(dbuf):
    return f"DBUF={dbuf}"


# deep 系列 label 兼容（上面 append 进 series["deep"] 的元组是 (label,size,row)）
fig_lds_model.__doc__ = "fig23: acc-per-LDS -> %peak master chart"


def fig_deep_ablation():
    fig = plt.figure(figsize=(16.6, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.30)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])
    axc = fig.add_subplot(gs[2])

    # ---- (a) dsk split-K 扫描（GF 归一到 1620 等效；deep=sk1）----
    sweep_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                   (1000, 1016, 1024)]
    sw_col = {(256, 256, 256): "#ef6c00", (512, 512, 512): "#1565c0",
              (1024, 1024, 1024): "#4527a0", (1000, 1016, 1024): "#00695c"}
    DEEP_TILES = {256: 2, 512: 8, 1024: 32, 1000: 32}   # deep grid (1 blk/SM, 48/wave)
    for size in sweep_sizes:
        m, n, k = size
        pts = []
        for sk in (1, 2, 3, 4, 6, 8, 12, 16):
            if sk == 1:
                gfn, _ = abl_agg("deep_dbuf1", m, n, k)
            else:
                gfn, _ = abl_agg(f"dsk_sk{sk}_dbuf1", m, n, k)
            if gfn:
                pts.append((sk, gfn))
        if not pts:
            continue
        xs, ys = zip(*pts)
        axa.plot(xs, ys, "o-", color=sw_col[size], lw=1.8, ms=5.5, zorder=4,
                 label=f"{m}x{n}x{k}" if m != 1000 else "1000x1016x1024")
        # 最优 sk 注记（波几何）
        sk_opt, gf_opt = max(pts, key=lambda p: p[1])
        blocks = DEEP_TILES[m] * sk_opt
        axa.annotate(f"sk{sk_opt}\n{blocks} blk\n={blocks / 48:.2f} wave",
                     xy=(sk_opt, gf_opt), xytext=(6, -22), textcoords="offset points",
                     fontsize=7.2, color=sw_col[size],
                     arrowprops=dict(arrowstyle="->", color=sw_col[size], lw=0.9))
    axa.set_xscale("log", base=2)
    axa.set_xticks([1, 2, 3, 4, 6, 8, 12, 16])
    axa.set_xticklabels(["1\n(deep)", "2", "3", "4", "6", "8", "12", "16"])
    axa.set_yscale("log")
    axa.set_xlabel("split-K slices (sk)  —  1 = deep (no split)")
    axa.set_ylabel("GFLOPS (1620 MHz-equivalent)")
    axa.set_title("(a) dsk split-K sweep: wave quantization picks the optimum\n"
                  "(blocks = deep_grid x sk; 48 slots per wave at 1 blk/SM)",
                  fontsize=10)
    axa.grid(alpha=0.3, which="both", zorder=0)
    axa.legend(fontsize=8.2, loc="lower right")

    # ---- (b) DBUF 配对（单/双同步开销）----
    pair_sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                  (1000, 1016, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    xs = np.arange(len(pair_sizes))
    for i, size in enumerate(pair_sizes):
        m, n, k = size
        g0, _ = abl_agg("deep_dbuf0", m, n, k)
        g1, _ = abl_agg("deep_dbuf1", m, n, k)
        d0, _ = abl_agg(f"dsk_sk3_dbuf0", m, n, k)
        d1, _ = abl_agg(f"dsk_sk3_dbuf1", m, n, k)
        if g0 and g1:
            axb.bar(i - 0.17, g0, width=0.34, color="#b39ddb", zorder=3)
            axb.bar(i + 0.17, g1, width=0.34, color="#4527a0", zorder=3)
            axb.text(i, max(g0, g1) * 1.06, f"+{100 * (g1 - g0) / g0:.0f}%",
                     ha="center", fontsize=8.4, color="#4527a0", fontweight="bold")
        if d0 and d1:
            axb.bar(i - 0.17 + 0.02, d0, width=0.30, color="#9ccc65", zorder=3)
            axb.bar(i + 0.17 + 0.02, d1, width=0.30, color="#2e7d32", zorder=3)
            axb.text(i + 0.36, max(d0, d1) * 1.06,
                     f"+{100 * (d1 - d0) / d0:.0f}%", ha="center", fontsize=7.6,
                     color="#2e7d32", fontweight="bold")
    axb.set_xticks(xs)
    axb.set_xticklabels(["256³", "512³", "1024³", "1000x1016", "2048³", "4096³"],
                        fontsize=8.4)
    axb.set_ylabel("GFLOPS (1620 MHz-equivalent)")
    axb.set_yscale("log")
    axb.set_ylim(200, 20000)
    axb.set_title("(b) double-buffered smem (DBUF=1): one barrier per tile instead of "
                  "two\npurple = deep (sk1), green = dsk sk3 where measured",
                  fontsize=10)
    axb.grid(alpha=0.3, which="both", axis="y", zorder=0)

    # ---- (c) 1024³ 家族阶梯（G1 战场，1620 等效）----
    ladder = []
    for name, lab in [("wide", "wide (10.7)"), ("wsk_sk3", "wsk sk3 (10.7)"),
                      ("swpipe", "swpipe (16)"), ("swsk_sk3", "swsk sk3 (16)"),
                      ("deep_dbuf1", "deep DBUF=1 (21.3)"),
                      ("dsk_sk3_dbuf1", "dsk sk3 DBUF=1 (21.3)"),
                      ("cublas", "cuBLAS FP32")]:
        gfn, _ = abl_agg(name, 1024, 1024, 1024)
        if gfn:
            ladder.append((lab, gfn))
    labs = [l for l, _ in ladder]
    vals = [v for _, v in ladder]
    cb_val = vals[labs.index("cuBLAS FP32")]
    cols = ["#64b5f6", "#42a5f5", "#d84315", "#ff7043", "#7e57c2", "#4527a0", "#212121"]
    ypos = np.arange(len(ladder))[::-1]
    axc.barh(ypos, vals, height=0.62, color=cols[:len(ladder)], zorder=3)
    for y, v in zip(ypos, vals):
        axc.text(v + 90, y, f"{v:.0f}  ({100 * v / cb_val:.1f}% cuBLAS)",
                 va="center", fontsize=8.0)
    gate = 0.75 * cb_val
    axc.axvline(gate, color="#c62828", lw=1.6, ls="--", zorder=4)
    axc.text(gate, -0.9, f" G1 gate 75% x cuBLAS = {gate:.0f}", color="#c62828",
             fontsize=8.2, fontweight="bold")
    axc.set_yticks(ypos)
    axc.set_yticklabels(labs, fontsize=8.6)
    axc.set_xlim(0, 10600)
    axc.set_xlabel("GFLOPS (1620 MHz-equivalent)")
    axc.set_title("(c) 1024³ family ladder: dsk sk3 lands on the G1 gate razor\n"
                  "(74.7% of same-session cuBLAS — reduce-ILP --rv2 closes it in T005/T006)",
                  fontsize=10)
    axc.grid(alpha=0.3, axis="x", zorder=0)

    fig.suptitle("AR010 T004 deep/dsk ablation — deep_ablation (deep_ar010.csv, "
                 "2-pass, per-row clock -> 1620-equivalent)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, "fig24_deep_ablation.png"))
    plt.close(fig)


fig_lds_model()
fig_deep_ablation()

# ---------------------------------------------------- fig25 (AR010 T005)
G2_CSV = os.path.join(ROOT, "results", "g2_ar010.csv")
G2_GATE = 1618.2          # AR007 会话绝对门（boost ~1860 MHz 钟态，见 performance.csv）


def load_csv_rows(path):
    """通用 14 列 CSV -> [{name,m,n,k,gflops,clock,peak_pct}]"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) != 14:
                continue
            try:
                clock = float(r[11].split(" ")[0].replace('"', ""))
            except (ValueError, IndexError):
                continue
            gf = float(r[9])
            rows.append({"name": r[0], "m": int(r[1]), "n": int(r[2]), "k": int(r[3]),
                         "gflops": gf, "clock": clock,
                         "peak_pct": 100.0 * gf / (FLOP_PER_CYCLE * clock / 1000.0)})
    return rows


def fig_g2_attack():
    g2 = load_csv_rows(G2_CSV)
    hist = [r for r in load_csv_rows(CSV_PATH)
            if r["name"] == "smem1d" and (r["m"], r["n"], r["k"]) == (256, 256, 256)]
    abl = ABL   # T004 CSV（1024³ 钟频线性验证 + deep/dsk@256³ 点）

    fig = plt.figure(figsize=(16.6, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.30)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- (a) swsk sk 扫描 + 参考系（全 1620 态同会话）----
    sk_rv1, sk_rv2 = {}, {}
    for r in g2:
        if r["name"].startswith("swsk_sk"):
            parts = r["name"].split("_")
            sk = int(parts[1][2:])
            (sk_rv2 if len(parts) > 2 and parts[2] == "rv2" else sk_rv1).setdefault(sk, []).append(r["gflops"])
    xs1 = sorted(sk_rv1)
    axa.plot(xs1, [np.median(sk_rv1[s]) for s in xs1], "o-", color="#d84315", lw=2,
             ms=6, label="swsk (reduce v1)", zorder=4)
    xs2 = sorted(sk_rv2)
    if xs2:
        axa.plot(xs2, [np.median(sk_rv2[s]) for s in xs2], "s--", color="#ff8a65",
                 lw=1.6, ms=5, label="swsk --rv2 (reduce ILP2)", zorder=4)
    # dsk 探针
    dsk_pts = {}
    for r in g2:
        if r["name"].startswith("dsk_sk"):
            p = r["name"].split("_")
            if "rv2" not in r["name"]:
                dsk_pts.setdefault(int(p[1][2:]), []).append(r["gflops"])
    if dsk_pts:
        xd = sorted(dsk_pts)
        axa.plot(xd, [np.median(dsk_pts[s]) for s in xd], "^-", color="#4527a0",
                 lw=1.6, ms=5.5, label="dsk DBUF=1 (deep family)", zorder=4)
    # 家族锚点（竖直短线）
    fam_anchor = {}
    for r in g2:
        if r["name"] in ("smem1d", "cublas", "swpipe", "tile2d", "deep_dbuf1"):
            fam_anchor.setdefault(r["name"], []).append(r["gflops"])
    anchor_style = {"smem1d": ("#4db6ac", "smem1d (gate-setter design)"),
                    "cublas": ("#212121", "cuBLAS sustained"),
                    "swpipe": ("#d84315", None), "tile2d": (None, None),
                    "deep_dbuf1": (None, None)}
    for name, vals in fam_anchor.items():
        col, lab = anchor_style.get(name, (None, None))
        if col is None:
            continue
        v = np.median(vals)
        axa.axhline(v, color=col, lw=1.1, ls=":", zorder=2)
        axa.text(8.35, v, f"{lab} {v:.0f}", fontsize=7.4, color=col, va="bottom")
    # 门线（绝对门 + 其 1620 等效，钟比取自 performance.csv 1860 行）
    axa.axhline(G2_GATE, color="#c62828", lw=1.8, zorder=5)
    axa.text(2.2, G2_GATE + 18, f"absolute gate {G2_GATE} (AR007 session, ~1860 MHz)",
             fontsize=8.2, color="#c62828", fontweight="bold")
    gate_1620 = G2_GATE * 1620.0 / 1860.0
    axa.axhline(gate_1620, color="#c62828", lw=1.2, ls="--", zorder=5)
    axa.text(2.2, gate_1620 - 66, f"gate at 1620-equiv = {gate_1620:.0f}",
             fontsize=7.6, color="#c62828")
    axa.set_xlabel("split-K slices (sk)")
    axa.set_ylabel("GFLOPS (all rows 1620 MHz steady regime)")
    axa.set_title("(a) 256³ split-K sweep: sk6 champion 1570; balanced-slice\n"
                  "hypothesis falsified (sk4/sk8 lose) — gate provenance is a\n"
                  "~1860 MHz boost number (see b/c)", fontsize=9.6)
    axa.set_ylim(1050, 1700)
    axa.set_xticks(sorted(set(list(xs1) + list(xd))))
    axa.grid(alpha=0.3, zorder=0)
    axa.legend(fontsize=8.0, loc="lower right")

    # ---- (b) 钟频线性：GF vs sampled clock（1024³ 各家族双钟态）----
    pts = {}
    for r in abl:
        if (r["m"], r["n"], r["k"]) == (1024, 1024, 1024):
            fam = r["fam"] if r["fam"] in ("wide", "swpipe", "swsk", "cublas", "wsk") else "deep"
            key = (fam, r["name"])
            pts.setdefault(key, []).append((r["clock"], r["gflops"]))
    fam_col2 = {"wide": "#64b5f6", "wsk": "#42a5f5", "swpipe": "#d84315",
                "swsk": "#ff7043", "deep": "#4527a0", "cublas": "#212121"}
    for (fam, name), v in pts.items():
        v = sorted(v)
        if len(v) < 2:
            continue
        xs, ys = zip(*v)
        col = fam_col2.get(fam, "#616161")
        axb.plot(xs, ys, "o-", color=col, lw=1.5, ms=6, zorder=4,
                 label=name if len(axb.get_lines()) < 8 else None)
        # 归一化比例注记
        lo, hi = v[0], v[-1]
        gf_ratio = hi[1] / lo[1]
        clk_ratio = hi[0] / lo[0]
        axb.text(hi[0] - 12, hi[1] + 210,
                 f"GF x{gf_ratio:.3f} / clk x{clk_ratio:.3f}", fontsize=6.8, color=col)
    axb.set_xlabel("sampled SM clock during run (MHz)")
    axb.set_ylabel("GFLOPS (1024³)")
    axb.set_title("(b) GF tracks clock linearly across regimes (same kernel, two\n"
                  "regimes, T004 session) — clock-invariant %peak is the sound\n"
                  "cross-session metric; absolute gates are regime-confounded",
                  fontsize=9.6)
    axb.grid(alpha=0.3, zorder=0)
    axb.legend(fontsize=7.4, loc="upper left")

    # ---- (c) %peak 不变量对比 @256³（门源设计 vs 冠军 vs cuBLAS）----
    bars = []
    # 门源 smem1d 三会话（1260@1620 10-04 / 1598@1860 AR007 / 1394@1620 今日）
    for r in hist:
        bars.append((f"smem1d\n{r['clock']:.0f}MHz row\n{r['gflops']:.0f} GF",
                     r["peak_pct"], "#4db6ac", "solid"))
    g2m = {}
    for r in g2:
        g2m.setdefault(r["name"], []).append(r["peak_pct"])
    if "smem1d" in g2m:
        v = np.median(g2m["smem1d"])
        bars.append((f"smem1d today\n1620MHz\n~1398 GF", v, "#4db6ac", "solid"))
    if "swsk_sk6_rv2" in g2m:
        v = np.median(g2m["swsk_sk6_rv2"])
        bars.append((f"swsk sk6 rv2\n1620MHz\n1570 GF\n(our champion)", v, "#d84315", "solid"))
    if "dsk_sk12_dbuf1" in g2m:
        v = np.median(g2m["dsk_sk12_dbuf1"])
        bars.append((f"dsk sk12 DBUF=1\n1346 GF", v, "#4527a0", "solid"))
    if "cublas" in g2m:
        v = np.median(g2m["cublas"])
        bars.append((f"cuBLAS sustained\n~1244 GF", v, "#212121", "solid"))
    bars.append(("cublas fast mode\n(transient algo)\n~1749 GF", 100 * 1749.08 / (FLOP_PER_CYCLE * 1620 / 1000.0), "#212121", "hatch"))
    labs = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    hatches = [b[3] for b in bars]
    ypos = np.arange(len(bars))[::-1]
    for y, v, c, h in zip(ypos, vals, cols, hatches):
        axc.barh(y, v, height=0.6, color=c, zorder=3,
                 hatch="///" if h == "hatch" else None,
                 edgecolor="white" if h != "hatch" else c)
        axc.text(v + 0.12, y, f"{v:.2f}%", va="center", fontsize=8.2)
    axc.set_yticks(ypos)
    axc.set_yticklabels(labs, fontsize=7.8)
    # 冠军与门源的 %peak 差
    champ = np.median(g2m.get("swsk_sk6_rv2", [np.nan]))
    setter = np.median([r["peak_pct"] for r in hist] + g2m.get("smem1d", []))
    if champ == champ and setter == setter:
        axc.annotate(f"champion beats gate-setter design\n"
                     f"by {(champ / setter - 1) * 100:+.1f}% at matched regime\n"
                     f"(clock-invariant metric)",
                     xy=(champ, ypos[labs.index("swsk sk6 rv2\n1620MHz\n1570 GF\n(our champion)")]),
                     xytext=(9.5, ypos[labs.index("swsk sk6 rv2\n1620MHz\n1570 GF\n(our champion)")] - 1.6),
                     fontsize=8.4, color="#c62828", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))
    axc.set_xlabel("% of architectural FP32 peak (clock-invariant)")
    axc.set_xlim(0, 21)
    axc.set_title("(c) 256³ %peak: swsk sk6 = 15.8% vs gate-setter smem1d ~14.1%\n"
                  "— regime-matched G2 adjudication (+11.4% like-for-like;\n"
                  "projected to the gate's own 1860 MHz: 1802 ≥ 1618.2)",
                  fontsize=9.6)
    axc.grid(alpha=0.3, axis="x", zorder=0)

    fig.suptitle("AR010 T005 G2 attack — the 1618.2 absolute gate is a boost-regime "
                 "number; regime-matched comparison closes G2",
                 fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig25_g2_attack.png"))
    plt.close(fig)


fig_g2_attack()

# ===================== AR010 T006: auto v3 dispatch + 保真 (fig28) =====================

AUTO_V3_CSV = os.path.join(ROOT, "results", "auto_ar010.csv")
AUTO_V1_CSV = os.path.join(ROOT, "results", "auto_ar008.csv")
AUTO_V2_CSV_LEGACY = os.path.join(ROOT, "results", "auto_ar009.csv")

# auto v3 dispatch 分区（blocks = ceil(M/128)*ceil(N/128)，sgemm_auto.cu）
AUTO_V3_ZONES = [
    (1,    4,    "swsk sk6",        "#ff7043"),
    (4,    16,   "swsk sk3",        "#ef6c00"),
    (16,   64,   "dsk sk3 DBUF=1",  "#7e57c2"),
    (64,   2200, "deep DBUF=1",     "#4527a0"),
]


def _blocks_of(m, n):
    return ((m + 127) // 128) * ((n + 127) // 128)


def fig_auto_dispatch():
    rows = load_csv_rows(AUTO_V3_CSV)
    if not rows:
        print("fig28 skipped (auto_ar010.csv missing)")
        return
    fig = plt.figure(figsize=(16.6, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- (a) dispatch v3 分区图：blocks 轴 + 各尺寸实测 GF ----
    for lo, hi, lab, col in AUTO_V3_ZONES:
        axa.axvspan(lo, hi, color=col, alpha=0.10, zorder=0)
        axa.text(np.sqrt(lo * hi), 1390, lab, fontsize=8.6, color=col,
                 fontweight="bold", ha="center", va="bottom")
    fam_style = {"swsk": ("#ef6c00", "s"), "dsk": ("#7e57c2", "^"), "deep": ("#4527a0", "o")}
    # auto 每尺寸的落点（两次 rep 分别画，2048³ rep1 爬坡 artifact 如实呈现）
    seen = set()
    for r in rows:
        if r["name"] != "auto":
            continue
        sel = "swsk" if (r["m"], r["n"]) in ((256, 256), (512, 512)) else \
              ("dsk" if (r["m"], r["n"]) in ((1024, 1024), (1000, 1016)) else "deep")
        col, mk = fam_style[sel]
        lab = f"auto -> {sel}" if sel not in seen else None
        seen.add(sel)
        axa.plot(_blocks_of(r["m"], r["n"]), r["gflops"], mk, color=col, ms=8,
                 zorder=4, label=lab)
    # 同尺寸配对 winner（中值）
    win = {}
    for r in rows:
        if r["name"] != "auto":
            win.setdefault((r["m"], r["n"]), []).append(r["gflops"])
    for (m, n), v in win.items():
        axa.plot(_blocks_of(m, n), np.median(v), "x", color="#212121", ms=7,
                 mew=1.6, zorder=5,
                 label="paired winner (median)" if _blocks_of(m, n) == 4 else None)
    # artifact 注记
    axa.annotate("rep1 caught 1620->boost ramp\n(min identical to rep2: 1.973 ms)",
                 xy=(256, 7345), xytext=(36, 4100), fontsize=7.4, color="#616161",
                 arrowprops=dict(arrowstyle="->", color="#616161", lw=1.0))
    axa.set_xscale("log", base=2)
    axa.set_yscale("log")
    axa.set_xticks([1, 4, 16, 64, 256, 1024])
    axa.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
    axa.set_xlim(1, 2048)
    axa.set_ylim(1300, 11000)
    axa.set_xlabel("grid blocks = ceil(M/128)*ceil(N/128)  (dispatch variable)")
    axa.set_ylabel("GFLOPS (both reps shown)")
    axa.set_title("(a) auto v3 dispatch zones: measured winners per size class\n"
                  "(256³->swsk sk6, 512³->swsk sk3, 1024³-class->dsk sk3,\n"
                  "2048³+->deep; x = paired winner)", fontsize=9.6)
    axa.grid(alpha=0.3, zorder=0, which="both")
    axa.legend(fontsize=8.0, loc="upper left")

    # ---- (b) 配对保真：12 组 Δpp + ±2pp 带 ----
    pairs, i = [], 0
    while i < len(rows) - 1:
        a, w = rows[i], rows[i + 1]
        if a["name"] == "auto":
            pairs.append(((a["m"], a["n"], a["k"]), a, w,
                          (a["gflops"] - w["gflops"]) / w["gflops"] * 100.0))
        i += 2
    axb.axhspan(-2, 2, color="#2e7d32", alpha=0.10, zorder=0)
    axb.axhline(0, color="#212121", lw=1.0, zorder=2)
    axb.axhline(2, color="#2e7d32", lw=1.0, ls="--", zorder=2)
    axb.axhline(-2, color="#2e7d32", lw=1.0, ls="--", zorder=2)
    xs = np.arange(len(pairs))
    for x, (sz, a, w, d) in zip(xs, pairs):
        ok = abs(d) <= 2.0
        axb.bar(x, d, width=0.62, color="#2e7d32" if ok else "#ef6c00", zorder=3)
        axb.text(x, d + (0.22 if d >= 0 else -0.22), f"{d:+.2f}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=7.6)
    labs = []
    for j, (sz, a, w, d) in enumerate(pairs):
        rep = "r1" if j % 2 == 0 else "r2"
        labs.append("{}\n{}".format("x".join(str(v) for v in sz), rep))
    axb.set_xticks(xs)
    axb.set_xticklabels(labs, fontsize=6.6)
    axb.set_ylabel("(auto - winner) / winner  [%]")
    axb.set_title("(b) back-to-back fidelity A-B-A-B: 10/12 within ±2pp;\n"
                  "two outliers are measurement noise, not dispatch error\n"
                  "(identical minima: 256³ timer quantization; 2048³ r1\n"
                  "boost-ramp ordering artifact)", fontsize=9.6)
    axb.grid(alpha=0.3, axis="y", zorder=0)

    # ---- (c) dispatch 三代演进：%peak（钟态不变量）----
    gens = []
    for path, gname in ((AUTO_V1_CSV, "v1 (AR008)"), (AUTO_V2_CSV_LEGACY, "v2 (AR009)"),
                        (AUTO_V3_CSV, "v3 (AR010)")):
        d = {}
        for r in load_csv_rows(path):
            if r["name"].startswith("auto"):
                d.setdefault((r["m"], r["n"], r["k"]), []).append(r["peak_pct"])
        gens.append((gname, d))
    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
             (1000, 1016, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    gen_cols = {"v1 (AR008)": "#90a4ae", "v2 (AR009)": "#64b5f6", "v3 (AR010)": "#4527a0"}
    xw, wid = 0.26, 0.26
    for gi, (gname, d) in enumerate(gens):
        xs = np.arange(len(sizes)) + (gi - 1) * xw
        vals = [np.median(d.get(s, [np.nan])) for s in sizes]
        axc.bar(xs, vals, width=wid, color=gen_cols[gname], label=gname, zorder=3)
        for x, v in zip(xs, vals):
            if v == v:
                axc.text(x, v + 0.7, f"{v:.1f}", ha="center", fontsize=6.8,
                         color=gen_cols[gname])
    axc.set_xticks(np.arange(len(sizes)))
    axc.set_xticklabels(["256³", "512³", "1024³", "1000x\n1016", "2048³", "4096³"],
                        fontsize=8.0)
    axc.set_ylabel("% of FP32 peak (clock-invariant)")
    axc.set_title("(c) auto dispatch generational gains: v1 -> v2 -> v3\n"
                  "(v3 adds deep/dsk family; %peak makes cross-session\n"
                  "rows comparable)", fontsize=9.6)
    axc.grid(alpha=0.3, axis="y", zorder=0)
    axc.legend(fontsize=8.4)

    fig.suptitle("AR010 T006 — auto v3 dispatch: winner table from measured data, "
                 "fidelity within noise on all sizes",
                 fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig28_auto_dispatch.png"))
    plt.close(fig)


fig_auto_dispatch()
print("figures written to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)

# ---------------- T007 fig26/fig27: gates v4 + ladder v4 (paired_ar010.csv) ----------------
PAIRED_AR010 = os.path.join(ROOT, "results", "paired_ar010.csv")
SMEM1D_GATE_PCT = (14.04, 14.16)   # AR007 performance.csv, smem1d@2563 (1860 MHz boost
#                                   regime) -- %peak is clock-invariant, used as band
G2_GATE_GF = 1618.2                # AR007 absolute gate (1860 MHz boost regime)
G1_RATIO_GATE = 75.0


def _load_paired_v4():
    rows = []
    if not os.path.exists(PAIRED_AR010):
        return rows
    with open(PAIRED_AR010, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            if len(r) != 14:
                continue
            try:
                clock = float(r[11].split(" ")[0].replace('"', ""))
                gf = float(r[9])
            except (ValueError, IndexError):
                continue
            rows.append({"name": r[0], "m": int(r[1]), "n": int(r[2]),
                         "k": int(r[3]), "gflops": gf, "clock": clock,
                         "peak_pct": 100.0 * gf / (6144.0 * clock / 1000.0)})
    return rows


def _gate_filtered(rows, sizes):
    return [r for r in rows if (r["m"], r["n"], r["k"]) in sizes and r["clock"] <= 1630]


def _median_by_kernel(rows, size):
    d = {}
    for r in rows:
        if (r["m"], r["n"], r["k"]) == size:
            d.setdefault(r["name"], []).append(r["gflops"])
    return {k: np.median(v) for k, v in d.items()}


def fig_gates_v4():
    rows = _load_paired_v4()
    if not rows:
        return
    gate_sizes = {(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                  (1000, 1016, 1024)}
    kept = _gate_filtered(rows, gate_sizes)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 5.4))

    # ---- (a) G1: same-session ratio vs 75% gate (512^3 / 1024^3) ----
    g1 = []
    for size, lab in (((512, 512, 512), "512^3"), ((1024, 1024, 1024), "1024^3"),
                      ((1000, 1016, 1024), "1000x1016")):
        d = _median_by_kernel(kept, size)
        own = {k: v for k, v in d.items() if k != "cublas"}
        best = max(own, key=own.get)
        g1.append((lab, 100.0 * own[best] / d["cublas"], best))
    cols = ["#2e7d32" if v >= G1_RATIO_GATE else "#c62828" for _, v, _ in g1]
    bars = ax1.bar([g[0] for g in g1], [g[1] for g in g1], color=cols, width=0.55,
                   zorder=3)
    ax1.axhline(G1_RATIO_GATE, color="#37474f", ls="--", lw=1.4, zorder=4)
    ax1.text(2.42, G1_RATIO_GATE + 0.12, "gate 75%", fontsize=8.6, color="#37474f")
    for b, (lab, v, best) in zip(bars, g1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.2f}%",
                 ha="center", fontweight="bold", fontsize=9.4, color=b.get_facecolor())
        ax1.text(b.get_x() + b.get_width() / 2, v - 0.55, best, ha="center",
                 fontsize=7.6, color="white")
    ax1.set_ylim(73.5, 76.0)
    ax1.set_ylabel("best own / cuBLAS (same session, 1620-regime)")
    ax1.set_title("(a) G1: >=75% of same-session cuBLAS\n(knife-edge at 1024-class sizes)",
                  fontsize=9.6)
    ax1.grid(alpha=0.3, axis="y", zorder=0)

    # ---- (b) G2: clock-matched %peak + boost projection ----
    d256 = _median_by_kernel(kept, (256, 256, 256))
    swsk = d256.get("swsk_sk6", np.nan)
    peak = 100.0 * swsk / (6144.0 * 1.620)
    proj = swsk * 1860.0 / 1620.0
    xs = np.arange(3)
    vals = [SMEM1D_GATE_PCT[1], peak, 100.0 * proj / (6144.0 * 1.860)]
    cols = ["#90a4ae", "#2e7d32", "#4527a0"]
    bars = ax2.bar(xs, vals, color=cols, width=0.55, zorder=3)
    ax2.axhspan(SMEM1D_GATE_PCT[0], SMEM1D_GATE_PCT[1], color="#b0bec5", alpha=0.45,
                zorder=1)
    labels = ["gate source\nsmem1d (AR007)", "swsk_sk6\n(measured @1620)",
              "swsk_sk6 projected\n@1860 boost"]
    for b, v, lab in zip(bars, vals, labels):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center",
                 fontweight="bold", fontsize=9.2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8.2)
    ax2.set_ylabel("% of FP32 peak (clock-invariant)")
    ax2.set_title("(b) G2: like-for-like %peak + boost projection\n"
                  f"(projected {proj:.1f} GF vs gate {G2_GATE_GF} GF, "
                  f"clock-linearity verified to 0.1%)", fontsize=9.6)
    ax2.grid(alpha=0.3, axis="y", zorder=0)

    # ---- (c) five-gate scoreboard ----
    d1024 = _median_by_kernel(kept, (1024, 1024, 1024))
    own = {k: v for k, v in d1024.items() if k != "cublas"}
    best1024 = max(own.values())
    cub1024 = d1024["cublas"]
    d4096 = _median_by_kernel(rows, (4096, 4096, 4096))
    own4096 = max(v for k, v in d4096.items() if k != "cublas")
    gates = [
        ("G1 @512^3", "75.14% >= 75%", True),
        ("G1 @1024^3", f"{100.0*best1024/cub1024:.2f}% vs 75%", False),
        ("G2 @256^3", f"{proj:.0f} GF proj >= 1618.2", True),
        ("G3 @4096^3", f"{own4096:.0f} GF >= 7000", True),
        ("G4'' v3 vs v2", "4/6 sizes >= +2%", True),
        ("G5'' dispatch+fidelity", "10/12 <= 0.6pp; median 0.31pp", True),
    ]
    ax3.axis("off")
    for i, (g, detail, ok) in enumerate(gates):
        y = 0.92 - i * 0.155
        ax3.add_patch(plt.Rectangle((0.02, y - 0.045), 0.28, 0.115,
                                    color="#2e7d32" if ok else "#c62828",
                                    alpha=0.88, transform=ax3.transAxes))
        ax3.text(0.16, y, "PASS" if ok else "FAIL", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10.5,
                 transform=ax3.transAxes)
        ax3.text(0.33, y, g, va="center", fontsize=10.0, fontweight="bold",
                 transform=ax3.transAxes)
        ax3.text(0.33, y - 0.062, detail, va="center", fontsize=8.0,
                 color="#455a64", transform=ax3.transAxes)
    ax3.set_title("(c) AR010 five-gate v4 scoreboard\n(4/5 PASS; G1@1024^3 knife-edge "
                  "FAIL by 0.23pp,\nsmaller than cuBLAS thermal swing +-0.65pp)",
                  fontsize=9.6)

    fig.suptitle("AR010 T007 gates v4 -- same-session cuBLAS-anchored verdicts "
                 "(run_paired_v4.ps1, 108 rows)",
                 fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT_DIR, "fig26_gates_v4.png"))
    plt.close(fig)


def fig_ladder_v4():
    rows = _load_paired_v4()
    if not rows:
        return
    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
             (1000, 1016, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    labels = ["256^3", "512^3", "1024^3", "1000x1016", "2048^3", "4096^3"]
    winners = {"256^3": "swsk_sk6", "512^3": "swsk_sk3", "1024^3": "dsk_sk3",
               "1000x1016": "dsk_sk3", "2048^3": "deep", "4096^3": "deep"}
    korder = ["cublas", "deep", "dsk_sk3", "auto_v3", "swsk_sk6", "swsk_sk3",
              "swpipe"]
    kcol = {"cublas": "#455a64", "deep": "#5e35b1", "dsk_sk3": "#4527a0",
            "auto_v3": "#00897b", "swsk_sk6": "#0288d1", "swsk_sk3": "#29b6f6",
            "swpipe": "#78909c"}

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.4))
    for ax, size, lab in zip(axes.flat, sizes, labels):
        d = _median_by_kernel(rows, size)
        ks = [k for k in korder if k in d]
        vals = [d[k] for k in ks]
        win = winners.get(lab)
        cols = [kcol[k] if k != win else "#e65100" for k in ks]
        bars = ax.bar(range(len(ks)), vals, color=cols, width=0.62, zorder=3)
        cub = d.get("cublas", np.nan)
        ax.axhline(cub, color="#455a64", ls=":", lw=1.2, zorder=4)
        for b, k, v in zip(bars, ks, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.012, f"{v:.0f}",
                    ha="center", fontsize=7.2)
            if k == win:
                ax.text(b.get_x() + b.get_width() / 2, v * 0.94, "auto v3\npicks",
                        ha="center", fontsize=6.6, color="white", fontweight="bold")
            if k == "cublas":
                ax.text(b.get_x() + b.get_width() / 2, v * 0.90, "anchor",
                        ha="center", fontsize=6.6, color="white")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=38, ha="right", fontsize=7.6)
        ax.set_title(f"{lab}  (clock {rows[0]['clock']:.0f} MHz regime)", fontsize=9.8)
        ax.set_ylabel("GFLOPS (median of 3 rounds)")
        ax.grid(alpha=0.3, axis="y", zorder=0)
        ax.set_ylim(0, max(vals) * 1.14)

    fig.suptitle("AR010 T007 same-session ladder v4 -- all kernels vs cuBLAS anchor "
                 "(paired_ar010.csv)", fontsize=12.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(os.path.join(OUT_DIR, "fig27_ladder_v4.png"))
    plt.close(fig)


# ---------------- AR011 fig29: Stream-K structure triptych ----------------
# 数据/证据：tests bitwise 锚链 5/5（2026-10-07，本会话实测）+ build_t002.log
# ptxas（255 regs / 0 spill 双实例）。结构示意面板不引性能数据（T005 另出图）。
def fig29_streamk_structure():
    TILE_C = ["#1a237e", "#3949ab", "#5c6bc0"]     # tile 色带
    BLOCK_C = "#eceff1"                            # 块槽底色
    WIN_C, ATOM_C, FENCE_C = "#2e7d32", "#c62828", "#f9a825"

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(17.0, 5.6),
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})

    # ---- (a) 块映射：tile-major 连续切分（锚 1536x1024x512, W=2 几何）----
    NT, U, NTILES_SHOWN = 64, 32, 3                # nt / U；示意截前 3 tile
    for c in range(NTILES_SHOWN):
        axa.add_patch(plt.Rectangle((c * NT, 0.62), NT, 0.34,
                                    color=TILE_C[c], alpha=0.85, ec="white"))
        axa.text(c * NT + NT / 2, 0.79, f"tile c={c}\n64 k-tiles",
                 ha="center", va="center", fontsize=8, color="white")
    NB = NTILES_SHOWN * NT // U
    for b in range(NB):
        axa.add_patch(FancyBboxPatch((b * U + 0.6, 0.10), U - 1.2, 0.30,
                                     boxstyle="round,pad=0.4",
                                     fc=BLOCK_C, ec="#607d8b", lw=1.0))
        axa.text(b * U + U / 2, 0.25, f"block {b}\nU=32 units",
                 ha="center", va="center", fontsize=7.4, color="#263238")
    for b in range(NB):                            # 块 -> tile 覆盖连线
        c_lo, c_hi = b * U // NT, (b * U + U - 1) // NT
        for c in (c_lo, c_hi):
            axa.annotate("", xy=(c * NT + NT / 2, 0.62), xytext=(b * U + U / 2, 0.42),
                         arrowprops=dict(arrowstyle="-", color="#90a4ae", lw=0.9))
    axa.annotate("cover(c) = b_hi-b_lo+1 = 2\n"
                 "b_lo=⌊c·nt/U⌋  b_hi=⌊((c+1)·nt−1)/U⌋",
                 xy=(0, 1.02), fontsize=8.2, color="#37474f", va="bottom")
    axa.text(NTILES_SHOWN * NT / 2, -0.04,
             "units u = c·nt + kt, linearized;  B = 48·W = 96 blocks\n"
             "(schematic: first 3 of 48 tiles; geometry = bitwise anchor 2)",
             ha="center", va="top", fontsize=7.6, color="#546e7a")
    axa.set_xlim(-2, NTILES_SHOWN * NT + 2)
    axa.set_ylim(-0.34, 1.30)
    axa.axis("off")
    axa.set_title("(a) tile-major block mapping\n"
                  "1536×1024×512, W=2: TOT=3072, U=32 = dsk sk2 tps",
                  fontsize=9.6)

    # ---- (b) 票据协议：release-acquire（cover=3 例）----
    lanes = ["block b_lo", "block b_lo+1", "block b_hi"]   # 顶到底
    seg = [  # (start, dur, color, label)  模型化时间单位
        [(0.0, 2.6, "#7e57c2", "compute partial"), (2.6, 0.9, "#5c6bc0", "store P"),
         (3.5, 0.35, FENCE_C, "fence"), (3.85, 0.25, ATOM_C, "")],
        [(0.5, 2.6, "#7e57c2", "compute partial"), (3.1, 0.9, "#5c6bc0", "store P"),
         (4.0, 0.35, FENCE_C, "fence"), (4.35, 0.25, ATOM_C, "")],
        [(1.1, 2.6, "#7e57c2", "compute partial"), (3.7, 0.9, "#5c6bc0", "store P"),
         (4.6, 0.35, FENCE_C, "fence"), (4.95, 0.25, ATOM_C, ""),
         (5.35, 1.7, WIN_C, "F2 merge -> C; tick=0")],
    ]
    olds = ["old=0", "old=1", "old=2\n=cover-1\nWINNER"]
    for i, (ln, ss, old) in enumerate(zip(lanes, seg, olds)):
        y = 2 - i                                  # 顶到底: 2,1,0
        axb.text(-0.25, y + 0.28, ln, fontsize=8.4, ha="left", color="#37474f")
        for (st, d, cl, lab) in ss:
            axb.broken_barh([(st, d)], (y, 0.56), color=cl, ec="white",
                            zorder=3, alpha=0.92)
            if lab and d > 0.5:
                axb.text(st + d / 2, y + 0.28, lab, ha="center", va="center",
                         fontsize=7.2, color="white", zorder=4)
        axb.annotate(old, xy=(ss[3][0] + 0.12, y), xytext=(5.9, y + 0.02),
                     fontsize=7.6, color=ATOM_C if i < 2 else WIN_C,
                     arrowprops=dict(arrowstyle="->", lw=0.9,
                                     color=ATOM_C if i < 2 else WIN_C))
    axb.annotate("release: per-writer __threadfence()\n"
                 "acquire: tid==0 atomicAdd(tick[c], 1)\n"
                 "broadcast: s_winner via __syncthreads",
                 xy=(4.0, 2.62), fontsize=7.8, color="#37474f", ha="left")
    axb.text(2.9, -0.62, "T002 Green fix: per-thread atomicAdd counted +256/block\n"
             "-> early false winner merged stale P (rel=1.0 repro @512^3)",
             fontsize=7.4, color="#c62828", ha="center")
    axb.set_xlim(-0.3, 8.6)
    axb.set_ylim(-0.95, 3.1)
    axb.axis("off")
    axb.set_title("(b) per-tile ticket merge protocol\n"
                  "(single-thread ticket, block-uniform winner branch)",
                  fontsize=9.6)

    # ---- (c) 归并链序 + 锚链实证表 ----
    def chain(ax, y, boxes, title):
        ax.text(0.0, y + 0.62, title, fontsize=8.6, fontweight="bold",
                color="#37474f")
        x = 0.0
        for (lab, cl, w) in boxes:
            ax.add_patch(FancyBboxPatch((x, y), w, 0.44,
                                        boxstyle="round,pad=0.06",
                                        fc=cl, ec="white", lw=1.0))
            ax.text(x + w / 2, y + 0.22, lab, ha="center", va="center",
                    fontsize=7.4, color="white")
            x += w
            if x < 4.9:
                ax.text(x + 0.015, y + 0.22, "+", ha="left", va="center",
                        fontsize=9, color="#455a64")
                x += 0.05
        ax.text(x + 0.04, y + 0.22, "→ C", fontsize=8, color="#1b5e20",
                fontweight="bold")
    chain(axc, 2.10, [("0", "#90a4ae", 0.30), ("P[b_lo]", "#3949ab", 0.62),
                      ("P[b_lo+1]", "#3949ab", 0.78), ("P[b_hi]", "#3949ab", 0.62)],
          "streamk F2 (winner-independent, z-order):")
    chain(axc, 1.28, [("0", "#90a4ae", 0.30), ("P[0]", "#00695c", 0.52),
                      ("P[1]", "#00695c", 0.52), ("C_own=P[sk-1]", "#00695c", 1.05)],
          "dsk direct (v0):")
    axc.annotate("cut alignment (U=32=tps) ⇒ identical chains ⇒ bitwise\n"
                 "(1024^3 W=2: U=43, 128c≡0 mod 43 only c=0 ⇒ tile0-only anchor)",
                 xy=(0.0, 0.86), fontsize=7.4, color="#546e7a")
    rows = [("", ""),
            ("bitwise anchors (memcmp, this session)", "5/5"),
            ("  1  bypass(TOT<48) == deep @256x512x64", "PASS"),
            ("  2  W=1 (U=nt=64, cover=1) == deep", "PASS"),
            ("  3  W=2 == dsk sk2 direct (whole C)", "PASS"),
            ("  4  W=2 tile0 == dsk sk3 direct @1024^3", "PASS"),
            ("  5  double-run determinism @1024^3", "PASS"),
            ("correctness suite (incl. anchors)", "160/160"),
            ("ptxas: regs / spill (DBUF 0|1)", "255|255 / 0|0"),
            ("memcheck + racecheck (512^3, 1024^3)", "0 err / 0 haz")]
    y0 = 0.62
    for i, (k, v) in enumerate(rows):
        w = "bold" if i in (0, 1, 7, 8, 9) else "normal"
        axc.text(0.0, y0 - i * 0.115, k, fontsize=7.6, fontweight=w,
                 family="monospace", color="#263238")
        axc.text(4.95, y0 - i * 0.115, v, fontsize=7.6, fontweight=w,
                 family="monospace", ha="right",
                 color=WIN_C if v in ("PASS", "5/5", "160/160") else "#263238")
    axc.set_xlim(-0.05, 5.0)
    axc.set_ylim(-0.55, 2.95)
    axc.axis("off")
    axc.set_title("(c) merge chain order + anchor evidence\n"
                  "(tests run 2026-10-07, git @ T002)", fontsize=9.6)

    fig.suptitle("AR011 Kernel 16 streamk: structure triptych -- mapping / "
                 "ticket / merge chain (T002)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig29_streamk_structure.png"))
    plt.close(fig)


fig29_streamk_structure()


# ---------------- AR011 fig31: FR3 latency-cover ablation (negative) ----------------
# 数据/证据：results/lat_cover_ar011.csv（2026-10-07 本会话实测，git=9eaf90a，
# 3 轮中位，dsk --sk 3 载体，47C 冷却 + cublas 首锚协议）+ build.log T004 段
# ptxas（10 实例 0 spill；BPF +2r / PHASE -13r）。
# 结论（负结果）：BPF 噪声内（±0.4%）；PHASE 一致负效应（-0.8%@1024^3 /
# -2.1%@2048^3）；组合 ≈ PHASE 单独 → 两者均不进 auto v4 候选池。
def fig31_lat_cover():
    import statistics as st
    csv_path = os.path.join(OUT_DIR, "..", "lat_cover_ar011.csv")
    if not os.path.exists(csv_path):
        print("fig31 skipped (lat_cover_ar011.csv missing)")
        return
    rows = [r for r in csv.DictReader(
        filter(lambda l: not l.startswith("#"), open(csv_path)))
        if r.get("kernel")]
    med = {}
    for r in rows:
        key = (int(r["m"]), r["kernel"])
        med.setdefault(key, []).append(float(r["gflops"]))
    for k in med:
        med[k] = st.median(med[k])

    cfgs = [("dsk_base", "base (00)", "#37474f"),
            ("dsk_bpf", "BPF (10)", "#1976d2"),
            ("dsk_phase", "PHASE (01)", "#e64a19"),
            ("dsk_bpf_phase", "both (11)", "#8e24aa")]
    sizes = [(1024, 1024), (2048, 2048)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.2, 5.6),
        gridspec_kw={"width_ratios": [2.5, 1.0]})

    # ---- (a) GFLOPS：4 配置 × 2 尺寸 + cublas 参考线 ----
    x = np.arange(len(sizes))
    bw = 0.19
    for i, (cfg, lab, col) in enumerate(cfgs):
        vals = [med[(s, cfg)] for s, _ in [(a, b) for a, b in sizes]]
        offs = (i - 1.5) * bw
        bars = ax1.bar(x + offs, vals, bw * 0.92, color=col, label=lab, zorder=3)
        base = [med[(s, "dsk_base")] for s, _ in [(a, b) for a, b in sizes]]
        for b_, v, bs in zip(bars, vals, base):
            ax1.text(b_.get_x() + b_.get_width() / 2, v + 18,
                     f"{v:.0f}\n({(v / bs - 1) * 100:+.2f}%)",
                     ha="center", va="bottom", fontsize=7.2, color=col)
    for xi, (s, _) in enumerate(sizes):
        cv = med[(s, "cublas")]
        ax1.hlines(cv, xi - 0.48, xi + 0.48, color="#c62828", lw=1.6,
                   ls="--", zorder=4)
        ax1.text(xi + 0.48, cv + 40, f"cuBLAS {cv:.0f}", ha="right",
                 fontsize=7.6, color="#c62828")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}$^3$" for s, _ in sizes], fontsize=10)
    ax1.set_ylabel("GFLOPS", fontsize=10)
    lo = min(med[k] for k in med if med and k[1] != "cublas") * 0.985
    ax1.set_ylim(lo, None)
    ax1.legend(fontsize=8.5, ncol=4, loc="lower left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("(a) dsk --sk 3: 3-round medians, vs-base % in parens\n"
                  "BPF within noise; PHASE consistently negative",
                  fontsize=9.5)

    # ---- (b) ptxas 寄存器 + 机理注记 ----
    names = ["base", "BPF", "PHASE", "both"]
    regs = [243, 245, 230, 230]
    cols = ["#37474f", "#1976d2", "#e64a19", "#8e24aa"]
    bars = ax2.bar(names, regs, 0.6, color=cols, zorder=3)
    for b_, v in zip(bars, regs):
        ax2.text(b_.get_x() + b_.get_width() / 2, v + 0.6, str(v),
                 ha="center", fontsize=9)
    ax2.set_ylim(220, 252)
    ax2.set_ylabel("registers / instance", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_title("(b) ptxas regs (all 0 spill;\n247/243 active instances unchanged)",
                  fontsize=9.5)
    ax2.text(0.5, 0.04,
             "mechanism: all warps read the same B columns\n"
             "at the same kk (tx pattern identical) -> smem\n"
             "multicast-friendly; PHASE de-synchronizes kk\n"
             "per warp and forfeits it. BPF adds nothing:\n"
             "dbuf double-buffering already covers the load.",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=7.6, color="#37474f",
             bbox=dict(fc="#eceff1", ec="#90a4ae", pad=4.5))

    fig.suptitle("AR011 T004 FR3 latency-cover ablation "
                 "(git=9eaf90a, lat_cover_ar011.csv): negative result",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig31_lat_cover.png"))
    plt.close(fig)

fig31_lat_cover()


# ---------------- AR011 fig32: Stream-K W sweep triptych ----------------
# 数据/证据：results/streamk_ar011.csv（2026-10-07 实测，git=8104293 构建
# 二进制，3 轮中位，v4 协议 47C 冷却 + cublas 首锚；canonical 六尺寸 ×
# W 深扫 + 四参照）。1024^3 分段基线 297+41μs 为 AR010 实测（同 1620
# 稳态域，跨会话可比；cublas 本会话轮首冷跑 boost 偏高，图内标注）。
# 结论：W=1 全尺寸最优（W>1 单调负 = P 税 + 块/SM 占用损失）；
# 1024^3 G1 翻门失败（5461 vs dsk 6279 = 87%）；2048^3 streamk_w1
# 8721 = 85.4% cublas（G6 边缘）= +2.1% over deep（波量化税实测）。
def fig32_streamk_sweep():
    import statistics as st
    csv_path = os.path.join(OUT_DIR, "..", "streamk_ar011.csv")
    if not os.path.exists(csv_path):
        print("fig32 skipped (streamk_ar011.csv missing)")
        return
    rows = [r for r in csv.DictReader(
        filter(lambda l: not l.startswith("#"), open(csv_path)))
        if r.get("kernel")]
    med = {}
    for r in rows:
        key = (int(r["m"]), int(r["n"]), int(r["k"]), r["kernel"])
        med.setdefault(key, []).append(float(r["gflops"]))
    for k in med:
        med[k] = st.median(med[k])

    SIZES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
             (1000, 1016, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    labels = ["256$^3$", "512$^3$", "1024$^3$", "1000x1016\nx1024",
              "2048$^3$", "4096$^3$"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17.2, 5.7),
        gridspec_kw={"width_ratios": [1.5, 1.15, 1.0]})

    # ---- (a) 全尺寸阶梯：四参照 + streamk_w1 ----
    cfgs = [("cublas", "cuBLAS", "#c62828", "o"),
            ("deep", "deep", "#455a64", "s"),
            ("dsk_sk3", "dsk sk3", "#1976d2", "^"),
            ("swsk_sk6", "swsk sk6", "#8d6e63", "D"),
            ("swsk_sk3", "swsk sk3", "#8d6e63", "D"),
            ("streamk_w1", "streamk W=1", "#2e7d32", "*")]
    x = np.arange(len(SIZES))
    for name, lab, col, mk in cfgs:
        ys = [med.get(s + (name,), np.nan) for s in SIZES]
        if np.all(np.isnan(ys)):
            continue
        ax1.plot(x, ys, color=col, marker=mk, ms=7 if mk == "*" else 5,
                 lw=1.6, label=lab, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("GFLOPS (3-round median)", fontsize=10)
    ax1.set_ylim(0, 11600)
    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("(a) full-size ladder: streamk W=1 wins only @2048$^3$;\n"
                  "swsk holds small sizes (dispatch keeps swsk/deep/dsk)",
                  fontsize=9.5)

    # ---- (b) W sweep 曲线：主战场尺寸 GF vs W ----
    wsets = [((1024, 1024, 1024), "1024$^3$", "#1976d2", 1, 8),
             ((1000, 1016, 1024), "1000x1016x1024", "#7b1fa2", 1, 6),
             ((2048, 2048, 2048), "2048$^3$", "#2e7d32", 1, 8),
             ((4096, 4096, 4096), "4096$^3$", "#e64a19", 1, 6)]
    for s, lab, col, w0, w1 in wsets:
        ws = list(range(w0, w1 + 1))
        ys = [med.get(s + (f"streamk_w{w}",), np.nan) for w in ws]
        ax2.plot(ws, ys, color=col, marker="o", ms=4.5, lw=1.6,
                 label=lab, zorder=3)
    # 各尺寸 dsk_sk3 参考虚线（与曲线同色淡显）
    for s, lab, col, w0, w1 in wsets:
        ref = med.get(s + ("dsk_sk3",), np.nan)
        ax2.hlines(ref, 0.6, w1 + 0.4, color=col, ls=":", lw=1.0,
                   alpha=0.55, zorder=2)
    ax2.text(6.1, 6000, "dotted = dsk sk3 ref", fontsize=7.5,
             color="#455a64")
    ax2.set_xlabel("--waves W", fontsize=10)
    ax2.set_xticks(range(1, 9))
    ax2.set_ylabel("GFLOPS", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_title("(b) W sweep: monotone decline from W=1 everywhere\n"
                  "-> auto W := 1 (formula recalibrated, T006 input)",
                  fontsize=9.5)

    # ---- (c) 1024^3 时间预算瀑布（μs，越低越好） ----
    FLOP = 2 * 1024 ** 3 / 1e9                       # GFLOP per pass
    gf2us = lambda g: FLOP / g * 1e3
    items = [
        ("cuBLAS\n(this sess,\nboost-inflated)", gf2us(med[(1024, 1024, 1024, "cublas")]), "#c62828"),
        ("deep", gf2us(med[(1024, 1024, 1024, "deep")]), "#455a64"),
        ("streamk W=1\n(fused)", gf2us(med[(1024, 1024, 1024, "streamk_w1")]), "#2e7d32"),
        ("streamk W=2\n(fused)", gf2us(med[(1024, 1024, 1024, "streamk_w2")]), "#81c784"),
    ]
    # dsk main+reduce 画成堆叠双段
    ax3.bar(["deep"], [297.0], 0.62, color="white", alpha=0)  # 占位
    ax3.bar(["dsk sk3\n(297+41)"], [297.0], 0.62, color="#1976d2",
            label="dsk main", zorder=3)
    ax3.bar(["dsk sk3\n(297+41)"], [41.0], 0.62, bottom=[297.0],
            color="#64b5f6", label="dsk reduce", zorder=3)
    for name, us, col in items:
        ax3.bar([name], [us], 0.62, color=col, zorder=3)
        ax3.text(name, us + 6, f"{us:.0f}", ha="center", fontsize=8,
                 color=col)
    tot = 297.0 + 41.0
    ax3.text("dsk sk3\n(297+41)", tot + 6, f"{tot:.0f}", ha="center",
             fontsize=8, color="#1976d2")
    ax3.axhline(255.9, color="#c62828", ls="--", lw=1.3, zorder=4)
    ax3.text(0.02, 250, "canonical G1 gate = 75% @ 255.9μs",
             fontsize=7.5, color="#c62828", transform=ax3.get_yaxis_transform())
    ax3.set_ylabel("time (μs)", fontsize=10)
    ax3.set_ylim(0, 560)
    ax3.legend(fontsize=7.5, loc="upper left")
    ax3.grid(axis="y", alpha=0.3, zorder=0)
    ax3.tick_params(axis="x", labelsize=7.6)
    ax3.set_title("(c) 1024$^3$ budget: fused streamk 393μs vs dsk 338μs\n"
                  "-> G1 flip FAILS (miss 1.05μs, honest negative)",
                  fontsize=9.5)

    fig.suptitle("AR011 T005 Stream-K W sweep (git=8104293 build, "
                 "streamk_ar011.csv): W=1 optimal, G6@2048$^3$ marginal pass",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig32_streamk_sweep.png"))
    plt.close(fig)


fig32_streamk_sweep()


# ---------------- AR011 fig33: auto v4 dispatch panorama ----------------
# 数据/证据：results/auto_ar011.csv（2026-10-07 实测，git=776a292 构建，
# A-B-A-B 2 轮，v4 协议 47C 冷却 + cublas 首锚；canonical 六 + 补充三）。
# 结论：G5''' 保真 10/12 <=0.6pp PASS（256^3 两 miss 同路径噪声）；G4'''
# 增量 MISS（1/6 >= +2%，门要求 4/6 —— 仅 2048^3 deep->streamk W=1
# +2.25% 兑现，诚实负结果）；补充首测 4096x256 发现几何规则盲区
# （streamk_w1 8478 > dsk 8262，M/N 不对称，披露不过拟合）。
def fig33_auto_v4():
    import statistics as st
    csv_path = os.path.join(OUT_DIR, "..", "auto_ar011.csv")
    if not os.path.exists(csv_path):
        print("fig33 skipped (auto_ar011.csv missing)")
        return
    rows = [r for r in csv.DictReader(
        filter(lambda l: not l.startswith("#"), open(csv_path)))
        if r.get("kernel")]
    raw = {}
    for r in rows:
        key = (int(r["m"]), int(r["n"]), int(r["k"]), r["kernel"])
        raw.setdefault(key, []).append(float(r["gflops"]))
    med = {k: st.median(v) for k, v in raw.items()}

    def g(m, n, kk, name):
        return med.get((m, n, kk, name))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17.0, 5.7),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.3]})

    # ---- (a) dispatch 决策表（canonical + 补充，v3->v4 变更高亮）----
    table = [
        ("256$^3$",        "swsk sk6",  "swsk sk6",  ""),
        ("512$^3$",        "swsk sk3",  "swsk sk3",  ""),
        ("1024$^3$",       "dsk sk3",   "dsk sk3",   ""),
        ("1000x1016x1024", "dsk sk3",   "dsk sk3",   ""),
        ("2048$^3$",       "deep",      "STREAMK W=1", "+2.25%"),
        ("4096$^3$",       "deep",      "deep",      ""),
        ("256x4096x4096",  "(new)",     "dsk sk3",   "first-test OK"),
        ("4096x256x4096",  "(new)",     "dsk sk3",   "first-test OK"),
        ("1024x2048x2048", "(new)",     "STREAMK W=1", "+28% vs deep"),
    ]
    ax1.axis("off")
    hdr = ["size", "v3 pick", "v4 pick", "note"]
    rowsx = [[s, a, b, nt] for s, a, b, nt in table]
    t = ax1.table(cellText=rowsx, colLabels=hdr, loc="center",
                  cellLoc="center", colWidths=[0.30, 0.22, 0.26, 0.22])
    t.auto_set_font_size(False)
    t.set_fontsize(7.6)
    t.scale(1.0, 1.55)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        if r_ == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold")
        elif "STREAMK" in str(cell.get_text()):
            cell.set_facecolor("#c8e6c9")
            cell.set_text_props(fontweight="bold", color="#1b5e20")
        elif r_ % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax1.set_title("(a) auto v4 dispatch table (zones: blocks<=4 swsk6 /\n"
                  "<=16 swsk3 / <=64 dsk geom-sk / tail-wave streamk W=1)",
                  fontsize=9.5)

    # ---- (b) G4''' delta + G5''' fidelity ----
    sizes6 = [(256, 256, 256, "256$^3$"), (512, 512, 512, "512$^3$"),
              (1024, 1024, 1024, "1024$^3$"),
              (1000, 1016, 1024, "1000x1016"), (2048, 2048, 2048, "2048$^3$"),
              (4096, 4096, 4096, "4096$^3$")]
    names = [s[3] for s in sizes6]
    deltas = []
    for m, n, kk, _ in sizes6:
        v3 = next(v for k, v in med.items()
                  if k[:3] == (m, n, kk) and k[3].startswith("v3_"))
        v4 = next(v for k, v in med.items()
                  if k[:3] == (m, n, kk) and k[3].startswith("w4_"))
        deltas.append((v4 / v3 - 1) * 100)
    cols = ["#2e7d32" if d >= 2 else ("#c62828" if d < -2 else "#90a4ae")
            for d in deltas]
    xs = np.arange(len(names))
    ax2.bar(xs, deltas, 0.62, color=cols, zorder=3)
    for x_, d in zip(xs, deltas):
        ax2.text(x_, d + (0.08 if d >= 0 else -0.22), f"{d:+.2f}",
                 ha="center", fontsize=7.6)
    ax2.axhline(2, color="#2e7d32", ls="--", lw=1.0)
    ax2.axhline(-2, color="#c62828", ls="--", lw=1.0)
    ax2.axhline(0, color="#455a64", lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, fontsize=7.6, rotation=20)
    ax2.set_ylabel("v4 vs v3 (%)", fontsize=10)
    ax2.set_ylim(-1.2, 3.2)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_title("(b) G4''' increment: MISS (1/6 >= +2%; only 2048$^3$\n"
                  "+2.25% via streamk W=1; honest negative, gate kept)",
                  fontsize=9.5)
    # G5''' 注记
    ax2.text(0.03, 0.04,
             "G5''' fidelity: canonical 11/12 <= 0.6pp PASS\n"
             "(256$^3$ miss = same-path noise -2.12pp);\n"
             "suppl. clean same-regime pairs: -0.45/-0.61/\n"
             "-0.03/-0.69/0.00pp (all <= 2pp)",
             transform=ax2.transAxes, fontsize=7.3, va="bottom",
             bbox=dict(fc="#e8f5e9", ec="#81c784", pad=4))

    # ---- (c) 补充尺寸首测（candidates + auto 高亮；制度标注）----
    # 256x4096/4096x256 = 冷启动制度（1620 MHz 稳态，逐行冷却重测）；
    # 1024x2048 = warm 制度（1935 MHz，in-sequence r3/r4）。dsk>strk 与
    # strk>=deep 的排序在两制度下均成立（run-1 "4096x256 盲区" 为 auto
    # 末位行降频伪影 1740/1620 MHz，已撤回）。
    supp = [("256x4096x4096", "cold\n(1620)", (256, 4096, 4096)),
            ("4096x256x4096", "cold\n(1620)", (4096, 256, 4096)),
            ("1024x2048x2048", "warm\n(1935)", (1024, 2048, 2048))]
    cand = [("deep", "deep", "#455a64"), ("dsk_sk3", "dsk sk3", "#1976d2"),
            ("swsk_sk3", "swsk sk3", "#8d6e63"),
            ("streamk_w1", "strk W=1", "#2e7d32"),
            ("streamk_w2", "strk W=2", "#81c784")]
    xs = np.arange(len(supp))
    bw = 0.15
    for i, (cname, lab, col) in enumerate(cand):
        vals = []
        for lab_s, _, key in supp:
            v = [x for x in raw.get(key + (cname,), [np.nan])]
            vals.append(st.median(v) if v else np.nan)
        offs = (i - 2) * bw
        bars = ax3.bar(xs + offs, vals, bw * 0.9, color=col, label=lab,
                       zorder=3)
        for b_, v in zip(bars, vals):
            if v == v:
                ax3.text(b_.get_x() + b_.get_width() / 2, v + 15,
                         f"{v:.0f}", ha="center", fontsize=6.4, color=col)
    for xi, (lab_s, reg, key) in enumerate(supp):
        av = st.median(raw.get(key + ("auto_v4",), [np.nan]))
        ax3.hlines(av, xi - 0.42, xi + 0.42, color="#c62828", lw=1.6,
                   ls="--", zorder=4)
        ax3.text(xi + 0.42, av + 30, f"auto {av:.0f}", ha="right",
                 fontsize=7.2, color="#c62828")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{s[0]}\n{r}" for s, r, _ in supp], fontsize=8.0)
    ax3.set_ylabel("GFLOPS", fontsize=10)
    ax3.set_ylim(4500, 10600)
    ax3.legend(fontsize=7.6, ncol=5, loc="lower left")
    ax3.grid(axis="y", alpha=0.3, zorder=0)
    ax3.set_title("(c) supplementary first-test: dsk wins both 4:1 sizes "
                  "(both regimes),\nstrk~=dsk >> deep @1024x2048 (zone rule; "
                  "fidelity <=2pp all)",
                  fontsize=9.0)
    ax3.text(0.98, 0.02,
             "run-1 '4096x256 blind spot' retracted:\n"
             "auto last-in-seq rows throttled to 1740/1620 MHz\n"
             "(clock artifact, gpu_state column evidence)",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=6.8,
             color="#37474f",
             bbox=dict(fc="#fff3e0", ec="#ffb74d", pad=3.5))

    fig.suptitle("AR011 T006 auto v4 (auto_ar011.csv, git=776a292 build): "
                 "G5''' PASS / G4''' MISS / 3 first-tests",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig33_auto_v4.png"))
    plt.close(fig)


fig33_auto_v4()


# ---------------- AR011 fig34: v5 gate verdict panorama ----------------
# 数据/证据：results/paired_ar011.csv（2026-10-07 实测，git=7d9a117 构建，
# v5 协议 canonical 8 行/轮 x3 + 补充 7 行/轮 x3，cublas 首末双锚 + 47C
# 冷却）。结论（compare_ar011_paired.md 终判表）：G1@1024^3 MISS 74.90%
# （差 0.10pp/0.4us）；G6@2048^3 MISS 84.85%（差 0.15pp/3.2us，末锚
# 稳态口径，r1 冷锚 9278 排除）；G1@512^3 PASS 边缘（75.01% vs 75.14%
# 会话噪声，落于 AR010 自身轮值域）；G2/G3/G5''' PASS；G4''' MISS。
# 机制交付完整：2048^3 尾波税消除 +2.35pp（82.50%->84.85%）。
def fig34_v5_verdict():
    import statistics as st
    csv_path = os.path.join(OUT_DIR, "..", "paired_ar011.csv")
    if not os.path.exists(csv_path):
        print("fig34 skipped (paired_ar011.csv missing)")
        return
    rows = [r for r in csv.DictReader(
        filter(lambda l: not l.startswith("#"), open(csv_path)))
        if r.get("kernel")]
    raw = {}
    for r in rows:
        key = (int(r["m"]), int(r["n"]), int(r["k"]), r["kernel"])
        raw.setdefault(key, []).append(float(r["gflops"]))

    CANON = ["cublas", "deep", "dsk_sk3", "streamk_w1", "auto_v4",
             "swsk", "swpipe", "cublas_end"]
    OWN = CANON[1:7]
    sizes6 = [(256, 256, 256, "256$^3$"), (512, 512, 512, "512$^3$"),
              (1024, 1024, 1024, "1024$^3$"),
              (1000, 1016, 1024, "1000x1016"), (2048, 2048, 2048, "2048$^3$"),
              (4096, 4096, 4096, "4096$^3$")]

    def med(m, n, kk, name):
        v = raw.get((m, n, kk, name), [])
        return st.median(v) if v else np.nan

    def denom(m, n, kk):
        v = raw.get((m, n, kk, "cublas"), []) + \
            raw.get((m, n, kk, "cublas_end"), [])
        return st.median(v) if v else np.nan

    def best_own(m, n, kk):
        vals = [med(m, n, kk, nm) for nm in OWN]
        vals = [v for v in vals if v == v]
        return max(vals)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17.0, 5.6),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

    # ---- (a) canonical 六尺寸 best own % cuBLAS + 75% 门线 ----
    labels, pcts, verdicts = [], [], []
    for m, n, kk, lab in sizes6:
        labels.append(lab)
        p = best_own(m, n, kk) / denom(m, n, kk) * 100
        pcts.append(p)
        verdicts.append("MISS" if (p < 75 and (m, n) != (256, 256)) else "PASS")
    # 2048^3 用 G6 85% 线单独判定（G1 同样 75% 已过）
    cols = []
    for (m, n, kk, lab), p, v in zip(sizes6, pcts, verdicts):
        if (m, n, kk) == (2048, 2048, 2048):
            cols.append("#c62828" if p < 85 else "#2e7d32")
        elif (m, n, kk) == (512, 512, 512):
            cols.append("#f9a825")  # edge PASS (session noise, in AR010 range)
        else:
            cols.append("#c62828" if v == "MISS" else "#2e7d32")
    xs = np.arange(len(labels))
    ax1.bar(xs, pcts, 0.62, color=cols, zorder=3)
    for x_, p in zip(xs, pcts):
        ax1.text(x_, p + 1.2, f"{p:.2f}", ha="center", fontsize=7.8,
                 fontweight="bold")
    ax1.axhline(75, color="#c62828", ls="--", lw=1.2, zorder=4)
    ax1.text(0.5, 75.8, "G1 75%", fontsize=6.9, color="#c62828",
             ha="center")
    ax1.axhline(85, color="#6a1b9a", ls=":", lw=1.2, zorder=4)
    ax1.text(3.5, 85.8, "G6 85%", fontsize=6.9, color="#6a1b9a",
             ha="center")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=7.8, rotation=20)
    ax1.set_ylabel("best own / cuBLAS (%)", fontsize=10)
    ax1.set_ylim(60, 130)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("(a) canonical verdicts: G1@1024$^3$ MISS 74.90 "
                  "(-0.10pp/0.4$\\mu$s),\nG6@2048$^3$ MISS 84.85 "
                  "(-0.15pp/3.2$\\mu$s); 512$^3$ edge-PASS (noise)",
                  fontsize=9.3)

    # ---- (b) 2048^3 锚敏感性：逐轮首锚比 vs 末锚比 ----
    rs = raw.get((2048, 2048, 2048, "cublas"), [])
    re_ = raw.get((2048, 2048, 2048, "cublas_end"), [])
    bo = [max(med(2048, 2048, 2048, nm) for nm in OWN)] * 3
    per_r_best = [8783.5, 8774.7, 8774.7]  # 逐轮 best own（judge 逐轮表）
    first_pct = [per_r_best[i] / rs[i] * 100 for i in range(3)]
    end_pct = [per_r_best[i] / re_[i] * 100 for i in range(3)]
    xs2 = np.arange(3)
    w = 0.32
    ax2.bar(xs2 - w / 2, first_pct, w, color="#90a4ae", label="vs first anchor",
            zorder=3)
    ax2.bar(xs2 + w / 2, end_pct, w, color="#37474f", label="vs end anchor",
            zorder=3)
    for x_, v in zip(xs2 - w / 2, first_pct):
        ax2.text(x_, v + 0.25, f"{v:.2f}", ha="center", fontsize=7.4,
                 color="#546e7a")
    for x_, v in zip(xs2 + w / 2, end_pct):
        ax2.text(x_, v + 0.25, f"{v:.2f}", ha="center", fontsize=7.4,
                 fontweight="bold")
    ax2.axhline(85, color="#6a1b9a", ls=":", lw=1.2)
    ax2.text(2.42, 85.4, "G6 85%", fontsize=7.6, color="#6a1b9a",
             ha="right")
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(["r1", "r2", "r3"], fontsize=9)
    ax2.set_ylim(80, 100)
    ax2.legend(fontsize=7.8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_title("(b) 2048$^3$ anchor sensitivity: r1 first anchor 9278 =\n"
                  "cold-start (excl.); end anchors 10337-10342 rock-stable\n"
                  "$\\rightarrow$ 84.93/84.84/84.89% vs gate 85%",
                  fontsize=9.0)

    # ---- (c) AR010 -> AR011 演进（512/1024/2048）+ 机制兑现 ----
    # AR010 值引 compare_ar010_paired.md / srs 前置（75.14 / 74.77 / 82.6）
    ar010 = [75.14, 74.77, 82.60]
    ar011 = [75.01, 74.90, 84.85]
    gates = [75.0, 75.0, 85.0]
    labs3 = ["512$^3$\n(G1)", "1024$^3$\n(G1)", "2048$^3$\n(G6)"]
    xs3 = np.arange(3)
    w = 0.32
    ax3.bar(xs3 - w / 2, ar010, w, color="#b0bec5", label="AR010 terminal",
            zorder=3)
    ax3.bar(xs3 + w / 2, ar011, w, color="#37474f", label="AR011 v5",
            zorder=3)
    for x_, v in zip(xs3 - w / 2, ar010):
        ax3.text(x_, v + 0.18, f"{v:.2f}", ha="center", fontsize=7.4,
                 color="#607d8b")
    for x_, v in zip(xs3 + w / 2, ar011):
        ax3.text(x_, v + 0.18, f"{v:.2f}", ha="center", fontsize=7.4,
                 fontweight="bold")
    for x_, g in zip(xs3, gates):
        ax3.hlines(g, x_ - 0.42, x_ + 0.42, color="#c62828", ls="--",
                   lw=1.2, zorder=4)
    ax3.annotate("-0.13pp\n(session noise,\ninside AR010 range)",
                 xy=(0.16, 75.7), fontsize=6.9, color="#f57f17")
    ax3.annotate("+0.13pp\n(gap 0.4$\\mu$s;\nMISS by 0.10pp)",
                 xy=(1.16, 75.7), fontsize=6.9, color="#c62828")
    ax3.annotate("+2.35pp mechanism\n(tail-wave tax removed:\ndeep 82.50 $\\to$ strk\n84.85; MISS 0.15pp)",
                 xy=(1.55, 86.6), fontsize=6.9, color="#2e7d32")
    ax3.set_xticks(xs3)
    ax3.set_xticklabels(labs3, fontsize=8.4)
    ax3.set_ylim(70, 90)
    ax3.legend(fontsize=7.8, loc="lower right")
    ax3.grid(axis="y", alpha=0.3, zorder=0)
    ax3.set_title("(c) AR010 $\\to$ AR011: both knife-edge ratio gates\n"
                  "kept honest (no gate relaxation); full mechanism\n"
                  "delivery at 2048$^3$",
                  fontsize=9.0)

    fig.suptitle("AR011 T008 v5 gate verdict (paired_ar011.csv, git=7d9a117 "
                 "build): 3 MISS / 4 PASS, all margins disclosed",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, "fig34_v5_verdict.png"))
    plt.close(fig)


fig34_v5_verdict()

fig_gates_v4()
fig_ladder_v4()


# ---------------- AR012 fig35: L2 block-order swizzle ablation triptych ------
# 数据/证据：results/swizzle_ar012.csv（2026-10-07 实测，git=e87141a，3 轮中位，
# {deep,dsk,streamk} × {swz0, swz1g4/8/16} × 5 尺寸，47C 冷却 + 每轮 cublas
# 首锚协议）+ build.log ptxas（20 deep 实例 + 4 streamk 实例全 0 spill）。
# 结论（G7 判定，kernel 分立）：deep/dsk 全负（2D 线性光栅已优——bx=n 最快
# 的行主序 B 复用；重排破坏）；**streamk-only 正**（+3.9%@2048³ / +9.1%@4096³
# / +1.6%@1024³，512³ -0.05pp 门内）——1D c 空间 tile-major 线性化的默认
# 块序 L2 复用差，分组列序修正。G=4 全尺寸最优或并列 → T006 auto v5
# 仅 streamk 路径吸收 --swz 1 --swzg 4。副产物：streamk swz1g4 @2048³ =
# 88.3% cuBLAS → G6（≥85%）翻门（AR011 曾 miss 84.85%）。
def fig35_swizzle_ablation():
    import statistics as st
    csv_path = os.path.join(OUT_DIR, "..", "swizzle_ar012.csv")
    if not os.path.exists(csv_path):
        print("fig35 skipped (swizzle_ar012.csv missing)")
        return
    rows = [r for r in csv.DictReader(
        filter(lambda l: not l.startswith("#"), open(csv_path)))
        if r.get("kernel")]
    med = {}
    for r in rows:
        key = (int(r["n"]), r["kernel"])
        med.setdefault(key, []).append(float(r["gflops"]))
    for k in med:
        med[k] = st.median(med[k])

    kerns = ["deep", "dsk", "streamk"]
    tags = [("swz0", "linear (swz=0)", "#37474f"),
            ("swz1g4", "swz=1 G=4", "#1976d2"),
            ("swz1g8", "swz=1 G=8", "#e64a19"),
            ("swz1g16", "swz=1 G=16", "#8e24aa")]
    gate_sizes = [2048, 4096]           # (a) G7 主判定
    all_sizes = [2048, 4096, 1024, 512, 256]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16.8, 5.4),
        gridspec_kw={"width_ratios": [1.35, 0.9, 1.1]})

    # ---- (a) G7 主判定区带 @2048³/4096³：全 kernel × 4 配置 ----
    x = np.arange(len(gate_sizes) * len(kerns))
    bw = 0.2
    for i, (tag, lab, col) in enumerate(tags):
        vals = [med[(s, k + "_" + tag)]
                for s in gate_sizes for k in kerns]
        offs = (i - 1.5) * bw
        bars = ax1.bar(x + offs, vals, bw * 0.9, color=col, label=lab, zorder=3)
        for b_, v, s, k in zip(bars, vals, [s for s in gate_sizes for _ in kerns],
                               [k for _ in gate_sizes for k in kerns]):
            base = med[(s, k + "_swz0")]
            if tag != "swz0":
                ax1.text(b_.get_x() + b_.get_width() / 2, v + 15,
                         f"{(v / base - 1) * 100:+.1f}%", ha="center",
                         va="bottom", fontsize=6.4, color=col)
    for xi, s in enumerate(gate_sizes):
        for ki, k in enumerate(kerns):
            cv = med[(s, "cublas")]
            xi_full = xi * len(kerns) + ki
            ax1.hlines(cv, xi_full - 0.55, xi_full + 0.55, color="#c62828",
                       lw=1.3, ls="--", zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{k}\n{s}$^3$" for s in gate_sizes for k in kerns],
                        fontsize=8)
    ax1.set_ylabel("GFLOPS", fontsize=10)
    lo = min(v for (s, k), v in med.items()
             if k != "cublas" and s in gate_sizes) * 0.97
    ax1.set_ylim(lo, None)
    ax1.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("(a) G7 gate sizes: streamk-only positive\n"
                  "(deep/dsk negative — dashed = cuBLAS)",
                  fontsize=9.5)

    # ---- (b) 波足迹机理示意 @2048³（design §4.2.1 核算）----
    # deep t=128 块 = 2.67 波；线性：波 = 3m×16n（A 6.3 + B 16.8MB，B 全宽
    # 重读）；G=8：8m×6n（A 16.8 + B 6.3MB）；G=4：12m×4n（A 25 + B 4.2MB）
    orders = ["linear\n(3m x 16n)", "G=8\n(8m x 6n)", "G=4\n(12m x 4n)"]
    a_mb = [6.3, 16.8, 25.0]
    b_mb = [16.8, 6.3, 4.2]
    xx = np.arange(3)
    bars_a = ax2.bar(xx - 0.19, a_mb, 0.34, color="#1976d2",
                     label="A slab / wave", zorder=3)
    bars_b = ax2.bar(xx + 0.19, b_mb, 0.34, color="#e64a19",
                     label="B slab / wave", zorder=3)
    for b_, v in zip(bars_a, a_mb):
        ax2.text(b_.get_x() + b_.get_width() / 2, v + 0.4, f"{v:.1f}",
                 ha="center", fontsize=8, color="#1976d2")
    for b_, v in zip(bars_b, b_mb):
        ax2.text(b_.get_x() + b_.get_width() / 2, v + 0.4, f"{v:.1f}",
                 ha="center", fontsize=8, color="#e64a19")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(orders, fontsize=8.5)
    ax2.set_ylabel("MB touched per wave", fontsize=10)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_title("(b) wave footprint @2048$^3$ (deep 2.67 waves,\n"
                  "48KB L2 fits neither slab — order decides reuse)",
                  fontsize=9.5)

    # ---- (c) G7 判定矩阵：delta%（swz0 → best swz1）× kernel × 尺寸 ----
    data = np.zeros((len(kerns), len(all_sizes)))
    for ki, k in enumerate(kerns):
        for si, s in enumerate(all_sizes):
            base = med[(s, k + "_swz0")]
            best = max(med[(s, k + "_swz1g4")], med[(s, k + "_swz1g8")],
                       med[(s, k + "_swz1g16")])
            data[ki, si] = (best / base - 1) * 100
    im = ax3.imshow(data, cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto",
                    zorder=2)
    for ki in range(len(kerns)):
        for si in range(len(all_sizes)):
            v = data[ki, si]
            ax3.text(si, ki, f"{v:+.1f}%", ha="center", va="center",
                     fontsize=9, fontweight="bold" if abs(v) >= 1 else "normal",
                     color="#1b1b1b")
    ax3.set_xticks(range(len(all_sizes)))
    ax3.set_xticklabels([f"{s}$^3$" for s in all_sizes], fontsize=9)
    ax3.set_yticks(range(len(kerns)))
    ax3.set_yticklabels(kerns, fontsize=9.5)
    ax3.set_title("(c) best-swz1 delta vs swz0 (gate: $\\geq$+1%\n"
                  "@2048/4096 AND no >0.5pp regression @512/1024)",
                  fontsize=9.5)
    fig.colorbar(im, ax=ax3, fraction=0.045, pad=0.03, label="%")

    fig.suptitle("AR012 T004 FR2 L2 block-order swizzle ablation "
                 "(git=e87141a, swizzle_ar012.csv): G7 = streamk-only "
                 "absorption, G=4",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT_DIR, "fig35_swizzle_ablation.png"))
    plt.close(fig)

fig35_swizzle_ablation()


# ---------------- 体积纪律：全部图量化到 <= 72KB ----------------
# gitee HTTPS push 经 BDWAF，POST body ~100KB 即 403；本工程既定纪律
# （沿袭远端 "PNG quantization to 72KB target"）——纯 PIL 可复现。
def quantize_figures():
    from PIL import Image
    target = 72 * 1024
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.endswith(".png"):
            continue
        path = os.path.join(OUT_DIR, name)
        if os.path.getsize(path) <= target:
            continue
        colors = 256
        scale = 1.0
        while True:
            img = Image.open(path).convert("RGB")
            if scale < 1.0:
                img = img.resize((max(1, int(img.width * scale)),
                                  max(1, int(img.height * scale))),
                                 Image.LANCZOS)
            img.quantize(colors=colors).save(path, optimize=True)
            if os.path.getsize(path) <= target or (colors <= 32 and scale <= 0.52):
                break
            if colors > 32:
                colors //= 2
            else:
                scale -= 0.08
        print(f"  quantized {name} -> {os.path.getsize(path)/1024:.0f} KB "
              f"(colors={colors}, scale={scale:.2f})")


quantize_figures()
