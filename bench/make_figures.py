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

def optimize_png(path):
    """存储层压缩：白底合成 + 调色板量化 + optimize，迭代降至 SIZE_TARGET 以下。

    图表像素内容由 matplotlib 渲染决定，本步骤只做存储层压缩（同一脚本、同一 CSV
    输入 → 同一输出），保持"无手工修饰"纪律。从 256 色起逐档减半直至达标。
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [warn] Pillow 不可用，跳过 PNG 压缩:", path)
        return
    SIZE_TARGET = 72 * 1024  # 72 KB：图表归档体积目标
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    colors = 256
    while True:
        img.quantize(colors=colors, method=Image.MEDIANCUT).save(path, optimize=True)
        if os.path.getsize(path) <= SIZE_TARGET or colors <= 32:
            break
        colors //= 2
    # 色数到下限仍未达标（如密集多面板图）：逐档轻微降采样（0.9x），直至达标
    if os.path.getsize(path) > SIZE_TARGET:
        scale = 0.9
        while os.path.getsize(path) > SIZE_TARGET and scale >= 0.6:
            w, h = img.size
            small = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            small.quantize(colors=colors, method=Image.MEDIANCUT).save(path, optimize=True)
            scale *= 0.9


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
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        optimize_png(os.path.join(OUT_DIR, f))
print("figures written to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)
