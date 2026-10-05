#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# compare.py — AR007 回归对比：基线 CSV vs 新会话 CSV → delta 表 + 四门判定
# ---------------------------------------------------------------
# 用法：
#   python bench\compare.py <baseline.csv> <new.csv> [-o results\compare_ar007.md]
# 语义（与 run_matrix.ps1 的数据流配套）：
#   - 基线文件：每个 (kernel, m, n, k) 取【首行】= E1 矩阵会话默认配置行
#   - 新文件  ：每个 (kernel, m, n, k) 取【末行】= AR007 矩阵会话行
#     （消融行已由 SGEMM_CSV 分流至 ablation csv，不在主文件中制造歧义）
#   - CSV 为 14 字段位置解析（表头 gpu_state 含未引用逗号，DictReader 不可用）：
#     kernel=0 m=1 n=2 k=3 med=4 min=5 max=6 mean=7 rsd=8 gf=9 note=10
#     state=11 git=12 ts=13
# 四门判定（srs AR007 §1）：
#   G-小尺寸：256^3（及 512^3）自研最优 > 同会话 cuBLAS
#   G-大尺寸：4096^3 自研最优 >= 7200 GF 且 >= cuBLAS 的 75%
#   G-全线  ：>= 4/6 尺寸自研最优较基线刷新 >= 2%
#   G-K6    ：4096^3 swpipe > vec4（全尺寸附列；负结果如实归档）
# ----
# AR008 追加 --paired 模式（thermal-paired 协议，design §4.2.4）：
#   python bench\compare.py --paired results\paired_ar008.csv [-o ...] [--baseline-kernel swpipe]
#   语义：配对 CSV 行序即配对关系（run_paired.ps1 产出：baseline 行 + 紧随挑战者行）；
#   对内 delta = (挑战者 GF - 基线 GF)/基线 GF——共模热漂移相消后的真实差值。
#   四门 v2（srs AR008 §4）：G1 中尺寸 75% cuBLAS / G2 小尺寸守成 /
#   G3 大尺寸 7.0TF / G4 全线 >=4/6 对内 delta>=+2% / G5 delta 轮间波动(极差)<2pp
# =====================================================================
import argparse
import csv
import sys
from collections import OrderedDict

CUSTOM_KERNELS = ["naive", "coalesced", "smem1d", "tile2d", "vec4",
                  "cpasync", "cpasync2", "swpipe"]


def load_rows(path):
    """读取 CSV → {(kernel, m, n, k): row_dict}（保留读序）。"""
    rows = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw or raw[0].lstrip().startswith("#"):
                continue
            if raw[0] == "kernel":       # 表头行
                continue
            if len(raw) < 14:
                continue
            key = (raw[0], int(raw[1]), int(raw[2]), int(raw[3]))
            row = {
                "kernel": raw[0], "m": int(raw[1]), "n": int(raw[2]),
                "k": int(raw[3]), "ms": float(raw[4]), "rsd": float(raw[8]),
                "gf": float(raw[9]), "git": raw[12], "ts": raw[13],
            }
            if key not in rows:          # 首见保留（读序）
                rows[key] = row
    return rows


def load_rows_last(path):
    """同 load_rows，但同 key 取末行（新会话语义）。"""
    rows = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw or raw[0].lstrip().startswith("#") or raw[0] == "kernel":
                continue
            if len(raw) < 14:
                continue
            key = (raw[0], int(raw[1]), int(raw[2]), int(raw[3]))
            row = {
                "kernel": raw[0], "m": int(raw[1]), "n": int(raw[2]),
                "k": int(raw[3]), "ms": float(raw[4]), "rsd": float(raw[8]),
                "gf": float(raw[9]), "git": raw[12], "ts": raw[13],
            }
            rows[key] = row               # 覆盖 → 末行
    return rows


def size_of(t):
    return t[0] * t[1] * t[2]


def fmt_delta(base, new):
    if base is None or base == 0:
        return "n/a"
    d = (new - base) / base * 100.0
    return f"{d:+.1f}%"


# ---------------------------------------------------------------------
# --paired 模式（AR008 thermal-paired 协议）
# ---------------------------------------------------------------------

def load_pairs(path, baseline_kernel):
    """按行序读配对 CSV：baseline 行 + 紧随的非 baseline 行 = 一对。

    返回 OrderedDict：{(size, challenger): [(base_gf, ch_gf, base_rsd, ch_rsd, ts)]}
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw or raw[0].lstrip().startswith("#") or raw[0] == "kernel":
                continue
            if len(raw) < 14:
                continue
            rows.append({
                "kernel": raw[0], "m": int(raw[1]), "n": int(raw[2]),
                "k": int(raw[3]), "ms": float(raw[4]), "rsd": float(raw[8]),
                "gf": float(raw[9]), "git": raw[12], "ts": raw[13],
            })
    groups = OrderedDict()
    i = 0
    while i + 1 < len(rows):
        b, c = rows[i], rows[i + 1]
        if b["kernel"] == baseline_kernel and c["kernel"] != baseline_kernel:
            key = ((b["m"], b["n"], b["k"]), c["kernel"])
            groups.setdefault(key, []).append(
                (b["gf"], c["gf"], b["rsd"], c["rsd"], c["ts"]))
            i += 2
        else:
            i += 1          # 非配对行（容错跳过）
    return groups


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def paired_report(path, baseline_kernel, out):
    groups = load_pairs(path, baseline_kernel)
    if not groups:
        print(f"[compare] no pairs found in {path} "
              f"(baseline kernel = {baseline_kernel})", file=sys.stderr)
        return 1

    L = []
    L.append("# AR008 thermal-paired 对内 delta 报告\n")
    L.append(f"- paired csv : `{path}`")
    L.append(f"- baseline   : {baseline_kernel}（run_paired.ps1 交替配对产出）")
    L.append("- 语义       ：对内 delta = 挑战者 vs 基线（同轮同热状态，共模漂移相消）\n")

    L.append("## 对内 delta 明细\n")
    L.append("| size | challenger | rounds | baseline GF | challenger GF | "
             "delta%（逐轮） | median | 极差(pp) |")
    L.append("|---|---|---:|---:|---:|---|---:|---:|")

    # 汇总结构：size -> challenger -> (median_delta, med_base_gf, med_ch_gf, spread)
    by_size = OrderedDict()
    for (size, ch), pairs in groups.items():
        deltas = [(c - b) / b * 100.0 for (b, c, _, _, _) in pairs]
        med_d = median(deltas)
        spread = (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0
        med_b = median([b for (b, c, _, _, _) in pairs])
        med_c = median([c for (b, c, _, _, _) in pairs])
        ds = ", ".join(f"{d:+.2f}" for d in deltas)
        m, n, k = size
        L.append(f"| {m}x{n}x{k} | {ch} | {len(pairs)} | {med_b:.1f} | {med_c:.1f} "
                 f"| {ds} | {med_d:+.2f} | {spread:.2f} |")
        by_size.setdefault(size, OrderedDict())[ch] = (med_d, med_b, med_c, spread)
    L.append("")

    # ---- 四门 v2 ----
    L.append("## 四门 v2 判定（srs AR008 §4）\n")

    def ch_group(size, ch):
        return by_size.get(size, {}).get(ch)

    # G1 中尺寸：512^3 / 1024^3 自研最优（swsk 或 auto）>= 75% cuBLAS（同协议 cublas 组）
    L.append("### G1 中尺寸（512^3/1024^3 >= 75% cuBLAS）\n")
    L.append("| size | 自研最优 GF | kernel | cuBLAS GF | 比值 | 判定 |")
    L.append("|---|---:|---|---:|---:|---|")
    g1_pass = 0
    g1_n = 0
    for size in [(512, 512, 512), (1024, 1024, 1024)]:
        cands = {ch: v for ch, v in by_size.get(size, {}).items() if ch != "cublas"}
        cb = ch_group(size, "cublas")
        if not (cands and cb):
            continue
        g1_n += 1
        ch = max(cands, key=lambda c: cands[c][2])
        g = cands[ch][2]
        ratio = g / cb[2] * 100.0 if cb[2] else 0.0
        ok = ratio >= 75.0
        g1_pass += 1 if ok else 0
        L.append(f"| {size[0]}^3 | {g:.1f} | {ch} | {cb[2]:.1f} | {ratio:.1f}% "
                 f"| {'PASS' if ok else 'FAIL'} |")
    if g1_n:
        L.append(f"\n- **判定：{'PASS' if g1_pass == g1_n else 'FAIL'}**"
                 f"（{g1_pass}/{g1_n} 达标）\n")

    # G2 小尺寸守成：256^3 挑战者组中 auto/swsk 保持 > 基线且绝对值 >= 1618.2
    L.append("### G2 小尺寸守成（256^3 auto/swsk >= 1618.2 GF）\n")
    size = (256, 256, 256)
    cands = {ch: v for ch, v in by_size.get(size, {}).items()
             if ch in ("auto", "swsk")}
    if cands:
        for ch, v in cands.items():
            ok = v[2] >= 1618.2
            L.append(f"- 256^3 {ch}: {v[2]:.1f} GF → {'PASS' if ok else 'FAIL'}")
        best = max(cands.values(), key=lambda v: v[2])
        L.append(f"- **判定：{'PASS' if best[2] >= 1618.2 else 'FAIL'}**\n")
    else:
        L.append("- （本 CSV 无 256^3 auto/swsk 组，跳过）\n")

    # G3 大尺寸：4096^3 ws 对内 delta（issue-slot 假说检验；>=7.0TF 绝对门）
    L.append("### G3 大尺寸（4096^3 ws：对内 delta + 绝对值 >= 7.0 TF）\n")
    ws4096 = ch_group((4096, 4096, 4096), "ws")
    if ws4096:
        d, mb, mc, sp = ws4096
        abs_ok = mc >= 7000.0
        rel_ok = d >= 5.0
        L.append(f"- ws vs swpipe 对内 delta = {d:+.2f}%（极差 {sp:.2f}pp）")
        L.append(f"- ws 绝对值 = {mc:.1f} GF（{'OK' if abs_ok else 'NG'} 7.0 TF 门；"
                 f"力争 7.2 TF {'OK' if mc >= 7200.0 else 'NG'}）")
        L.append(f"- **判定：{'PASS' if (abs_ok and rel_ok) else 'FAIL'}**"
                 f"（假说检验：{'issue-slot 假说获支持' if d >= 5.0 else '负结果——如实归档'}）\n")
    else:
        L.append("- （本 CSV 无 4096^3 ws 组，跳过）\n")

    # G4 全线：>= 4/6 尺寸存在挑战者对内 delta >= +2%
    L.append("### G4 全线（>= 4/6 尺寸对内 delta >= +2%）\n")
    L.append("| size | 最佳挑战者 | delta | 判定 |")
    L.append("|---|---|---:|---|")
    hits = 0
    for size, chs in by_size.items():
        cands = {ch: v for ch, v in chs.items() if ch != "cublas"}
        if not cands:
            continue
        ch = max(cands, key=lambda c: cands[c][0])
        d = cands[ch][0]
        ok = d >= 2.0
        hits += 1 if ok else 0
        m, n, k = size
        L.append(f"| {m}x{n}x{k} | {ch} | {d:+.2f}% | {'HIT' if ok else 'miss'} |")
    ok = hits >= 4
    L.append(f"\n- 命中 {hits}/{len(by_size)}（门槛 4/6）→ **判定：{'PASS' if ok else 'FAIL'}**\n")

    # G5 方法学：全体组 delta 极差中位数 < 2pp
    L.append("### G5 方法学（对内 delta 轮间极差 < 2pp）\n")
    spreads = [v[3] for chs in by_size.values() for v in chs.values()]
    if spreads:
        med_sp = median(spreads)
        worst = max(spreads)
        ok = med_sp < 2.0
        L.append(f"- 全组 delta 极差：median = {med_sp:.2f}pp，最大 = {worst:.2f}pp"
                 f"（{len(spreads)} 组）")
        L.append(f"- **判定：{'PASS' if ok else 'FAIL'}**"
                 f"（对照：AR007 热浸没有会话绝对值漂移 ~5%）\n")

    L.append("## 备注\n")
    L.append("- 协议：run_paired.ps1 交替配对（A,B）×rounds，组间冷却门控；"
             "冷却超时组由脚本标注 thermal-contaminated（本表不区分，见执行日志）。")
    L.append("- delta 为 GF 比值；cublas 作为\"挑战者\"运行时其组值即同协议 cuBLAS 参考"
             "（G1 分母），其 delta 行 = swpipe/cuBLAS 相对关系。\n")

    report = "\n".join(L)
    print(report)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[compare] report written -> {out}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--paired", default=None, metavar="PAIRED_CSV",
                    help="AR008 thermal-paired 模式：读配对 CSV 输出对内 delta 报告")
    ap.add_argument("--baseline-kernel", default="swpipe",
                    help="paired 模式基线 kernel 名（默认 swpipe）")
    args = ap.parse_args()

    if args.paired:
        if not args.baseline:
            return paired_report(args.paired, args.baseline_kernel, args.out)
        print("[compare] --paired 模式忽略多余的位置参数", file=sys.stderr)
        return paired_report(args.paired, args.baseline_kernel, args.out)

    if not (args.baseline and args.new):
        ap.error("需要 <baseline.csv> <new.csv>，或 --paired <paired.csv>")

    base = load_rows(args.baseline)
    new = load_rows_last(args.new)

    # 新会话的尺寸集合（按规模排序）
    sizes = sorted({(r["m"], r["n"], r["k"]) for r in new.values()},
                   key=size_of)

    lines = []
    lines.append("# AR007 回归对比 — 基线 vs 新会话\n")
    lines.append(f"- baseline: `{args.baseline}`（每 key 首行 = E1 默认配置）")
    lines.append(f"- new     : `{args.new}`（每 key 末行 = AR007 矩阵）\n")

    # ---- delta 表（逐尺寸 × 全 kernel）----
    lines.append("## 逐尺寸 delta（GFLOPS）\n")
    for (m, n, k) in sizes:
        lines.append(f"### {m}x{n}x{k}\n")
        lines.append("| kernel | baseline GF | new GF | delta | new RSD% |")
        lines.append("|---|---:|---:|---:|---:|")
        cublas_new = new.get(("cublas", m, n, k))
        for kn in CUSTOM_KERNELS + ["cublas"]:
            b = base.get((kn, m, n, k))
            nw = new.get((kn, m, n, k))
            if nw is None:
                continue
            bg = f"{b['gf']:.1f}" if b else "n/a"
            delta = fmt_delta(b["gf"] if b else None, nw["gf"])
            lines.append(f"| {kn} | {bg} | {nw['gf']:.1f} | {delta} | {nw['rsd']:.2f} |")
        if cublas_new:
            best_custom = max(
                (new[(kn, m, n, k)]["gf"] for kn in CUSTOM_KERNELS
                 if (kn, m, n, k) in new), default=0.0)
            ratio = best_custom / cublas_new["gf"] * 100.0 if cublas_new["gf"] else 0.0
            lines.append(f"\n自研最优 {best_custom:.1f} GF = cuBLAS 的 {ratio:.1f}%\n")

    # ---- 四门判定 ----
    lines.append("## 四门判定（srs AR007 §1）\n")

    def cell(kn, m, n, k):
        return new.get((kn, m, n, k))

    def best_custom(m, n, k):
        cands = {kn: new[(kn, m, n, k)]["gf"] for kn in CUSTOM_KERNELS
                 if (kn, m, n, k) in new}
        if not cands:
            return None, None
        kn = max(cands, key=cands.get)
        return kn, cands[kn]

    # G-小尺寸
    lines.append("### G-小尺寸（自研最优 > cuBLAS）\n")
    lines.append("| size | 自研最优 | kernel | cuBLAS | 判定 |")
    lines.append("|---|---:|---|---:|---|")
    for (m, n, k) in sizes:
        if (m, n, k) not in [(256, 256, 256), (512, 512, 512)]:
            continue
        kn, g = best_custom(m, n, k)
        cb = cell("cublas", m, n, k)
        verdict = "PASS" if (g is not None and cb and g > cb["gf"]) else "FAIL"
        lines.append(f"| {m}^3 | {g:.1f} | {kn} | {cb['gf']:.1f} | {verdict} |")
    lines.append("")

    # G-大尺寸
    lines.append("### G-大尺寸（4096^3：>=7.2 TF 且 >= cuBLAS 75%）\n")
    kn, g = best_custom(4096, 4096, 4096)
    cb = cell("cublas", 4096, 4096, 4096)
    if g is not None and cb:
        ok = (g >= 7200.0) and (g >= 0.75 * cb["gf"])
        lines.append(f"- 自研最优 {kn} = {g:.1f} GF；cuBLAS = {cb['gf']:.1f} GF"
                     f"（{g / cb['gf'] * 100:.1f}%）")
        lines.append(f"- **判定：{'PASS' if ok else 'FAIL'}**\n")

    # G-全线
    lines.append("### G-全线（>=4/6 尺寸自研最优较基线刷新 >=2%）\n")
    lines.append("| size | 基线自研最优 | 新自研最优 | 刷新 |")
    lines.append("|---|---:|---:|---:|")
    hits = 0
    nsz = 0
    for (m, n, k) in sizes:
        kn_new, g_new = best_custom(m, n, k)
        if g_new is None:
            continue
        nsz += 1
        cands_b = {kn: base[(kn, m, n, k)]["gf"] for kn in CUSTOM_KERNELS
                   if (kn, m, n, k) in base}
        g_base = max(cands_b.values()) if cands_b else None
        if g_base:
            improved = (g_new - g_base) / g_base * 100.0
            mark = ">=2%" if improved >= 2.0 else "<2%"
            hits += 1 if improved >= 2.0 else 0
            lines.append(f"| {m}x{n}x{k} | {g_base:.1f} | {g_new:.1f} ({kn_new}) "
                         f"| {improved:+.1f}% {mark} |")
        else:
            lines.append(f"| {m}x{n}x{k} | n/a | {g_new:.1f} ({kn_new}) | n/a |")
    ok = hits >= 4
    lines.append(f"\n- 命中 {hits}/{nsz}（门槛 4/6）→ **判定：{'PASS' if ok else 'FAIL'}**\n")

    # G-K6
    lines.append("### G-K6（swpipe > vec4）\n")
    lines.append("| size | vec4 GF | swpipe GF | delta | 判定 |")
    lines.append("|---|---:|---:|---:|---|")
    k6_hits = 0
    k6_total = 0
    for (m, n, k) in sizes:
        v = cell("vec4", m, n, k)
        s = cell("swpipe", m, n, k)
        if not (v and s):
            continue
        k6_total += 1
        d = (s["gf"] - v["gf"]) / v["gf"] * 100.0
        win = d > 0
        k6_hits += 1 if win else 0
        lines.append(f"| {m}x{n}x{k} | {v['gf']:.1f} | {s['gf']:.1f} | {d:+.1f}% "
                     f"| {'WIN' if win else 'LOSS'} |")
    big_v = cell("vec4", 4096, 4096, 4096)
    big_s = cell("swpipe", 4096, 4096, 4096)
    ok = big_s is not None and big_v is not None and big_s["gf"] > big_v["gf"]
    lines.append(f"\n- 主判据（4096^3）：swpipe {'>' if ok else '<='} vec4 → "
                 f"**判定：{'PASS' if ok else 'FAIL'}**")
    lines.append(f"- 全尺寸战况：swpipe 胜 {k6_hits}/{k6_total}\n")

    lines.append("## 备注\n")
    lines.append("- WDDM 动态时钟环境（无 admin 锁频）：小尺寸 RSD 受单点离群影响，"
                 "median 语义保持稳健；跨轮 RSD>5% 已由 --rounds 门控重试。")
    lines.append("- 基线 git=84e261f（E1 会话，smem1d 为 bk16 默认）；新会话默认已按 "
                 "AR007 固化 bk32 —— smem1d delta 含默认值变更效应（授权见 srs §3.4）。\n")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[compare] report written -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

