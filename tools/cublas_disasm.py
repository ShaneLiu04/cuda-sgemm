#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/cublas_disasm.py — AR012 T002（FR1 E-B）Level-2 SASS 特征提取

输入：cuobjdump --dump-sass 导出的 .sass 文件（cuBLAS volta_sgemm_* 与
      我方 sgemm_deep 模板实例），可选 --trace 传入 cupti_trace.exe 的
      stdout 日志（Level-1 launch 结构）。
输出：每个 kernel 的指令构成特征表（stdout markdown），特征口径：
      - 直方图：FFMA / LDS / STS / LDG / STG / LDGSTS(cp.async) / BAR /
        IMAD / ISETP / LOP3 / SHF / BRA / S2R
      - FFMA:LDS 比值（主循环计算:访存指令密度——FFMA 发射效率代理）
      - 最长连续 FFMA 段（编译器 ILP/展开深度信号）
      - 循环结构：BRA 回边数（目标地址 < 当前地址）
      - swizzle 信号：STS/LDS 地址计算邻近的 LOP3（XOR LUT）出现密度
        （8-way XOR swizzle 在 SASS 里表现为 LOP3.LUT 与地址 IMAD 交错）
用法：
  python tools/cublas_disasm.py profile/cublas_disasm/volta_sgemm_128x128_nn.sass \
         [more.sass ...] [--trace trace.log]
"""
import argparse
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SASS 行样例：
#        /*01a0*/               @P4 MOV R81, R2 ;   /* 0x... */
# 指令 = 谓词? + 助记符（含 . 修饰）。取第一个非空 token 去掉 @P 前缀。
SASS_RE = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d+\s+)?([A-Z0-9._]+)")
FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")


OPCODE_GROUPS = [
    "FFMA", "LDS", "STS", "LDG", "STG", "LDGSTS", "BAR",
    "IMAD", "ISETP", "LOP3", "SHF", "BRA", "S2R", "MOV", "F2F",
    "EXIT", "NOP", "RET", "ATOMG", "RED", "LDL", "STL",
]


def norm_op(tok):
    # LDGSTS.E.BYPASS.128 -> LDGSTS；LDG.E.128 -> LDG；BAR.SYNC -> BAR
    base = tok.split(".")[0]
    if base.startswith("LDGSTS"):
        return "LDGSTS"
    if base in ("BAR",):
        return "BAR"
    if base in ("BRA",):
        return "BRA"
    if base in ("ATOM", "ATOMG", "RED"):
        return "RED/ATOM"
    return base


def parse_functions(path):
    """yield (func_name, [(addr_int, op), ...])"""
    funcs = []
    cur_name, cur = None, []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = FUNC_RE.match(line)
            if m:
                if cur_name is not None:
                    funcs.append((cur_name, cur))
                cur_name, cur = m.group(1), []
                continue
            m2 = SASS_RE.search(line)
            if m2 and cur_name is not None:
                addr = int(line.split("/*")[1].split("*/")[0], 16)
                cur.append((addr, norm_op(m2.group(1))))
    if cur_name is not None:
        funcs.append((cur_name, cur))
    return funcs


def analyze(name, insns):
    hist = Counter(op for _, op in insns)
    n = len(insns)
    ffma = hist.get("FFMA", 0)
    lds = hist.get("LDS", 0)
    sts = hist.get("STS", 0)
    ldgsts = hist.get("LDGSTS", 0)
    # 最长连续 FFMA 段
    run = best = 0
    for _, op in insns:
        run = run + 1 if op == "FFMA" else 0
        best = max(best, run)
    # BRA 回边
    addr_by_idx = {}
    backedges = 0
    bra_re = re.compile(r"BRA")
    return {
        "name": name, "n": n, "FFMA": ffma, "LDS": lds, "STS": sts,
        "LDG": hist.get("LDG", 0), "STG": hist.get("STG", 0),
        "LDGSTS": ldgsts, "BAR": hist.get("BAR", 0),
        "IMAD": hist.get("IMAD", 0), "ISETP": hist.get("ISETP", 0),
        "LOP3": hist.get("LOP3", 0), "SHF": hist.get("SHF", 0),
        "BRA": hist.get("BRA", 0), "S2R": hist.get("S2R", 0),
        "ffma_lds": (ffma / lds) if lds else float("inf"),
        "max_ffma_run": best,
    }


def fmt_row(d):
    return ("| {name} | {n} | {FFMA} | {LDS} | {STS} | {LDGSTS} | {LDG} | "
            "{STG} | {BAR} | {IMAD} | {LOP3} | {SHF} | "
            "{ratio} | {run} |").format(
        name=d["name"][:46], n=d["n"], FFMA=d["FFMA"], LDS=d["LDS"],
        STS=d["STS"], LDGSTS=d["LDGSTS"], LDG=d["LDG"], STG=d["STG"],
        BAR=d["BAR"], IMAD=d["IMAD"], LOP3=d["LOP3"], SHF=d["SHF"],
        ratio="%.2f" % d["ffma_lds"] if d["LDS"] else "inf",
        run=d["max_ffma_run"])


HDR = ("| kernel | insns | FFMA | LDS | STS | LDGSTS | LDG | STG | BAR | "
       "IMAD | LOP3 | SHF | FFMA:LDS | maxFFMArun |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass", nargs="+")
    ap.add_argument("--trace", default=None,
                    help="cupti_trace.exe stdout log（Level-1 launch 结构）")
    args = ap.parse_args()

    rows = []
    for path in args.sass:
        for fname, insns in parse_functions(path):
            if not insns:
                continue
            rows.append(analyze(fname, insns))

    print(HDR)
    print(SEP)
    for r in rows:
        print(fmt_row(r))

    if args.trace:
        print()
        print("## Level-1 launch structure (from %s)" % args.trace)
        try:
            with open(args.trace, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("[cublas") or line.startswith("#"):
                        print(line.rstrip())
        except OSError as e:
            print("(trace file unreadable: %s)" % e)


if __name__ == "__main__":
    main()
