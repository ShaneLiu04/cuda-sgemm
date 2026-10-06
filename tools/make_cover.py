#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成仓库封面 docs/cover.png（纯 PIL 绘制，无外部素材，可复现）。

设计语言：深空蓝渐变 + NVIDIA 绿强调 + 性能阶梯柱状主视觉。
尺寸 1600x800；输出经调色板量化控制在 72KB 以内（与图表归档体积纪律一致）。
"""

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "cover.png")
SIZE_TARGET = 72 * 1024

W, H = 1600, 800
FONTS = r"C:\Windows\Fonts"
GREEN = (118, 185, 0)
WHITE = (255, 255, 255)
GRAY = (168, 181, 198)
DIM = (122, 136, 155)


def font(name, size, index=0):
    return ImageFont.truetype(os.path.join(FONTS, name), size, index=index)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_background(img):
    """深空蓝对角渐变 + 细网格 + 右上角辉光。"""
    top = (8, 18, 32)
    bottom = (14, 34, 56)
    px = img.load()
    for y in range(H):
        for x in range(0, W, 4):
            t = (x / W * 0.35 + y / H * 0.65)
            c = lerp(top, bottom, t)
            for dx in range(4):
                if x + dx < W:
                    px[x + dx, y] = c
    draw = ImageDraw.Draw(img, "RGBA")
    step = 64
    for x in range(0, W, step):
        draw.line([(x, 0), (x, H)], fill=(255, 255, 255, 7))
    for y in range(0, H, step):
        draw.line([(0, y), (W, y)], fill=(255, 255, 255, 7))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = W - 350, -80
    for r in range(480, 40, -20):
        alpha = int(9 * (480 - r) / 440) + 2
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 130, 60, alpha))
    img.alpha_composite(glow)


def draw_tracked(draw, pos, text, fnt, fill, tracking=6):
    """带字距的文本（PIL 无原生 tracking，逐字符绘制）。"""
    x, y = pos
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + tracking


def ladder_heights():
    """K0-K9 实测 GFLOPS（4096^3, median）→ 归一化柱高（log 标度）。

    十一根柱：K0-K6 正向演进 + swsk(K6')/ws(K7)/wide(K8) 含负结果柱 +
    deep(K9) 峰值——负结果同样上封面，与本工程"失败实验归档"军规一致。
    """
    import math
    perf = [157.0, 792.4, 2874.7, 4683.6, 6391.3, 5647.1,
            6584.8, 6250.4, 5377.7, 4408.8, 8560.3]
    lo, hi = math.log(120), math.log(12000)
    return perf, [0.16 + 0.84 * (math.log(p) - lo) / (hi - lo) for p in perf]


def draw_ladder(img):
    """右侧性能阶梯主视觉：11 根渐变柱 + cuBLAS 参考线。"""
    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, bw, gap = 900, 620, 46, 13
    perf, hs = ladder_heights()
    labels = ["K0", "K1", "K2", "K3", "K4", "K5",
              "K6", "K6'", "K7", "K8", "K9"]
    neg = {5, 8, 9}   # cpasync / ws / wide —— 负结果柱降饱和
    max_h = 430
    for i, (p, h) in enumerate(zip(perf, hs)):
        bh = int(max_h * h)
        x = x0 + i * (bw + gap)
        y_top = y0 - bh
        t = i / 10
        if i in neg:
            c_top = lerp((30, 40, 48), (74, 92, 60), t)
            c_bot = lerp((12, 18, 24), (30, 40, 26), t)
        else:
            c_top = lerp((38, 70, 44), GREEN, t)
            c_bot = lerp((14, 30, 22), (46, 92, 20), t)
        for yy in range(bh):
            draw.line([(x, y_top + yy), (x + bw, y_top + yy)],
                      fill=lerp(c_top, c_bot, yy / max(1, bh - 1)))
        draw.rectangle([x, y_top - 3, x + bw, y_top], fill=WHITE)
        lbl = font("consola.ttf", 18)
        draw.text((x + bw // 2, y_top - 32), labels[i], font=lbl,
                  fill=GRAY, anchor="ma")
        val = font("consola.ttf", 14)
        draw.text((x + bw // 2, y0 + 12), f"{p:,.0f}", font=val,
                  fill=DIM, anchor="ma")
    draw.text((x0 + 5 * (bw + gap) + bw // 2, y0 + 40),
              "kernel 版本（4096³ 实测 GFLOPS，log 标度；暗色柱 = 负结果归档）",
              font=font("msyh.ttc", 17, 1), fill=DIM, anchor="ma")
    import math as _m
    lo, hi = _m.log(120), _m.log(12000)
    cub_t = 0.16 + 0.84 * (_m.log(10136.2) - lo) / (hi - lo)
    cub_y = y0 - int(max_h * cub_t)
    for xx in range(x0 - 30, x0 + 11 * (bw + gap), 18):
        draw.line([(xx, cub_y), (xx + 9, cub_y)], fill=(220, 190, 90, 200))
    draw.text((x0 + 11 * (bw + gap) - 26, cub_y - 26), "cuBLAS",
              font=font("consola.ttf", 17), fill=(220, 190, 90), anchor="ra")


def draw_text_block(img):
    draw = ImageDraw.Draw(img, "RGBA")
    draw_tracked(draw, (96, 96), "CUDA  PERFORMANCE  ENGINEERING",
                 font("consola.ttf", 22), GREEN, tracking=4)
    draw.text((92, 140), "cuda-sgemm", font=font("segoeuib.ttf", 112), fill=WHITE)
    draw.text((96, 288), "十六版 SGEMM 逐层实测 — 负结果同样归档",
              font=font("msyhbd.ttc", 36, 1), fill=(220, 228, 238))
    draw.text((96, 344), "从教科书 kernel 到逼近 cuBLAS：每一步优化都被实测证据钉住",
              font=font("msyh.ttc", 22, 1), fill=GRAY)
    draw.rectangle([96, 410, 320, 414], fill=GREEN)
    stats = [
        ("54.6×", "naive → deep 提升比"),
        ("8,568", "GFLOPS @ 2048³（同会话）"),
        ("124.6%", "vs cuBLAS @ 256³"),
        ("146/146", "正确性 + 21/21 逐位"),
    ]
    x = 96
    for value, label in stats:
        draw.text((x, 452), value, font=font("segoeuib.ttf", 46), fill=WHITE)
        draw.text((x, 516), label, font=font("msyh.ttc", 19, 1), fill=DIM)
        x += 220
    draw.line([(96, 736), (W - 96, 736)], fill=(255, 255, 255, 28))
    draw.text((96, 754), "Quadro RTX 5000   ·   Turing sm_75   ·   48 SM   ·   CUDA 12.5   ·   严格 FP32（IEEE FMA）",
              font=font("consola.ttf", 19), fill=DIM)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img = Image.new("RGBA", (W, H))
    draw_background(img)
    draw_ladder(img)
    draw_text_block(img)
    img = img.convert("RGB")
    colors = 256
    while True:
        img.quantize(colors=colors).save(OUT, optimize=True)
        if os.path.getsize(OUT) <= SIZE_TARGET or colors <= 64:
            break
        colors //= 2
    print(f"cover written: {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, colors={colors})")


if __name__ == "__main__":
    main()
