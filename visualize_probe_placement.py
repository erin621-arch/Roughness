import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans JP'

# ====== パラメータ（guiyou_0113.py から） ======
mesh_length = 1.0e-5
x_length = 0.02
y_length = 0.04
nx = int(x_length / mesh_length)  # 2000
ny = int(y_length / mesh_length)  # 4000

f_pitch = 2.00e-3
f_width = 0.25e-3
f_depth = 0.10e-3

mn_p  = int(f_pitch / mesh_length)   # 200
mn_nf = int((f_pitch - f_width) / mesh_length)  # 175
mn_d  = int(f_depth / mesh_length)   # 10
num_f = int(y_length / f_pitch)      # 20

probe_d = 0.005  # 5 mm
sy    = (1799 + 1975) / 2            # 1887.0
sy_l  = sy - int(probe_d / mesh_length / 2)  # 1637
sy_r  = sy + int(probe_d / mesh_length / 2)  # 2137

flaw_starts = [mn_nf + i * mn_p for i in range(num_f) if (i + 1) * mn_p < ny]
flaw_ends   = [(i + 1) * mn_p   for i in range(num_f) if (i + 1) * mn_p < ny]

fs8, fe8 = flaw_starts[8], flaw_ends[8]   # 1775, 1800
fs9, fe9 = flaw_starts[9], flaw_ends[9]   # 1975, 2000

# ====== 描画 ======
fig, axes = plt.subplots(1, 2, figsize=(15, 9),
                         gridspec_kw={"width_ratios": [1, 1.7]})
fig.suptitle("探触子と傷の位置関係\n（pitch=2.00mm, depth=0.10mm, probe_d=5mm）",
             fontsize=13)

# ========== 左パネル：全体俯瞰 ==========
ax = axes[0]
# x=0 (探触子面) が上、x=nx (背面) が下（+60 mesh の余白を下方に追加）
ax.set_xlim(0, ny)
ax.set_ylim(nx + 60, 0)
ax.set_xlabel("y 方向 [mesh]", labelpad=6)
ax.set_ylabel("x 方向（深さ） [mesh]", labelpad=6)
ax.set_title("全体俯瞰\n（上: 探触子面 x=0、下: 背面 x=nx）")

# 金属ブロック（黄色塗りつぶし＋黒枠）
ax.set_facecolor("lightyellow")
for sp in ax.spines.values():
    sp.set_edgecolor("black")
    sp.set_linewidth(2.0)

# 全傷（赤）
for fs, fe in zip(flaw_starts, flaw_ends):
    ax.add_patch(mpatches.Rectangle(
        (fs, nx - mn_d), fe - fs, mn_d,
        facecolor="tomato", edgecolor="none", alpha=0.85, zorder=2
    ))

# 傷 i=8・i=9 を強調（濃い赤）
for idx in [8, 9]:
    fs, fe = flaw_starts[idx], flaw_ends[idx]
    ax.add_patch(mpatches.Rectangle(
        (fs, nx - mn_d), fe - fs, mn_d,
        facecolor="crimson", edgecolor="black", lw=0.8, zorder=3
    ))

# 探触子（青バー、探触子面 x=0 の上端に表示）
probe_h_left = int(nx * 0.012)
ax.add_patch(mpatches.Rectangle(
    (sy_l, 0), sy_r - sy_l, probe_h_left,
    facecolor="royalblue", edgecolor="black", lw=0.8, alpha=0.9, zorder=4
))
ax.axvline(x=sy, color="royalblue", lw=0.9, ls="--", alpha=0.7, zorder=3)

# 背面ライン（x=nx）と探触子面ライン（x=0）を明示
ax.axhline(y=nx, color="black", lw=2.5, zorder=6)
ax.text(ny * 0.02, nx + 8, "背面（傷面）  x = nx = 2000",
        ha="left", va="top", fontsize=8, color="black")
ax.axhline(y=0, color="dimgray", lw=1.5, ls="--", zorder=6)
ax.text(ny * 0.02, 0 + 8, "探触子面  x = 0",
        ha="left", va="top", fontsize=8, color="dimgray")

legend_elements = [
    mpatches.Patch(facecolor="lightyellow", edgecolor="black", lw=1.5, label="金属（鋼材）"),
    mpatches.Patch(facecolor="tomato",      edgecolor="none",         label="傷（矩形溝）"),
    mpatches.Patch(facecolor="crimson",     edgecolor="black", lw=0.8, label="傷 i=8・i=9（注目）"),
    mpatches.Patch(facecolor="royalblue",   edgecolor="black", lw=0.8, label="探触子（sy±250）"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

# ========== 右パネル：拡大図 ==========
ax2 = axes[1]

y_lo, y_hi   = 1600, 2200   # x方向（横軸）の表示範囲：探触子全幅を包含
x_depth_show = 35            # 材料内部の表示深さ [mesh]
ann_space    = 30            # 表面より外側の注釈スペース [mesh]

# ylim: 底 = nx + ann_space（空間側）、天 = nx - x_depth_show（材料深部）
#        ※ 大きい値が下 → inverted y
ax2.set_xlim(y_lo, y_hi)
ax2.set_ylim(nx + ann_space, nx - x_depth_show)

ax2.set_xlabel("y 方向 [mesh]  (1 mesh = 10 µm)", labelpad=6)
ax2.set_ylabel("x 方向 [mesh]", labelpad=6)
ax2.set_title("拡大図：傷 i=8・i=9 と探触子の位置\n（表面 = 破線、上方向 = 材料内部）")

# ---- 金属ブロック（黄色＋黒枠） ----
ax2.add_patch(mpatches.Rectangle(
    (y_lo, nx - x_depth_show), y_hi - y_lo, x_depth_show,
    facecolor="lightyellow", edgecolor="black", linewidth=2.0, zorder=1
))

# ---- 表面ライン ----
ax2.axhline(y=nx, color="black", lw=2.0, zorder=5)

# ---- 傷 ----
ax2.add_patch(mpatches.Rectangle(
    (fs8, nx - mn_d), fe8 - fs8, mn_d,
    facecolor="tomato", edgecolor="black", lw=0.8, zorder=3,
    label=f"傷 i=8  y=[{fs8}, {fe8}]"
))
ax2.add_patch(mpatches.Rectangle(
    (fs9, nx - mn_d), fe9 - fs9, mn_d,
    facecolor="tomato", edgecolor="black", lw=0.8, zorder=3,
    label=f"傷 i=9  y=[{fs9}, {fe9}]"
))

# ---- 探触子バー（表面より外側の注釈スペースに描画） ----
probe_bar_y = nx + 4
probe_bar_h = 6
ax2.add_patch(mpatches.Rectangle(
    (sy_l, probe_bar_y), sy_r - sy_l, probe_bar_h,
    facecolor="royalblue", edgecolor="black", lw=0.8, alpha=0.85, zorder=5,
    label=f"探触子  [{int(sy_l)}, {int(sy_r)}]"
))

# 探触子中心線
ax2.axvline(x=sy, color="royalblue", lw=1.5, ls="--", zorder=6,
            label=f"探触子中心 sy={int(sy)}")

# 傷端の点線
ax2.axvline(x=fe8, color="gray", lw=0.8, ls=":", alpha=0.7)
ax2.axvline(x=fs9, color="gray", lw=0.8, ls=":", alpha=0.7)

# ---- 注釈（表面外側の空間に集約） ----
# 注釈の y 位置（表面=nx より下方 = 大きい y 値）
y_ann1 = nx + 13   # 平坦部の矢印
y_ann2 = nx + 21   # 探触子径の矢印

# (1) 平坦部（傷 i=8 末端 〜 傷 i=9 先端）
ax2.annotate("", xy=(fs9, y_ann1), xytext=(fe8, y_ann1),
             arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2.0), zorder=7)
ax2.text((fe8 + fs9) / 2, y_ann1 + 1.5,
         f"平坦部: {(fs9 - fe8) * mesh_length * 1e3:.2f} mm  ({fs9 - fe8} mesh)",
         ha="center", va="top", fontsize=9, color="darkgreen")

# (2) 探触子径
ax2.annotate("", xy=(sy_r, y_ann2), xytext=(sy_l, y_ann2),
             arrowprops=dict(arrowstyle="<->", color="royalblue", lw=2.0), zorder=7)
ax2.text(sy, y_ann2 + 1.5,
         f"探触子径: {probe_d * 1e3:.1f} mm  ({int(sy_r - sy_l)} mesh)",
         ha="center", va="top", fontsize=9, color="royalblue")

# ---- 注釈（材料内部） ----
# (3) ピッチ矢印（材料内、傷より深い位置）
y_pitch = nx - x_depth_show + 9
ax2.annotate("", xy=(fe9, y_pitch), xytext=(fe8, y_pitch),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=2.0), zorder=6)
ax2.text((fe8 + fe9) / 2, y_pitch - 2,
         f"pitch: {f_pitch * 1e3:.2f} mm",
         ha="center", va="bottom", fontsize=9, color="purple")

# (4) 傷の深さ矢印（傷 i=8 と i=9 の間、材料内）
y_depth_x = (fe8 + fs9) / 2   # x位置（y方向中間）
ax2.annotate("", xy=(y_depth_x, nx), xytext=(y_depth_x, nx - mn_d),
             arrowprops=dict(arrowstyle="<->", color="sienna", lw=1.8), zorder=6)
ax2.text(y_depth_x + 8, nx - mn_d / 2,
         f"depth: {f_depth * 1e3:.2f} mm",
         ha="left", va="center", fontsize=8.5, color="sienna")

# (5) 傷幅ラベル（傷の外側上方）
for fs, fe, tag in [(fs8, fe8, "i=8"), (fs9, fe9, "i=9")]:
    ax2.text((fs + fe) / 2, nx - mn_d - 3,
             f"幅\n{f_width*1e3:.2f}mm",
             ha="center", va="bottom", fontsize=7.5, color="darkred",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

# ---- 凡例 ----
ax2.legend(loc="upper left", fontsize=8.5)

# ---- 軸ラベル ----
# y軸 (x方向) の主要ティック
ax2.set_yticks([nx - x_depth_show, nx - mn_d, nx, nx + ann_space])
ax2.set_yticklabels([
    f"{nx - x_depth_show}\n(材料内)",
    f"{nx - mn_d}\n(傷底)",
    f"{nx}\n(表面)",
    f"{nx + ann_space}\n(空間側)"
], fontsize=7.5)

# x軸 (y方向) のティック
xticks = np.arange(1600, 2250, 100)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f"{v}\n({v * mesh_length * 1e3:.1f}mm)" for v in xticks], fontsize=7.5)

plt.tight_layout()
plt.savefig("probe_placement.png", dpi=150, bbox_inches="tight")
print("saved: probe_placement.png")
plt.show()
