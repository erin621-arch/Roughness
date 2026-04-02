import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 日本語フォントの設定
try:
    font_path = ""
    font_names = [f.name for f in fm.fontManager.ttflist]
    if 'MS Gothic' in font_names:
        plt.rcParams['font.family'] = 'MS Gothic'
    elif 'AppleGothic' in font_names:
        plt.rcParams['font.family'] = 'AppleGothic'
    elif 'Noto Sans CJK JP' in font_names:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    else:
        plt.rcParams['font.family'] = 'sans-serif' 
except:
    pass

def draw_rectangular_groove_fixed():
    """画像: 矩形形の溝 (Rectangular Groove) - 修正版"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # パラメータ設定
    p = 10.0   # ピッチ
    w = 5.0    # 溝の幅
    d = 4.0    # 深さ
    
    # 描画範囲
    x_start = -p * 0.8
    x_end = p * 1.5
    
    # 形状データ
    x_points = [-p, -p+w, -p+w, 0, 0, w, w, p, p, p+w, p+w]
    y_points = [d,  d,    0,    0, d, d, 0, 0, d, d,   0]
    
    # プロファイル描画
    ax.plot(x_points, y_points, color='black', linewidth=2)
    ax.fill_between(x_points, y_points, d+2, color='lightgray', alpha=0.5)

    # --- 寸法線と注釈 ---
    
    # タイトルなど
    ax.text(w/2, d+3.5, "矩形形の溝", fontsize=24, fontweight='bold', ha='center')
    ax.plot([w/2-4, w/2+4], [d+3.3, d+3.3], color='black', linewidth=2)

    # --- 修正箇所: 深さ d (Depth) ---
    # 矢印の位置を少し左にずらします（オフセット）
    offset_d = -1.5
    
    # 矢印を描画 (x=offset_d の位置)
    ax.annotate('', xy=(offset_d, 0), xytext=(offset_d, d), arrowprops=dict(arrowstyle='<->', lw=1.5))
    
    # テキスト
    ax.text(offset_d - 0.5, d/2, "深さ $d$", fontsize=18, va='center', ha='right')
    
    # 補助線 (Extension lines)
    # 形状の端(x=0)から矢印の位置(offset_d)まで線を引きます
    # 少しだけ突き出させるために -0.2 を加えています
    ax.plot([0, offset_d - 0.2], [0, 0], 'k-', lw=0.5) # 下レベル
    ax.plot([0, offset_d - 0.2], [d, d], 'k-', lw=0.5) # 上レベル (ここを修正)

    # --- 幅 w (Width) ---
    # 前回の修正を維持: テキストを下に配置
    ax.annotate('', xy=(0, d/2), xytext=(w, d/2), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(w/2, d/2 - 0.3, "幅 $w$", fontsize=18, ha='center', va='top') 

    # --- ピッチ p (Pitch) ---
    y_pitch = -1.0
    ax.annotate('', xy=(0, y_pitch), xytext=(p, y_pitch), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(p/2, y_pitch - 0.5, "ピッチ $p$", fontsize=18, ha='center', va='top')
    ax.plot([0, 0], [0, y_pitch - 0.2], 'k-', lw=1)
    ax.plot([p, p], [0, y_pitch - 0.2], 'k-', lw=1)

    # 設定
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-3, d + 5)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("修正版を描画します...")
    draw_rectangular_groove_fixed()