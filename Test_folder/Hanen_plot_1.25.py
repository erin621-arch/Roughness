import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_arch_shape():
    # --- パラメータ設定 (画像の数値に基づきます) ---
    w = 0.25          # 全幅
    d = 0.20          # 全高
    h_side = 0.075    # 直線部分の高さ
    r = 0.125         # 半径 (R)
    
    # 検証: 高さの関係 (直線部 + 半径 = 全高) が正しいか確認
    # 0.075 + 0.125 = 0.20 なので正しい

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- 1. 形状の描画 (青い線) ---
    # 色は画像に近い落ち着いた青色に設定
    line_color = '#4c6ea5'
    line_width = 3

    # (A) 左側の直線 (下から上へ)
    ax.plot([0, 0], [0, h_side], color=line_color, lw=line_width)

    # (B) 上部の半円 (アーチ)
    # 角度を180度(pi)から0度まで変化させて円弧を描く
    theta = np.linspace(np.pi, 0, 100)
    # 中心の座標は (x=w/2, y=h_side)
    center_x = w / 2
    center_y = h_side
    
    x_arc = center_x + r * np.cos(theta)
    y_arc = center_y + r * np.sin(theta)
    
    ax.plot(x_arc, y_arc, color=line_color, lw=line_width)

    # (C) 右側の直線 (上から下へ)
    ax.plot([w, w], [h_side, 0], color=line_color, lw=line_width)

    # --- 2. 寸法線とテキストの描画 ---
    
    # 矢印の共通スタイル
    arrow_props = dict(arrowstyle='<->', color='black', lw=1.5)
    text_offset = 0.02

    # (A) 全高 (0.2) - 左側
    ax.annotate('', xy=(-0.03, 0), xytext=(-0.03, d), arrowprops=arrow_props)
    ax.text(-0.04, d/2, '0.2', va='center', ha='right', fontsize=16)

    # (B) 全幅 (0.25) - 下側
    ax.annotate('', xy=(0, -0.03), xytext=(w, -0.03), arrowprops=arrow_props)
    ax.text(w/2, -0.08, '0.25', va='center', ha='center', fontsize=16)

    # (C) 直線部の高さ (0.075) - 右側
    ax.annotate('', xy=(w + 0.02, 0), xytext=(w + 0.02, h_side), arrowprops=arrow_props)
    ax.text(w + 0.03, h_side/2, '0.075', va='center', ha='left', fontsize=16)

    # (D) 半径 (R0.125)
    # 中心から45度(pi/4)の方向へ矢印を引く
    arrow_angle = np.pi / 4
    arrow_len = r
    # 矢印の始点（中心）と終点（円周）
    start_x = center_x
    start_y = center_y
    end_x = center_x + r * np.cos(arrow_angle)
    end_y = center_y + r * np.sin(arrow_angle)
    
    # 半径の矢印 (片側矢印)
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(end_x + 0.01, end_y + 0.01, 'R0.125', fontsize=16)

    # --- 3. その他のテキスト ---
    # 日本語フォントが環境にない場合に備え、英語表記か標準フォントを使用
    # タイトル風に下部にテキスト配置
    plt.figtext(0.5, 0.05, f"w={w}, d={d:.2f}の場合", ha="center", fontsize=16)

    # --- グラフの設定 ---
    ax.set_aspect('equal') # アスペクト比を1:1にして円を正円にする
    ax.axis('off')         # 軸を消す
    
    # マージン調整
    plt.xlim(-0.15, w + 0.15)
    plt.ylim(-0.1, d + 0.05)
    
    plt.show()

if __name__ == "__main__":
    draw_arch_shape()