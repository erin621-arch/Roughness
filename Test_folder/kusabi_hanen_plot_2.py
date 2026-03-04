import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 日本語フォントの設定（環境に合わせて調整してください）
# Windows: 'MS Gothic', Mac: 'AppleGothic', Linux: 'IPAGothic' など
import matplotlib.font_manager as fm
try:
    # 一般的な日本語フォントを探して設定
    font_path = ""
    font_names = [f.name for f in fm.fontManager.ttflist]
    if 'MS Gothic' in font_names:
        plt.rcParams['font.family'] = 'MS Gothic'
    elif 'AppleGothic' in font_names:
        plt.rcParams['font.family'] = 'AppleGothic'
    elif 'Noto Sans CJK JP' in font_names:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    else:
        # フォントが見つからない場合のフォールバック（豆腐になります）
        print("日本語フォントが見つかりませんでした。font.familyを手動で設定してください。")
except:
    pass

def draw_wedge_groove():
    """画像1: くさび形の溝 の描画"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # パラメータ（比率を再現）
    p = 10.0  # ピッチ
    d = 4.0   # 深さ
    offset_x = -5.0 # 左側の開始位置
    
    # 座標の定義
    x = [offset_x, 0, 0, p, p + 4]
    y = [d * (1 + offset_x/p * -1), 0, d, 0, d * (1 + -4/p * -1)] 
    # 左側の坂の計算を修正: y = - (d/p) * x
    # 左端: x=-10, y=4 -> x=0, y=0. 
    # 右側: x=0, y=4 -> x=10, y=0
    
    # 描画用の点を整理
    # 左の坂（途中から）
    x_line = [-p*0.8, 0, 0, p, p*1.3]
    y_line = [d*0.8, 0, d, 0, d * (-0.3 * -1)] # 簡易的な傾き再現
    
    # 正確な線分データの作成
    # Line 1: 左上から原点へ
    ax.plot([-p, 0], [d, 0], color='black', linewidth=2)
    # Line 2: 原点から真上へ (深さd)
    ax.plot([0, 0], [0, d], color='black', linewidth=2)
    # Line 3: 真上から右下へ
    ax.plot([0, p], [d, 0], color='black', linewidth=2)
    # Line 4: 右下から真上へ
    ax.plot([p, p], [0, d], color='black', linewidth=2)
    # Line 5: 右への続き
    ax.plot([p, p+3], [d, d*(1-3/p)], color='black', linewidth=2)

    # 塗りつぶし（試験体内部）
    # 左側
    ax.fill_between([-p, 0], [d, 0], [d+2, d+2], color='lightgray', alpha=0.5)
    # メイン部分
    ax.fill_between([0, p], [d, 0], [d+2, d+2], color='lightgray', alpha=0.5)
    # 右側
    ax.fill_between([p, p+3], [d, d*(1-3/p)], [d+2, d+2], color='lightgray', alpha=0.5)

    # テキストと注釈
    
    # タイトル
    ax.text(-p, d+3, "くさび形の溝", fontsize=24, fontweight='bold')
    ax.plot([-p, -p+8], [d+2.8, d+2.8], color='black', linewidth=2) # アンダーライン

    # 試験体内部
    ax.text(-p/2, d+0.5, "試験体内部", fontsize=18)
    
    # 底面
    ax.text(-p, -1, "底面", fontsize=18)

    # 寸法線: 深さ d
    ax.annotate('', xy=(0.5, 0), xytext=(0.5, d), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(1.0, d/2, "深さ $d$", fontsize=18, va='center')
    # 補助線
    ax.plot([0, 2], [d, d], 'k-', lw=0.5)
    ax.plot([0, 2], [0, 0], 'k-', lw=0.5)

    # 寸法線: ピッチ p
    ax.annotate('', xy=(0, -1.5), xytext=(p, -1.5), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(p/2, -2.5, "ピッチ $p$", fontsize=18, ha='center')
    # 補助線
    ax.plot([0, 0], [0, -2], 'k-', lw=1)
    ax.plot([p, p], [0, -2], 'k-', lw=1)

    # 角度 alpha
    # 円弧を描く
    arc = patches.Arc((p/2 + 2, 0), 4, 4, theta1=155, theta2=180, color='steelblue', lw=1.5)
    ax.add_patch(arc)
    # 矢印の先端だけそれっぽく描画
    ax.annotate('', xy=(p/2 + 2 - 2, 0), xytext=(p/2 + 2, 0), arrowprops=dict(arrowstyle='<->', color='steelblue')) # 簡易表現
    # 実際には角度の円弧に矢印をつけるのは複雑なので、直感的な位置に配置
    ax.annotate('', xy=(p*0.6, d*0.25), xytext=(p*0.6, 0), arrowprops=dict(arrowstyle='<->', color='steelblue'))
    ax.text(p*0.55, d*0.15, r'$\alpha$', fontsize=16)

    # 右上の注釈: 溝の底R
    ax.annotate('溝の底RはR0.03以下', xy=(p, d), xytext=(p-2, d+4),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.5), fontsize=16)
    # 角の丸印
    circle = patches.Circle((p, d), radius=0.2, fill=False, color='black', lw=1)
    ax.add_patch(circle)

    # 設定
    ax.set_xlim(-p-1, p+4)
    ax.set_ylim(-3, d+5)
    ax.axis('off') # 軸を消す
    plt.tight_layout()
    plt.show()

def draw_rounded_profile():
    """画像2: 半円形のプロファイル の描画"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # パラメータ
    w = 0.25
    h_total = 0.2
    h_straight = 0.075
    r = 0.125  # w/2 と一致

    # 中心座標
    center_x = 0
    center_y = h_straight

    # 図形の描画（青い線）
    # 左の直線
    ax.plot([-r, -r], [0, h_straight], color='steelblue', linewidth=3)
    # 右の直線
    ax.plot([r, r], [0, h_straight], color='steelblue', linewidth=3)
    
    # 上の半円
    theta = np.linspace(0, np.pi, 100)
    x_arc = r * np.cos(theta)
    y_arc = h_straight + r * np.sin(theta)
    ax.plot(x_arc, y_arc, color='steelblue', linewidth=3)

    # 寸法線とテキスト

    # 幅 0.25
    ax.annotate('', xy=(-r, -0.02), xytext=(r, -0.02), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(0, -0.05, "0.25", fontsize=20, ha='center')

    # 高さ 0.2
    ax.annotate('', xy=(-r - 0.05, 0), xytext=(-r - 0.05, h_total), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(-r - 0.12, h_total/2, "0.2", fontsize=20, va='center')

    # 直線部高さ 0.075
    ax.annotate('', xy=(r + 0.05, 0), xytext=(r + 0.05, h_straight), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(r + 0.07, h_straight/2, "0.075", fontsize=20, va='center')

    # 半径 R0.125
    # 中心から円弧への矢印
    arrow_angle = np.pi / 3 # 60度
    arrow_len = r
    ax.annotate('', xy=(r*np.cos(arrow_angle), h_straight + r*np.sin(arrow_angle)), 
                xytext=(0, h_straight), 
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.05, h_straight + r + 0.02, "R0.125", fontsize=20)

    # 下部のテキスト
    ax.text(0, -0.1, "w=0.25, d=0.20の場合", fontsize=20, ha='center')

    # 設定
    ax.set_aspect('equal')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.15, 0.3)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# 描画の実行
if __name__ == "__main__":
    print("画像1を描画します...")
    draw_wedge_groove()
    print("画像2を描画します...")
    draw_rounded_profile()