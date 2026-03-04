import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_staggered_grid_refined():
    # 図のサイズ設定
    fig, ax = plt.subplots(figsize=(10, 8))

    # グリッドの設定（3x3セル分を表示）
    rows = 3  # i インデックス (縦方向 / x軸)
    cols = 3  # j インデックス (横方向 / z軸)
    
    # レイヤーの順序 (zorder) の定義
    grid_zorder = 1        # 格子線（最背面）
    component_zorder = 10  # 図形と文字（前面）

    # 1. 格子線を描画
    # 横線 (i方向 / x軸)
    for i in range(rows):
        ax.hlines(-i, -0.2, cols-0.3, color='red', linestyle='--', alpha=0.6, linewidth=1.5, zorder=grid_zorder)
        label_text = f'$i = {i}$' if i < rows-1 else '$i = n_x$'
        ax.text(-0.5, -i, label_text, color='black', va='center', ha='right', fontsize=12, fontweight='bold')

    # 縦線 (j方向 / z軸)
    for j in range(cols):
        ax.vlines(j, -rows+0.3, 0.2, color='blue', linestyle='--', alpha=0.6, linewidth=1.5, zorder=grid_zorder)
        label_text = f'$j = {j}$' if j < cols-1 else '$j = n_z$'
        ax.text(j, 0.5, label_text, color='black', va='bottom', ha='center', fontsize=12, fontweight='bold')

    # 2. コンポーネント（速度・応力）を配置
    for i in range(rows):
        for j in range(cols):
            # Uz (青色矢印): 格子点 (j, i)
            # 矢印
            ax.annotate('', xy=(j + 0.35, -i), xytext=(j - 0.25, -i),
                        arrowprops=dict(facecolor='dodgerblue', edgecolor='black', shrink=0, width=8, headwidth=14),
                        zorder=component_zorder)
            # 文字（不透明背景で線を隠す）
            ax.text(j, -i, '$U_z$', color='white', ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='dodgerblue', edgecolor='black', alpha=1.0, pad=2, lw=0.5),
                    zorder=component_zorder+1)
            
            # T1, T3 (黄色ボックス): (j+0.5, i)
            if j < cols - 1:
                # alpha=1.0で背後の線を隠す
                rect = patches.Rectangle((j + 0.25, -i - 0.18), 0.5, 0.36, linewidth=1.5, 
                                        edgecolor='darkorange', facecolor='orange', alpha=1.0, zorder=component_zorder)
                ax.add_patch(rect)
                ax.text(j + 0.5, -i, '$T_1, T_3$', color='white', ha='center', va='center', fontsize=10, fontweight='bold', 
                        zorder=component_zorder+1)
            
            # T5 (紫色ボックス): (j, i+0.5)
            if i < rows - 1:
                rect = patches.Rectangle((j - 0.25, -i - 0.75), 0.5, 0.5, linewidth=1.5, 
                                        edgecolor='indigo', facecolor='purple', alpha=1.0, zorder=component_zorder)
                ax.add_patch(rect)
                ax.text(j, -i - 0.5, '$T_5$', color='white', ha='center', va='center', fontsize=10, fontweight='bold',
                        zorder=component_zorder+1)
                
            # Ux (赤色矢印): (j+0.5, i+0.5)
            if i < rows - 1 and j < cols - 1:
                ax.annotate('', xy=(j + 0.5, -i - 0.8), xytext=(j + 0.5, -i - 0.2),
                            arrowprops=dict(facecolor='tomato', edgecolor='black', shrink=0.05, width=8, headwidth=14),
                            zorder=component_zorder)
                ax.text(j + 0.5, -i - 0.5, '$U_x$', color='white', ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='tomato', edgecolor='black', alpha=1.0, pad=2, lw=0.5),
                        zorder=component_zorder+1)
                

    # レイアウト調整
    ax.set_xlim(-1, cols)
    ax.set_ylim(-rows, 1)
    ax.axis('off')
    plt.tight_layout()
    
    # 保存
    plt.savefig('staggered_grid_refined.png', dpi=300, bbox_inches='tight')
    plt.show()

draw_staggered_grid_refined()