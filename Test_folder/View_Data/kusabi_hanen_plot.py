import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Font Settings ---
plt.rcParams['font.family'] = 'sans-serif'

def draw_wedge_groove_fixed():
    """Image 1: Wedge Groove (English)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p = 10.0
    d = 4.0
    y_top_limit = d + 2.5
    
    vertices = [
        (-p, y_top_limit), (-p, d), (0, 0), (0, d),
        (p, 0), (p, d), (p + 3, d * (1 - 3/p)), (p + 3, y_top_limit)
    ]
    
    poly = patches.Polygon(vertices, closed=True, facecolor='#E0E0E0', edgecolor='none')
    ax.add_patch(poly)

    ax.plot([-p, 0], [d, 0], color='black', linewidth=2)
    ax.plot([0, 0], [0, d], color='black', linewidth=2)
    ax.plot([0, p], [d, 0], color='black', linewidth=2)
    ax.plot([p, p], [0, d], color='black', linewidth=2)
    ax.plot([p, p+3], [d, d * (1 - 3/p)], color='black', linewidth=2)

    # English Translations
    ax.text(-p, d + 3.5, "Wedge Groove", fontsize=22, fontweight='bold')
    ax.plot([-p, -p+8], [d + 3.3, d + 3.3], color='black', linewidth=2)
    
    ax.text(-p/2, d + 1.0, "Specimen", fontsize=20)
    # ax.text(-p, -1.5, "Bottom", fontsize=16)

    ax.annotate('', xy=(0.5, 0), xytext=(0.5, d), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(1.0, d/2, "Depth $d$", fontsize=16, va='center')
    ax.plot([0, 2], [d, d], 'k-', lw=0.5)
    ax.plot([0, 2], [0, 0], 'k-', lw=0.5)

    ax.annotate('', xy=(0, -2.5), xytext=(p, -2.5), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(p/2, -3.5, "Pitch $p$", fontsize=16, ha='center')
    ax.plot([0, 0], [0, -3], 'k-', lw=1)
    ax.plot([p, p], [0, -3], 'k-', lw=1)

    origin_x = p
    origin_y = 0
    r_alpha = 3.0
    theta_rad = np.arctan2(d, p)
    
    ax.plot([origin_x - r_alpha - 1.0, origin_x], [0, 0], color='steelblue', linewidth=1.5, linestyle='--')
    
    start_x = origin_x - r_alpha
    start_y = 0
    end_x = origin_x + r_alpha * np.cos(np.pi - theta_rad)
    end_y = origin_y + r_alpha * np.sin(np.pi - theta_rad)
    
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1.5, 
                                connectionstyle='arc3,rad=-0.2'))
    
    mid_angle = np.pi - (theta_rad / 2)
    text_dist = r_alpha + 0.6
    text_x = origin_x + text_dist * np.cos(mid_angle)
    text_y = origin_y + text_dist * np.sin(mid_angle)
    ax.text(text_x, text_y, r'$\alpha$', fontsize=16, color='black', va='center', ha='center')

    # Fix: Use raw string and \leq for safety
    # ax.annotate(r'Groove bottom R $\leq$ 0.03', xy=(p, d), xytext=(p-1, d+4),arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.5), fontsize=14)
    # circle = patches.Circle((p, d), radius=0.2, fill=False, color='black', lw=1)
    # ax.add_patch(circle)

    ax.set_xlim(-p-1, p+4)
    ax.set_ylim(-4, d+5)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def draw_rounded_profile_fixed():
    """Image 2: Rounded Groove (English)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    w = 4.0        
    d_val = w / 2  
    offset_p = 8.0 
    
    margin_x = 3.0
    x_min = -margin_x
    x_max = offset_p + w + margin_x
    y_max = d_val + 3.0
    y_base = 0.0

    theta = np.linspace(np.pi, 0, 50)
    
    x_flat1 = np.linspace(x_min, 0, 10)
    y_flat1 = np.full_like(x_flat1, y_base)
    x_arc1 = (w/2) + (w/2) * np.cos(theta)
    y_arc1 = y_base + (w/2) * np.sin(theta)
    
    x_flat2 = np.linspace(w, offset_p, 10)
    y_flat2 = np.full_like(x_flat2, y_base)
    
    x_arc2 = offset_p + (w/2) + (w/2) * np.cos(theta)
    y_arc2 = y_base + (w/2) * np.sin(theta)
    x_flat3 = np.linspace(offset_p + w, x_max, 10)
    y_flat3 = np.full_like(x_flat3, y_base)
    
    x_all = np.concatenate([x_flat1, x_arc1, x_flat2, x_arc2, x_flat3])
    y_all = np.concatenate([y_flat1, y_arc1, y_flat2, y_arc2, y_flat3])
    
    poly_x = np.concatenate([x_all, [x_max, x_min]])
    poly_y = np.concatenate([y_all, [y_max, y_max]])
    poly = patches.Polygon(np.column_stack((poly_x, poly_y)), closed=True, facecolor='#D0D0D0', edgecolor='none')
    ax.add_patch(poly)

    ax.plot(x_all, y_all, color='black', linewidth=2)

    # English Translations
    ax.text(x_min + 0.5, y_max - 1.0, "Specimen", fontsize=20, ha='left')
    # ax.text(x_min + 0.5, -1.0, "Bottom", fontsize=20, ha='left')

    w_arrow_y = -0.8 
    w_start = offset_p
    w_end = offset_p + w
    
    ax.annotate('', xy=(w_start, w_arrow_y), xytext=(w_end, w_arrow_y), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    ax.plot([w_start, w_start], [0, w_arrow_y - 0.3], 'k-', lw=1)
    ax.plot([w_end, w_end], [0, w_arrow_y - 0.3], 'k-', lw=1)
    ax.text(offset_p + w/2, w_arrow_y - 0.8, "Width $w$", fontsize=18, ha='center', va='top')

    arrow_y = -3.5 
    p_start = 0
    p_end = offset_p + w
    
    ax.annotate('', xy=(p_start, arrow_y), xytext=(p_end, arrow_y), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    ax.plot([p_start, p_start], [0, arrow_y - 0.5], 'k-', lw=1) 
    ax.plot([p_end, p_end], [0, arrow_y - 0.5], 'k-', lw=1)
    ax.text((p_start + p_end)/2, arrow_y - 0.8, "Pitch $p$", fontsize=18, ha='center', va='top')

    # Depth d - Positioned ABOVE the arrow
    top_y = d_val
    center_x = offset_p + w/2
    
    ax.annotate('', xy=(center_x, 0), xytext=(center_x, top_y), 
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    ax.plot([center_x - 0.5, center_x + 0.5], [0, 0], 'k-', lw=1)
    
    # Text position adjusted: y = top_y + margin, ha='center', va='bottom'
    ax.text(center_x, top_y + 0.2, "Depth $d$", fontsize=18, ha='center', va='bottom')

    ax.set_xlim(x_min, x_max + 2)
    ax.set_ylim(-5.5, y_max + 1.5) # Increase upper limit for text space
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_wedge_groove_fixed()
    draw_rounded_profile_fixed()