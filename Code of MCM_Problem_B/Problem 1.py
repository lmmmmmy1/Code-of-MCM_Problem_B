import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 1. Basic Parameters (无变化) --------------------------
total_material = 100_000_000  
elevator_capacity = 537_000    
rocket_capacity = 195_000      
elevator_cost = 5823.79          
rocket_cost = 400000     

ratio_elevator = np.linspace(0, 1, 35)
x_data = ratio_elevator * 100
time_data = []
cost_data = []

for r in ratio_elevator:
    mat_e = total_material * r
    mat_r = total_material * (1 - r)
    t_e = mat_e / elevator_capacity if mat_e != 0 else 0
    t_r = mat_r / rocket_capacity if mat_r != 0 else 0
    time_data.append(max(t_e, t_r))
    cost = (mat_e * elevator_cost + mat_r * rocket_cost) / 1e12
    cost_data.append(cost)

x_array = np.array(x_data)
time_array = np.array(time_data)
cost_array = np.array(cost_data)

min_time_idx = np.argmin(time_array)
opt_ratio_time = x_array[min_time_idx]
min_total_time = time_array[min_time_idx]

# -------------------------- 2. Plot Configuration (无变化) --------------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(18, 10))

cost_cmap = LinearSegmentedColormap.from_list('CostGrad', [(0.2, 0.6, 0.9), (0.9, 0.3, 0.2)], N=256)

# -------------------------- 3. Draw 2.5D Bar Chart (无变化) --------------------------
bar_width = 2.9
depth = 1.5
alpha = 0.8

bars = ax.bar(
    x_array, time_array, width=bar_width,
    color=cost_cmap(cost_array / cost_array.max()),
    alpha=alpha, edgecolor='white', linewidth=1.2,
    label='Transport Plan (Height=Time, Color=Cost)'
)

ax.bar(
    x_array + depth, time_array, width=bar_width,
    color=cost_cmap(cost_array / cost_array.max()),
    alpha=alpha * 0.3, edgecolor='white', linewidth=0.8
)

# -------------------------- 4. Highlight Time-Optimal Combination (调整标注位置) --------------------------
ax.scatter(
    opt_ratio_time, min_total_time,
    s=300, c='#FF7F0E', edgecolor='black', linewidth=2.5, marker='D',
    label=f'Time-Optimal Combination ({opt_ratio_time:.0f}% Elevator)'
)
# 调整标注位置，避开右侧图例
ax.annotate(
    f'Time-Optimal Combination\nRatio: {opt_ratio_time:.0f}% Elevator + {(100-opt_ratio_time):.0f}% Rocket\nMin Time: {min_total_time:.0f} Years\nCost: {cost_array[min_time_idx]:.2f} Trillion USD',
    xy=(opt_ratio_time, min_total_time),
    xytext=(opt_ratio_time -18, min_total_time + 80),  # 上移更多，避开图例
    fontsize=11, fontweight='bold', color='#FF7F0E',
    bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF3CD', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#FF7F0E', lw=2.5)
)

ax.axvline(x=opt_ratio_time, color='#FF7F0E', linewidth=2, linestyle='--',
           label=f'Time Balance Point ({opt_ratio_time:.0f}% Ratio)')
ax.fill_betweenx([0, min_total_time*1.2], opt_ratio_time-2, opt_ratio_time+2,
                 color='#FF7F0E', alpha=0.1, label='Time-Optimal Zone')

# -------------------------- 5. Annotate Key Plans (调整纯电梯标注位置) --------------------------
pure_r_idx = np.argmin(x_array)
ax.annotate(
    f'Pure Rocket\nTime: {time_array[pure_r_idx]:.0f} Years\nCost: {cost_array[pure_r_idx]:.0f} Trillion USD',
    xy=(x_array[pure_r_idx], time_array[pure_r_idx]),
    xytext=(x_array[pure_r_idx]+ 5, time_array[pure_r_idx] + 20),
    fontsize=11, fontweight='bold', color='#E74C3C',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FADBD8', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2)
)

# 大幅调整纯电梯标注位置，彻底避开时间最优标注
pure_e_idx = np.argmax(x_array)
ax.annotate(
    f'Pure Space Elevator\nTime: {time_array[pure_e_idx]:.0f} Years\nCost: {cost_array[pure_e_idx]:.2f} Trillion USD',
    xy=(x_array[pure_e_idx], time_array[pure_e_idx]),
    xytext=(x_array[pure_e_idx] -10, time_array[pure_e_idx] + 10),  # 左移+下移更多
    fontsize=11, fontweight='bold', color='#3498DB',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#D6EAF8', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2)
)

# -------------------------- 6. Legend & Axes Settings (核心：图例移至右上角) --------------------------
sm = plt.cm.ScalarMappable(cmap=cost_cmap, norm=plt.Normalize(vmin=0, vmax=cost_array.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Transport Cost (Trillion USD)', fontsize=12, fontweight='bold', labelpad=15)

# 图例移至右上角，并通过bbox_to_anchor固定在角落，避免重叠
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=cost_cmap(0.5), alpha=alpha, edgecolor='white',
                  label='Transport Plan (Height=Time, Color=Cost)'),
    plt.Line2D([0], [0], color='#FF7F0E', linewidth=2, linestyle='--',
               label=f'Time Balance Point ({opt_ratio_time:.0f}% Ratio)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF7F0E', markersize=10,
               label=f'Time-Optimal Combination ({min_total_time:.0f} Years)')
]
ax.legend(handles=legend_elements, loc='upper right', 
          bbox_to_anchor=(0.99, 0.99),  # 固定在右上角角落
          fontsize=11, framealpha=0.9, edgecolor='lightgray')

ax.set_xlabel('Space Elevator Transport Ratio (%)', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Transport Time (Years)', fontsize=14, fontweight='bold', labelpad=15)
ax.set_title('Lunar Colony Transport Plan: Time-Optimal Combination Analysis (2.5D Chart)\n(Total Material: 100 Million Tons, Perfect Condition)',
             fontsize=16, fontweight='bold', pad=25, color='#333')

ax.set_xlim(-3, 108)
ax.set_ylim(0, np.max(time_array) * 1.15)
ax.set_xticks(np.arange(0, 110, 10))
ax.set_yticks(np.arange(0, np.max(time_array)+100, 100))
ax.grid(True, axis='y', linestyle='--', linewidth=0.8)

# -------------------------- 7. Save & Show (无变化) --------------------------
plt.tight_layout()
plt.savefig('lunar_colony_time_optimal_no_overlap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

# -------------------------- 8. Key Results Output (无变化) --------------------------
print("="*60)
print("Time-Optimal Combination Key Results:")
print("="*60)
print(f"1. Optimal Ratio: {opt_ratio_time:.0f}% Space Elevator + {(100-opt_ratio_time):.0f}% Rocket")
print(f"2. Minimum Transport Time: {min_total_time:.0f} Years")
print(f"3. Corresponding Cost: {cost_array[min_time_idx]:.2f} Trillion USD")
print(f"4. Advantages Over Pure Plans: Saves {time_array[pure_r_idx]-min_total_time:.0f} years vs Pure Rocket, {time_array[pure_e_idx]-min_total_time:.0f} years vs Pure Elevator")
print("="*60)
