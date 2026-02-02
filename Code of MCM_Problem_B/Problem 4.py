import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D polygon drawing
import matplotlib.colors as mcolors

# ======================== 1. Core AHP and TOPSIS Functions (Unmodified) ========================
def ahp_calculate(decision_matrix, indicator_names):
    A = np.array(decision_matrix, dtype=np.float64)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Judgment matrix must be square!")
    eig_values, eig_vectors = np.linalg.eig(A)
    lambda_max = np.max(np.real(eig_values))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1:0.0, 2:0.0, 3:0.58, 4:0.89, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    ri = ri_table.get(n, 0.0)
    cr = ci / ri if ri != 0 else 0.0
    is_pass = cr < 0.1
    max_eig_vector = np.real(eig_vectors[:, np.argmax(np.real(eig_values))])
    weights = max_eig_vector / np.sum(max_eig_vector)
    print("="*50)
    print("[AHP Analytic Hierarchy Process Results]")
    print(f"Maximum eigenvalue of judgment matrix λ_max: {lambda_max:.6f}")
    print(f"Consistency Index (CI): {ci:.6f}")
    print(f"Consistency Ratio (CR): {cr:.6f} (CR < 0.1 = Pass test: {'Yes' if is_pass else 'No'})")
    print("Indicator weight distribution:")
    for idx, (name, w) in enumerate(zip(indicator_names, weights)):
        print(f"  {name}: {w:.6f}")
    print("="*50)
    return weights.tolist(), cr, is_pass

def topsis_calculate(original_data, weights, scheme_names, indicator_names, indicator_types):
    X = np.array(original_data, dtype=np.float64)
    n_schemes, n_indicators = X.shape
    # 1. Indicator direction unification (negative → positive)
    X_forward = X.copy()
    for j in range(n_indicators):
        if indicator_types[j] == 'negative':
            max_val = np.max(X_forward[:, j])
            X_forward[:, j] = max_val - X_forward[:, j]
    # 2. Vector normalization
    X_standard = X_forward / np.sqrt(np.sum(X_forward**2, axis=0))
    # 3. Weighted normalization
    X_weighted = X_standard * np.array(weights)
    # 4. Determine ideal solutions
    R_plus = np.max(X_weighted, axis=0)
    R_minus = np.min(X_weighted, axis=0)
    # 5. Calculate Euclidean distance
    D_plus = np.sqrt(np.sum((X_weighted - R_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((X_weighted - R_minus)**2, axis=1))
    # 6. Calculate closeness and ranking
    closeness = D_minus / (D_plus + D_minus)
    rank_indices = np.argsort(-closeness)
    rank = [0]*n_schemes
    for idx, pos in enumerate(rank_indices):
        rank[pos] = idx + 1
    print("\n" + "="*50)
    print("[TOPSIS Final Evaluation Results]")
    print("Scheme Ranking (Higher closeness = Better performance):")
    for i in range(n_schemes):
        print(f"  Rank {rank[i]}: {scheme_names[i]} → Closeness={closeness[i]:.6f}")
    print("="*50)
    return closeness.tolist(), rank, X_weighted, D_plus, D_minus

# ======================== 2. Core Visualization Functions (Only removed vertical line drawing) ========================
def plot_visualizations(weights, closeness, X_weighted, scheme_names, indicator_names):
    plt.rcParams['font.sans-serif'] = ['Arial']  # English font
    plt.rcParams['axes.unicode_minus'] = False    # Negative sign display
    fig = plt.figure(figsize=(20, 18))  # Canvas size adaptation

    # ---------------------- Subplot 1: Indicator Weight Donut Chart (Unmodified) ----------------------
    ax1 = plt.subplot(2, 1, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Indicator colors
    wedges, texts, autotexts = ax1.pie(weights, labels=indicator_names, autopct='%1.2f%%',
                                       colors=colors, startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('AHP Indicator Weight Distribution', fontsize=18, fontweight='bold', pad=20)

    # ---------------------- Subplot 2: Scheme-Indicator Weighted Value Heatmap (White = Optimal) ----------------------
    ax2 = plt.subplot(2, 2, 3)
    # Custom colormap: Red (Poor) → Yellow → White (Optimal)
    cmap = mcolors.LinearSegmentedColormap.from_list('RdYlW', ['#E8EDF1', '#7091C7', '#375093'])
    im = ax2.imshow(X_weighted, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(X_weighted))
    # Axis and value annotation
    ax2.set_xticks(np.arange(len(indicator_names)))
    ax2.set_yticks(np.arange(len(scheme_names)))
    ax2.set_xticklabels(indicator_names)
    ax2.set_yticklabels(scheme_names)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(scheme_names)):
        for j in range(len(indicator_names)):
            text = ax2.text(j, i, f'{X_weighted[i, j]:.4f}',
                           ha="center", va="center", color="black", fontsize=10)
    ax2.set_title('Scheme-Indicator Weighted Normalized Value Heatmap (BLUE=Optimal)', fontsize=16, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Weighted Value', rotation=270, labelpad=15)

    # ---------------------- Subplot 3: 3D Line Chart (Core Modification: Removed vertical lines) ----------------------
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    n_schemes = len(scheme_names)
    n_indicators = len(indicator_names)
    # Basic definition: X=Indicators, Y=Schemes, transpose weighted matrix
    x = np.arange(n_indicators)  # X-axis: Indicators (0-3)
    X_weighted_T = X_weighted.T  # (n_indicators, n_schemes)
    # Scheme colors + markers (5 schemes)
    color_list = ['#7A5799', '#576799', '#4787C8', '#57A899', '#4AB86A']
    markers = ['o', 'o', 'o', 'o', 'o']
    # Key indicator indices (Order: Carbon Emission/Noise Pollution/Habitat Loss/Air Pollution)
    carbon_idx = 0         # Carbon Emission
    noise_idx = 1          # Noise Pollution
    habitat_idx = 2        # Habitat Loss
    air_pollution_idx = 3  # Air Pollution

    # Step 1: Draw 3D lines for each scheme (Unmodified, black annotations)
    for i in range(n_schemes):
        z = X_weighted_T[:, i]  # Weighted values of current scheme's indicators
        # Draw 3D line
        ax3.plot(x, [i]*n_indicators, z, color=color_list[i], linewidth=2,
                 marker=markers[i], markersize=3, markerfacecolor=color_list[i],
                 label=scheme_names[i])
        # Annotate line points (black text)
        for j in range(n_indicators):
            if z[j] > 0.01:
                ax3.text(x[j], i, z[j]+0.005, f'{z[j]:.4f}',
                        ha='center', va='bottom', fontsize=8, color='black')

    # Step 2: Draw borderless polygons for each scheme (Removed vertical line drawing)
    for i in range(n_schemes):
        # 1. Get Z values (weighted values) of 4 indicators for current scheme
        z_carbon = X_weighted_T[carbon_idx, i]       # Carbon Emission Z value
        z_noise = X_weighted_T[noise_idx, i]         # Noise Pollution Z value
        z_habitat = X_weighted_T[habitat_idx, i]     # Habitat Loss Z value
        z_air = X_weighted_T[air_pollution_idx, i]   # Air Pollution Z value

        # -------- Core Modification: Removed vertical line drawing code --------

        # 3. Construct 3D polygon vertices (Unmodified)
        poly_vertices = [
            [carbon_idx, i, z_carbon],          # 1. Carbon Emission point
            [noise_idx, i, z_noise],            # 2. Noise Pollution point
            [habitat_idx, i, z_habitat],        # 3. Habitat Loss point
            [air_pollution_idx, i, z_air],      # 4. Air Pollution point
            [air_pollution_idx, i, 0],          # 5. Air Pollution vertical bottom
            [carbon_idx, i, 0],                 # 6. Carbon Emission vertical bottom
            [carbon_idx, i, z_carbon]           # 7. Close: Back to Carbon Emission point
        ]

        # 4. Draw 3D polygon (Unmodified, borderless + 75% transparency)
        poly = Poly3DCollection([poly_vertices], alpha=0.2)
        poly.set_facecolor(color_list[i])
        poly.set_edgecolor('none')
        poly.set_linewidth(0)
        ax3.add_collection3d(poly)

    # Axis + View + Legend settings (Unmodified)
    ax3.set_xticks(x)
    ax3.set_xticklabels(indicator_names, rotation=45, ha='right', fontsize=9)  # X-axis: Indicators
    ax3.set_yticks(np.arange(n_schemes))
    ax3.set_yticklabels(scheme_names, fontsize=8)  # Y-axis: Schemes
    ax3.set_zlabel('Weighted Normalized Value', fontsize=12, labelpad=10)
    ax3.set_zlim(0, np.max(X_weighted)+0.05)
    ax3.view_init(elev=30, azim=-35)  # Optimize view to avoid occlusion
    ax3.set_title('Scheme-Indicator Weighted Value 3D Line Chart ', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(loc='upper left', fontsize=7)

    # Adjust subplot spacing and save high-resolution image
    plt.tight_layout(pad=6.0)
    plt.savefig('AHP_TOPSIS_Visualization_No_Vertical_Lines.png', dpi=300, bbox_inches='tight')
    plt.show()

# ======================== 3. Main Program (Unmodified) ========================
if __name__ == "__main__":
    # 1. Basic information definition (Schemes/Indicators/Judgment Matrix)
    scheme_names = [
        "CER",               # Clean Energy Rocket
        "FER",               # Fossil Energy Rocket
        "CESE",              # Clean Energy Space Elevator
        "FESE",              # Fossil Energy Space Elevator
        "CER+CESE Combo"     # Clean Energy Rocket + Elevator Combo
    ]
    indicator_names = ["Carbon Emission", "Noise Pollution", "Habitat Loss", "Air Pollution"]
    indicator_types = ["negative", "negative", "negative", "negative"]  # All negative indicators
    ahp_matrix_input = [  # Specified judgment matrix
        [1, 0.4, 0.8, 0.6],
        [2.5, 1, 2, 1.5],
        [1.25, 0.5, 1, 0.75],
        [5/3, 2/3, 4/3, 1]
    ]
    # 2. Original indicator data (Replace with actual data as needed)
    original_data = [
        [0, 100450, 28000, 0],        # CER
        [1572900, 100450, 28000, 4200],# FER
        [0, 5012000, 1500, 0],         # CESE
        [4601522, 12000, 1500, 6421],  # FESE
        [0, 34997, 29500, 0]           # CER+CESE Combo
    ]
    # 3. Run AHP+TOPSIS calculation
    weights, cr, is_pass = ahp_calculate(ahp_matrix_input, indicator_names)
    closeness, rank, X_weighted, D_plus, D_minus = topsis_calculate(
        original_data, weights, scheme_names, indicator_names, indicator_types
    )
    # 4. Generate visualization charts
    plot_visualizations(weights, closeness, X_weighted, scheme_names, indicator_names)