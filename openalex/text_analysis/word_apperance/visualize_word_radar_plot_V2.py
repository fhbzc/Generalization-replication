import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import sys
# utils.py lives two levels up (openalex/), not in this word_apperance/ directory;
# add it to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

plt.rcParams['font.family'] = 'Gill Sans MT'

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'text_appear_gen_foundation_ext')
figure_dire = os.path.join('./figures/', 'text_radar_plot')
os.makedirs(figure_dire, exist_ok=True)

reference_threshold = 1
citation_threshold = 5
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)

df_res = pd.read_excel(
    os.path.join(
        load_dire,
        f'manual_evaluation_word_c_t_{citation_threshold}_r_t_{reference_threshold}_V3.xlsx'
    )
)

# ---------- 自定义：把 r 映射到等间距环的坐标 ----------
BREAKS = np.array([0.0, 0.7, 1.0, 1.4, 2], dtype=float)

def equalize_radius(r, breaks=BREAKS):
    """
    将原始半径 r 映射到等环距坐标 r'∈[0, len(breaks)-1]。
    每个区间 [b_i, b_{i+1}] 映射为长度 1，区间内线性。
    超出范围的 r 会被裁剪到 [breaks[0], breaks[-1]]。
    """
    r = float(r)
    if r <= breaks[0]:
        return 0.0
    if r >= breaks[-1]:
        return float(len(breaks) - 1)
    i = np.searchsorted(breaks, r) - 1
    left, right = breaks[i], breaks[i+1]
    return i + (r - left) / (right - left)

# ---------- 绘图数据 ----------
labels = ['F', 'E', 'G']
gen_type_list = ['ratio']

for gen_type in tqdm(gen_type_list):
    if gen_type == 'ratio':
        word2foundation = dict(zip(df_res['word'], df_res['f_y5_word_ratio']))
        word2extension = dict(zip(df_res['word'], df_res['e_y5_word_ratio']))
        word2generalization = dict(zip(df_res['word'], df_res['g_y5_word_ratio']))
    else:
        raise NotImplementedError

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    key_word_list = [
        'device', 'tool', 'software',
        'review', 'guideline', 'tutorial',
        'discovery', 'dataset', 'observation',
        'theory', 'metric', 'hypothesis'
    ]

    main_list = []
    for key_word in key_word_list:
        vals = [word2foundation[key_word], word2extension[key_word], word2generalization[key_word]]
        vals += vals[:1]
        main_list.append(vals)

    # ---------- 画图 ----------
    fig, axes = plt.subplots(4, 3, subplot_kw=dict(polar=True), figsize=(14, 14))
    axes = axes.flatten()
    groups = key_word_list
    shared_color = '#1a4f8a'  # 深蓝色

    # 等间距环坐标范围：0..4
    pos_ticks = np.arange(len(BREAKS))       # 0,1,2,3,4
    ticks_to_show = pos_ticks[1:]            # 不显示 r=0
    tick_labels   = [f"r={g}" for g in ["0.7", "1", "1.4", "2"]]

    label_offset = 0.6
    r_ring_max   = pos_ticks[-1]             # 外圈半径（=4）
    r_max_for_labels = r_ring_max + label_offset

    for idx, (ax, title, main) in enumerate(zip(axes, groups, main_list)):
        is_first = (idx == 0)

        # --- 基本设置 ---
        ax.spines['polar'].set_visible(False)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([''] * len(labels))
        ax.set_ylim(0, r_max_for_labels)

        # --- 圆环网格：r 刻度只在第一个图显示 ---
        ax.set_yticks(ticks_to_show)
        if is_first:
            ax.set_yticklabels(tick_labels, fontsize=12)
        else:
            ax.set_yticklabels([''] * len(ticks_to_show))
        ax.yaxis.grid(True, color='gray', linewidth=0.5)

        # 把 r=1 的那圈加粗、红色
        theta = np.linspace(0, 2*np.pi, 512)
        ax.plot(theta, np.full_like(theta, 2.0), color='red', linewidth=2, zorder=5)

        # 把 r=1 的刻度标签设为红色加粗（仅第一个图）
        if is_first:
            for lbl in ax.get_yticklabels():
                if lbl.get_text() == "r=1":
                    lbl.set_color("red")
                    lbl.set_fontweight("bold")

        # --- 辐条：不要超过外圈 ---
        ax.xaxis.grid(False)
        for a in angles[:-1]:
            ax.plot([a, a], [0, r_ring_max], color='gray', linewidth=0.8, alpha=0.7, zorder=1)

        # 外侧放 F/E/G 字母：只在第一个图标注
        if is_first:
            for angle, lab in zip(angles[:-1], labels):
                ax.text(angle, r_max_for_labels, lab, ha='center', va='center',
                        fontsize=18, color='black')

        # ---- 数据：等环距变换后再画 ----
        main_eq = [equalize_radius(v) for v in main]
        ax.plot(angles, main_eq, color=shared_color, linewidth=2, zorder=4)
        ax.fill(angles, main_eq, color=shared_color, alpha=0.65, zorder=3)

        # ---- 判断中心是否被 shade 实质性覆盖 ----
        # 将极坐标顶点转换为笛卡尔坐标（含 theta_offset=pi/2, direction=-1）
        display_angles = [np.pi / 2 - a for a in angles[:-1]]
        cart_pts = np.array([
            (r * np.cos(da), r * np.sin(da))
            for r, da in zip(main_eq[:-1], display_angles)
        ])
        # 计算原点到多边形每条边的有符号距离（正值=在内侧）
        # 有符号距离最小值越大，说明中心越深入 shade 内部
        n_verts = len(cart_pts)
        min_signed_dist = float('inf')
        for i in range(n_verts):
            a = cart_pts[i]
            b = cart_pts[(i + 1) % n_verts]
            ab = b - a
            length = np.linalg.norm(ab)
            if length < 1e-9:
                continue
            # 有符号距离：正值表示原点在边 a→b 的左侧（即多边形内侧）
            signed_d = (ab[0] * (-a[1]) - ab[1] * (-a[0])) / length
            min_signed_dist = min(min_signed_dist, signed_d)
        # 阈值：0.5（数据单位）—— 中心够深才用白字，否则黑字
        text_color = 'white' if min_signed_dist > 0.5 else 'black'

        # ---- 词语放在 radar 中央 ----
        ax.text(
            0, 0,
            title.capitalize(),
            ha='center', va='center',
            fontsize=22, fontweight='bold',
            color=text_color,
            transform=ax.transData,
            zorder=6
        )

    plt.subplots_adjust(wspace=-0.3, hspace=0.0, top=0.9)

    # at rt=1 / ct=5 this IS Fig. 2d, so the figure and the data table behind it are named
    # after the panel; every other threshold pair keeps the descriptive name.
    if reference_threshold == 1 and citation_threshold == 5:
        out_path = os.path.join(figure_dire, 'Fig. 2d.svg')
        csv_path = os.path.join(figure_dire, 'Fig. 2d.csv')
    else:
        out_path = os.path.join(
            figure_dire,
            f'type_{gen_type}_radar_plot_text_gen_exten_foundation_c_t_{citation_threshold}_r_t_{reference_threshold}_equalrings_V2.svg'
        )
        csv_path = os.path.join(
            figure_dire,
            f'type_{gen_type}_radar_plot_text_gen_exten_foundation_c_t_{citation_threshold}_r_t_{reference_threshold}_equalrings_V2.csv'
        )
    plt.savefig(out_path, format='svg', transparent=True)
    plt.close()

    pd.DataFrame({
        'word': key_word_list,
        'foundation': [word2foundation[w] for w in key_word_list],
        'extension': [word2extension[w] for w in key_word_list],
        'generalization': [word2generalization[w] for w in key_word_list],
    }).to_csv(csv_path, index=False)
    print(f"[CSV saved] {csv_path}")

print("finished")
