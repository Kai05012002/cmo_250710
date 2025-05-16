import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load data
files = {
    'DRQN': 'xml/DRQN資料.txt',
    'FeUdal': 'xml/FeUdal資料.txt',
    'GAT': 'xml/GAT資料.txt',
    'GBRL': 'xml/GBRL資料.txt'
}

data = {}
for alg, fn in files.items():
    with open(fn, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # 所有資料*1000
    data[alg] = {
        'RL_only': np.array(obj['step_times_rl'])*1000,
        'RL+CMO': np.array(obj['step_times_cmo'])*1000
    }

# Compute means and stds
algorithms = list(files.keys())
rl_means = np.array([data[alg]['RL_only'].mean() for alg in algorithms])
rl_stds = np.array([data[alg]['RL_only'].std(ddof=0) for alg in algorithms])
cmo_means = np.array([data[alg]['RL+CMO'].mean() for alg in algorithms])
cmo_stds = np.array([data[alg]['RL+CMO'].std(ddof=0) for alg in algorithms])

# Positions for boxes
positions_rl = [i * 3 + 1 for i in range(len(algorithms))]
positions_cmo = [i * 3 + 2 for i in range(len(algorithms))]
width = 0.6

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Draw boxes and mean lines for RL_only
for pos, mean, std in zip(positions_rl, rl_means, rl_stds):
    # rectangle for ±std
    rect = Rectangle((pos - width/2, mean - std), width, 2*std,
                     fill=False, edgecolor='black', linewidth=1.5, zorder=2)
    ax1.add_patch(rect)
    # mean line
    ax1.hlines(mean, pos - width/2, pos + width/2, colors='red', linewidth=2, zorder=3)

# Draw boxes and mean lines for RL+CMO
for pos, mean, std in zip(positions_cmo, cmo_means, cmo_stds):
    rect = Rectangle((pos - width/2, mean - std), width, 2*std,
                     fill=False, edgecolor='black', linewidth=1.5, zorder=2)
    ax2.add_patch(rect)
    ax2.hlines(mean, pos - width/2, pos + width/2, colors='blue', linewidth=2, zorder=3)

# Labels and ticks
ax1.set_ylabel('RL_only Step Time (ms)', color='red') # 顯示紅色
ax2.set_ylabel('RL+CMO Step Time (ms)', color='blue') # 顯示藍色
ax1.set_xticks([i * 3 + 1.5 for i in range(len(algorithms))])
ax1.set_xticklabels(algorithms)
ax1.set_title('Mean ± Std Boxes with Visible Mean Lines')

plt.tight_layout()
plt.show()
