import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from JSON files
files = {
    'DQN': 'xml/7V7/DQN資料.txt',
    'DRQN': 'xml/7V7/DRQN資料.txt',
    'FeUdal': 'xml/7V7/FeUdal資料.txt',
    'GBRL': 'xml/7V7/GBRL資料.txt',
    # 'DQN_maple': 'xml/7V7/DQN資料_maple.txt'
}

data = {}
for alg, fn in files.items():
    with open(fn, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # 將列表轉換為 NumPy 數組
    step_times_rl = np.array(obj['step_times_rl'])
    step_times_cmo = np.array(obj['step_times_cmo'])
    
    # 計算5%分位數
    threshold_rl = np.percentile(step_times_rl, 80)
    threshold_cmo = np.percentile(step_times_cmo, 95)
    
    # 剔除前5%的值
    step_times_rl = step_times_rl[step_times_rl <= threshold_rl]
    step_times_cmo = step_times_cmo[step_times_cmo <= threshold_cmo]
    
    # 所有資料*1000
    data[alg] = {
        'RL_only': step_times_rl * 1000,
        'RL+CMO': step_times_cmo * 1000
    }

# 新增：分別繪製 RL_only 和 RL+CMO 的獨立箱形圖
algorithms = list(files.keys())
positions = [i + 1 for i in range(len(algorithms))]

# Plot RL_only boxplot
fig1, ax1 = plt.subplots(figsize=(8, 5))
rl_data = [data[alg]['RL_only'] for alg in algorithms]
ax1.boxplot(rl_data, positions=positions, widths=0.6, showfliers=False, boxprops=dict(color='red'))
ax1.set_ylim(0, max(max(arr) for arr in rl_data) * 1.1)
ax1.set_ylabel('RL_only Step Time (ms)', color='red')
ax1.set_xticks(positions)
ax1.set_xticklabels(algorithms)
ax1.set_title('RL_only Step Times 7V7 80%')

# Plot RL+CMO boxplot
fig2, ax2 = plt.subplots(figsize=(8, 5))
cmo_data = [data[alg]['RL+CMO'] for alg in algorithms]
ax2.boxplot(cmo_data, positions=positions, widths=0.6, showfliers=False, boxprops=dict(color='blue'))
ax2.set_ylim(0, max(max(arr) for arr in cmo_data) * 1.1)
ax2.set_ylabel('RL+CMO Step Time (ms)', color='blue')
ax2.set_xticks(positions)
ax2.set_xticklabels(algorithms)
ax2.set_title('RL+CMO Step Times 7V7')

# 輸出平均值和標準差
for alg in algorithms:
    print(f"{alg}:")
    print(f"  RL_only: {np.mean(data[alg]['RL_only']):.4f} ± {np.std(data[alg]['RL_only']):.4f}")
    print(f"  RL+CMO: {np.mean(data[alg]['RL+CMO']):.4f} ± {np.std(data[alg]['RL+CMO']):.4f}")

plt.tight_layout()
plt.show()
