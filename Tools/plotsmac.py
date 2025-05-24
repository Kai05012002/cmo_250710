import json
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# 讀取多個 JSON 檔案
def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(file_path) as f:
            data = json.load(f)
            # 處理特殊的數據格式（numpy.float64 對象）
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, list):
                    # 檢查是否包含字典格式的數值
                    if value and isinstance(value[0], dict) and 'value' in value[0]:
                        processed_data[key] = [item['value'] for item in value]
                    else:
                        processed_data[key] = value
                else:
                    processed_data[key] = value
            all_data.append(processed_data)
    return all_data

def smooth(y, alpha=0.1, sigma=2):
    def exp_moving_average(y):
        ema = np.zeros_like(y)
        ema[0] = y[0]
        for i in range(1, len(y)):
            ema[i] = alpha * y[i] + (1 - alpha) * ema[i-1]
        return ema
    
    return gaussian_filter1d(exp_moving_average(y), sigma)

# 定義要繪製的數據
def plot_data(data_list, keys, name_list, battle_name, smooth_window=2):
    # 計算需要的窗口數量
    num_keys = len(keys)
    num_windows = (num_keys + 3) // 4  # 向上取整，每個窗口最多4個圖表
    
    for window_idx in range(num_windows):
        # 創建一個包含多個子圖的窗口
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(battle_name, fontsize=16)
        
        # 確定當前窗口要顯示的keys
        start_idx = window_idx * 4
        end_idx = min(start_idx + 4, num_keys)
        window_keys = keys[start_idx:end_idx]
        
        # 計算行列數
        num_plots = len(window_keys)
        rows = min(2, num_plots)
        cols = min(2, (num_plots + 1) // 2)
        
        for i, key in enumerate(window_keys):
            # 創建子圖
            ax = fig.add_subplot(rows, cols, i + 1)
            # 設置坐標軸背景顏色
            ax.set_facecolor('lightyellow')
            # 設置網格顏色
            ax.grid(color='green', linestyle='--', linewidth=0.5)
            
            for data, name in zip(data_list, name_list):
                if key in data:
                    x = data[key + '_T']
                    y = data[key]
                    y = smooth(y)
                    ax.plot(x, y, label=name)
                    #ax.plot(data[key + '_T'], data[key], label=name) 
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()
        
        # 調整子圖之間的間距
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 為頂部標題留出空間

    plt.show()

def create_dynamic_window(data_list, keys, name_list, battle_name):
    root = tk.Tk()
    root.title("動態選擇要顯示的數據")
    
    # 創建左側的選擇面板
    select_frame = ttk.Frame(root)
    select_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
    
    # 創建變量來存儲選擇狀態
    vars = []
    
    def update_plot(*args):
        # 獲取選中的數據
        selected_names = [name for name, var in zip(name_list, vars) if var.get()]
        filtered_data = [data for data, name in zip(data_list, name_list) if name in selected_names]
        filtered_names = [name for name in name_list if name in selected_names]
        
        # 清除所有現有的圖表
        plt.close('all')
        # 重新繪製圖表
        if filtered_data:  # 確保至少選擇了一個數據
            plot_data(filtered_data, keys, filtered_names, battle_name)
    
    # 創建複選框
    for name in name_list:
        var = tk.BooleanVar(value=True)  # 默認全選
        var.trace('w', update_plot)  # 添加跟踪器，當值改變時更新圖表
        vars.append(var)
        cb = ttk.Checkbutton(select_frame, text=name, variable=var)
        cb.pack(anchor='w', padx=5, pady=2)
    
    # 添加全選/取消全選按鈕
    def select_all():
        for var in vars:
            var.set(True)
    
    def deselect_all():
        for var in vars:
            var.set(False)
    
    ttk.Button(select_frame, text="全選", command=select_all).pack(pady=5)
    ttk.Button(select_frame, text="取消全選", command=deselect_all).pack(pady=5)
    
    # 初始繪圖
    update_plot()
    
    root.mainloop()

# 使用者選擇要繪製的數據
file_paths = [
            #   'logs/run_20250327_173035/stats.json',
            #   'logs/run_20250330_235037_CPU_4X/stats.json',
            #   'logs/run_20250402_013318_GPU_5X/stats.json',
            #   'logs/run_20250402_053653/stats.json',
            #   'logs/run_20250409_043516/stats.json',
            #   'logs/run_20250414_052243/stats.json',
            #   'logs/run_20250416_031716/stats.json',
            #   'logs/run_20250428_020031/stats.json',
            #   'logs/run_20250430_015530/stats.json',
            #   'logs/run_20250430_051852/stats.json',
            #   'logs/run_20250430_053312/stats.json',
            #   'logs/run_20250504_235008/stats.json',
            #   'logs/run_20250504_190944/stats.json',
            #   'logs/run_20250504_221035/stats.json',
            'Logs/DQN_I5_CPU/stats.json',
            'Logs/DQN_R9_GPU/stats.json',
            'Logs/DQN_R9_GPU_batch/stats.json',
              ]

data_list = load_data(file_paths)
# selected_keys = ['distance','episode_return','loss','best_distance','worker_loss','manager_loss','critic_loss']
selected_keys = ['distance', 'loss', 'return']
name_list = [ 
                # "NO_Reset",
                # "CPU_4X",
                # "GPU_5X",
                # "RAMDISK_5X",
                # "Feudal_5X",
                # "Feudal_5X+singleEpisode_beta",
                # "Feudal_5X+singleEpisode_beta_2",
                # "RDQN+singleEpisode",
                # "MyNet+singleEpisode",
                # "MyNet+singleEpisode_1",
                # "MyNet+singleEpisode_2",
                # "bayesian",
                # "FeUdal",
                # "DRQN",
                "DQN_I5_CPU",
                "DQN_R9_GPU",
                "DQN_R9_GPU_batch",
            ]

battle_name = 'Model_1'

# 使用新的動態選擇窗口
create_dynamic_window(data_list, selected_keys, name_list, battle_name)
