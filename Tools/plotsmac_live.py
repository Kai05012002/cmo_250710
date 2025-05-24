import json
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
import time
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 讀取多個 JSON 檔案
def load_data(file_paths):
    all_data = []
    file_names = []
    
    for file_path in file_paths:
        try:
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
                file_names.append(os.path.basename(os.path.dirname(file_path)))
        except Exception as e:
            print(f"無法載入 {file_path}: {e}")
    
    return all_data, file_names

def smooth(y, alpha=0.1, sigma=1):
    def exp_moving_average(y):
        ema = np.zeros_like(y)
        ema[0] = y[0]
        for i in range(1, len(y)):
            ema[i] = alpha * y[i] + (1 - alpha) * ema[i-1]
        return ema
    
    return gaussian_filter1d(exp_moving_average(y), sigma)

class LivePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("即時監控訓練指標")
        self.root.geometry("1200x800")
        
        # 初始化數據
        self.data_list = []
        self.name_list = []
        self.keys = []
        self.battle_name = "模型訓練監控"
        self.file_paths = []
        self.update_interval = 5  # 默認更新間隔（秒）
        self.watching_dirs = []
        
        # 創建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 創建左側控制面板
        self.control_frame = ttk.Frame(self.main_frame, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 創建右側圖表顯示區域
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 設置控制元素
        self.setup_controls()
        
        # 創建圖表區域
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化選擇的指標變量
        self.metric_vars = []
        self.experiment_vars = []
        
        # 更新定時器
        self.update_timer = None
        self.is_updating = False
    
    def setup_controls(self):
        # 標題
        ttk.Label(self.control_frame, text="控制面板", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 選擇資料夾按鈕
        ttk.Button(self.control_frame, text="選擇監控資料夾", command=self.select_directories).pack(pady=5, fill=tk.X)
        
        # 選擇檔案按鈕
        ttk.Button(self.control_frame, text="選擇檔案", command=self.select_files).pack(pady=5, fill=tk.X)
        
        # 更新間隔設置
        interval_frame = ttk.Frame(self.control_frame)
        interval_frame.pack(pady=5, fill=tk.X)
        ttk.Label(interval_frame, text="更新間隔(秒):").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="5")
        interval_entry = ttk.Entry(interval_frame, textvariable=self.interval_var, width=5)
        interval_entry.pack(side=tk.LEFT, padx=5)
        
        # 監控標題設置
        title_frame = ttk.Frame(self.control_frame)
        title_frame.pack(pady=5, fill=tk.X)
        ttk.Label(title_frame, text="監控標題:").pack(side=tk.LEFT)
        self.title_var = tk.StringVar(value="模型訓練監控")
        title_entry = ttk.Entry(title_frame, textvariable=self.title_var)
        title_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 創建分隔線
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 創建實驗選擇框架
        self.experiment_frame = ttk.LabelFrame(self.control_frame, text="實驗選擇")
        self.experiment_frame.pack(pady=5, fill=tk.X)
        
        # 創建指標選擇框架
        self.metrics_frame = ttk.LabelFrame(self.control_frame, text="指標選擇")
        self.metrics_frame.pack(pady=5, fill=tk.X)
        
        # 創建按鈕框架
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        # 更新和停止按鈕
        self.start_button = ttk.Button(button_frame, text="開始監控", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.stop_button = ttk.Button(button_frame, text="停止監控", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 手動更新按鈕
        ttk.Button(self.control_frame, text="手動更新", command=self.update_now).pack(pady=5, fill=tk.X)
        
        # 指標全選/取消全選按鈕
        select_frame = ttk.Frame(self.control_frame)
        select_frame.pack(pady=5, fill=tk.X)
        ttk.Button(select_frame, text="全選指標", command=self.select_all_metrics).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(select_frame, text="取消全選", command=self.deselect_all_metrics).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 實驗全選/取消全選按鈕
        exp_select_frame = ttk.Frame(self.control_frame)
        exp_select_frame.pack(pady=5, fill=tk.X)
        ttk.Button(exp_select_frame, text="全選實驗", command=self.select_all_experiments).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(exp_select_frame, text="取消全選", command=self.deselect_all_experiments).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 狀態標籤
        self.status_var = tk.StringVar(value="就緒")
        status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        status_label.pack(pady=10)
    
    def select_directories(self):
        dirs = filedialog.askdirectory(title="選擇要監控的logs資料夾")
        if dirs:
            self.watching_dirs = [dirs]
            self.scan_directories()
    
    def select_files(self):
        files = filedialog.askopenfilenames(title="選擇JSON檔案", filetypes=[("JSON files", "*.json")])
        if files:
            self.file_paths = list(files)
            self.load_files()
    
    def scan_directories(self):
        self.status_var.set("掃描資料夾中...")
        self.root.update_idletasks()
        
        # 清理之前的選擇
        self.file_paths = []
        
        # 掃描選擇的目錄中的所有stats.json檔案
        for dir_path in self.watching_dirs:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file == 'stats.json':
                        self.file_paths.append(os.path.join(root, file))
        
        if self.file_paths:
            self.load_files()
            self.status_var.set(f"已找到 {len(self.file_paths)} 個檔案")
        else:
            self.status_var.set("未找到任何stats.json檔案")
    
    def load_files(self):
        self.status_var.set("讀取檔案中...")
        self.root.update_idletasks()
        
        self.data_list, self.name_list = load_data(self.file_paths)
        
        if not self.data_list:
            self.status_var.set("無法讀取任何檔案")
            return
        
        # 收集所有可能的指標鍵
        all_keys = set()
        for data in self.data_list:
            for key in data.keys():
                if key.endswith('_T'):
                    continue
                if isinstance(data[key], list) and key + '_T' in data:
                    all_keys.add(key)
        
        self.keys = sorted(list(all_keys))
        
        # 更新實驗選擇區
        self.update_experiment_selection()
        
        # 更新指標選擇區
        self.update_metric_selection()
        
        self.status_var.set(f"已載入 {len(self.data_list)} 個檔案，找到 {len(self.keys)} 個指標")
    
    def update_experiment_selection(self):
        # 清理現有的選擇框
        for widget in self.experiment_frame.winfo_children():
            widget.destroy()
        
        # 重置變量
        self.experiment_vars = []
        
        # 判斷是否需要滾動
        if len(self.name_list) > 10:
            # 創建滾動條和框架
            canvas = tk.Canvas(self.experiment_frame, height=200)
            scrollbar = ttk.Scrollbar(self.experiment_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            target_frame = scrollable_frame
        else:
            target_frame = self.experiment_frame
        
        # 創建新的選擇框
        for i, name in enumerate(self.name_list):
            var = tk.BooleanVar(value=True)
            self.experiment_vars.append(var)
            cb = ttk.Checkbutton(target_frame, text=name, variable=var)
            cb.pack(anchor='w', padx=5, pady=2)
    
    def update_metric_selection(self):
        # 清理現有的選擇框
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # 創建標籤和滾動框架
        if len(self.keys) > 10:
            # 創建滾動條和框架
            canvas = tk.Canvas(self.metrics_frame, height=200)
            scrollbar = ttk.Scrollbar(self.metrics_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            target_frame = scrollable_frame
        else:
            target_frame = self.metrics_frame
        
        # 重置變量
        self.metric_vars = []
        
        # 創建新的選擇框
        default_metrics = ['episode_return', 'loss', 'distance', 'best_distance']
        for key in self.keys:
            var = tk.BooleanVar(value=key in default_metrics)
            self.metric_vars.append((key, var))
            cb = ttk.Checkbutton(target_frame, text=key, variable=var)
            cb.pack(anchor='w', padx=5, pady=2)
    
    def select_all_metrics(self):
        for _, var in self.metric_vars:
            var.set(True)
    
    def deselect_all_metrics(self):
        for _, var in self.metric_vars:
            var.set(False)
    
    def select_all_experiments(self):
        for var in self.experiment_vars:
            var.set(True)
    
    def deselect_all_experiments(self):
        for var in self.experiment_vars:
            var.set(False)
    
    def start_monitoring(self):
        if not self.data_list or not self.keys:
            self.status_var.set("沒有資料可監控")
            return
        
        try:
            self.update_interval = int(self.interval_var.get())
        except ValueError:
            self.status_var.set("更新間隔必須是整數")
            return
        
        self.battle_name = self.title_var.get()
        
        self.is_updating = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.status_var.set(f"監控中...每 {self.update_interval} 秒更新一次")
        self.update_plot()
        
        # 設置定時更新
        self.schedule_update()
    
    def stop_monitoring(self):
        self.is_updating = False
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_var.set("監控已停止")
    
    def schedule_update(self):
        if self.is_updating:
            self.update_timer = self.root.after(self.update_interval * 1000, self.update_handler)
    
    def update_handler(self):
        if self.is_updating:
            self.update_now()
            self.schedule_update()
    
    def update_now(self):
        # 重新載入數據
        self.status_var.set("更新數據中...")
        self.root.update_idletasks()
        
        # 重新載入同一批文件
        self.data_list, self.name_list = load_data(self.file_paths)
        
        # 更新圖表
        self.update_plot()
        
        self.status_var.set(f"最後更新: {time.strftime('%H:%M:%S')}")
    
    def update_plot(self):
        # 獲取選中的數據和指標
        selected_indices = [i for i, var in enumerate(self.experiment_vars) if var.get()]
        selected_metrics = [key for key, var in self.metric_vars if var.get()]
        
        if not selected_indices or not selected_metrics:
            self.status_var.set("請選擇至少一個實驗和一個指標")
            return
        
        filtered_data = [self.data_list[i] for i in selected_indices]
        filtered_names = [self.name_list[i] for i in selected_indices]
        
        # 清除圖表
        self.fig.clear()
        
        # 計算子圖排列
        num_plots = len(selected_metrics)
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        
        self.fig.suptitle(self.battle_name, fontsize=16)
        
        for i, key in enumerate(selected_metrics):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            
            # 設置坐標軸背景顏色
            ax.set_facecolor('lightyellow')
            # 設置網格顏色
            ax.grid(color='green', linestyle='--', linewidth=0.5)
            
            for data, name in zip(filtered_data, filtered_names):
                if key in data and key + '_T' in data:
                    x = data[key + '_T']
                    y = data[key]
                    if len(y) > 3:  # 只有在有足夠的資料點時才進行平滑
                        y = smooth(y)
                    ax.plot(x, y, label=name)
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(key)
            ax.set_title(key)
            
            # 如果有多個實驗，添加圖例
            if len(filtered_names) > 1:
                ax.legend()
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # 為頂部標題留出空間
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotter(root)
    root.mainloop() 