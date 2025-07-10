from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed, manual_attack_contact, delete_unit, add_unit, set_unit_to_mission, auto_attack_contact
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import Multi_Side_FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
from scripts.MyNet_multi_battle.Mylib import *
import numpy as np
from collections import deque
import time
import random
import pprint
import os
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy   

import logging

# 导入模型
from module.batch_agent.GBV15_agent import my_Agent
# from module.batch_agent.GBV2_agent import GB_Belief_Agent, GB_Worker_Agent, GB_Belief_Critic
from module.batch_agent.GAT_agent import CenterGAT_QNet
from module.batch_agent.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic
from module.batch_agent.DRQN_agent import RNN_Agent
from module.batch_agent.DQN_agent import DQN_Agent
from module.mixer.qmix import QMixer

class ReplayBuffer:
    """
    簡單的環形緩衝區，用於離線隨機抽樣訓練。
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    

class MyAgent(BaseAgent):
    class EnemyInfo:
        def __init__(self, player_side, enemy_side):
            self.player_side = player_side
            self.enemy_side = enemy_side
            self.enemy_alive = {}
            self.initial_enemy_count = 0
            self.enemy_alive_count = 0
            self.prev_enemy_alive_count = 0
            self.order = []
        def init_episode(self, features):
            self.enemy_alive = {u.Name: 1 for u in features.units[self.enemy_side]}
            self.enemy_alive_count = len(self.enemy_alive)
            self.prev_enemy_alive_count = len(self.enemy_alive)
            if not self.order:
                self.initial_enemy_count = len(self.enemy_alive)
                self.order = [u.Name for u in features.units[self.enemy_side]]
        def get_enemy_found(self, features):
            return 1 if len(features.contacts[self.player_side]) > 0 else 0
        def update_alive(self, features):
            current_ids = {u.Name for u in features.units[self.enemy_side]}
            for name, alive in list(self.enemy_alive.items()):
                if alive == 1 and name not in current_ids:
                    self.enemy_alive[name] = 0
            for name in current_ids:
                if name not in self.enemy_alive:
                    self.enemy_alive[name] = 1
            self.enemy_alive_count = sum(self.enemy_alive.values())
        def alive_ratio(self):
            return (sum(self.enemy_alive.values()) / self.initial_enemy_count) if self.initial_enemy_count > 0 else 0.0
    class FriendlyInfo:
        def __init__(self, player_side):
            self.player_side = player_side
            self.order = []
            self.alive = {}
        def init_episode(self, features):
            if not self.order:
                all_names = [u.Name for u in features.units[self.player_side]]
                for name in all_names:
                    if name not in self.order:
                        self.order.append(name)
            self.alive = {name: 1 for name in self.order}
        def update_alive(self, features):
            current_ids = {u.Name for u in features.units[self.player_side]}
            for name, alive in list(self.alive.items()):
                if alive == 1 and name not in current_ids:
                    self.alive[name] = 0
                elif alive == 0 and name in current_ids:
                    self.alive[name] = 1
        def alive_mask(self):
            return [self.alive.get(n, 0) for n in self.order]

    def __init__(self, player_side: str, enemy_side: str = None,checkpoint_dir: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param enemy_side: 敵人所屬陣營
        :param checkpoint_dir: 檢查點資料夾路徑
        """
        super().__init__(player_side)
        self.player_side = player_side
        self.enemy_side = enemy_side
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent")
        self.logger.setLevel(logging.INFO)
        
        # 初始化 Mylib 並傳入 self
        self.mylib = Mylib(self)

        self.agent_type = 'GAT' #'GBV15','GBV15', 'GBV2', 'Feudal', 'DRQN', 'DQN'
        self.enable_mixer = False
        self.enable_double_q = True
        self.enable_on_policy = True
        # 自動生成運行時間及 checkpoint 資料夾
        self.run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join('checkpoints', self.agent_type, self.run_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 建立state_action_logs資料夾
        self.logs_dir = os.path.join(self.checkpoint_dir, 'state_action_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 初始化記錄文件相關變量
        self.current_log_file = None
        self.log_file_handle = None
        
        # 複製 sample_agent.py 到 checkpoint 資料夾
        src = os.path.abspath(__file__)
        dst = os.path.join(self.checkpoint_dir, os.path.basename(__file__))
        shutil.copy2(src, dst)
        # 複製 demo.py 到 checkpoint 資料夾
        src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'demo.py'))
        dst = os.path.join(self.checkpoint_dir, os.path.basename('demo.py'))
        shutil.copy2(src, dst)
        # 複製 config.yaml 到 checkpoint 資料夾
        src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))
        dst = os.path.join(self.checkpoint_dir, os.path.basename('config.yaml'))
        shutil.copy2(src, dst)

        # 網路參數
        class Args:
            def __init__(self):
                self.hidden_dim = 64

                #變更地圖須變更參數
                self.n_agents = 7
                self.enemy_num = 7

                self.input_size = 7 + 5 * (self.n_agents-1) + 4 * self.enemy_num  # [相對X, 相對Y, 敵人是否存在, 敵人存活比率, 敵人位置, 彈藥比率]
                self.n_actions = 4  # 前進、左轉、右轉、攻擊
                self.goal_dim = 4   # 修改：讓 Manager 生成更豐富的策略目標
                # Goal 可以表示：[探索方向_x, 探索方向_y, 攻擊傾向, 防守傾向, 速度偏好, 協調偏好, 風險偏好, 目標優先級]
                

                self.manager_hidden_dim = 64
                self.worker_hidden_dim = 64
                self.state_dim_d = 3
                self.embedding_dim_k = 16

        self.args = Args()
        self.input_size = self.args.input_size
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # 修改記憶存儲方式，使用列表存儲完整的episode
        self.episode_memory = []  # 存儲當前episode的經驗
        self.completed_episodes = []  # 存儲已完成的episodes
        self.max_episodes = 32  # 最多保存的episode數量


        # 基本超參數
        self.gamma = 0.99
        self.lr = 5e-4
        self.batch_size = 32   # B
        self.sequence_len = 64    # T
        # self.train_interval = 50        # 每隔多少 steps 學習一次
        self.update_freq = 10000        # 每隔多少 steps 同步 target network
        self.save_freq = 50000        # 每隔多少 steps 保存 checkpoint
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_steps = 100000
        
        self.done_condition = 0.1
        self.max_distance = 90.0
        self.win_reward = 150
        self.min_win_reward = 50
        self.reward_scale = 25
        self.loss_threshold = 1.0  # 當 loss 超過此閾值時輸出訓練資料
        self.loss_log_file = 'large_loss_episodes.txt'  # 記錄異常 loss 的 episode 到文字檔

        # ===============================初始化網路========================================
        if self.agent_type == 'GBV15':
            # 建立 policy_net 與 target_net
            self.my_agent = my_Agent(self.input_size, self.args).to(self.device)
            self.target_my_agent = deepcopy(self.my_agent)
            self.target_my_agent.eval() # 設置為評估模式
            self.params = list(self.my_agent.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
            
            # 用於記錄上一時刻的特徵以計算全域狀態
            self.last_features = None
        elif self.agent_type == 'GBV2':
            self.manager_agent = GB_Belief_Agent(self.input_size, self.args).to(self.device)
            self.worker_agent = GB_Worker_Agent(self.input_size, self.args).to(self.device)
            self.target_worker_agent = deepcopy(self.worker_agent)
            self.target_worker_agent.eval() # 設置為評估模式
            self.manager_critic = GB_Belief_Critic(self.input_size, self.args).to(self.device)
            # 新增: 為 GBV2 創建 manager 和 worker 優化器
            self.manager_optimizer = torch.optim.Adam(
                list(self.manager_agent.parameters()) + list(self.manager_critic.parameters()),
                lr=self.lr
            )
            self.worker_optimizer = torch.optim.Adam(
                list(self.worker_agent.parameters()),
                lr=self.lr
            )
        elif self.agent_type == 'Feudal':
            self.manager_agent = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
            self.worker_agent = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
            self.critic = FeUdalCritic(self.input_size, self.args).to(self.device)
        elif self.agent_type == 'DRQN':
            self.rnn_agent = RNN_Agent(self.input_size, self.args).to(self.device)
        elif self.agent_type == 'DQN':
            self.dqn_agent = DQN_Agent(self.input_size, self.args).to(self.device)
        elif self.agent_type == 'GAT':
            # --- GAT 相關超參數 ---
            self.node_dim  = 29 + 1 + self.args.n_agents   # 29 + type + one-hot(id)             
            self.edge_dim   = 1
            self.embed_dim  = 128
            self.n_actions  = 4              # 你的動作數 (前進/左/右/打)
            self.gat_layers = 3

            # --- 建圖網路 ---
            self.gat_net = CenterGAT_QNet(
                    in_dim    = self.node_dim,   # == 37
                    hid_dim   = self.embed_dim,  # 128
                    n_layers  = self.gat_layers, # 3
                    n_actions = self.n_actions,   # 4
                    edge_dim  = self.node_dim     # 33
            ).to(self.device)

            self.gat_target = deepcopy(self.gat_net)
            self.gat_target.eval()
            self.optimizer  = torch.optim.Adam(self.gat_net.parameters(), lr=self.lr)

        # 初始化 QMIX 混合網路
        if self.enable_mixer:
            global_state_dim = self.args.n_agents * 5 + self.args.enemy_num * 4
            self.mixer = QMixer(self.args.n_agents, global_state_dim, self.args.hidden_dim).to(self.device)
            self.target_mixer = deepcopy(self.mixer)
            self.target_mixer.eval()
            self.params += list(self.mixer.parameters())
        
        

        # 初始化隱藏狀態
        if self.agent_type == 'GBV15':
            self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
        elif self.agent_type == 'GBV2':
            self.manager_hidden, self.worker_hidden = self.manager_agent.init_hidden(), self.worker_agent.init_hidden()
        elif self.agent_type == 'Feudal':
            self.manager_hidden, self.worker_hidden = self.manager_agent.init_hidden(), self.worker_agent.init_hidden()
        elif self.agent_type == 'DRQN':
            self.rnn_agent_hidden = self.rnn_agent.init_hidden()

        self.best_distance = 1000000
        self.worst_distance = 0
        self.total_reward = 0
        self.prev_score = 0

        # 初始化緩衝區
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.total_steps = 0
        self.epsilon = 1.0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.alive = None

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_init = True
        self.episode_step = 0
        self.episode_count = 0
        self.max_episode_steps = 500
        self.min_episode_steps = 200
        self.episode_done = False
        self.episode_reward = 0
        self.done = False

        self.step_times_rl = []
        self.reset_cmd = ""
        # 新增：追蹤每個 episode 的統計以計算 5 期平均
        self.episode_steps_history = []
        self.episode_loss_history = []
        self.episode_return_history = []

        # __init__ 
        self.enemy_info = MyAgent.EnemyInfo(self.player_side, self.enemy_side)
        self.friendly_info = MyAgent.FriendlyInfo(self.player_side)

        if checkpoint_dir:
            self._load_checkpoint(checkpoint_dir)

    def _load_checkpoint(self, path: str):
        """內部使用：從 path 載入模型、optimizer、epsilon、steps"""
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get('agent_type') == 'GAT':
            self.gat_net.load_state_dict(ckpt['gat_net'])
            self.gat_target.load_state_dict(ckpt['gat_target'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:   # 其他
            self.my_agent.load_state_dict(ckpt['model'])
            self.target_my_agent.load_state_dict(ckpt['target_model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        # 有存就還原，沒有就保留原本初始化的值
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.total_steps   = ckpt.get('total_steps',   self.total_steps)
        self.episode_count = ckpt.get('episode_count', self.episode_count)
        self.logger.info(f"載入 checkpoint：{path}，從 step={self.total_steps} 繼續")
        
    def _save_checkpoint(self, path: str):
        # 儲存模型
        if self.total_steps % self.save_freq == 0:
            ckpt = {    
                'agent_type': self.agent_type,
                'epsilon':      self.epsilon,
                'total_steps':  self.total_steps,
                'episode_count':self.episode_count                
            }
            if self.agent_type == 'GBV2':
                ckpt['manager_agent'] = self.manager_agent.state_dict()
                ckpt['manager_critic'] = self.manager_critic.state_dict()
                ckpt['worker_agent'] = self.worker_agent.state_dict()
                ckpt['target_worker_agent'] = self.target_worker_agent.state_dict()
                ckpt['manager_optimizer'] = self.manager_optimizer.state_dict()
                ckpt['worker_optimizer'] = self.worker_optimizer.state_dict()
            if self.agent_type == 'GBV15':
                ckpt['model'] = self.my_agent.state_dict()
                ckpt['target_model'] = self.target_my_agent.state_dict()
                ckpt['optimizer'] = self.optimizer.state_dict()
            if self.agent_type == 'GAT':
                ckpt['gat_net']    = self.gat_net.state_dict()
                ckpt['gat_target'] = self.gat_target.state_dict()
                ckpt['optimizer']  = self.optimizer.state_dict()
            if self.enable_mixer:
                ckpt['mixer'] = self.mixer.state_dict()
                ckpt['target_mixer'] = self.target_mixer.state_dict()
            # 使用預先創建的 checkpoint 目錄
            path = os.path.join(self.checkpoint_dir, f'ckpt_{self.total_steps}.th')
            torch.save(ckpt, path)
            self.logger.info(f"已儲存 checkpoint: {path}")

    def get_unit_info_from_observation(self, features: Multi_Side_FeaturesFromSteam, side: str, unit_name: str) -> Unit:
        """
        從觀察中獲取指定單位的資訊。
        """
        units = features.units[side]
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None
    
    def get_contact_info_from_observation(self, features: Multi_Side_FeaturesFromSteam, side: str, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        """
        contacts = features.contacts[side]
        for contact in contacts:
            if contact['Name'] == contact_name:
                return contact
        return None

    # -------------------------------------------------------
    #  Graph 前處理：把 CMO observation 轉成 (Xl, El)
    # -------------------------------------------------------
    def obs_to_Xl_El(self, features: Multi_Side_FeaturesFromSteam):
        """
        依照目前設定動態產生：
            Xl : (N, 31)  = [local-29 | type-1 | id-n_agents]  
            El : (N, N, self.node_dim)  可視邊 (值全 0/1)
        N = n_agents + enemy_num
        """
        N = self.args.n_agents + self.args.enemy_num
        X_rows, units_all = [], []

        # ---------- 友軍節點 ----------
        for idx, name in enumerate(self.friendly_info.order):
            u = self.get_unit_info_from_observation(features, self.player_side, name)
            units_all.append(u)
            local29 = self.get_state(features, u)[:29] if u is not None else np.zeros(29, np.float32)
            one_hot = np.eye(self.args.n_agents, dtype=np.float32)[idx]          # 動態 one-hot
            X_rows.append(np.concatenate([local29, [0], one_hot]))               # type = 0

        # ---------- 敵軍節點 ----------
        for name in self.enemy_info.order:
            u = self.get_unit_info_from_observation(features, self.enemy_side, name)
            units_all.append(u)
            local29 = np.zeros(29, np.float32)                                   # 目前不取敵方 local
            X_rows.append(np.concatenate([local29, [1],
                                        np.zeros(self.args.n_agents, np.float32)]))  # type = 1

        Xl = torch.tensor(X_rows, dtype=torch.float32, device=self.device)       # (N,31)

        # ---------- visibility edge ----------
        vis = np.zeros((N, N, self.node_dim), dtype=np.float32)
        for i in range(self.args.n_agents):                                      # 只有友軍能「看」
            friendly = units_all[i]
            if friendly is None:
                continue
            for c in features.contacts[self.player_side]:
                if c['Name'] in self.enemy_info.order:
                    j = self.args.n_agents + self.enemy_info.order.index(c['Name'])
                    vis[i, j, :] = 1.0                                           # 整條 33-dim 設 1
        El = torch.tensor(vis, dtype=torch.float32, device=self.device)
        return Xl, El

    
    def get_done(self,state: list[np.ndarray]):
        # 跳過第一步的 done 檢測，避免場景尚未更新時誤判
        if self.episode_step == 0:
            return False
        # 如果已達最大步數限制，強制結束 episode
        if self.episode_step >= self.max_episode_steps:
            return True
        done = True
        # 到達目的地
        for i, name in enumerate(self.friendly_info.order):
            if state[i][0] > self.done_condition: 
                done = False
        return done
    
    def get_win(self, state: list[np.ndarray]):
        win = True
        # 所有船抵達目標
        for i, name in enumerate(self.friendly_info.order):
            if state[i][0] >= self.done_condition:
                win = False
            elif state[i][0] == 0.0:
                win = False
        return win

    def get_distance(self, dx, dy):
        """计算智能体与目标之间的距离，支持NumPy数组和PyTorch张量"""
        if isinstance(dx, torch.Tensor):
            # 使用PyTorch操作
            return torch.sqrt((dx)**2 + (dy)**2)
        else:
            # 使用NumPy操作
            return np.sqrt((dx)**2 + (dy)**2)
            
    def action(self, features: Multi_Side_FeaturesFromSteam) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        if self.episode_init:
            # 第一次執行 action()，初始化敵人與友軍資訊
            self.enemy_info.init_episode(features)
            self.friendly_info.init_episode(features)
            self.episode_init = False
            self.episode_count += 1
            self.logger.info(f"episode: {self.episode_count}")
            
            # 創建新的日志文件用於記錄state和action
            self._create_new_log_file()

        if features.sides_[self.player_side].TotalScore == 0:
            self.prev_score = 0
        # print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        has_unit = False
        for unit in features.units[self.player_side]:
            has_unit = True
        if not has_unit:
            self.logger.warning(f"找不到任何單位")
            return self.reset()  # 如果找不到單位，返回初始化
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        current_state = self.get_states(features)
        
        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            self.done = self.get_done(current_state)
            score_change = features.sides_[self.player_side].TotalScore - self.prev_score
            rewards = self.get_rewards(features, self.prev_state, current_state, score_change)
            self.prev_score = features.sides_[self.player_side].TotalScore
            # 将 rewards 列表转换为 numpy 数组并计算平均值
            rewards_arr = np.array(rewards, dtype=np.float32)
            avg_reward = rewards_arr.mean() if rewards_arr.size > 0 else 0.0
            self.total_reward += avg_reward
            self.episode_reward += avg_reward
            
            # 計算並記錄全域狀態
            prev_global_state = self.get_global_state(self.last_features, self.prev_state)
            current_global_state = self.get_global_state(features, current_state)
            #self.episode_memory.append((self.prev_state, prev_global_state, current_state, current_global_state, self.prev_action, rewards, self.done, self.alive))
            # 先把目前觀測轉成 graph
            Xl_now, El_now   = self.obs_to_Xl_El(features)
            Xl_prev, El_prev = self.obs_to_Xl_El(self.last_features) if self.last_features is not None else (Xl_now, El_now)

            self.episode_memory.append((
                    Xl_prev.cpu(), El_prev.cpu(),        # t
                    self.prev_action,                              # a_t (list[int] 長度3)
                    rewards,                             # r_t (list[float] 長度3)
                    Xl_now.cpu(), El_now.cpu(),          # t+1
                    self.done                            # done_t
            ))

            # 檢查遊戲是否結束
            if self.done or self.episode_step > self.max_episode_steps:
                self.episode_done = True
                self.logger.info(f"遊戲結束! 總獎勵: {self.episode_reward:.4f}")
                
                # 將完成的episode添加到已完成episodes列表中
                if len(self.episode_memory) > 0:
                    self.completed_episodes.append(self.episode_memory)
                    # 限制已完成的episodes數量
                    if len(self.completed_episodes) > self.max_episodes:
                        self.completed_episodes.pop(0)


                                # 在遊戲結束時進行訓練
                # 根據 agent 類型決定訓練次數：GBV2 為 on-policy，只訓練一次，其它 off-policy 可用多次隨機抽樣
                if self.agent_type == 'GBV2':
                    loss = self.train()
                else:
                    loss = 0
                    for _ in range(self.batch_size):
                        loss += self.train()
                    loss = loss / self.batch_size
                # 重置遊戲狀態

                self.episode_steps_history.append(self.episode_step)
                self.episode_loss_history.append(loss)
                self.episode_return_history.append(self.episode_reward)
                if self.episode_count % 5 == 0:

                    # 計算最近 5 個 episode 的平均值
                    window = 5
                    count = len(self.episode_steps_history)
                    avg_steps = sum(self.episode_steps_history[-window:]) / min(window, count)
                    avg_loss = sum(self.episode_loss_history[-window:]) / min(window, count)
                    avg_return = sum(self.episode_return_history[-window:]) / min(window, count)
                    # 記錄平均值
                    self.stats_logger.log_stat("episode_step", float(avg_steps), self.total_steps)
                    self.stats_logger.log_stat("loss", float(avg_loss), self.total_steps)
                    self.stats_logger.log_stat("episode_return", float(avg_return), self.total_steps)

                    # 重置統計
                    self.episode_steps_history = []
                    self.episode_loss_history = []
                    self.episode_return_history = []
                
                return self.reset()
            
        
        
        # 選擇動作
        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)   # → [1, Agent, feat]
        state_tensor = state_tensor.unsqueeze(1)       # → [seq_len=1, batch=1, Agent, feat]
        t0_rl = time.perf_counter()
        with torch.no_grad():
            if self.agent_type == 'GBV15':
                q_values, (self.manager_hidden, self.worker_hidden) = \
                    self.my_agent(state_tensor, (self.manager_hidden, self.worker_hidden))
            elif self.agent_type == 'GBV2':
                goal, self.manager_hidden = \
                    self.manager_agent(state_tensor, self.manager_hidden)
                q_values, self.worker_hidden = self.worker_agent(
                    state_tensor, 
                    self.worker_hidden,
                    goal
                )
            elif self.agent_type == 'Feudal':
                _, goal, self.manager_hidden = self.manager_agent(state_tensor, self.manager_hidden)
            
                # Worker根据目标选择动作
                q_values, self.worker_hidden = self.worker_agent(
                    state_tensor, 
                    self.worker_hidden,
                    goal
                )
                critic_value = self.critic(state_tensor)
            elif self.agent_type == 'DRQN':
                q_values, self.rnn_agent_hidden = self.rnn_agent(state_tensor, self.rnn_agent_hidden)
            elif self.agent_type == 'DQN':
                q_values = self.dqn_agent(state_tensor)
            elif self.agent_type == 'GAT':
                Xl, El = self.obs_to_Xl_El(features)        # (6,31), (6,6,1)
                q_all  = self.gat_net(Xl, El)              # (6, n_actions)
                q_values = q_all.unsqueeze(0).unsqueeze(0) # → [T=1, B=1, 6, n_actions]


        dt_rl = time.perf_counter() - t0_rl
        # self.step_times_rl.append(dt_rl)
        # if dt_rl > 0.01:
        #     print(prof.key_averages().table(sort_by="cuda_time_total"))
        
        # 檢查是否有敵人
        has_enemy = len(features.contacts[self.player_side]) > 0

        # q_values shape: [T, B, A, n_actions]
        # 取出第一個時間步＆batch
        #q_vals = q_values[0, 0]            # shape [A, n_actions]
        #A, n_actions = q_vals.shape
        # 取 3 艘友軍的 Q 值
        # 只保留友軍 3 節點的 Q 值
        q_vals = q_all[: self.args.n_agents]  # (3, n_actions)

        # ── 選動作（下面的全用 A = 3）────────────────────
        A, n_actions = q_vals.shape           # A = 3
        # ─── ε-greedy 選動作 ───────────────────────────────
        masks, actions = [], []
        for i in range(A):
            mask = torch.ones(n_actions, dtype=torch.bool, device=self.device)
            mount_ratio = current_state[i][5]
            if not has_enemy or mount_ratio <= 0:
                mask[3] = False
            masks.append(mask)

            # ε-greedy
            if random.random() < self.epsilon:
                allowed = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                act = random.choice(allowed) if allowed else 0
            else:
                q_masked = q_vals[i].clone()
                q_masked[~mask] = -float('inf')
                act = int(q_masked.argmax())
            actions.append(act)

        # 產生 action_mask: 限制在以目標點為左上角，邊長 self.max_distance 的正方形內行動
        #masks = []
        #for i in range(A):
            # 從當前狀態列表取得第 i 個 agent 的 dx_norm, dy_norm
            # dx = current_state[i][0] * self.max_distance
            # dy = current_state[i][1] * self.max_distance
            #mask = torch.ones(n_actions, dtype=torch.bool, device=self.device)
            # # 上邊界: dy <= 0 無法再向北
            # if dy <= 0:
            #     mask[0] = False
            # # 下邊界: dy >= self.max_distance 無法再向南
            # if dy >= self.max_distance:
            #     mask[1] = False
            # # 左邊界: dx >= 0 無法再向西
            # if dx >= 0:
            #     mask[2] = False
            # # 右邊界: dx <= -self.max_distance 無法再向東
            # if dx <= -self.max_distance:
            #     mask[3] = False
            # 無敵人時禁止攻擊
            #if not has_enemy:
                #mask[3] = False
            # 無彈藥時禁止攻擊
            #mount_ratio = current_state[i][5]
            #if mount_ratio <= 0:
                #mask[3] = False
            #masks.append(mask)
        actions = []
        action_cmd = ""

        for ai in range(A):
            q_agent = q_vals[ai]           # shape [n_actions]
            # 根據 action_mask 執行 ε-greedy 或隨機策略
            mask = masks[ai]
            if random.random() < self.epsilon:
                # 隨機從允許的動作中選擇
                allowed = mask.nonzero().squeeze(-1).tolist()
                act = random.choice(allowed) if allowed else 0
                self.logger.debug(f"Agent {ai} 隨機選擇動作: {act}")
            else:
                # ε-greedy：先將不允許的動作設為 -inf，再取 argmax
                q_agent_masked = q_agent.clone()
                q_agent_masked[~mask] = -float('inf')
                act = int(q_agent_masked.argmax().item())
                self.logger.debug(f"Agent {ai} 根據Q值選擇動作: {act}")
            actions.append(act)

        # 更新友軍存活狀態並分配動作
        self.friendly_info.update_alive(features)
        alive_mask = self.friendly_info.alive_mask()
        self.alive = np.array(alive_mask, dtype=bool)
        action_cmd = ""
        for idx, name in enumerate(self.friendly_info.order):
            if not self.alive[idx]:
                continue
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            # rule-based stop if reached goal
            if current_state[idx][0] < self.done_condition:
                action_cmd += "\n" + set_unit_heading_and_speed(
                    side=self.player_side,
                    unit_name=name,
                    heading=unit.CH,
                    speed=0
                )
            else:
                action_cmd += "\n" + self.apply_action(actions[idx], unit, features)

        # 記錄state和action到文件
        self._log_state_action(current_state, actions, self.episode_step)

        if self.episode_step < 10:
            for unit in features.units[self.enemy_side]:
                action_cmd += "\n" + set_unit_to_mission(
                    unit_name=unit.Name,
                    mission_name='Kinmen patrol'
                )
        self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = actions
        self.total_steps += 1
        self.episode_step += 1
        self.alive = alive_mask
        # print("alive:", self.alive)

        # 更新 epsilon
        self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * self.total_steps / self.eps_decay_steps
        self.epsilon = max(self.eps_end, self.epsilon)
        self.logger.debug(f"更新epsilon: {self.epsilon:.4f}")

        # 更新目標網路
        if self.total_steps % self.update_freq == 0:
            if self.agent_type == 'GBV15':
                self.target_my_agent.load_state_dict(self.my_agent.state_dict())
            if self.enable_mixer:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            # 同步 GBV2 worker 目標網路
            if self.agent_type == 'GBV2':
                self.target_worker_agent.load_state_dict(self.worker_agent.state_dict())
            if self.agent_type == 'GAT':
                self.gat_target.load_state_dict(self.gat_net.state_dict())
        
        # 儲存模型
        self._save_checkpoint(self.checkpoint_dir)
        
        # 保存當前features以供下一步計算全域狀態
        self.last_features = features
        
        return action_cmd

    def get_state(self, features: Multi_Side_FeaturesFromSteam, ac: Unit) -> np.ndarray:
        """
        獲取當前狀態向量，包含自身單位和目標的相對位置資訊。
        :return: numpy 陣列，例如 
        [相對X, 相對Y, 敵人是否存在, 敵人存活比率,所有敵人位置]
        """
        target_lon = float(118.27954108343)
        target_lat = float(24.333113806906)
        max_distance = self.max_distance
        
        # 計算相對位置
        ac_lon = float(ac.Lon)
        ac_lat = float(ac.Lat)
        
        # 計算相對座標 (X,Y)，將經緯度差轉換為大致的平面座標
        # 注意：這是簡化的轉換，對於小範圍有效
        # X正方向為東，Y正方向為北
        earth_radius = 6371  # 地球半徑（公里）
        lon_scale = np.cos(np.radians(ac_lat))  # 經度在當前緯度的縮放因子
        
        # 1. 計算目標相對 X 和 Y（公里）
        dx = (target_lon - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
        dy = (target_lat - ac_lat) * np.pi * earth_radius / 180.0
        dist = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / max_distance
        dy_norm = dy / max_distance
        dist_norm = dist / max_distance

        # 計算目標方位角（0=東，逆時針為正）
        target_angle = np.arctan2(dy, dx)

        # CMO 的 CH: 0=北，順時針增
        # 轉到 0=東，逆時針增：heading_math = 90°−CH
        heading_rad = np.deg2rad(90.0 - ac.CH)

        # 相對角度 = 目標方位 − 自身航向
        relative_angle = target_angle - heading_rad

        # 正規化到 [-π, π]
        relative_angle = (relative_angle + np.pi) % (2*np.pi) - np.pi

        # 如需用 sin/cos 表示
        relative_sin = np.sin(relative_angle)
        relative_cos = np.cos(relative_angle)

        # 敵方資訊處理: 檢查是否有敵人 & 更新存活狀態
        enemy_found = self.enemy_info.get_enemy_found(features)
        self.enemy_info.update_alive(features)
        alive_ratio = self.enemy_info.alive_ratio()
        step_ratio = self.episode_step / self.max_episode_steps

        # 計算彈藥持有比率
        mount_ratio = 0.0
        mounts = getattr(ac, 'Mounts', None)
        if mounts:
            for mount in mounts:
                name = getattr(mount, 'Name', None)
                weapons = getattr(mount, 'Weapons', [])
                if not weapons:
                    continue
                curr = weapons[0].QuantRemaining
                maxq = weapons[0].MaxQuant
                ratio = curr / maxq if maxq > 0 else 0.0
                if name == 'Hsiung Feng II Quad':
                    mount_ratio += ratio
                elif name == 'Hsiung Feng III Quad':
                    mount_ratio += ratio
            mount_ratio /= 2

        # 構建基礎狀態向量[距離, 方位sin, 方位cos, 敵人是否存在, 敵人存活比率,彈藥比率,步數比率]
        base_state = np.array([
                                dist_norm, #0
                                relative_sin, #1
                                relative_cos, #2
                                enemy_found, #3
                                alive_ratio, #4
                                mount_ratio, #5
                                step_ratio #6
                                ], dtype=np.float32)

        state_vec = base_state
        # 友軍狀態向量
        friendly_positions = []
        for name in self.friendly_info.order:
            friendly_unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if friendly_unit is not None :
                if friendly_unit.Name == ac.Name:
                    continue
                # 計算相對位置
                friendly_alive = 1.0
                friendly_dx = (float(friendly_unit.Lon) - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
                friendly_dy = (float(friendly_unit.Lat) - ac_lat) * np.pi * earth_radius / 180.0
                friendly_dx_norm = friendly_dx / max_distance
                friendly_dy_norm = friendly_dy / max_distance
                friendly_dist_norm = np.sqrt(friendly_dx_norm**2 + friendly_dy_norm**2)

                # 計算方位角
                friendly_angle = np.arctan2(friendly_dy, friendly_dx)
                friendly_relative_angle = friendly_angle - heading_rad
                friendly_relative_angle = (friendly_relative_angle + np.pi) % (2*np.pi) - np.pi
                friendly_relative_sin = np.sin(friendly_relative_angle)
                friendly_relative_cos = np.cos(friendly_relative_angle)
                # 計算彈藥持有比率
                mount_ratio = 0.0
                mounts = getattr(friendly_unit, 'Mounts', None)
                if mounts:
                    for mount in mounts:
                        name = getattr(mount, 'Name', None)
                        weapons = getattr(mount, 'Weapons', [])
                        if not weapons:
                            continue
                        curr = weapons[0].QuantRemaining
                        maxq = weapons[0].MaxQuant
                        ratio = curr / maxq if maxq > 0 else 0.0
                        if name == 'Hsiung Feng II Quad':
                            mount_ratio += ratio
                        elif name == 'Hsiung Feng III Quad':
                            mount_ratio += ratio
                mount_ratio /= 2
            else:
                friendly_alive = 0.0
                friendly_dist_norm = 0.0
                friendly_relative_sin = 0.0
                friendly_relative_cos = 0.0
                mount_ratio = 0.0
            # 構建友軍狀態向量[存活, 距離, 方位sin, 方位cos, 彈藥比率]
            friendly_positions += [
                                    friendly_alive,
                                    friendly_dist_norm,
                                    friendly_relative_sin,
                                    friendly_relative_cos,
                                    mount_ratio
                                ]
        state_vec = np.concatenate([state_vec, friendly_positions])


        # 敵人狀態向量
        enemy_positions = []
        for name in self.enemy_info.order:
            enemy_unit = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if enemy_unit is not None:
                enemy_alive = 1.0
                enemy_dx = (float(enemy_unit.Lon) - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
                enemy_dy = (float(enemy_unit.Lat) - ac_lat) * np.pi * earth_radius / 180.0
                enemy_dx_norm = enemy_dx / max_distance
                enemy_dy_norm = enemy_dy / max_distance
                enemy_dist_norm = np.sqrt(enemy_dx_norm**2 + enemy_dy_norm**2)
                # 計算方位角
                enemy_angle = np.arctan2(enemy_dy, enemy_dx)
                enemy_relative_angle = enemy_angle - heading_rad
                enemy_relative_angle = (enemy_relative_angle + np.pi) % (2*np.pi) - np.pi
                enemy_relative_sin = np.sin(enemy_relative_angle)
                enemy_relative_cos = np.cos(enemy_relative_angle)
            else:
                enemy_alive = 0.0
                enemy_dist_norm = 0.0
                enemy_relative_sin = 0.0
                enemy_relative_cos = 0.0
            # 構建敵人狀態向量[存活, 距離, 方位sin, 方位cos]
            enemy_positions += [
                                    enemy_alive,
                                    enemy_dist_norm,
                                    enemy_relative_sin,
                                    enemy_relative_cos
                                ]

        # 整合敵人位置
        state_vec = np.concatenate([state_vec, enemy_positions])
       
        # 整合最終狀態向量
        raw_state = state_vec

        # normalized_state = self.normalize_state(raw_state)
        # print("Name:", ac.Name)
        # print("raw_state:", raw_state)
        # print("normalized_state:", normalized_state)
        
        # 返回狀態向量
        return raw_state
    
    def get_states(self, features: Multi_Side_FeaturesFromSteam) -> list[np.ndarray]:
        states = []
        # 對每個初始友軍單位按順序生成狀態，死亡的單位回傳默認值
        for name in self.friendly_info.order:
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                # 單位死亡或不存在，返回預設零state
                state = np.zeros(self.input_size, dtype=np.float32)
                # state = self.normalize_state(raw_state)
                # print(f"單位 {name} 死亡或不存在，返回預設零state: {state}")
            else:
                state = self.get_state(features, unit)
            states.append(state)
        return states
    
    def get_global_state(self, features: Multi_Side_FeaturesFromSteam, states: list[np.ndarray]) -> np.ndarray:
        target_lon = float(118.27954108343)
        target_lat = float(24.333113806906)
        max_distance = self.max_distance
        global_state = []
        for name in self.friendly_info.order:
            ac = self.get_unit_info_from_observation(features, self.player_side, name)
            if ac is not None:
                # 計算相對位置
                ac_lon = float(ac.Lon)
                ac_lat = float(ac.Lat)
                # 計算相對座標 (X,Y)，將經緯度差轉換為大致的平面座標
                # 注意：這是簡化的轉換，對於小範圍有效
                # X正方向為東，Y正方向為北
                earth_radius = 6371  # 地球半徑（公里）
                lon_scale = np.cos(np.radians(ac_lat))  # 經度在當前緯度的縮放因子
                
                # 1. 計算目標相對 X 和 Y（公里）
                dx = (target_lon - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
                dy = (target_lat - ac_lat) * np.pi * earth_radius / 180.0
                dist = np.sqrt(dx**2 + dy**2)
                dx_norm = dx / max_distance
                dy_norm = dy / max_distance
                dist_norm = dist / max_distance

                # 計算目標方位角（0=東，逆時針為正）
                target_angle = np.arctan2(dy, dx)

                # CMO 的 CH: 0=北，順時針增
                # 轉到 0=東，逆時針增：heading_math = 90°−CH
                heading_rad = np.deg2rad(90.0 - ac.CH)

                # 相對角度 = 目標方位 − 自身航向
                relative_angle = target_angle - heading_rad

                # 正規化到 [-π, π]
                relative_angle = (relative_angle + np.pi) % (2*np.pi) - np.pi

                # 如需用 sin/cos 表示
                relative_sin = np.sin(relative_angle)
                relative_cos = np.cos(relative_angle)

                alive = 1.0

                # 計算彈藥持有比率
                mount_ratio = 0.0
                mounts = getattr(ac, 'Mounts', None)
                if mounts:
                    for mount in mounts:
                        name = getattr(mount, 'Name', None)
                        weapons = getattr(mount, 'Weapons', [])
                        if not weapons:
                            continue
                        curr = weapons[0].QuantRemaining
                        maxq = weapons[0].MaxQuant
                        ratio = curr / maxq if maxq > 0 else 0.0
                        if name == 'Hsiung Feng II Quad':
                            mount_ratio += ratio
                        elif name == 'Hsiung Feng III Quad':
                            mount_ratio += ratio
                    mount_ratio /= 2
            else:
                alive = 0.0
                dist_norm = 0.0
                relative_sin = 0.0
                relative_cos = 0.0
                mount_ratio = 0.0
            # 構建基礎狀態向量[距離, 方位sin, 方位cos, 敵人是否存在, 敵人存活比率,彈藥比率,步數比率]
            base_state = np.array([
                                    alive, #0
                                    dist_norm, #1
                                    relative_sin, #2
                                    relative_cos, #3
                                    mount_ratio, #4
                                    ], dtype=np.float32)
            global_state = np.concatenate([global_state, base_state])

        for name in self.enemy_info.order:
            enemy_unit = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if enemy_unit is not None:
                enemy_alive = 1.0
                enemy_dx = (float(enemy_unit.Lon) - target_lon) * np.pi * earth_radius * lon_scale / 180.0
                enemy_dy = (float(enemy_unit.Lat) - target_lat) * np.pi * earth_radius / 180.0
                enemy_dx_norm = enemy_dx / max_distance
                enemy_dy_norm = enemy_dy / max_distance
                enemy_dist_norm = np.sqrt(enemy_dx_norm**2 + enemy_dy_norm**2)
                # 計算方位角
                enemy_angle = np.arctan2(enemy_dy, enemy_dx)
                enemy_relative_angle = enemy_angle - heading_rad
                enemy_relative_angle = (enemy_relative_angle + np.pi) % (2*np.pi) - np.pi
                enemy_relative_sin = np.sin(enemy_relative_angle)
                enemy_relative_cos = np.cos(enemy_relative_angle)
            else:
                enemy_alive = 0.0
                enemy_dist_norm = 0.0
                enemy_relative_sin = 0.0
                enemy_relative_cos = 0.0
            # 構建敵人狀態向量[存活, 距離, 方位sin, 方位cos]
            enemy_state = np.array([
                                    enemy_alive,
                                    enemy_dist_norm,
                                    enemy_relative_sin,
                                    enemy_relative_cos
                                ], dtype=np.float32)
            global_state = np.concatenate([global_state, enemy_state])


        return global_state
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray, score: int, win: bool) -> np.ndarray:
        # 計算全局 reward
        reward = 0
        
        # 場景score變化獎勵
        reward += score
        # 發現敵人獎勵
        if state[3] == 0.0 and next_state[3] == 1.0:
            reward += 10
        # 往敵方探索獎勵
        reward += 200 * (state[0] - next_state[0])
        # 少一個敵人+10
        # if self.enemy_info.enemy_alive_count < self.enemy_info.prev_enemy_alive_count:
        #     reward += 10 *(self.enemy_info.prev_enemy_alive_count - self.enemy_info.enemy_alive_count)
        # self.enemy_info.prev_enemy_alive_count = self.enemy_info.enemy_alive_count

        if state[0] >= self.done_condition and next_state[0] < self.done_condition:
            reward += 20

        if win:
            win_reward = self.win_reward * (1- (self.episode_step - self.min_episode_steps) / (self.max_episode_steps - self.min_episode_steps))
            win_reward = max(win_reward + self.min_win_reward, self.min_win_reward)
            # if self.enable_mixer:
            #     win_reward = win_reward / self.args.n_agents
            
            reward += win_reward

        # 原始獎勵
        raw_reward = reward
        # 獲勝獎勵200 + 敵軍總數 7 *擊殺獎勵 20 + 最大距離獎勵 200*7
        max_return = self.win_reward + self.min_win_reward + self.enemy_info.initial_enemy_count * 20 +  100
        scaled_reward = raw_reward/(max_return/self.reward_scale)
        # self.logger.info(f"raw reward: {raw_reward:.4f}, scaled reward: {scaled_reward:.4f}")
        # 將標量 reward 擴展為多代理人向量
        # return raw_reward
        return scaled_reward
    
    def get_rewards(self,features: Multi_Side_FeaturesFromSteam, state: list[np.ndarray], next_state: list[np.ndarray], score: int) -> list[np.ndarray]:
        rewards = []
        # 對每個初始友軍單位按順序生成狀態，死亡的單位回傳默認值
        for i, name in enumerate(self.friendly_info.order):
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                # 單位死亡或不存在，給予0獎勵
                reward = 0
            else:
                reward = self.get_reward(state[i], next_state[i], score, self.get_win(next_state))
            # 無論單位是否存活，都添加對應獎勵，確保長度一致
            rewards.append(reward)

        return rewards

    def apply_action(self, action: int, ac: Unit, features: Multi_Side_FeaturesFromSteam) -> str:
        """將動作轉換為 CMO 命令"""
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0: #前進
            heading = ac.CH
        elif action == 1: #左轉
            heading = ac.CH-30
        elif action == 2: #右轉
            heading = ac.CH+30
        elif action == 3:  # 攻擊
            # 檢查是否有彈藥
            has_ammo = False
            enemy = random.choice(features.contacts[self.player_side])
            for mount in ac.Mounts:
                name = getattr(mount, 'Name', None)
                if name not in ('Hsiung Feng II Quad', 'Hsiung Feng III Quad'):
                    continue
                        
                weapons = getattr(mount, 'Weapons', [])
                if weapons and weapons[0].QuantRemaining > 0:
                    if name == 'Hsiung Feng III Quad':
                        weapon_id = 1133
                    elif name == 'Hsiung Feng II Quad':
                        weapon_id = 1934
                    has_ammo = True
                    break
            if not has_ammo:
                # 無彈藥，保持前進
                heading = ac.CH
            else:
                # 有彈藥，執行攻擊
                return manual_attack_contact(
                    attacker_id=ac.ID,
                    contact_id=enemy['ID'],
                    weapon_id=weapon_id,
                    qty=1
                )
            
        if heading > 360:
            heading = heading - 360
        elif heading < 0:
            heading = 360 + heading
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=ac.Name,
            heading=heading,
            speed=30
        )
    
    def train(self):
        """訓練網路"""
        if len(self.completed_episodes) < 1:
            self.logger.warning("沒有足夠的episodes進行訓練")
            return
        if self.agent_type == 'GAT':
            # 直接走 GAT 專屬函式，不要先解包
            return self.GAT_train()
        if self.enable_on_policy:
            # 使用最新的episode
            episode = self.completed_episodes[-1]
        else:
            # 隨機選一個已完成 episode
            episode = random.choice(self.completed_episodes)
            # 解包批次: (states, prev_global_states, next_states, next_global_states, actions_list, rewards, dones, _)
        states, prev_global_states, next_states, next_global_states, actions_list, rewards, dones, _ = zip(*episode)
        # 張量轉換: states [T,A,feat] -> [T,1,A,feat]
        states = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device).unsqueeze(1)
        # 張量轉換 global_states [T,S] -> [T,1,S]
        global_states = torch.tensor(prev_global_states, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_global_states = torch.tensor(next_global_states, dtype=torch.float32, device=self.device).unsqueeze(1)
        # actions_list [T,A] -> [T,1,A]
        actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=self.device).unsqueeze(1)
        # rewards [T,A] -> [T,1,A]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        # dones [T] -> [T,1,1]
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1,1,1)

        #----------------------------- Calculate loss -----------------------------------
        if self.agent_type == 'GBV15':
            loss = self.GBV15_train(states, next_states, actions_tensor, rewards_tensor, dones_tensor, global_states, next_global_states)
            return loss
        elif self.agent_type == 'GBV2':
            loss = self.GBV2_train(states, next_states, actions_tensor, rewards_tensor, dones_tensor, global_states, next_global_states)
            return loss
        elif self.agent_type == 'GAT':
            return self.GAT_train()

        # big loss check
        # self.big_loss_check(loss)

    # -------------------------------------------------------
    #  GAT + multi-head DQN 訓練 (on-policy)
    # -------------------------------------------------------
    def GAT_train(self):
        """
        取 self.completed_episodes[-1] 作為 on-policy 批次。
        T = episode 長度, A = n_agents, n_act = 4
        """
        ep = self.completed_episodes[-1]
        T = len(ep)

        # 解析 episode
        Xl_list, El_list, Xl_next_list, El_next_list = [], [], [], []
        act_list, rew_list, done_list = [], [], []
        for Xl, El, a, r, Xl_n, El_n, d in ep:
            Xl_list.append(Xl);   El_list.append(El)
            Xl_next_list.append(Xl_n); El_next_list.append(El_n)
            act_list.append(a);   rew_list.append(r);   done_list.append(d)

        # 堆成 tensor
        Xl      = torch.stack(Xl_list,      dim=0).to(self.device)   # [T, N,31]
        El      = torch.stack(El_list,      dim=0).to(self.device)   # [T, N,N,33]
        Xl_next = torch.stack(Xl_next_list, dim=0).to(self.device)
        El_next = torch.stack(El_next_list, dim=0).to(self.device)

        actions = torch.tensor(act_list, dtype=torch.long,
                            device=self.device)                   # [T, n_agents]
        rewards = torch.tensor(rew_list, dtype=torch.float32,
                            device=self.device)                   # [T, n_agents]
        dones   = torch.tensor(done_list, dtype=torch.float32,
                            device=self.device).unsqueeze(-1)     # [T,1]

        # ---- Q(s,·) ----------------------------------------------------
        Q_all   = self.gat_net(Xl, El)                               # [T, N, n_act]
        Q_next  = self.gat_target(Xl_next, El_next).detach()

        Q_friend      = Q_all[:, : self.args.n_agents]               # [T, n_agents, n_act]
        Q_next_friend = Q_next[:, : self.args.n_agents]

        # 取 a_t
        Q_taken = Q_friend.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [T, n_agents]

        # Double-DQN
        with torch.no_grad():
            best_next = Q_friend.max(dim=2, keepdim=True)[1]             # [T, n_agents,1]
            max_next  = Q_next_friend.gather(2, best_next).squeeze(-1)   # [T, n_agents]

        td_target = rewards + (1 - dones) * self.gamma * max_next
        loss = F.smooth_l1_loss(Q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gat_net.parameters(), 5)
        self.optimizer.step()
        return loss.item()

    def train_batch_test(self):
        valid_eps = [ep for ep in self.completed_episodes if len(ep) >= self.sequence_len]
        if len(valid_eps) < self.batch_size:
            return 0.0

        batch_eps = random.sample(valid_eps, self.batch_size)

        batch_seqs = []
        for ep in batch_eps:
            L = len(ep)
            # 確保至少有 T 長度
            assert L >= self.sequence_len

            r = random.random()
            if r < 0.05:
                start = 0
            elif r < 0.1:
                start = L - self.sequence_len
            else:
                start = random.randint(0, L - self.sequence_len)

            batch_seqs.append(ep[start : start + self.sequence_len])

        # 解包成 numpy array，形狀 [T, B, ...]
        states      = np.stack([[step[0] for step in seq] for seq in batch_seqs], axis=1)
        next_states = np.stack([[step[1] for step in seq] for seq in batch_seqs], axis=1)
        actions     = np.stack([[step[2] for step in seq] for seq in batch_seqs], axis=1)
        rewards     = np.stack([[step[3] for step in seq] for seq in batch_seqs], axis=1)
        dones       = np.stack([[step[4] for step in seq] for seq in batch_seqs], axis=1)

        # 轉成 tensor
        states      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,     dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=self.device).unsqueeze(-1)  # 增加一個維度以匹配 [T, B, A]

        # 初始化 hidden，這裡將 batch_size 傳給 init_hidden
        if self.agent_type == 'GBV15':
            mh0, wh0 = self.my_agent.init_hidden(batch_size=self.batch_size)
        elif self.agent_type == 'Feudal':
            mh0, wh0 = self.manager_agent.init_hidden(batch_size=self.batch_size), self.worker_agent.init_hidden(batch_size=self.batch_size)
        elif self.agent_type == 'DRQN':
            h0 = self.rnn_agent.init_hidden(batch_size=self.batch_size)

        # 正向計算
        q_values, _       = self.my_agent(states,      (mh0, wh0))  # [T, B, A, n_actions]
        with torch.no_grad():
            q_next, _    = self.target_my_agent(next_states, (mh0, wh0))

        # 計算 TD 目標
        q_taken    = q_values.gather(3, actions.unsqueeze(-1)).squeeze(-1)
        max_next_q = q_next.max(dim=3)[0]

        # 更新 TD 目標與損失計算
        td_target = rewards + (1 - dones) * self.gamma * max_next_q  # [T, B, A]
        mse = (q_taken - td_target).pow(2)
        loss = mse.mean()

        # 檢查極端 loss，並列印 episode 訓練資料
        loss_value = loss.item()
        if loss_value > self.loss_threshold:
            self.logger.warning(f"Large loss: {loss_value:.4f} > threshold {self.loss_threshold}")
            pprint.pprint(batch_eps)
            # 將異常 episode 寫入文字檔
            with open(self.loss_log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== Large loss: {loss_value:.4f} ===\n")
                f.write(pprint.pformat(batch_eps) + "\n\n")
        self.my_agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.my_agent.parameters(), max_norm=5)
        self.my_agent_optimizer.step()

        return loss.item()
    
    def train_all_batch(self):
        # 使用整個 episode 並對齊 batch
        valid_eps = self.completed_episodes
        if len(valid_eps) < self.batch_size:
            return 0.0

        batch_eps = random.sample(valid_eps, self.batch_size)
        # 計算最大 episode 長度
        seq_lengths = [len(ep) for ep in batch_eps]
        max_len = max(seq_lengths)
        # 將每個 episode pad 到相同長度，並將 state 轉為 numpy array
        padded_batch_seqs = []
        for ep in batch_eps:
            padded_ep = []
            L = len(ep)
            # 獲取 agent 數量和特徵維度
            first_state_list = ep[0][0]  # list[np.ndarray]
            A = len(first_state_list)
            feat = first_state_list[0].shape[0]
            state_shape = (A, feat)
            for t in range(max_len):
                if t < L:
                    step = ep[t]
                    # 將列表形式的 state/next_state 轉為 numpy array
                    state_arr = np.stack(step[0], axis=0)
                    next_state_arr = np.stack(step[1], axis=0)
                    action_list = step[2]
                    reward_arr = np.array(step[3], dtype=np.float32)
                    done_val = step[4]
                    padded_ep.append((state_arr, next_state_arr, action_list, reward_arr, done_val))
                else:
                    # pad 步驟
                    padded_state = np.zeros(state_shape, dtype=np.float32)
                    padded_next_state = np.zeros(state_shape, dtype=np.float32)
                    padded_action = [0] * A
                    padded_reward = np.zeros(A, dtype=np.float32)
                    padded_done = 1.0
                    padded_ep.append((padded_state, padded_next_state, padded_action, padded_reward, padded_done))
            padded_batch_seqs.append(padded_ep)
        batch_seqs = padded_batch_seqs

        # 解包成 numpy array，形狀 [T, B, ...]
        states      = np.stack([[step[0] for step in seq] for seq in batch_seqs], axis=1)
        next_states = np.stack([[step[1] for step in seq] for seq in batch_seqs], axis=1)
        actions     = np.stack([[step[2] for step in seq] for seq in batch_seqs], axis=1)
        rewards     = np.stack([[step[3] for step in seq] for seq in batch_seqs], axis=1)
        dones       = np.stack([[step[4] for step in seq] for seq in batch_seqs], axis=1)

        # 轉成 tensor
        states      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,     dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=self.device).unsqueeze(-1)  # 增加一個維度以匹配 [T, B, A]

        # 初始化 hidden，這裡將 batch_size 傳給 init_hidden
        mh0, wh0 = self.my_agent.init_hidden(batch_size=self.batch_size)

        # 正向計算
        q_values, _       = self.my_agent(states,      (mh0, wh0))  # [T, B, A, n_actions]
        with torch.no_grad():
            q_next, _    = self.target_my_agent(next_states, (mh0, wh0))

        # 計算 TD 目標
        q_taken    = q_values.gather(3, actions.unsqueeze(-1)).squeeze(-1)
        max_next_q = q_next.max(dim=3)[0]

        # 更新 TD 目標與損失計算
        td_target = rewards + (1 - dones) * self.gamma * max_next_q  # [T, B, A]
        mse = (q_taken - td_target).pow(2)
        loss = mse.mean()

        # # 檢查極端 loss，並列印 episode 訓練資料
        # loss_value = loss.item()
        # if loss_value > self.loss_threshold:
        #     self.logger.warning(f"Large loss: {loss_value:.4f} > threshold {self.loss_threshold}")
        #     pprint.pprint(batch_eps)
        #     # 將異常 episode 寫入文字檔
        #     with open(self.loss_log_file, 'a', encoding='utf-8') as f:
        #         f.write(f"=== Large loss: {loss_value:.4f} ===\n")
        #         f.write(pprint.pformat(batch_eps) + "\n\n")
        self.my_agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.my_agent.parameters(), max_norm=5)
        self.my_agent_optimizer.step()

        return loss.item()
    
    def reset(self):
        """重置遊戲狀態，準備開始新的episode"""
        # 關閉當前日志文件
        self._close_log_file()
        
        self.episode_init = True
        self.best_distance = 1000000
        self.worst_distance = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.episode_memory = []
        self.done = False
        if self.agent_type == 'GBV15':
            self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
        elif self.agent_type == 'GBV2':
            self.manager_hidden, self.worker_hidden = self.manager_agent.init_hidden(), self.worker_agent.init_hidden()
        elif self.agent_type == 'Feudal':
            self.manager_hidden, self.worker_hidden = self.manager_agent.init_hidden(), self.worker_agent.init_hidden()
        elif self.agent_type == 'DRQN':
            self.rnn_agent_hidden = self.rnn_agent.init_hidden()
        # 清空當前episode的記憶
        
        self.logger.info("重置遊戲狀態，準備開始新的episode")

        # 組合多個命令
        action_cmd = ""
        action_cmd = self.reset_cmd
        
        return action_cmd
    
    def get_reset_cmd(self, features: Multi_Side_FeaturesFromSteam):
        action_cmd = ""
        for ac in features.units[self.player_side]:
            action_cmd += delete_unit(
                side=self.player_side,
                unit_name=ac.Name
            ) + "\n"
            action_cmd += add_unit(
                type='Ship',
                unitname=ac.Name,
                dbid=ac.DBID,
                side=self.player_side,
                Lat=ac.Lat,
                Lon=ac.Lon
            ) + "\n"
        for ac in features.units[self.enemy_side]:
            action_cmd += delete_unit(
                side=self.enemy_side,
                unit_name=ac.Name
            ) + "\n"
            action_cmd += add_unit(
                type='Ship',
                unitname=ac.Name,
                dbid=ac.DBID,
                side=self.enemy_side,
                Lat=ac.Lat,
                Lon=ac.Lon
            ) + "\n"
        self.reset_cmd = action_cmd

    def big_loss_check(self, loss: float):
        # # 檢查極端 loss，並列印 episode 訓練資料
        # loss_value = loss.item()
        # if loss_value > self.loss_threshold:
        #     self.logger.warning(f"Large loss: {loss_value:.4f} > threshold {self.loss_threshold}")
        #     pprint.pprint(episode)
        #     # 將異常 episode 寫入文字檔
        #     with open(self.loss_log_file, 'a', encoding='utf-8') as f:
        #         f.write(f"=== Large loss: {loss_value:.4f} ===\n")
        #         f.write(pprint.pformat(episode) + "\n\n")
        # Optimize agent networks and mixer
        return
    
    def GBV15_train(self, states: torch.Tensor,
                            next_states: torch.Tensor,
                            actions_tensor: torch.Tensor,
                            rewards_tensor: torch.Tensor,
                            dones_tensor: torch.Tensor,
                            global_states: torch.Tensor, 
                            next_global_states: torch.Tensor):
        """訓練 GBV15 網路"""
        # Forward
        mh0, wh0 = self.my_agent.init_hidden()
        q_values, _ = self.my_agent(states, (mh0, wh0))  
        
        # [T,1,A,n_actions]
        with torch.no_grad():
            target_q_values, _ = self.target_my_agent(next_states, (mh0, wh0))  # [T,1,A,n_actions]
        # Gather current Q for taken actions
        current_q = q_values.gather(3, actions_tensor.unsqueeze(-1)).squeeze(-1)  # [T,1,A]
        
        # double Q
        if self.enable_double_q:
            with torch.no_grad():
                online_next_q, _ = self.my_agent(next_states, (mh0, wh0))
            best_actions = online_next_q.argmax(dim=3, keepdim=True)       # [T,1,A,1]
            max_next_q   = target_q_values.gather(3, best_actions).squeeze(-1)  # [T,1,A]
        else:
            max_next_q = target_q_values.max(dim=3)[0]  # [T,1,A]

        # mixer
        if self.enable_mixer:
            current_q = self.mixer(current_q, global_states)
            max_next_q = self.target_mixer(max_next_q, next_global_states)
            # 平均rewards
            rewards_tensor = rewards_tensor.mean(dim=2, keepdim=True)
        # else:
        #    # 一律使用平均的joint_reward
        #     rewards_tensor = rewards_tensor.mean(dim=2, keepdim=True)
        #     # 將 rewards_tensor 擴展為 [T,1,A]
        #     rewards_tensor = rewards_tensor.repeat(1, 1, self.args.n_agents)

        # TD target
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        mse = (current_q - target_q).pow(2)
        loss = mse.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=5)
        self.optimizer.step()

        return loss.item()

    def GBV2_train(self, states: torch.Tensor,
                            next_states: torch.Tensor,
                            actions_tensor: torch.Tensor,
                            rewards_tensor: torch.Tensor,
                            dones_tensor: torch.Tensor,
                            global_states: torch.Tensor, 
                            next_global_states: torch.Tensor):
        """訓練 GBV2 網路"""

        # manager
        # Critic 返回 [T,B,A,1]，squeeze 最後一維到 [T,B,A]
        values_M = self.manager_critic(states).squeeze(-1)
        values_M_next = self.manager_critic(next_states).squeeze(-1)

        # GAE
        # rewards_tensor [T,B,A], dones_tensor [T,B,1], values_M_next [T,B,A]
        
        # 修復：為 Manager 設計專用獎勵，關注長期戰略目標
        # Manager 關心：整體進度、生存率、協調性
        # 使用滑動平均來平滑短期波動，關注長期趨勢
        window_size = min(5, rewards_tensor.size(0))
        if rewards_tensor.size(0) >= window_size:
            # 計算滑動平均獎勵，讓 Manager 關注長期趨勢
            manager_rewards = torch.zeros_like(rewards_tensor)
            for t in range(rewards_tensor.size(0)):
                start_idx = max(0, t - window_size + 1)
                manager_rewards[t] = rewards_tensor[start_idx:t+1].mean(dim=0, keepdim=True)
        else:
            manager_rewards = rewards_tensor
            
        deltas_M = manager_rewards + (1 - dones_tensor) * self.gamma * values_M_next - values_M
        advantage_M = torch.zeros_like(rewards_tensor, device=self.device)
        last_adv_M = 0.0
        for t in reversed(range(len(rewards_tensor))):
            last_adv_M = deltas_M[t] + self.gamma * 0.95 * (1 - dones_tensor[t]) * last_adv_M
            advantage_M[t] = last_adv_M
        
        # 正規化
        advantage_M = (advantage_M - advantage_M.mean()) / (advantage_M.std() + 1e-8)

        # Manager 策略梯度與 Critic 更新
        mh0 = self.manager_agent.init_hidden(batch_size=1)
        b_values, _ = self.manager_agent(states, mh0)  # [T, B*A, goal_dim]
        # 重新形狀到 [T, B, A, goal_dim]
        T, B, A, _ = states.shape
        goal_dim = b_values.size(-1)
        b_values = b_values.view(T, B, A, goal_dim)
        # 計算選中目標的 log-prob
        # 修復：使用與執行時一致的確定性策略
        # 執行時我們直接使用 softmax 輸出，所以這裡應該用連續的 goal
        
        # 方法1：直接使用 softmax 輸出作為連續 goal，不進行離散化
        # 這樣就避免了策略梯度的離散化問題
        continuous_goal = b_values  # [T, B, A, goal_dim]
        
        # 策略損失：鼓勵 goal 朝著高 advantage 方向發展
        # 使用 advantage 作為權重，直接優化 goal 分佈
        policy_loss = -(advantage_M.detach() * continuous_goal.mean(dim=-1)).mean()
        
        # 或者方法2：如果一定要用離散策略，至少要與執行一致
        # 使用最大概率的動作（類似 greedy 執行）
        # goal_actions = b_values.argmax(dim=-1)  # [T, B, A]
        # dist = torch.distributions.Categorical(b_values.view(-1, goal_dim))
        # m_log_probs = dist.log_prob(goal_actions.view(-1)).view(T, B, A)
        # policy_loss = -(advantage_M.detach() * m_log_probs).mean()
        # 價值損失: 返回與 values_M 同形狀 [T,B,A]
        returns_M = advantage_M.detach() + values_M  # [T, B, A]
        value_loss = F.mse_loss(values_M, returns_M)
        loss_M = policy_loss + value_loss

        # 更新 manager 參數
        self.manager_optimizer.zero_grad()
        loss_M.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_agent.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(self.manager_critic.parameters(), max_norm=5)
        self.manager_optimizer.step()

        # Worker Q-learning 更新
        wh0 = self.worker_agent.init_hidden(batch_size=1)
        q_values_w, _ = self.worker_agent(states, wh0, b_values.detach().view(T, B*A, goal_dim))

        
        with torch.no_grad():
            b_next, _ = self.manager_agent(next_states, mh0)
            q_next_w, _ = self.target_worker_agent(next_states, wh0, b_next)
        q_taken = q_values_w.gather(3, actions_tensor.unsqueeze(-1)).squeeze(-1)  # [T, B, A]

        # double Q
        if self.enable_double_q:
            with torch.no_grad():
                online_next_q_w, _ = self.worker_agent(next_states, wh0, b_next)
            best_actions = online_next_q_w.argmax(dim=3, keepdim=True)       # [T, B, A, 1]
            max_next_q_w = q_next_w.gather(3, best_actions).squeeze(-1)  # [T, B, A]
        else:
            max_next_q_w = q_next_w.max(dim=3)[0]  # [T, B, A]

        # TD target
        target_q_w = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_w


        loss_W = (q_taken - target_q_w.detach()).pow(2).mean()

        # 更新 worker 參數
        self.worker_optimizer.zero_grad()
        loss_W.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_agent.parameters(), max_norm=5)
        self.worker_optimizer.step()

        # 返回總損失
        return (loss_M + loss_W).item()

    def _create_new_log_file(self):
        """創建新的日志文件用於記錄state和action"""
        # 關閉之前的文件（如果有的話）
        self._close_log_file()
        
        # 生成文件名：包含episode編號和時間戳
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"episode_{self.episode_count:05d}_{timestamp}.txt"
        self.current_log_file = os.path.join(self.logs_dir, filename)
        
        try:
            self.log_file_handle = open(self.current_log_file, 'w', encoding='utf-8')
            # 寫入文件頭信息
            self.log_file_handle.write(f"=== Episode {self.episode_count} State-Action Log ===\n")
            self.log_file_handle.write(f"Agent Type: {self.agent_type}\n")
            self.log_file_handle.write(f"Player Side: {self.player_side}\n")
            self.log_file_handle.write(f"Enemy Side: {self.enemy_side}\n")
            self.log_file_handle.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file_handle.write("=" * 60 + "\n\n")
            self.log_file_handle.flush()
            self.logger.info(f"創建新的日志文件: {self.current_log_file}")
        except Exception as e:
            self.logger.error(f"創建日志文件時發生錯誤: {e}")
            self.log_file_handle = None

    def _log_state_action(self, states: list[np.ndarray], actions: list[int], step: int):
        """記錄state和action到文件"""
        if self.log_file_handle is None:
            return
            
        try:
            self.log_file_handle.write(f"Step {step}:\n")
            self.log_file_handle.write("-" * 40 + "\n")
            
            # 記錄每個agent的state和action
            for idx, (state, action) in enumerate(zip(states, actions)):
                agent_name = self.friendly_info.order[idx] if idx < len(self.friendly_info.order) else f"Agent_{idx}"
                self.log_file_handle.write(f"Agent {idx} ({agent_name}):\n")
                
                # 格式化state向量，每行最多8個數值
                state_str = "  State: ["
                for i, val in enumerate(state):
                    if i > 0 and i % 8 == 0:
                        state_str += "\n         "
                    state_str += f"{val:8.4f}, "
                state_str = state_str.rstrip(", ") + "]\n"
                self.log_file_handle.write(state_str)
                
                # 記錄action
                action_names = ["前進", "左轉", "右轉", "攻擊"]
                action_name = action_names[action] if 0 <= action < len(action_names) else f"未知動作({action})"
                self.log_file_handle.write(f"  Action: {action} ({action_name})\n")
                self.log_file_handle.write("\n")
            
            self.log_file_handle.write("\n")
            self.log_file_handle.flush()
            
        except Exception as e:
            self.logger.error(f"記錄state-action時發生錯誤: {e}")

    def _close_log_file(self):
        """關閉當前日志文件"""
        if self.log_file_handle is not None:
            try:
                self.log_file_handle.write(f"\n=== Episode End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                self.log_file_handle.close()
                self.logger.info(f"已關閉日志文件: {self.current_log_file}")
            except Exception as e:
                self.logger.error(f"關閉日志文件時發生錯誤: {e}")
            finally:
                self.log_file_handle = None
                self.current_log_file = None



