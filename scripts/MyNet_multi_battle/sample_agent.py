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

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy   

import logging

# 导入模型
from module.batch_agent.GBV15_agent import my_Agent
from module.batch_agent.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic
from module.batch_agent.DRQN_agent import RNN_Agent
from module.batch_agent.DQN_agent import DQN_Agent

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

    def __init__(self, player_side: str, enemy_side: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param ac_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        self.player_side = player_side
        self.enemy_side = enemy_side
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent")
        self.logger.setLevel(logging.INFO)
        
        # 初始化 Mylib 並傳入 self
        self.mylib = Mylib(self)
        
        # FeUdal网络参数
        class Args:
            def __init__(self):
                self.hidden_dim = 64
                self.n_agents = 3
                self.enemy_num = 3
                self.input_size = 7 + 5 * (self.n_agents-1) + 4 * self.enemy_num  # [相對X, 相對Y, 敵人是否存在, 敵人存活比率, 敵人位置, 彈藥比率]
                self.n_actions = 4  # 前進、左轉、右轉、攻擊
                self.goal_dim = 4
                

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
        self.sequence_len = 16    # T
        # self.train_interval = 50        # 每隔多少 steps 學習一次
        self.update_freq = 10000        # 每隔多少 steps 同步 target network
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

        # ===============================MYNET========================================
        # 建立 policy_net 與 target_net
        self.my_agent = my_Agent(self.input_size, self.args).to(self.device)
        self.target_my_agent = deepcopy(self.my_agent)
        self.target_my_agent.eval() # 設置為評估模式
        self.my_agent_optimizer = torch.optim.Adam(self.my_agent.parameters(), lr=self.lr)
        # self.manager_agent = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
        # self.worker_agent = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
        # self.critic = FeUdalCritic(self.input_size, self.args).to(self.device)

        # self.rnn_agent = RNN_Agent(self.input_size, self.args).to(self.device)
        # self.dqn_agent = DQN_Agent(self.input_size, self.args).to(self.device)

        # 初始化隐藏状态
        self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
        # self.manager_hidden = self.manager_agent.init_hidden()
        # self.worker_hidden = self.worker_agent.init_hidden()
        # self.rnn_agent_hidden = self.rnn_agent.init_hidden()

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
            rewards = self.get_rewards(features, self.prev_state, current_state, features.sides_[self.player_side].TotalScore)
            # distance = self.get_distance(current_state)
            # reward = 1
            distance = 1
            
            # done = False
            # 将 rewards 列表转换为 numpy 数组并计算平均值
            rewards_arr = np.array(rewards, dtype=np.float32)
            avg_reward = rewards_arr.mean() if rewards_arr.size > 0 else 0.0
            self.total_reward += avg_reward
            self.episode_reward += avg_reward
            
            # 將經驗添加到當前episode的記憶中
            self.episode_memory.append((self.prev_state, current_state, self.prev_action, rewards, self.done, self.alive))
            
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
                loss = 0
                for _ in range(self.batch_size):
                    episode_loss = self.train()
                    loss = loss + episode_loss
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
            q_values, (self.manager_hidden, self.worker_hidden) = \
                self.my_agent(state_tensor, (self.manager_hidden, self.worker_hidden))
            # Manager生成目标
            # _, goal, self.manager_hidden = self.manager_agent(state_tensor, self.manager_hidden)
            
            # # Worker根据目标选择动作
            # q_values, self.worker_hidden = self.worker_agent(
            #     state_tensor, 
            #     self.worker_hidden,
            #     goal
            # )
            # critic_value = self.critic(state_tensor)
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                # q_values, self.rnn_agent_hidden = self.rnn_agent(state_tensor, self.rnn_agent_hidden)
            # q_values = self.dqn_agent(state_tensor)
        dt_rl = time.perf_counter() - t0_rl
        # self.step_times_rl.append(dt_rl)
        # if dt_rl > 0.01:
        #     print(prof.key_averages().table(sort_by="cuda_time_total"))
        
        # 檢查是否有敵人
        has_enemy = len(features.contacts[self.player_side]) > 0

        # q_values shape: [T, B, A, n_actions]
        # 取出第一個時間步＆batch
        q_vals = q_values[0, 0]            # shape [A, n_actions]
        A, n_actions = q_vals.shape
        # 產生 action_mask: 限制在以目標點為左上角，邊長 self.max_distance 的正方形內行動
        masks = []
        for i in range(A):
            # 從當前狀態列表取得第 i 個 agent 的 dx_norm, dy_norm
            # dx = current_state[i][0] * self.max_distance
            # dy = current_state[i][1] * self.max_distance
            mask = torch.ones(n_actions, dtype=torch.bool, device=self.device)
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
            if not has_enemy:
                mask[3] = False
            # 無彈藥時禁止攻擊
            mount_ratio = current_state[i][5]
            if mount_ratio <= 0:
                mask[3] = False
            masks.append(mask)
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
        # 線性更新
        self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * self.total_steps / self.eps_decay_steps
        self.epsilon = max(self.eps_end, self.epsilon)
        self.logger.debug(f"更新epsilon: {self.epsilon:.4f}")

        if self.total_steps % self.update_freq == 0:


            self.target_my_agent.load_state_dict(self.my_agent.state_dict())
        
        # 儲存模型
        # if self.total_steps % 1000 == 0:
        #     torch.save(self.my_agent.state_dict(), f'models/my_agent_{self.total_steps}.pth')
        
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray, score: int) -> np.ndarray:
        # 計算全局 reward
        reward = 0
        
        # 場景score
        current_score = score
        # 場景score變化
        score_change = current_score - self.prev_score
        self.prev_score = current_score
        # 場景score變化獎勵
        reward += score_change
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

        if next_state[0] < self.done_condition and self.done:
            win_reward = self.win_reward * (1- (self.episode_step - self.min_episode_steps) / (self.max_episode_steps - self.min_episode_steps))
            win_reward = max(win_reward, self.min_win_reward)
            reward += win_reward

        # 原始獎勵
        raw_reward = reward
        # 獲勝獎勵200 + 敵軍總數 7 *擊殺獎勵 20 + 最大距離獎勵 200*7
        max_return = self.win_reward + self.enemy_info.initial_enemy_count * 20 +  100
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
                reward = self.get_reward(state[i], next_state[i], score)
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
        # 隨機選一個已完成 episode
        episode = random.choice(self.completed_episodes)
        # 解包批次: (states, next_states, actions_list, reward, done, alive_mask)
        states, next_states, actions_list, rewards, dones, alive_masks = zip(*episode)
        # 張量轉換: states [T,A,feat] -> [T,1,A,feat]
        states = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device).unsqueeze(1)
        # actions_list [T,A] -> [T,1,A]
        actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=self.device).unsqueeze(1)
        # rewards [T,A] -> [T,1,A]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        # dones [T] -> [T,1,1]
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1,1,1)
        # alive_masks [T,A] -> [T,1,A]
        alive_masks = torch.tensor(alive_masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        #----------------------------- Calculate loss -----------------------------------
        # Forward
        mh0, wh0 = self.my_agent.init_hidden()
        q_values, _ = self.my_agent(states, (mh0, wh0))  
        
        # [T,1,A,n_actions]
        with torch.no_grad():
            target_q_values, _ = self.target_my_agent(next_states, (mh0, wh0))  # [T,1,A,n_actions]
        # Gather current Q for taken actions
        current_q = q_values.gather(3, actions_tensor.unsqueeze(-1)).squeeze(-1)  # [T,1,A]
        # Max next Q
        max_next_q = target_q_values.max(dim=3)[0]  # [T,1,A]
        # TD target
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        current_q = current_q * alive_masks
        target_q = target_q * alive_masks

        mse = (current_q - target_q).pow(2)
        loss = mse.sum() / alive_masks.sum().clamp(min=1.0)

        # 檢查極端 loss，並列印 episode 訓練資料
        loss_value = loss.item()
        if loss_value > self.loss_threshold:
            self.logger.warning(f"Large loss: {loss_value:.4f} > threshold {self.loss_threshold}")
            pprint.pprint(episode)
            # 將異常 episode 寫入文字檔
            with open(self.loss_log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== Large loss: {loss_value:.4f} ===\n")
                f.write(pprint.pformat(episode) + "\n\n")
        self.my_agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.my_agent.parameters(), max_norm=5)
        self.my_agent_optimizer.step()

        return loss.item()


        return loss.item()
    
    def reset(self):
        """重置遊戲狀態，準備開始新的episode"""
        self.episode_init = True
        self.best_distance = 1000000
        self.worst_distance = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
        # 清空當前episode的記憶
        self.episode_memory = []
        self.done = False
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
