from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed, manual_attack_contact, delete_unit, add_unit, set_unit_to_mission, auto_attack_contact
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import Multi_Side_FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
import numpy as np
from collections import deque
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy   

import logging

# 导入FeUdal模型
from scripts.MyNet_multi_battle.GB_agent import my_Agent
from scripts.MyNet_multi_battle.FeUdal_agent import Feudal_ManagerAgent, Feudal_WorkerAgent, FeUdalCritic
from scripts.MyNet_multi_battle.DRQN_agent import RNN_Agent
from scripts.MyNet_multi_battle.DQN_agent import DQN_Agent

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
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
    def __init__(self, player_side: str, ac_name: str, target_name: str = None):
        """
        初始化 Agent。
        :param player_side: 玩家所屬陣營
        :param ac_name: 控制的單位名稱（例如 B 船）
        :param target_name: 目標單位名稱（例如 A 船），可選
        :param log_level: 日誌級別，預設為INFO，可設置為logging.DEBUG啟用詳細日誌
        """
        super().__init__(player_side)
        self.ac_name = ac_name
        self.target_name = target_name  # 用於追蹤特定目標（例如 A 船）
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(f"MyAgent_{ac_name}")
        self.logger.setLevel(logging.INFO)
        
        
        # FeUdal网络参数
        class Args:
            def __init__(self):
                self.hidden_dim = 64
                self.input_size = 5  # [相對X, 相對Y, 相對方向sin, 相對方向cos, 敵人是否存在]
                self.n_actions = 5  # 上下左右、攻擊
                self.goal_dim = 3
                self.n_agents = 7

                self.manager_hidden_dim = 64
                self.worker_hidden_dim = 64
                self.state_dim_d = 3
                self.embedding_dim_k = 16

        self.args = Args()
        self.input_size = 5  # [相對X, 相對Y, 相對方向sin, 相對方向cos, 敵人是否存在]
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        # ===============================MYNET========================================
        # 建立 policy_net 與 target_net
        # self.my_agent = my_Agent(self.input_size, self.args).to(self.device)
        # self.target_my_agent = deepcopy(self.my_agent)
        # self.target_my_agent.eval() # 設置為評估模式
        # self.my_agent_optimizer = torch.optim.Adam(self.my_agent.parameters(), lr=3e-4)
        # self.manager_agent = Feudal_ManagerAgent(self.input_size, self.args).to(self.device)
        # self.worker_agent = Feudal_WorkerAgent(self.input_size, self.args).to(self.device)
        # self.critic = FeUdalCritic(self.input_size, self.args).to(self.device)

        # self.rnn_agent = RNN_Agent(self.input_size, self.args).to(self.device)
        self.dqn_agent = DQN_Agent(self.input_size, self.args).to(self.device)



        
        # 修改記憶存儲方式，使用列表存儲完整的episode
        self.episode_memory = []  # 存儲當前episode的經驗
        self.completed_episodes = []  # 存儲已完成的episodes
        self.max_episodes = 32  # 最多保存的episode數量


        # 基本超參數
        self.gamma = 0.99
        self.lr = 5e-4
        self.batch_size = 256
        self.train_interval = 50        # 每隔多少 steps 學習一次
        self.update_freq = 10000        # 每隔多少 steps 同步 target network
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_steps = 100000
        
        self.done_condition = 0.1

        # 初始化隐藏状态
        # self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
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

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()
        
        # 添加遊戲結束標記
        self.episode_init = True
        self.episode_step = 0
        self.episode_count = 0
        self.max_episode_steps = 2000
        self.episode_done = False
        self.episode_reward = 0

        self.step_times_rl = []
        self.reset_cmd = ""

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
    def get_distance(self, state):
        """计算智能体与目标之间的距离，支持NumPy数组和PyTorch张量"""
        if isinstance(state, torch.Tensor):
            # 使用PyTorch操作
            return torch.sqrt((state[0])**2 + (state[1])**2)
        else:
            # 使用NumPy操作
            return np.sqrt((state[0])**2 + (state[1])**2)
    def get_done(self, state):
        done = False
        distance = self.get_distance(state)
        # 如果距離大於0.5公里，則認為達到目標
        if abs(state[0]) > 1:
            done = True
        if abs(state[1]) > 1:
            done = True
        if distance < self.done_condition:
            done = True
        return done
            
    def action(self, features: Multi_Side_FeaturesFromSteam) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        if self.episode_init:
            self.episode_init = False
            self.episode_count += 1
            self.logger.info(f"episode: {self.episode_count}")
            time.sleep(0.1)
            

        if features.sides_[self.player_side].TotalScore == 0:
            self.prev_score = 0
        print("total_step:", self.total_steps)
        self.logger.debug("開始執行動作")
        action = ""
        for unit in features.units[self.player_side]:
            has_unit = True
        if not has_unit:
            self.logger.warning(f"找不到單位: {self.ac_name}")
            return self.reset()  # 如果找不到單位，返回初始化
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        current_state = self.get_states(features)
        
        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            # reward = self.get_reward(self.prev_state, current_state, features.sides_[self.player_side].TotalScore)
            # distance = self.get_distance(current_state)
            reward = 1
            distance = 1
            # done = self.get_done(current_state)
            done = False
            self.total_reward += reward
            self.episode_reward += reward
            
            # 將經驗添加到當前episode的記憶中
            self.episode_memory.append((self.prev_state, current_state, self.prev_action, reward, done))
            
            # 檢查遊戲是否結束
            if done or self.episode_step > self.max_episode_steps:
                self.episode_done = True
                self.logger.info(f"遊戲結束! 總獎勵: {self.episode_reward:.4f}")
                
                # 將完成的episode添加到已完成episodes列表中
                if len(self.episode_memory) > 0:
                    self.completed_episodes.append(self.episode_memory)
                    # 限制已完成的episodes數量
                    if len(self.completed_episodes) > self.max_episodes:
                        self.completed_episodes.pop(0)
                
                # # 在遊戲結束時進行訓練
                # loss = 0
                # for _ in range(10):
                #     episode_loss = self.train()
                #     loss = loss + episode_loss
                # loss = loss / 10
                # # 重置遊戲狀態
                # if self.episode_count % 5 == 0:
                #     self.logger.info(f"步數: {self.total_steps}, 距離: {distance:.4f}, 損失: {loss:.4f}, 總獎勵: {self.total_reward:.4f}")
                #     self.stats_logger.log_stat("distance", distance, self.total_steps)
                #     self.stats_logger.log_stat("best_distance", self.best_distance, self.total_steps)
                #     self.stats_logger.log_stat("loss", loss, self.total_steps)
                #     self.stats_logger.log_stat("episode_return", self.episode_reward, self.total_steps)
                return self.reset()
            
        
        
        # 選擇動作
        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)   # → [1, Agent, feat]
        state_tensor = state_tensor.unsqueeze(1)       # → [seq_len=1, batch=1, Agent, feat]
        t0_rl = time.perf_counter()
        with torch.no_grad():
            # q_values, (self.manager_hidden, self.worker_hidden) = \
            #     self.my_agent(state_tensor, (self.manager_hidden, self.worker_hidden))
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
            q_values = self.dqn_agent(state_tensor)
        dt_rl = time.perf_counter() - t0_rl
        self.step_times_rl.append(dt_rl)
        # if dt_rl > 0.01:
        #     print(prof.key_averages().table(sort_by="cuda_time_total"))
        
        # 檢查是否有敵人
        has_enemy = len(features.contacts[self.player_side]) > 0

        # q_values shape: [T, B, A, n_actions]
        # 取出第一個時間步＆batch
        q_vals = q_values[0, 0]            # shape [A, n_actions]
        A, n_actions = q_vals.shape
        actions = []
        action_cmd = ""

        for ai in range(A):
            q_agent = q_vals[ai]           # shape [n_actions]
            if random.random() < self.epsilon:
                # 隨機策略
                if has_enemy:
                    act = random.randint(0, n_actions - 1)
                else:
                    act = random.randint(0, 3)  # 只挑移動動作
                self.logger.debug(f"Agent {ai} 隨機選擇動作: {act}")
            else:
                # ε-greedy
                act = int(q_agent.argmax().item())
                # 如果選到攻擊但沒敵人，排除再選
                if act == 4 and not has_enemy:
                    q_agent_masked = q_agent.clone()
                    q_agent_masked[4] = -float('inf')
                    act = int(q_agent_masked.argmax().item())
                self.logger.debug(f"Agent {ai} 根據Q值選擇動作: {act}")
            actions.append(act)

        # 對應到每個 unit 執行動作
        processed_units = set()  # 用於追蹤已經處理過的單位
        action_cmd = ""
        
        for ai, unit in enumerate(features.units[self.player_side]):
            # 檢查單位是否已經被處理過
            if unit.Name in processed_units:
                continue
            
            action_cmd += "\n" + self.apply_action(actions[ai], unit, features)
            processed_units.add(unit.Name)  # 將處理過的單位名稱加入集合


        # if self.episode_step < 10:
        #     action_cmd += "\n" + set_unit_to_mission(
        #         unit_name=ac.Name,
        #         mission_name='test'
        #     )
        self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = action
        self.total_steps += 1
        self.episode_step += 1

        # 更新 epsilon
        # 線性更新
        self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * self.total_steps / self.eps_decay_steps
        self.epsilon = max(0.05, self.epsilon)
        self.logger.debug(f"更新epsilon: {self.epsilon:.4f}")

        if self.total_steps % self.update_freq == 0:


            self.target_my_agent.load_state_dict(self.my_agent.state_dict())
        
        # 儲存模型
        # if self.total_steps % 1000 == 0:
        #     torch.save(self.my_agent.state_dict(), f'models/my_agent_{self.total_steps}.pth')
        
        return action_cmd
    
    def normalize_state(self, state):
        max_distanceX = 100  # 根據你的地圖範圍調整
        max_distanceY = 100
        norm_state = np.zeros_like(state)
        norm_state[0] = state[0] / max_distanceX  # 相對X
        norm_state[1] = state[1] / max_distanceY  # 相對Y
        norm_state[2] = state[2]  # 角度sin
        norm_state[3] = state[3]  # 角度cos
        norm_state[4] = state[4]  # 敵人是否存在
        return norm_state

    def get_state(self, features: Multi_Side_FeaturesFromSteam, ac: Unit) -> np.ndarray:
        """
        獲取當前狀態向量，包含自身單位和目標的相對位置資訊。
        :return: numpy 陣列，例如 [相對X, 相對Y, 相對方向sin, 相對方向cos, 敵人是否存在]
        """
        # 獲取自身單位（B 船）的資訊
        if ac is None:
            # 如果找不到單位，返回預設狀態
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        target_lon = float(118.27954108343)
        target_lat = float(24.333113806906)
        
        # 計算相對位置
        ac_lon = float(ac.Lon)
        ac_lat = float(ac.Lat)
        
        # 計算相對座標 (X,Y)，將經緯度差轉換為大致的平面座標
        # 注意：這是簡化的轉換，對於小範圍有效
        # X正方向為東，Y正方向為北
        earth_radius = 6371  # 地球半徑（公里）
        lon_scale = np.cos(np.radians(ac_lat))  # 經度在當前緯度的縮放因子
        
        # 1. 計算相對 X 和 Y（公里）
        dx = (target_lon - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
        dy = (target_lat - ac_lat) * np.pi * earth_radius / 180.0
        
        # 計算兩船的相對方向（不考慮船頭方向）
        # 2. 計算相對方向（弧度，範圍 [-π, π]）
        relative_angle = np.arctan2(dy, dx) 
        # 3. 用 sin/cos 編碼角度，消除「0↔360°」不連續問題
        sin_angle = np.sin(relative_angle)
        cos_angle = np.cos(relative_angle)

        # 檢查是否存在敵人
        enemy_found = 0
        for enemy in features.contacts[self.player_side]:
            enemy_found = 1
        
        # 構建新的狀態向量：[相對X, 相對Y, 相對方向sin, 相對方向cos, 敵人是否存在(0-1)]
        raw_state = np.array([dx, dy, sin_angle, cos_angle, enemy_found])
        normalized_state = self.normalize_state(raw_state)
        # print("raw_state:", raw_state)
        # print("normalized_state:", normalized_state)
        
        # 返回狀態向量
        return normalized_state
    
    def get_states(self, features: Multi_Side_FeaturesFromSteam) -> list[np.ndarray]:
        states = []
        processed_units = set()  # 用於追蹤已經處理過的單位
        
        for ac in features.units[self.player_side]:
            # 檢查單位是否已經被處理過
            if ac.Name in processed_units:
                continue
            
            # print(f"ac: {ac.ID}")
            # print(f"ac: {ac.Name}")
            states.append(self.get_state(features, ac))
            processed_units.add(ac.Name)  # 將處理過的單位ID加入集合
        
        return states
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray, score: int) -> float:
        reward = 0

        # # 場景score
        # current_score = score
        # # 場景score變化
        # score_change = current_score - self.prev_score
        # self.prev_score = current_score
        # # 場景score變化獎勵
        # reward += score_change

        # distance = self.get_distance(state)
        # next_distance = self.get_distance(next_state)
        
        # self.logger.debug(f"距離變化: {distance:.4f} -> {next_distance:.4f}")
            
        # reward += 100 * (distance-next_distance)
        # # # print(f"Reward: {reward}")
        # if next_distance + 0.01 < self.best_distance:
        #     self.best_distance = next_distance   #如果當前距離比最佳距離近0.1公里 換他當最佳距離 然後給獎勵
        #     reward += 1
        #     self.logger.debug(f"新的最佳距離: {self.best_distance:.4f}")
        # if next_distance - 0.01 > self.best_distance:
        #     self.worst_distance = next_distance
        #     reward -= 1
        #     self.logger.debug(f"新的最差距離: {self.worst_distance:.4f}")
        # if next_distance < self.done_condition:
        #     reward += 200
        # # if next_distance < 0.2:
        # #     reward += 0.25
        # # elif next_distance < 0.3:
        # #     reward += 0.2
        # # elif next_distance < 0.4:
        # #     reward += 0.15
        # # elif next_distance < 0.5:
        # #     reward += 0.1
        # # 超出範圍懲罰
        # if abs(next_state[0]) > 1:
        #     reward -= 10
        # if abs(next_state[1]) > 1:
        #     reward -= 10
        reward -= 0.01
        self.logger.debug(f"獎勵: {reward:.4f}")
            
        return reward

    def apply_action(self, action: int, ac: Unit, features: Multi_Side_FeaturesFromSteam) -> str:
        """將動作轉換為 CMO 命令"""
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0:  # 上
            heading = 0
        elif action == 1:  # 下
            heading = 180
        elif action == 2:  # 左
            heading = 270
        elif action == 3:  # 右
            heading = 90
        elif action == 4:  # 攻擊
            for enemy in features.contacts[self.player_side]:
                return auto_attack_contact(
                        attacker_id=ac.ID,
                        contact_id=enemy['ID'],
                )
                # return manual_attack_contact(
                #         attacker_id=self.ac_name,
                #         contact_id=enemy['ID'],
                #         weapon_id=1133,
                #         qty=1
                # )
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=ac.Name,
            heading=heading,
            speed=30
        )

    def train(self):
        """訓練MyNet網路"""
        # 檢查是否有足夠的episodes進行訓練
        if len(self.completed_episodes) < 1:
            self.logger.warning("沒有足夠的episodes進行訓練")
            return
            
        # 從已完成的episodes中隨機選擇一個episode
        episode = random.choice(self.completed_episodes)
        
        batch = episode
            
        # 解包批次數據
        states, next_states, actions, rewards, dones = zip(*batch)
        
        # 轉換為張量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 獲取實際的批次大小
        actual_batch_size = states.size(0)

        #----------------------------- Calculate loss -----------------------------------
        
        current_q = []
        target_q = []
        mh0, wh0 = self.my_agent.init_hidden()
        # states shape: [T, feat]  ->  [T, 1, feat]  (seq_len, batch, feat)
        seq = states.unsqueeze(1)
        q_seq, _ = self.my_agent(seq, (mh0, wh0))     # q_seq: [T, 1, n_actions]

        # 取動作 Q(s,a)
        act_idx = actions.view(actual_batch_size, 1, 1)               # [T, 1, 1]
        current_q = q_seq.gather(2, act_idx).squeeze(2)   # [T, 1] -> 再 squeeze
        

        mh1, wh1 = self.my_agent.init_hidden()
        with torch.no_grad():
            next_q_seq, _ = self.target_my_agent(next_states.unsqueeze(1), (mh1, wh1))
            # max over actions dim=2, keepdim=False -> [T,1]
            target_q = rewards + (1-dones)*self.gamma*next_q_seq.max(2)[0]
        self.logger.debug("forward完成")
        
        self.my_agent_optimizer.zero_grad()
        loss = F.mse_loss(current_q, target_q.detach())
        loss = torch.clamp(loss, max=10.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.my_agent.parameters(), max_norm=5)
        self.my_agent_optimizer.step()
         # 記錄訓練統計數據  
        
        # 每10步記錄一次
        # if self.total_steps % 10 == 0:

        self.logger.debug("訓練完成")

        # if self.total_steps % self.update_freq == 0:
        #     self.manager_hidden = self.manager.init_hidden()
        #     self.worker_hidden = self.worker.init_hidden()
        #     self.logger.debug(f"更新FeUdal網路，步數: {self.total_steps}")
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
        # self.manager_hidden, self.worker_hidden = self.my_agent.init_hidden()
        # 清空當前episode的記憶
        self.episode_memory = []
        self.logger.info("重置遊戲狀態，準備開始新的episode")

        # 組合多個命令
        action_cmd = ""
        # # 刪除舊單位
        # action_cmd += delete_unit(
        #     side=self.player_side,
        #     unit_name='1101 Cheng Kung [Perry Class, Kuang Hua I]'
        # ) + "\n"
        # action_cmd += add_unit(
        #     type='Ship',
        #     unitname='1101 Cheng Kung [Perry Class, Kuang Hua I]',
        #     dbid=649,
        #     side=self.player_side,
        #     # Lat=23.578745387803,
        #     # Lon=119.41307176516
        #     Lat=23.90,
        #     Lon=118.75
        # ) + "\n"
        # action_cmd += delete_unit(
        #     side='C',
        #     unit_name='Type 056A Jiangdao II [593 Sanmenxia]'
        # ) + "\n"
        # action_cmd += add_unit(
        #     type='Ship',
        #     unitname='Type 056A Jiangdao II [593 Sanmenxia]',
        #     dbid=4720,
        #     side='C',
        #     Lat=24.384352428685,
        #     Lon=118.46657514864
        # ) + "\n"
        # action_cmd += set_unit_to_mission(
        #     unit_name='Type 056A Jiangdao II [593 Sanmenxia]',
        #     mission_name='test'
        # )
        print(f"reset_cmd: {self.reset_cmd}")
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
        for ac in features.units['C']:
            action_cmd += delete_unit(
                side='C',
                unit_name=ac.Name
            ) + "\n"
            action_cmd += add_unit(
                type='Ship',
                unitname=ac.Name,
                dbid=ac.DBID,
                side='C',
                Lat=ac.Lat,
                Lon=ac.Lon
            ) + "\n"
        self.reset_cmd = action_cmd


def compute_returns(rewards, gamma, values, terminated): 
    """
    rewards: 奖励张量 [batch_size]
    gamma: 折扣因子
    values: 价值估计 [batch_size]
    terminated: 终止标志 [batch_size]
    返回: returns张量 [batch_size]
    """
    # 在您的代码中，这些都是一维张量，没有序列长度维度
    batch_size = rewards.shape[0]
    
    # 预分配returns张量
    returns = torch.zeros_like(values)
    
    # 计算每个样本的回报
    for b in range(batch_size):
        # 如果终止，则回报就是奖励
        if terminated[b]:
            returns[b] = rewards[b]
        else:
            # 否则，回报是奖励加上折扣的价值估计
            returns[b] = rewards[b] + gamma * values[b]
            
    return returns