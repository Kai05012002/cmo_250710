from pycmo.lib.actions import AvailableFunctions, set_unit_position, set_unit_heading_and_speed
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
import numpy as np
from collections import deque

import random

import torch
import torch.nn as nn

import logging

# DQN 模型
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        

        # DQN 參數
        self.input_size = 6  # [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        self.hidden_size = 64
        self.num_actions = 4  # 上下左右
        
        # 檢查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.logger.info(f"使用設備: {self.device}")
        
        self.model = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        self.target_model = DQN(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1
        self.gamma = 0.99
        self.update_freq = 100
        self.steps = 0
        self.done_condition = 0.005

        self.best_distance = 1000000
        self.total_reward = 0

        # 用於解決 next_state 延遲的屬性
        self.prev_state = None
        self.prev_action = None
        self.prev_features = None

        # 初始化訓練統計記錄器（如果有的話）
        self.stats_logger = Logger()

    def get_unit_info_from_observation(self, features: FeaturesFromSteam, unit_name: str) -> Unit:
        """
        從觀察中獲取指定單位的資訊。
        """
        units = features.units
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None
    
    def get_contact_info_from_observation(self, features: FeaturesFromSteam, contact_name: str) -> dict:
        """
        從觀察中獲取指定接觸點（敵方單位）的資訊。
        """
        contacts = features.contacts
        for contact in contacts:
            if contact['Name'] == contact_name:
                return contact
        return None
    def get_distance(self, state: np.ndarray) -> float:
        """
        計算兩船之間的距離
        """
        return np.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2)

    def debug_print(self, message):
        """只在調試模式下打印信息"""
        if self.debug_mode:
            print(message)
            
    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS: AvailableFunctions) -> str:
        """
        根據觀察到的特徵執行動作。
        :param features: 當前環境的觀察資料
        :param VALID_FUNCTIONS: 可用的動作函數
        :return: 執行的動作命令（字串）
        """
        self.logger.debug("開始執行動作")
        action = ""
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            self.logger.warning(f"找不到單位: {self.ac_name}")
            return action  # 如果找不到單位，返回空動作
        self.logger.debug("已獲取單位資訊")
        
        # 獲取當前狀態
        current_state = self.get_state(features)
        
        # 如果有前一步資料，進行訓練
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.get_reward(self.prev_state, current_state)
            done = self.get_distance(current_state) < self.done_condition
            self.total_reward += reward
            self.logger.debug(f"訓練前")
            # 每步都进行训练
            self.logger.info(f"訓練中...")
            self.train(self.prev_state, self.prev_action, reward, current_state, done)
            
        self.logger.debug("訓練完成")
        
        # 選擇動作
        state_tensor = torch.tensor([current_state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            self.logger.debug(f"隨機選擇動作: {action}")
        else:
            action = q_values.argmax().item()
            self.logger.debug(f"根據Q值選擇動作: {action}, Q值: {q_values}")
        
        # 執行動作
        action_cmd = self.apply_action(action, ac)
        self.logger.debug(f"應用動作命令: {action_cmd}")
        
        # 更新前一步資料
        self.prev_state = current_state
        self.prev_action = action
        self.prev_features = features
        self.steps += 1
        print("steps = ",self.steps)
        # 更新 epsilon
        self.epsilon = max(0.1, self.epsilon * 0.999)
        self.logger.debug(f"更新epsilon: {self.epsilon:.4f}")
        
        return action_cmd
    
    def normalize_state(self, state):
        min_lon, max_lon = 121.0, 122.5  # 根據你的地圖範圍調整
        min_lat, max_lat = 23.5, 24.5
        min_heading, max_heading = 0.0, 360.0
        min_speed, max_speed = 0.0, 30.0
        norm_state = np.zeros_like(state)
        norm_state[0] = (state[0] - min_lon) / (max_lon - min_lon)  # B_lon
        norm_state[1] = (state[1] - min_lat) / (max_lat - min_lat)  # B_lat
        norm_state[2] = state[2] / max_heading  # B_heading
        norm_state[3] = state[3] / max_speed  # B_speed   
        norm_state[4] = (state[4] - min_lon) / (max_lon - min_lon)  # A_lon
        norm_state[5] = (state[5] - min_lat) / (max_lat - min_lat)  # A_lat
        return norm_state   

    def get_state(self, features: FeaturesFromSteam) -> np.ndarray:
        """
        獲取當前狀態向量，包含自身單位和目標的資訊。
        :return: numpy 陣列，例如 [B_lon, B_lat, B_heading, B_speed, A_lon, A_lat]
        """
        # 獲取自身單位（B 船）的資訊
        ac = self.get_unit_info_from_observation(features, self.ac_name)
        if ac is None:
            # 如果找不到單位，返回預設狀態
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 獲取目標單位（A 船）的資訊
        if self.target_name:
            target = self.get_contact_info_from_observation(features, self.target_name) or \
                    self.get_unit_info_from_observation(features, self.target_name)
            if target:
                # 根據 target 的型別提取經緯度
                if isinstance(target, dict):  # 來自 features.contacts
                    target_lon = float(target.get('Lon', 0.0))
                    target_lat = float(target.get('Lat', 0.0))
                else:  # 來自 features.units (Unit 物件)
                    target_lon = float(target.Lon)
                    target_lat = float(target.Lat)
            else:
                target_lon, target_lat = 0.0, 0.0  # 目標未找到時的預設值
        else:   
            target_lon, target_lat = 0.0, 0.0  # 未指定目標時的預設值
        raw_state = np.array([float(ac.Lon), float(ac.Lat), float(ac.CH), float(ac.CS), target_lon, target_lat])
        normalized_state = self.normalize_state(raw_state)

        # 返回狀態向量：[自身經度, 自身緯度, 自身航向, 自身航速, 目標經度, 目標緯度]
        return normalized_state
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        distance = self.get_distance(state)
        next_distance = self.get_distance(next_state)
         # print(f"Distance: {distance:.4f} -> {next_distance:.4f}")
        #reward = -100 * next_distance
        # if (distance - next_distance) > 0:
        #     reward = 2*(1-next_distance)  # 放大距離變化
        # else:
        #     reward = 10*(-next_distance)  # 放大距離變化
        
        self.logger.debug(f"距離變化: {distance:.4f} -> {next_distance:.4f}")
            
        reward = 500 * (distance-next_distance)
        # print(f"Reward: {reward}")
        if next_distance + 0.01 < self.best_distance:
            self.best_distance = next_distance   #如果當前距離比最佳距離近0.5公里 換他當最佳距離 然後給獎勵
            reward += 3
            self.logger.debug(f"新的最佳距離: {self.best_distance:.4f}")
        if next_distance < self.done_condition:
            reward += 20
        # if next_distance > 0.25:
        #     reward -= 100
        # print(f"FinalReward: {reward:.4f}")
            self.logger.debug("達到目標條件!")
            
        self.logger.debug(f"獎勵: {reward:.4f}")
            
        return reward

    def apply_action(self, action: int, ac: Unit) -> str:
        """將動作轉換為 CMO 命令"""
        step_size = 0.005
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0:  # 上
            heading = 0
        elif action == 1:  # 下
            heading = 180
        elif action == 2:  # 左
            heading = 270
        elif action == 3:  # 右
            heading = 90
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=self.ac_name,
            heading=heading,
            speed=30
        )

    def train(self, state, action, reward, next_state, done):
        """訓練 DQN 模型"""
        # 将经验添加到记忆中
        self.memory.append((state, action, reward, next_state, done))
        
        # 每步训练，不使用批量
        states = torch.tensor([state], dtype=torch.float32).to(self.device)
        actions = torch.tensor([action], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([reward], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([next_state], dtype=torch.float32).to(self.device)
        dones = torch.tensor([done], dtype=torch.float32).to(self.device)
        
        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.logger.debug(f"損失: {loss.item():.4f}")
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 記錄訓練統計數據  
        distance = self.get_distance(state)
        
        # 每10步記錄一次
        if self.steps % 10 == 0:
            self.logger.info(f"步數: {self.steps}, 距離: {distance:.4f}, 損失: {loss.item():.4f}, 總獎勵: {self.total_reward:.4f}")
            self.stats_logger.log_stat("distance", distance, self.steps)
            self.stats_logger.log_stat("loss", loss.item(), self.steps)
            self.stats_logger.log_stat("return", self.total_reward, self.steps)

        if self.steps % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.logger.debug(f"更新目標網路，步數: {self.steps}")
    
    # def reset(self):
    #     self.best_distance = 1000000
    #     self.prev_state = None
    #     self.prev_action = None
    #     self.prev_features = None
    #     self.state_history.clear()  # 清空狀態歷史
    #     # self.hidden = self.model.init_hidden(1)  # 重置隱藏狀態
    #     # self.total_reward = 0
    #     return set_unit_position(
    #         side=self.player_side,
    #         unit_name=self.ac_name,
    #         latitude=24.04,
    #         longitude=122.18
    #     )
