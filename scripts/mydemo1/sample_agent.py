# Author: Minh Hua
# Date: 10/21/2023
# Purpose: A sample agent to interact with the steam_demo scenario, demonstrating our ability to work with the Steam version of CMO.
# Randomly moves an aircraft every timestep.

import random

from pycmo.lib.actions import AvailableFunctions, set_unit_course, auto_attack_contact
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import FeaturesFromSteam, Unit

class MyAgent1(BaseAgent):
    def __init__(self, player_side:str, ac_name:str):
        super().__init__(player_side)
        self.ac_name = ac_name

    def get_unit_info_from_observation(self, features: FeaturesFromSteam, unit_name:str) -> Unit:
        units = features.units
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None

    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS:AvailableFunctions) -> str:
        action = ""
        ac = self.get_unit_info_from_observation(features=features, unit_name=self.ac_name)
        print("ac:", ac)
        # 取得敵方單位
        enemy_units = features.contacts
        for enemy_unit in enemy_units:
            if enemy_unit['Name'] == "enemy1":  # 使用字典訪問方式
                # 直接使用敵方單位的位置作為目標
                target_longitude = float(enemy_unit['Lon'])  # 使用字典訪問方式
                target_latitude = float(enemy_unit['Lat'])   # 使用字典訪問方式
                action = set_unit_course(
                    side=self.player_side, 
                    unit_name=self.ac_name, 
                    latitude=target_latitude, 
                    longitude=target_longitude
                )
                break
        
        # 如果沒有找到敵人，則使用預設的移動方式
        if not action:
            delta_longitude = 0.1    
            delta_latitude = 0.1
            new_longitude = float(ac.Lon) + delta_longitude
            new_latitude = float(ac.Lat) + delta_latitude
            action = set_unit_course(
                side=self.player_side, 
                unit_name=self.ac_name, 
                latitude=new_latitude, 
                longitude=new_longitude
            )
        
        return action
