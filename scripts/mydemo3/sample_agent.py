# Author: Minh Hua
# Date: 10/21/2023
# Purpose: A sample agent to interact with the steam_demo scenario, demonstrating our ability to work with the Steam version of CMO.
# Randomly moves an aircraft every timestep.

import random

from pycmo.lib.actions import AvailableFunctions, set_unit_course, auto_attack_contact, set_unit_heading, set_unit_speed
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

    # ac: Unit(XML_ID=0, ID='0YS6W7-0HNAENMQ57D2M', Name='Cheng Kung', Side='Taiwan', DBID=649, Type='Ship', CH=0.0, CS=0.0, CA=0.0, Lon=122.31600167879, Lat=24.064523349445, CurrentFuel=64000.0, MaxFuel=64000.0, Mounts=[Mount(XML_ID=0, ID='0YS6W7-0HNAENMQ57D3A', Name='20mm/85 Mk15 Phalanx Blk 0 CIWS', DBID=553, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D3D', WeaponID=1702, QuantRemaining=5, MaxQuant=5)]), Mount(XML_ID=1, ID='0YS6W7-0HNAENMQ57D3I', Name='324mm Mk32 TT Triple', DBID=789, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D3K', WeaponID=1647, QuantRemaining=3, MaxQuant=3)]), Mount(XML_ID=2, ID='0YS6W7-0HNAENMQ57D3T', Name='324mm Mk32 TT Triple', DBID=789, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D3V', WeaponID=1647, QuantRemaining=3, MaxQuant=3)]), Mount(XML_ID=3, ID='0YS6W7-0HNAENMQ57D48', Name='40mm/70 Single Breda Type 564', DBID=1770, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D4A', WeaponID=1177, QuantRemaining=27, MaxQuant=27)]), Mount(XML_ID=4, ID='0YS6W7-0HNAENMQ57D4F', Name='40mm/70 Single Breda Type 564', DBID=1770, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D4H', WeaponID=1177, QuantRemaining=27, MaxQuant=27)]), Mount(XML_ID=5, ID='0YS6W7-0HNAENMQ57D4M', Name='76mm/62 OTO Melara Compact', DBID=550, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D4O', WeaponID=858, QuantRemaining=20, MaxQuant=20)]), Mount(XML_ID=6, ID='0YS6W7-0HNAENMQ57D52', Name='AN/SLQ-25A Nixie', DBID=1309, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D54', WeaponID=1066, QuantRemaining=2, MaxQuant=2)]), Mount(XML_ID=7, ID='0YS6W7-0HNAENMQ57D58', Name='Hsiung Feng II Quad', DBID=269, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D5A', WeaponID=1934, QuantRemaining=4, MaxQuant=4)]), Mount(XML_ID=8, ID='0YS6W7-0HNAENMQ57D5K', Name='Hsiung Feng III Quad', DBID=1037, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D5M', WeaponID=1133, QuantRemaining=4, MaxQuant=4)]), Mount(XML_ID=9, ID='0YS6W7-0HNAENMQ57D5V', Name='Kung Fen DL', DBID=795, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D61', WeaponID=1255, QuantRemaining=16, MaxQuant=16)]), Mount(XML_ID=10, ID='0YS6W7-0HNAENMQ57D65', Name='Kung Fen DL', DBID=795, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D67', WeaponID=1255, QuantRemaining=16, MaxQuant=16)]), Mount(XML_ID=11, ID='0YS6W7-0HNAENMQ57D6B', Name='Kung Fen DL', DBID=795, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D6D', WeaponID=1255, QuantRemaining=16, MaxQuant=16)]), Mount(XML_ID=12, ID='0YS6W7-0HNAENMQ57D6H', Name='Kung Fen DL', DBID=795, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D6J', WeaponID=1255, QuantRemaining=16, MaxQuant=16)]), Mount(XML_ID=13, ID='0YS6W7-0HNAENMQ57D6N', Name='Mk13 Mod 4 Single Rail', DBID=609, Weapons=[Weapon(XML_ID=0, ID='0YS6W7-0HNAENMQ57D6P', WeaponID=844, QuantRemaining=1, MaxQuant=1), Weapon(XML_ID=1, ID='0YS6W7-0HNAENMQ57D71', WeaponID=220, QuantRemaining=0, MaxQuant=1)])], Loadout=None)
    def action(self, features: FeaturesFromSteam, VALID_FUNCTIONS:AvailableFunctions) -> str:
        action = ""
        ac = self.get_unit_info_from_observation(features=features, unit_name=self.ac_name)
        # print("features:", features.contacts)
        # print("ac:", ac)

        
        print("TotalScore:", features.side_.TotalScore)
        # 艦艇自身狀態
        print("----------1.艦艇自身狀態--------")
        print("當前位置:", ac.Lon, ac.Lat)
        print("當前航向:", ac.CH)
        print("當前航速:", ac.CS)
        print("當前燃料:", ac.CurrentFuel)
        print("最大燃料:", ac.MaxFuel)
        print("裝備資訊:")
        for mount in ac.Mounts:
            print("ID:", mount.XML_ID, "Name:", mount.Name, "目前數量:", mount.Weapons[0].QuantRemaining, "最大數量:", mount.Weapons[0].MaxQuant)
        print("----------2.友軍艦艇狀態----------")
        for friend in features.units:
            print("ID:", friend.ID, "Name:", friend.Name, "位置:", friend.Lon, friend.Lat, "航向:", friend.CH, "航速:", friend.CS, "燃料:", friend.CurrentFuel, "最大燃料:", friend.MaxFuel)
        print("----------3.敵方單位----------")
        for enemy in features.contacts:
            print("ID:", enemy['ID'], "Name:", enemy['Name'], "位置:", enemy['Lon'], enemy['Lat'])
        print("----------4.任務----------")       
        # OrderedDict([('ID', '0YS6W7-0HNAM1OVIN50V'), ('Name', 'mission1'), ('Type', 'Strike: 1'), ('Subtype', 'Air Intercept'), ('IsActive', 'true'), ('StartTime', None), ('EndTime', None), ('SISH', 'false'), ('UnitList', None)])
        for mission in features.missions:
            print("ID:", mission['ID'], "Name:", mission['Name'], "Type:", mission['Type'], "Subtype:", mission['Subtype'], "IsActive:", mission['IsActive'], "UnitList:", mission['UnitList'], "StartTime:", mission['StartTime'], "EndTime:", mission['EndTime'])
            if mission['UnitList'] is not None:
                print("執行單位:")
                for unit in mission['UnitList'].get("Unit", []):
                    for friend in features.units:
                        if unit == friend.ID:
                            print(friend.Name)
        # 自訂function(設定航向)
        action = set_unit_heading( 
            side=self.player_side, 
            unit_name=self.ac_name, 
            heading=float(ac.CH)+5.0
        )

        # enemy_units = features.contacts
        # for enemy_unit in enemy_units:
        #     print("action")
        #     if enemy_unit['Name'] == "enemy2":  # 使用字典訪問方式
        #         # 直接使用敵方單位的位置作為目標
        #         if ac.CS == 20.0:
        #             action = set_unit_course(
        #                 side=self.player_side, 
        #                 unit_name=self.ac_name, 
        #                 latitude=float(ac.Lat), 
        #                 longitude=float(ac.Lon)
        #             )
        #         else:
        #             action = auto_attack_contact(
        #                 attacker_id=ac.ID,
        #                 contact_id=enemy_unit['ID']
        #             )
        #         break
        
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
