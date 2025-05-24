# Author: Minh Hua
# Date: 08/16/2021
# Last Update: 06/16/2022
# Purpose: A run loop for agent/environment interaction.

# imports
from pycmo.lib.protocol import Server # Server to handle connection
from pycmo.env.cmo_env import CMOEnv, StepType # CPE environment, functions as the client that sends actions and receives observations
# agents
from pycmo.agents.base_agent import BaseAgent
# auxiliary functions
import threading, time, statistics, json, os
from pycmo.lib.tools import print_env_information, clean_up_steps, ticks_to_unix, parse_datetime, parse_utc
import logging


def run_loop_steam_new(env: CMOEnv,
                   agent: BaseAgent=None, 
                   max_steps: int=None,
                   train: bool=True) -> None:    
    env.logger.setLevel(logging.INFO)
    # start the game
    env.reset(close_scenario_end_and_player_eval_messages=False)
    time.sleep(0.5)
    action = ""
    action = agent.reset()
    state = env.step(action)
    agent.get_reset_cmd(state.observation)
    print("reset_cmd = ", agent.reset_cmd)
    action = ""

    # Configure a limit for the maximum number of steps
    total_steps = 0
    step_times_cmo = []
    step_times_rl = []
    # main loop
    while (not max_steps) or (total_steps < max_steps):
        
        
        t0_cmo = time.perf_counter()
        state = env.step(action)

        action = agent.action(state.observation)
        dt_cmo = time.perf_counter() - t0_cmo
        # step_times_cmo.append(dt_cmo)


        # if train:
        #     print("train")
            # agent.train()

        total_steps += 1
        # 遊戲運行很久後重置
        if state.step_type == StepType(2) or env.check_game_ended():
            print_env_information(state.step_id, parse_utc(int(state.observation.meta.Time)), action, state.reward, state.reward)
            env.reset(close_scenario_end_and_player_eval_messages=True)
            action = agent.reset()


    
    env.end_game()
    # ---- 新增：寫出 JSON 檔，結構同 GAT資料.txt ----
    step_times_rl = agent.step_times_rl
    if step_times_rl and step_times_cmo:
        result = {
            "RL_only": {
                "mean": statistics.mean(step_times_rl),
                "std":  statistics.stdev(step_times_rl) if len(step_times_rl) > 1 else 0.0
            },
            "RL+CMO": {
                "mean": statistics.mean(step_times_cmo),
                "std":  statistics.stdev(step_times_cmo) if len(step_times_cmo) > 1 else 0.0
            },
            "step_times_rl": step_times_rl,
            "step_times_cmo": step_times_cmo
        }
        outfile = f"GAT資料_{int(time.time())}.txt"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        env.logger.info(f"[run_loop_steam] 時間統計已寫入: {os.path.abspath(outfile)}")

