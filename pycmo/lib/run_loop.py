# Author: Minh Hua
# Date: 08/16/2021
# Last Update: 06/16/2022
# Purpose: A run loop for agent/environment interaction.

# imports
from pycmo.lib.protocol import Server # Server to handle connection
from pycmo.env.cmo_env import CPEEnv, CMOEnv, StepType # CPE environment, functions as the client that sends actions and receives observations
# agents
from pycmo.agents.base_agent import BaseAgent
# auxiliary functions
import threading, time, statistics, json, os
from pycmo.lib.tools import print_env_information, clean_up_steps, ticks_to_unix, parse_datetime, parse_utc
import logging

def run_loop(player_side:str, config:dict, step_size:list=['0', '0', '1'], agent=None, max_steps=None, server=False, scen_file=None) -> None:
    """
    Description:
        Generic function to run an observe-act loop.

    Keyword Arguments:
        player_side: the name of the player's side.
        config: a dictionary of configuration paths to important folders and files.
        step_size: a list containing the step size in the format ["h", "m", "s"]. Default is step size of 1 seconds.
        agent: an agent to control the game.
        max_steps: the maximum number of allowable steps.
        server: whether or not to initialize a server.
        scen_file: whether or not to use a custom scenario file.

    Returns:
        None
    """        
    # config and set up, clean up steps folder
    steps_path = config["observation_path"]
    clean_up_steps(steps_path)
    
    # set up a Command TCP/IP socket if the game is not already running somewhere
    if server:
        server = Server(scen_file)
        x = threading.Thread(target=server.start_game)
        x.start()
        time.sleep(10)
    
    # build CPE environment
    env = CPEEnv(steps_path, step_size, player_side, config["scen_ended"])
    
    # initial variables and state
    step_id = 0
    initial_state = env.reset()
    state_old = initial_state
    cur_time = ticks_to_unix(initial_state.observation.meta.Time)
    print(parse_datetime(int(initial_state.observation.meta.Time)))

    # Configure a limit for the maximum number of steps
    if max_steps == None:
        max_steps = 1000

    # main loop
    while not (env.check_game_ended() or (step_id > max_steps)):
        # get current time
        cur_time = ticks_to_unix(state_old.observation.meta.Time)

        # perform random actions or choose the action
        available_actions = env.action_spec(state_old.observation)
        if agent != None:
            final_move = agent.get_action(state_old.observation, available_actions.VALID_FUNCTIONS)
        else:
            final_move = '--script \nTool_EmulateNoConsole(true)' # No action if no agent is loaded

        # get new state and observation, rewards, discount
        step_id += 1
        new_state = env.step(cur_time, step_id, action=final_move)
        current_score = new_state.observation.side_.TotalScore
        current_reward = new_state.reward
        print_env_information(step_id, parse_datetime(int(state_old.observation.meta.Time)), final_move, current_score, current_reward)

        # store new data into a long-term memory

        # set old state as the previous new state
        old_state = new_state

        if step_id % 10 == 0:
            clean_up_steps(steps_path)

def run_loop_steam(env: CPEEnv | CMOEnv,
                   agent:BaseAgent=None, 
                   max_steps=None) -> None:    
    env.logger.setLevel(logging.INFO)
    # start the game
    state = env.reset(close_scenario_end_and_player_eval_messages=False)
    action = ""
    action = agent.reset()
    state = env.step(action)

    # Configure a limit for the maximum number of steps
    total_steps = 0
    step_times_cmo = []
    step_times_rl = []
    # main loop
    while (not max_steps) or (total_steps < max_steps):
        t0_cmo = time.perf_counter()
        env.logger.debug("run_loop_steam")
        print_env_information(state.step_id, parse_utc(int(state.observation.meta.Time)), action, state.reward, state.reward)
        # perform random actions or choose the action
        available_actions = env.action_spec(state.observation)
        if agent:
            action = agent.action(state.observation, available_actions.VALID_FUNCTIONS)
        else:
            action = env.action_space.sample() # sample random action if no agent is loaded
        env.logger.debug("loop_action")
        # get new state and observation, rewards, discount
        # 2) 測包含 env.step 的整段步驟
        
        state = env.step(action)
        dt_cmo = time.perf_counter() - t0_cmo
        step_times_cmo.append(dt_cmo)
        env.logger.debug("loop_step")
        total_steps += 1
        if state.step_type == StepType(2) or env.check_game_ended():
            print_env_information(state.step_id, parse_utc(int(state.observation.meta.Time)), action, state.reward, state.reward)
            state = env.reset(close_scenario_end_and_player_eval_messages=True)
            action = ''
            agent.reset()
        env.logger.debug("loop_end")

    
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
