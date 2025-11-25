# Experimental Protocol

## Preparation 

First, launch the simulator in _Terminal 1_: 
```
/CarlaUE4.sh 
```

Second, in another terminal _Terminal 2_, run the following commands:
```
# 1. Activate the environment
conda activate garage_2

# 2. Perform the standard CARLA Garage setup
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# 3. Update PYTHONPATH --- we need this with the added Perception Simplex code
export PYTHONPATH=$PYTHONPATH:$WORK_DIR/team_code 

# 4. Set the save path
export SAVE_PATH=$WORK_DIR/results/DSN

# 5. Set the config files directory path
export CONFIGS_DIR=$WORK_DIR/configs

# 6. Enable visualization and runtime data saving
export VISUALIZE=1
export SAVE_RUNTIME_DATA=1
```

## Selecting Model

To select the SS model (one of PS, M2S, S2M, SS) set the following environmental variable to an appropriate value:
```
# One of [PS, M2S, S2M, SS]
export FAULT_HANDLER=M2S 
```

## Running Experiments

For each route (e.g. for route 01 of Scenario 1), perform the following steps:
1. **Free Run** --- We need the free run to record actions in the absence of obstacles.
    1. Run the free run experiment:
        ```
        SAFETY=0 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent-config model_ckpt/pretrained_models/all_towns/ --agent team_code/safe_agent/safe_agent.py --routes $CONFIGS_DIR/FS1/01.xml
        ```
    2. Record the `REPLAY_ID` of the resulting run from the save directory, e.g. `01_route0_11_22_04_35_51` from `$SAVE_PATH`.
    3. Put the `REPLAY_ID` in the corresponding column in the [Experiments spreadsheet](https://uillinoisedu-my.sharepoint.com/:x:/r/personal/myeghiaz_illinois_edu/Documents/Conferences/DSN%202026/Experiments.xlsx?d=w791ac1b6df334241924786752d55a9c9&csf=1&web=1&e=srJrjp).
2. **ML Run**
    1. Run in the ML setting. Note the added `REPLAY_ID` and changed `--routes` parameter.
        ```
        REPLAY_ID=<REPLAY_ID> SAFETY=0 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent-config model_ckpt/pretrained_models/all_towns/ --agent team_code/safe_agent/safe_agent.py --routes $CONFIGS_DIR/S1/01.xml

        # For example
        REPLAY_ID=01_route0_11_22_04_35_51 SAFETY=0 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent-config model_ckpt/pretrained_models/all_towns/ --agent team_code/safe_agent/safe_agent.py --routes $CONFIGS_DIR/S1/01.xml
        ```
    2. Record the values of the resulting metrics: `Game Time` and `CollisionTest`. Record the `Experiment ID` of this run.
    3. Put the recorded values (both metrics and `Experiment ID`) in the [Experiments spreadsheet](https://uillinoisedu-my.sharepoint.com/:x:/r/personal/myeghiaz_illinois_edu/Documents/Conferences/DSN%202026/Experiments.xlsx?d=w791ac1b6df334241924786752d55a9c9&csf=1&web=1&e=srJrjp).
3. **Perception Simplex Run**
    1. Run in the PS setting. Note that safety changes to `SAFETY=1`.
        ```
        REPLAY_ID=<REPLAY_ID> SAFETY=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent-config model_ckpt/pretrained_models/all_towns/ --agent team_code/safe_agent/safe_agent.py --routes $CONFIGS_DIR/S1/01.xml

        # For example
        REPLAY_ID=01_route0_11_22_04_35_51 SAFETY=1 python leaderboard/leaderboard/leaderboard_evaluator_local.py --agent-config model_ckpt/pretrained_models/all_towns/ --agent team_code/safe_agent/safe_agent.py --routes $CONFIGS_DIR/S1/01.xml
        ```
    2. Record the values of the resulting metrics: `Game Time` and `CollisionTest`. Record the `Experiment ID` of this run.
    3. Put the recorded values (both metrics and `Experiment ID`) in the [Experiments spreadsheet](https://uillinoisedu-my.sharepoint.com/:x:/r/personal/myeghiaz_illinois_edu/Documents/Conferences/DSN%202026/Experiments.xlsx?d=w791ac1b6df334241924786752d55a9c9&csf=1&web=1&e=srJrjp).
4. Repeat the steps above for the next route (e.g. route 02).