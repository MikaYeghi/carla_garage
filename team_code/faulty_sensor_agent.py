import os
import sys
import gzip
import json
import carla
import ujson
import pickle
import numpy as np

from srunner.scenariomanager.timer import GameTime
from nav_planner import extrapolate_waypoint_route

# Append the current directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from team_code.sensor_agent import SensorAgent


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'FaultySensorAgent'

class EgoActionsLoader:
    def __init__(self, *args, **kwargs):
        # Retrieve the replay ID
        self.replay_id = self.get_replay_id()

        # Retrieve the replay save path
        self.replay_path = self.get_save_path()

        # Retrieve replay data
        self.replay_data = self.retrieve_replay_data()

        # Convert the replay data into a dict
        self.replay_data_dict = {x["timestamp"]: x["control"] for x in self.replay_data}

        print(f"[EgoActionsLoader]: Initialized{'. Loading from: ' + self.replay_path if self.replay_path else '. No source to load from.'}")

    def get_save_path(self):
        save_path = os.environ.get('SAVE_PATH', None)
        if self.replay_id:
            return os.path.join(save_path, self.replay_id, "ego_actions.pkl.gz")
        else:
            return None

    def get_replay_id(self):
        if "REPLAY_ID" in os.environ:
            return os.environ.get("REPLAY_ID")
        return None
    
    def retrieve_replay_data(self):
        if self.replay_path:
            if os.path.exists(self.replay_path):
                return self.load_ego_actions(self.replay_path)
            else:
                return []
        else:
            return []
    
    def load_ego_actions(self, path: str, verbose=0):
        """
        Load and reconstruct ego control actions from a .pkl.gz file.

        Args:
            path (str): Full path to the saved ego_actions.pkl.gz file.

        Returns:
            list[dict]: Each element has:
                {
                    "timestamp": float,
                    "control": carla.VehicleControl
                }
        """
        # Load pickled list of dicts
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)

        # Convert each dict["control"] back to VehicleControl 
        def reconstruct_control(d):
            ctrl = carla.VehicleControl()
            ctrl.steer = float(d["steer"])
            ctrl.throttle = float(d["throttle"])
            ctrl.brake = float(d["brake"])
            ctrl.hand_brake = bool(d["hand_brake"])
            ctrl.reverse = bool(d["reverse"])
            ctrl.manual_gear_shift = bool(d["manual_gear_shift"])
            ctrl.gear = int(d["gear"])
            return ctrl

        for entry in data:
            entry["control"] = reconstruct_control(entry["control"])

        if verbose:
            print(f"[EgoActionsLoader] Loaded {len(data)} ego actions from {path}")

        return data
    
    def choose_action(self, input_data, timestamp, sensors, control):
        if timestamp in self.replay_data_dict.keys():
            print(f"[EgoActionsLogger]: Replaced the control action.")
            control = self.replay_data_dict[timestamp]
        return control

class EgoActionsLogger:
    def __init__(self, *args, **kwargs):
        self.save_path = self.get_save_path(args[1])
        self.step = 0
        self.ego_actions_list = []
        print(f"[EgoActionsLogger]: Initialized. Saving to: {self.save_path}.")

    def get_save_path(self, run_id):
        save_path = os.environ.get('SAVE_PATH', None)
        assert save_path is not None
        return os.path.join(save_path, run_id, "ego_actions.pkl.gz")

    def log_step(self, input_data, timestamp, sensors, control, verbose=0):
        self.ego_actions_list.append(
            {
                "timestamp": timestamp,
                "control": control
            }
        )

        if verbose:
            print(f"[EgoActionsLogger]: Logged step #{self.step}.")

        self.step += 1
    
    def reset(self):
        self.step = 0
        self.ego_actions_list = []
        print("[EgoActionsLogger]: Reset.")
    
    def dump_to_json(self):
        def make_serializable(obj):
            if isinstance(obj, carla.VehicleControl):
                # Convert to simple Python dict
                return {
                    "steer": float(obj.steer),
                    "throttle": float(obj.throttle),
                    "brake": float(obj.brake),
                    "hand_brake": bool(obj.hand_brake),
                    "reverse": bool(obj.reverse),
                    "manual_gear_shift": bool(obj.manual_gear_shift),
                    "gear": int(obj.gear),
                }
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj

        safe_dict = make_serializable(self.ego_actions_list)

        with gzip.open(self.save_path, "wb") as f:
            pickle.dump(safe_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

class FaultySensorAgent(SensorAgent):
    """
    A subclass of SensorAgent that currently behaves identically to the base class.
    This serves as a testbed for future controlled fault injection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[FaultySensorAgent] Initialized.")

        # Ego actions logger
        self.ego_actions_logger = EgoActionsLogger(*args, **kwargs)

        # Ego actions loader
        self.ego_actions_loader = EgoActionsLoader(*args, **kwargs)

    def run_step(self, input_data, timestamp, sensors=None):
        control = super().run_step(input_data, timestamp, sensors)
        
        # Choose a control action between the one proposed by the current algoprithm and a pre-recorded one
        control = self.ego_actions_loader.choose_action(input_data, timestamp, sensors, control)

        # Log the current control action
        self.ego_actions_logger.log_step(input_data, timestamp, sensors, control)
        
        return control

    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """
        # NOTE: might need different ego action save paths for different repetitions!
        if self.save_path is not None:
            self.lon_logger.dump_to_json()
            self.ego_actions_logger.dump_to_json()
            if len(self.nets[0].speed_histogram) > 0:
                with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
                    ujson.dump(self.nets[0].speed_histogram, f, indent=4)

            if self.config.tp_attention:
                if len(self.tp_attention_buffer) > 0:
                    print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
                    with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
                        ujson.dump(self.tp_attention_buffer, f, indent=4)

                del self.tp_attention_buffer

        del self.nets
        del self.config
        del self.metric_info