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

        if verbose > 0:
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
        print("[FaultySensorAgent] Initialized (currently identical to SensorAgent).")

        # Ego actions logger
        self.ego_actions_logger = EgoActionsLogger(*args, **kwargs)

    def run_step(self, input_data, timestamp, sensors=None):
        control = super().run_step(input_data, timestamp, sensors)

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