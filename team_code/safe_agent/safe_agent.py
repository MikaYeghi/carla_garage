from sensor_agent import SensorAgent
from perception_simplex.safety_layer import SafetyLayer

def save_run_data(frame_id, run_data):
    import os, pickle
    assert frame_id is not None
    save_dir = "run_data/1-1_Obstacle_Lidar+Mission_Detections+Speed+Mission_Control"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"frame-{frame_id}.pkl"), "wb") as handler:
        pickle.dump(run_data, handler, protocol=pickle.HIGHEST_PROTOCOL)

# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'SafeAgent'


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


class SafeAgent(SensorAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_layer = SafetyLayer()
        print("[SafeAgent] Initialized.")

    def run_step(self, input_data, timestamp, sensors=None):
        speed = input_data['speed'][1]['speed']

        # === 1. Mission control ===
        control_mission = super().run_step(input_data, timestamp, sensors)

        # === 2. Safety layer perception ===
        safety_obstacles = self.safety_layer.detect_obstacles(input_data.get('lidar'))
        mission_detections = self.get_mission_detections()

        # Save the run data
        run_data = {
            "lidar_points": input_data.get('lidar')[1],
            "mission_detections": mission_detections,
            "speed": speed,
            "control_mission": {
                "throttle": control_mission.throttle,
                "steer": control_mission.steer,
                "brake": control_mission.brake,
                "hand_brake": control_mission.hand_brake, 
                "reverse": control_mission.reverse,
                "manual_gear_shift": control_mission.manual_gear_shift,
                "gear": control_mission.gear                
            }
        }
        # save_run_data(input_data.get('lidar')[0], run_data)

        # === 3. Fault detection ===
        fault = self.safety_layer.detect_faults(mission_detections, safety_obstacles)
        collision_risk = self.safety_layer.assess_collision_risk(safety_obstacles, speed)

        # === 4. Decision logic (Simplex supervisor) ===
        if fault and collision_risk:
            control_final = self.safety_layer.override_control()
        else:
            control_final = self.safety_layer.limit_velocity(control_mission, speed)

        # === 5. Logging ===
        # if self.logger:
        #     self.logger.log_step(input_data, timestamp, sensors, control_final, mode=mode)

        return control_final
    
    def get_mission_detections(self):
        return self.bb_buffer