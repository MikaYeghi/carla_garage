from sensor_agent import SensorAgent
from safety_layer import SafetyLayer


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
        # === 1. Mission control ===
        control_mission = super().run_step(input_data, timestamp, sensors)

        # === 2. Safety layer perception ===
        safety_obstacles = self.safety_layer.detect_obstacles(input_data.get('lidar'))
        mission_detections = self.get_mission_detections()

        # === 3. Fault detection ===
        fault = self.safety_layer.detect_faults(mission_detections, safety_obstacles)
        collision_risk = self.safety_layer.assess_collision_risk(safety_obstacles)

        # === 4. Decision logic (Simplex supervisor) ===
        if fault and collision_risk:
            control_final = self.safety_layer.override_control()
            mode = "SAFETY_OVERRIDE"
        else:
            control_final = self.safety_layer.limit_velocity(control_mission)
            mode = "MISSION"

        # === 5. Logging ===
        # if self.logger:
        #     self.logger.log_step(input_data, timestamp, sensors, control_final, mode=mode)

        return control_final
    
    def get_mission_detections(self):
        return self.bb_buffer