import os
import carla
from sensor_agent import SensorAgent
from perception_simplex.utils import visualize_bev
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
        
        # Initialize the safety layer
        self.safety_layer = SafetyLayer()

        print("[SafeAgent] Initialized.")

    def run_step(self, input_data, timestamp, sensors=None):
        # Extract speed & lidar data
        speed = input_data['speed'][1]['speed']
        lidar_data = input_data.get('lidar')[1]
        print(f"Speed: {round(speed, 2)} m/s")

        # Extract mission layer control action
        control_mission = super().run_step(input_data, timestamp, sensors)

        # Detect obstacles using the safety layer
        safety_layer_detections = self.safety_layer.detect_obstacles(lidar_data)

        # Detect faults in the mission detections
        mission_layer_detections = self.get_mission_detections()
        faulty_detections = self.safety_layer.detect_faults(safety_layer_detections, mission_layer_detections)

        # Assess collision risk for each safety layer detection
        collision_risks = self.safety_layer.assess_collision_risk(safety_layer_detections, speed, faulty_detections)

        # Implement the simplex logic
        control_final, safety_override = self.safety_layer.fault_handler(control_mission, faulty_detections, collision_risks, speed)
        
        # Convert the safety override into a CARLA VehicleControl object
        if safety_override:
            control_final = carla.VehicleControl(
                steer=control_final['steer'],
                brake=control_final['brake'],
                throttle=control_final['throttle']
            )

        # Save the run data
        run_data = {
            "lidar_points": input_data.get('lidar')[1],
            "mission_detections": mission_layer_detections,
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

        # Visualize from safety layer's perspective
        if self.config.debug and self.save_path:
            visualize_bev(
                lidar_data,
                safety_layer_detections,
                save_path=os.path.join(self.save_path, f"{self.step:04}-SL.png"),
                xlim=(-30, 30),
                ylim=(-60, 0),
                mission_layer_detections=mission_layer_detections,
                faulty_detections=faulty_detections,
                collision_risks=collision_risks,
                speed=speed,
                safety_override=safety_override
            )

        return control_final
    
    def get_mission_detections(self):
        bb_buffer = self.bb_buffer
        if len(bb_buffer) == 0:
            return []
        else:
            return bb_buffer[0]