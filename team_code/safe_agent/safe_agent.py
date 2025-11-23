import os
import carla
from sensor_agent import SensorAgent
from faulty_sensor_agent import FaultySensorAgent
from perception_simplex.safety_layer import SafetyLayer
from perception_simplex.utils import visualize_bev, preprocess_lidar_data, preprocess_mission_layer_detections

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


class SafeAgent(FaultySensorAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the safety layer
        self.safety_layer = SafetyLayer(
            a_brake_max=7.0,
            d_margin=0.5
        )
        self.safety_override = False
        self.safety_enabled = int(os.environ.get('SAFETY', 0)) == 1
        self.emergency = False

        # Visualization config
        self.visualize = int(os.environ.get('VISUALIZE', 0)) == 1

        print(f"[SafeAgent] Initialized. Safety: {'enabled' if self.safety_enabled else 'disabled'}.")

    def run_step(self, input_data, timestamp, sensors=None):
        # Run only the mission layer if safety is disabled
        if not self.safety_enabled:
            return super().run_step(input_data, timestamp, sensors)

        # Extract speed, lidar data and mission detections
        speed = input_data['speed'][1]['speed'].copy()
        lidar_data = input_data['lidar'][1].copy()
        mission_layer_detections = self.get_mission_detections()

        # Preprocess some of the data
        lidar_data = preprocess_lidar_data(lidar_data)
        mission_layer_detections = preprocess_mission_layer_detections(mission_layer_detections)

        # Extract mission layer control action
        control_mission = super().run_step(input_data, timestamp, sensors)

        # Detect obstacles using the safety layer
        safety_layer_detections = self.safety_layer.detect_obstacles(lidar_data)

        # Detect faults in the mission detections
        faulty_detections = self.safety_layer.detect_faults(safety_layer_detections, mission_layer_detections)

        # Assess collision risk for each safety layer detection
        collision_risks, braking_area_box = self.safety_layer.assess_collision_risk(safety_layer_detections, speed, faulty_detections)

        # Implement the simplex logic
        control_final, safety_override = self.safety_layer.fault_handler(control_mission, faulty_detections, collision_risks, speed)

        # Record emergency if there is a collision risk. In case of an emergency full stop is applied.
        if not self.emergency and safety_override and any(collision_risks):
            self.emergency = True

        # Safety Layer does not work for even steps. If it was `safety_override` last time -- keep applying it.
        if (self.step % 2 == 0 and self.safety_override) or self.emergency:
            control_final = self.safety_layer.override_control()
            safety_override = True
        
        # Convert the safety override into a CARLA VehicleControl object
        if safety_override or self.emergency:
            control_final = carla.VehicleControl(
                steer=control_final['steer'],
                brake=control_final['brake'],
                throttle=control_final['throttle']
            )
        self.safety_override = safety_override

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
        if self.visualize and self.save_path:
            visualize_bev(
                lidar_data,
                safety_layer_detections,
                save_path=os.path.join(self.save_path, f"SL-{self.step:04}.png"),
                xlim=(-30, 30),
                ylim=(-60, 0),
                mission_layer_detections=mission_layer_detections,
                faulty_detections=faulty_detections,
                collision_risks=collision_risks,
                speed=speed,
                safety_override=safety_override,
                braking_area_box=braking_area_box
            )

        print(f"Speed: {round(speed, 2)} m/s. Brake: {safety_override}.")

        return control_final
    
    def get_mission_detections(self):
        bb_buffer = self.bb_buffer
        if len(bb_buffer) == 0:
            return []
        else:
            return bb_buffer[0]