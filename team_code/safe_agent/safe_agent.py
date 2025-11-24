import os
import carla
import pickle
from typing import Dict
from sensor_agent import SensorAgent
from faulty_sensor_agent import FaultySensorAgent
from perception_simplex.safety_layer import SafetyLayer
from synergistic_simplex.safety_layer import SafetyLayerSS
from perception_simplex.utils import visualize_bev, preprocess_lidar_data, preprocess_mission_layer_detections

def save_runtime_data_to_file(save_dir: str, runtime_data: Dict):
    assert "frame_id" in runtime_data.keys()
    frame_id = runtime_data['frame_id']
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"frame-{frame_id}.pkl"), "wb") as handler:
        pickle.dump(runtime_data, handler, protocol=pickle.HIGHEST_PROTOCOL)

# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'SafeAgent'

def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')

class SafeAgent(FaultySensorAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the safety layer
        self.safety_layer = SafetyLayerSS(
            a_brake_max=7.0,
            d_margin=0.5
        )
        self.safety_override = 0
        self.safety_enabled = int(os.environ.get('SAFETY', 0)) == 1
        self.emergency = False

        # Logging config
        self.save_runtime_data = int(os.environ.get('SAVE_RUNTIME_DATA', 0)) == 1

        # Visualization config
        self.visualize = int(os.environ.get('VISUALIZE', 0)) == 1

        # BEV semantic map
        self.pred_bev_semantic = None

        print(f"[SafeAgent] Initialized. Safety: {'enabled' if self.safety_enabled else 'disabled'}.")

    def run_step(self, input_data, timestamp, sensors=None):
        # Extract speed, lidar data and mission detections
        speed = input_data['speed'][1]['speed'].copy()
        lidar_data = input_data['lidar'][1].copy()
        mission_layer_detections = self.get_mission_detections()

        # Preprocess some of the data
        lidar_data = preprocess_lidar_data(lidar_data)
        mission_layer_detections = preprocess_mission_layer_detections(mission_layer_detections)

        # Extract mission layer control action
        control_mission = super().run_step(input_data, timestamp, sensors)

        # BEV semantic map
        if self.pred_bev_semantic is not None:
            pred_bev_semantic = self.pred_bev_semantic
        else:
            pred_bev_semantic = None

        # Run only the mission layer if safety is disabled
        if not self.safety_enabled:
            # Save the runtime data (ML layer only)
            if self.save_runtime_data:
                runtime_data = {
                    "step": self.step,
                    "frame_id": input_data.get('lidar')[0],
                    "lidar_points": input_data.get('lidar')[1],
                    "mission_layer_detections": mission_layer_detections,
                    "safety_override": False,
                    "safety_enabled": self.safety_enabled,
                    "emergency": False,
                    "pred_bev_semantic": pred_bev_semantic,
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
                save_runtime_data_to_file(os.path.join(self.save_path, "runtime_data"), runtime_data)

            return control_mission

        # Detect obstacles using the safety layer
        safety_layer_detections = self.safety_layer.detect_obstacles(lidar_data)

        # Detect faults in the mission detections
        faulty_detections = self.safety_layer.detect_faults(safety_layer_detections, mission_layer_detections)

        # Assess collision risk for each safety layer detection
        collision_risks, braking_area_box = self.safety_layer.assess_collision_risk(safety_layer_detections, speed, faulty_detections)

        # Implement the simplex logic
        control_final, safety_override = self.safety_layer.fault_handler(control_mission, faulty_detections, collision_risks, speed, pred_bev_semantic=pred_bev_semantic, safety_layer_detections=safety_layer_detections)

        # Record emergency if there is a collision risk. In case of an emergency full stop is applied.
        if not self.emergency and safety_override == 2 and any(collision_risks):
            self.emergency = True

        # Safety Layer does not work for even steps. If it was `safety_override` last time -- keep applying it.
        if (self.step % 2 == 0 and self.safety_override == 2) or self.emergency:
            control_final = self.safety_layer.override_control()
            safety_override = 2
        
        # Convert the safety override into a CARLA VehicleControl object
        if safety_override > 0 or self.emergency:
            control_final = carla.VehicleControl(
                steer=control_final['steer'],
                brake=control_final['brake'],
                throttle=control_final['throttle']
            )
        self.safety_override = safety_override

        # Save the runtime data (ML + Safety layers)
        if self.save_runtime_data:
            runtime_data = {
                "step": self.step,
                "frame_id": input_data.get('lidar')[0],
                "lidar_points": input_data.get('lidar')[1],
                "mission_layer_detections": mission_layer_detections,
                "safety_layer_detections": safety_layer_detections,
                "faulty_detections": faulty_detections,
                "collision_risks": collision_risks,
                "braking_area_box": braking_area_box,
                "safety_override": safety_override,
                "safety_enabled": self.safety_enabled,
                "emergency": self.emergency,
                "pred_bev_semantic": pred_bev_semantic,
                "speed": speed,
                "control_mission": {
                    "throttle": control_mission.throttle,
                    "steer": control_mission.steer,
                    "brake": control_mission.brake,
                    "hand_brake": control_mission.hand_brake, 
                    "reverse": control_mission.reverse,
                    "manual_gear_shift": control_mission.manual_gear_shift,
                    "gear": control_mission.gear                
                },
                "control_final": {
                    "throttle": control_final.throttle,
                    "steer": control_final.steer,
                    "brake": control_final.brake,
                    "hand_brake": control_final.hand_brake, 
                    "reverse": control_final.reverse,
                    "manual_gear_shift": control_final.manual_gear_shift,
                    "gear": control_final.gear 
                }
            }
            save_runtime_data_to_file(os.path.join(self.save_path, "runtime_data"), runtime_data)

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

        print(f"Speed: {round(speed, 2)} m/s. Safety override: {safety_override}. Emergency: {self.emergency}.")

        return control_final
    
    def get_mission_detections(self):
        bb_buffer = self.bb_buffer
        if len(bb_buffer) == 0:
            return []
        else:
            return bb_buffer[0]