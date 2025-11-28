import os
import torch
import carla
import pickle
import ujson  # Like json but faster
import numpy as np
import transfuser_utils as t_u
import torch.nn.functional as F
from copy import deepcopy
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
            a_brake_max=5.5,
            d_margin=0.5,
            iou_thresh=2.0
        )
        self.safety_override = 0
        self.safety_enabled = int(os.environ.get('SAFETY', 0)) == 1
        self.emergency = False
        
        # Record the fault handler type
        self.fault_handler_type = self.safety_layer.get_fault_handler_type()

        # Logging config
        self.save_runtime_data = int(os.environ.get('SAVE_RUNTIME_DATA', 0)) == 1

        # Visualization config
        self.visualize = int(os.environ.get('VISUALIZE', 0)) == 1

        # BEV semantic map
        self.pred_bev_semantic = None
        
        # Last frame data
        self.speed_last_custom = None
        self.lidar_last_custom = None

        print(f"[SafeAgent] Initialized. Safety: {'enabled' if self.safety_enabled else 'disabled'}.")

    def update_lidar_points_custom(self, speed, lidar_data, dt=0.05):
        """
        Approximate lidar data for the missing frames.
        """
        lidar_data[:, 1] += speed * dt
        return lidar_data

    def run_step(self, input_data, timestamp, sensors=None):
        # Extract speed, lidar data and mission detections
        speed = input_data['speed'][1]['speed'].copy()
        lidar_data = input_data['lidar'][1].copy()
        mission_layer_detections = self.get_mission_detections()

        # Preprocess some of the data
        lidar_data = preprocess_lidar_data(lidar_data)
        mission_layer_detections = preprocess_mission_layer_detections(mission_layer_detections)
        
        # Save lidar_data for odd frames, "predict" for even frames
        if self.step % 2 == 0:
            self.speed_last_custom = speed.copy()
            self.lidar_last_custom = lidar_data.copy()
        elif self.lidar_last_custom is not None:
            lidar_data = self.update_lidar_points_custom(self.speed_last_custom, self.lidar_last_custom)

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
        # 3: emergency braking
        # 2: velocity control braking
        # 1: throttle release (soft emergency)
        # 0: no override
        control_final, safety_override = self.safety_layer.fault_handler(control_mission, 
                                                                         faulty_detections, 
                                                                         collision_risks, 
                                                                         speed, 
                                                                         pred_bev_semantic=pred_bev_semantic, 
                                                                         safety_layer_detections=safety_layer_detections,
                                                                         input_data=input_data,
                                                                         timestamp=timestamp,
                                                                         agent=self)
        if self.fault_handler_type in ("SS", "S2M"):
            self.step -= 1

        # Record emergency if there is a collision risk. In case of an emergency full stop is applied.
        if not self.emergency and safety_override == 3:
            self.emergency = True

        # # Safety Layer does not work for even steps. If it was `safety_override` last time -- keep applying it.
        # if self.step % 2 == 0:
        #     if self.safety_override == 2 or self.safety_override == 3:
        #         control_final = self.safety_layer.override_control()
        #         safety_override = self.safety_override
        #     elif self.emergency:
        #         control_final = self.safety_layer.override_control()
        #         safety_override = 3
        #     elif self.safety_override == 1:
        #         control_final = self.safety_layer.soft_override_control()
        #         safety_override = 1
        
        # Convert the safety override into a CARLA VehicleControl object
        if safety_override > 1 or self.emergency:
            control_final = carla.VehicleControl(
                steer=0,
                brake=1,
                throttle=0
            )
        elif safety_override == 1:
            control_final = carla.VehicleControl(
                steer=0,
                brake=0,
                throttle=0
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

        # Stop the simulation if it's been stalled due to an emergency
        if speed < 0.1 and self.emergency:
            # raise KeyboardInterrupt
            import signal
            os.kill(os.getpid(), signal.SIGINT)

        return control_final
    
    def get_mission_detections(self):
        bb_buffer = self.bb_buffer
        if len(bb_buffer) == 0:
            return []
        else:
            return bb_buffer[0]
        
    @torch.inference_mode()  # Turns off gradient computation
    def _run_step(self, input_data, timestamp, sensors=None, safety_layer_detections=[]):  # pylint: disable=locally-disabled, unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            if self.config.backbone not in ('aim'):
                self.lidar_last = deepcopy(tick_data['lidar'])
            return control

        # Need to run this every step for GPS filtering
        tick_data = self.tick(input_data)

        lidar_indices = []
        for i in range(self.config.lidar_seq_len):
            lidar_indices.append(i * self.config.data_save_freq)

        #Current position of the car
        ego_x = self.state_log[-1][0]
        ego_y = self.state_log[-1][1]
        ego_theta = self.state_log[-1][2]

        ego_x_last = self.state_log[-2][0]
        ego_y_last = self.state_log[-2][1]
        ego_theta_last = self.state_log[-2][2]

        # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
        if self.config.backbone not in ('aim'):
            lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

        # Updates stop boxes by vehicle movement converting past predictions into the current frame.
        if self.stop_sign_controller:
            self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

        if self.config.backbone not in ('aim'):
            lidar_current = deepcopy(tick_data['lidar'])
            lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

            self.lidar_buffer.append(lidar_full)

        if self.config.backbone not in ('aim'):
            # We wait until we have sufficient LiDARs
            if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
                self.lidar_last = deepcopy(tick_data['lidar'])
                tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
                self.control = tmp_control

                return tmp_control

        if self.config.backbone in ('aim'):  # Image only method
            # Dummy data
            lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height,
                                    self.config.lidar_resolution_width)).to(self.device, dtype=torch.float32)
        else:
            # Voxelize LiDAR and stack temporal frames
            lidar_bev = []
            # prepare LiDAR input
            for i in lidar_indices:
                lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])

                # For single frame there is no point in realignment. The state_log index will also differ.
                if self.config.realign_lidar and self.config.lidar_seq_len > 1:
                    # Position of the car when the LiDAR was collected
                    curr_x = self.state_log[i][0]
                    curr_y = self.state_log[i][1]
                    curr_theta = self.state_log[i][2]

                    # Voxelize to BEV for NN to process
                    lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

                lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                                        use_ground_plane=self.config.use_ground_plane)

                lidar_histogram = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
                lidar_bev.append(lidar_histogram)

                lidar_bev = torch.cat(lidar_bev, dim=1)

        if self.config.backbone not in ('aim'):
            self.lidar_last = deepcopy(tick_data['lidar'])

        # prepare velocity input
        gt_velocity = tick_data['speed']
        velocity = gt_velocity.reshape(1, 1)  # used by transfuser

        compute_debug_output = self.config.debug and (self.save_path is not None)

        # new checkpoint lookahead: calculate which checkpoint to use for control
        speed = gt_velocity.item()

        if self.stop_after_meter > 0:
            dt = self.config.carla_frame_rate
            self.meters_travelled = self.meters_travelled + speed * dt

        # forward pass
        pred_wps = []
        pred_target_speeds = []
        pred_checkpoints = []
        bounding_boxes = []
        wp_selected = None
        for i in range(self.model_count):
            if self.config.backbone in ('transFuser', 'aim', 'bev_encoder'):
                pred_wp, \
                pred_target_speed, \
                pred_checkpoint, \
                pred_semantic, \
                pred_bev_semantic, \
                pred_depth, \
                pred_bb_features,\
                attention_weights,\
                pred_wp_1,\
                selected_path = self.nets[i].forward(
                    rgb=tick_data['rgb'],
                    lidar_bev=lidar_bev,
                    target_point=tick_data['target_point'],
                    target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
                    ego_vel=velocity,
                    command=tick_data['command'])
                # Only convert bounding boxes when they are used.
                if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                                self.stop_sign_controller):
                    pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
                else:
                    pred_bounding_box = None

                # Save the BEV semantic map for the 0-th model
                if i == 0:
                    self.pred_bev_semantic = pred_bev_semantic
            else:
                raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

            if self.config.use_wp_gru:
                if self.config.multi_wp_output:
                    wp_selected = 0
                    if F.sigmoid(selected_path)[0].item() > 0.5:
                        wp_selected = 1
                        pred_wps.append(pred_wp_1)
                    else:
                        pred_wps.append(pred_wp)
                else:
                    pred_wps.append(pred_wp)
            if self.config.use_controller_input_prediction:
                pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
                pred_checkpoints.append(pred_checkpoint[0])

            bounding_boxes.append(pred_bounding_box)

        # Append safety layer detections if they are supplied
        if len(safety_layer_detections) > 0:
            bounding_boxes.append(safety_layer_detections)

        # Average the predictions from ensembles
        if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                        self.stop_sign_controller):
            # We average bounding boxes by using non-maximum suppression on the set of all detected boxes.
            bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)

            self.bb_buffer.append(bbs_vehicle_coordinate_system)
        else:
            bbs_vehicle_coordinate_system = None

        if self.stop_sign_controller:
            stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

        if self.config.tp_attention:
            self.tp_attention_buffer.append(attention_weights[2])

        if self.config.use_wp_gru:
            self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

        # calculate target speed scalar from model predictions
        if self.config.use_controller_input_prediction:
            pred_target_speed_ensemble = torch.stack(pred_target_speeds,
                                                    dim=0).mean(dim=0)  # average across ensemble models' prediction

            if self.uncertainty_weight:
                uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
                if uncertainty[0] > self.config.brake_uncertainty_threshold:
                    pred_target_speed_scalar = self.inference_target_speeds[0]
                else:
                    pred_target_speed_scalar = sum(uncertainty * self.inference_target_speeds)
            else:
                pred_target_speed_index = torch.argmax(pred_target_speed_ensemble)
                pred_target_speed_scalar = self.inference_target_speeds[pred_target_speed_index]

        # Visualize the output of the last model
        if compute_debug_output:
            if self.config.use_controller_input_prediction:
                prob_target_speed = F.softmax(pred_target_speed, dim=1)
            else:
                prob_target_speed = pred_target_speed

            self.nets[0].visualize_model(
                self.save_path,
                self.step,
                tick_data['rgb'],
                lidar_bev,
                tick_data['target_point'],
                pred_wp,
                target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
                pred_semantic=pred_semantic,
                pred_bev_semantic=pred_bev_semantic,
                pred_depth=pred_depth,
                pred_checkpoint=pred_checkpoint,
                pred_speed=prob_target_speed,
                pred_target_speed_scalar=pred_target_speed_scalar,
                pred_bb=bbs_vehicle_coordinate_system,
                gt_speed=gt_velocity,
                gt_wp=pred_wp_1,
                wp_selected=wp_selected)

        if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
            pred_checkpoints = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
            steer, throttle, brake = self.nets[0].control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)
        elif self.config.use_wp_gru and not self.config.inference_direct_controller:
            steer, throttle, brake = self.nets[0].control_pid(self.pred_wp,
                                                                gt_velocity,
                                                                tuned_aim_distance=bool(self.tuned_aim_distance))
        else:
            raise ValueError('An output representation was chosen that was not trained.')

        # 0.1 is just an arbitrary low number to threshold when the car is stopped
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            emergency_stop = False
            if self.config.backbone not in ('aim'):
                # safety check
                safety_box = deepcopy(self.lidar_buffer[-1])

                # z-axis
                safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
                safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

                # y-axis
                safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
                safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

                # x-axis
                safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
                safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
                emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

            if not emergency_stop:
                print('Detected agent being stuck. Step: ', self.step)
                throttle = max(self.config.creep_throttle, throttle)
                brake = False
                self.force_move -= 1
            else:
                print('Creeping stopped by safety box. Step: ', self.step)
                throttle = 0.0
                brake = True
                self.force_move = self.config.creep_duration

        if self.stop_sign_controller:
            if stop_for_stop_sign:
                throttle = 0.0
                brake = True

        if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
            print(f'Stopping after {self.stop_after_meter} meters.')
            throttle = 0.0
            brake = True

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.IS_BENCH2DRIVE:
            # TODO doesn't seem to work
            metric_info = self.get_metric_info()
            self.metric_info[self.step] = metric_info
            if self.save_path is not None and self.step % 1 == 0:
                with open(self.save_path / 'metric_info.json', 'w') as outfile:
                    ujson.dump(self.metric_info, outfile, indent=4)

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        return control