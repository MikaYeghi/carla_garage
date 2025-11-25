import os
import numpy as np
from scipy import ndimage
from typing import Dict
from perception_simplex.safety_layer import SafetyLayer as SafetyLayerPS

def meters_to_pixel(x, y, bev_size=256, world_size=64.0):
    scale = bev_size / world_size
    half = world_size / 2
    px = int((x + half) * scale)
    py = int((half - y) * scale)
    return px, py

def obstacle_overlapping_lanes(obstacle, labeled_lanes):
    xmin, ymin, _ = obstacle['bbox_min']
    xmax, ymax, _ = obstacle['bbox_max']

    px_min, py_max = meters_to_pixel(xmin, ymin)
    px_max, py_min = meters_to_pixel(xmax, ymax)

    px_min = np.clip(px_min, 0, 256)
    px_max = np.clip(px_max, 0, 256)
    py_min = np.clip(py_min, 0, 256)
    py_max = np.clip(py_max, 0, 256)

    region = labeled_lanes[py_min:py_max, px_min:px_max]
    labels = np.unique(region)
    return labels

class SafetyLayerSS(SafetyLayerPS):
    def __init__(self, return_half = True, left_fov = False, threshold_deg = 10, min_obs_height = 0.2, max_dist_lidar = 85, angle_clustering_thresh = 5, min_points_cluster = 2, iou_thresh = 0.75, clustering_method = "original", a_brake_max = 7, latency_max = 0.01, d_margin = 0.1, dt = 0.1, ego_length = 5.02, ego_width = 2.13):
        super().__init__(return_half, left_fov, threshold_deg, min_obs_height, max_dist_lidar, angle_clustering_thresh, min_points_cluster, iou_thresh, clustering_method, a_brake_max, latency_max, d_margin, dt, ego_length, ego_width)

        # TODO: implement selection logic between the PS, M2S, S2M, SS fault handlers
        self.fault_handler_type = self.get_fault_handler()
        print(f"Fault Handler: {self.fault_handler_type}")

    def get_fault_handler(self):
        valid_handlers = {"PS", "M2S", "S2M", "SS"}

        fault_handler_env = os.environ.get("FAULT_HANDLER", "").upper()

        if fault_handler_env in valid_handlers:
            return fault_handler_env
        
        # Fallback to PS
        return "PS"

    def fault_handler(self, 
                      control_mission, 
                      faulty_detections, 
                      collision_risks, 
                      speed,
                      pred_bev_semantic=None,
                      safety_layer_detections=[],
                      road_id=1,
                      vehicles_id=9
        ) -> tuple[Dict, bool]:
        if self.fault_handler_type == "PS":
            return self.fault_handler_PS(control_mission, faulty_detections, collision_risks, speed)
        elif self.fault_handler_type == "M2S":
            return self.fault_handler_M2S(
                control_mission,
                faulty_detections,
                collision_risks,
                speed,
                pred_bev_semantic=pred_bev_semantic,
                safety_layer_detections=safety_layer_detections,
                road_id=road_id,
                vehicles_id=vehicles_id
            )
        elif self.fault_handler_type == "S2M":
            raise NotImplementedError
        elif self.fault_handler_type == "SS":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid fault handler type {self.fault_handler_type}. Expected one of: PS, M2S, S2M, SS.")

    def fault_handler_PS(self, control_mission, faulty_detections, collision_risks, speed) -> tuple[Dict, bool]:
        """
        Implement the simplex logic.

        Args:
            control_mission (dict): Control commands from the mission system.
            faulty_detections (List[bool]): List of faulty detections. True means the detection is missed by the mission layer.
            collision_risks (List[bool]): List of collision risks with each obstacle detected by the safety layer. True if there is a collision risk.
            speed (float): Current speed of the vehicle.

        Returns:
            dict: Modified control commands with limited velocity.
        """
        # Check if there is a risk of collision with any of the faulty obstacles
        for is_faulty, collision_risk in zip(faulty_detections, collision_risks):
            if is_faulty and collision_risk:
                return self.override_control(), 3
        
        # Otherwise, limit velocity if needed
        control_final, safety_override = self.limit_velocity(control_mission, speed)
        if safety_override:
            return control_final, 2
        else:
            return control_final, 0

    def fault_handler_M2S(self, 
                      control_mission, 
                      faulty_detections, 
                      collision_risks, 
                      speed,
                      pred_bev_semantic=None,
                      safety_layer_detections=[],
                      road_id=1,
                      vehicles_id=9
        ) -> tuple[Dict, bool]:
        """
        Implement the simplex logic.

        Args:
            control_mission (dict): Control commands from the mission system.
            faulty_detections (List[bool]): List of faulty detections. True means the detection is missed by the mission layer.
            collision_risks (List[bool]): List of collision risks with each obstacle detected by the safety layer. True if there is a collision risk.
            speed (float): Current speed of the vehicle.

        Returns:
            dict: Modified control commands with limited velocity.
        """
        # Use the PS fault handler if no BEV semantic map is provided
        if pred_bev_semantic is None:
            print("WARNING: No BEV semantic map provided, rolling back to the PS Fault Handler.")
            return self.fault_handler_PS(control_mission, faulty_detections, collision_risks, speed)
        
        # Pre-process the BEV semantic map
        bev_semantic_map = pred_bev_semantic.squeeze().argmax(axis=0).clone().detach().cpu().numpy()

        # Extract the lanes as the road labels
        lanes_map = bev_semantic_map == road_id
        # vehicles_map = bev_semantic_map == vehicles_id
        # lanes_map = lanes_map + vehicles_map # consider vehicles as part of lanes for connectivity

        # Identify each lane as a blob that does not touch other blobs
        labeled_lanes, num_labels = ndimage.label(lanes_map)
        
        # Identify which lane is touching the ego vehicle
        ego_vehicle_lane_id = self.get_ego_vehicle_lane(labeled_lanes)

        # Rotate the BEV segmentation map
        labeled_lanes_aligned = np.rot90(labeled_lanes, k=-1)

        # Check where the faulty and collision risk obstacles are located and respond respectively. Override levels:
        # 0: no override needed, return limit_velocity (Zone 3)
        # 1: release throttle, no braking (Zone 2)
        # 3: brake (Zone 1)
        response_ids = [0 for _ in range(len(collision_risks))]
        for i, (is_faulty, is_risky, obstacle) in enumerate(zip(faulty_detections, collision_risks, safety_layer_detections)):
            if is_faulty and is_risky:
                overlap_ids = obstacle_overlapping_lanes(obstacle, labeled_lanes_aligned)
                
                # Check which zone the obstacle falls into
                if any(overlap_ids == ego_vehicle_lane_id): # Zone 1: brake
                    response_id = 3
                elif any(overlap_ids != 0):                 # Zone 2: release the throttle, no braking
                    response_id = 1
                else:                                       # Zone 3: no override action
                    response_id = 0
                
                # Assign the response ID
                response_ids[i] = response_id

        # Extract the maximum response_id
        if len(response_ids) > 0:
            max_response_id = max(response_ids)
        else:
            max_response_id = 0

        # Assign the override
        # 3: emergency braking
        # 2: velocity control braking (no emergency)
        # 1: release throttle (soft emergency)
        # 0: no override
        if max_response_id == 3:
            return self.override_control(), 3
        else:
            control_final, safety_override = self.limit_velocity(control_mission, speed)
            if safety_override:
                return control_final, 2
            elif max_response_id == 1:
                return self.soft_override_control(), 1
            else:
                return control_mission, 0

    def get_ego_vehicle_lane(self, labeled_lanes):
        ego_y = labeled_lanes.shape[0] // 2
        ego_x = labeled_lanes.shape[1] // 2
        ego_vehicle_lane_id = labeled_lanes[ego_y, ego_x]
        if ego_vehicle_lane_id == 0:
            print("WARNING: ego vehicle is not standing on any lane!")
        return ego_vehicle_lane_id
    
    def soft_override_control(self):
        """
        Override vehicle control command.

        Returns:
            dict: Control commands to override the vehicle's current control.
        """
        override_control = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0
        }
        return override_control