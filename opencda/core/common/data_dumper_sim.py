# -*- coding: utf-8 -*-
"""
Dumping sensor data.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import cv2
import open3d as o3d
import numpy as np
import yaml

from opencda.core.common.misc import get_speed
from opencda.core.sensing.perception import sensor_transformation as st
from opencda.scenario_testing.utils.yaml_utils import save_yaml


class SimDataDumper(object):
    """
    Customized Data dumper class for openCDA simulator integration.

    """

    def __init__(self,
                 vehicle,
                 sim_controller,
                 ego_ttc,
                 is_tailgate_warning,
                 lane_invasion_sensor,
                 collision_sensor,
                 save_time):

        self.vehicle = vehicle
        self.sim_controller = sim_controller
        self.ego_ttc = ego_ttc
        self.is_tailgate_warning = is_tailgate_warning
        self.lane_invasion_sensor = lane_invasion_sensor
        self.collision_sensor = collision_sensor

        self.save_time = save_time

        current_path = os.path.dirname(os.path.realpath(__file__))
        self.save_parent_folder = \
            os.path.join(current_path,
                         '../../../sim_data_dumping',
                         save_time,
                         'opencda_ego_vehicle')

        if not os.path.exists(self.save_parent_folder):
            os.makedirs(self.save_parent_folder)

        self.count = 0

    def run_step(self, reduce_frequency=False):
        """
        Dump data at running time.

        """
        # increment self count 
        self.count += 1

        # 10hz
        if self.count % 2 != 0 and reduce_frequency:
            return

        # save data 
        self.save_yaml_file(self.vehicle,
                            self.count)

    # helper function 
    def is_close_to_destination(self, ego_pos, end_waypoint_pos_x, end_waypoint_pos_y):
            """
            Check if the current ego vehicle's position is close to destination

            Returns
            -------
            flag : boolean
                It is True if the current ego vehicle's position is close to destination

            """
            flag = abs(ego_pos.x - end_waypoint_pos_x) <= 12 and \
                   abs(ego_pos.y - end_waypoint_pos_y) <= 12
            return flag

    # yaml helper function 
    def numpy_to_python(self, data):
        if isinstance(data, dict):
            return {key: self.numpy_to_python(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.numpy_to_python(element) for element in data]
        elif isinstance(data, np.bool_):  # Convert NumPy boolean to Python bool
            return bool(data)
        elif isinstance(data, np.integer):  # Convert NumPy int to Python int
            return int(data)
        elif isinstance(data, np.floating):  # Convert NumPy float to Python float
            return float(data)
        else:
            return data

    def save_yaml_file(self,
                       veh,
                       count):
        """
        Save ego vehicle positions/spped, true ego position,
        predicted ego position, sensor transformations.

        Parameters
        ----------
        perception_manager : opencda object
            OpenCDA perception manager.

        localization_manager : opencda object
            OpenCDA localization manager.

        behavior_agent : opencda object
            OpenCDA behavior agent.
        """
        frame = count

        dump_yml = {}
        vehicle_dict = {}

        # 1. vehicle info
        veh_pos = veh.get_transform()
        veh_bbx = veh.bounding_box
        # note: this calculates speed magnitude in km/h
        veh_speed = get_speed(veh)
        # update vehicle dict
        vehicle_dict.update({
            "vehicle_id": veh.id,
            "role_name": veh.attributes['role_name'], 
            "bp_id": veh.type_id,
            "location": [veh_pos.location.x,
                         veh_pos.location.y,
                         veh_pos.location.z],
            "center": [veh_bbx.location.x,
                       veh_bbx.location.y,
                       veh_bbx.location.z],
            "angle": [veh_pos.rotation.roll,
                      veh_pos.rotation.yaw,
                      veh_pos.rotation.pitch],
            "extent": [veh_bbx.extent.x,
                       veh_bbx.extent.y,
                       veh_bbx.extent.z],
            "speed": veh_speed # km/h
        })
        # update output dict
        dump_yml.update({'vehicle': vehicle_dict})

        # 2. simulaiton controller info
        sim_control_dict = {}
        sim_control_dict['human_take_over'] = self.sim_controller.human_take_over
        sim_control_dict['throttle'] = self.sim_controller._control.throttle
        sim_control_dict['steer'] = self.sim_controller._control.steer
        sim_control_dict['brake'] = self.sim_controller._control.brake
        sim_control_dict['hand_brake'] = self.sim_controller._control.hand_brake
        sim_control_dict['reverse'] = self.sim_controller._control.reverse
        # update output dict
        dump_yml.update({'simulation controller signal': sim_control_dict})

        # 3. tailgate warning 
        ego_following_dict = {}
        ego_following_dict['ego ttc'] = self.ego_ttc
        ego_following_dict['taigate warning'] = self.is_tailgate_warning
        # cast to local python vals
        dump_yml.update({'ego car following data': self.numpy_to_python(ego_following_dict)})

        # 4. lane invasion
        ego_lane_position_dict = {}
        # ego_lane_position_dict['is_crossed_lane'] = self.lane_invasion_sensor.is_crossed_lane
        ego_lane_position_dict['warning_text'] = self.lane_invasion_sensor.warning_text
        
        dump_yml.update({'ego lane position data': ego_lane_position_dict})

        # 5. collision sensor 
        collision_sensor_dict = {'is_vehicle_collided':self.collision_sensor.is_collided}
        dump_yml.update({'collision sensor': collision_sensor_dict})

        # 6. end simulation 
        destination_x = 599.10
        destination_y = 237.73
        is_near_end = self.is_close_to_destination(veh_pos.location, destination_x, destination_y)
        dump_yml.update({'is simulation end': is_near_end})

        # 7. save yaml file
        yml_name = '%06d' % frame + '.yaml'
        save_path = os.path.join(self.save_parent_folder,
                                 yml_name)
        save_yaml(dump_yml, save_path)

    @staticmethod
    def matrix2list(matrix):
        """
        To generate readable yaml file, we need to convert the matrix
        to list format.

        Parameters
        ----------
        matrix : np.ndarray
            The extrinsic/intrinsic matrix.

        Returns
        -------
        matrix_list : list
            The matrix represents in list format.
        """

        assert len(matrix.shape) == 2
        return matrix.tolist()