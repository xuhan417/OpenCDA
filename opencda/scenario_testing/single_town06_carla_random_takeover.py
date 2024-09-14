# -*- coding: utf-8 -*-
"""
Scenario testing: single vehicle behavior in intersection
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import json

import carla
import pygame
import numpy as np
from carla import ColorConverter as cc
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

from opencda.version import __version__
import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time

# pygame render
from opencda.core.common.pygame_render import pygame_loop

# multi-processing
from multiprocessing import Process, Queue, get_context, shared_memory
import multiprocessing
# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def run_scenario(opt, scenario_params):
    try:
        # init simulation tick count 
        tick = 0
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_intersection_town06_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()

        # multi-process
        ctx = get_context('spawn')
        input_queue = ctx.Queue(maxsize=1)
        output_queue = ctx.Queue(maxsize=1)

        # Set up shared memory for communication
        shared_array_size = (2,)  # Example size for throttle, steer, brake, reverse
        shm = shared_memory.SharedMemory(create=True, size=np.prod(shared_array_size) * np.dtype(np.float64).itemsize)
        shared_array = np.ndarray(shared_array_size, dtype=np.float64, buffer=shm.buf)
        shared_array[:] = [0.0, 0.0] 

        # Start Pygame subprocess
        pygame_process = ctx.Process(target=pygame_loop, 
                                     args=(input_queue, output_queue, shm.name, shared_array_size))
        
        # update the warning setting 
        opt.display_warning = True
        # put opt to input queue
        input_queue.put(opt)
        human_takeover = False
        reduce_speed = False

        # run steps
        while True:
            scenario_manager.tick()
            # increment simulation tick 
            tick += 1

            # update shared memory 
            ego_ttc = single_cav_list[0].agent.ttc
            shared_array[0] = ego_ttc

            # pygame rendering 
            if not input_queue.empty() and \
                not pygame_process.is_alive() and\
                tick >= 2:
                pygame_process.start()
                print('start multi-processing!!')

            # catch output queue
            if not output_queue.empty():
                human_controls = output_queue.get()
                human_takeover = human_controls['human_take_over']
                # print('human control signal is: ' + str(human_controls))

            # ---------- tailgate behavior -------------
            human_takeover_sec = random.uniform(1, 100) # random float from 1 to 100 with uniform distribution
            human_takeover_sec = 5 # hard code for debug purpose
            sim_dt = scenario_params['world']['fixed_delta_seconds']
            # factor to reduce the look-ahead dist
            reducing_factor = 0.35

            # activate tailgate when reaches desired time
            if tick*sim_dt == human_takeover_sec:
                print('[OpenCDA Side]: Reduce collision time, human takeover !!!')
                # reduce safety distance 
                single_cav = single_cav_list[0].agent.reduce_following_dist(reducing_factor)
                # check collision checker state 
                new_collision_time = single_cav_list[0].agent._collision_check.time_ahead
                print('New collision checker is enabled with: ' + \
                        str(new_collision_time) + 'second ahead time! ')

            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=50),
                carla.Rotation(
                    pitch=-
                    90)))

            single_cav = single_cav_list[0]
            single_cav.update_info()
            control = single_cav.run_step()
            if human_takeover:
                manual_control = carla.VehicleControl()
                manual_control.throttle = human_controls['throttle']
                manual_control.steer = human_controls['steer']
                manual_control.brake = human_controls['brake']
                manual_control.reverse = human_controls['reverse']
                single_cav.vehicle.apply_control(manual_control)
            elif single_cav.agent.is_close_to_destination():
                print('Simulation is Over. ')
                brake_control = carla.VehicleControl(brake=1.0)
                single_cav.vehicle.apply_control(brake_control)

            else:
                single_cav.vehicle.apply_control(control)

            # logic to maintain background vehicle speed
            # this is specific to town06, used to reduce 90km/h to 50km/h
            leading_v = bg_veh_list[0]
            trailing_v = bg_veh_list[-1]
            speed_limit = max(leading_v.get_speed_limit(), \
                              trailing_v.get_speed_limit()) 
            if speed_limit >= 75 and not reduce_speed:
                print('set reduce speed to true.')
                reduce_speed = True
            if reduce_speed:
                print('reduce speed limit to all TM to 35%')
                # traffic_manager.global_percentage_speed_difference(90)
                for v in bg_veh_list:
                    traffic_manager.vehicle_percentage_speed_difference(v, 40)
                tm_spd = leading_v.get_velocity()
                tm_kmh = math.sqrt((tm_spd.x**2 + tm_spd.y**2 + tm_spd.z**2))*3.6
                print('The current tm speed is: ' + str(tm_kmh))

    finally:
        input_queue.put(None)  # Signal the GPU process to terminate
        pygame_process.join()
        eval_manager.evaluate()

        # Clean up shared memory
        shm.close()
        shm.unlink()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()

