# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os
import sys

from omegaconf import OmegaConf
from opencda.version import __version__
# --- pygame imports ---
import collections
import datetime
import glob
import logging
import math
import random
import re
import json

import carla
import pygame
from carla import ColorConverter as cc
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time

def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    
    # add opencda arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('-v', "--version", type=str, default='0.9.14',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.14')
    # add argument of whether use pygame rendering interface
    parser.add_argument("--pygame_render",
                        action='store_true',
                        help='whether to use Pygame to render the simulation with a first person view. '
                             'Inside the ego vehicle. Note the ego vehicle needs to have a proper role_name.')
    # add argument of whether use pygame rendering interface
    parser.add_argument("--sim_wheel",
                        action='store_true',
                        help='whether to use simulation steering wheel as input device.')
    parser.add_argument("--sim_wheel_config_path",
                        action='store_true',
                        help='config file path for the sim wheel.')
    parser.add_argument("--num_screens",
                        default=1,
                        type=int,
                        help='Number of screens rendered by pygame.')

    # add pygame arguments to parser 
    parser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    parser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    parser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='4800x1080',
        help='window resolution (default: 3840x720)')

    # help='window resolution (default: 5760x1620)')
    # 3840x720
    parser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    parser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    parser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    parser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    parser.add_argument(
        '--sync',
        action='store_false',
        help='Activate synchronous mode execution')

    # parse the arguments and return the result
    opt = parser.parse_args()

    # print pygame info
    if opt.pygame_render:
        opt.width, opt.height = [int(x) for x in opt.res.split('x')]
    # return args
    return opt

def main():
    # parse the arguments
    opt = arg_parse()
    # print the version of OpenCDA
    print("OpenCDA Version: %s" % __version__)
    # sim wheel 
    if opt.sim_wheel:
        opt.sim_wheel_config_path = r'/home/ccorreaj/OpenCDA/opencda/scenario_testing/utils/'

    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)

    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # import the testing script
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')
    # run the scenario testing
    scenario_runner(opt, scene_dict)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
