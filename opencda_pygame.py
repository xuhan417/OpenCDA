#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

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
from opencda.core.common.pygame_render import World, HUD, KeyboardControl, pygame_loop

# from opencda.core.plan.behavior_agent_carla import BehaviorAgent  # pylint: disable=import-error
# from opencda.core.plan.roaming_agent import RoamingAgent  # pylint: disable=import-error
# from opencda.core.plan.basic_agent import BasicAgent  # pylint: disable=import-error


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def pygame_loop(args):
    """ Main loop for agent"""
    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # pygame clock 
        clock = pygame.time.Clock()

        connected_text = f'Connection established with OpenCDA vehicle...'
        world.hud.notification(connected_text, seconds=1.0)
        count = 0

        while True:
            count += 1
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

                # as soon as the server is ready continue!
                world.world.wait_for_tick(10.0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                
            else:
                # agent.update_information(world)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()


    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def pygame_main():
    """Main method"""

    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")

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
        default='1920x1080',
        help='window resolution (default: 1920x1080)')
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
    parser.add_argument("--num_screens",
                        default=1,
                        type=int,
                        help='Number of screens rendered by pygame.')

    # parse the arguments and return the result
    args = parser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        pygame_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    pygame_main()