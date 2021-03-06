# -*- coding: utf-8 -*-
"""
Starter code for the controls project.
This is the solution of the backyard flyer script, 
modified for all the changes required to get it working for controls.
"""

import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from unity_drone import UnityDrone
from controller import NonlinearController
from udacidrone.connection import MavlinkConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class ControlsFlyer(UnityDrone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        self.controller = NonlinearController()
        # debug counters
        self.attitude_cnt = 0
        self.gyro_cnt = 0
        self.velocity_cnt = 0
        self.start_time = time.time()

        # register all your callbacks here
        self.register_callback(MsgID.STATE,          self.state_callback)
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.ATTITUDE,       self.attitude_callback)
        self.register_callback(MsgID.RAW_GYROSCOPE,  self.gyro_callback)

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                # PLAN PATH
                print("plan trajectory")
                #self.all_waypoints = self.calculate_box()
                time_mult = 0.5
                (self.position_trajectory,
                 self.time_trajectory,
                 self.yaw_trajectory) = self.load_test_trajectory(time_mult)
                self.all_waypoints = self.position_trajectory.copy()
                self.waypoint_number = -1
                # EXECUTE PATH: start
                print("execute trajectory") # not quite. that just tracks where we should be in time, but does not drive the controls
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if time.time() > self.time_trajectory[self.waypoint_number]:
                if len(self.all_waypoints) > 0:
                    # EXECUTE PATH: continue
                    self.waypoint_transition()
                    #pass
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        # EXECUTE PATH: finish
                        self.landing_transition()

    def attitude_callback(self):
        if self.flight_state == States.WAYPOINT:
            self.attitude_cnt += 1
            #self.attitude_controller()
            self.thrust_cmd = self.controller.altitude_control(
                -self.local_position_target[2],
                -self.local_velocity_target[2],
                -self.local_position[2],
                -self.local_velocity[2],
                self.attitude,
                9.81)
            roll_pitch_rate_cmd = self.controller.roll_pitch_controller(
                self.local_acceleration_target[0:2],
                self.attitude,
                self.thrust_cmd)
            yawrate_cmd = self.controller.yaw_control(
                self.attitude_target[2],
                self.attitude[2])
            self.body_rate_target = np.array([roll_pitch_rate_cmd[0], roll_pitch_rate_cmd[1], yawrate_cmd])

    def gyro_callback(self):
        if self.flight_state == States.WAYPOINT:
            #self.bodyrate_controller()
            # use body_rate_target set in attitude callback
            # this controller runs a faster loop

            self.gyro_cnt += 1
            if self.gyro_cnt % 20 == 0:
                print('time: {:.4f} secs'.format(time.time()-self.start_time))
                print('gyro {}, att {}, vel {}'.format(self.gyro_cnt, self.attitude_cnt, self.velocity_cnt))

            moment_cmd = self.controller.body_rate_control(self.body_rate_target, self.gyro_raw)

#            print("alt {:.4f}/{:.4f}; alt_vel {:.4f}/{:.4f}; thrust {:.4f}".format(
#                self.local_position[2], self.local_position_target[2],
#                self.local_velocity[2], self.local_velocity_target[2],
#                self.thrust_cmd))

#            print("roll {:.4f}/{:.4f}; pitch {:.4f}/{:.4f}; yaw {:.4f}/{:.4f}; moments {:.4f}, {:.4f}, {:.4f}; thrust {:.4f}".format(
#                self.attitude[0], self.attitude_target[0],
#                self.attitude[1], self.attitude_target[1],
#                self.attitude[2], self.attitude_target[2],
#                moment_cmd[0], moment_cmd[1], moment_cmd[2],
#                self.thrust_cmd
#            )
#            )

            self.cmd_moment(moment_cmd[0],
                               moment_cmd[1],
                               moment_cmd[2],
                               self.thrust_cmd)

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            # landed? disarm
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()
        if self.flight_state == States.WAYPOINT:
            # in mission? control position and speed
            #self.position_controller()
            self.velocity_cnt += 1

            (self.local_position_target,
             self.local_velocity_target,
             yaw_cmd) = self.controller.trajectory_control(
                             self.position_trajectory,
                             self.yaw_trajectory,
                             self.time_trajectory, time.time())
#            print('time: {:.4f}; target position: {:.4f}/{:.4f}; target vel: {:.4f}/{:.4f}'.format(time.time() - self.start_time,
#                                                                           self.local_position_target[0], self.local_position_target[1],
#                                                                           self.local_velocity_target[0], self.local_velocity_target[1]))

            self.attitude_target = np.array((0.0, 0.0, yaw_cmd))

            acceleration_cmd = self.controller.lateral_position_control(
                self.local_position_target[0:2],
                self.local_velocity_target[0:2],
                self.local_position[0:2],
                self.local_velocity[0:2])
            #acceleration_cmd = np.array([0.0, 0.0])
#            print("              acc:{:.4f}/{:.4f} --- pos diff:{:.4f}/{:.4f} --- vel diff:{:.4f}/{:.4f}".format(
#                acceleration_cmd[0], acceleration_cmd[1],
#                self.local_position_target[0] - self.local_position[0], self.local_position_target[1] - self.local_position[1],
#                self.local_velocity_target[0] - self.local_velocity[0], self.local_velocity_target[1] - self.local_velocity[1]
#            ))
            self.local_acceleration_target = np.array([acceleration_cmd[0],
                                                       acceleration_cmd[1],
                                                       0.0])

    def arming_transition(self):
        print("arming transition")
        self.take_control()
        self.arm()
        # set the current location to be the home position
        self.set_home_position(self.global_position[0],
                               self.global_position[1],
                               self.global_position[2])  

        self.flight_state = States.ARMING

    def takeoff_transition(self):
        print("takeoff transition")
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        self.waypoint_number = self.waypoint_number + 1
        self.target_position = self.all_waypoints.pop(0)
        print('time: {:.4f}; planned waypoint position: {}, {}'.format(time.time() - self.start_time, self.target_position[0], self.target_position[1]))
        self.flight_state = States.WAYPOINT


    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print("manual transition")
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        # self.connect()

        print("starting connection")
        # self.connection.start()

        super().start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://127.0.0.1:5760')
    drone = ControlsFlyer(conn)
    time.sleep(2)
    drone.start()
    drone.print_mission_score()
