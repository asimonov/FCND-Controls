"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0
MAX_PITCH = 20.0
MAX_ROLL = MAX_PITCH

class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""

        body_t_rise = 0.35
        body_delta = 0.7
        body_omega_n = 1.57 / body_t_rise
        self.body_rate_Kp = body_omega_n ** 2
        print('body: Kp={}'.format(self.body_rate_Kp))

        roll_t_rise = 0.55
        roll_delta = 0.9
        roll_omega_n = 1.57 / roll_t_rise
        self.roll_Kp = roll_omega_n ** 2
        print('roll=pitch: Kp={}'.format(self.roll_Kp))
        self.pitch_Kp = self.roll_Kp

        yaw_t_rise = 1.0 # yaw controller needs to be more relaxed. probably because of bigger inertia around z
        yaw_delta = 0.9
        yaw_omega_n = 1.57 / yaw_t_rise
        self.yaw_Kp = yaw_omega_n ** 2
        print('yaw: Kp={}'.format(self.yaw_Kp))

        alt_t_rise = 0.7
        alt_delta = 0.9
        alt_omega_n = 1.57 / alt_t_rise
        self.alt_Kp = alt_omega_n ** 2
        self.alt_Kd = 2 * alt_delta * alt_omega_n
        print('alt: Kp={} Kd={}'.format(self.alt_Kp, self.alt_Kd))

        pos_t_rise = 0.85
        pos_delta = 0.9
        pos_omega_n = 1.57 / pos_t_rise
        self.pos_Kp = pos_omega_n ** 2
        self.pos_Kd = 2 * pos_delta * pos_omega_n
        print('pos: Kp={} Kd={}'.format(self.pos_Kp, self.pos_Kd))


    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]
        
        
        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                
                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]
            
        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)
        
        return (position_cmd, velocity_cmd, yaw_cmd)


    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        err = body_rate_cmd - body_rate
        u_bar = self.body_rate_Kp * err * MOI
        return np.clip(u_bar, -MAX_TORQUE, MAX_TORQUE)



    # def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
    #     """ Generate the rollrate and pitchrate commands in the body frame
    #
    #     Args:
    #         target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
    #         attitude: 3-element numpy array (roll, pitch, yaw) in radians
    #         thrust_cmd: vehicle thruts command in Newton
    #
    #     Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
    #     """
    #     if thrust_cmd < 1e-5:
    #         return np.array([0.0, 0.0])
    #     R = euler2RM(attitude[0], attitude[1], attitude[2])
    #
    #     b_x_c = acceleration_cmd[0] / thrust_cmd # desired
    #     b_x = R[0, 2]
    #     b_x_err = b_x_c - b_x
    #     b_x_p_term = self.roll_Kp * b_x_err
    #
    #     b_y_c = acceleration_cmd[1] / thrust_cmd # desired
    #     b_y = R[1, 2]
    #     b_y_err = b_y_c - b_y
    #     b_y_p_term = self.pitch_Kp * b_y_err
    #
    #     b_x_commanded_dot = b_x_p_term
    #     b_y_commanded_dot = b_y_p_term
    #
    #     rot_mat1 = np.array([[R[1, 0], -R[0, 0]],
    #                          [R[1, 1], -R[0, 1]]]) / R[2, 2]
    #
    #     rot_rate = np.matmul(rot_mat1, np.array([b_x_commanded_dot, b_y_commanded_dot]).T)
    #     p_c = np.clip(rot_rate[0], -MAX_ROLL/180.0, MAX_ROLL/180.0)
    #     q_c = np.clip(rot_rate[1], -MAX_PITCH/180.0, MAX_PITCH/180.0)
    #
    #     return np.array([p_c, q_c])

    def roll_pitch_controller(self,
                              acceleration_cmd,
                              attitude,
                              thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        # Calculate rotation matrix
        R = euler2RM(attitude[0], attitude[1], attitude[2])
        c_d = thrust_cmd / DRONE_MASS_KG

        if thrust_cmd > 0.0:
            target_R13 = -min(max(acceleration_cmd[0].item() / c_d, -1.0), 1.0)
            target_R23 = -min(max(acceleration_cmd[1].item() / c_d, -1.0), 1.0)

            p_cmd = (1 / R[2, 2]) * \
                    (-R[1, 0] * self.roll_Kp * (R[0, 2] - target_R13) + \
                     R[0, 0] * self.pitch_Kp * (R[1, 2] - target_R23))
            q_cmd = (1 / R[2, 2]) * \
                    (-R[1, 1] * self.pitch_Kp * (R[0, 2] - target_R13) + \
                     R[0, 1] * self.pitch_Kp * (R[1, 2] - target_R23))
        else:  # Otherwise command no rate
            p_cmd = 0.0
            q_cmd = 0.0
        return np.array([p_cmd, q_cmd])


    def altitude_control(self,
                         altitude_cmd,
                         vertical_velocity_cmd,
                         altitude,
                         vertical_velocity,
                         attitude,
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """

        err = altitude_cmd - altitude
        vel_err = vertical_velocity_cmd - vertical_velocity
        R = euler2RM(attitude[0], attitude[1], attitude[2])

        u1_cmd = ((self.alt_Kp * err + self.alt_Kd * vel_err + DRONE_MASS_KG*acceleration_ff) / R[2,2])

        return np.clip(u1_cmd, 0.0, MAX_THRUST)

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        err = yaw_cmd - yaw
        yaw_r = self.yaw_Kp * err
        return yaw_r

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command
            
        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        err = local_position_cmd - local_position
        err_vel = local_velocity_cmd - local_velocity
        acc_cmd = self.pos_Kp * err + self.pos_Kd * err_vel + DRONE_MASS_KG*acceleration_ff

        return acc_cmd





