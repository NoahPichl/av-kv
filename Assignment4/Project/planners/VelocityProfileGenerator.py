# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(glob.glob('%s/../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        PATH,
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ==============================================================================
# -- Imports  ---------------------------------------------------------
# ==============================================================================
import time
from typing import Tuple, List

import numpy as np

from .Structures import State, Maneuver, TrajectoryPoint, PathPoint
from .utils import path_point_distance, get_magnitude
# =============================================================================
# -- Velocity Profile Generator  ----------------------------------------------
# =============================================================================
dbl_epsilon = sys.float_info.epsilon

class VelocityProfileGenerator(object):
    """
        This class computes a velocity trajectory from a starting speed to a desired
        speed. It works in unison with the Behavioral plannner  as it needs to build a
        velocity profile for each of the states that the vehicle can be in.
        In the "Follow_lane" state we need to either speed up or speed down to maintain
        a speed target. In the "decel_to_stop" state we need to create a profile that
        allows us to decelerate smoothly to a stop line.

        The order of precedence for handling these cases is stop sign handling and then
        nominal lane maintenance. In a real velocity planner you would need to handle
        the coupling between these states, but for simplicity this project can be
        implemented by isolating each case.

        For all trajectories, the required acceleration is given by _a_max (confortable
        accel).
        Look at the structs.h for details on the types of manuevers/states that the
        behavior planner can be in.
    """

    def __init__(self):
        self._time_gap = None
        self._a_max = None
        self._slow_speed = None

    def setup(self, time_gap : float, a_max : float, slow_speed : float):
        self._time_gap = time_gap
        self._a_max = a_max
        self._slow_speed = slow_speed
    
    def generate_trajectory(self, spiral : List[PathPoint], desired_speed : float, 
                            ego_state : State, lead_car_state : State, maneuver : Maneuver):
        
        trajectory = []
        start_speed = get_magnitude(ego_state.velocity)

        # Generate a trapezoidal trajectory to decelerate to stop.
        if maneuver == Maneuver.DECEL_TO_STOP:
            trajectory = self.decelerate_trajectory(spiral, start_speed) 
        
        # If we need to follow the lead vehicle, make sure we decelerate to its speed
        # by the time we reach the time gap point.
        elif maneuver == Maneuver.FOLLOW_VEHICLE:
            trajectory = self.follow_trajectory(spiral, start_speed, desired_speed, lead_car_state)

        # Otherwise, compute the trajectory to reach our desired speed.
        else:
            trajectory = self.nominal_trajectory(spiral, start_speed, desired_speed)

        # Interpolate between the zeroth state and the first state.
        # This prevents the controller from getting stuck at the zeroth state
        if len(trajectory)>1:
            x = (trajectory[1].path_point.x - trajectory[0].path_point.x)*0.1+trajectory[0].path_point.x
            y = (trajectory[1].path_point.y - trajectory[0].path_point.y)*0.1+trajectory[0].path_point.y
            z = (trajectory[1].path_point.z - trajectory[0].path_point.z)*0.1+trajectory[0].path_point.z
            v = (trajectory[1].v - trajectory[0].v)*0.1+trajectory[0].v

            p = PathPoint(x,y,z,0,0,0,0,0)
            trajectory[0] = TrajectoryPoint(p,v,0,0)
        
        return trajectory
    
    def decelerate_trajectory(self, spiral: List[PathPoint], start_speed: float):
        """
        Computes a velocity trajectory for deceleration to a full stop.
        """
        trajectory = []

        # Using d = (v_f^2 - v_i^2) / (2 * a)
        decel_distance = self.calc_distance(start_speed, self._slow_speed, -self._a_max)
        brake_distance = self.calc_distance(self._slow_speed, 0, -self._a_max)

        path_length = 0
        stop_index = len(spiral) - 1

        for i in range(stop_index):
            path_length += path_point_distance(spiral[i], spiral[i + 1])

        # If the brake distance exceeds the length of the path, then we cannot
        # perform a smooth deceleration and require a harder deceleration. Build the path
        # up in reverse to ensure we reach zero speed at the required time.

        if brake_distance + decel_distance > path_length:
            speeds = []
            vf = 0
            # Let's add the last point, i.e at the stopping line we should have speed
            # 0.0.
            speeds.append(vf)

            # Let's now go backwards until we get to the very beginning of the path
            for i in range(stop_index - 1, -1, -1):
                dist = path_point_distance(spiral[i + 1], spiral[i])
                vi = self.calc_final_speed(vf, -self._a_max, dist)
                if vi > start_speed:
                    vi = start_speed
                # Let's add it
                speeds.append(vi)
                vf = vi

            # At this point we have all the speeds. Now we need to create the
            # trajectory
            time_step = 0
            time = 0

            for i in range(len(speeds) - 1):
                path_point = spiral[i]
                v = speeds[i]
                relative_time = time
                traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
                trajectory.append(traj_point)

                time_step = np.abs(speeds[i] - speeds[i + 1]) / self._a_max
                time += time_step

            # We still need to add the last one
            i = len(spiral) - 1
            path_point = spiral[i]
            v = speeds[i]
            relative_time = time
            traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
            trajectory.append(traj_point)

        # If the brake distance DOES NOT exceed the length of the path
        else:
            brake_index = stop_index
            temp_dist = 0

            # Compute the index at which to start braking down to zero.
            for i in range(stop_index, 0, -1):
                temp_dist += path_point_distance(spiral[i], spiral[i - 1])
                if temp_dist >= brake_distance:
                    brake_index = i-1  # added bias for the range function
                    break

            # Compute the index to stop decelerating to the slow speed.
            decel_index = 0
            temp_dist = 0.0

            for i in range(brake_index):
                temp_dist += path_point_distance(spiral[i], spiral[i + 1])
                if temp_dist >= decel_distance:
                    decel_index = i  # does not need bias since already have it in brake_index
                    break

            # At this point we have all the speeds. Now we need to create the
            # trajectory
            time_step = 0
            time = 0
            vi = start_speed

            # Calculation of the deceleration trajectory
            for i in range(decel_index):
                dist = path_point_distance(spiral[i], spiral[i + 1])
                vf = self.calc_final_speed(vi, -self._a_max, dist)
                vf = max(vf, self._slow_speed)

                path_point = spiral[i]
                v = vi
                relative_time = time
                traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
                trajectory.append(traj_point)

                time_step = np.abs(vf - vi) / self._a_max
                time += time_step
                vi = vf

            # Calculation of constant part of the hybrid trajectory
            for i in range(decel_index, brake_index):
                path_point = spiral[i]
                v = vi
                relative_time = time
                traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
                trajectory.append(traj_point)

                dist = path_point_distance(spiral[i], spiral[i + 1])
                if dist > 0:
                    time_step = dist / vi
                else:
                    time_step = 0
                time += time_step

            # Calculation of deceleration trajectory until stopped
            for i in range(brake_index, stop_index):
                dist = path_point_distance(spiral[i], spiral[i + 1])
                vf = self.calc_final_speed(vi, -self._a_max, dist)

                path_point = spiral[i]
                v = vi
                relative_time = time
                traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
                trajectory.append(traj_point)

                time_step = np.abs(vf - vi) / self._a_max
                time += time_step
                vi = vf

            # Now we just need to add the last point.
            i = stop_index
            path_point = spiral[i]
            v = vi
            relative_time = time
            traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
            trajectory.append(traj_point)

        return trajectory


    def follow_trajectory(self, spiral: List[PathPoint], start_speed: float, desired_speed: float, lead_car_state: State):
        """
        Computes a velocity trajectory for following a lead vehicle.
        """
        trajectory = []

        # Calculate time to collision (TTC)
        rel_speed = start_speed - lead_car_state.speed
        if rel_speed <= 0:
            ttc = np.inf
        else:
            ttc = path_point_distance(spiral[0], lead_car_state.position) / rel_speed

        # Safe distance to maintain from the lead vehicle
        safe_distance = self._time_gap * start_speed

        time_step = 0
        time = 0
        vi = start_speed
        target_speed = min(lead_car_state.speed, desired_speed)

        for i in range(len(spiral) - 1):
            dist = path_point_distance(spiral[i], spiral[i + 1])
            vf = self.calc_final_speed(vi, self._a_max, dist)
            vf = min(target_speed, vf)

            path_point = spiral[i]
            v = vi
            relative_time = time
            traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
            trajectory.append(traj_point)

            time_step = np.abs(vf - vi) / self._a_max
            time += time_step
            vi = vf

        i = len(spiral) - 1
        path_point = spiral[i]
        v = vi
        relative_time = time
        traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
        trajectory.append(traj_point)

        return trajectory
 
    
    def nominal_trajectory(self, spiral: List[PathPoint], start_speed: float, desired_speed: float):
        """
        Computes a velocity trajectory for nominal speed tracking, a.k.a. Lane Follow
        or Cruise Control
        """
        trajectory = []
        accel_distance = 0

        if desired_speed < start_speed:
            accel_distance = self.calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = self.calc_distance(start_speed, desired_speed, self._a_max)

        ramp_end_index = 0
        distance = 0

        # Compute the index at which to start braking down to zero.
        for i in range(len(spiral) - 1):
            distance += path_point_distance(spiral[i], spiral[i + 1])
            if distance >= accel_distance:
                ramp_end_index = i+1
                break

        time_step = 0
        time = 0
        vi = start_speed

        for i in range(ramp_end_index):
            dist = path_point_distance(spiral[i], spiral[i + 1])
            if desired_speed < start_speed:
                vf = self.calc_final_speed(vi, -self._a_max, dist)
                vf = max(desired_speed, vf)
            else:
                vf = self.calc_final_speed(vi, self._a_max, dist)
                vf = min(desired_speed, vf)

            path_point = spiral[i]
            v = vi
            relative_time = time
            traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
            trajectory.append(traj_point)

            time_step = np.abs(vf - vi) / self._a_max
            time += time_step
            vi = vf

        for i in range(ramp_end_index, len(spiral) - 1):
            path_point = spiral[i]
            v = desired_speed
            relative_time = time
            traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
            trajectory.append(traj_point)

            dist = path_point_distance(spiral[i], spiral[i + 1])

            if desired_speed < dbl_epsilon:
                time_step = 0
            else:
                time_step = dist / desired_speed

            time += time_step

        i = len(spiral) - 1
        path_point = spiral[i]
        v = desired_speed
        relative_time = time
        traj_point = TrajectoryPoint(path_point, v, 0, relative_time)
        trajectory.append(traj_point)

        return trajectory

    
    def calc_distance(self, v_i: float, v_f: float, a: float):
        """
        Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
        required for a given acceleration/deceleration.
        """
        if abs(a) < dbl_epsilon:
            return np.inf
        else:
            d = (v_f ** 2 - v_i ** 2) / (2 * a)
            return d
    
    def calc_final_speed(self, v_i : float, a : float, d : float):
        """
        Using v_f = sqrt(v_i ^ 2 + 2ad), compute the final speed for a given
        acceleration across a given distance, with initial speed v_i.
        Make sure to check the discriminant of the radical. If it is negative,
        return zero as the final speed.
        """

        v_f_squared = v_i ** 2 + 2 * a * d

        if v_f_squared <= 0:
            return 0
        elif np.isinf(v_f_squared) or np.isnan(v_f_squared):
            return np.inf
        else:
            return np.sqrt(v_f_squared)