import numpy as np
import rospy
import time
from geometry_msgs.msg import Twist, PoseStamped
from traj_service.srv import Traj_call
from nav_msgs.msg import Path
from waypoint_generator import Waypoints
from utils import *

rospy.init_node("test")
rospy.wait_for_service("/test_primitive/trajectory_generation_service")
mpl_ros_caller = rospy.ServiceProxy("/test_primitive/trajectory_generation_service", Traj_call)

st = (21,18,5)
g = (-22,19,5)

trajectory_info, trajectory_done = GetTrajectory(mpl_ros_caller, st, g)
if not trajectory_done:
    print("Trajectory is not generated. Try again...")

st = convert2NED(st)
g = convert2NED(st)
cxs, cys, czs, ts = get_polynomial_coefficient(trajectory_info)        
global_init_time = time.time()


# step1. trajectory gen
args = None
current_time = time.time()
position, velocity, acceleration, stop_flag = trajectory_generator(cxs, cys, czs, ts, current_time, global_init_time, args)
# step2. geometric controller 
# stpe2-1. input: current_drone_states, position, velocity, acceleration
# step3. airsim action API input -> p,q,r,thrust from geometric controller

print(position)
print(velocity)
print(acceleration)
print(stop_flag)


