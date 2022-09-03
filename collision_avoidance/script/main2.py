import numpy as np
import airsim
import rospy
import time
from geometry_msgs.msg import Twist, PoseStamped
from traj_service.srv import Traj_call
from nav_msgs.msg import Path
from waypoint_generator import Waypoints
from utils import *
from airgym.envs.drone_env import AirSimDroneEnv
from geometric_controller import GeometricCtrl
import pickle
from tqdm import tqdm
def main():

    with open("pos_list.pickle", 'rb') as f:
        pos_list = pickle.load(f)

    env = AirSimDroneEnv(ip_address='127.0.0.1') 
    controller = GeometricCtrl()
    
    global_path = []
    for i in tqdm(range(len(pos_list[:]))):
        global_path.append(np.array([-pos_list[i][1], pos_list[i][0]]))
    print("global path generated.")

    st = (21.0, 18.0,-5.0)
    

    env._setup_flight(init_pos=st)
    
    done = False
    i=0
    while not done:

        i += 1
        t = time.time()

        pos_x = global_path[i][1]
        pos_y = global_path[i][0]
        print(pos_x, pos_y)
        env.drone.moveToPositionAsync(
            x = pos_x,
            y = pos_y,
            z = -5.0,
            velocity = 3.0,
            drivetrain= airsim.DrivetrainType.ForwardOnly, 
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=np.arctan2(pos_x, pos_y)), ).join()
            # timeout_sec=1,
            # drivetrain= airsim.DrivetrainType.ForwardOnly, 
            # yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=np.arctan2(pos_x, pos_y)), 
            # lookahead=-1, 
            # adaptive_lookahead=1, 
            # vehicle_name='').join()
        
        print(time.time()-t)
        # done = env.drone.simGetCollisionInfo().has_collided
        # print(done)
    
    

  
if __name__ == '__main__':

    main()