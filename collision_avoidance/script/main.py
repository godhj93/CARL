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
def main():
    env = AirSimDroneEnv(ip_address='127.0.0.1') 
    controller = GeometricCtrl()
    
    rospy.init_node("test")
    rospy.wait_for_service("/test_primitive/trajectory_generation_service")
    mpl_ros_caller = rospy.ServiceProxy("/test_primitive/trajectory_generation_service", Traj_call)


    st = (21.0, 18.0,5.0)
    # g = (-22.0, 19.0,5.0)
    g = (0., 0., 5.0)

    trajectory_info, trajectory_done = GetTrajectory(mpl_ros_caller, st, g)
    if not trajectory_done:
        print("Trajectory is not generated. Try again...")

    st = convert2NED(st)
    g = convert2NED(st)
    cxs, cys, czs, ts = get_polynomial_coefficient(trajectory_info)        
    
    env._setup_flight(init_pos=st)
    done = False
    stop_flag = False
    
    pos_list = []
    vel_list = []
    acc_list = []
    global_init_time = time.time()
    while True:
    
        # step1. trajectory gen
        args = None
        current_time = time.time()

        position, velocity, acceleration, stop_flag = trajectory_generator(cxs, cys, czs, ts, current_time, global_init_time, args)
        
        
        targetPos = np.array([position[1], position[0], -position[2]])
        targetVel = np.array([velocity[1], velocity[0], -velocity[2]])
        targetAcc = np.array([acceleration[1], acceleration[0], -acceleration[2]])

        pos_list.append(targetPos)
        vel_list.append(targetVel)
        acc_list.append(targetAcc)

        mavState = env.GetMavStatesInput()  
        # step2. geometric controller
        # stpe2-1. input: current_drone_states, position, velocity, acceleration
        try:
            p,q,r,t = controller.calc_cmd(mavState, targetPos, targetVel, targetAcc)  
        except Exception as e:
            print(e)
            break
        # step3. airsim action API input -> p,q,r,thrust from geometric controller
        env.PosePublisher(targetPos, mavState["att"], mavState["pos"], mavState["att"])
        env.PublishMarker()
        env.ArcControl(p,q,r,t,duration=1)
        done = env.state['collision']
    
    with open('pos_list.pickle', 'wb') as f:
        pickle.dump(pos_list, f)
    with open('vel_list.pickle', 'wb') as f:
        pickle.dump(vel_list, f)
    with open('acc_list.pickle', 'wb') as f:
        pickle.dump(acc_list, f)
    

  
if __name__ == '__main__':

    main()