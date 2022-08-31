import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped

def GetTrajectory(srv_caller, st, g):
    twist = Twist()
    twist.linear.x = st[0]
    twist.linear.y = st[1]
    twist.linear.z = st[2]
    twist.angular.x = g[0]
    twist.angular.y = g[1]
    twist.angular.z = g[2]
    
    # publish service to mpl_ros to make trajectory
    result_traj = srv_caller(twist).result_traj
    
    if len(result_traj.primitives) == 0:
        trajectory_info = 0
        done = False
    else:
        print("Trajectory is generated")
        trajectory_info = result_traj
        done = True
    return trajectory_info, done

def get_polynomial_coefficient(trajectory):
    cxs = []
    cys = []
    czs = []
    dts = []
    
    for primitive in trajectory.primitives:
        cxs.append(primitive.cx)
        cys.append(primitive.cy)
        czs.append(primitive.cz)
        dts.append(primitive.t)
    dts = np.asarray(dts) # ex) dts = [0.2 0.2 0.2 ... 0.2]
    ts = np.cumsum(dts) # ex) ts = [0.0 0.2 0.4 ... 5.8]
    return cxs, cys, czs, ts

def trajectory_generator(cxs, cys, czs, ts, curr_t, init_t, args):
    stop_flag = False # flag to stop cmd
    coeff_position = np.array([1/120, 1/24, 1/6, 1/2, 1, 1])
    t0 = init_t
    t_current = curr_t - t0    
    if t_current < 0.001:
        t_current = 0

    if t_current > np.max(ts):
        stop_flag = True

    if not stop_flag:
        t_ind = np.where(t_current <= ts)[0][0]
        total_time_with_zero = np.insert(ts, 0, np.array([0]))        
        dt = t_current - total_time_with_zero[t_ind]
        # dt = t_current - ts[t_ind]
        # print('t_ind', t_ind)
        # print(cxs)
        # print(cxs.shape)
        cx = cxs[t_ind]
        cy = cys[t_ind]
        cz = czs[t_ind]
        # define polynomials
        px = np.poly1d(coeff_position*cx,False)
        py = np.poly1d(coeff_position*cy,False)
        pz = np.poly1d(coeff_position*cz,False)
        vx = np.polyder(px)
        vy = np.polyder(py)
        vz = np.polyder(pz)
        ax = np.polyder(vx)
        ay = np.polyder(vy)
        az = np.polyder(vz)
        # get position, velocity, acceleration
        x_t = np.polyval(px,dt)
        y_t = np.polyval(py,dt)
        z_t = np.polyval(pz,dt)
        vx_t = np.polyval(vx,dt)
        vy_t = np.polyval(vy,dt)
        vz_t = np.polyval(vz,dt)
        ax_t = np.polyval(ax,dt)
        ay_t = np.polyval(ay,dt)
        az_t = np.polyval(az,dt)        

        position = np.array([x_t, y_t, z_t])
        velocity = np.array([vx_t, vy_t, vz_t])
        acceleration = np.array([ax_t, ay_t, az_t])
    else:
        position = np.zeros([3,1])
        velocity = np.zeros([3,1])
        acceleration = np.zeros([3,1])        

    return position, velocity, acceleration, stop_flag

def local_trajectory_generator(cxs, cys, czs, ts, curr_t, init_t, args):    
    stop_flag = False # flag to stop cmd
    coeff_position = np.array([1/120, 1/24, 1/6, 1/2, 1, 1])
    t0 = init_t
    t_current = curr_t - t0        

    if t_current > np.max(ts):
        stop_flag = True

    if not stop_flag:
        t_ind = np.where(t_current <= ts)[0][0]
        # total_time_with_zero = np.insert(ts, 0, np.array([0]))        
        # dt = t_current - total_time_with_zero[t_ind]
        dt = t_current - ts[t_ind]
        # print('t_ind', t_ind)
        # print(cxs)
        # print(cxs.shape)
        cx = cxs[t_ind]
        cy = cys[t_ind]
        cz = czs[t_ind]
        # define polynomials
        px = np.poly1d(coeff_position*cx,False)
        py = np.poly1d(coeff_position*cy,False)
        pz = np.poly1d(coeff_position*cz,False)
        vx = np.polyder(px)
        vy = np.polyder(py)
        vz = np.polyder(pz)
        ax = np.polyder(vx)
        ay = np.polyder(vy)
        az = np.polyder(vz)
        # get position, velocity, acceleration
        x_t = np.polyval(px,dt)
        y_t = np.polyval(py,dt)
        z_t = np.polyval(pz,dt)
        vx_t = np.polyval(vx,dt)
        vy_t = np.polyval(vy,dt)
        vz_t = np.polyval(vz,dt)
        ax_t = np.polyval(ax,dt)
        ay_t = np.polyval(ay,dt)
        az_t = np.polyval(az,dt)        

        position = np.array([x_t, y_t, z_t])
        velocity = np.array([vx_t, vy_t, vz_t])
        acceleration = np.array([ax_t, ay_t, az_t])
    else:
        position = np.zeros([3,1])
        velocity = np.zeros([3,1])
        acceleration = np.zeros([3,1])        

    return position, velocity, acceleration, stop_flag


def convert2NED(value):
    # convert all values from rviz to NED coordinates
    value_x = value[1]
    value_y = value[0]
    value_z = -value[2]
    return np.array([value_x, value_y, value_z])

def get_position_error(current_position, target_position):
    em = current_position - target_position
    error = np.sqrt(pow(em[0],2) + pow(em[1], 2) + pow(em[2], 2))
    return error