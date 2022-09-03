import airsim
import rospy
import numpy as np
from airgym.envs.airsim_env import AirSimEnv
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from utils import get_position_error
class AirSimDroneEnv(AirSimEnv):

    def __init__(self, ip_address):
        super().__init__()

        #ROS

        self.target_pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size = 1)
        self.current_pose_pub = rospy.Publisher("/current_pose", PoseStamped, queue_size = 1)
        self.traj_pub = rospy.Publisher("/traj",Path,queue_size=1,latch=True)
        self.marker_array_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/waypoint", Marker, queue_size=100)

        self.state = {
            "prev_position": np.zeros(3),
            "position": np.zeros(3),
            "dynamic_state": np.zeros([4,4]),
            "depth_image": np.zeros([128,160,4]),
            "depth_lidar": np.zeros(1),
            "collision": False
        }
        self.lidar_depth_prev = 0

        self.drone = airsim.MultirotorClient(ip=ip_address)

        self._setup_flight()

        self.MIN_DEPTH_METERS = 0
        self.MAX_DEPTH_METERS = 100.0


    def __del__(self):
        self.drone.reset()

    def _setup_flight(self,init_pos=(0.0,0.0,-5.0)):

        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        pose = self.drone.simGetVehiclePose()
       
        '''Initializing orientations'''
        # position = [init_x, init_y, 20]
 
       
        x,y,z,w = airsim.utils.to_quaternion(0, 0, 180)
        self.drone.moveByVelocityAsync(0, 0, 0.1, 1.0).join()             
        self.drone.simSetVehiclePose(pose=airsim.Pose(airsim.Vector3r(init_pos[0], init_pos[1], init_pos[2]), airsim.Quaternionr(x, y, z, w)), 
                                        ignore_collision=True)
        # while True:
        #     mavState = self.GetMavStatesInput()
        #     self.drone.moveByVelocityAsync(0, 0, 0.1, 1.0).join()             
        #     self.drone.simSetVehiclePose(pose=airsim.Pose(airsim.Vector3r(init_pos[0], init_pos[1], init_pos[2]), airsim.Quaternionr(0, 0, 0, 1)), 
        #                                 ignore_collision=True)
        #     error = get_position_error(mavState["pos"], init_pos)
        #     if error < 0.1:
        #         break 
        # print("Finished resetting drone.")
        # self.drone.simSetVehiclePose(pose, True)
        
        # self.drone.moveToPositionAsync(pose.position.x_val, pose.position.y_val, -5, 10).join()
       
    def GetMavStatesInput(self):
        state = self.GetDroneState()
        position = np.array([state[0], state[1], state[2]])
        velocity = np.array([state[3], state[4], state[5]])
        attitude = np.array([state[12], state[9], state[10], state[11]])
        mav_states = {"pos": position, "vel": velocity, "att": attitude}
        
        return mav_states

    def GetDroneState(self):
        state = self.drone.getMultirotorState()
        self.pos_x = state.kinematics_estimated.position.x_val
        self.pos_y = state.kinematics_estimated.position.y_val
        self.pos_z = state.kinematics_estimated.position.z_val
        self.vx = state.kinematics_estimated.linear_velocity.x_val
        self.vy = state.kinematics_estimated.linear_velocity.y_val
        self.vz = state.kinematics_estimated.linear_velocity.z_val
        self.ax = state.kinematics_estimated.angular_velocity.x_val
        self.ay = state.kinematics_estimated.angular_velocity.y_val
        self.az = state.kinematics_estimated.angular_velocity.z_val
        self.ori_x = state.kinematics_estimated.orientation.x_val
        self.ori_y = state.kinematics_estimated.orientation.y_val
        self.ori_z = state.kinematics_estimated.orientation.z_val
        self.ori_w = state.kinematics_estimated.orientation.w_val
        # linear acceleration이 필요한가?
        return np.array([self.pos_x,
                         self.pos_y,
                         self.pos_z,
                         self.vx,
                         self.vy,
                         self.vz,
                         self.ax,
                         self.ay,
                         self.az,
                         self.ori_x, # 9
                         self.ori_y, # 10
                         self.ori_z, # 11
                         self.ori_w]) # 12

    def ArcControl(self, roll_rate, pitch_rate, yaw_rate, throttle, duration):
        self.drone.moveByAngleRatesThrottleAsync(roll_rate=roll_rate, pitch_rate=pitch_rate, yaw_rate=yaw_rate, throttle=throttle, duration=duration)
    
    def PositionControl(self, st):
        pose = self.drone.simGetVehiclePose()
        pose.position.x_val = st[0]
        pose.position.y_val = st[1]
        pose.position.z_val = st[2]
        # self.client.moveToPositionAsync(x=py, y=-px, z=pz, velocity=speed)
        self.drone.simSetVehiclePose(pose, True)        

    def AngleControl(self, pitch, roll, throttle, yaw, duration):        
        self.drone.moveByAngleThrottleAsync(pitch, roll, throttle, yaw, duration)

    def PosePublisher(self, target_pos, target_att, current_pos, current_att):
        target_pos_msg = PoseStamped()
        current_pos_msg = PoseStamped()
        target_pos_msg.header.frame_id='map'
        target_pos_msg.header.stamp= rospy.Time.now()
        target_pos_msg.pose.position.x = -target_pos[1]
        target_pos_msg.pose.position.y = target_pos[0]
        target_pos_msg.pose.position.z = target_pos[2]
        target_pos_msg.pose.orientation.w = target_att[0]
        target_pos_msg.pose.orientation.x = target_att[1]
        target_pos_msg.pose.orientation.y = target_att[2]
        target_pos_msg.pose.orientation.z = target_att[3]

        current_pos_msg.header.frame_id='map'
        current_pos_msg.header.stamp= rospy.Time.now()
        current_pos_msg.pose.position.x = current_pos[1]
        current_pos_msg.pose.position.y = current_pos[0]
        current_pos_msg.pose.position.z = -current_pos[2]
        current_pos_msg.pose.orientation.w = current_att[0]
        current_pos_msg.pose.orientation.x = current_att[1]
        current_pos_msg.pose.orientation.y = current_att[2]
        current_pos_msg.pose.orientation.z = current_att[3]
        self.target_pose_pub.publish(target_pos_msg)
        self.current_pose_pub.publish(current_pos_msg) 

    def PublishTrajectory(self, traj): # traj: waypoint class
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        times = np.linspace(0, traj.total_time[-1], 100)
        points = []
        for time in times:
            point, _, _ = traj.get_point_position(time)
            points.append(point)
        for i, point in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = point[1]
            pose.pose.position.y = point[0]
            pose.pose.position.z = -point[2]
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1
            msg.poses.append(pose)
        self.traj_pub.publish(msg)
    
    def PublishMarkerArray(self, waypoint):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(waypoint):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint[1]
            marker.pose.position.y = waypoint[0]
            marker.pose.position.z = -waypoint[2]
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)
        self.marker_array_pub.publish(marker_array)
    
    def PublishMarker(self):
        # get position
        state = self.drone.getMultirotorState()
        self.pos_x = state.kinematics_estimated.position.x_val
        self.pos_y = state.kinematics_estimated.position.y_val
        self.pos_z = state.kinematics_estimated.position.z_val

        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 123
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.pos_y
        marker.pose.position.y = self.pos_x
        marker.pose.position.z = -self.pos_z
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker) 

    def _get_obs(self):
        
        # response, = self.drone.simGetImages([self.image_request])
        
        # self.depth_image = self._query_depth_img_uint8()
        self.depth_image = self._query_depth_img()
        self.depth_lidar = self._query_lidar()
        
        import cv2
        cv2.imshow("DEPTH IMAGE", self.depth_image)
        cv2.waitKey(1)

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        # self.state["dynamic_state"] = np.array([
        #     *self.drone_state.kinematics_estimated.linear_velocity, 
        #     *self.drone_state.kinematics_estimated.linear_acceleration, 
        #     *self.drone_state.kinematics_estimated.angular_velocity,
        #     *self.drone_state.kinematics_estimated.angular_acceleration])
        
        self.state["dynamic_state"] = np.vstack( (self.state["dynamic_state"],
        
            np.array([self.drone_state.kinematics_estimated.linear_velocity.x_val, 
            self.drone_state.kinematics_estimated.linear_acceleration.x_val, 
            self.drone_state.kinematics_estimated.angular_velocity.z_val,
            self.drone_state.kinematics_estimated.angular_acceleration.z_val])))[1:,:]
        
        # self.state["depth_image"] = self.depth_image
        self.state["depth_image"] = np.dstack((self.state["depth_image"], self.depth_image))[:,:,1:]
        self.state["depth_lidar"] = self.depth_lidar

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return [self.state["depth_image"], self.state["dynamic_state"]]

    def _query_lidar(self):

        lidarData = self.drone.getLidarData()
        if (len(lidarData.point_cloud) < 1):
            print("\tNo points received from Lidar data.")
            return self.lidar_depth_prev
        
        else:
            points = np.array(lidarData.point_cloud,dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3),3))
            
            pt_range = []
            for pt in points:
                
                distance_from_vehicle = np.sqrt(pt[0]**2 + pt[1]**2)
                pt_range.append(distance_from_vehicle)
            
                self.lidar_depth = np.min(pt_range)
            self.lidar_depth_prev = self.lidar_depth
            return self.lidar_depth
    
    def _query_depth_img_uint8(self):

        responses = self.drone.simGetImages(
            [
                airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, False, False
            )])

        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # original image is fliped vertically
        img_rgb = np.flipud(img_rgb)
        return np.expand_dims(img_rgb[:,:,0]/255.0,axis=-1)

    def _query_depth_img(self):

        response = self.drone.simGetImages(
            [
                airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False
            )])[0]
        
        depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)

        limit_max_range_to_10m = np.interp(depth_img_in_meters, (self.MIN_DEPTH_METERS, self.MAX_DEPTH_METERS), (0,10))
        # Lerp 0..100m to 0..255 gray values
        depth_8bit_lerped = np.interp(limit_max_range_to_10m, (0, 10), (0, 255))
        depth_normalized = depth_8bit_lerped / 255.0

        return depth_normalized

    def _do_action(self, action):
        
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = action[0],
            vy = 0.0,
            z = -5.0,
            duration = 10,
            drivetrain = airsim.DrivetrainType.ForwardOnly,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=action[1])
        )

    def _compute_reward(self):
        '''
        vel_x: Action is m/s, State is m/s
        vel_w: Action is degree/sec, State is rad/sec!!!
        '''
        MAX_DEPTH = 40.0
        depth_norm = np.clip(self.state["depth_lidar"], 0, MAX_DEPTH) / MAX_DEPTH
        r_depth = (depth_norm - 0.5) * 2 # [-1, +1]

        vx = self.state["dynamic_state"][-1][0]
        w = self.state["dynamic_state"][-1][2]
        
        if self.state["collision"]:
            done = True
            r_collision = 0.0

        else:
            done=False
            r_collision = 0.0
        
        # r_dyn = np.clip(1/np.abs(depth_norm*3 - vx), 0, 1)
        
        if r_depth >= 0:
            # print("r_depth >= 0")
            r_dyn = np.clip(vx * np.cos(w), 0, 1)
        else:
            # print("r_depth < 0")
            r_dyn = 0#np.abs(w)
        

        reward = r_collision + r_depth + r_dyn
        
        # print(f"Reward: {reward:.4f}, r_depth: {r_depth:.4f}, r_dyn: {r_dyn:.4f}, r_collision: {r_collision}")
        return reward, done
    

    def step(self, action, global_step):

        self._do_action(action)
        
    
        self.obs = self._get_obs()
        self.reward, self.done = self._compute_reward()
    
        return self.obs, self.reward, self.done, self.state

    def reset(self):

        self._setup_flight()

        return self._get_obs()

    