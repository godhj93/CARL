#include <decomp_ros_utils/data_ros_utils.h>
#include <mpl_external_planner/ellipsoid_planner/ellipsoid_planner.h>
#include <planning_ros_utils/data_ros_utils.h>
#include <planning_ros_utils/primitive_ros_utils.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
//#include <sensor_msgs/point_field_conversion.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <traj_service/Traj_call.h>
#include <geometry_msgs/Twist.h>
#include "bag_reader.hpp"



class ellipsoidalPlanner{
public:
  bool test(traj_service::Traj_call::Request &req, traj_service::Traj_call::Response &res);
  ellipsoidalPlanner();

private:
  ros::NodeHandle nh_;
  ros::Publisher es_pub, sg_pub, traj_pub, prior_traj_pub, cloud_pub, ps_pub;
  ros::ServiceServer service;
  
  ros::Time t0;  
  std::string file_name, topic_name;
  sensor_msgs::PointCloud2 tmp;
  sensor_msgs::PointCloud map;
  double robot_radius;
  Vec3f origin, dim;
  double dt, v_max, a_max, w, epsilon;
  double u_max_z, u_max;
  int max_num, num;
  bool use_3d;
  std::unique_ptr<MPL::EllipsoidPlanner> planner_;
  double start_x, start_y, start_z;
  double start_vx, start_vy, start_vz;
  double goal_x, goal_y, goal_z;
  bool use_acc, use_jrk;
  Waypoint3D start;
  Waypoint3D goal;
  
  sensor_msgs::PointCloud sg_cloud;
  geometry_msgs::Point32 pt1, pt2;
  std::string traj_file_name, traj_topic_name;
  bool use_prior;
  planning_ros_msgs::Trajectory prior_traj;
  vec_E<VecDf> U;
  decimal_t du;

  
  bool valid;
  sensor_msgs::PointCloud ps;
  
};
// constructor
ellipsoidalPlanner::ellipsoidalPlanner():
nh_("~")
{
  
  service = nh_.advertiseService("trajectory_generation_service", &ellipsoidalPlanner::test, this);
  es_pub = nh_.advertise<decomp_ros_msgs::EllipsoidArray>("ellipsoids", 1, true);
  sg_pub = nh_.advertise<sensor_msgs::PointCloud>("start_and_goal", 1, true);
  traj_pub = nh_.advertise<planning_ros_msgs::Trajectory>("trajectory", 1, true);
  prior_traj_pub = nh_.advertise<planning_ros_msgs::Trajectory>("prior_trajectory", 1, true);
  cloud_pub = nh_.advertise<sensor_msgs::PointCloud>("cloud", 1, true);
  ps_pub = nh_.advertise<sensor_msgs::PointCloud>("ps", 1, true);
  t0 = ros::Time::now();
  // Read map from bag file
  nh_.param("file", file_name, std::string("voxel_map"));
  nh_.param("topic", topic_name, std::string("voxel_map"));   
  ROS_INFO("load pcd success");    
  tmp = read_bag<sensor_msgs::PointCloud2>(file_name, topic_name, 0).back();
  ROS_INFO("read pcl2 success"); 
  map.header.frame_id = "world";
  map.header.stamp = ros::Time::now();
  bool success = sensor_msgs::convertPointCloud2ToPointCloud(tmp, map);
  if (!success) {
    printf("PointCloud loading failed");
  }
  
  ROS_INFO("convert pcl1 success");
  cloud_pub.publish(map);
  
  nh_.param("robot_r", robot_radius, 0.5);
  nh_.param("origin_x", origin(0), 0.0);
  nh_.param("origin_y", origin(1), 0.0);
  nh_.param("origin_z", origin(2), 0.0);
  nh_.param("range_x", dim(0), 0.0);
  nh_.param("range_y", dim(1), 0.0);
  nh_.param("range_z", dim(2), 0.0);

  ROS_INFO("Takse %f sec to set up map!", (ros::Time::now() - t0).toSec());
  t0 = ros::Time::now();

  // Initialize planner
  nh_.param("dt", dt, 1.0);
  nh_.param("epsilon", epsilon, 1.0);
  nh_.param("v_max", v_max, -1.0);
  nh_.param("a_max", a_max, -1.0);
  nh_.param("u_max", u_max, 1.0);
  nh_.param("u_max_z", u_max_z, 1.0);
  nh_.param("w", w, 10.);
  nh_.param("num", num, 1);
  nh_.param("max_num", max_num, -1);
  nh_.param("use_3d", use_3d, false);

  //
  planner_.reset(new MPL::EllipsoidPlanner(true));
  planner_->setMap(cloud_to_vec(map), robot_radius, origin,
                   dim);          // Set collision checking function
  planner_->setEpsilon(epsilon);  // Set greedy param (default equal to 1)
  planner_->setVmax(v_max);       // Set max velocity
  planner_->setAmax(a_max);       // Set max acceleration
  planner_->setDt(dt);            // Set dt for each primitive
  planner_->setW(w);              // Set time weight for each primitive
  planner_->setMaxNum(
      max_num);  // Set maximum allowed expansion, -1 means no limitation
  planner_->setTol(6.0, 100.0,
                   100.0);  // Tolerance for goal region as pos, vel, acc
  

  // Set start and goal
  nh_.param("start_x", start_x, 12.5);
  nh_.param("start_y", start_y, 1.4);
  nh_.param("start_z", start_z, 0.0);
  
  nh_.param("start_vx", start_vx, 0.0);
  nh_.param("start_vy", start_vy, 0.0);
  nh_.param("start_vz", start_vz, 0.0);
 
  nh_.param("goal_x", goal_x, 6.4);
  nh_.param("goal_y", goal_y, 16.6);
  nh_.param("goal_z", goal_z, 0.0);

  nh_.param("use_acc", use_acc, true);
  nh_.param("use_jrk", use_jrk, true);

  
  start.pos = Vec3f(start_x, start_y, start_z);
  start.vel = Vec3f(start_vx, start_vy, start_vz);
  start.acc = Vec3f(0, 0, 0);
  start.jrk = Vec3f(0, 0, 0);
  start.use_pos = true;
  start.use_vel = true;
  start.use_acc = use_acc;
  start.use_jrk = use_jrk;
  start.use_yaw = false;

  goal = Waypoint3D(start.control);
  goal.pos = Vec3f(goal_x, goal_y, goal_z);
  goal.vel = Vec3f(0, 0, 0);
  goal.acc = Vec3f(0, 0, 0);
  goal.jrk = Vec3f(0, 0, 0);
  
  // Publish location of start and goal
  sg_cloud.header.frame_id = "map";
  
  pt1.x = start_x, pt1.y = start_y, pt1.z = start_z;
  pt2.x = goal_x, pt2.y = goal_y, pt2.z = goal_z;
  
  sg_cloud.points.push_back(pt1), sg_cloud.points.push_back(pt2);
  sg_pub.publish(sg_cloud);

  // Read prior traj
  
  nh_.param("traj_file", traj_file_name, std::string(""));
  nh_.param("traj_topic", traj_topic_name, std::string(""));
  nh_.param("use_prior", use_prior, false);
  
  if (!traj_file_name.empty()) {
    prior_traj =
        read_bag<planning_ros_msgs::Trajectory>(traj_file_name, traj_topic_name,
                                                0)
            .back();
    if (!prior_traj.primitives.empty()) {
      prior_traj_pub.publish(prior_traj);
      if (use_prior) {
        planner_->setPriorTrajectory(toTrajectory3D(prior_traj));
        goal.use_acc = false;
        goal.use_jrk = false;
      }
    }
  }

  // Set input control

  decimal_t du = u_max / num;
  if (use_3d) {
    decimal_t du_z = u_max_z / num;
    for (decimal_t dx = -u_max; dx <= u_max; dx += du)
      for (decimal_t dy = -u_max; dy <= u_max; dy += du)
        for (decimal_t dz = -u_max_z; dz <= u_max_z;
             dz += du_z)  // here we reduce the z control
          U.push_back(Vec3f(dx, dy, dz));
  } else {
    for (decimal_t dx = -u_max; dx <= u_max; dx += du)
      for (decimal_t dy = -u_max; dy <= u_max; dy += du)
        U.push_back(Vec3f(dx, dy, 0));
  }
  planner_->setU(U);  // Set discretization with 1 and efforts
  // planner_->setMode(num, use_3d, start); // Set discretization with 1 and
  // efforts
  // Planning thread!

  t0 = ros::Time::now();
  bool valid = planner_->plan(start, goal);
  
  // Publish expanded nodes
  ps = vec_to_cloud(planner_->getCloseSet());
  ps.header.frame_id = "map";
  ps_pub.publish(ps);

  if (!valid) {
    ROS_WARN("Failed! Takes %f sec for planning, expand [%zu] nodes",
             (ros::Time::now() - t0).toSec(), planner_->getCloseSet().size());
  } else {
    ROS_INFO("Succeed! Takes %f sec for planning, expand [%zu] nodes",
             (ros::Time::now() - t0).toSec(), planner_->getCloseSet().size());

    // Publish trajectory
    auto traj = planner_->getTraj();
    planning_ros_msgs::Trajectory traj_msg = toTrajectoryROSMsg(traj);
    traj_msg.header.frame_id = "map";
    traj_pub.publish(traj_msg);

    // Write result.txt
    ros::Time result_time = ros::Time::now();
    std::stringstream ss;
    ss << result_time.sec <<"_" << result_time.nsec;
    
    std::ofstream myfile;
    std::string pwd = "/root/catkin_ws/src/mpl_ros/mpl_test_node/launch/ellipsoid_planner_node/rosbag_file/";
    std::string txt = ".txt";
    std::string full_name = pwd + ss.str() + txt;
    std::cout << full_name << std::endl;
    myfile.open (full_name);
    if(myfile.is_open())
    {
        std::cout<<"Success"<<std::endl;
        myfile << "traj.J(Control::VEL), traj.J(Control::ACC), traj.J(Control::SNP), traj.getTotalTime())" << std::endl;
        myfile << traj.J(Control::VEL) << ", " << traj.J(Control::ACC) << ", " << traj.J(Control::SNP) << ", "  << traj.getTotalTime();
    }
    myfile.close();
    std::cout << "save success" << std::endl;
    //

    printf(
        "================== Traj -- total J(VEL): %f, J(ACC): %F, J(JRK): %f, "
        "total time: %f\n",
        traj.J(Control::VEL), traj.J(Control::ACC), traj.J(Control::SNP),
        traj.getTotalTime());

    vec_E<Ellipsoid3D> Es =
        sample_ellipsoids(traj, Vec3f(robot_radius, robot_radius, 0.1), 50);
    decomp_ros_msgs::EllipsoidArray es_msg =
        DecompROS::ellipsoid_array_to_ros(Es);
    es_msg.header.frame_id = "map";
    es_pub.publish(es_msg);

    max_attitude(traj, 1000);
  }
  
}

bool ellipsoidalPlanner::test(traj_service::Traj_call::Request &req, traj_service::Traj_call::Response &res){
  res.result_traj = planning_ros_msgs::Trajectory(); // remove?

  start_x = req.start_and_goal.linear.x;
  start_y = req.start_and_goal.linear.y;
  start_z = req.start_and_goal.linear.z;
  goal_x = req.start_and_goal.angular.x;
  goal_y = req.start_and_goal.angular.y;
  goal_z = req.start_and_goal.angular.z;
  start.pos = Vec3f(start_x, start_y, start_z);
  goal.pos = Vec3f(goal_x, goal_y, goal_z);

  
  t0 = ros::Time::now();
  bool valid = planner_->plan(start, goal);
  
  // Publish expanded nodes
  ps = vec_to_cloud(planner_->getCloseSet());
  ps.header.frame_id = "map";
  ps_pub.publish(ps);

  if (!valid) {
    ROS_WARN("Failed! Takes %f sec for planning, expand [%zu] nodes",
             (ros::Time::now() - t0).toSec(), planner_->getCloseSet().size());
  } else {
    ROS_INFO("Succeed! Takes %f sec for planning, expand [%zu] nodes",
             (ros::Time::now() - t0).toSec(), planner_->getCloseSet().size());

    // Publish trajectory
    auto traj = planner_->getTraj();
    planning_ros_msgs::Trajectory traj_msg = toTrajectoryROSMsg(traj);
    res.result_traj = toTrajectoryROSMsg(traj); // service result!!!!!!
    traj_msg.header.frame_id = "map";
    traj_pub.publish(traj_msg);

    // Write result.txt
    //ros::Time result_time = ros::Time::now();
    //std::stringstream ss;
    //ss << result_time.sec <<"_" << result_time.nsec;
    
    //std::ofstream myfile;
    //std::string pwd = "/root/catkin_ws/src/mpl_ros/mpl_test_node/launch/ellipsoid_planner_node/rosbag_file/";
    //std::string txt = ".txt";
    //std::string full_name = pwd + ss.str() + txt;
    //std::cout << full_name << std::endl;
    //myfile.open (full_name);
    //if(myfile.is_open())
    //{
    //    std::cout<<"Success"<<std::endl;
    //    myfile << "traj.J(Control::VEL), traj.J(Control::ACC), traj.J(Control::SNP), traj.getTotalTime())" << std::endl;
    //    myfile << traj.J(Control::VEL) << ", " << traj.J(Control::ACC) << ", " << traj.J(Control::SNP) << ", "  << traj.getTotalTime();
    //}
    //myfile.close();
    //std::cout << "save success" << std::endl;
    //

    printf(
        "================== Traj -- total J(VEL): %f, J(ACC): %F, J(JRK): %f, "
        "total time: %f\n",
        traj.J(Control::VEL), traj.J(Control::ACC), traj.J(Control::SNP),
        traj.getTotalTime());

    vec_E<Ellipsoid3D> Es =
        sample_ellipsoids(traj, Vec3f(robot_radius, robot_radius, 0.1), 50);
    decomp_ros_msgs::EllipsoidArray es_msg =
        DecompROS::ellipsoid_array_to_ros(Es);
    es_msg.header.frame_id = "map";
    es_pub.publish(es_msg);

    max_attitude(traj, 1000);
  }
  
  
  return true;
}



int main(int argc, char **argv) {

  ros::init(argc, argv, "test");

  ellipsoidalPlanner e;

  ros::spin();

  return 0;
}
