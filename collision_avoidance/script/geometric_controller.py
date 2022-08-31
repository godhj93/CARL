#!/usr/bin/env python
import numpy as np

# common math functions #
def quaternion_multiply(q):
    a2 = q[0] # w
    b2 = q[1] # x
    c2 = -q[2] # y
    d2 = -q[3] # z
    a1 = 0
    b1 = 1/np.sqrt(2)
    c1 = 1/np.sqrt(2)
    d1 = 0
    # q1 * q_ned
    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2
    q_new = np.array([a,b,c,d])
    return q_new

def quat2RotMatrix(q):
    rotmat = np.array([\
        [q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3], 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[0] * q[2] + 2 * q[1] * q[3]],\
        [2 * q[0] * q[3] + 2 * q[1] * q[2], q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3], 2 * q[2] * q[3] - 2 * q[0] * q[1]],\
        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[0] * q[1] + 2 * q[2] * q[3], q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]]])
    return rotmat
def rot2Quaternion(R):
    quat = np.zeros(4)
    tr = R.trace()
    if (tr > 0.0):
        S = np.sqrt(tr + 1.0) * 2.0 # S = 4*qw
        quat[0] = 0.25 * S
        quat[1] = (R[2, 1] - R[1, 2]) / S
        quat[2] = (R[0, 2] - R[2, 0]) / S
        quat[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0 # S=4*qx
        quat[0] = (R[2, 1] - R[1, 2]) / S
        quat[1] = 0.25 * S
        quat[2] = (R[0, 1] + R[1, 0]) / S
        quat[3] = (R[0, 2] + R[2, 0]) / S
    elif (R[1, 1] > R[2, 2]):
        S = np.sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0 # S=4*qy
        quat[0] = (R[0, 2] - R[2, 0]) / S
        quat[1] = (R[0, 1] + R[1, 0]) / S
        quat[2] = 0.25 * S
        quat[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0 # S=4*qz
        quat[0] = (R[1, 0] - R[0, 1]) / S
        quat[1] = (R[0, 2] + R[2, 0]) / S
        quat[2] = (R[1, 2] + R[2, 1]) / S
        quat[3] = 0.25 * S
    return quat
def acc2quaternion(acc, yaw):
    proj_xb_des = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    zb_des = acc / np.linalg.norm(acc)
    yb_des = np.cross(zb_des, proj_xb_des) / np.linalg.norm(np.cross(zb_des, proj_xb_des))
    xb_des = np.cross(yb_des, zb_des) / np.linalg.norm(np.cross(yb_des, zb_des))
    rotmat = np.array([\
        [xb_des[0], yb_des[0], zb_des[0]],\
        [xb_des[1], yb_des[1], zb_des[1]],\
        [xb_des[2], yb_des[2], zb_des[2]]])
    quat = rot2Quaternion(rotmat)
    return quat
def matrix_hat(v):
    Vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return Vx
def matrix_hat_inv(Vx):
    v = -np.array([Vx[2,1], Vx[0,2], Vx[1,0]])
    return v
def getVelocityYaw(velocity): # input : 3d vector
    velYaw = np.arctan2(velocity[1], velocity[0])
    return velYaw # generate pseudo heading command to point traj!

class GeometricCtrl:
    def __init__(self):
        # ctrl_mode_ = ERROR_QUATERNION
        self.velocity_yaw_ = False # use yaw?
        self.max_fb_acc_ = 10.0 #max_acc
        self.mavYaw_ = 0.0 # yaw_heading
        # geometric controller parameter for iris model
        self.attctrl_tau_ = 0.36 # attctrl_constant : large -> small rate cmd!        
        self.norm_thrust_const_ = 0.065 # = 9.8/hovering thrust
        self.norm_thrust_offset_ = 0.031 # minimum thrust
        self.Kpos_x_ = 8.0 
        self.Kpos_y_ = 8.0 
        self.Kpos_z_ = 20.0 
        self.Kvel_x_ = 5.0 
        self.Kvel_y_ = 5.0 
        self.Kvel_z_ = 10.0
        self.targetPos_ = np.array([1.0, 1.0, 1.0])
        self.targetVel_ = np.array([0.0, 0.0, 0.0])
        self.targetAcc_ = np.array([0.0, 0.0, 0.0])
        #
        self.targetPos_prev_ = np.array([0.0, 0.0, 0.0])
        self.targetVel_prev_ = np.array([0.0, 0.0, 0.0])
        #
        self.mavPos_ = np.array([0.0, 0.0, 0.0])
        self.mavVel_ = np.array([0.0, 0.0, 0.0])
        #self.mavYaw_ = 0 #
        self.mavAtt_ = np.zeros(4)
        self.g_ = np.array([0.0, 0.0, -9.8])
        self.Kpos_ = np.array([-self.Kpos_x_, -self.Kpos_y_, -self.Kpos_z_])
        self.Kvel_ = np.array([-self.Kvel_x_, -self.Kvel_y_, -self.Kvel_z_])
        self.dragx_ = 0.0
        self.dragy_ = 0.0
        self.dragz_ = 0.0
        self.rotorD_ = np.array([self.dragx_, self.dragy_, self.dragz_]) # rotor drag
        #self.tau = np.array([tau_x, tau_y, tau_z])
    
    def poscontroller(self, pos_error, vel_error):
        a_fb = self.Kpos_* pos_error + self.Kvel_ * vel_error
        if (np.linalg.norm(a_fb) > self.max_fb_acc_):
            a_fb = self.max_fb_acc_/np.linalg.norm(a_fb)*a_fb # Clip acceleration if reference is too large
        return a_fb
    def controlPosition(self, target_pos, target_vel, target_acc):
        # Compute BodyRate commands using differential flatness
        # Controller based on Faessler 2017
        a_ref = target_acc
        if (self.velocity_yaw_):
            self.mavYaw_ = getVelocityYaw(self.mavVel_)
        q_ref = acc2quaternion(a_ref - self.g_, self.mavYaw_)
        R_ref = quat2RotMatrix(q_ref)
        pos_error = self.mavPos_ - target_pos
        vel_error = self.mavVel_ - target_vel
        # Position contorller
        a_fb = self.poscontroller(pos_error, vel_error)
        #print("a_fb", a_fb)
        # Rotor Drag compensation
        #a_rd = R_ref @ np.diag(self.rotorD_) @ R_ref.T @ target_vel # rotor drag
        #print("a_rd", a_rd)
        # Reference acceleration
        #a_des = a_fb + a_ref - a_rd - self.g_
        a_des = a_fb + a_ref - self.g_
        return a_des
    def geometric_attcontroller(self, q_des, a_des):
        # Geometric attitude controller
        # Attitude error is defined as in Lee, Taeyoung, Melvin Leok, and N. Harris McClamroch. "Geometric tracking control
        # of a quadrotor UAV on SE (3)." 49th IEEE conference on decision and control (CDC). IEEE, 2010.
        # The original paper inputs moment commands, but for offboard control, angular rate commands are sent
        ratecmd = np.zeros(4)
        ref_att = q_des
        ref_acc = a_des
        curr_att = self.mavAtt_ # hard coded
        rotmat = quat2RotMatrix(curr_att)
        rotmat_d = quat2RotMatrix(ref_att)
        test = rotmat_d.T @ rotmat - rotmat.T @ rotmat_d
        error_att = 0.5 * matrix_hat_inv(rotmat_d.T @ rotmat - rotmat.T @ rotmat_d)
        ratecmd[0:3] = (2.0 / self.attctrl_tau_) * error_att
        # rotmat = quat2RotMatrix(mavAtt_); --> redefinition? removed.
        zb = rotmat[:,2] # ndarray
        ratecmd[3] = np.max([0.0, np.min([1.0, self.norm_thrust_const_ * ref_acc.dot(zb) + self.norm_thrust_offset_])]) # thrust! check!
        return ratecmd
    
    def computeBodyRateCmd(self, a_des):
        q_des = acc2quaternion(a_des, self.mavYaw_)
        bodyrate_cmd = self.geometric_attcontroller(q_des, a_des) # Calculate BodyRate
        return bodyrate_cmd, q_des
    
    def calc_cmd(self, mavState, targetPos, targetVel, targetAcc): # use it!
        self.mavPos_ = np.array([mavState["pos"][0], -mavState["pos"][1], -mavState["pos"][2]])
        self.mavVel_ = np.array([mavState["vel"][0], -mavState["vel"][1], -mavState["vel"][2]])
        
        self.mavAtt_[0] = mavState["att"][0]
        self.mavAtt_[1] = mavState["att"][1]
        self.mavAtt_[2] = -mavState["att"][2]
        self.mavAtt_[3] = -mavState["att"][3]
        # self.mavAtt_ = np.array([b2, -a2, d2,-c2])
        # self.mavAtt_ = quaternion_multiply(mavState["att"])
        # feedthrough_enable_ = False ?? future development..
        # transform from airsim target to px4 target
        targetPos[1] = -targetPos[1]
        targetVel[1] = -targetVel[1]
        targetAcc[1] = -targetAcc[1]
        targetPos[2] = -targetPos[2]
        targetVel[2] = -targetVel[2]
        targetAcc[2] = -targetAcc[2]

        desired_acc = self.controlPosition(targetPos, targetVel, targetAcc)
        cmdBodyRate, q_des = self.computeBodyRateCmd(desired_acc)
        p = cmdBodyRate[0]
        q = cmdBodyRate[1]
        r = cmdBodyRate[2]
        thrust = cmdBodyRate[3]
        return p, q, r, thrust
# controller = GeometricCtrl()
# mavState = {"pos":np.array([0,0,0]), "vel":np.array([0,0,0]), "att":np.array([1,0,0,0])}
# targetPos = np.array([0,0,1])
# targetVel = np.array([0,0,0])
# targetAcc = np.array([0,0,0])
# print(controller.calc_cmd(mavState, targetPos, targetVel, targetAcc))

# if __name__ == '__main__':
#     controller = GeometricCtrl()