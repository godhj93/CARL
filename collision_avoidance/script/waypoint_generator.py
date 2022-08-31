import numpy as np

class Waypoints:
    def __init__(self, waypoints, dt=1):
        self.dt = dt
        self.waypoints = waypoints
        self.num_seg = len(self.waypoints)-1
        self.waypoint_dt = np.ones(self.num_seg)*dt
        self.total_time = np.cumsum(self.waypoint_dt)
        self.A = np.zeros([8*self.num_seg, 8*self.num_seg])
        self.b = np.zeros([8*self.num_seg, 3])
        
        
        self.pos_at_zero = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        self.vel_at_zero = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        self.acc_at_zero = np.array([0, 0, 1, 0, 0, 0, 0, 0])
        self.jerk_at_zero = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        self.snap_at_zero = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        self.crackle_at_zero = np.array([0, 0, 0, 0, 0, 1, 0, 0])
        self.pop_at_zero = np.array([0, 0, 0, 0, 0, 0, 1, 0])
        
        self.pos_at_dt = np.array([1, dt, dt**2/2, dt**3/6, dt**4/24, dt**5/120, dt**6/720, dt**7/5040])
        self.vel_at_dt = np.array([0,  1,      dt, dt**2/2,  dt**3/6,  dt**4/24, dt**5/120, dt**6/720])
        self.acc_at_dt = np.array([0,  0,       1,      dt,  dt**2/2,   dt**3/6,  dt**4/24, dt**5/120])
        self.jerk_at_dt = np.array([0,  0,       0,       1,      dt,   dt**2/2,   dt**3/6, dt**4/24])
        self.snap_at_dt = np.array([0,  0,       0,       0,       1,        dt,   dt**2/2, dt**3/6])
        self.crackle_at_dt = np.array([0,  0,    0,       0,       0,         1,        dt, dt**2/2])
        self.pop_at_dt = np.array([0,  0,        0,        0,      0,         0,         1,      dt])
        
        self.A_jerk = np.zeros([6*self.num_seg, 6*self.num_seg])
        self.b_jerk = np.zeros([6*self.num_seg, 3])
        self.A_acc = np.zeros([4*self.num_seg, 4*self.num_seg])
        self.b_acc = np.zeros([4*self.num_seg, 3])
        
    def calc_coeff(self, initial_condition):
        # initial condition : list type, ex) initial_condtion[0] = np.array([current_velocity_x, current_velocity_y, current_velocity_z])
        self.initial_condition = initial_condition
        cx, cy, cz = self.calc_coeff_min_jerk()
        return cx, cy, cz

    def calc_coeff_min_jerk(self):
        self.coeff_position = np.array([1/120, 1/24, 1/6, 1/2, 1, 1])
        
        for i in range(self.num_seg):
            if i != (self.num_seg-1):
                waypoint_constraint0      = self.pos_at_zero[:-2]
                waypoint_constraint1      = self.pos_at_dt[:-2]
                waypoint_constraint_right = np.zeros(6)

                segment_constraint1_left  = self.vel_at_dt[:-2]
                segment_constraint1_right = -self.vel_at_zero[:-2]
                segment_constraint2_left  = self.acc_at_dt[:-2]
                segment_constraint2_right = -self.acc_at_zero[:-2]
                segment_constraint3_left  = self.jerk_at_dt[:-2]
                segment_constraint3_right = -self.jerk_at_zero[:-2]
                segment_constraint4_left  = self.snap_at_dt[:-2]
                segment_constraint4_right = -self.snap_at_zero[:-2]

                waypoint_constraint0 = np.append(waypoint_constraint0, waypoint_constraint_right)
                waypoint_constraint1 = np.append(waypoint_constraint1, waypoint_constraint_right)
                segment_constraint1 = np.append(segment_constraint1_left, segment_constraint1_right)
                segment_constraint2 = np.append(segment_constraint2_left, segment_constraint2_right)
                segment_constraint3 = np.append(segment_constraint3_left, segment_constraint3_right)
                segment_constraint4 = np.append(segment_constraint4_left, segment_constraint4_right)
                segment_constraint = np.vstack([waypoint_constraint0, waypoint_constraint1, segment_constraint1, segment_constraint2, segment_constraint3, segment_constraint4])
                # print(segment_constraint)
                bx = np.array([self.waypoints[i][0], self.waypoints[i+1][0], 0, 0, 0, 0])
                by = np.array([self.waypoints[i][1], self.waypoints[i+1][1], 0, 0, 0, 0])
                bz = np.array([self.waypoints[i][2], self.waypoints[i+1][2], 0, 0, 0, 0])
                self.A_jerk[6*i:(6*i+6),6*i:(6*i+12)] = segment_constraint
                self.b_jerk[6*i:(6*i+6),:] = np.vstack([bx,by,bz]).T
            else:
                # terminal waypoint
                waypoint_constraint0      = self.pos_at_zero[:-2]
                waypoint_constraint1      = self.pos_at_dt[:-2]
                # initial zero vel, acc
                initial_constraint1_left  = self.vel_at_zero[:-2]
                initial_constraint2_left  = self.acc_at_zero[:-2]
                # terminal zero vel, acc
                terminal_constraint1_left  = self.vel_at_dt[:-2]
                terminal_constraint2_left  = self.acc_at_dt[:-2]

                self.A_jerk[6*i,   6*i:6*i+6] = waypoint_constraint0
                self.A_jerk[6*i+1, 6*i:6*i+6] = waypoint_constraint1
                self.A_jerk[6*i+2, 6*i:6*i+6] = terminal_constraint1_left
                self.A_jerk[6*i+3, 6*i:6*i+6] = terminal_constraint2_left
                self.A_jerk[6*i+4, 0:6] = initial_constraint1_left
                self.A_jerk[6*i+5, 0:6] = initial_constraint2_left
                
                pos_0 = self.initial_condition[0] # wrong
                vel_0 = self.initial_condition[1]
                acc_0 = self.initial_condition[2]

                v0 = 0
                a0 = 0
                # # jz0 = acc_0
                vT = 0
                aT = 0
                jT = 0
                # bx = np.array([self.waypoints[i][0], self.waypoints[i+1][0], vT, aT, vel_0[0], acc_0[0]])
                # by = np.array([self.waypoints[i][1], self.waypoints[i+1][1], vT, aT, vel_0[1], acc_0[1]])
                # bz = np.array([self.waypoints[i][2], self.waypoints[i+1][2], vT, aT, vel_0[2], acc_0[2]])
                bx = np.array([self.waypoints[i][0], self.waypoints[i+1][0], vT, aT, v0, a0])
                by = np.array([self.waypoints[i][1], self.waypoints[i+1][1], vT, aT, v0, a0])
                bz = np.array([self.waypoints[i][2], self.waypoints[i+1][2], vT, aT, v0, a0])
                self.b_jerk[6*i:(6*i+6),:] = np.vstack([bx,by,bz]).T
        coeff = np.linalg.solve(self.A_jerk,self.b_jerk)
        self.coeff_x = np.fliplr(coeff[:,0].reshape(-1,6))
        self.coeff_y = np.fliplr(coeff[:,1].reshape(-1,6))
        self.coeff_z = np.fliplr(coeff[:,2].reshape(-1,6))  
        return self.coeff_x, self.coeff_y, self.coeff_z
             
    def get_point_position(self, t):
        
        t_ind = np.where(t<=self.total_time)[0][0]
        
        total_time_with_zero = np.insert(self.total_time, 0, np.array([0]))
        t_reg = (t - total_time_with_zero[t_ind])
        # print("t_reg{}".format(t_reg))
        cx = self.coeff_position*self.coeff_x[t_ind] 
        px = np.poly1d(cx, False)
        vx = np.polyder(px)
        ax = np.polyder(vx)

        cy = self.coeff_position*self.coeff_y[t_ind]
        py = np.poly1d(cy, False)
        vy = np.polyder(py)
        ay = np.polyder(vy)

        cz = self.coeff_position*self.coeff_z[t_ind]
        pz = np.poly1d(cz, False)
        vz = np.polyder(pz)
        az = np.polyder(vz)
        
        px_t = np.polyval(px, t_reg)
        vx_t = np.polyval(vx, t_reg)
        ax_t = np.polyval(ax, t_reg)
        py_t = np.polyval(py, t_reg)
        vy_t = np.polyval(vy, t_reg)
        ay_t = np.polyval(ay, t_reg)
        pz_t = np.polyval(pz, t_reg)
        vz_t = np.polyval(vz, t_reg)
        az_t = np.polyval(az, t_reg)
        
        position = np.array([px_t, py_t, pz_t])
        velocity = np.array([vx_t, vy_t, vz_t])
        acceleration = np.array([ax_t, ay_t, az_t])
        return position, velocity, acceleration