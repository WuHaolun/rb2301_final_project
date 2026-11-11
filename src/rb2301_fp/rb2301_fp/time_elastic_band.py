from dataclasses import dataclass
import math

import casadi as ca
import numpy as np

# LiDAR translation from base_link in nanocar_description.urdf, in metres.
LIDAR_X_OFFSET = 0.11423


@dataclass(frozen=True)
class LidarFootprint:
    """Keep-out distances from the LiDAR, with axes aligned to the robot."""

    front: float
    rear: float
    left: float
    right: float

    def __post_init__(self):
        distances = (self.front, self.rear, self.left, self.right)
        if (any(not math.isfinite(value) or value < 0 for value in distances)
                or self.front + self.rear == 0 or self.left + self.right == 0):
            raise ValueError('LiDAR keep-out lengths must be finite and non-negative with positive width and length')

    @property
    def base_bounds(self):
        """Return xmin, xmax, ymin, ymax in the robot base frame."""
        return LIDAR_X_OFFSET - self.rear, LIDAR_X_OFFSET + self.front, -self.right, self.left

    @property
    def radius(self):
        xmin, xmax, ymin, ymax = self.base_bounds
        return math.hypot(max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax)))


class TebplanSolver:
    """局部规划求解器（支持障碍约束自适应、轨迹点数量自动调整）"""
    
    def __init__(self, x0, xf, obstacles=None, n=None, *,
                 v_max=0.2, omega_max=0.4, r_min=0.3, a_max=1.0, epsilon=1e-2,
                 w_p=2.0, w_t=0.5, w_kin=2.0, w_r=2.0, w_obs=2.0, T_min=0.05, T_max=0.2,
                 lidar_front=0.00577, lidar_rear=0.23423, lidar_left=0.12, lidar_right=0.12):
        """
        初始化路径规划求解器
        
        参数:
            x0: 起点坐标和姿态 [x, y, theta]
            xf: 终点坐标和姿态 [x, y, theta]
            obstacles: 障碍物坐标数组，形状为 (m, 2)，空数组表示无障碍
            n: 中间点数(None时自动计算)
            lidar_front/rear/left/right: 从雷达到车头、车尾、左侧、右侧禁区边缘的距离（米）
            
            v_max=0.3     每段线速度上限，单位 m/s。求解器用 `段位移 / (dt + epsilon)` 算速度
            omega_max=1.0 每段角速度的绝对值上限，单位 rad/s。`1.0 rad/s` 约等于 `57°/s`
            r_min=0.5     希望转弯半径至少为 `0.5 m`。半径更小会产生惩罚，惩罚力度由 `w_r` 决定
            a_max=2.0     相邻两段之间的线加速度绝对值上限，单位 m/s²
            epsilon=1e-2  加在速度、加速度和转弯半径公式的分母上，避免除零，也会影响约束的实际松紧。比如 `dt=0.05 s` 时，速度公式用 `0.06 s` 作分母
            T_min=0.05vv  T_max=0.2 单段 `dt` 的上下限，单位秒。总行驶时间是所有 `dt` 加起来

            w_p=0.5    减少各段位移的平方和，让相邻轨迹点更近、步长更均匀。 
            w_t=1.0    减少各段 `dt` 的平方和，倾向于更快通过。 
            w_kin  让每段位移方向更贴近车身朝向，减少轨迹里差速车难以执行的横向移动。 
            w_r    更强地惩罚小于 `r_min` 的转弯半径，倾向于走更大的弯。 
            w_obs 更强地惩罚靠近障碍物。求解器里这一项是软惩罚，节点还会用原始 LiDAR 点检查整条轨迹和首段速度指令。 
        """
        self.x0 = np.array(x0)
        self.xf = np.array(xf)
        self.obstacles = np.empty((0, 2)) if obstacles is None else np.asarray(obstacles)
        self.footprint = LidarFootprint(lidar_front, lidar_rear, lidar_left, lidar_right)
        self.v_max = v_max
        self.omega_max = omega_max
        self.r_min = r_min
        self.a_max = a_max
        self.epsilon = epsilon
        self.w_p = w_p
        self.w_t = w_t
        self.w_kin = w_kin
        self.w_r = w_r
        self.w_obs = w_obs
        self.T_min = T_min
        self.T_max = T_max
        
        # 自动计算轨迹点数量n（若未指定）
        self._auto_resize = n is None
        self.n = self._auto_calculate_n() if self._auto_resize else n
        self.n = max(5, self.n)  # 确保最少5个中间点
        
        # 求解结果
        self.trajectory = None
        self.solver_result = None
        self.cost = None
    
    def _auto_calculate_n(self):
        """根据起点终点距离、最大速度和障碍数量自动计算中间点数n"""
        # 1. 计算起点到终点的直线距离
        pos0 = self.x0[:2]
        posf = self.xf[:2]
        dist_total = np.linalg.norm(posf - pos0)
        
        # 若起点终点重合，返回最小点数
        if dist_total < 1e-6:
            return 5
        
        # 2. 估算每个时间步的最大移动距离（基于平均时间步）
        T_avg = (self.T_min + self.T_max) / 2  # 平均时间步
        max_step_dist = self.v_max * T_avg     # 每步最大移动距离
        
        # 3. 基础点数：总距离 / 每步最大距离（减1是因为总点数为n+2）
        base_n = int(dist_total / max_step_dist) - 1
        base_n = max(2, base_n)  # 至少2个中间点
        
        return base_n
    
    def solve(self, x0=None, xf=None, obstacles=None, *, reference=None):
        """
        求解路径规划问题，支持同时更新起点、终点与障碍物。
        构造时未指定 n 的情况下，更新问题数据会重新计算 n。
        
        参数:
            x0 : 新起点 [x, y, theta],None 表示沿用旧值
            xf : 新终点 [x, y, theta],None 表示沿用旧值
            obstacles : 新障碍物数组 shape (m,2),None 表示沿用旧值
            reference : 可选的初始轨迹，形状 (n+2, 3)
        返回:
            trajectory : 优化后的轨迹 (n+2, 3)
        """
        # 1. 按需更新起点/终点
        if x0 is not None:
            self.x0 = np.array(x0)
        if xf is not None:
            self.xf = np.array(xf)
        if obstacles is not None:
            self.obstacles = np.array(obstacles)

        # 2. 仅在启用自动采样时重新计算 n。
        if self._auto_resize and any(arg is not None for arg in (x0, xf, obstacles)):
            self.n = max(3, self._auto_calculate_n())

        # 3. 构建并求解
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            if reference.shape != (self.n + 2, 3) or not np.isfinite(reference).all():
                raise ValueError('reference must contain one finite pose per TEB knot')
        res = self._build_and_solve(self.obstacles, reference)

        # 4. 提取结果
        self.solver_result = res
        self.trajectory, dt_seq = self._extract_trajectory(res)
        self.cost = float(res['f'])
        return self.trajectory, dt_seq
    
    def _build_and_solve(self, obs_now, reference=None):
        """构建优化问题并求解（支持无障碍时忽略障碍约束）"""
        # 变量定义（n+2个轨迹点，n+1个时间步）
        x = ca.SX.sym('x', self.n + 2)
        y = ca.SX.sym('y', self.n + 2)
        theta = ca.SX.sym('theta', self.n + 2)
        dt = ca.SX.sym('dt', self.n + 1)
        z = ca.vertcat(x, y, theta, dt)
        
        # 目标函数
        f = 0
        for i in range(self.n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            f += self.w_p * (dx**2 + dy**2)  # 路径平滑性
            f += self.w_t * dt[i]**2          # 时间惩罚
        
        g_ineq = []

        if reference is not None:
            ref_x, ref_y = reference[:, :2].T
            f += 40.0 * (ca.sumsqr(x - ref_x) + ca.sumsqr(y - ref_y))

        # Soft signed clearance from the same rectangle used by the swept collision check.
        if len(obs_now):
            obs_x, obs_y = obs_now[:, :2].T
            shape = (self.n + 2, 1)
            dx = np.tile(obs_x, shape) - ca.repmat(x, 1, len(obs_now))
            dy = np.tile(obs_y, shape) - ca.repmat(y, 1, len(obs_now))
            cos_yaw = ca.repmat(ca.cos(theta), 1, len(obs_now))
            sin_yaw = ca.repmat(ca.sin(theta), 1, len(obs_now))
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            xmin, xmax, ymin, ymax = self.footprint.base_bounds
            # Smooth absolute values and maximum for IPOPT. The exact rectangle is checked separately.
            outside_x = ca.sqrt((local_x - (xmin + xmax) / 2)**2 + 1e-6) - (xmax - xmin) / 2
            outside_y = ca.sqrt((local_y - (ymin + ymax) / 2)**2 + 1e-6) - (ymax - ymin) / 2
            clearance = (outside_x + outside_y + ca.sqrt((outside_x - outside_y)**2 + 1e-6)) / 2
            f += self.w_obs * ca.sum1(ca.vec(ca.exp(-clearance / 0.02)))

        # 3. 运动学约束（速度、角速度、加速度、转弯半径）
        for i in range(self.n + 1):
            # 位移计算
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            dist_step = ca.sqrt(dx**2 + dy**2 + 1e-12)
            
            # 线速度约束：|v| <= v_max
            v = dist_step / (dt[i] + self.epsilon)
            g_ineq.extend([v - self.v_max, -v - self.v_max])
            
            # 角速度约束：|omega| <= omega_max
            dth = ca.atan2(ca.sin(theta[i+1]-theta[i]),
                          ca.cos(theta[i+1]-theta[i]))  # 角度差（[-pi, pi]）
            omega = dth / (dt[i] + self.epsilon)
            g_ineq.extend([omega - self.omega_max, -omega - self.omega_max])
            
            # 转弯半径软约束（惩罚小于最小半径的情况）
            radius = v / (ca.fabs(omega) + self.epsilon)
            f += self.w_r * ca.fmax(0, self.r_min - radius)**2
            
            # 加速度约束（除最后一个时间步）
            if i < self.n:
                dx2 = x[i+2] - x[i+1]
                dy2 = y[i+2] - y[i+1]
                dist_step2 = ca.sqrt(dx2**2 + dy2**2 + 1e-12)
                v2 = dist_step2 / (dt[i+1] + self.epsilon)
                acc = (v2 - v) / (0.5*(dt[i] + dt[i+1]) + self.epsilon)
                g_ineq.extend([acc - self.a_max, -acc - self.a_max])
        
        # 4. 非完整约束（惩罚运动方向与姿态偏离）
        for i in range(self.n + 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]

            li = ca.vertcat(ca.cos(theta[i]), ca.sin(theta[i]))
            li1 = ca.vertcat(ca.cos(theta[i+1]), ca.sin(theta[i+1]))
            cross = (li[0] + li1[0]) * dy - (li[1] + li1[1]) * dx
            f += self.w_kin * cross**2
        
        # 约束边界设置
        g = ca.vertcat(*g_ineq)
        lbg = [-ca.inf] * len(g_ineq)
        ubg = [0] * len(g_ineq)
        
        # 变量上下界
        lbx = -np.inf * np.ones(z.shape[0])
        ubx = np.inf * np.ones(z.shape[0])
        
        # 固定起点和终点的位置与姿态
        fix_idx = [
            0, self.n+1,                # x的起点和终点索引
            self.n+2, 2*self.n+3,       # y的起点和终点索引
            2*self.n+4, 3*self.n+5      # theta的起点和终点索引
        ]
        lbx[fix_idx] = ubx[fix_idx] = [
            self.x0[0], self.xf[0],
            self.x0[1], self.xf[1],
            self.x0[2], self.xf[2]
        ]
        
        # 时间步上下界
        dt_start_idx = 3 * (self.n + 2)
        lbx[dt_start_idx:] = self.T_min
        ubx[dt_start_idx:] = self.T_max
        
        # 获取样条曲线初始化值
        if reference is None:
            init_x, init_y, init_theta = self._get_spline_initial_guess()
        else:
            init_x, init_y, init_theta = reference.T

        z0 = np.zeros(z.shape[0])
        z0[:self.n+2] = init_x
        z0[self.n+2:2*self.n+4] = init_y
        z0[2*self.n+4:3*self.n+6] = init_theta

        # 时间步初始化
        z0[3*self.n+6:] = np.ones(self.n+1) * ((self.T_min + self.T_max)/2)
        
        # 求解NLP
        nlp = {'x': z, 'f': f, 'g': g}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.max_cpu_time': 0.15,
            'print_time': False,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        stats = solver.stats()
        if not stats['success']:
            raise RuntimeError(f"IPOPT failed: {stats['return_status']}")
        return res
    
    def _get_spline_initial_guess(self):
        """使用三次埃尔米特样条生成平滑的初始猜测轨迹"""
        n_points = self.n + 2
        t = np.linspace(0, 1, n_points)
        
        # 计算起点和终点的切线向量 (方向由 theta 决定)
        # 这里的 scale 决定了曲线的“张力”，通常取位移距离的一半
        dist = np.linalg.norm(self.xf[:2] - self.x0[:2])
        scale = dist * 0.5 
        
        v0 = np.array([np.cos(self.x0[2]), np.sin(self.x0[2])]) * scale
        vf = np.array([np.cos(self.xf[2]), np.sin(self.xf[2])]) * scale
        
        p0 = self.x0[:2]
        pf = self.xf[:2]
        
        # 三次埃尔米特样条基函数
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        # 生成位置插值 (n_points, 2)
        path_xy = (np.outer(h00, p0) + np.outer(h10, v0) + 
                np.outer(h01, pf) + np.outer(h11, vf))
        
        # 生成角度插值：通过路径的切向计算 theta
        # 计算路径上每一点的导数（速度矢量）
        dh00 = 6*t**2 - 6*t
        dh10 = 3*t**2 - 4*t + 1
        dh01 = -6*t**2 + 6*t
        dh11 = 3*t**2 - 2*t
        
        path_v = (np.outer(dh00, p0) + np.outer(dh10, v0) + 
                np.outer(dh01, pf) + np.outer(dh11, vf))
        
        path_theta = np.arctan2(path_v[:, 1], path_v[:, 0])
        
        # 修正：样条生成的 theta 在起点和终点必须严格等于设定值
        path_theta[0] = self.x0[2]
        path_theta[-1] = self.xf[2]
        
        # 处理角度突变 (unwrap) 确保插值平滑
        path_theta = np.unwrap(path_theta)
        
        return path_xy[:, 0], path_xy[:, 1], path_theta

    def _extract_trajectory(self, res):
        """从求解结果中提取轨迹（包含位置、姿态和时间差序列）"""
        # 提取位置和姿态
        x = res['x'][:self.n+2].full().flatten()
        y = res['x'][self.n+2:2*self.n+4].full().flatten()
        th = res['x'][2*self.n+4:3*self.n+6].full().flatten()
        
        # 提取时间差序列（dt）
        dt_start_idx = 3 * (self.n + 2)  # dt变量的起始索引
        dt = res['x'][dt_start_idx:dt_start_idx + self.n + 1].full().flatten()
        
        # 返回轨迹（位置+姿态）和时间差序列
        trajectory = np.column_stack((x, y, th))
        return trajectory, dt

    
    def get_trajectory(self):
        return self.trajectory
    
    def get_cost(self):
        return self.cost