import casadi as ca
import numpy as np
from parameters import vehicle_parameters

class LPVMPC:
    def __init__(self, N=20, dt=0.1, params=vehicle_parameters):
        self.N = N
        self.dt = dt
        self.params = params

        self.nx = 2
        self.nu = 1

        # ===== 决策变量 =====
        X = ca.SX.sym('X', self.nx, N + 1)
        U = ca.SX.sym('U', self.nu, N)

        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        x0 = ca.SX.sym('x0', self.nx)
        ref = ca.SX.sym('ref', self.nx, N)
        Ad = ca.SX.sym('Ad', self.nx, self.nx)
        Bd = ca.SX.sym('Bd', self.nx, self.nu)
        
        # ===== 权重 =====
        alpha_Q = ca.SX.sym('alpha_Q', self.nx)
        alpha_R = ca.SX.sym('alpha_R', self.nu)
        Q_base = ca.vertcat(10000, 10000)  
        R_base = ca.vertcat(0.0001) 
        Q = ca.diag(Q_base * ca.exp(alpha_Q))
        R = ca.diag(R_base * ca.exp(alpha_R))

        p = ca.vertcat(x0, ca.reshape(ref, -1, 1), ca.reshape(Ad, -1, 1), 
            ca.reshape(Bd, -1, 1), alpha_Q, alpha_R)

        cost = 0
        g = [X[:, 0] - x0]

        # ===== 构建 MPC =====
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            err = xk - ref[:, k]
            cost += ca.mtimes([err.T, Q, err])
            cost += ca.mtimes([uk.T, R, uk])

            # 动力学约束
            x_next = Ad @ xk + Bd @ uk
            g.append(X[:, k + 1] - x_next)

        g = ca.vertcat(*g)

        nlp = {'x': w, 'p': p, 'f': cost, 'g': g}

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 50,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # ========= 变量维度 =========
        self.nw = w.size1()
        self.ng = g.size1()

        # ========= box 约束 =========
        self.lbx = -ca.inf * np.ones(self.nw)
        self.ubx =  ca.inf * np.ones(self.nw)

        # 控制约束
        offset_u = self.nx * (N + 1)
        for k in range(N):
            self.lbx[offset_u + k] = -3000
            self.ubx[offset_u + k] = 3000

        # 等式约束 g = 0
        self.lbg = np.zeros(self.ng)
        self.ubg = np.zeros(self.ng)

        # 初始猜测
        self.w0 = np.zeros(self.nw)

    # ===== LPV 离散化 =====
    def get_lpv_discrete_model(self, vx, tire):
        p = self.params
        m, Iz = p['mass'], p['Iz']
        lf, lr = p['lf'], p['lr']
        Cf, Cr = p['Cf'], p['Cr']
        if tire == 1 or tire == -1:
            Cf = Cf * 0.5 
        elif tire == 2 or tire == -2:
            Cr = Cr * 0.5  

        vx = max(vx, 0.1)  # 防止奇异

        A = np.array([
            [-(Cf + Cr) / (m * vx),
             -vx - (Cf * lf - Cr * lr) / (m * vx)],
            [-(Cf * lf - Cr * lr) / (Iz * vx),
             -(Cf * lf**2 + Cr * lr**2) / (Iz * vx)]
        ])

        B = np.array([
            [0.0],
            [1.0 / Iz]
        ])

        Ad = np.eye(self.nx) + A * self.dt
        Bd = B * self.dt

        return Ad, Bd

    # ===== 在线求解 =====
    def solve(self, x_full, ref_traj, tire, alpha_Q, alpha_R):
        vx = x_full[0] / 3.6
        vy = x_full[1] / 3.6
        r = x_full[2] / 180 * np.pi

        Ad, Bd = self.get_lpv_discrete_model(vx, tire)

        p_val = np.concatenate([np.array([vy, r]),
                                ref_traj.reshape(-1, order='F'),
                                Ad.reshape(-1, order='F'),
                                Bd.reshape(-1, order='F'),
                                alpha_Q, 
                                alpha_R
                                ])
        try:
            sol = self.solver(x0=self.w0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p_val)
            
            w_opt = sol['x'].full().flatten()
            offset_u = self.nx * (self.N + 1)
            u_opt = w_opt[offset_u: offset_u + self.nu]
            return float(u_opt[0])
        
        except RuntimeError:
            return 0.0
