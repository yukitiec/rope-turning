from IPython.display import clear_output
import os
import sys
import time

import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import yaml

# Add the parent directory to the path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from other directories
try:
    from ur5e import ur_custom_casadi as ur_kin  # Commented out - not used
except ImportError:
    print("Warning: Could not import ur_custom_casadi")
    ur_kin = None

# from utils import mpc_utils 
try:
    import utils._create_update_model as update_model
except ImportError:
    print("Warning: Could not import utils._create_update_model")
    update_model = None

try:
    import utils.kalman_filter_3d as kf
except ImportError:
    print("Warning: Could not import utils.kalman_filter_3d")
    kf = None

os.makedirs('images', exist_ok=True)
if os.name == 'nt':
    plt.rcParams['font.family'] = 'MS Gothic'


class MPCController:
    def __init__(self, config_env=r"config/env/mpc.yaml", 
                 config_robot=r"config/policy/mpc.yaml"):
        # Read config files with UTF-8 encoding
        try:
            with open(config_env, 'r', encoding='utf-8') as f:
                self.config_env = yaml.load(f, Loader=yaml.FullLoader)
        except UnicodeDecodeError:
            # Fallback to system default encoding if UTF-8 fails
            with open(config_env, 'r') as f:
                self.config_env = yaml.load(f, Loader=yaml.FullLoader)
                
        try:
            with open(config_robot, 'r', encoding='utf-8') as f:
                self.config_policy = yaml.load(f, Loader=yaml.FullLoader)
        except UnicodeDecodeError:
            # Fallback to system default encoding if UTF-8 fails
            with open(config_robot, 'r') as f:
                self.config_policy = yaml.load(f, Loader=yaml.FullLoader)
                
        # Convert parameters to proper types
        self.ns = int(self.config_env['params']['ns'])
        self.length_rope = float(self.config_env['params']['_length_rope'])
        self.g = float(self.config_env['params']['g'])
        self.rho = float(self.config_env['params']['pho']) # density : 1.14e3 kg/m^3
        self.radius = float(self.config_env['params']['r']) # radius : 1e-2 m.
        self.k_s = float(self.config_env['params']['k_s'])
        self.unit_length = round(self.length_rope / self.ns,3)  #unit length
        self.m = self.rho*(np.pi*self.radius**2)*self.unit_length #mass of the unit length. kg
        self.nu = int(self.config_env['params']['nu'])
        self.weight_default = float(self.config_env['params']['weight_default'])
        
        self.nx = 2 * 3 * (self.ns + 1) # State dimension. Exclude the velocity of the human and the robot.

        self.G = casadi.DM([0, 0, self.g])
        
        # Create the cost weight.
        self._create_cost_weight()
        
        # Policy setting.
        self.horizon = int(self.config_policy["horizon"])
        self.K = self.horizon
        self.control_frequency = int(self.config_policy["control_frequency"])
        self.dt = 1.0/self.control_frequency
        self.time_horizon = self.horizon*self.dt
        
        self.type_solver = str(self.config_policy["solver"])
        self.tol_equality = float(self.config_policy["tol_equality"])
        
        # Constraint.
        self.x_lb = [-1000] * self.nx
        self.x_ub = [1000] * self.nx
        self.u_lb = [-6.28] * self.nu
        self.u_ub = [6.28] * self.nu
        self.eq_lb = [-self.tol_equality] * self.nx * self.K
        self.eq_ub = [self.tol_equality] * self.nx * self.K
        
        # Number of parameters to optimize
        self.total = self.nx * (self.K + 1) + self.nu * self.K

    def _create_cost_weight(self):
        """
        Create the cost weight for the MPC.
        The cost weight is a list of lists.
        The first list is the weight for the position of the robot.
        The second list is the weight for the position of the human.
        The third list is the weight for the velocity of the robot.
        The fourth list is the weight for the velocity of the human.
        """
        cost_weight = []
        cost_weight.append([0, 0, 0])
        if self.ns == 2:
            cost_weight.append([self.weight_default, self.weight_default, self.weight_default])
            cost_weight.append([0.0, 0.0, 0.0])
        else:
            for i in range(1, self.ns):  # for the rope point weights.

                if i < self.ns // 2:  # robot to the middle.

                    weight = (1 + (i / (self.ns // 2 - 1))) * self.weight_default

                else:  # middle to human.

                    weight = (2 - ((i - (self.ns // 2)) / 
                                (self.ns // 2 - 1))) * self.weight_default

                cost_weight.append([weight, weight, weight])

            cost_weight.append([0, 0, 0])  # for the human's position.

        cost_weight = np.array(cost_weight,dtype=np.float64)

        cost_weight = cost_weight.ravel()
        cost_weight = cost_weight.tolist()
        

        self.Q = casadi.diag(
            cost_weight  # position, (ns+1)
            + [0.0, 0.0, 0.0]  # robot's angular speed.
            + [self.weight_default/100., self.weight_default/100., self.weight_default/100.] * (self.ns - 1)  # velocity except human
            + [0, 0, 0]
        )  # controlled point is 12.5, and other points's speed is 125.

        self.Q_f = casadi.diag(cost_weight + [0.0, 0.0, 0.0] * (self.ns + 1))

        self.R = casadi.diag([self.weight_default/100., self.weight_default/100., self.weight_default/100.])

    def make_RK4(self):
        """Make a RK4 integrator to estimate the next state.
        """

        states = casadi.SX.sym("states", self.nx)

        ctrls = casadi.SX.sym("ctrls", self.nu)

        f = update_model.create_update_model(
            k_s=self.k_s, l=self.unit_length, m=self.m, G=self.G, 
            ns=self.ns, nx=self.nx, nu=self.nu
        )

        r1 = f(x=states, u=ctrls)["x_dot"]

        r2 = f(x=states + self.dt * r1 / 2, u=ctrls)["x_dot"]

        r3 = f(x=states + self.dt * r2 / 2, u=ctrls)["x_dot"]

        r4 = f(x=states + self.dt * r3, u=ctrls)["x_dot"]

        states_next = states + self.dt * (r1 + 2 * r2 + 2 * r3 + r4) / 6

        RK4 = casadi.Function("RK4", [states, ctrls], [states_next], 
                             ["x", "u"], ["x_next"])

        return RK4

    def make_integrator(self):
        """Make an integrator to proceed with the simulation step by step.
        """

        states = casadi.SX.sym("states", self.nx)

        ctrls = casadi.SX.sym("ctrls", self.nu)

        f = update_model.create_update_model(
            k_s=self.k_s, l=self.unit_length, m=self.m, G=self.G, 
            ns=self.ns, nx=self.nx, nu=self.nu
        )

        ode = f(x=states, u=ctrls)["x_dot"]

        # Add ctrl_hum to the parameters so the integrator can use it

        dae = {"x": states, "p": ctrls, "ode": ode}

        integrator = casadi.integrator("I", "cvodes", dae, 0, self.dt)

        return integrator

    def compute_stage_cost(self, x, u, x_ref):
        """Compute the stage cost of MPC

        Args:
            x (casadi.SX): state
            u (casadi.SX): control input
            x_ref (casadi.SX): referential data
        """
        # for robot's state. from joint angle to end-effector pose.

        x_diff = x - x_ref

        cost = (casadi.dot(self.Q @ x_diff, x_diff) + casadi.dot(self.R @ u, u)) / 2
        return cost

    def compute_terminal_cost(self, x, x_ref):
        """Compute the terminal cost.

        Args:
            x (casadi.SX): state
            x_ref (casadi.SX): referential data_
        """
        x_diff = x - x_ref

        cost = casadi.dot(self.Q_f @ x_diff, x_diff) / 2
        return cost
    
    def make_qp_formulation(self):
        """Create a QP formulation for QP solvers (OSQP, qpoases)
        
        Returns:
            dict: QP problem formulation
        """
        RK4 = self.make_RK4()

        U = [
            casadi.SX.sym(f"u_{k}", self.nu) for k in range(self.K)
        ]  # robot's control input. 3D velocity.

        X = [
            casadi.SX.sym(f"x_{k}", self.nx) for k in range(self.K + 1)
        ]  # state vector. from robot's end-effector to the human's turning point.

        # Reference trajectory as parameters
        X_ref = [
            casadi.SX.sym(f"x_ref_{k}", self.nx) for k in range(self.K + 1)
        ]  # reference state vector as parameters

        G = []

        J = 0

        # compute the stage cost and the constraint.
        for k in range(self.K):
            J += self.compute_stage_cost(X[k], U[k], X_ref[k]) * self.dt
            eq = X[k + 1] - RK4(x=X[k], u=U[k])["x_next"]
            G.append(eq)
        J += self.compute_terminal_cost(X[-1], X_ref[-1])

        # Pass reference trajectory as parameters
        qp = {"x": casadi.vertcat(*X, *U), "f": J, "g": casadi.vertcat(*G), "p": casadi.vertcat(*X_ref)}
        return qp

    def make_nlps(self):
        """construct NLP and QP solvers

        Returns:
            tuple: (S_ipopt, S_qrsqp, S_osqp, S_qpoases) - solver instances
        """
        RK4 = self.make_RK4()

        U = [
            casadi.SX.sym(f"u_{k}", self.nu) for k in range(self.K)
        ]  # robot's control input. 3D velocity.

        X = [
            casadi.SX.sym(f"x_{k}", self.nx) for k in range(self.K + 1)
        ]  # state vector. from robot's end-effector to the human's turning point.

        # Reference trajectory as parameters
        X_ref = [
            casadi.SX.sym(f"x_ref_{k}", self.nx) for k in range(self.K + 1)
        ]  # reference state vector as parameters

        G = []

        J = 0

        # compute the stage cost and the constraint.
        for k in range(self.K):
            J += self.compute_stage_cost(X[k], U[k], X_ref[k]) * self.dt
            eq = X[k + 1] - RK4(x=X[k], u=U[k])["x_next"]
            G.append(eq)
        J += self.compute_terminal_cost(X[-1], X_ref[-1])

        # Pass reference trajectory as parameters
        nlp = {"x": casadi.vertcat(*X, *U), "f": J, "g": casadi.vertcat(*G), "p": casadi.vertcat(*X_ref)}
        
        # IPOPT solver
        option_ipopt = {"print_time": False, "ipopt": {"print_level": 0}}
        S_ipopt = casadi.nlpsol("S_ipopt", "ipopt", nlp, option_ipopt)
        
        # QR-SQP solver
        option_qrsqp = {
            "print_time": False,
            "print_iteration": False,
            "print_header": False,
            "qpsol_options": {
                "print_info": False,
                "print_iter": False,
                "print_header": False,
            },
        }
        S_qrsqp = casadi.nlpsol("S_qrsqp", "qrsqp", nlp, option_qrsqp)
        
        # Create QP formulation for QP solvers
        qp = self.make_qp_formulation()
        
        # OSQP solver with print suppression
        option_osqp = {
            "print_time": False,
            "print_out": False,
            "print_in": False
        }
        S_osqp = casadi.qpsol("S_osqp", "osqp", qp, option_osqp)
        
        # qpoases solver
        option_qpoases = {"printLevel": "none", "sparse": False}
        S_qpoases = casadi.qpsol("S_qpoases", "qpoases", qp, option_qpoases)
        
        return S_ipopt, S_qrsqp, S_osqp, S_qpoases
    
    def get_solver(self, solver_type=None):
        """Get the appropriate solver based on solver type
        
        Args:
            solver_type (str, optional): Solver type. If None, uses self.type_solver
            
        Returns:
            casadi solver: The selected solver instance
        """
        if solver_type is None:
            solver_type = self.type_solver
            
        # Create all solvers
        S_ipopt, S_qrsqp, S_osqp, S_qpoases = self.make_nlps()
        
        # Return the appropriate solver
        if solver_type == "ipopt":
            print("Using IPOPT solver")
            return S_ipopt
        elif solver_type == "qrsqp":
            print("Using QR-SQP solver")
            return S_qrsqp
        elif solver_type == "osqp":
            print("Using OSQP solver")
            return S_osqp
        elif solver_type == "qpoases":
            print("Using qpoases solver")
            return S_qpoases
        else:
            raise ValueError(f"Invalid solver type: {solver_type}. Available solvers: ipopt, qrsqp, osqp, qpoases")
    
    def get_all_solvers(self):
        """Get all available solvers
        
        Returns:
            tuple: (S_ipopt, S_qrsqp, S_osqp, S_qpoases) - all solver instances
        """
        return self.make_nlps()
    
    def solve_optimization(self, x_init, x_ref, x0=None, S=None):
        """Solve the optimization problem with the specified solver
        
        Args:
            x_init (casadi.DM): Initial state
            x_ref (list): Reference trajectory
            S (casadi solver, optional): Solver instance. If None, uses default solver
            
        Returns:
            tuple: (u_opt, x_trajectory, solver_stats) - optimal control, 
                   state trajectory, and solver statistics
        """
        if S is None:
            S = self.get_solver(self.type_solver)
 
        # Prepare the initial state. (self.nx,1) -> (self.nx,)
        x_init_list = x_init.full().ravel().tolist()
        
        # Check for NaN values in initial state
        if any(np.isnan(val) for val in x_init_list):
            print("Error: NaN values detected in initial state after validation")
            return None, None, None
        
        # Create bounds for states and controls only (no reference in variables)
        lbx = x_init_list + self.x_lb * self.K + self.u_lb * self.K  # self.nx*(self.K+1)+self.nu*self.K
        ubx = x_init_list + self.x_ub * self.K + self.u_ub * self.K
        lbx = np.array(lbx,dtype=np.float64).tolist()
        ubx = np.array(ubx,dtype=np.float64).tolist()
        lbg = np.array(self.eq_lb,dtype=np.float64).tolist()
        ubg = np.array(self.eq_ub,dtype=np.float64).tolist()
        #lbg = [val if not np.isnan(val) else -self.tol_equality for val in lbg]
        #ubg = [val if not np.isnan(val) else self.tol_equality for val in ubg]
        
        # Flatten reference trajectory for parameters
        x_ref_flat = []
        for k in range(self.K + 1):
            if k < len(x_ref):
                ref_k = x_ref[k]
                if isinstance(ref_k, casadi.DM):
                    ref_k_flat = ref_k.full().ravel().tolist()
                else:
                    ref_k_flat = np.array(ref_k).ravel().tolist()
                x_ref_flat.extend(ref_k_flat)
            else:
                # Pad with zeros if not enough reference points
                x_ref_flat.extend([0.0] * self.nx)
        
        
        # Create initial guess for states and controls
        if x0 is None:
            total_vars = self.nx * (self.K + 1) + self.nu * self.K
            x0 = np.zeros(total_vars, dtype=np.float64)
        
        # Solve the problem with parameters
        try:
            res = S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0, p=x_ref_flat)
            #res = S(lbx=lbx, ubx=ubx, x0=x0, p=x_ref_flat)
            # Extract results
            x_sol = res["x"]
            offset = self.nx * (self.K + 1)
            u_opt = x_sol[offset:offset + self.nu]
            
            # Extract state trajectory
            x_trajectory = []
            for k in range(self.K + 1):
                start_idx = k * self.nx
                end_idx = (k + 1) * self.nx
                x_trajectory.append(x_sol[start_idx:end_idx])
            
            print(f"Optimization successful. u_opt shape: {u_opt.shape}")
            return u_opt, x_trajectory, res
            
        except Exception as e:
            print(f"Error in solve_optimization: {e}")
            print(f"x_init shape: {x_init.shape}, x_ref length: {len(x_ref)}")
            print(f"Solving optimization with {len(x_init_list)} initial states, {len(x_ref_flat)} reference parameters")
            print(f"Bounds: lbx={len(lbx)}, ubx={len(ubx)}, lbg={len(self.eq_lb)}, ubg={len(self.eq_ub)}")
            print(f"Initial state range: [{min(x_init_list):.3f}, {max(x_init_list):.3f}]")
            print(f"Reference range: [{min(x_ref_flat):.3f}, {max(x_ref_flat):.3f}]")
            if len(x_ref) > 0:
                print(f"First x_ref element shape: {x_ref[0].shape if hasattr(x_ref[0], 'shape') else 'unknown'}")
            return None, None, None
    
    def compare_solvers(self, x_init, x_ref):
        """Compare all available solvers
        
        Args:
            x_init (casadi.DM): Initial state
            x_ref (list): Reference trajectory
            
        Returns:
            dict: Dictionary with solver comparison results
        """
        solvers = ["ipopt", "qrsqp", "osqp", "qpoases"]
        results = {}
        
        for solver_type in solvers:
            print(f"Testing {solver_type}...")
            start_time = time.time()
            
            u_opt, x_trajectory, res = self.solve_optimization(
                x_init, x_ref, solver_type)
            
            end_time = time.time()
            solve_time = end_time - start_time
            
            if u_opt is not None:
                results[solver_type] = {
                    "solve_time": solve_time,
                    "success": True,
                    "u_opt": u_opt,
                    "x_trajectory": x_trajectory,
                    "solver_stats": res
                }
            else:
                results[solver_type] = {
                    "solve_time": solve_time,
                    "success": False,
                    "u_opt": None,
                    "x_trajectory": None,
                    "solver_stats": None
                }
        
        return results

    def interpolate(self,p_st, p_end, length_rope, n_points):
        """
        Create a 3D position list by linear interpolation between start and end points.

        Args:
            p_st (np.array): Start point position (3,)
            p_end (np.array): End point position (3,)
            length_rope (float): Length of the rope (not used in linear interpolation)
            n_points (int): Number of points to generate (including endpoints)

        Returns:
            positions (np.array): Array of shape (n_points, 3) containing interpolated positions
        """
        p_intersect = self.cal_intersect(p_st, p_end, length_rope)
        positions = np.zeros((n_points, 3))
        positions[0] = p_st.copy()
        positions[n_points - 1] = p_end.copy()
        for i in range(1, n_points - 1):
            if i < n_points // 2:
                t = i / (n_points // 2)
                positions[i] = (1 - t) * p_st + t * p_intersect
            else:
                t = (i - n_points // 2) / (n_points // 2)
                positions[i] = (1 - t) * p_intersect + t * p_end

        return positions


    def cal_intersect(self,p_st, p_end, length_rope):
        """Calculate the intersection between the orthogonal vector to base_vec and the circle
        whose center is p_st and radius is length_rope/2.

        Args:
            p_st (np.array): start point position. (3,)
            p_end (np.array): end point position. (3,)

        Returns:
            p_intersect (np.array): intersection point position. (3,)
        """

        p_middle = (p_st + p_end) / 2
        vec_end_to_st = p_end - p_st
        vec_end_to_st = vec_end_to_st / (np.linalg.norm(vec_end_to_st) + 1e-4)  # normalize

        # orthogonal vector
        vec_orthogonal = np.array([0, vec_end_to_st[2], -vec_end_to_st[1]])
        vec_orthogonal = vec_orthogonal / (
            np.linalg.norm(vec_orthogonal) + 1e-10
        )  # normalize

        p_intersect = p_middle
        t = 0
        delta_t = 0.01  # 1 cm resolution
        max_iter = 10000
        tol = delta_t * 1.5
        iter_count = 0
        prev_dist = None
        while iter_count < max_iter:
            dist = np.linalg.norm(p_intersect - p_st)
            if np.linalg.norm(dist - length_rope / 2) < tol:
                print(
                    f"Finish :: iter_count: {iter_count}, \n dist: {round(dist,2)}, length_rope: {round(length_rope,2)}\n err={round(np.linalg.norm(dist - length_rope / 2),2)}"
                )
                break
            if iter_count % 10 == 0:
                print(
                    f"iter_count: {iter_count}, \n dist: {round(dist,2)}, length_rope: {round(length_rope,2)}\n err={round(np.linalg.norm(dist - length_rope / 2),2)}"
                )
            p_intersect_pos = p_middle + vec_orthogonal * (t + delta_t)
            p_intersect_neg = p_middle + vec_orthogonal * (t - delta_t)
            dist_pos = np.linalg.norm(p_intersect_pos - p_st)
            dist_neg = np.linalg.norm(p_intersect_neg - p_st)
            if np.linalg.norm(dist_pos - length_rope / 2) < np.linalg.norm(
                dist_neg - length_rope / 2
            ):
                t += delta_t
            else:
                t -= delta_t
            p_intersect = p_middle + vec_orthogonal * t
            prev_dist = dist
            iter_count += 1

        # Choose the candidates whose z-coordinate is lower.
        p_intersect_neg = p_middle - vec_orthogonal * t
        p_intersect_pos = p_middle + vec_orthogonal * t
        if p_intersect_pos[2] < p_intersect_neg[2]:
            p_intersect = p_intersect_pos.copy()
        else:
            p_intersect = p_intersect_neg.copy()
        dist_to_st = np.linalg.norm(p_intersect - p_st)

        print(f"dist_to_st: {dist_to_st}")

        return p_intersect


    def make_x_init(self,pose_eef, pos_human, length_rope):
        """Initialize the rope state.

        Args:
            pose_eef (np.array): robot's end-effector position. (6,)
            pos_human (np.array): human's turning point position. (3,)
            length_rope (float): length of the rope.

        Returns:
            x_init (np.array): initial rope state. (2 * ns - 1, 3)
        """
        # make the initial rope state.

        x_init = np.zeros(
            (
                2 * (self.ns+1),
                3,
            )  # except the velocity of robot and the human
        )  # 0--ns-1 : position, ns--2ns-3 : velocity except two points(0:robot, ns:human)

        position_list = self.interpolate(pose_eef[:3], pos_human, length_rope, self.ns+1)
        # Rope position.
        x_init[:self.ns+1] = position_list.copy()  # (10,3)
        # Rope velocity = 0
        x_init = x_init.ravel()  # (30), position+velocity
        print(f"x_init: {x_init.shape}")
        x_init = casadi.DM(x_init)

        return x_init


    def make_x0(self, x_init):
        """Make an input storage including state, and control inputs for the next iteration
        x0  #(nx,)
        """
        # initial optimized variables including state and control inputs
        integrator = self.make_integrator()

        x0 = [x_init]  # (30,)

        x_tmp = x_init

        for i in range(self.K):

            x_tmp = integrator(x0=x_tmp)["xf"]

            x0.append(x_tmp)

        u0 = casadi.DM.zeros(self.nu * self.K)  # K steps

        x0 = casadi.vertcat(*x0, u0)  # state + control input. nx*(K+1)+nu*K

        return x0


    def get_window_input(self,data, idx_st, window_size):
        """Get a window of data from idx_st to idx_st+window_size.

        Args:
            data (np.array): data to get window from. (n,3) #(velocity)
            idx_st (int): start index of the window.
            window_size (int): size of the window.

        Returns:
            window_data (np.array): window of data. (window_size,ns,3)
        """
        window_size = window_size + 1  # add a current state
        window_data = data[idx_st : min(idx_st + window_size, data.shape[0])]
        if window_data.shape[0] < window_size:
            window_data = np.concatenate(
                [window_data, data[: window_size - window_data.shape[0]]]
            )  # add a initial data.
        return window_data