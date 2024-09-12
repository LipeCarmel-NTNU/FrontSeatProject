import casadi as ca
import numpy as np

class BioReactorMPC:
    class Constraint():
        def __init__(self, g, lbg, ubg, name=None):
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.name = name
    def __init__(self, x0, u0, nx, nu, N, T, n_int=2):
        # Number of states
        self.nx = nx
        # Number of control input
        self.nu = nu
        # Number of time steps
        self.N = N
        # Time horizon
        self.T = T
        # Discretization time step
        self.DT = T / N
        # Number of integrations between two steps
        self.n_int = n_int
        # Integration step
        self.h = self.DT / self.n_int
        # Number of optimization variables
        self.n_opti = (self.nx + self.nu) * (self.N) + self.nx
        
        # Define symbolic variables for the computations
        self.x = ca.SX.sym('x', self.nx)
        self.u = ca.SX.sym('u', self.nu)
        
        # Define the symbolic variables used for the optimization
        self.state = [ca.SX.sym("X_0", self.nx)]
        self.inputs = []
        self.opt_var = self.state.copy()
        for i in range(self.N):
            self.inputs += [ca.SX.sym(f"U_{i}", self.nu)]
            self.state += [ca.SX.sym(f"X_{i+1}", self.nx)]
            self.opt_var += [self.inputs[i], self.state[i+1]]
                
        # self.opt_var_0 = self.init_values(x0, u0)
        
        self.lb_opt_var = - np.ones(self.n_opti) * np.inf
        self.ub_opt_var =   np.ones(self.n_opti) * np.inf
        
        self.cost = 0
        self.constraints = []
    
    def bioreactor_model(self):
        # States
        V, X, S, CO2 = self.x[0], self.x[1], self.x[2], self.x[3]

        # Manipulated variables
        Fin = self.u[0]
        Fout = self.u[1]

        # Parameters
        Sin = 200
        Q = 2
        V_total = 2.7  # L
        CO2in = 0.04  # 100*volume fraction of CO2 in feed

        Y_XS = 0.4204
        mu_max = 0.1945
        Ks = 0.0070
        Y_CO2X = 0.5430
        kd = 0.0060

        # Define the rate
        mu = mu_max * (S / (Ks + S))

        # Differential equations
        dV = Fin - Fout
        dX = -X * (Fin / V) + mu * X - kd * X  # Biomass
        dS = (Sin - S) * (Fin / V) - mu * X / Y_XS  # Substrate
        dCO2 = ((CO2in - CO2) * Q + mu * X / Y_CO2X) / (V_total - V)  # CO2

        # Output
        self.dx = ca.vertcat(dV, dX, dS, dCO2)
        
    def cost_function(self, gains, xdes=None):
        """
        This method is used to define the cost function of the optimization problem.
        
        Args:
            gains: dict
                Dictionary containing the gains of the cost function.
        """
        # Check if the desired state is defined
        if xdes is None:
            xdes = np.zeros(self.nx)
        
        controls = self.u
        x_dot = self.dx
        
        k1 = gains['k_volume']
        k2 = gains['k_biomass']
        r = gains['reg']
        
        # Cost function
        self.L = k1*(self.x[0] - xdes[0])**2 + k2*(self.x[1] - xdes[1])**2 + r * ca.sumsqr(controls)
    
    def ode(self):
        """
        This method is used to define the ordinary differential equation of the optimization problem.
        """     
        
        dynamics = ca.Function('dynamics', [self.x, self.u], [self.dx, self.L], ['x0', 'u0'], ['xf', 'qf'])
        
        return dynamics
    
    def integrator(self):
        """
        This method is used to define the integrator of the optimization problem.
        """
        # Define the integrator
        ode = self.ode()
        X0 = ca.SX.sym('X0', self.nx)
        U = ca.SX.sym('U', self.nu)
        X = X0
        Q = 0
        for j in range(self.n_int):
            k1, k1_q = ode(X, U)
            k2, k2_q = ode(X + self.h/2 * k1, U)
            k3, k3_q = ode(X + self.h/2 * k2, U)
            k4, k4_q = ode(X + self.h * k3, U)
            X = X+self.h/6*(k1 + 2*k2 + 2*k3 + k4)
            Q = Q + self.h/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            # X = X + self.h*k1
            # Q = Q + self.h*k1_q
        self.update_state = ca.Function('update_state', [X0, U], [X, Q], ['x0', 'u0'], ['xf', 'qf'])
        
    def init_values(self, x0, u0):
        """
        This method is used to define the initial values of the optimization variables.
        """    
        # Initial dynamics propagation with constant input
        w0 = []
        w0 += x0.tolist()
        x0_k = x0
        u0_k = u0.tolist()
        v_init = []
        x_init = []
        s_init = []
        co2_init = []
        
        F0_init = []
        F1_init = []
        
        # Set the initial state in the constraints of the optimization variables
        self.lb_opt_var[:self.nx] = x0.tolist()
        self.ub_opt_var[:self.nx] = x0.tolist()    

        for k in range(self.N):
            w0 += u0_k
            x0_k = self.update_state(x0=x0_k, u0=u0_k)  # return a DM type structure

            x0_k = x0_k['xf'].full().flatten()

            w0 += x0_k.tolist()
            
            # Extract the state variables
            v_init += [x0_k[0]]
            x_init += [x0_k[1]]
            s_init += [x0_k[2]]
            co2_init += [x0_k[3]]
            
            # Extract the control input variables
            F0_init += [u0_k[0]]
            F1_init += [u0_k[1]]
            
            
        
        return w0, v_init, x_init, s_init, co2_init, F0_init, F1_init
    
    def set_bounds_x(self, inputs_lb, inputs_ub):
        """
        This method is used to set the bounds of the state variables.
        """
        for i in range(self.N):
            self.lb_opt_var[i*(self.nx+self.nu):i*(self.nx+self.nu)+self.nx] = inputs_lb
            self.ub_opt_var[i*(self.nx+self.nu):i*(self.nx+self.nu)+self.nx] = inputs_ub
        
    def set_bounds_u(self, inputs_lb, inputs_ub):
        """
        This method is used to set the bounds of the control input variables.
        """
        for i in range(self.N):
            self.lb_opt_var[i*(self.nx+self.nu)+self.nx:i*(self.nx+self.nu)+self.nx+self.nu] = inputs_lb
            self.ub_opt_var[i*(self.nx+self.nu)+self.nx:i*(self.nx+self.nu)+self.nx+self.nu] = inputs_ub
        
    def add_constraint(self, g, lbg, ubg, name=None):
        """
        This method is used to add a constraint to the optimization problem.
        """
        if name is not None:
            for constraint in self.constraints:
                if constraint.name == name:
                    raise ValueError(f"Constraint {name} already exists.")
        
        self.constraints += [self.Constraint(
            g=g,
            lbg=np.array([lbg]).flatten(),
            ubg=np.array([ubg]).flatten(),
            name=name,
        )]
        
    def update_constraint(self, name, g=None, lbg=None, ubg=None):
        for constraint in self.constraints:
            if constraint.name == name:
                if g is not None:
                    constraint.g = g
                if lbg is not None:
                    constraint.lbg = np.array([lbg]).flatten()
                if ubg is not None:
                    constraint.ubg = np.array([ubg]).flatten()
                return
            
        raise ValueError(f"Constraint {name} not found.")
    
    def multiple_shooting(self):
        """
        This method is used to define the multiple shooting constraint of the optimization problem.
        """
        for i in range(self.N):
            F = self.update_state(x0=self.state[i], u0=self.inputs[i])
            Xk_end = F['xf']
            self.cost += F['qf']
            
            # Continuity constraint
            self.add_constraint(
                g=[Xk_end - self.state[i+1]],
                lbg=np.zeros(self.nx),
                ubg=np.zeros(self.nx),
                name=f"Multiple shooting {i}",
            )
    
    def set_terminal_cost(self, xdes, gains):
        """
        This method is used to set the terminal cost of the optimization problem.
        
        Args:
            xdes: np.array
                Desired state.
            gains: dict
                Dictionary containing the gains of the terminal cost.
        """
        self.cost += gains['alpha']*(self.state[-1][0] - xdes[0])**2 + gains['beta']*(self.state[-1][1] - xdes[1])**2 + gains['gamma']*(self.state[-1][2] - xdes[2])**2    
    
    def create_solver(self):
        """
        This method is used to create the solver of the optimization problem.
        """
        g = []
        for constraint in self.constraints:
            g += constraint.g
        
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.opt_var),
            'g': ca.vertcat(*g)
        }
                
        opts = {
            'ipopt.max_iter': 5e2,
            # 'ipopt.gradient_approximation': 'finite-difference-values',
            # 'ipopt.hessian_approximation': 'limited-memory', 
            # 'ipopt.hsllib': "/usr/local/libhsl.so",
            # 'ipopt.linear_solver': 'mumps',
            # 'ipopt.mu_strategy': 'adaptive',
            # 'ipopt.adaptive_mu_globalization': 'kkt-error',
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.print_level': 3
        }
                        
        self.solver = ca.nlpsol('solver', 'ipopt', problem, opts)
        
    def solve(self):
        """
        This method is used to solve the optimization problem.
        """
        lbg = np.empty(0)
        ubg = np.empty(0)
        for constraint in self.constraints:
            lbg = np.concatenate((lbg, constraint.lbg))
            ubg = np.concatenate((ubg, constraint.ubg))
    
        r = self.solver(
            x0=self.opt_var_0,
            lbx=self.lb_opt_var, ubx=self.ub_opt_var,
            lbg=lbg, ubg=ubg,
        )
        
        sol = r['x'].full().flatten()
        
        self.opt_var_0 = sol
        
        # Optimal state and control input
        v_opt = []
        x_opt = []
        s_opt = []
        co2_opt = []
        
        F0_opt = []
        F1_opt = []
        
        for i in range(self.N):
            v_opt += [sol[i*(self.nx+self.nu)]]
            x_opt += [sol[i*(self.nx+self.nu) + 1]]
            s_opt += [sol[i*(self.nx+self.nu) + 2]]
            co2_opt += [(sol[i*(self.nx+self.nu) + 3])]
            F0_opt += [(sol[i*(self.nx+self.nu) + 4])]
            F1_opt += [(sol[i*(self.nx+self.nu) + 5])]
            

        return v_opt, x_opt, s_opt, co2_opt, F0_opt, F1_opt
    
    def step(self, x0, u0):
        """
        This method is used to solve the optimization problem (use it in an MPC fashion).
        
        Args:
            x0: np.array
                Initial state.
            u0: np.array
                Initial control input.
        """
        self.init_values(x0, u0)
        
        # self.create_solver()
        
        return self.solve()
        